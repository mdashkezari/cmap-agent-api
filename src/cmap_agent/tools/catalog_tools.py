from __future__ import annotations

import re
import time
import logging
from typing import Any, Optional, Literal

from pydantic import BaseModel, Field
from sqlalchemy import text

from cmap_agent.rag.retrieval import get_kb, KBBackend
from cmap_agent.storage.sqlserver import SQLServerStore

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# UDF-backed in-memory catalog cache
# ---------------------------------------------------------------------------

_CACHE_TTL_SECONDS = 6 * 3600  # 6 hours

# Columns pulled from udfCatalog() — enough for search, ranking, and display.
# Heavy columns (statistics, unstructured metadata) are excluded to keep memory low.
_UDF_SELECT = """
    Table_Name        AS table_name,
    Dataset_ID        AS dataset_id,
    Dataset_Name      AS dataset_name,
    Dataset_Short_Name AS dataset_short_name,
    Make              AS make,
    Sensor            AS sensor,
    Temporal_Resolution AS temporal_resolution,
    Spatial_Resolution  AS spatial_resolution,
    Time_Min          AS time_min,
    Time_Max          AS time_max,
    Lat_Min           AS lat_min,
    Lat_Max           AS lat_max,
    Lon_Min           AS lon_min,
    Lon_Max           AS lon_max,
    Depth_Min         AS depth_min,
    Depth_Max         AS depth_max,
    Variable          AS variable,
    Long_Name         AS long_name,
    Unit              AS unit,
    Keywords          AS keywords,
    Data_Source       AS data_source,
    Distributor       AS distributor,
    Dataset_Description AS description,
    Acknowledgement   AS acknowledgement
"""


class CatalogCache:
    """Lazy-loading, TTL-refreshed in-memory snapshot of the CMAP catalog.

    Loaded from udfCatalog() once at first use; refreshed every _CACHE_TTL_SECONDS.
    All filtering (text search, spatial overlap, variable lookup) runs in Python
    against this cache — no per-query SQL round trips for catalog operations.
    """

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []
        self._loaded_at: float = 0.0
        # Dataset IDs that map to more than one Table_Name (e.g. Darwin).
        # Detected dynamically from the cache — never hardcoded.
        self._multi_table_dataset_ids: set[int] = set()

    def _is_fresh(self) -> bool:
        return bool(self._rows) and (time.time() - self._loaded_at) < _CACHE_TTL_SECONDS

    def load(self, store: SQLServerStore) -> None:
        """(Re-)load the cache from the database."""
        log.info("CatalogCache: loading from udfCatalog()...")
        t0 = time.time()
        with store.engine.connect() as conn:
            rows = conn.execute(
                text(f"SELECT {_UDF_SELECT} FROM udfCatalog()")
            ).mappings().all()
        self._rows = [dict(r) for r in rows]
        self._loaded_at = time.time()

        # Detect multi-table datasets dynamically
        from collections import defaultdict
        ds_tables: dict[int, set[str]] = defaultdict(set)
        for r in self._rows:
            ds_id = r.get("dataset_id")
            tbl = r.get("table_name")
            if ds_id is not None and tbl:
                ds_tables[int(ds_id)].add(str(tbl))
        self._multi_table_dataset_ids = {
            ds_id for ds_id, tables in ds_tables.items() if len(tables) > 1
        }

        elapsed = time.time() - t0
        log.info(
            "CatalogCache: loaded %d rows (%d datasets, %d multi-table) in %.1fs",
            len(self._rows),
            len(ds_tables),
            len(self._multi_table_dataset_ids),
            elapsed,
        )

    def ensure_loaded(self, store: SQLServerStore) -> None:
        if not self._is_fresh():
            self.load(store)

    @property
    def rows(self) -> list[dict[str, Any]]:
        return self._rows

    @property
    def multi_table_dataset_ids(self) -> set[int]:
        return self._multi_table_dataset_ids


# Module-level singleton
_catalog_cache = CatalogCache()


def _get_cache(ctx: dict) -> tuple[CatalogCache, SQLServerStore]:
    store = _get_store(ctx)
    _catalog_cache.ensure_loaded(store)
    return _catalog_cache, store


# ---------------------------------------------------------------------------
# Text search helpers
# ---------------------------------------------------------------------------

def _contains(text_val: Any, q_lower: str) -> bool:
    """Case-insensitive substring check."""
    return q_lower in str(text_val or "").lower()


def _match_score(row: dict[str, Any], q: str) -> float:
    """Score a cache row by how well it matches query string q.

    Each field contributes at most once per token (binary hit). For multi-word
    queries, scores are summed across all matched tokens and then averaged, so
    a dataset matching all query terms scores higher than one matching only one.

    Field specificity: exact variable name > variable LIKE > long_name > keywords >
    dataset names > table/sensor/source > description/acknowledgement (broad/noisy).
    """
    tokens = [t.strip() for t in q.lower().split() if len(t.strip()) >= 3]
    if not tokens:
        return 0.0
    # Score each token separately and average — prevents single-term domination
    total = 0.0
    for ql in tokens:
        total += _match_score_single(row, ql)
    return total / len(tokens)


def _match_score_single(row: dict[str, Any], ql: str) -> float:
    """Score for a single lowercase query token against a cache row."""
    s = 0.0
    var = str(row.get("variable") or "").lower()
    ln = str(row.get("long_name") or "").lower()
    kw = str(row.get("keywords") or "").lower()
    ds_name = str(row.get("dataset_name") or "").lower()
    ds_short = str(row.get("dataset_short_name") or "").lower()
    tbl = str(row.get("table_name") or "").lower()
    sensor = str(row.get("sensor") or "").lower()
    src = str(row.get("data_source") or "").lower()
    dist = str(row.get("distributor") or "").lower()
    desc = str(row.get("description") or "").lower()
    ack = str(row.get("acknowledgement") or "").lower()

    # Each field: binary — either it matches or it doesn't
    if var == ql:
        s += 5.0
    elif ql in var:
        s += 3.0

    if ln == ql:
        s += 4.0
    elif ql in ln:
        s += 2.0

    if ql in kw:
        s += 1.5

    if ql in ds_name or ql in ds_short:
        s += 1.0

    if ql in tbl:
        s += 0.8

    if ql in sensor or ql in src or ql in dist:
        s += 0.5

    # Description and acknowledgement are long and noisy — small binary bonus only
    if ql in desc:
        s += 0.3

    if ql in ack:
        s += 0.2

    return s


def _search_rows(rows: list[dict[str, Any]], q: str) -> list[dict[str, Any]]:
    """Return rows matching query q in any searchable field.

    For multi-word queries (e.g. "nitrate phosphate silicate"), a row matches
    if ANY of the tokens appear in any searchable field. Single-word queries
    behave as before.
    """
    tokens = [t for t in q.lower().split() if len(t) >= 3]
    if not tokens:
        return []

    def _row_matches(r: dict[str, Any]) -> bool:
        blob = " ".join([
            str(r.get("variable") or ""),
            str(r.get("long_name") or ""),
            str(r.get("keywords") or ""),
            str(r.get("dataset_name") or ""),
            str(r.get("dataset_short_name") or ""),
            str(r.get("table_name") or ""),
            str(r.get("sensor") or ""),
            str(r.get("data_source") or ""),
            str(r.get("distributor") or ""),
            str(r.get("description") or ""),
            str(r.get("acknowledgement") or ""),
        ]).lower()
        return any(t in blob for t in tokens)

    return [r for r in rows if _row_matches(r)]


def _row_to_dataset_dict(row: dict[str, Any], kb_score: float = 0.0) -> dict[str, Any]:
    """Convert a cache variable-level row into a dataset-level result dict."""
    return {
        "table": row.get("table_name"),
        "name": row.get("dataset_short_name") or row.get("dataset_name") or row.get("table_name"),
        "title": row.get("dataset_name"),
        "dataset_id": row.get("dataset_id"),
        "make": row.get("make"),
        "sensor": row.get("sensor"),
        "description": (str(row.get("description") or ""))[:400],
        "kb_score": kb_score,
        "time_min": row.get("time_min"),
        "time_max": row.get("time_max"),
        "lat_min": row.get("lat_min"),
        "lat_max": row.get("lat_max"),
        "lon_min": row.get("lon_min"),
        "lon_max": row.get("lon_max"),
        "spatial_resolution": row.get("spatial_resolution"),
        "temporal_resolution": row.get("temporal_resolution"),
    }


def _dataset_type_bonus(row: dict[str, Any]) -> float:
    """Bonus/penalty for dataset type to rank gridded/satellite products above
    cruise in-situ datasets when the variable name match is ambiguous.

    Cruise datasets often have variables literally named "Chlorophyll",
    "Nitrate" etc. — exact matches for common queries — but gridded satellite
    or model products are almost always what users want for mapping requests.
    The bonuses here are large enough to overcome a variable-name exact match.
    """
    tres = str(row.get("temporal_resolution") or "").lower()
    sres = str(row.get("spatial_resolution") or "").lower()
    sensor = str(row.get("sensor") or "").lower()
    s = 0.0
    # Gridded/regular temporal resolution
    if any(t in tres for t in ("daily", "eight day", "weekly", "monthly", "annual")):
        s += 3.0
    # Gridded spatial resolution
    if any(t in sres for t in ("km", "degree", "°")):
        s += 2.0
    # Satellite sensor
    if "satellite" in sensor:
        s += 2.0
    # Point-like / cruise: both irregular → penalty
    if "irregular" in tres and "irregular" in sres:
        s -= 2.0
    return s


def _deduplicate_to_datasets(
    matched_rows: list[dict[str, Any]],
    q: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Deduplicate variable-level matched rows to dataset/table level.

    For each unique Table_Name, pick the best-scoring variable row as the
    representative. The effective ranking score combines field-specificity
    match_score with a small dataset-type bonus (gridded/satellite preferred).
    This prevents cruise datasets with a variable literally named "Chlorophyll"
    from always outranking satellite products whose canonical variable is "chlor_a".

    Variable-availability gate (v222): after per-table scoring, apply a
    penalty to tables whose variable list has zero concept-token overlap
    with the query.  This is the same gate used in
    ``_post_rank_catalog_results`` (v221) — mirrored here so
    ``catalog_search`` (SQL-LIKE path) gets the same protection as
    ``catalog_search_kb_first`` (KB-ranked path).  The penalty is
    sized to match the gate's original tuning (~5 points) scaled to
    the smaller score magnitudes produced by ``_match_score`` +
    ``_dataset_type_bonus`` (typical range ~4–8).

    Modality intent adjustment (v223): when the query names a single
    modality (``model``, ``in-situ``, ``observation``, etc.), boost
    matching candidates and penalize recognised-opposite candidates.
    Also mirrored from ``_post_rank_catalog_results`` so both catalog
    paths apply the same intent.  Magnitudes scaled to this path's
    score range.
    """
    # Concept tokens extracted once for the whole query.
    concept = _concept_tokens_from_query(q)
    modality_intent = _modality_intent_from_query(q)

    best: dict[str, tuple[float, dict[str, Any]]] = {}
    for row in matched_rows:
        tbl = str(row.get("table_name") or "")
        if not tbl:
            continue
        score = _match_score(row, q) + _dataset_type_bonus(row)
        if tbl not in best or score > best[tbl][0]:
            best[tbl] = (score, row)

    # Apply variable-availability gate.  Dataset-level — check each
    # distinct table once, not per variable row (cheaper and correct).
    # Penalty is scaled to this score range: here scores are typically
    # in [0, 10] from match + type bonus, vs [0, 40+] in the
    # post-rank function where kb_score contributes *3.  A penalty of
    # 3.5 on this scale has the same relative effect as 5 on that
    # scale.
    _GATE_PENALTY = 3.5
    # Modality adjustment is scaled similarly: the KB path uses
    # +3.0 / -4.0, so the plain path uses +2.0 / -2.5 to keep the
    # relative weight to the surrounding ~0-10 scores comparable.
    _MODALITY_BOOST = 2.0
    _MODALITY_PENALTY = -2.5
    gated_best: dict[str, tuple[float, dict[str, Any]]] = {}
    for tbl, (score, row) in best.items():
        # Build a minimal dataset dict for the gate (it only reads
        # `table`).  Passing the variable-level row directly would also
        # work — the gate uses `r.get("table") or r.get("table_name")`
        # — but we construct a small dict for clarity.
        if _variable_availability_score({"table": tbl}, concept) == 0.0:
            score -= _GATE_PENALTY
        # Modality intent.  Use the representative variable row we
        # already picked (``row``) since it carries the candidate's
        # make and sensor.  Compute via the shared helper then rescale
        # from the KB path's magnitudes to this path's magnitudes.
        if modality_intent:
            adj = _modality_score_adjustment(row, modality_intent)
            if adj > 0:
                score += _MODALITY_BOOST
            elif adj < 0:
                score += _MODALITY_PENALTY
        gated_best[tbl] = (score, row)

    # Sort by descending effective score
    ranked = sorted(gated_best.values(), key=lambda x: -x[0])
    results = []
    for score, row in ranked[:limit]:
        d = _row_to_dataset_dict(row, kb_score=score)
        _add_reason(d, q)
        results.append(d)
    return results


def _add_reason(d: dict[str, Any], q: str) -> None:
    ql = q.lower()
    tbl = (d.get("table") or "").lower()
    name = (d.get("name") or "").lower()
    title = (d.get("title") or "").lower()
    if tbl == ql or name == ql or title == ql:
        d["reason"] = "exact"
    elif ql in tbl or ql in name or ql in title:
        d["reason"] = "partial"
    else:
        d["reason"] = "metadata"


# ---------------------------------------------------------------------------
# Arg models
# ---------------------------------------------------------------------------

class CatalogSearchArgs(BaseModel):
    query: str = Field(..., description="Search term — variable name, dataset name, or scientific concept.")
    limit: int = Field(15, ge=1, le=50, description="Max number of results")


class CatalogSearchVariablesArgs(BaseModel):
    query: str = Field(..., description="Variable name or scientific term to search for.")
    limit: int = Field(10, ge=1, le=50, description="Max number of results")
    table_hint: str | None = Field(None, description="Restrict search to this table name.")


class DatasetMetadataArgs(BaseModel):
    table: str


class ListVariablesArgs(BaseModel):
    table: str


class CountDatasetsArgs(BaseModel):
    """No-arg tool: returns the number of datasets in the catalog."""
    pass


class DatasetSummaryArgs(BaseModel):
    query: str = Field(..., description="Dataset table name or a text query (short name/title/keywords).")
    max_variables: int = Field(25, ge=0, le=200, description="Max number of variables to return")
    max_matches: int = Field(10, ge=1, le=25, description="If the query is not an exact match, return up to this many matching datasets. Use 20 when the user asks for a comprehensive summary of all datasets matching a program or topic.")


# ---------------------------------------------------------------------------
# Internal helpers (keep store accessor for KB tools that still need SQL)
# ---------------------------------------------------------------------------

def _get_store(ctx: dict) -> SQLServerStore:
    store = ctx.get("store")
    if isinstance(store, SQLServerStore):
        return store
    return SQLServerStore.from_env()


# ---------------------------------------------------------------------------
# catalog_search — now cache-backed
# ---------------------------------------------------------------------------

def catalog_search(args: CatalogSearchArgs, ctx: dict) -> dict:
    cache, _ = _get_cache(ctx)
    q = (args.query or "").strip()
    if not q:
        return {"results": [], "selected": None, "alternates": [], "total_returned": 0}

    matched = _search_rows(cache.rows, q)
    results = _deduplicate_to_datasets(matched, q, limit=args.limit)

    selected = results[0] if results else None
    alternates = results[1:6] if len(results) > 1 else []
    return {
        "tool": "catalog.search",
        "query": q,
        "selected": selected,
        "alternates": alternates,
        "results": results,
        "total_returned": len(results),
    }


# ---------------------------------------------------------------------------
# Spatial helpers (unchanged)
# ---------------------------------------------------------------------------

class CatalogSearchROIArgs(BaseModel):
    lat1: float = Field(..., description="Southern latitude bound")
    lat2: float = Field(..., description="Northern latitude bound")
    lon1: float = Field(..., description="Western longitude bound")
    lon2: float = Field(..., description="Eastern longitude bound")
    make: str | None = Field(None, description="Filter by make: Observation, Model, or Assimilation")
    sensor: str | None = Field(None, description="Filter by sensor type")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of datasets to return.")
    rank_mode: Literal["mixed", "bbox_area", "overlap_area"] = Field(
        "mixed",
        description="Ranking strategy: 'mixed' (default) interleaves coverage and tight-bbox ranking.",
    )


def _norm_lat_bounds(lat1: float, lat2: float) -> tuple[float, float]:
    a, b = float(lat1), float(lat2)
    return (min(a, b), max(a, b))


def _lon_intervals(lon1: float, lon2: float) -> list[tuple[float, float]]:
    a = float(lon1)
    b = float(lon2)
    if abs(b - a) >= 359.0:
        return [(-180.0, 180.0)]
    def norm(x: float) -> float:
        y = ((x + 180.0) % 360.0) - 180.0
        if y == -180.0 and x > 0:
            return 180.0
        return y
    a = norm(a)
    b = norm(b)
    if a <= b:
        return [(a, b)]
    return [(a, 180.0), (-180.0, b)]


def _intervals_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def _bbox_overlaps(
    ds_lat_min: Optional[float],
    ds_lat_max: Optional[float],
    ds_lon_min: Optional[float],
    ds_lon_max: Optional[float],
    roi_lat1: float,
    roi_lat2: float,
    roi_lon1: float,
    roi_lon2: float,
) -> bool:
    if ds_lat_min is None or ds_lat_max is None or ds_lon_min is None or ds_lon_max is None:
        return False
    lat1, lat2 = _norm_lat_bounds(roi_lat1, roi_lat2)
    if float(ds_lat_max) < lat1 or float(ds_lat_min) > lat2:
        return False
    roi_iv = _lon_intervals(roi_lon1, roi_lon2)
    ds_iv = _lon_intervals(float(ds_lon_min), float(ds_lon_max))
    for a in roi_iv:
        for b in ds_iv:
            if _intervals_overlap(a, b):
                return True
    return False


def _interval_width(iv: tuple[float, float]) -> float:
    return max(0.0, float(iv[1]) - float(iv[0]))


def _lon_span(lon1: float, lon2: float) -> float:
    ivs = _lon_intervals(lon1, lon2)
    return sum(_interval_width(iv) for iv in ivs)


def _bbox_area(
    lat_min: Optional[float],
    lat_max: Optional[float],
    lon_min: Optional[float],
    lon_max: Optional[float],
) -> float:
    if any(v is None for v in (lat_min, lat_max, lon_min, lon_max)):
        return float("inf")
    try:
        lat_span = max(0.0, float(lat_max) - float(lat_min))
        lon_span = _lon_span(float(lon_min), float(lon_max))
        return lat_span * lon_span
    except Exception:
        return float("inf")


def _lon_overlap_width(
    ds_lon_min: float,
    ds_lon_max: float,
    roi_lon1: float,
    roi_lon2: float,
) -> float:
    ds_ivs = _lon_intervals(ds_lon_min, ds_lon_max)
    roi_ivs = _lon_intervals(roi_lon1, roi_lon2)
    total = 0.0
    for a in ds_ivs:
        for b in roi_ivs:
            lo = max(a[0], b[0])
            hi = min(a[1], b[1])
            if hi > lo:
                total += hi - lo
    return total


def _bbox_overlap_area(
    ds_lat_min: float,
    ds_lat_max: float,
    ds_lon_min: float,
    ds_lon_max: float,
    roi_lat1: float,
    roi_lat2: float,
    roi_lon1: float,
    roi_lon2: float,
) -> float:
    lat1, lat2 = _norm_lat_bounds(roi_lat1, roi_lat2)
    ov_lat = max(0.0, min(float(ds_lat_max), lat2) - max(float(ds_lat_min), lat1))
    ov_lon = _lon_overlap_width(float(ds_lon_min), float(ds_lon_max), roi_lon1, roi_lon2)
    return ov_lat * ov_lon


def _roi_area(lat1: float, lat2: float, lon1: float, lon2: float) -> float:
    la1, la2 = _norm_lat_bounds(lat1, lat2)
    return max(0.0, la2 - la1) * _lon_span(lon1, lon2)


def _matches_make_sensor(record: dict, make: Optional[str] = None, sensor: Optional[str] = None) -> bool:
    if make:
        rec_make = str(record.get("make") or "").strip().lower()
        if rec_make and make.strip().lower() not in rec_make:
            return False
    if sensor:
        rec_sensor = str(record.get("sensor") or "").strip().lower()
        if rec_sensor and sensor.strip().lower() not in rec_sensor:
            return False
    return True


# ---------------------------------------------------------------------------
# catalog_search_roi — now cache-backed
# ---------------------------------------------------------------------------

def catalog_search_roi(args: CatalogSearchROIArgs, ctx: dict) -> dict:
    """ROI-only dataset search using the in-memory catalog cache."""
    cache, _ = _get_cache(ctx)
    lat1, lat2 = _norm_lat_bounds(args.lat1, args.lat2)

    # Deduplicate cache to one row per table (for ROI, we don't need variable-level)
    seen_tables: set[str] = set()
    dataset_rows: list[dict[str, Any]] = []
    for r in cache.rows:
        tbl = str(r.get("table_name") or "")
        if tbl and tbl not in seen_tables:
            seen_tables.add(tbl)
            dataset_rows.append(r)

    filtered: list[dict[str, Any]] = []
    for r in dataset_rows:
        if not _matches_make_sensor(r, make=args.make, sensor=args.sensor):
            continue
        if _bbox_overlaps(
            r.get("lat_min"), r.get("lat_max"),
            r.get("lon_min"), r.get("lon_max"),
            lat1, lat2, args.lon1, args.lon2,
        ):
            d = _row_to_dataset_dict(r)
            filtered.append(d)

    roi_area = _roi_area(args.lat1, args.lat2, args.lon1, args.lon2)
    roi_area = roi_area if roi_area > 0 else 1.0

    for d in filtered:
        d["_bbox_area"] = _bbox_area(d.get("lat_min"), d.get("lat_max"), d.get("lon_min"), d.get("lon_max"))
        try:
            d["_overlap_area"] = _bbox_overlap_area(
                float(d.get("lat_min")), float(d.get("lat_max")),
                float(d.get("lon_min")), float(d.get("lon_max")),
                args.lat1, args.lat2, args.lon1, args.lon2,
            )
        except Exception:
            d["_overlap_area"] = 0.0
        d["_overlap_frac"] = float(d["_overlap_area"]) / float(roi_area)

    mode = (args.rank_mode or "mixed").strip().lower()
    if mode == "bbox_area":
        filtered.sort(key=lambda r: (float(r.get("_bbox_area") or 1e18), float(r.get("dataset_id") or 1e18)))
    elif mode == "overlap_area":
        filtered.sort(key=lambda r: (
            -float(r.get("_overlap_frac") or 0.0),
            -float(r.get("_overlap_area") or 0.0),
            -float(r.get("_bbox_area") or 0.0),
            float(r.get("dataset_id") or 1e18),
        ))
    else:
        by_cov = sorted(filtered, key=lambda r: (
            -float(r.get("_overlap_frac") or 0.0),
            -float(r.get("_overlap_area") or 0.0),
            -float(r.get("_bbox_area") or 0.0),
            float(r.get("dataset_id") or 1e18),
        ))
        by_tight = sorted(filtered, key=lambda r: (
            float(r.get("_bbox_area") or 1e18),
            -float(r.get("_overlap_frac") or 0.0),
            -float(r.get("_overlap_area") or 0.0),
            float(r.get("dataset_id") or 1e18),
        ))
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        i = j = 0
        while len(out) < args.limit and (i < len(by_cov) or j < len(by_tight)):
            if i < len(by_cov):
                r = by_cov[i]; i += 1
                t = str(r.get("table") or "")
                if t and t not in seen:
                    out.append(r); seen.add(t)
                    if len(out) >= args.limit:
                        break
            if j < len(by_tight):
                r = by_tight[j]; j += 1
                t = str(r.get("table") or "")
                if t and t not in seen:
                    out.append(r); seen.add(t)
        filtered = out

    def _clean_row(r: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in (r or {}).items() if not str(k).startswith("_")}

    results = [_clean_row(r) for r in filtered[:args.limit]]
    selected = results[0] if results else None
    alternates = results[1:6] if len(results) > 1 else []
    return {
        "tool": "catalog.search_roi",
        "query": {"roi": {"lat1": args.lat1, "lat2": args.lat2, "lon1": args.lon1, "lon2": args.lon2},
                  "make": args.make, "sensor": args.sensor},
        "selected": selected,
        "alternates": alternates,
        "results": results,
        "total_returned": len(results),
    }


# ---------------------------------------------------------------------------
# KB-first search (ChromaDB semantic + cache-backed SQL augmentation)
# ---------------------------------------------------------------------------

class CatalogSearchKBFArgs(BaseModel):
    query: str = Field(..., description="Scientific variable/dataset search query.")
    lat1: float | None = None
    lat2: float | None = None
    lon1: float | None = None
    lon2: float | None = None
    dt1: str | None = None
    dt2: str | None = None
    make: str | None = None
    sensor: str | None = None
    limit: int = Field(10, ge=1, le=50, description="Maximum number of datasets to return.")


def _field_family_from_query(q: str) -> str | None:
    ql = (q or "").lower()
    families = {
        "chlorophyll": ["chlorophyll", "chl", "chlor"],
        "temperature": ["temperature", "sst", "sst_"],
        "salinity": ["salinity", "sal", "psu"],
        "nutrients": ["nitrate", "phosphate", "silicate", "ammonium", "nutrient", "no3", "po4", "si"],
        "oxygen": ["oxygen", "o2", "aou"],
        "carbon": ["carbon", "doc", "poc", "dic", "co2", "alkalinity"],
        "iron": ["iron", "fe"],
        "wind": ["wind", "u10", "v10"],
        "precipitation": ["precipitation", "rain", "tp"],
        "current": ["current", "velocity", "u_vel", "v_vel"],
        "fluorescence": ["fluorescence", "fluor"],
    }
    for family, terms in families.items():
        if any(t in ql for t in terms):
            return family
    return None


def _candidate_blob(r: dict[str, Any]) -> str:
    return " ".join([
        str(r.get("table") or r.get("table_name") or ""),
        str(r.get("name") or r.get("dataset_short_name") or ""),
        str(r.get("title") or r.get("dataset_name") or ""),
        str(r.get("description") or ""),
        str(r.get("sensor") or ""),
    ]).lower()


def _field_match_score(r: dict[str, Any], family: str | None) -> float:
    if not family:
        return 0.0
    blob = _candidate_blob(r)
    family_terms = {
        "chlorophyll": ["chlorophyll", "chl"],
        "temperature": ["temperature", "sst"],
        "salinity": ["salinity"],
        "nutrients": ["nitrate", "phosphate", "silicate", "nutrient"],
        "oxygen": ["oxygen", "o2"],
        "carbon": ["carbon", "doc", "poc", "dic"],
        "iron": ["iron", "fe"],
        "wind": ["wind"],
        "precipitation": ["precipitation", "rain"],
        "current": ["current", "velocity"],
        "fluorescence": ["fluorescence"],
    }
    terms = family_terms.get(family, [family])
    return float(sum(1 for t in terms if t in blob))


def _looks_point_like(r: dict[str, Any]) -> bool:
    tres = str(r.get("temporal_resolution") or "").lower()
    sres = str(r.get("spatial_resolution") or "").lower()
    return "irregular" in tres or "irregular" in sres


def _looks_map_friendly(r: dict[str, Any]) -> bool:
    tres = str(r.get("temporal_resolution") or "").lower()
    sres = str(r.get("spatial_resolution") or "").lower()
    return any(t in tres for t in ("daily", "eight day", "weekly", "monthly")) or \
           any(t in sres for t in ("km", "degree", "°"))


def _parse_date_safe(s: Any) -> str | None:
    if not s:
        return None
    try:
        return str(s)[:10]
    except Exception:
        return None


def _time_score_for_request(r: dict[str, Any], dt1: Any, dt2: Any) -> float:
    if not dt1 and not dt2:
        return 0.0
    t_min = _parse_date_safe(r.get("time_min"))
    t_max = _parse_date_safe(r.get("time_max"))
    if not t_min and not t_max:
        return 0.5  # No coverage info — neutral
    req_start = _parse_date_safe(dt1) or "0000-01-01"
    req_end = _parse_date_safe(dt2) or "9999-12-31"
    covers_start = (not t_min) or (t_min <= req_start)
    covers_end = (not t_max) or (t_max >= req_end)
    if covers_start and covers_end:
        return 1.0
    if covers_start or covers_end:
        return 0.5
    return 0.0


def _is_gridded(r: dict[str, Any]) -> bool:
    tres = str(r.get("temporal_resolution") or "").lower()
    sres = str(r.get("spatial_resolution") or "").lower()
    desc = str(r.get("description") or "").lower()
    if any(t in tres for t in ("daily", "eight day", "weekly", "monthly", "annual", "8 day")):
        return True
    if any(t in sres for t in ("km", "degree", "deg", "°", "min", "arc")):
        return True
    if any(t in desc for t in ("gridded", "satellite", "reanalysis")):
        return True
    return False


# ---------------------------------------------------------------------------
# Variable-availability gate (v221 / reviewer 3.4 first intervention)
# ---------------------------------------------------------------------------
#
# Motivation.  The pre-v221 ranker scored candidates by ``kb_score`` (dense
# retrieval) plus blob-text family-term matches.  Neither signal checks
# whether the candidate dataset *actually contains a variable* matching
# the user's query concept.  Consequence: a query like "surface dissolved
# nitrate" can return ``tblSSS_NRT`` at rank 1 because the KB embedding
# matched strongly on "surface" and "ocean" tokens, even though SSS has
# exactly one variable (``sss``) and nothing nitrate-related.
#
# This gate queries the variable cache for each candidate dataset and
# penalises candidates whose variables have zero concept-token overlap
# with the query.  The penalty is large enough to sink a false match
# with a high ``kb_score`` past legitimate matches, but not infinite —
# the dataset can still appear as an alternate if there's no better
# candidate in the merged pool.
#
# Design notes.
#   * No hardcoded dataset or query strings.  The concept set is derived
#     from the same family vocabulary used by the rest of the ranker
#     (``_field_family_from_query``), with a fallback to raw query
#     tokens (minus stopwords/regional words) when no family matches.
#   * Queries with no extractable concept (e.g. a single region name,
#     or purely temporal queries like "2020 data") produce an empty
#     concept set; the gate becomes a no-op in that case.
#   * The gate reads from ``_catalog_cache.rows`` which is already
#     loaded for all catalog operations — no extra database round trips.


# Words that identify neither a variable nor a concept.  These are
# stripped before computing concept tokens so they don't drive the gate.
# Deliberately conservative — only purely spatial, temporal, or
# articles/connectives.  Anything that could plausibly appear in a
# variable name or long_name (e.g. "surface", which is part of "Sea
# Surface Salinity") is NOT included here.
_CONCEPT_STOPWORDS: frozenset[str] = frozenset({
    # Articles / connectives
    "a", "an", "the", "of", "and", "or", "in", "on", "at", "for",
    "from", "to", "with", "by", "over", "under",
    # Directional / regional (region already scored by ROI overlap)
    "north", "south", "east", "west", "northern", "southern",
    "eastern", "western",
    # Ocean basins by name (scored by ROI overlap when coordinates given)
    "atlantic", "pacific", "indian", "arctic", "antarctic",
    "mediterranean", "ocean", "sea", "gulf", "bay",
    # Year/date-ish
    "year", "years", "month", "months", "day", "days", "date",
})


def _concept_tokens_from_query(q: str) -> set[str]:
    """Extract concept tokens from a user query for the variable-availability
    gate.

    Strategy: prefer the family vocabulary when a family is detected, since
    it already encodes abbreviation/synonym pairs (``chl``/``chlorophyll``,
    ``sst``/``temperature``).  Fall back to raw non-stopword query tokens
    so that concepts outside the family dict still gate correctly.
    """
    ql = (q or "").lower().strip()
    if not ql:
        return set()

    # Family vocabulary — already defined in _field_family_from_query.
    # Recompute the full term lists here so the gate uses the richer
    # (abbreviation-inclusive) list from the family definition.
    _FAMILY_TERMS: dict[str, list[str]] = {
        "chlorophyll": ["chlorophyll", "chl", "chlor"],
        "temperature": ["temperature", "sst"],
        "salinity": ["salinity", "sal", "psu"],
        "nutrients": ["nitrate", "phosphate", "silicate", "ammonium",
                      "nutrient", "no3", "po4"],
        "oxygen": ["oxygen", "o2", "aou"],
        "carbon": ["carbon", "doc", "poc", "dic", "co2", "alkalinity"],
        "iron": ["iron", "fe"],
        "wind": ["wind", "u10", "v10"],
        "precipitation": ["precipitation", "rain"],
        "current": ["current", "velocity"],
        "fluorescence": ["fluorescence", "fluor"],
    }

    # Collect all families triggered by the query.  A query can hit
    # multiple families ("carbon and nitrate"); gate on the union.
    concept: set[str] = set()
    for _family, terms in _FAMILY_TERMS.items():
        if any(t in ql for t in terms):
            concept.update(terms)

    # Fallback: raw query tokens (minus stopwords, minus punctuation)
    # when no family matched.  Also add raw tokens even when a family
    # did match — a query like "satellite POC" still benefits from
    # having "poc" itself as a concept token.
    import re as _re
    raw_tokens = _re.findall(r"[a-z0-9]+", ql)
    for tok in raw_tokens:
        if tok in _CONCEPT_STOPWORDS:
            continue
        if len(tok) < 2:
            continue
        # Skip tokens that are pure make/sensor modality words —
        # these are routing cues, not variable concepts.
        if tok in {"satellite", "model", "observation", "insitu",
                   "assimilation", "climatology", "monthly", "daily",
                   "seasonal", "surface", "hourly", "annual",
                   "map", "plot", "show", "get", "give"}:
            continue
        concept.add(tok)

    return concept


def _variable_availability_score(
    r: dict[str, Any],
    concept: set[str],
) -> float:
    """Return 0 when the candidate has no variable matching any concept
    token, 1 otherwise.

    Reads ``_catalog_cache.rows`` to look up variables for the candidate
    table.  ``r`` comes from search results / metadata lookups and
    carries only dataset-level fields, so variable information has to be
    joined in.

    Returns 1.0 when concept is empty (gate disabled — no signal to gate
    on).  Returns 1.0 when the candidate table has no variables listed
    in the cache (cannot gate without data; do not penalize).
    """
    if not concept:
        return 1.0
    table = str(r.get("table") or r.get("table_name") or "").strip()
    if not table:
        return 1.0
    cache_rows = _catalog_cache.rows
    if not cache_rows:
        return 1.0

    saw_any_variable = False
    for row in cache_rows:
        if str(row.get("table_name") or "").strip() != table:
            continue
        saw_any_variable = True
        var_name = str(row.get("variable") or "").lower()
        long_name = str(row.get("long_name") or "").lower()
        # Token-level containment in either the short or long name.
        for tok in concept:
            if tok in var_name or tok in long_name:
                return 1.0

    if not saw_any_variable:
        # Dataset unknown to the cache — cannot gate, don't penalize.
        return 1.0
    # Variables exist but none match any concept token.
    return 0.0


# ---------------------------------------------------------------------------
# Modality intent (v223 / reviewer 3.4 second intervention)
# ---------------------------------------------------------------------------
#
# The variable-availability gate (v221/v222) penalises candidates that
# have zero concept-token overlap with the query.  That closes one
# class of failure: wrong dataset because no matching variable exists.
#
# A second class remains: the user specifies a *modality* (model,
# observation, in-situ) but the ranker ignores it.  Example: CS008
# ``model nitrate climatology`` returns ``tblWOA_Climatology`` at
# rank 1 — WOA has nitrate (so the gate doesn't fire) but it's
# observational, not a model, so it doesn't match the user's stated
# intent.  Example: CS009 ``in-situ chlorophyll from cruises`` returns
# ``tblCHL_REP`` (satellite) at rank 1 — CHL_REP has chlorophyll (so
# the gate doesn't fire) but it's satellite, not in-situ.
#
# The existing ranker already has asymmetric satellite handling:
# ``wants_satellite = "satellite" in ql`` triggers +3.0 boost on
# satellite-sensor candidates and -4.0 on cruise-sensor candidates.
# v223 generalises this into a symmetric modality-intent system
# covering all three make values (Observation / Model / Assimilation)
# and the main sensor-level distinctions (Satellite / In-Situ).
#
# Design principles:
#
#   * No hardcoded dataset names or prompt-specific logic.  Modality
#     cues are phrase tokens; field values come from the catalog rows
#     (``make``, ``sensor``).
#   * Bidirectional: matching modality gets a boost, opposite
#     modality gets a penalty, unrelated queries get nothing.
#   * Mixed intent (user mentions multiple modalities) disables the
#     mechanism entirely — we don't know which one they want.
#   * Works from a small phrase→field table rather than per-phrase
#     special-casing.


# Modality cues.  Each entry maps a set of phrase tokens to a
# (field, value-token) pair.  The field is ``make`` (values:
# Observation/Model/Assimilation) or ``sensor`` (value tokens from the
# catalog's Sensor column, e.g. "satellite", "in-situ").
#
# Phrase-token matching is substring-based on the lowercased query.
# That catches "in-situ", "in situ", "insitu", "in-situ measurements"
# all via the substring "in" + "situ" checks (see
# ``_modality_intent_from_query``).  It also catches "modeled" /
# "modelled" / "modeling" all via the substring "model".
#
# Field values in ``_candidate_make_sensor`` are compared token-wise
# (lowercase substring) so "In-Situ" and "in-situ" both match.
_MODALITY_CUES: tuple[tuple[tuple[str, ...], str, str], ...] = (
    # phrase tokens,             field,    value-token
    (("satellite",),             "sensor", "satellite"),
    (("in situ", "in-situ", "insitu", "cruise", "cruises"),
                                 "sensor", "in-situ"),
    (("model", "modeled", "modelled", "modeling", "modelling",
      "simulation", "simulated"),
                                 "make",   "model"),
    (("observation", "observational", "observed", "measured"),
                                 "make",   "observation"),
    (("assimilation", "reanalysis"),
                                 "make",   "assimilation"),
)


def _modality_intent_from_query(q: str) -> dict[str, str]:
    """Extract a single-modality intent from the query, or {} when
    ambiguous/absent.

    Returns a dict with at most one entry: ``{"sensor": "satellite"}``,
    ``{"make": "model"}``, etc.  When the query mentions more than one
    modality (e.g. "satellite chlorophyll model comparison") the
    function returns ``{}`` — the ranker should not apply an
    exclusionary penalty when the user's intent is mixed.

    Mechanically, this scans ``_MODALITY_CUES`` and collects every
    matched (field, value) tuple.  If exactly one matches, return it;
    if zero or more than one, return empty.

    No hardcoded dataset or prompt logic — the phrase tokens are
    generic modality words, not domain-specific handling.
    """
    ql = (q or "").lower()
    if not ql:
        return {}

    matched: list[tuple[str, str]] = []
    for phrases, field, value in _MODALITY_CUES:
        if any(p in ql for p in phrases):
            matched.append((field, value))

    # Collapse duplicates (e.g. two cue-families that map to the same
    # (field, value) pair — shouldn't happen with the current table but
    # defend against future additions).
    uniq = list(dict.fromkeys(matched))
    if len(uniq) != 1:
        return {}
    field, value = uniq[0]
    return {field: value}


def _modality_score_adjustment(
    r: dict[str, Any],
    intent: dict[str, str],
) -> float:
    """Return a score adjustment reflecting modality intent match.

    Returns:
      * +3.0 when the candidate's (make or sensor) matches the intent
        value (substring, case-insensitive).
      * -4.0 when the candidate's field value is a known alternative
        to the intent (e.g. intent is ``sensor: satellite`` but the
        candidate has ``sensor: in-situ``).
      * 0.0 when the candidate's field is empty/unknown or the intent
        is empty — no signal to apply.

    Magnitudes chosen to match the existing ``wants_satellite``
    handling at line ~820 (boost +3.0, penalty -4.0), so the new
    modality logic is a generalisation of that existing pattern
    rather than an additional, uncalibrated layer.
    """
    if not intent:
        return 0.0
    field = next(iter(intent))           # "sensor" or "make"
    target = intent[field].lower()
    cand_value = str(r.get(field) or "").lower()
    if not cand_value:
        return 0.0

    if target in cand_value:
        return 3.0

    # Target absent — is this a known opposite?  Only penalise when
    # the candidate has a recognised opposite value from the catalog
    # vocabulary.  Unknown or exotic values (e.g. "Blend", "Uncategorized")
    # are treated as neutral so we don't over-penalise legitimate
    # hybrid datasets.
    _KNOWN_OPPOSITES: dict[str, tuple[str, ...]] = {
        # field: (target -> recognised other-values)
        # Entry form: "{field}:{target}" -> other values that are known
        # explicit opposites.  Anything not in this table is neutral.
    }
    key = f"{field}:{target}"
    # Build opposites on first use (cheap).  Each modality cue in
    # _MODALITY_CUES that shares a field contributes its value as a
    # potential opposite for the others.
    opposites_by_key: dict[str, set[str]] = {}
    for _phrases, f, v in _MODALITY_CUES:
        opposites_by_key.setdefault(f, set()).add(v.lower())
    same_field_values = opposites_by_key.get(field, set())
    opposites = {v for v in same_field_values if v != target}
    if any(o in cand_value for o in opposites):
        return -4.0

    return 0.0


def _post_rank_catalog_results(
    results: list[dict[str, Any]],
    *,
    query: str,
    dt1: Any = None,
    dt2: Any = None,
    lat1: Any = None,
    lat2: Any = None,
    lon1: Any = None,
    lon2: Any = None,
    modality_hint: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    family = _field_family_from_query(query)
    ql = (query or "").lower()
    wants_satellite = "satellite" in ql
    wants_gridded = any(t in ql for t in ["map", "plot", "gridded", "surface", "global"])
    has_roi = all(v is not None for v in (lat1, lat2, lon1, lon2))
    concept = _concept_tokens_from_query(query)
    # Modality intent: use caller-provided hint when available (the
    # caller may have combined signals from structured intent fields +
    # raw user message that aren't present in ``query``).  Fall back to
    # parsing ``query`` directly when no hint is supplied.  An empty
    # dict from the caller is treated as "no signal" and falls back,
    # but an explicit caller decision to suppress intent can be made
    # by passing a sentinel ``{}`` — however, since empty intent means
    # "no effect" in ``_modality_score_adjustment``, the distinction
    # doesn't matter in practice.
    if modality_hint:
        modality_intent = modality_hint
    else:
        modality_intent = _modality_intent_from_query(query)

    def score(r: dict[str, Any]) -> tuple[float, float, float]:
        is_point = _looks_point_like(r)
        tier = 1.0 if is_point else 0.0

        s = float(r.get("kb_score") or 0.0) * 3.0

        s += _field_match_score(r, family) * 1.5

        # Variable-availability gate: if the candidate has a known
        # variable list that contains zero concept-token matches, sink
        # it.  Penalty tuned to outrank a high-``kb_score`` false match
        # against a moderate-``kb_score`` correct match (delta ~2.5 in
        # typical nitrate→SSS vs nitrate→Darwin case; penalty -5
        # provides clear separation without making the candidate
        # invisible as an alternate).  Returns 1.0 when the gate
        # cannot apply (empty concept set, unknown table, etc.).
        if _variable_availability_score(r, concept) == 0.0:
            s -= 5.0

        if _is_gridded(r):
            s += 2.5
        elif is_point:
            s -= 3.0

        if has_roi:
            r_lat1 = r.get("lat_min")
            r_lat2 = r.get("lat_max")
            r_lon1 = r.get("lon_min")
            r_lon2 = r.get("lon_max")
            if all(v is not None for v in (r_lat1, r_lat2, r_lon1, r_lon2)):
                try:
                    if _bbox_overlaps(
                        float(r_lat1), float(r_lat2), float(r_lon1), float(r_lon2),
                        float(lat1), float(lat2), float(lon1), float(lon2),
                    ):
                        s += 2.0
                    else:
                        s -= 3.0
                except Exception:
                    pass
            else:
                # Dataset has no spatial coverage info (NULL bbox) — penalize when
                # the user specified a region. These are typically cruise datasets
                # with sparse, unindexed coverage that won't help a regional query.
                s -= 1.5

        sensor_val = str(r.get("sensor") or "").lower()
        # Modality intent adjustment.  Generalises the asymmetric
        # pre-v223 satellite handling into a symmetric mechanism
        # covering satellite / in-situ / model / observation /
        # assimilation.  Returns 0.0 when intent is empty or the
        # candidate's field is unknown; ``+3.0`` on match; ``-4.0``
        # when the candidate carries a recognised opposite value.
        s += _modality_score_adjustment(r, modality_intent)
        # Retain the narrow ``wants_satellite`` + in-situ-analyzer
        # anti-pattern: ``thermosalinograph`` / ``elemental analyzer``
        # are sensor-level cues for specific instruments that are
        # rarely what a user asking for "satellite" wants, but they
        # aren't covered by the modality intent table.
        if wants_satellite:
            if "thermosalinograph" in sensor_val or "elemental analyzer" in sensor_val:
                s -= 4.0

        if wants_gridded and _looks_map_friendly(r):
            s += 1.0

        s += _time_score_for_request(r, dt1, dt2) * 1.5

        blob = _candidate_blob(r)
        if family and "climatology" in blob and dt1:
            s -= 1.5

        return (tier, -s, str(r.get("table") or ""))

    return sorted(results or [], key=score)


def _fetch_datasets_by_tables(store: SQLServerStore, tables: list[str]) -> list[dict[str, Any]]:
    """Look up datasets by table names — hits the in-memory cache.

    Tables not present in the cache (e.g. deprecated/stale KB entries) are
    silently skipped — this prevents stale ChromaDB references from surfacing
    datasets that no longer exist in the live catalog.
    """
    if not tables:
        return []
    cache = _catalog_cache
    if not cache.rows:
        cache.ensure_loaded(store)
    tables_set = {str(t).strip() for t in tables if t}
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for r in cache.rows:
        tbl = str(r.get("table_name") or "")
        if tbl in tables_set and tbl not in seen:
            seen.add(tbl)
            out.append(_row_to_dataset_dict(r))
    # Log any KB tables not found in current catalog (stale entries)
    missing = tables_set - seen
    if missing:
        log.debug("_fetch_datasets_by_tables: %d KB tables not in live catalog: %s",
                  len(missing), sorted(missing)[:5])
    return out


def _expand_query_text_for_kb(q: str) -> str:
    """Minimal expansion for KB semantic search — no hardcoded domain synonyms."""
    return (q or "").strip()


def _kb_semantic_table_scores(
    kb: KBBackend,
    *,
    query: str,
    tables: list[str],
    doc_types: tuple[str, ...] = ("dataset", "variable"),
    k_per_type: int = 250,
) -> dict[str, float]:
    if not tables:
        return {}
    qx = _expand_query_text_for_kb(query)
    tables_set = {t for t in tables if t}
    out: dict[str, float] = {}

    for dt in doc_types:
        where_filtered = {"doc_type": dt, "table": {"$in": list(tables_set)}}
        try:
            hits = kb.query(qx, k=min(k_per_type, max(25, len(tables_set) * 2)), where=where_filtered)
        except Exception:
            hits = kb.query(qx, k=k_per_type, where={"doc_type": dt})

        for h in hits or []:
            meta = h.get("metadata") or {}
            t = str(meta.get("table") or "")
            if t not in tables_set:
                continue
            dist = h.get("distance")
            try:
                d = float(dist) if dist is not None else 1e9
            except Exception:
                d = 1e9
            s = 1.0 / (1.0 + d)
            if s > out.get(t, 0.0):
                out[t] = s

    return out


def _roi_lex_score(r: dict, terms: list[str]) -> int:
    blob = " ".join([
        str(r.get("table", "") or r.get("table_name", "")),
        str(r.get("name", "") or r.get("dataset_short_name", "")),
        str(r.get("title", "") or r.get("dataset_name", "")),
        str(r.get("description", "")),
    ]).lower()
    return sum(1 for t in terms if t and t in blob)


def _roi_rank_key(
    r: dict,
    *,
    sem_scores: dict[str, float],
    terms: list[str],
    roi_lat1: float,
    roi_lat2: float,
    roi_lon1: float,
    roi_lon2: float,
    roi_area: float,
) -> tuple[float, float, float, float]:
    t = str(r.get("table") or "")
    sem = float(sem_scores.get(t, 0.0))
    lex = float(_roi_lex_score(r, terms))
    try:
        ov = _bbox_overlap_area(
            float(r.get("lat_min")), float(r.get("lat_max")),
            float(r.get("lon_min")), float(r.get("lon_max")),
            roi_lat1, roi_lat2, roi_lon1, roi_lon2,
        )
    except Exception:
        ov = 0.0
    ov_frac = float(ov) / float(roi_area) if roi_area > 0 else 0.0
    bbox = _bbox_area(r.get("lat_min"), r.get("lat_max"), r.get("lon_min"), r.get("lon_max"))
    return (-sem, -lex, -ov_frac, bbox)


def _bare_query(q: str) -> str:
    """Strip sensor/make words from a query to get the bare variable name."""
    result = (q or "").lower()
    for word in ("satellite", "model", "assimilation", "observation", "in-situ", "insitu"):
        result = result.replace(word, "").strip()
    return result.strip() or q


def _strip_sensor_words(q: str) -> str:
    return _bare_query(q)


# ---------------------------------------------------------------------------
# catalog_search_kb_first — KB semantic + cache-backed augmentation
# ---------------------------------------------------------------------------

def catalog_search_kb_first(args: CatalogSearchKBFArgs, ctx: dict) -> dict:
    """KB-first semantic dataset discovery with cache-backed SQL augmentation."""
    cache, store = _get_cache(ctx)
    kb: KBBackend | None = ctx.get("kb")

    q = (args.query or "").strip()
    bare_q = _bare_query(q)

    # 1) SQL-first: search the cache with the bare query
    sql_rows = _search_rows(cache.rows, bare_q) if bare_q else []
    sql_datasets = _deduplicate_to_datasets(sql_rows, bare_q, limit=args.limit * 2)

    # 2) KB semantic search
    kb_tables: list[str] = []
    kb_scores: dict[str, float] = {}
    if kb and bare_q:
        try:
            kb_k = max(50, args.limit * 10)
            hits = kb.query(_expand_query_text_for_kb(bare_q), k=kb_k)
            for h in hits or []:
                meta = h.get("metadata") or {}
                t = str(meta.get("table") or "")
                if not t:
                    continue
                dist = h.get("distance")
                try:
                    d = float(dist) if dist is not None else 1e9
                except Exception:
                    d = 1e9
                s = 1.0 / (1.0 + d)
                if s > kb_scores.get(t, 0.0):
                    kb_scores[t] = s
                    if t not in kb_tables:
                        kb_tables.append(t)
        except Exception as e:
            log.warning("KB search failed: %s", e)

    # Fetch KB-matched datasets from cache
    kb_datasets = _fetch_datasets_by_tables(store, kb_tables)
    for d in kb_datasets:
        tbl = str(d.get("table") or "")
        if tbl in kb_scores:
            d["kb_score"] = kb_scores[tbl]

    # 3) Merge: KB results first (semantic relevance), then SQL augmentation
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for d in kb_datasets:
        t = str(d.get("table") or "")
        if t and t not in seen:
            seen.add(t)
            merged.append(d)
    for d in sql_datasets:
        t = str(d.get("table") or "")
        if t and t not in seen:
            seen.add(t)
            # SQL match score already set as kb_score via _deduplicate_to_datasets
            merged.append(d)

    if not merged:
        # Fallback: use sql_datasets directly
        merged = sql_datasets

    # 4) Make/sensor filter
    if args.make or args.sensor:
        merged = [r for r in merged if _matches_make_sensor(r, make=args.make, sensor=args.sensor)]

    # 5) Post-rank
    merged = _post_rank_catalog_results(
        merged, query=bare_q, dt1=args.dt1, dt2=args.dt2,
        lat1=args.lat1, lat2=args.lat2, lon1=args.lon1, lon2=args.lon2,
    )
    merged = merged[:args.limit]

    selected = merged[0] if merged else None
    alternates = merged[1:6] if len(merged) > 1 else []

    return {
        "tool": "catalog.search_kb_first",
        "query": {
            "q": q, "lat1": args.lat1, "lat2": args.lat2,
            "lon1": args.lon1, "lon2": args.lon2,
            "dt1": args.dt1, "dt2": args.dt2,
            "make": args.make, "sensor": args.sensor, "limit": args.limit,
        },
        "selected": selected,
        "alternates": alternates,
        "results": merged,
    }


# ---------------------------------------------------------------------------
# catalog_search_variables — cache-backed
# ---------------------------------------------------------------------------

def catalog_search_variables(args: CatalogSearchVariablesArgs, ctx: dict) -> dict:
    cache, _ = _get_cache(ctx)
    q = (args.query or "").strip()
    if not q:
        return {"results": [], "selected": None, "alternates": [], "total_returned": 0}

    ql = q.lower()
    table_hint = (args.table_hint or "").strip() or None

    rows = cache.rows
    if table_hint:
        rows = [r for r in rows if str(r.get("table_name") or "").strip() == table_hint]

    matched = []
    for r in rows:
        var = str(r.get("variable") or "").lower()
        ln = str(r.get("long_name") or "").lower()
        kw = str(r.get("keywords") or "").lower()
        ds_name = str(r.get("dataset_name") or "").lower()
        ds_short = str(r.get("dataset_short_name") or "").lower()
        if ql in var or ql in ln or ql in kw or ql in ds_name or ql in ds_short:
            matched.append(r)

    def _vreason(var: str | None, long_name: str | None) -> str:
        v = (var or "").lower()
        ln = (long_name or "").lower()
        if v == ql or ln == ql:
            return "exact"
        if ql in v or ql in ln:
            return "partial"
        return "metadata"

    def _vscore(r: dict) -> float:
        var = str(r.get("variable") or "").lower()
        ln = str(r.get("long_name") or "").lower()
        s = 0.0
        if var == ql: s += 5.0
        elif ql in var: s += 3.0
        if ln == ql: s += 4.0
        elif ql in ln: s += 2.0
        return s

    matched.sort(key=lambda r: -_vscore(r))
    matched = matched[:args.limit]

    results = []
    for r in matched:
        var = r.get("variable")
        ln = r.get("long_name")
        results.append({
            "table": r.get("table_name"),
            "variable": var,
            "long_name": ln,
            "keywords": r.get("keywords"),
            "unit": r.get("unit"),
            "dataset_id": r.get("dataset_id"),
            "dataset_short_name": r.get("dataset_short_name"),
            "dataset_title": r.get("dataset_name"),
            "reason": _vreason(var, ln),
        })

    selected = results[0] if results else None
    alternates = results[1:6] if len(results) > 1 else []
    return {
        "query": q,
        "table_hint": table_hint,
        "selected": selected,
        "alternates": alternates,
        "results": results,
        "total_returned": len(results),
    }


# ---------------------------------------------------------------------------
# dataset_metadata — still uses SQL (needs full metadata, References, etc.)
# ---------------------------------------------------------------------------

def dataset_metadata(args: DatasetMetadataArgs, ctx: dict) -> dict:
    store = _get_store(ctx)
    table = args.table.strip()
    cache, _ = _get_cache(ctx)
    rows = [r for r in cache.rows if str(r.get("table_name") or "").strip() == table]
    if not rows:
        return {"metadata": []}

    # Fetch references from dbo.tblDataset_References via Dataset_ID
    ds_id = rows[0].get("dataset_id")
    refs = []
    if ds_id is not None:
        try:
            with store.engine.begin() as conn:
                refs = conn.execute(
                    text("SELECT TOP 20 Reference_ID AS ReferenceId, Reference FROM dbo.tblDataset_References WHERE Dataset_ID=:ds_id ORDER BY Reference_ID"),
                    {"ds_id": int(ds_id)}
                ).mappings().all()
        except Exception:
            refs = []

    # Build dataset-level metadata from first row + all variables
    first = rows[0]
    variables = [
        {"variable": r.get("variable"), "long_name": r.get("long_name"), "unit": r.get("unit")}
        for r in sorted(rows, key=lambda r: str(r.get("variable") or ""))
    ]
    md = {
        "TableName": table,
        "DatasetId": first.get("dataset_id"),
        "ShortName": first.get("dataset_short_name"),
        "DatasetName": first.get("dataset_name"),
        "Description": first.get("description"),
        "Make": first.get("make"),
        "Sensor": first.get("sensor"),
        "SpatialResolution": first.get("spatial_resolution"),
        "TemporalResolution": first.get("temporal_resolution"),
        "LatMin": first.get("lat_min"),
        "LatMax": first.get("lat_max"),
        "LonMin": first.get("lon_min"),
        "LonMax": first.get("lon_max"),
        "DepthMin": first.get("depth_min"),
        "DepthMax": first.get("depth_max"),
        "References": [dict(x) for x in refs] if refs else [],
        "Variables": variables,
    }
    return {"metadata": [md]}


# ---------------------------------------------------------------------------
# list_variables — cache-backed
# ---------------------------------------------------------------------------

def list_variables(args: ListVariablesArgs, ctx: dict) -> dict:
    cache, _ = _get_cache(ctx)
    table = args.table.strip()
    rows = [r for r in cache.rows if str(r.get("table_name") or "").strip() == table]
    rows.sort(key=lambda r: str(r.get("variable") or ""))
    out = [
        {"variable": r.get("variable"), "long_name": r.get("long_name"), "unit": r.get("unit")}
        for r in rows
    ]
    return {"variables": out}


# ---------------------------------------------------------------------------
# count_datasets — cache-backed
# ---------------------------------------------------------------------------

def count_datasets(args: CountDatasetsArgs, ctx: dict) -> dict:
    cache, _ = _get_cache(ctx)
    n = len({r.get("dataset_id") for r in cache.rows if r.get("dataset_id") is not None})
    return {"count": n}


# ---------------------------------------------------------------------------
# dataset_summary — cache-backed
# ---------------------------------------------------------------------------

def dataset_summary(args: DatasetSummaryArgs, ctx: dict) -> dict:
    cache, store = _get_cache(ctx)
    q = (args.query or "").strip()
    if not q:
        return {"matches": [], "selected": None, "total_matches": 0, "truncated": False}

    max_vars = int(args.max_variables)
    max_matches = int(getattr(args, "max_matches", 10))

    # Exact match: table name or dataset short/long name
    ql = q.lower()
    exact_tables: list[str] = []
    seen_for_exact: set[str] = set()
    for r in cache.rows:
        tbl = str(r.get("table_name") or "")
        ds_short = str(r.get("dataset_short_name") or "").lower()
        ds_name = str(r.get("dataset_name") or "").lower()
        if tbl and tbl not in seen_for_exact:
            if ql == tbl.lower() or ql == ds_short or ql == ds_name:
                exact_tables.append(tbl)
                seen_for_exact.add(tbl)

    def _build_match(table: str) -> dict | None:
        rows = [r for r in cache.rows if str(r.get("table_name") or "").strip() == table]
        if not rows:
            return None
        first = rows[0]
        vars_sorted = sorted(rows, key=lambda r: str(r.get("variable") or ""))
        variables = [
            {"variable": r.get("variable"), "long_name": r.get("long_name"), "unit": r.get("unit")}
            for r in vars_sorted[:max_vars]
        ]
        # References from dbo.tblDataset_References via Dataset_ID
        ds_id = rows[0].get("dataset_id") if rows else None
        refs_list = []
        if ds_id is not None:
            try:
                with store.engine.begin() as conn:
                    refs = conn.execute(
                        text("SELECT TOP 20 Reference_ID AS ReferenceId, Reference FROM dbo.tblDataset_References WHERE Dataset_ID=:ds_id ORDER BY Reference_ID"),
                        {"ds_id": int(ds_id)}
                    ).mappings().all()
                refs_list = [dict(r) for r in refs]
            except Exception:
                refs_list = []

        return {
            "table": table,
            "dataset_id": first.get("dataset_id"),
            "short_name": first.get("dataset_short_name"),
            "description": first.get("description"),
            "keywords": first.get("keywords"),
            "source": first.get("data_source"),
            "spatial_resolution": first.get("spatial_resolution"),
            "temporal_resolution": first.get("temporal_resolution"),
            "lat_min": first.get("lat_min"),
            "lat_max": first.get("lat_max"),
            "lon_min": first.get("lon_min"),
            "lon_max": first.get("lon_max"),
            "depth_min": first.get("depth_min"),
            "depth_max": first.get("depth_max"),
            "variables": variables,
            "references": refs_list,
        }

    if exact_tables:
        matches = [m for t in exact_tables if (m := _build_match(t)) is not None]
        selected = matches[0] if matches else None
        return {"selected": selected, "matches": matches, "total_matches": len(matches), "truncated": False}

    # Fuzzy match: search all fields, deduplicate by table
    matched_rows = _search_rows(cache.rows, q)
    datasets = _deduplicate_to_datasets(matched_rows, q, limit=max_matches)
    total_matches = len(_deduplicate_to_datasets(matched_rows, q, limit=10000))
    truncated = total_matches > max_matches

    matches = [m for d in datasets if (m := _build_match(str(d.get("table") or ""))) is not None]
    selected = matches[0] if matches else None
    return {
        "selected": selected,
        "matches": matches,
        "total_matches": total_matches,
        "truncated": truncated,
    }
