from __future__ import annotations

import re
from typing import Any, Optional, Literal

from pydantic import BaseModel, Field
from sqlalchemy import text

from cmap_agent.rag.chroma_kb import ChromaKB
from cmap_agent.storage.sqlserver import SQLServerStore


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}

# Minimal synonym expansion to boost recall for common CMAP queries.
_SYNONYMS: dict[str, list[str]] = {
    "sst": ["sea surface temperature"],
    "chl": ["chlorophyll"],
    "chla": ["chlorophyll"],
    "chlor_a": ["chlorophyll"],
    "mld": ["mixed layer depth"],
    "par": ["photosynthetically active radiation"],

    # Precipitation
    "precip": ["precipitation", "rain", "rainfall"],
    "precipitation": ["rain", "rainfall", "total precipitation", "rain rate", "tp"],
    "rainfall": ["precipitation", "rain rate"],
    "tp": ["total precipitation", "precipitation", "rainfall"],

    # Nutrients / biogeochemistry
    "nutrient": [
        "nutrients",
        "nitrate", "nitrite", "ammonium", "phosphate", "silicate",
        "no3", "no2", "nh4", "po4", "si",
        "din", "dissolved inorganic nitrogen",
        "dip", "dissolved inorganic phosphorus",
    ],
    "nutrients": [
        "nitrate", "nitrite", "ammonium", "phosphate", "silicate",
        "no3", "no2", "nh4", "po4", "si",
        "din", "dissolved inorganic nitrogen",
        "dip", "dissolved inorganic phosphorus",
    ],
    "no3": ["nitrate"],
    "no2": ["nitrite"],
    "nh4": ["ammonium"],
    "po4": ["phosphate"],
    "si": ["silicate"],
}


def _tokenize(q: str) -> list[str]:
    q = (q or "").strip().lower()
    if not q:
        return []
    # Keep alphanumerics and underscores; replace other punctuation with spaces.
    q = re.sub(r"[^a-z0-9_]+", " ", q)
    raw = [t for t in q.split() if t]

    toks: list[str] = []
    for t in raw:
        if t in _STOPWORDS:
            continue
        # Many users express variable names in snake_case (e.g., total_precipitation).
        # Split underscores into separate tokens *unless* the token is a known synonym key
        # (e.g., chlor_a) where the underscore is meaningful.
        if '_' in t and t not in _SYNONYMS:
            for p in t.split('_'):
                if not p or p in _STOPWORDS:
                    continue
                if len(p) >= 2:
                    toks.append(p)
            continue
        if len(t) >= 2:
            toks.append(t)

    return toks



def _expand_tokens(tokens: list[str]) -> list[str]:
    """Normalize/de-duplicate tokens.

    NOTE: We do *not* explode synonyms here because that would turn synonyms into
    additional required AND-terms. Synonyms are handled inside _build_token_where
    as OR-alternatives per token.
    """
    out: list[str] = []
    seen: set[str] = set()
    for t in tokens or []:
        tt = (t or "").strip()
        if not tt:
            continue
        if tt.lower() in seen:
            continue
        out.append(tt)
        seen.add(tt.lower())
    return out

def _expand_query_text_for_kb(q: str) -> str:
    """Expand common synonyms into the KB query to improve recall.

    This keeps the original query intact while appending synonym phrases, e.g.:
    - "nutrients" -> adds nitrate/phosphate/etc.
    - "sst" -> adds "sea surface temperature"
    """
    q = (q or "").strip()
    if not q:
        return q
    toks = _tokenize(q)
    extra_phrases: list[str] = []
    seen: set[str] = set()
    for t in toks:
        for phrase in _SYNONYMS.get(t, []):
            p = (phrase or "").strip()
            if p and p.lower() not in seen:
                extra_phrases.append(p)
                seen.add(p.lower())
    if not extra_phrases:
        return q
    return q + " ; " + " ; ".join(extra_phrases)



def _build_token_where(
    *,
    tokens: list[str],
    fields: list[str],
    param_prefix: str,
) -> tuple[str, dict[str, str]]:
    """Build a SQL WHERE fragment that requires each token to match (with synonyms as OR-alternatives).

    Semantics:
      (token0 OR synonyms(token0)) AND (token1 OR synonyms(token1)) AND ...

    Returns: (where_sql, params)
    """
    params: dict[str, str] = {}
    if not tokens:
        return "", params

    token_clauses: list[str] = []
    for i, tok in enumerate(tokens):
        alts = [tok] + list(_SYNONYMS.get(tok, []))
        alt_clauses: list[str] = []
        for j, alt in enumerate(alts):
            p = f"{param_prefix}{i}_{j}"
            params[p] = f"%{alt}%"
            ors = " OR ".join([f"{f} LIKE :{p}" for f in fields])
            alt_clauses.append(f"({ors})")
        token_clauses.append("(" + " OR ".join(alt_clauses) + ")")

    return " AND ".join(token_clauses), params

class CatalogSearchArgs(BaseModel):
    query: str = Field(..., description="Free text query, e.g. 'chlorophyll', 'SST', 'ARGO oxygen'")
    limit: int = Field(10, ge=1, le=50, description="Max number of results")


class CatalogSearchVariablesArgs(BaseModel):
    query: str = Field(..., description="Free text query for variables, e.g. 'chlorophyll', 'sst', 'oxygen'")
    table_hint: str | None = Field(None, description="Optional dataset table name to restrict search")
    limit: int = Field(10, ge=1, le=50, description="Max number of results")

class DatasetMetadataArgs(BaseModel):
    table: str

class ListVariablesArgs(BaseModel):
    table: str


class CountDatasetsArgs(BaseModel):
    """No-arg tool: returns the number of datasets in the cached catalog."""
    pass


class DatasetSummaryArgs(BaseModel):
    query: str = Field(..., description="Dataset table name or a text query (short name/title/keywords).")
    max_variables: int = Field(25, ge=0, le=200, description="Max number of variables to return")
    max_matches: int = Field(5, ge=1, le=25, description="If the query is not an exact match, return up to this many matching datasets")

def _get_store(ctx: dict) -> SQLServerStore:
    store = ctx.get("store")
    if isinstance(store, SQLServerStore):
        return store
    return SQLServerStore.from_env()

def catalog_search(args: CatalogSearchArgs, ctx: dict) -> dict:
    store = _get_store(ctx)
    q = (args.query or "").strip()
    if not q:
        return {"results": []}

    like_full = f"%{q}%"
    # Include Make and Sensor in the metadata search so queries like "model" or "satellite"
    # can match without needing special-cased SQL.
    fields = [
        "TableName",
        "ShortName",
        "DatasetName",
        "Description",
        "Keywords",
        "Make",
        "Sensor",
    ]

    tokens = _expand_tokens(_tokenize(q))
    where_tokens_and, token_params = _build_token_where(tokens=tokens, fields=fields, param_prefix="t")

    # Prefer AND-over-tokens for multi-word queries; if it returns no rows, fall back to
    # a broader OR-over-full-query search.
    where_sql = (
        f"({where_tokens_and})" if where_tokens_and else "(" + " OR ".join([f"{f} LIKE :like_full" for f in fields]) + ")"
    )

    sql = text(
        f"""
        SELECT TOP (:limit)
            TableName AS [table],
            DatasetId AS [dataset_id],
            ShortName AS [name],
            DatasetName AS [title],
            Make AS [make],
            Sensor AS [sensor],
            Description AS [description],
            Keywords AS [keywords]
        FROM agent.CatalogDatasets
        WHERE {where_sql}
        ORDER BY
            CASE
                WHEN TableName = :q THEN 0
                WHEN ShortName = :q THEN 1
                WHEN DatasetName = :q THEN 2
                WHEN TableName LIKE :like_full THEN 3
                WHEN ShortName LIKE :like_full THEN 4
                WHEN DatasetName LIKE :like_full THEN 5
                ELSE 6
            END,
            UpdatedAt DESC
        """
    )

    with store.engine.begin() as conn:
        params = {"limit": args.limit, "like_full": like_full, "q": q}
        params.update(token_params)
        rows = conn.execute(sql, params).mappings().all()

        # Fallback: if token-AND search yields no results, use a broader OR search.
        if not rows and where_tokens_and:
            where_or = " OR ".join([f"{f} LIKE :like_full" for f in fields])
            sql2 = text(
                f"""
                SELECT TOP (:limit)
                    TableName AS [table],
                    DatasetId AS [dataset_id],
                    ShortName AS [name],
                    DatasetName AS [title],
                    Make AS [make],
                    Sensor AS [sensor],
                    Description AS [description],
                    Keywords AS [keywords]
                FROM agent.CatalogDatasets
                WHERE {where_or}
                ORDER BY
                    CASE
                        WHEN TableName = :q THEN 0
                        WHEN ShortName = :q THEN 1
                        WHEN DatasetName = :q THEN 2
                        ELSE 3
                    END,
                    UpdatedAt DESC
                """
            )
            rows = conn.execute(sql2, {"limit": args.limit, "like_full": like_full, "q": q}).mappings().all()
    ql = q.lower()

    def _reason(table: str | None, name: str | None, title: str | None) -> str:
        t = (table or "").lower()
        n = (name or "").lower()
        ti = (title or "").lower()
        if t == ql or n == ql or ti == ql:
            return "exact"
        if ql and (ql in t or ql in n or ql in ti):
            return "partial"
        return "metadata"

    results: list[dict] = []
    for r in rows:
        table = r["table"]
        name = r.get("name") or r.get("title") or table
        title = r.get("title")
        results.append(
            {
                "table": table,
                "name": name,
                "title": title,
                "dataset_id": r.get("dataset_id"),
                "make": r.get("make"),
                "sensor": r.get("sensor"),
                "description": (r.get("description") or "")[:400],
                "reason": _reason(table, name, title),
            }
        )

    selected = results[0] if results else None
    alternates = results[1:6] if len(results) > 1 else []

    # Keep backwards-compatible `results`, but add structured fields so the agent can
    # reliably present "chosen" vs "also found".
    return {
        "query": q,
        "selected": selected,
        "alternates": alternates,
        "results": results,
        "total_returned": len(results),
    }


class CatalogSearchROIArgs(BaseModel):
    """Search datasets that overlap a Region Of Interest (ROI) bounding box."""

    lat1: float = Field(..., description="Southern latitude (degrees).")
    lat2: float = Field(..., description="Northern latitude (degrees).")
    lon1: float = Field(..., description="Western longitude (degrees).")
    lon2: float = Field(..., description="Eastern longitude (degrees).")
    make: Optional[str] = Field(
        None, description="Optional dataset Make filter (e.g., Observation, Model, Assimilation)."
    )
    sensor: Optional[str] = Field(None, description="Optional dataset Sensor filter (e.g., in-Situ, Satellite).")
    rank_mode: Literal["mixed", "overlap_area", "bbox_area"] = Field(
    "mixed",
    description="ROI ranking strategy. 'mixed' interleaves global/large-coverage products with tight regional products; "
                "'overlap_area' prefers datasets with the largest ROI overlap; 'bbox_area' reproduces the legacy behavior "
                "(smallest bbox first).",
    )

    limit: int = Field(10, ge=1, le=100, description="Maximum number of datasets to return.")


def _norm_lat_bounds(lat1: float, lat2: float) -> tuple[float, float]:
    a = float(lat1)
    b = float(lat2)
    return (a, b) if a <= b else (b, a)


def _lon_intervals(lon1: float, lon2: float) -> list[tuple[float, float]]:
    """Return 1 or 2 lon intervals in [-180, 180] handling date-line crossing.

    lon1 is treated as "west" and lon2 as "east". If lon1 > lon2, we assume the ROI crosses the date-line.
    """

    a = float(lon1)
    b = float(lon2)

    # Special-case: some catalog extents represent *full globe* coverage using end points
    # that differ by ~360 degrees (e.g. [0, 360], [-180, 180], [10, 370]).
    # If we naively normalize the endpoints into [-180, 180], both may map to the same
    # value (e.g. 0 and 360 both map to 0), which collapses the interval to ~0° and
    # causes ROI overlap checks to incorrectly exclude global datasets.
    if abs(b - a) >= 359.0:
        return [(-180.0, 180.0)]
    # Normalize into [-180, 180]
    def norm(x: float) -> float:
        y = ((x + 180.0) % 360.0) - 180.0
        # Keep +180 as +180 (not -180) for readability
        if y == -180.0 and x > 0:
            return 180.0
        return y

    a = norm(a)
    b = norm(b)
    if a <= b:
        return [(a, b)]
    # Date-line crossing: [a, 180] U [-180, b]
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
    """Conservative overlap check for ROI vs dataset bounds.

    - If dataset bounds are missing, we do not treat it as overlapping.
    - Handles ROI date-line crossing.
    - Handles dataset date-line crossing if lon_min > lon_max.
    """

    if ds_lat_min is None or ds_lat_max is None or ds_lon_min is None or ds_lon_max is None:
        return False

    lat1, lat2 = _norm_lat_bounds(roi_lat1, roi_lat2)
    if float(ds_lat_max) < lat1 or float(ds_lat_min) > lat2:
        return False

    roi_iv = _lon_intervals(roi_lon1, roi_lon2)

    # Dataset may also cross date-line
    # Always normalize lon bounds and handle dateline crossing consistently.
    ds_iv = _lon_intervals(float(ds_lon_min), float(ds_lon_max))
    for a in roi_iv:
        for b in ds_iv:
            if _intervals_overlap(a, b):
                return True
    return False



def _interval_width(iv: tuple[float, float]) -> float:
    return max(0.0, float(iv[1]) - float(iv[0]))


def _lon_span(lon1: float, lon2: float) -> float:
    """Return longitudinal span in degrees, handling date-line crossing and global extents."""
    return float(sum(_interval_width(iv) for iv in _lon_intervals(lon1, lon2)))


def _bbox_area(
    lat_min: float | None,
    lat_max: float | None,
    lon_min: float | None,
    lon_max: float | None,
) -> float:
    """Approximate bbox area in degree^2 (lat_span * lon_span), robust to date-line/global bounds."""
    try:
        if lat_min is None or lat_max is None or lon_min is None or lon_max is None:
            return 1e18
        lat_span = abs(float(lat_max) - float(lat_min))
        lon_span = _lon_span(float(lon_min), float(lon_max))
        return float(lat_span * lon_span)
    except Exception:
        return 1e18


def _lon_overlap_width(
    roi_lon1: float,
    roi_lon2: float,
    ds_lon_min: float,
    ds_lon_max: float,
) -> float:
    roi_iv = _lon_intervals(roi_lon1, roi_lon2)
    ds_iv = _lon_intervals(ds_lon_min, ds_lon_max)
    ov = 0.0
    for a in roi_iv:
        for b in ds_iv:
            ov += max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    return float(ov)


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
    lat_ov = max(0.0, min(float(ds_lat_max), lat2) - max(float(ds_lat_min), lat1))
    if lat_ov <= 0.0:
        return 0.0
    lon_ov = _lon_overlap_width(roi_lon1, roi_lon2, float(ds_lon_min), float(ds_lon_max))
    if lon_ov <= 0.0:
        return 0.0
    return float(lat_ov * lon_ov)


def _roi_area(lat1: float, lat2: float, lon1: float, lon2: float) -> float:
    a, b = _norm_lat_bounds(lat1, lat2)
    return float(abs(b - a) * _lon_span(lon1, lon2))


def _matches_make_sensor(record: dict, make: Optional[str] = None, sensor: Optional[str] = None) -> bool:
    """Filter helper for KB results.

    Records may store fields directly (e.g., {"make": "...", "sensor": "..."})
    or nested under a "metadata" dict. Matching is case-insensitive substring.
    """
    if not make and not sensor:
        return True
    if not isinstance(record, dict):
        return False

    # Prefer explicit keys, but also look under record["metadata"] if present.
    meta = record.get("metadata")
    if isinstance(meta, dict):
        merged = {**meta, **record}
    else:
        merged = record

    def _norm(x: object) -> str:
        return str(x).strip().lower() if x is not None else ""

    if make:
        rec_make = _norm(merged.get("make"))
        if not rec_make or _norm(make) not in rec_make:
            return False

    if sensor:
        rec_sensor = _norm(merged.get("sensor"))
        if not rec_sensor or _norm(sensor) not in rec_sensor:
            return False

    return True


def catalog_search_roi(args: CatalogSearchROIArgs, ctx: dict) -> dict:
    """ROI-only dataset search using SQL-cached extents."""

    store = _get_store(ctx)
    lat1, lat2 = _norm_lat_bounds(args.lat1, args.lat2)
    # Fetch a superset from SQL and filter robustly in Python (handles date-line crossing).
    sql = text(
        """
        SELECT
            DatasetId AS dataset_id,
            TableName AS [table],
            ShortName AS [name],
            DatasetName AS [title],
            Description AS [description],
            Make AS [make],
            Sensor AS [sensor],
            LatMin AS lat_min,
            LatMax AS lat_max,
            LonMin AS lon_min,
            LonMax AS lon_max
        FROM agent.CatalogDatasets
        """
    )
    with store.engine.connect() as conn:
        rows = [dict(r._mapping) for r in conn.execute(sql)]

    filtered: list[dict[str, Any]] = []
    for r in rows:
        if not _matches_make_sensor(r.get("make"), args.make):
            continue
        if not _matches_make_sensor(r.get("sensor"), args.sensor):
            continue
        if _bbox_overlaps(
            r.get("lat_min"),
            r.get("lat_max"),
            r.get("lon_min"),
            r.get("lon_max"),
            lat1,
            lat2,
            args.lon1,
            args.lon2,
        ):
            filtered.append(r)

    # Rank ROI overlaps. We compute both (1) dataset bbox area (coverage size) and (2) overlap with the ROI.
    roi_area = _roi_area(args.lat1, args.lat2, args.lon1, args.lon2)
    roi_area = roi_area if roi_area > 0 else 1.0

    for r in filtered:
        r["_bbox_area"] = _bbox_area(r.get("lat_min"), r.get("lat_max"), r.get("lon_min"), r.get("lon_max"))
        try:
            r["_overlap_area"] = _bbox_overlap_area(
                float(r.get("lat_min")),
                float(r.get("lat_max")),
                float(r.get("lon_min")),
                float(r.get("lon_max")),
                args.lat1,
                args.lat2,
                args.lon1,
                args.lon2,
            )
        except Exception:
            r["_overlap_area"] = 0.0
        r["_overlap_frac"] = float(r["_overlap_area"]) / float(roi_area)

    mode = (args.rank_mode or "mixed").strip().lower()
    if mode == "bbox_area":
        # Legacy: smallest bbox first.
        filtered.sort(key=lambda r: (float(r.get("_bbox_area") or 1e18), float(r.get("dataset_id") or 1e18)))
    elif mode == "overlap_area":
        # Prefer the *largest* overlap with ROI; break ties by broader coverage (so global products don't get starved).
        filtered.sort(
            key=lambda r: (
                -float(r.get("_overlap_frac") or 0.0),
                -float(r.get("_overlap_area") or 0.0),
                -float(r.get("_bbox_area") or 0.0),
                float(r.get("dataset_id") or 1e18),
            )
        )
    else:
        # Mixed: interleave "coverage" ranking with "tight bbox" ranking for diversity.
        by_cov = sorted(
            filtered,
            key=lambda r: (
                -float(r.get("_overlap_frac") or 0.0),
                -float(r.get("_overlap_area") or 0.0),
                -float(r.get("_bbox_area") or 0.0),
                float(r.get("dataset_id") or 1e18),
            ),
        )
        by_tight = sorted(
            filtered,
            key=lambda r: (
                float(r.get("_bbox_area") or 1e18),
                -float(r.get("_overlap_frac") or 0.0),
                -float(r.get("_overlap_area") or 0.0),
                float(r.get("dataset_id") or 1e18),
            ),
        )
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        i = j = 0
        while len(out) < args.limit and (i < len(by_cov) or j < len(by_tight)):
            if i < len(by_cov):
                r = by_cov[i]
                i += 1
                t = str(r.get("table") or "")
                if t and t not in seen:
                    out.append(r)
                    seen.add(t)
                    if len(out) >= args.limit:
                        break
            if j < len(by_tight):
                r = by_tight[j]
                j += 1
                t = str(r.get("table") or "")
                if t and t not in seen:
                    out.append(r)
                    seen.add(t)
        filtered = out
    def _clean_row(r: dict[str, Any]) -> dict[str, Any]:
        # Hide internal ranking helpers (keys prefixed with "_")
        return {k: v for k, v in (r or {}).items() if not str(k).startswith("_")}

    results = [_clean_row(r) for r in filtered[: args.limit]]
    selected = results[0] if results else None
    alternates = results[1:6] if len(results) > 1 else []
    return {
        "query": {
            "roi": {"lat1": args.lat1, "lat2": args.lat2, "lon1": args.lon1, "lon2": args.lon2},
            "make": args.make,
            "sensor": args.sensor,
        },
        "selected": selected,
        "alternates": alternates,
        "results": results,
        "total_returned": len(results),
    }


class CatalogSearchKBFArgs(BaseModel):
    """KB-first dataset search (semantic discovery, then apply constraints)."""

    query: str = Field(..., description="Natural-language query describing the desired data (e.g., 'nutrients', 'nitrate', 'model NO3').")
    lat1: Optional[float] = Field(None, description="Optional ROI southern latitude.")
    lat2: Optional[float] = Field(None, description="Optional ROI northern latitude.")
    lon1: Optional[float] = Field(None, description="Optional ROI western longitude.")
    lon2: Optional[float] = Field(None, description="Optional ROI eastern longitude.")
    make: Optional[str] = Field(None, description="Optional dataset Make filter.")
    sensor: Optional[str] = Field(None, description="Optional dataset Sensor filter.")
    roi_rank_mode: Literal["mixed", "overlap_area", "bbox_area"] = Field(
    "mixed",
    description="When an ROI is provided, controls how ROI candidates are ranked for backfill. "
                "'mixed' interleaves global coverage with tight regional products; 'overlap_area' prefers maximum overlap; "
                "'bbox_area' reproduces legacy behavior.",
    )

    limit: int = Field(10, ge=1, le=50, description="Maximum number of datasets to return.")
    kb_k: int = Field(30, ge=5, le=500, description="Number of KB matches to consider (dynamic default may increase when ROI is provided).")


def _infer_make_sensor_from_query(q: str) -> tuple[Optional[str], Optional[str]]:
    s = (q or "").lower()
    make = None
    sensor = None
    if "assimilation" in s:
        make = "Assimilation"
    elif "model" in s or "forecast" in s or "reanalysis" in s:
        make = "Model"
    elif "observation" in s or "observations" in s:
        make = "Observation"

    if "satellite" in s:
        sensor = "Satellite"
    elif "in situ" in s or "insitu" in s or "cruise" in s or "ctd" in s:
        sensor = "in-Situ"
    return make, sensor


def _fetch_datasets_by_tables(store: SQLServerStore, tables: list[str]) -> list[dict[str, Any]]:
    if not tables:
        return []
    # Parameterize the IN list safely
    params = {f"t{i}": t for i, t in enumerate(tables)}
    in_clause = ", ".join([f":t{i}" for i in range(len(tables))])
    sql = text(
        f"""
        SELECT
            DatasetId AS dataset_id,
            TableName AS [table],
            ShortName AS [name],
            DatasetName AS [title],
            Description AS [description],
            Make AS [make],
            Sensor AS [sensor],
            LatMin AS lat_min,
            LatMax AS lat_max,
            LonMin AS lon_min,
            LonMax AS lon_max
        FROM agent.CatalogDatasets
        WHERE TableName IN ({in_clause})
        """
    )
    with store.engine.connect() as conn:
        return [dict(r._mapping) for r in conn.execute(sql, params)]


def _kb_semantic_table_scores(
    kb: ChromaKB,
    *,
    query: str,
    tables: list[str],
    doc_types: tuple[str, ...] = ("dataset", "variable"),
    k_per_type: int = 250,
) -> dict[str, float]:
    """Return best semantic score per table for the given query.

    We use (1 / (1 + distance)) as a monotonic similarity proxy (higher is better),
    and take the max score across doc_types (dataset/variable) for each table.
    """
    if not tables:
        return {}
    qx = _expand_query_text_for_kb(query)
    tables_set = {t for t in tables if t}
    out: dict[str, float] = {}

    for dt in doc_types:
        # Try a filtered query first (fast + focused). If the local chromadb version
        # doesn't support the operator form, fall back to an unfiltered query.
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


def catalog_search_kb_first(args: CatalogSearchKBFArgs, ctx: dict) -> dict:
    """Semantic dataset discovery via ChromaKB, then apply ROI / Make / Sensor constraints.

    The goal is to avoid hard-coded SQL branching for every query: we use the KB to find likely
    datasets/variables semantically, then we constrain and rank candidates using structured metadata
    (extents, make, sensor) from the catalog.
    """

    store = _get_store(ctx)
    q = (args.query or "").strip()
    if not q:
        return {"query": "", "results": [], "total_returned": 0}

    make = args.make
    sensor = args.sensor
    if make is None and sensor is None:
        inf_make, inf_sensor = _infer_make_sensor_from_query(q)
        make = inf_make
        sensor = inf_sensor

    use_roi = args.lat1 is not None and args.lat2 is not None and args.lon1 is not None and args.lon2 is not None

    # KB search: query variables (captures scientific terms) and datasets (captures higher-level descriptions).
    kb_error = None
    try:
        kb = ChromaKB.from_settings()
        # Use a larger KB candidate pool when ROI constraints are present,
        # because the ROI filter will eliminate many otherwise-relevant semantic matches.
        kb_k = int(args.kb_k)
        if use_roi:
            kb_k = max(kb_k, min(500, max(100, args.limit * 20)))
        else:
            kb_k = max(kb_k, min(200, max(50, args.limit * 10)))

        var_hits = kb.query(_expand_query_text_for_kb(q), k=kb_k, where={"doc_type": "variable"})
        ds_hits = kb.query(_expand_query_text_for_kb(q), k=kb_k, where={"doc_type": "dataset"})
    except Exception as e:
        kb_error = f"{type(e).__name__}: {e}"
        var_hits = []
        ds_hits = []

    # Aggregate signals per table
    table_score: dict[str, float] = {}
    table_vars: dict[str, list[str]] = {}

    def add_hit(table: str, score: float, var_name: Optional[str] = None) -> None:
        if not table:
            return
        table_score[table] = max(table_score.get(table, 0.0), float(score))
        if var_name:
            table_vars.setdefault(table, [])
            if var_name not in table_vars[table]:
                table_vars[table].append(var_name)

    for h in var_hits:
        meta = h.get("metadata") or {}
        dist = h.get("distance")
        try:
            d = float(dist) if dist is not None else 1e9
        except Exception:
            d = 1e9
        score = 1.0 / (1.0 + d)  # higher is better
        add_hit(str(meta.get("table") or ""), score, str(meta.get("var_name") or meta.get("variable") or ""))

    for h in ds_hits:
        meta = h.get("metadata") or {}
        dist = h.get("distance")
        try:
            d = float(dist) if dist is not None else 1e9
        except Exception:
            d = 1e9
        score = 1.0 / (1.0 + d)
        add_hit(str(meta.get("table") or ""), score, None)

    # If KB recall is empty, fall back to the SQL metadata search so callers still
    # get useful dataset candidates even when the KB index is incomplete or stale.
    if not table_score and not (args.lat1 is not None and args.lat2 is not None and args.lon1 is not None and args.lon2 is not None):
        base = catalog_search(CatalogSearchArgs(query=q, limit=max(int(args.limit or 5) * 6, 20)), ctx)
        results = list(base.get("results") or [])
        # Optional Make/Sensor filter (kept lightweight; improves precision for queries like 'model', 'satellite').
        if make or sensor:
            results = [r for r in results if _matches_make_sensor(r, make=make, sensor=sensor)]
        selected = base.get("selected") if isinstance(base.get("selected"), dict) else (results[0] if results else None)
        alternates = list(base.get("alternates") or [])
        # Trim to requested limit
        if args.limit and args.limit > 0:
            results = results[: int(args.limit)]
            if alternates:
                alternates = alternates[: max(0, int(args.limit) - 1)]
        return {
            "query": {"q": q, "lat1": args.lat1, "lat2": args.lat2, "lon1": args.lon1, "lon2": args.lon2, "make": make, "sensor": sensor, "limit": args.limit},
            "mode": "sql_fallback",
            "selected": selected,
            "alternates": alternates,
            "results": results,
        }

    # Pull dataset rows for candidate tables
    if not table_score and args.lat1 is not None and args.lat2 is not None and args.lon1 is not None and args.lon2 is not None:
            # KB returned no hits for this query, but we still have an ROI. Use an ROI superset from SQL and
            # rank it semantically (when KB is available) so we don't starve global / broad-coverage products.
            roi_limit = min(100, max(50, int(args.limit) * 10))
            roi = catalog_search_roi(
                CatalogSearchROIArgs(
                    lat1=args.lat1,
                    lat2=args.lat2,
                    lon1=args.lon1,
                    lon2=args.lon2,
                    limit=roi_limit,
                    make=make,
                    sensor=sensor,
                    rank_mode=args.roi_rank_mode,
                ),
                ctx,
            )
            roi_rows = roi.get("results", []) or []

            # Semantic score per table (best over dataset+variable KB docs) if KB is healthy.
            sem_scores: dict[str, float] = {}
            if kb_error is None and roi_rows:
                try:
                    kb2 = ChromaKB.from_settings()
                    sem_scores = _kb_semantic_table_scores(
                        kb2, query=q, tables=[str(r.get("table") or "") for r in roi_rows]
                    )
                except Exception:
                    sem_scores = {}

            # Lightweight lexical fallback (helps when KB has gaps).
            terms = [t for t in re.split(r"\W+", q.lower()) if t]

            def _lex_score_row(r: dict) -> int:
                blob = (
                    " ".join(
                        [
                            str(r.get("table", "")),
                            str(r.get("name", "")),
                            str(r.get("title", "")),
                            str(r.get("description", "")),
                        ]
                    )
                ).lower()
                return sum(1 for t in terms if t and t in blob)

            roi_area = _roi_area(args.lat1, args.lat2, args.lon1, args.lon2)
            roi_area = roi_area if roi_area > 0 else 1.0

            def _rank_row(r: dict) -> tuple[float, float, float, float]:
                t = str(r.get("table") or "")
                sem = float(sem_scores.get(t, 0.0))
                lex = float(_lex_score_row(r))
                try:
                    ov = _bbox_overlap_area(
                        float(r.get("lat_min")),
                        float(r.get("lat_max")),
                        float(r.get("lon_min")),
                        float(r.get("lon_max")),
                        args.lat1,
                        args.lat2,
                        args.lon1,
                        args.lon2,
                    )
                except Exception:
                    ov = 0.0
                ov_frac = float(ov) / float(roi_area)
                bbox = _bbox_area(r.get("lat_min"), r.get("lat_max"), r.get("lon_min"), r.get("lon_max"))
                # Higher semantic > higher lexical > better overlap > smaller bbox (tighter) as a last tie-breaker
                return (-sem, -lex, -ov_frac, bbox)

            roi_rows.sort(key=_rank_row)
            results = roi_rows[: args.limit]
            return {
                "tool": "catalog.search_kb_first",
                "query": q,
                "make": make,
                "sensor": sensor,
                "kb_error": kb_error,
                "kb_hits": [],
                "results": results,
                "total_returned": len(results),
                "mode": "roi_semantic_fallback",
            }

    candidate_tables = sorted(table_score.keys(), key=lambda t: table_score[t], reverse=True)

    # Keep some headroom in case ROI filters remove many
    candidate_tables = candidate_tables[: min(len(candidate_tables), 500)]

    ds_rows = _fetch_datasets_by_tables(ctx, candidate_tables)
    by_table = {r.get("table"): r for r in ds_rows if r.get("table")}

    # Apply constraints and build results
    use_roi = args.lat1 is not None and args.lat2 is not None and args.lon1 is not None and args.lon2 is not None
    if use_roi:
        lat1, lat2 = _norm_lat_bounds(args.lat1, args.lat2)
    results: list[dict[str, Any]] = []
    for t in candidate_tables:
        r = by_table.get(t)
        if not r:
            continue
        if not _matches_make_sensor(r.get("make"), make):
            continue
        if not _matches_make_sensor(r.get("sensor"), sensor):
            continue
        if use_roi and not _bbox_overlaps(
            r.get("lat_min"), r.get("lat_max"), r.get("lon_min"), r.get("lon_max"), lat1, lat2, args.lon1, args.lon2
        ):
            continue
        r_out = dict(r)
        r_out["kb_score"] = table_score.get(t, 0.0)
        if table_vars.get(t):
            r_out["matched_variables"] = table_vars[t][:5]
        results.append(r_out)

    # Rank by KB score (descending), then by tighter spatial bbox if ROI is present.
    def rank(r: dict[str, Any]) -> tuple[float, float]:
        kb_score = float(r.get("kb_score") or 0.0)
        if use_roi:
            try:
                area = abs(float(r.get("lat_max")) - float(r.get("lat_min"))) * abs(float(r.get("lon_max")) - float(r.get("lon_min")))
            except Exception:
                area = 1e18
        else:
            area = 0.0
        # Sort: higher kb_score, lower area
        return (-kb_score, area)

    results.sort(key=rank)

    # If ROI was requested and the KB candidate pool yields too few matches,
    # backfill from a fast SQL ROI search (then lightly score by text overlap).
    if use_roi and len(results) < args.limit:
        # Semantic ROI backfill: bring in additional ROI-overlapping datasets and rank them semantically
        # (falling back to light lexical scoring when KB is sparse).
        try:
            roi_limit = min(max(args.limit * 10, 50), 100)
            roi = catalog_search_roi(
                CatalogSearchROIArgs(
                    lat1=args.lat1,
                    lat2=args.lat2,
                    lon1=args.lon1,
                    lon2=args.lon2,
                    limit=roi_limit,
                    make=make,
                    sensor=sensor,
                    rank_mode=args.roi_rank_mode,
                ),
                ctx,
            )
            existing = {r.get("table") for r in results if r.get("table")}
            roi_rows = [
                r for r in (roi.get("results", []) or [])
                if r.get("table") and r.get("table") not in existing
            ]

            sem_scores: dict[str, float] = {}
            if roi_rows:
                try:
                    kb2 = ChromaKB.from_settings()
                    sem_scores = _kb_semantic_table_scores(
                        kb2, query=q, tables=[str(r.get("table") or "") for r in roi_rows]
                    )
                except Exception:
                    sem_scores = {}

            terms = [t for t in re.split(r"\W+", q.lower()) if t]

            def _lex_score_row(r: dict) -> int:
                blob = (
                    " ".join(
                        [
                            str(r.get("table", "")),
                            str(r.get("name", "")),
                            str(r.get("title", "")),
                            str(r.get("description", "")),
                        ]
                    )
                ).lower()
                return sum(1 for t in terms if t and t in blob)

            roi_area = _roi_area(args.lat1, args.lat2, args.lon1, args.lon2)
            roi_area = roi_area if roi_area > 0 else 1.0

            def _rank_row(r: dict) -> tuple[float, float, float, float]:
                t = str(r.get("table") or "")
                sem = float(sem_scores.get(t, 0.0))
                lex = float(_lex_score_row(r))
                try:
                    ov = _bbox_overlap_area(
                        float(r.get("lat_min")),
                        float(r.get("lat_max")),
                        float(r.get("lon_min")),
                        float(r.get("lon_max")),
                        args.lat1,
                        args.lat2,
                        args.lon1,
                        args.lon2,
                    )
                except Exception:
                    ov = 0.0
                ov_frac = float(ov) / float(roi_area)
                bbox = _bbox_area(r.get("lat_min"), r.get("lat_max"), r.get("lon_min"), r.get("lon_max"))
                return (-sem, -lex, -ov_frac, bbox)

            roi_rows.sort(key=_rank_row)

            for r in roi_rows:
                if len(results) >= args.limit:
                    break
                t = str(r.get("table") or "")
                sem = float(sem_scores.get(t, 0.0))
                lex = float(_lex_score_row(r))
                if sem <= 0.0 and lex <= 0.0 and len(results) >= max(1, args.limit // 2):
                    # avoid flooding with low-signal rows once we have some results
                    continue
                r_out = dict(r)
                # Use semantic score as a weak KB score for transparency/consistent sorting upstream.
                r_out["kb_score"] = max(float(r_out.get("kb_score") or 0.0), sem)
                results.append(r_out)
        except Exception:
            pass

    results = results[: args.limit]
    selected = results[0] if results else None
    alternates = results[1:6] if len(results) > 1 else []

    return {
        "query": {
            "q": q,
            "roi": (
                {"lat1": args.lat1, "lat2": args.lat2, "lon1": args.lon1, "lon2": args.lon2} if use_roi else None
            ),
            "make": make,
            "sensor": sensor,
        },
        "selected": selected,
        "alternates": alternates,
        "results": results,
        "total_returned": len(results),
    }


def catalog_search_variables(args: CatalogSearchVariablesArgs, ctx: dict) -> dict:
    """Search variables across the cached catalog.

    This is the core tool for resolving variable names when users provide only
    scientific terms (e.g., "chlorophyll") or abbreviations (e.g., "sst").
    """
    store = _get_store(ctx)
    q = (args.query or "").strip()
    if not q:
        return {"results": []}

    table_hint = (args.table_hint or "").strip() or None

    like_full = f"%{q}%"
    tokens = _expand_tokens(_tokenize(q))

    # Variable search fields
    fields = [
        "v.VarName",
        "v.LongName",
        "v.Keywords",
        "v.Unit",
        "d.ShortName",
        "d.DatasetName",
    ]

    where_tokens_and, token_params = _build_token_where(tokens=tokens, fields=fields, param_prefix="t")

    base_where = (
        f"({where_tokens_and})" if where_tokens_and else "(" + " OR ".join([f"{f} LIKE :like_full" for f in fields]) + ")"
    )
    if table_hint:
        base_where = base_where + " AND v.TableName = :table_hint"

    sql = text(
        f"""
        SELECT TOP (:limit)
            v.TableName AS [table],
            v.VarName AS [variable],
            v.LongName AS [long_name],
            v.Keywords AS [keywords],
            v.Unit AS [unit],
            d.DatasetId AS [dataset_id],
            d.ShortName AS [dataset_short_name],
            d.DatasetName AS [dataset_title]
        FROM agent.CatalogVariables v
        LEFT JOIN agent.CatalogDatasets d
            ON d.TableName = v.TableName
        WHERE {base_where}
        ORDER BY
            CASE
                WHEN v.VarName = :q THEN 0
                WHEN v.LongName = :q THEN 1
                WHEN v.VarName LIKE :like_full THEN 2
                WHEN v.LongName LIKE :like_full THEN 3
                ELSE 4
            END,
            d.UpdatedAt DESC,
            v.VarName
        """
    )

    with store.engine.begin() as conn:
        params = {"limit": args.limit, "like_full": like_full, "q": q, "table_hint": table_hint}
        params.update(token_params)
        rows = conn.execute(sql, params).mappings().all()

        # Fallback: if token-AND yields no results, broaden to OR-over-full-like.
        if not rows and where_tokens_and:
            where_or = " OR ".join([f"{f} LIKE :like_full" for f in fields])
            if table_hint:
                where_or = f"({where_or}) AND v.TableName = :table_hint"
            sql2 = text(
                f"""
                SELECT TOP (:limit)
                    v.TableName AS [table],
                    v.VarName AS [variable],
                    v.LongName AS [long_name],
                    v.Unit AS [unit],
                    d.DatasetId AS [dataset_id],
                    d.ShortName AS [dataset_short_name],
                    d.DatasetName AS [dataset_title]
                FROM agent.CatalogVariables v
                LEFT JOIN agent.CatalogDatasets d
                    ON d.TableName = v.TableName
                WHERE {where_or}
                ORDER BY
                    CASE
                        WHEN v.VarName = :q THEN 0
                        WHEN v.LongName = :q THEN 1
                        ELSE 2
                    END,
                    d.UpdatedAt DESC,
                    v.VarName
                """
            )
            rows = conn.execute(sql2, {"limit": args.limit, "like_full": like_full, "q": q, "table_hint": table_hint}).mappings().all()

    ql = q.lower()

    def _vreason(var: str | None, long_name: str | None) -> str:
        v = (var or "").lower()
        ln = (long_name or "").lower()
        if v == ql or ln == ql:
            return "exact"
        if ql and (ql in v or ql in ln):
            return "partial"
        return "metadata"

    results: list[dict] = []
    for r in rows:
        var = r.get("variable")
        ln = r.get("long_name")
        results.append(
            {
                "table": r.get("table"),
                "variable": var,
                "long_name": ln,
                "keywords": r.get("keywords"),
                "unit": r.get("unit"),
                "dataset_id": r.get("dataset_id"),
                "dataset_short_name": r.get("dataset_short_name"),
                "dataset_title": r.get("dataset_title"),
                "reason": _vreason(var, ln),
            }
        )

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

def dataset_metadata(args: DatasetMetadataArgs, ctx: dict) -> dict:
    store = _get_store(ctx)
    table = args.table.strip()
    with store.engine.begin() as conn:
        ds = conn.execute(text("SELECT * FROM agent.CatalogDatasets WHERE TableName=:t"), {"t": table}).mappings().first()
        vars_ = conn.execute(text("SELECT * FROM agent.CatalogVariables WHERE TableName=:t ORDER BY VarName"), {"t": table}).mappings().all()
        refs = conn.execute(text("SELECT ReferenceId, Reference FROM agent.CatalogDatasetReferences WHERE TableName=:t ORDER BY ReferenceId"), {"t": table}).mappings().all()

    if not ds:
        return {"metadata": []}

    # Return in a style similar to pycmap catalog tool output (list with one dict)
    md = dict(ds)
    md["References"] = [dict(x) for x in refs] if refs else []
    md["Variables"] = [dict(v) for v in vars_] if vars_ else []
    return {"metadata": [md]}

def list_variables(args: ListVariablesArgs, ctx: dict) -> dict:
    store = _get_store(ctx)
    table = args.table.strip()
    with store.engine.begin() as conn:
        rows = conn.execute(text("SELECT VarName, LongName, Unit FROM agent.CatalogVariables WHERE TableName=:t ORDER BY VarName"), {"t": table}).mappings().all()
    out=[{"variable": r["VarName"], "long_name": r.get("LongName"), "unit": r.get("Unit")} for r in rows]
    return {"variables": out}


def count_datasets(args: CountDatasetsArgs, ctx: dict) -> dict:
    store = _get_store(ctx)
    with store.engine.begin() as conn:
        n = conn.execute(text("SELECT COUNT(1) AS n FROM agent.CatalogDatasets")).mappings().first()
    return {"count": int(n["n"]) if n and "n" in n else 0}


def dataset_summary(args: DatasetSummaryArgs, ctx: dict) -> dict:
    """Return a compact overview for one *or more* datasets.

    This is meant to make 'summarize dataset X' possible in a single tool call.
    - If args.query exactly matches a dataset (table/short name/title), return that dataset as `selected`
      and also as the only item in `matches`.
    - If the query is not an exact match, perform a LIKE search and return up to `max_matches` dataset
      summaries in `matches`. In this case, `selected` is set to the first (best-ranked) match so the
      assistant can still produce a summary without failing.
    """
    store = _get_store(ctx)
    q = (args.query or "").strip()
    if not q:
        return {"matches": [], "selected": None, "total_matches": 0, "truncated": False}

    max_vars = int(args.max_variables)
    max_matches = int(getattr(args, "max_matches", 5))
    like_full = f"%{q}%"
    fields = [
        "TableName",
        "ShortName",
        "DatasetName",
        "Description",
        "Keywords",
    ]
    tokens = _expand_tokens(_tokenize(q))
    where_tokens_and, token_params = _build_token_where(tokens=tokens, fields=fields, param_prefix="t")

    def _fetch_vars(conn, table: str):
        if max_vars <= 0:
            return []
        try:
            rows = conn.execute(
                text(
                    """
                    SELECT TOP (:n) VarName, LongName, Unit
                    FROM agent.CatalogVariables
                    WHERE TableName=:t
                    ORDER BY VarName
                    """
                ),
                {"t": table, "n": max_vars},
            ).mappings().all()
        except Exception:
            # Some deployments may not have variable cache populated yet.
            return []
        return [
            {"variable": v.get("VarName"), "long_name": v.get("LongName"), "unit": v.get("Unit")}
            for v in (rows or [])
        ]

    def _fetch_refs(conn, table: str):
        # Keep references small; the assistant can request more if needed.
        rows = conn.execute(
            text(
                """
                SELECT TOP 10 ReferenceId, Reference
                FROM agent.CatalogDatasetReferences
                WHERE TableName=:t
                ORDER BY ReferenceId
                """
            ),
            {"t": table},
        ).mappings().all()
        return [dict(r) for r in (rows or [])]

    def _pack(ds_row, vars_list, refs_list):
        return {
            "table": ds_row.get("TableName"),
            "dataset_id": ds_row.get("DatasetId"),
            "short_name": ds_row.get("ShortName"),
            "title": ds_row.get("DatasetName"),
            "description": ds_row.get("Description"),
            "keywords": ds_row.get("Keywords"),
            "source": ds_row.get("Source"),
            "spatial_resolution": ds_row.get("SpatialResolution"),
            "temporal_resolution": ds_row.get("TemporalResolution"),
            "time_coverage_start": ds_row.get("TimeCoverageStart"),
            "time_coverage_end": ds_row.get("TimeCoverageEnd"),
            "lat_min": ds_row.get("LatMin"),
            "lat_max": ds_row.get("LatMax"),
            "lon_min": ds_row.get("LonMin"),
            "lon_max": ds_row.get("LonMax"),
            "depth_min": ds_row.get("DepthMin"),
            "depth_max": ds_row.get("DepthMax"),
            "updated_at": ds_row.get("UpdatedAt"),
            "variables": vars_list,
            "references": refs_list,
        }

    with store.engine.begin() as conn:
        # 1) Exact match by table/short name/title.
        exact = conn.execute(
            text(
                """
                SELECT TOP 1 *
                FROM agent.CatalogDatasets
                WHERE TableName = :q OR ShortName = :q OR DatasetName = :q
                """
            ),
            {"q": q},
        ).mappings().first()

        if exact:
            table = exact.get("TableName")
            vars_ = _fetch_vars(conn, table)
            refs = _fetch_refs(conn, table)
            selected = _pack(exact, vars_, refs)
            return {"selected": selected, "matches": [selected], "total_matches": 1, "truncated": False}

        # 2) Fallback: LIKE search across common metadata fields.
        # Prefer AND-over-tokens for multi-token queries; fall back to OR-over-full-like.
        where_and = where_tokens_and
        where_or = " OR ".join([f"{f} LIKE :like_full" for f in fields])
        where_sql_count = f"({where_and})" if where_and else f"({where_or})"

        total_row = conn.execute(
            text(
                f"""
                SELECT COUNT(1) AS n
                FROM agent.CatalogDatasets
                WHERE {where_sql_count}
                """
            ),
            {"like_full": like_full, **token_params},
        ).mappings().first()
        total_matches = int(total_row["n"]) if total_row and "n" in total_row else 0

        where_sql = f"({where_and})" if where_and else f"({where_or})"

        rows = conn.execute(
            text(
                f"""
                SELECT TOP (:limit) *
                FROM agent.CatalogDatasets
                WHERE {where_sql}
                ORDER BY
                    CASE
                        WHEN TableName = :q THEN 0
                        WHEN ShortName = :q THEN 1
                        WHEN DatasetName = :q THEN 2
                        WHEN ShortName LIKE :like_full THEN 3
                        WHEN DatasetName LIKE :like_full THEN 4
                        WHEN TableName LIKE :like_full THEN 5
                        ELSE 6
                    END,
                    UpdatedAt DESC
                """
            ),
            {"limit": max_matches, "q": q, "like_full": like_full, **token_params},
        ).mappings().all()

        # Fallback: if token-AND yields nothing, broaden to OR-over-full-like.
        if not rows and where_and:
            rows = conn.execute(
                text(
                    f"""
                    SELECT TOP (:limit) *
                    FROM agent.CatalogDatasets
                    WHERE ({where_or})
                    ORDER BY
                        CASE
                            WHEN TableName = :q THEN 0
                            WHEN ShortName = :q THEN 1
                            WHEN DatasetName = :q THEN 2
                            ELSE 3
                        END,
                        UpdatedAt DESC
                    """
                ),
                {"limit": max_matches, "q": q, "like_full": like_full},
            ).mappings().all()

        if not rows:
            return {"matches": [], "selected": None, "total_matches": 0, "truncated": False}

        matches = []
        for r in rows:
            table = r.get("TableName")
            vars_ = _fetch_vars(conn, table)
            refs = _fetch_refs(conn, table)
            matches.append(_pack(r, vars_, refs))

    # Pick the top-ranked match as selected so the LLM doesn't "give up".
    selected = matches[0] if matches else None
    return {"selected": selected, "matches": matches, "total_matches": total_matches if total_matches else (len(matches) if matches else 0), "truncated": bool(total_matches and total_matches > len(matches))}
