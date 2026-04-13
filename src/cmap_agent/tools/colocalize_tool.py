from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import base64
import io
import math
import re
from fractions import Fraction
from typing import Any, Literal, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field, model_validator

import pycmap
from pycmap.sample import Sample

from cmap_agent.tools.pycmap_safe import make_pycmap_api
from cmap_agent.tools.cmap_tools import _export_df
from cmap_agent.storage.sqlserver import SQLServerStore
from cmap_agent.tools.catalog_tools import _catalog_cache
from sqlalchemy import text



def _load_source_df_from_artifact(source_artifact: dict[str, Any] | str) -> pd.DataFrame:
    """
    Load a source dataframe from an artifact reference.

    Supported:
      - artifact dict with s3_bucket/s3_key (preferred)
      - s3://bucket/key URI
      - http(s) URL to CSV/parquet
      - local /artifacts/... URL (only when running local backend and file exists)
    """
    import io
    import os
    import urllib.request

    # Normalize
    if isinstance(source_artifact, str):
        ref = source_artifact.strip()
        art = {"url": ref}
    else:
        art = dict(source_artifact)

    url = str(art.get("url") or art.get("uri") or art.get("artifact_url") or art.get("s3_uri") or "").strip()
    s3_bucket = art.get("s3_bucket") or art.get("bucket")
    s3_key = art.get("s3_key") or art.get("key")

    def _read_bytes_to_df(data: bytes, fmt: str) -> pd.DataFrame:
        bio = io.BytesIO(data)
        if fmt == "parquet":
            return pd.read_parquet(bio)
        # csv
        return pd.read_csv(bio)

    # 1) S3 reference by bucket/key
    if s3_bucket and s3_key:
        import boto3  # type: ignore
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=str(s3_bucket), Key=str(s3_key))
        data = obj["Body"].read()
        fmt = "parquet" if str(s3_key).lower().endswith(".parquet") else "csv"
        return _read_bytes_to_df(data, fmt)

    # 2) s3:// URI
    if url.startswith("s3://"):
        m = re.match(r"^s3://([^/]+)/(.+)$", url)
        if not m:
            raise ValueError("Invalid s3 URI")
        import boto3  # type: ignore
        s3 = boto3.client("s3")
        b = m.group(1)
        k = m.group(2)
        obj = s3.get_object(Bucket=b, Key=k)
        data = obj["Body"].read()
        fmt = "parquet" if k.lower().endswith(".parquet") else "csv"
        return _read_bytes_to_df(data, fmt)

    # 3) Local FastAPI artifacts URL -> local path (only meaningful in local mode)
    if url.startswith("/artifacts/"):
        try:
            from cmap_agent.config.settings import settings as _settings
            parts = url.strip("/").split("/", 2)
            if len(parts) >= 3:
                cand = os.path.join(_settings.CMAP_AGENT_ARTIFACT_DIR, parts[1], parts[2])
                if os.path.exists(cand):
                    url = cand
        except Exception:
            pass

    # 4) Local path
    if os.path.exists(url):
        if url.lower().endswith(".parquet"):
            return pd.read_parquet(url)
        return pd.read_csv(url)

    # 5) http(s) URL
    if url.startswith("http://") or url.startswith("https://"):
        with urllib.request.urlopen(url) as r:
            data = r.read()
        fmt = "parquet" if url.lower().endswith(".parquet") else "csv"
        return _read_bytes_to_df(data, fmt)

    raise ValueError("Unsupported artifact reference")


class ColocalizeTarget(BaseModel):
    """Target dataset spec for colocalization via pycmap.Sample()."""

    table: str = Field(..., description="Target CMAP table name, e.g. 'tblSST_AVHRR_OI_NRT'.")
    variables: list[str] = Field(
        ..., description="Target variable name(s) to add (as columns) to the output dataframe."
    )
    dt_tol_days: float | None = Field(
        None,
        description=(
            "Temporal tolerance. For regular datasets this is in days (e.g., 1 for daily products). "
            "For climatology targets (monthly climatology), this is in MONTHS (e.g., 1 means ±1 month). "
            "If omitted, the tool will infer a reasonable default from the catalog TemporalResolution."
        ),
    )
    lat_tol_deg: float | None = Field(
        None,
        description=(
            "Latitude tolerance (degrees). Example: 0.25 for 1/4-degree products. "
            "If omitted, the tool will infer from the catalog SpatialResolution (deg/km/arc-second)."
        ),
    )
    lon_tol_deg: float | None = Field(
        None,
        description=(
            "Longitude tolerance (degrees). Example: 0.25 for 1/4-degree products. "
            "If omitted, the tool will infer from the catalog SpatialResolution (deg/km/arc-second)."
        ),
    )
    depth_tol_m: float | None = Field(
        None,
        description=(
            "Depth tolerance (meters). If omitted, defaults to 5 m when BOTH source and target have a depth field; otherwise 0."
        ),
    )


class ColocalizeArgs(BaseModel):
    """Arguments for cmap.colocalize tool.

    Source can be either:
      - a CMAP table (small / in-situ recommended), or
      - user-provided CSV / Parquet content.
    """

    # Source selection
    source_table: str | None = Field(
        None,
        description=(
            "Source CMAP table name (preferred for small in-situ tables). "
            "If provided, the tool will download the table and use it as the source dataframe."
        ),
    )

    # Inline file content (Swagger-friendly)
    source_csv: str | None = Field(
        None,
        description=(
            "CSV text content for a custom source dataset. Must include columns time, lat, lon (and optional depth)."
        ),
    )
    source_parquet_b64: str | None = Field(
        None,
        description=(
            "Base64-encoded Parquet file bytes for a custom source dataset. "
            "Must include columns time, lat, lon (and optional depth)."
        ),
    )
    
    # Artifact reference (preferred for larger inputs)
    source_artifact: dict[str, Any] | str | None = Field(
        None,
        description=(
            "Reference to an uploaded source dataset stored as an artifact. "
            "This can be an artifact dict returned by /files/presign_upload (recommended) "
            "or a string like an s3://bucket/key URI or an http(s) URL."
        ),
    )

    column_map: dict[str, str] | None = Field(
        None,
        description=(
            "Optional mapping if your source columns are named differently. Keys can include: time, lat, lon, depth."
        ),
    )

    # Targets
    targets: list[ColocalizeTarget] = Field(
        ..., description="One or more target datasets to colocalize against."
    )

    # Output
    # Default to CSV for easier UX (open in Excel / Sheets, view in browser, etc.).
    # Users can override with format="parquet" when they want typed columns + better compression.
    format: Literal["parquet", "csv"] = "csv"

    @model_validator(mode="after")
    def _validate_source(self):
        provided = [
            self.source_table is not None,
            self.source_csv is not None,
            self.source_parquet_b64 is not None,
            self.source_artifact is not None,
        ]
        if sum(provided) != 1:
            raise ValueError(
                "Provide exactly one source: source_table OR source_artifact OR source_csv OR source_parquet_b64."
            )
        return self


def _normalize_source_df(df: pd.DataFrame, column_map: dict[str, str] | None) -> pd.DataFrame:
    df = df.copy()
    cmap = column_map or {}
    # Rename to CMAP standard column names if needed
    rename_map = {v: k for k, v in cmap.items() if v in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)

    required = {"time", "lat", "lon"}
    missing = sorted([c for c in required if c not in df.columns])
    if missing:
        raise ValueError(
            f"Source dataframe missing required columns: {missing}. Required: time, lat, lon (depth optional)."
        )

    # Ensure time is string-like for pycmap.Sample (dateutil parser)
    if pd.api.types.is_datetime64_any_dtype(df["time"].dtype):
        # Convert to UTC-ish ISO strings
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    else:
        # Convert timestamps/objects to strings
        df["time"] = df["time"].apply(lambda x: x.isoformat() if hasattr(x, "isoformat") else str(x))

    # Coerce numeric columns when possible
    for col in ["lat", "lon", "depth"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _as_bool(v: Any) -> bool:
    """Best-effort coercion for bit/int/bool/text fields coming from SQL Server."""
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, (int, float)):
        return v != 0
    s = str(v).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def _clean_resolution_str(s: str | None) -> str:
    """Normalize catalog resolution strings.

    Handles common encoding artifacts (e.g., "Â°"), casing, and whitespace.
    """

    if not s:
        return ""
    out = str(s)
    # Fix common mojibake for degree symbol.
    out = out.replace("Â°", "°")
    out = out.replace("Â", "")
    out = out.replace("º", "°")
    out = out.strip().lower()
    out = re.sub(r"\s+", " ", out)
    return out


def _parse_number_token(tok: str) -> float | None:
    """Parse a token that may be a float, int, or fraction like '1/4'."""

    t = tok.strip()
    if not t:
        return None
    # fraction like 1/4
    if re.fullmatch(r"\d+\s*/\s*\d+", t):
        try:
            return float(Fraction(t.replace(" ", "")))
        except Exception:
            return None
    # plain number
    try:
        return float(t)
    except Exception:
        return None


def _infer_spatial_tolerances_deg(
    spatial_res: str, median_lat: float
) -> Optional[Tuple[float, float]]:
    """Infer (lat_tol_deg, lon_tol_deg) from CatalogDatasets.SpatialResolution.

    Notes:
    - CMAP stores mixed units: degrees (e.g., "1/4° X 1/4°") and kilometers (e.g., "70km X 70km").
    - This returns tolerances in **degrees** (as expected by pycmap.Sample()).
    - For degree-based resolutions, we return the degree spacing directly (no cosine correction).
    - For km-based resolutions, we convert km->deg using ~111 km/deg and correct longitude by cos(lat).
    - "Irregular" returns None.
    """
    s = _clean_resolution_str(spatial_res)
    if not s:
        return None

    if "irregular" in s:
        return None

    # e.g. "15 arc-second interval grid"
    m = re.search(r"(\d+(?:\.\d+)?)\s*arc-?second", s)
    if m:
        arcsec = float(m.group(1))
        deg = arcsec / 3600.0
        return (deg, deg)

    # km grid, e.g. "70km x 70km" or "9 km x 9 km"
    if "km" in s:
        m = re.search(r"(\d+(?:\.\d+)?)\s*km", s)
        if not m:
            return None
        km = float(m.group(1))
        lat_deg = km / 111.0
        coslat = max(math.cos(math.radians(median_lat)), 0.2)
        lon_deg = km / (111.0 * coslat)
        return (lat_deg, lon_deg)

    # Degree grid, e.g. "1/4° x 1/4°", "1° x 1°"
    def _num_to_float(tok: str) -> float:
        tok = tok.strip()
        if "/" in tok:
            a, b = [t.strip() for t in tok.split("/", 1)]
            return float(a) / float(b)
        return float(tok)

    # allow optional encoding artifacts between number and degree sign
    grid = re.search(
        r"(\d+(?:\.\d+)?|\d+\s*/\s*\d+)\s*(?:°|deg)\s*x\s*(\d+(?:\.\d+)?|\d+\s*/\s*\d+)\s*(?:°|deg)",
        s,
    )
    if grid:
        lat_deg = _num_to_float(grid.group(1))
        lon_deg = _num_to_float(grid.group(2))
        return (lat_deg, lon_deg)

    single = re.search(r"(\d+(?:\.\d+)?|\d+\s*/\s*\d+)\s*(?:°|deg)", s)
    if single:
        deg = _num_to_float(single.group(1))
        return (deg, deg)

    return None

def _infer_temporal_tolerance(temporal_res: str) -> tuple[float | None, bool, str | None]:
    """Infer (dt_tol, is_climatology) from CatalogDatasets.TemporalResolution.

    - For climatology targets, dt_tol is in MONTHS.
    - For regular targets, dt_tol is in DAYS.
    """

    s = _clean_resolution_str(temporal_res)
    if not s:
        return None, False, "No TemporalResolution found; using conservative defaults."

    if "climatology" in s:
        return 1.0, True, None

    # Common named cadences
    if "daily" in s:
        return 1.0, False, None
    if "weekly" in s:
        return 7.0, False, None
    if "monthly" in s:
        return 31.0, False, None
    if "eight day" in s or "eight days" in s:
        return 8.0, False, None
    if "three days" in s:
        return 3.0, False, None
    if "six hourly" in s:
        return 6.0 / 24.0, False, None
    if "hourly" in s:
        return 1.0 / 24.0, False, None

    # Minute/second patterns
    m = re.search(r"(?P<n>\d+)\s*minutes?", s)
    if m:
        n = float(m.group("n"))
        return n / (60.0 * 24.0), False, None
    if "one minute" in s:
        return 1.0 / (60.0 * 24.0), False, None
    m = re.search(r"(?P<n>\d+)\s*seconds?", s)
    if m:
        n = float(m.group("n"))
        return n / 86400.0, False, None
    # fraction seconds like '1/6 s'
    m = re.search(r"(?P<frac>\d+\s*/\s*\d+)\s*s", s)
    if m:
        v = _parse_number_token(m.group("frac"))
        if v is not None:
            return v / 86400.0, False, None

    if "irregular" in s:
        # If the string includes a hint like 'irregular, hourly', use the hinted cadence.
        if "hourly" in s:
            return 1.0 / 24.0, False, "TemporalResolution is irregular; using hourly default based on hint."
        return None, False, "TemporalResolution is 'Irregular'; using conservative defaults unless user specifies tolerances."

    return None, False, f"Unrecognized TemporalResolution '{temporal_res}'; using conservative defaults."


def cmap_colocalize(args: ColocalizeArgs, ctx: dict) -> dict[str, Any]:
    """Colocalize a source dataset with one or more target datasets using pycmap.Sample()."""

    # Create a pycmap client configured for this request. This patches pycmap's config_path
    # to a writable location and avoids pycmap calling sys.exit() on errors.
    api = make_pycmap_api(ctx["cmap_api_key"], base_url=None)

    warnings: list[str] = []
    source_table: Optional[str] = args.source_table if args.source_table else None

    # Load source dataframe
    if args.source_table:
        source_table = args.source_table
        source_table = args.source_table
        # Let pycmap fetch and validate the source dataset (row limits, bounds, etc.).
        df_source = api.get_dataset(args.source_table)
    elif args.source_artifact is not None:
        df_source = _load_source_df_from_artifact(args.source_artifact)
    elif args.source_csv is not None:
        df_source = pd.read_csv(io.StringIO(args.source_csv))
    else:
        raw = base64.b64decode(args.source_parquet_b64.encode("utf-8"))
        df_source = pd.read_parquet(io.BytesIO(raw))

    df_source = _normalize_source_df(df_source, args.column_map)

    source_has_depth = "depth" in df_source.columns

    # Infer tolerances from agent.CatalogDatasets (SpatialResolution/TemporalResolution)
    # when not supplied, and add guardrails for climatology temporal units.
    store = ctx.get("store") if isinstance(ctx.get("store"), SQLServerStore) else None
    if store is None:
        try:
            store = SQLServerStore.from_env()
        except Exception:
            store = None

    # Load tolerances from in-memory catalog cache — no SQL round trips needed.
    if store is not None:
        _catalog_cache.ensure_loaded(store)

    meta_by_table: dict[str, dict[str, Any]] = {}
    cache_rows = _catalog_cache.rows
    if cache_rows:
        for table in sorted({t.table for t in args.targets}):
            tbl_rows = [r for r in cache_rows
                        if str(r.get("table_name") or "").strip() == table.strip()]
            first = tbl_rows[0] if tbl_rows else {}
            meta: dict[str, Any] = {
                "SpatialResolution": str(first.get("spatial_resolution") or ""),
                "TemporalResolution": str(first.get("temporal_resolution") or ""),
                "HasDepth": None,  # not in udfCatalog — inferred later from depth bounds
            }
            # Climatology flag still needs SQL (not in udfCatalog)
            if store is not None:
                try:
                    with store.engine.begin() as conn:
                        row2 = conn.execute(
                            text(
                                "SELECT TOP 1 Climatology FROM dbo.tblDatasets WHERE TableName = :table"
                            ),
                            {"table": table},
                        ).mappings().first()
                        if row2 is not None and "Climatology" in row2:
                            meta["Climatology"] = row2["Climatology"]
                except Exception:
                    pass
            meta_by_table[table] = meta

# Representative latitude used when converting km -> lon degrees.
    median_lat = 0.0
    if "lat" in df_source.columns:
        try:
            median_lat = float(df_source["lat"].dropna().median())
        except Exception:
            median_lat = 0.0

    # Build pycmap.Sample targets dict
    targets: dict[str, dict[str, Any]] = {}
    resolved_targets: list[dict[str, Any]] = []

    # Suggest tolerances per target based on catalog metadata when the user omitted them.
    # For climatology targets, default temporal tolerance is 1 month unless user overrides.
    for t in args.targets:
        meta = meta_by_table.get(t.table, {})
        spatial_res = str(meta.get("SpatialResolution") or "")
        temporal_res = str(meta.get("TemporalResolution") or "")

        target_has_depth = _as_bool(meta.get("HasDepth"))
        is_clim_meta = _as_bool(meta.get("Climatology"))

        # Infer temporal tolerance (days for non-climatology; months for climatology)
        dt_inferred, is_clim_inferred, dt_units = _infer_temporal_tolerance(temporal_res)
        is_clim = bool(is_clim_meta or is_clim_inferred or any("clim" in v.lower() for v in t.variables))
        if is_clim:
            dt_units = "months"
            if t.dt_tol_days is None:
                dt_tol = 1.0
            else:
                dt_tol = float(t.dt_tol_days)
        else:
            dt_units = "days"
            dt_tol = float(t.dt_tol_days) if t.dt_tol_days is not None else (
                float(dt_inferred) if dt_inferred is not None else 1.0
            )

        # Infer spatial tolerance in degrees
        inferred_spatial = _infer_spatial_tolerances_deg(spatial_res, median_lat)
        if inferred_spatial is not None:
            lat_inferred, lon_inferred = inferred_spatial
        else:
            lat_inferred, lon_inferred = (None, None)
        if is_clim and (lat_inferred is None or lon_inferred is None):
            # Coarse/default climatology fallback when spatial resolution is missing/unparseable
            lat_inferred = lat_inferred or 1.0
            lon_inferred = lon_inferred or 1.0

        lat_tol = float(t.lat_tol_deg) if t.lat_tol_deg is not None else (
            float(lat_inferred) if lat_inferred is not None else 0.25
        )
        lon_tol = float(t.lon_tol_deg) if t.lon_tol_deg is not None else (
            float(lon_inferred) if lon_inferred is not None else 0.25
        )

        # Clamp to sane bounds; avoid ridiculously tiny or huge tolerances.
        lat_tol = float(min(max(lat_tol, 0.01), 5.0))
        lon_tol = float(min(max(lon_tol, 0.01), 5.0))

        # Infer depth tolerance in meters
        if t.depth_tol_m is None:
            depth_tol = 5.0 if (source_has_depth and target_has_depth) else 0.0
        else:
            depth_tol = float(t.depth_tol_m)
        if not (source_has_depth and target_has_depth):
            if t.depth_tol_m not in (None, 0, 0.0):
                warnings.append(
                    f"Ignoring provided depth tolerance for '{t.table}' because either the source or target has no depth column."
                )
            depth_tol = 0.0

        targets[t.table] = {
            "variables": t.variables,
            "tolerances": [dt_tol, lat_tol, lon_tol, depth_tol],
        }

        # User-facing preview of what will be used
        dt_note = (
            "climatology: dt tol is in months" if is_clim else "dt tol is in days"
        )
        sp_note = None
        if t.lat_tol_deg is None and lat_inferred is not None and abs(lat_tol - float(lat_inferred)) < 1e-9:
            sp_note = f"inferred from SpatialResolution='{spatial_res}'"
        if t.lon_tol_deg is None and lon_inferred is not None and abs(lon_tol - float(lon_inferred)) < 1e-9:
            sp_note = f"inferred from SpatialResolution='{spatial_res}'"

        # Guardrails: discourage accidental huge tolerances
        if is_clim and dt_tol > 12:
            warnings.append(
                f"Temporal tolerance for climatology target '{t.table}' is {dt_tol} months (default is 1)."
            )

        resolved_targets.append(
            {
                "table": t.table,
                "variables": t.variables,
                "has_depth": bool(target_has_depth),
                "is_climatology": bool(is_clim),
                "tolerances": {
                    "dt": {"value": dt_tol, "units": dt_units, "note": dt_note},
                    "lat_tol_deg": lat_tol,
                    "lon_tol_deg": lon_tol,
                    "depth_tol_m": depth_tol,
                    "spatial_note": sp_note,
                },
                "tolerance_sources": {
                    "dt": ("user" if t.dt_tol_days is not None else ("catalog" if (not is_clim and dt_inferred is not None and abs(dt_tol - float(dt_inferred)) < 1e-9) else ("default"))),
                    "lat": ("user" if t.lat_tol_deg is not None else ("catalog" if (lat_inferred is not None and abs(lat_tol - float(lat_inferred)) < 1e-9) else "default")),
                    "lon": ("user" if t.lon_tol_deg is not None else ("catalog" if (lon_inferred is not None and abs(lon_tol - float(lon_inferred)) < 1e-9) else "default")),
                    "depth": ("user" if t.depth_tol_m is not None else "default"),
                },                "provided": {
                    "dt": t.dt_tol_days is not None,
                    "lat": t.lat_tol_deg is not None,
                    "lon": t.lon_tol_deg is not None,
                    "depth": t.depth_tol_m is not None,
                },
                "inferred": {
                    "dt": t.dt_tol_days is None and (is_clim or dt_inferred is not None),
                    "lat": t.lat_tol_deg is None and lat_inferred is not None and abs(lat_tol - float(lat_inferred)) < 1e-9,
                    "lon": t.lon_tol_deg is None and lon_inferred is not None and abs(lon_tol - float(lon_inferred)) < 1e-9,
                    "depth": t.depth_tol_m is None,
                },
            }
        )

    # Run colocalization
    # IMPORTANT: Do not pass `servers` (let pycmap defaults choose), and force
    # replaceWithMonthlyClimatolog=False as requested.
    try:
        # Use positional arg for replaceWithMonthlyClimatolog for broad compatibility.
        df_out = Sample(df_source, targets, False)
    except TypeError:
        # Fallback for alternative signatures.
        df_out = Sample(df_source, targets, replaceWithMonthlyClimatolog=False)
    except Exception as e:
        # Surface useful context in logs (CloudWatch) to debug container vs local drift.
        try:
            import pandas as _pd
            import pyarrow as _pa
            import pycmap as _pc
            vinfo = {
                "pandas": getattr(_pd, "__version__", "unknown"),
                "pyarrow": getattr(_pa, "__version__", "unknown"),
                "pycmap": getattr(_pc, "__version__", "unknown"),
            }
        except Exception:
            vinfo = {}
        logger.exception(
            "cmap.colocalize failed (source_table=%s, targets=%s, versions=%s): %s",
            source_table,
            [t.get("table") for t in targets] if isinstance(targets, list) else targets,
            vinfo,
            str(e),
        )
        raise

    export = _export_df(
        df_out,
        ctx["thread_id"],
        prefix="colocalize",
        fmt=args.format,
    )

    # Merge any warnings from export (e.g., artifact publish/preview post-steps)
    # with colocalization-specific warnings.
    export_warnings = export.get("warnings") if isinstance(export, dict) else None
    merged_warnings: list[str] = []
    if isinstance(export_warnings, list):
        merged_warnings.extend([str(x) for x in export_warnings])

    # Include a machine-readable summary of what we actually did. This helps the
    # agent explain the chosen variable names and default tolerances in its
    # natural-language response.
    # NOTE: Earlier iterations of this tool supported custom on-disk source
    # inputs (CSV/Parquet). The current schema does not expose those fields, so
    # we must not assume they exist on `args` (otherwise we raise AttributeError
    # after the heavy sampling step, causing the agent to retry and duplicate
    # work).
    export["resolved"] = {
        "source": {
            "table": source_table,
            "custom_csv": bool(getattr(args, "source_csv", None)),
            "custom_parquet": bool(getattr(args, "source_parquet_b64", None)),
            "columns": list(df_source.columns),
            "rows": int(len(df_source)),
        },
        "targets": resolved_targets,
    }

    if warnings:
        merged_warnings.extend([str(x) for x in warnings])
    if merged_warnings:
        export["warnings"] = merged_warnings

    export["pycmap_code"] = (
        "import pycmap\n"
        "from pycmap.sample import Sample\n\n"
        "# 'source_df' is a pandas.DataFrame with columns time, lat, lon (and optional depth)\n"
        f"targets = {targets}\n"
        "df = Sample(source_df, targets, False)\n"
    )
    return export
