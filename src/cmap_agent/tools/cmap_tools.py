from __future__ import annotations

import uuid
import os
import tempfile
import urllib.request
import re
import difflib
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from sqlalchemy import text

from cmap_agent.artifacts import get_artifact_store
from cmap_agent.tools.errors import ToolInputError
from cmap_agent.tools.pycmap_safe import make_pycmap_api
from cmap_agent.tools.limits import LIMITS
from cmap_agent.tools import viz
from cmap_agent.tools.catalog_tools import (
    _tokenize,
    _expand_tokens,
    _build_token_where,
    catalog_search_variables,
    CatalogSearchVariablesArgs,
    catalog_search,
    CatalogSearchArgs,
)

_ART_STORE = get_artifact_store()


def _allocate_local_path(thread_id: str, filename: str) -> str:
    return _ART_STORE.allocate_local_path(thread_id=thread_id, filename=filename)


def _publish_file(
    local_path: str,
    thread_id: str,
    filename: str,
    artifact_type: str,
    content_type: str | None = None,
) -> dict[str, Any]:
    _pub, artifact = _ART_STORE.publish_file(
        local_path=local_path,
        thread_id=thread_id,
        filename=filename,
        artifact_type=artifact_type,
        content_type=content_type,
    )
    return artifact


def _validate_table_variable(*, store, table: str, variable: str) -> None:
    """Validate that (table, variable) exist in the cached catalog.

    This avoids pycmap's occasional sys.exit() behavior on invalid inputs and
    returns a structured error payload the agent can use to recover.

    Notes:
    - If the catalog variable cache is unavailable/unpopulated, we skip variable validation.
    """

    table = (table or "").strip()
    variable = (variable or "").strip()
    if not table:
        raise ToolInputError("Missing required field: table", code="missing_table")
    if not variable:
        raise ToolInputError("Missing required field: variable", code="missing_variable")

    # Some contexts may not provide a SQL store (e.g., standalone usage).
    if store is None or not hasattr(store, "engine"):
        return

    with store.engine.begin() as conn:
        ds = conn.execute(
            text(
                """
                SELECT TOP 1 TableName, ShortName, DatasetName
                FROM agent.CatalogDatasets
                WHERE TableName = :t
                """
            ),
            {"t": table},
        ).mappings().first()

        if not ds:
            # Suggest datasets using a tokenized search on the table hint.
            fields = ["TableName", "ShortName", "DatasetName", "Description", "Keywords"]
            tokens = _expand_tokens(_tokenize(table))
            where_and, params = _build_token_where(tokens=tokens, fields=fields, param_prefix="s")
            like_full = f"%{table}%"

            where_sql = (
                f"({where_and})"
                if where_and
                else "(" + " OR ".join([f"{f} LIKE :like_full" for f in fields]) + ")"
            )

            rows = conn.execute(
                text(
                    f"""
                    SELECT TOP 5 TableName, ShortName, DatasetName
                    FROM agent.CatalogDatasets
                    WHERE {where_sql}
                    ORDER BY UpdatedAt DESC
                    """
                ),
                {"like_full": like_full, **params},
            ).mappings().all()
            suggestions = [
                {
                    "table": r.get("TableName"),
                    "short_name": r.get("ShortName"),
                    "title": r.get("DatasetName"),
                }
                for r in (rows or [])
            ]

            raise ToolInputError(
                f"Unknown dataset table '{table}'.",
                code="unknown_table",
                details={"table": table},
                suggestions={
                    "datasets": suggestions,
                    "next": "Call catalog.search or catalog.dataset_summary to pick the right dataset table.",
                },
            )

        # Variable cache may not be populated everywhere; treat that as non-fatal.
        try:
            var_row = conn.execute(
                text(
                    """
                    SELECT TOP 1 VarName, LongName, Unit
                    FROM agent.CatalogVariables
                    WHERE TableName = :t AND VarName = :v
                    """
                ),
                {"t": table, "v": variable},
            ).mappings().first()
        except Exception:
            return

        if var_row:
            return

        # Suggest variables within this table.
        vq = variable
        like_v = f"%{vq}%"
        v_fields = ["VarName", "LongName", "Unit"]
        v_tokens = _expand_tokens(_tokenize(vq))
        v_where_and, v_params = _build_token_where(tokens=v_tokens, fields=v_fields, param_prefix="v")

        def _fetch_var_suggestions(where_sql: str, extra_params: dict[str, str]):
            rows = conn.execute(
                text(
                    f"""
                    SELECT TOP 15 VarName, LongName, Unit
                    FROM agent.CatalogVariables
                    WHERE TableName = :t AND ({where_sql})
                    ORDER BY VarName
                    """
                ),
                {"t": table, "like_v": like_v, **extra_params},
            ).mappings().all()
            return [
                {"variable": r.get("VarName"), "long_name": r.get("LongName"), "unit": r.get("Unit")}
                for r in (rows or [])
            ]

        if v_where_and:
            matches = _fetch_var_suggestions(v_where_and, v_params)
            if not matches:
                where_or = " OR ".join([f"{f} LIKE :like_v" for f in v_fields])
                matches = _fetch_var_suggestions(where_or, {})
        else:
            where_or = " OR ".join([f"{f} LIKE :like_v" for f in v_fields])
            matches = _fetch_var_suggestions(where_or, {})

        raise ToolInputError(
            f"Unknown variable '{variable}' for table '{table}'.",
            code="unknown_variable",
            details={"table": table, "variable": variable, "dataset_title": ds.get("DatasetName")},
            suggestions={
                "table": table,
                "variable_matches": matches,
                "next": "Call catalog.search_variables (recommended) or catalog.list_variables to resolve the correct variable name.",
            },
        )


def _norm_name(s: str | None) -> str:
    s = (s or "").strip().lower()
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def _sim(a: str | None, b: str | None) -> float:
    aa = _norm_name(a)
    bb = _norm_name(b)
    if not aa or not bb:
        return 0.0
    if aa == bb:
        return 1.0
    return difflib.SequenceMatcher(None, aa, bb).ratio()


def _pick_best_variable_match(requested: str, candidates: list[dict]) -> tuple[dict | None, float]:
    """Pick the best variable candidate based on VarName/LongName similarity."""
    best = None
    best_score = 0.0
    for c in candidates or []:
        s = max(_sim(requested, c.get("variable")), _sim(requested, c.get("long_name")))
        if s > best_score:
            best_score = s
            best = c
    return best, best_score


def _climatology_period(period: str) -> str:
    """Normalize climatology period tokens to CMAP field names.

    Mirrors pycmap.rest.CMAP._climatology_period.
    """
    p = (period or "").lower().strip()
    if p in ["d", "day", "dayofyear"]:
        return "dayofyear"
    if p in ["w", "week", "weekly"]:
        return "week"
    if p in ["m", "month", "monthly"]:
        return "month"
    if p in ["y", "a", "year", "yearly", "annual"]:
        return "year"
    raise ToolInputError(
        f"Invalid climatology period: {period}",
        code="invalid_climatology_period",
        suggestions={
            "allowed": ["dayofyear", "week", "month", "year"],
            "examples": ["month (10)", "week (1)", "dayofyear (250)", "year (2020)"],
        },
    )


def _validate_climatology_period_value(period: str, value: int) -> None:
    """Basic sanity checks for period_value."""
    if value is None:
        raise ToolInputError("Missing period_value", code="missing_period_value")
    try:
        v = int(value)
    except Exception:
        raise ToolInputError("period_value must be an integer", code="invalid_period_value")

    if period == "month" and not (1 <= v <= 12):
        raise ToolInputError("month period_value must be 1..12", code="invalid_period_value")
    if period == "week" and not (1 <= v <= 53):
        raise ToolInputError("week period_value must be 1..53", code="invalid_period_value")
    if period == "dayofyear" and not (1 <= v <= 366):
        raise ToolInputError("dayofyear period_value must be 1..366", code="invalid_period_value")


def _resolve_table_variable_best_effort(*, store, table: str, variable: str) -> tuple[str, str, dict | None]:
    """Best-effort resolution of (table, variable).

    Why this exists:
    - LLMs sometimes guess table names (e.g., "tblPrecipitation_Global").
    - Users often specify variables in different casing/snake_case.

    This resolver uses the cached SQL catalog to:
      1) resolve a missing/unknown table by searching variables globally
      2) resolve a variable within a known table using case-insensitive + fuzzy matching

    Returns (resolved_table, resolved_variable, resolved_block_or_None).
    """

    table_in = (table or "").strip()
    var_in = (variable or "").strip()
    if store is None or not hasattr(store, "engine"):
        return table_in, var_in, None

    resolved: dict[str, Any] = {
        "original": {"table": table_in, "variable": var_in},
        "resolved": {"table": table_in, "variable": var_in},
        "confidence": None,
        "candidates": [],
    }

    with store.engine.begin() as conn:
        # ---- Table existence (case-insensitive) ----
        ds = conn.execute(
            text(
                """
                SELECT TOP 1 TableName
                FROM agent.CatalogDatasets
                WHERE TableName = :t OR LOWER(TableName) = LOWER(:t)
                """
            ),
            {"t": table_in},
        ).mappings().first()

        if not ds:
            # Unknown table: try resolving via global variable search.
            q = _norm_name(var_in) or _norm_name(table_in)
            if q:
                var_res = catalog_search_variables(
                    CatalogSearchVariablesArgs(query=q, table_hint=None, limit=25),
                    {"store": store},
                )
                cand = list(var_res.get("results") or [])
                best, score = _pick_best_variable_match(var_in or q, cand)
                # Keep a few candidates for transparency/debugging
                resolved["candidates"] = [
                    {
                        "table": c.get("table"),
                        "variable": c.get("variable"),
                        "long_name": c.get("long_name"),
                        "unit": c.get("unit"),
                    }
                    for c in cand[:8]
                ]
                if best and score >= 0.70:
                    table_out = str(best.get("table") or table_in)
                    var_out = str(best.get("variable") or var_in)
                    resolved["resolved"] = {"table": table_out, "variable": var_out}
                    resolved["confidence"] = float(score)
                    return table_out, var_out, resolved

            # As a last resort, suggest datasets via metadata search (helps the LLM recover).
            q2 = _norm_name(var_in) or _norm_name(table_in)
            ds_res = catalog_search(CatalogSearchArgs(query=q2, limit=10), {"store": store}) if q2 else {"results": []}
            resolved["candidates"] = list(ds_res.get("results") or [])[:8]
            return table_in, var_in, resolved

        table_out = str(ds.get("TableName") or table_in)

        # ---- Variable resolution within table (case-insensitive and long name) ----
        # 1) direct / case-insensitive VarName or LongName
        try:
            vr = conn.execute(
                text(
                    """
                    SELECT TOP 1 VarName, LongName
                    FROM agent.CatalogVariables
                    WHERE TableName = :t
                      AND (
                        VarName = :v OR LongName = :v OR
                        LOWER(VarName) = LOWER(:v) OR LOWER(LongName) = LOWER(:v)
                      )
                    """
                ),
                {"t": table_out, "v": var_in},
            ).mappings().first()
        except Exception:
            vr = None

        if vr:
            var_out = str(vr.get("VarName") or var_in)
            resolved["resolved"] = {"table": table_out, "variable": var_out}
            resolved["confidence"] = 1.0
            return table_out, var_out, resolved

        # 2) fuzzy within-table search
        qv = _norm_name(var_in)
        var_res2 = catalog_search_variables(
            CatalogSearchVariablesArgs(query=qv or var_in, table_hint=table_out, limit=40),
            {"store": store},
        )
        cand2 = list(var_res2.get("results") or [])
        best2, score2 = _pick_best_variable_match(var_in or qv, cand2)
        resolved["candidates"] = [
            {
                "table": c.get("table"),
                "variable": c.get("variable"),
                "long_name": c.get("long_name"),
                "unit": c.get("unit"),
            }
            for c in cand2[:10]
        ]

        if best2 and score2 >= 0.70:
            var_out = str(best2.get("variable") or var_in)
            resolved["resolved"] = {"table": table_out, "variable": var_out}
            resolved["confidence"] = float(score2)
            return table_out, var_out, resolved

        # No confident match; return originals but include candidates for downstream error reporting.
        resolved["resolved"] = {"table": table_out, "variable": var_in}
        resolved["confidence"] = float(score2 or 0.0)
        return table_out, var_in, resolved


class SpaceTimeArgs(BaseModel):
    table: str
    variable: str
    dt1: str = Field(..., description="Start datetime (ISO string)")
    dt2: str = Field(..., description="End datetime (ISO string)")
    lat1: float
    lat2: float
    lon1: float
    lon2: float
    depth1: float = Field(-10000.0, description="Min depth (m). Use wide bounds if unknown.")
    depth2: float = Field(10000.0, description="Max depth (m). Use wide bounds if unknown.")
    servers: list[str] = Field(default_factory=lambda: ["rainier"])
    format: Literal["csv", "parquet"] = Field("csv", description="Export file format. Default is csv; set to parquet for parquet export.")
    base_url: str = Field("https://simonscmap.com", description="CMAP base URL")


class TimeSeriesArgs(BaseModel):
    table: str
    variable: str
    dt1: str
    dt2: str
    lat1: float
    lat2: float
    lon1: float
    lon2: float
    depth1: float = Field(0.0, description="Min depth (m). Defaults to 0 m when depth is not specified.")
    depth2: float = Field(5.0, description="Max depth (m). Defaults to 5 m when depth is not specified.")
    interval: str | None = Field(
        None,
        description="Aggregation interval (e.g., 'month', 'week', 'day'). None returns native.",
    )
    servers: list[str] = Field(default_factory=lambda: ["rainier"])
    format: Literal["csv", "parquet"] = Field("csv", description="Export file format. Default is csv; set to parquet for parquet export.")
    base_url: str = "https://simonscmap.com"


class DepthProfileArgs(BaseModel):
    table: str
    variable: str
    dt1: str
    dt2: str
    lat1: float
    lat2: float
    lon1: float
    lon2: float
    servers: list[str] = Field(default_factory=lambda: ["rainier"])
    format: Literal["csv", "parquet"] = Field("csv", description="Export file format. Default is csv; set to parquet for parquet export.")
    base_url: str = "https://simonscmap.com"


class ClimatologyArgs(BaseModel):
    table: str
    variable: str
    period: str = Field(
        ...,
        description=(
            "Climatology aggregation period (case-insensitive). Supported values: "
            "day/dayofyear, week/weekly, month/monthly, year/annual."
        ),
    )
    period_value: int = Field(
        ...,
        description=(
            "Value for the chosen period. Examples: month=10 (October), week=1..53, dayofyear=1..366, year=YYYY."
        ),
    )
    lat1: float
    lat2: float
    lon1: float
    lon2: float
    depth1: float = Field(-10000.0, description="Min depth (m). Use wide bounds if unknown.")
    depth2: float = Field(10000.0, description="Max depth (m). Use wide bounds if unknown.")
    servers: list[str] = Field(default_factory=lambda: ["rainier"])
    format: Literal["csv", "parquet"] = Field("csv", description="Export file format. Default is csv; set to parquet for parquet export.")
    base_url: str = Field("https://simonscmap.com", description="CMAP base URL")


class PlotTimeseriesArgs(TimeSeriesArgs):
    y_column: str | None = Field(None, description="Column to plot. If None, uses variable name.")
    x_column: str = Field("time", description="Time column name")


class PlotMapArgs(BaseModel):
    """Map plotting args.

    Two modes are supported:
      1) CMAP query mode: provide table/variable + space-time bounds; the tool fetches via pycmap.space_time.
      2) Artifact mode: provide `data_artifact` (preferred) or `data_url`; the tool loads the dataframe and plots it.

    Artifact mode is useful for plotting derived products (e.g., on-the-fly climatology results).
    """

    # ---- Artifact mode (preferred for derived / precomputed dataframes) ----
    data_artifact: dict[str, Any] | None = Field(
        None,
        description=(
            "A data artifact object previously returned by cmap.* tools (CSV or parquet). "
            "If provided, the tool loads this artifact and plots it."
        ),
    )
    data_url: str | None = Field(
        None,
        description=(
            "URL to a CSV or parquet file to plot. Use this when you only have a URL string (e.g., presigned S3 URL). "
            "Prefer passing `data_artifact` when available."
        ),
    )

    # ---- CMAP query mode ----
    table: str | None = Field(None, description="CMAP table name (dataset). Required unless data_artifact/data_url is provided.")
    variable: str | None = Field(None, description="Variable short name. Required unless data_artifact/data_url is provided.")
    dt1: str | None = Field(None, description="Start datetime (YYYY-MM-DD). Required in CMAP query mode.")
    dt2: str | None = Field(None, description="End datetime (YYYY-MM-DD). Required in CMAP query mode.")
    lat1: float | None = Field(None, description="Min latitude. Optional in artifact mode; used for bbox/extent.")
    lat2: float | None = Field(None, description="Max latitude. Optional in artifact mode; used for bbox/extent.")
    lon1: float | None = Field(None, description="Min longitude. Optional in artifact mode; used for bbox/extent.")
    lon2: float | None = Field(None, description="Max longitude. Optional in artifact mode; used for bbox/extent.")
    depth1: float = Field(-10000.0, description="Min depth (m). Only used in CMAP query mode.")
    depth2: float = Field(10000.0, description="Max depth (m). Only used in CMAP query mode.")
    servers: list[str] = Field(default_factory=lambda: ["rainier"], description="CMAP server routing list.")
    format: Literal["csv", "parquet"] = Field(
        "csv",
        description="Export file format for CMAP query mode. Default is csv; set to parquet for parquet export.",
    )
    base_url: str = Field("https://simonscmap.com", description="CMAP base URL")

    # ---- Plot options ----
    lat_column: str = Field("lat", description="Latitude column name in the dataframe.")
    lon_column: str = Field("lon", description="Longitude column name in the dataframe.")
    value_column: str | None = Field(None, description="Value column to color by. If None, uses `variable` when set, otherwise auto-detects.")
    projection: Literal[
        "PlateCarree",
        "Robinson",
        "Mollweide",
        "Mercator",
        "EqualEarth",
        "NorthPolarStereo",
        "SouthPolarStereo",
        "Orthographic",
    ] = Field(
        "PlateCarree",
        description=(
            "Cartopy map projection for the output plot. Default PlateCarree. Common alternatives: Robinson, Mollweide, "
            "Mercator, EqualEarth, NorthPolarStereo, SouthPolarStereo, Orthographic."
        ),
    )
    central_longitude: float | None = Field(
        None,
        description="Optional projection central_longitude in degrees. If omitted, uses 0° (or 180° if bbox crosses the antimeridian).",
    )
    central_latitude: float | None = Field(
        None,
        description="Optional central_latitude in degrees (used for Orthographic). If omitted, uses the bbox center latitude.",
    )

    method: Literal["auto", "pcolormesh", "contourf", "scatter"] = Field(
        "auto",
        description=(
            "How to render gridded data. 'auto' chooses a sensible default (typically contourf for small grids, "
            "pcolormesh for large grids). Use 'pcolormesh' for pixel-like rendering, 'contourf' for smooth filled contours, "
            "or 'scatter' for point rendering."
        ),
    )

    @model_validator(mode="after")
    def _validate_mode(self):
        if self.data_artifact is not None or (self.data_url and str(self.data_url).strip()):
            # Artifact mode
            return self

        # CMAP query mode
        missing = [k for k in ("table", "variable", "dt1", "dt2", "lat1", "lat2", "lon1", "lon2") if getattr(self, k) in (None, "")]
        if missing:
            raise ValueError(
                "viz.plot_map requires either `data_artifact`/`data_url` (artifact mode) OR the full CMAP query inputs: "
                + ", ".join(missing)
            )
        return self


def _export_df(df: pd.DataFrame, thread_id: str, prefix: str, fmt: str) -> dict[str, Any]:
    if len(df) > LIMITS.max_export_rows:
        raise ValueError(
            f"Result has {len(df):,} rows which exceeds CMAP_AGENT_MAX_EXPORT_ROWS={LIMITS.max_export_rows:,}. Narrow constraints."
        )

    uid = uuid.uuid4().hex[:10]
    ext = "parquet" if fmt == "parquet" else "csv"
    fname = f"{prefix}_{uid}.{ext}"
    fpath = _allocate_local_path(thread_id, fname)

    if fmt == "parquet":
        df.to_parquet(fpath, index=False)
        content_type = "application/x-parquet"
    else:
        df.to_csv(fpath, index=False)
        content_type = "text/csv"

    warnings: list[str] = []

    # IMPORTANT: artifact publishing is a post-step (e.g., S3 upload + presign).
    # If publishing fails, the data retrieval/sampling already succeeded, so we
    # must return a clean success payload with an unpublished artifact and a warning.
    try:
        artifact = _publish_file(
            fpath,
            thread_id=thread_id,
            filename=fname,
            artifact_type=fmt,
            content_type=content_type,
        )
    except Exception as e:
        # Best-effort local URL for local backend; otherwise keep url=None.
        backend = getattr(_ART_STORE, "__class__", type("X", (), {})).__name__.lower()
        url = None
        path = None
        if "localartifactstore" in backend:
            url = f"/artifacts/{thread_id}/{fname}"
            path = fpath
            backend_name = "local"
        else:
            backend_name = "unpublished"

        warnings.append(
            f"Artifact publish failed ({backend_name}): {type(e).__name__}: {str(e)[:300]}"
        )
        artifact = {
            "type": fmt,
            "filename": fname,
            "url": url,
            "backend": backend_name,
        }
        if path is not None:
            artifact["path"] = path

    preview: list[dict[str, Any]] = []
    try:
        head = df.head(min(len(df), LIMITS.max_inline_rows))
        preview = head.to_dict(orient="records")
    except Exception as e:
        warnings.append(f"Preview generation failed: {type(e).__name__}: {str(e)[:300]}")

    out: dict[str, Any] = {
        "status": "ok_with_warnings" if warnings else "ok",
        "rows": int(len(df)),
        "columns": list(df.columns),
        "preview": preview,
        "artifact": artifact,
    }
    if warnings:
        out["warnings"] = warnings
    return out


def cmap_space_time(args: SpaceTimeArgs, ctx: dict) -> dict[str, Any]:
    store = ctx.get("store")
    table, variable, resolved = _resolve_table_variable_best_effort(store=store, table=args.table, variable=args.variable)
    _validate_table_variable(store=store, table=table, variable=variable)
    api = make_pycmap_api(ctx["cmap_api_key"], base_url=args.base_url)
    df = api.space_time(
        table,
        variable,
        args.dt1,
        args.dt2,
        args.lat1,
        args.lat2,
        args.lon1,
        args.lon2,
        args.depth1,
        args.depth2,
        servers=args.servers,
    )
    export = _export_df(
        df,
        ctx["thread_id"],
        prefix=f"space_time_{table}_{variable}",
        fmt=args.format,
    )
    export["pycmap_code"] = (
        "import pycmap\n"
        f"api = pycmap.API(token=API_KEY, baseURL='{args.base_url}')\n"
        f"df = api.space_time('{table}','{variable}','{args.dt1}','{args.dt2}',{args.lat1},{args.lat2},{args.lon1},{args.lon2},{args.depth1},{args.depth2})\n"
    )
    if resolved and resolved.get('original') != resolved.get('resolved'):
        export['resolved'] = resolved
    return export


def cmap_time_series(args: TimeSeriesArgs, ctx: dict) -> dict[str, Any]:
    store = ctx.get("store")
    table, variable, resolved = _resolve_table_variable_best_effort(store=store, table=args.table, variable=args.variable)
    _validate_table_variable(store=store, table=table, variable=variable)
    api = make_pycmap_api(ctx["cmap_api_key"], base_url=args.base_url)
    df = api.time_series(
        table,
        variable,
        args.dt1,
        args.dt2,
        args.lat1,
        args.lat2,
        args.lon1,
        args.lon2,
        args.depth1,
        args.depth2,
        interval=args.interval,
        servers=args.servers,
    )
    export = _export_df(
        df,
        ctx["thread_id"],
        prefix=f"time_series_{table}_{variable}",
        fmt=args.format,
    )
    export["pycmap_code"] = (
        "import pycmap\n"
        f"api = pycmap.API(token=API_KEY, baseURL='{args.base_url}')\n"
        f"df = api.time_series('{table}','{variable}','{args.dt1}','{args.dt2}',{args.lat1},{args.lat2},{args.lon1},{args.lon2},{args.depth1},{args.depth2}, interval={repr(args.interval)})\n"
    )
    if resolved and resolved.get('original') != resolved.get('resolved'):
        export['resolved'] = resolved
    return export


def cmap_depth_profile(args: DepthProfileArgs, ctx: dict) -> dict[str, Any]:
    store = ctx.get("store")
    table, variable, resolved = _resolve_table_variable_best_effort(store=store, table=args.table, variable=args.variable)
    _validate_table_variable(store=store, table=table, variable=variable)
    api = make_pycmap_api(ctx["cmap_api_key"], base_url=args.base_url)
    df = api.depth_profile(
        table,
        variable,
        args.dt1,
        args.dt2,
        args.lat1,
        args.lat2,
        args.lon1,
        args.lon2,
        servers=args.servers,
    )
    export = _export_df(
        df,
        ctx["thread_id"],
        prefix=f"depth_profile_{table}_{variable}",
        fmt=args.format,
    )
    export["pycmap_code"] = (
        "import pycmap\n"
        f"api = pycmap.API(token=API_KEY, baseURL='{args.base_url}')\n"
        f"df = api.depth_profile('{table}','{variable}','{args.dt1}','{args.dt2}',{args.lat1},{args.lat2},{args.lon1},{args.lon2})\n"
    )
    if resolved and resolved.get('original') != resolved.get('resolved'):
        export['resolved'] = resolved
    return export


def cmap_climatology(args: ClimatologyArgs, ctx: dict) -> dict[str, Any]:
    """Compute on-the-fly climatology for qualified gridded datasets.

    This mirrors the behavior of pycmap.API().climatology(), but we:
      - provide structured ToolInputError payloads for unsupported datasets
      - allow `servers=` routing for the underlying query
      - keep behavior consistent with the rest of cmap.* tools
    """

    store = ctx.get("store")
    table, variable, resolved = _resolve_table_variable_best_effort(store=store, table=args.table, variable=args.variable)
    _validate_table_variable(store=store, table=table, variable=variable)

    api = make_pycmap_api(ctx["cmap_api_key"], base_url=args.base_url)

    clim_period = _climatology_period(args.period)
    _validate_climatology_period_value(clim_period, args.period_value)

    # Qualification checks (matching pycmap.rest.CMAP.climatology)
    try:
        if bool(api.is_climatology(table)):
            raise ToolInputError(
                f"Table {table} is already a climatology dataset; no on-the-fly climatology needed.",
                code="already_climatology",
                details={"table": table, "variable": variable},
                suggestions={"next": "Use cmap.space_time (or viz.plot_map) to subset and plot this climatology dataset."},
            )
    except ToolInputError:
        raise
    except Exception:
        # If the server doesn't expose the climatology flag, proceed (best-effort).
        pass

    try:
        is_grid = api.is_grid(table, variable)
        if is_grid is False:
            raise ToolInputError(
                f"Climatology computation only applies to uniformly gridded datasets. {table}:{variable} appears irregular.",
                code="not_gridded_dataset",
                details={"table": table, "variable": variable},
                suggestions={"next": "Use cmap.space_time or cmap.time_series directly (no climatology aggregation)."},
            )
    except ToolInputError:
        raise
    except Exception:
        # Best-effort: if we cannot determine griddedness, proceed and let the server decide.
        pass

    try:
        if not bool(api.has_field(table, clim_period, servers=args.servers)):
            raise ToolInputError(
                f"Climatology computation is not supported by {table}. (Missing field '{clim_period}'.)",
                code="climatology_not_supported",
                details={"table": table, "variable": variable, "required_field": clim_period},
                suggestions={
                    "next": "Choose a different dataset, or compute climatology offline after downloading a space_time subset.",
                    "hint": "Many satellite/model gridded products include month/week/dayofyear/year helper fields; some do not.",
                },
            )
    except ToolInputError:
        raise
    except Exception:
        # If has_field fails (permissions/latency), proceed best-effort.
        pass

    # NOTE: CMAP longitudes are in [-180, 180]. If the bbox crosses the antimeridian (lon1 > lon2),
    # split into two requests and concatenate.
    def _aggregate_bbox(lat1: float, lat2: float, lon1: float, lon2: float):
        q = (
            "uspAggregate "
            f"'{table}', '{variable}', '{clim_period}', {int(args.period_value)}, "
            f"{float(lat1)}, {float(lat2)}, {float(lon1)}, {float(lon2)}, {float(args.depth1)}, {float(args.depth2)}"
        )
        return api.query(q, servers=args.servers)

    if args.lon1 is not None and args.lon2 is not None and float(args.lon1) > float(args.lon2):
        df1 = _aggregate_bbox(args.lat1, args.lat2, float(args.lon1), 180.0)
        df2 = _aggregate_bbox(args.lat1, args.lat2, -180.0, float(args.lon2))
        df = pd.concat([df1, df2], ignore_index=True)
    else:
        df = _aggregate_bbox(args.lat1, args.lat2, float(args.lon1), float(args.lon2))

    if df is None or len(df) == 0:
        raise ToolInputError(
            "Climatology query returned no rows. Narrow bounds may miss the dataset coverage, or the dataset may not have data for the chosen period_value.",
            code="no_data",
            details={
                "table": table,
                "variable": variable,
                "period": clim_period,
                "period_value": int(args.period_value),
                "bbox": [args.lat1, args.lat2, args.lon1, args.lon2, args.depth1, args.depth2],
            },
            suggestions={
                "next": "Try a larger bounding box, or use catalog.dataset_metadata to confirm dataset coverage/time range.",
            },
        )

    export = _export_df(
        df,
        ctx["thread_id"],
        prefix=f"climatology_{table}_{variable}_{clim_period}_{int(args.period_value)}",
        fmt=args.format,
    )
    export["pycmap_code"] = (
        "import pycmap\n"
        f"api = pycmap.API(token=API_KEY, baseURL='{args.base_url}')\n"
        f"df = api.climatology('{table}','{variable}','{args.period}',{int(args.period_value)},{args.lat1},{args.lat2},{args.lon1},{args.lon2},{args.depth1},{args.depth2})\n"
    )
    export["climatology"] = {"period": clim_period, "period_value": int(args.period_value)}
    if resolved and resolved.get('original') != resolved.get('resolved'):
        export['resolved'] = resolved
    return export


def plot_timeseries(args: PlotTimeseriesArgs, ctx: dict) -> dict[str, Any]:
    store = ctx.get("store")
    table, variable, resolved = _resolve_table_variable_best_effort(store=store, table=args.table, variable=args.variable)
    _validate_table_variable(store=store, table=table, variable=variable)
    # Fetch data once, then export + plot from in-memory dataframe.
    api = make_pycmap_api(ctx["cmap_api_key"], base_url=args.base_url)
    df = api.time_series(
        table,
        variable,
        args.dt1,
        args.dt2,
        args.lat1,
        args.lat2,
        args.lon1,
        args.lon2,
        args.depth1,
        args.depth2,
        interval=args.interval,
        servers=args.servers,
    )

    export = _export_df(
        df,
        ctx["thread_id"],
        prefix=f"time_series_{table}_{variable}",
        fmt=args.format,
    )
    x = args.x_column
    y = args.y_column or variable

    uid = uuid.uuid4().hex[:10]
    fname = f"plot_timeseries_{uid}.png"
    out_png = _allocate_local_path(ctx["thread_id"], fname)

    viz.save_timeseries_png(df, x=x, y=y, out_png=out_png, title=f"{table}:{y}")

    plot_artifact = _publish_file(
        out_png,
        thread_id=ctx["thread_id"],
        filename=fname,
        artifact_type="png",
        content_type="image/png",
    )

    out = {
        "plot": plot_artifact,
        "data_artifact": export["artifact"],
        "pycmap_code": (
            "import pycmap\n"
            f"api = pycmap.API(token=API_KEY, baseURL='{args.base_url}')\n"
            f"df = api.time_series('{table}','{variable}','{args.dt1}','{args.dt2}',{args.lat1},{args.lat2},{args.lon1},{args.lon2},{args.depth1},{args.depth2}, interval={repr(args.interval)})\n"
        ),
    }

    if resolved and resolved.get('original') != resolved.get('resolved'):
        out['resolved'] = resolved
    return out


def _fetch_dataset_meta_for_table(store, table: str) -> dict[str, Any] | None:
    if store is None or not hasattr(store, "engine") or not table:
        return None
    with store.engine.begin() as conn:
        row = conn.execute(
            text(
                """
                SELECT TOP 1
                    TableName,
                    Description,
                    TemporalResolution,
                    SpatialResolution,
                    TimeMin,
                    TimeMax
                FROM agent.CatalogDatasets
                WHERE TableName = :t
                """
            ),
            {"t": table},
        ).mappings().first()
    return dict(row) if row else None


def _infer_temporal_window_days(meta: dict[str, Any] | None) -> int | None:
    if not meta:
        return None
    blob = " ".join([
        str(meta.get("TemporalResolution") or ""),
        str(meta.get("Description") or ""),
    ]).lower()
    m = re.search(r"(\d+)\s*[- ]?day", blob)
    if m:
        try:
            n = int(m.group(1))
            return n if n > 1 else None
        except Exception:
            return None
    m = re.search(r"(\d+)\s*[- ]?week", blob)
    if m:
        try:
            n = int(m.group(1))
            return n * 7 if n > 0 else None
        except Exception:
            return None
    if "weekly" in blob:
        return 7
    return None


def _expand_exact_date_window(dt1: str, dt2: str, window_days: int | None) -> tuple[str, str]:
    if not window_days or window_days <= 1:
        return dt1, dt2
    s1 = str(dt1 or "")[:10]
    s2 = str(dt2 or "")[:10]
    if not s1 or s1 != s2:
        return dt1, dt2
    ts = pd.Timestamp(s1)
    left = max(0, (window_days - 1) // 2)
    right = max(0, (window_days - 1) - left)
    return ((ts - pd.Timedelta(days=left)).date().isoformat(), (ts + pd.Timedelta(days=right)).date().isoformat())


def plot_map(args: PlotMapArgs, ctx: dict) -> dict[str, Any]:
    # ------------------------------------------------------------
    # Artifact mode: plot an existing CSV/parquet dataframe
    # ------------------------------------------------------------
    if args.data_artifact is not None or (args.data_url and str(args.data_url).strip()):

        def _read_df_from_path_or_url(loc: str) -> pd.DataFrame:
            loc = str(loc or "").strip()
            if not loc or loc.lower() in ("none", "null"):
                raise ToolInputError("Missing data artifact location", code="missing_data_location")

            # Local FastAPI static URL -> local filesystem path
            if loc.startswith("/artifacts/"):
                # /artifacts/<thread_id>/<filename>
                parts = loc.strip("/").split("/", 2)
                if len(parts) >= 3:
                    try:
                        from cmap_agent.config.settings import settings as _settings

                        cand = os.path.join(_settings.CMAP_AGENT_ARTIFACT_DIR, parts[1], parts[2])
                        if os.path.exists(cand):
                            loc = cand
                    except Exception:
                        pass

            # If it's a local file, read directly
            if os.path.exists(loc):
                if loc.lower().endswith(".parquet"):
                    return pd.read_parquet(loc)
                return pd.read_csv(loc)

            # Otherwise treat as URL
            if loc.startswith("http://") or loc.startswith("https://"):
                ext = ".parquet" if ".parquet" in loc.lower().split("?", 1)[0] else ".csv"
                tmp_path = os.path.join(tempfile.gettempdir(), f"cmap_agent_plot_input_{uuid.uuid4().hex[:10]}{ext}")
                try:
                    with urllib.request.urlopen(loc, timeout=30) as r, open(tmp_path, "wb") as f:
                        f.write(r.read())
                    if tmp_path.lower().endswith(".parquet"):
                        return pd.read_parquet(tmp_path)
                    return pd.read_csv(tmp_path)
                finally:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

            raise ToolInputError(
                "Could not read data for plotting. Provide a valid local path, a /artifacts/... URL, or an http(s) URL.",
                code="invalid_data_location",
                details={"location": loc},
            )

        art = args.data_artifact or {}
        loc = None
        if isinstance(art, dict):
            loc = (
                art.get("path")
                or art.get("local_path")
                or art.get("filepath")
                or art.get("file")
                or art.get("url")
                or art.get("artifact_url")
                or art.get("download_url")
                or art.get("uri")
            )
        if not loc:
            loc = args.data_url

        df = _read_df_from_path_or_url(loc)

        lat = args.lat_column
        lon = args.lon_column

        # Choose value column
        val = args.value_column or args.variable
        if not val:
            # Prefer the first numeric column that's not a common coordinate/time field
            exclude = {lat, lon, "time", "depth", "month", "week", "dayofyear", "year"}
            numeric_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
            if numeric_cols:
                val = numeric_cols[0]
            else:
                raise ToolInputError(
                    "Could not infer value_column for map coloring. Provide value_column explicitly.",
                    code="missing_value_column",
                    suggestions={"columns": list(df.columns)},
                )

        # Optional bbox (if provided). If missing, cartopy helper will infer from data extents.
        bbox = None
        if all(getattr(args, k) is not None for k in ("lat1", "lat2", "lon1", "lon2")):
            bbox = (float(args.lat1), float(args.lat2), float(args.lon1), float(args.lon2))

        uid = uuid.uuid4().hex[:10]
        fname = f"plot_map_{uid}.png"
        out_png = _allocate_local_path(ctx["thread_id"], fname)

        title = f"{args.table}:{val}" if args.table else str(val)
        ok = viz.save_cartopy_map_png(
            df,
            lat=lat,
            lon=lon,
            val=str(val),
            out_png=out_png,
            title=title,
            bbox=bbox,
            projection=args.projection,
            central_longitude=args.central_longitude,
            central_latitude=args.central_latitude,
            method=args.method,
        )
        render_mode = "cartopy" if ok else "scatter"
        if not ok:
            viz.save_scatter_map_png(df, lat=lat, lon=lon, val=str(val), out_png=out_png, title=title)

        plot_artifact = _publish_file(
            out_png,
            thread_id=ctx["thread_id"],
            filename=fname,
            artifact_type="png",
            content_type="image/png",
        )

        out: dict[str, Any] = {
            "plot": plot_artifact,
            "render_mode": render_mode,
            "source": "artifact",
        }

        if isinstance(args.data_artifact, dict):
            out["data_artifact"] = args.data_artifact

        return out

    # ------------------------------------------------------------
    # CMAP query mode: fetch via pycmap.space_time then plot
    # ------------------------------------------------------------
    store = ctx.get("store")
    table, variable, resolved = _resolve_table_variable_best_effort(store=store, table=str(args.table), variable=str(args.variable))
    _validate_table_variable(store=store, table=table, variable=variable)

    meta = _fetch_dataset_meta_for_table(store, table)
    eff_dt1, eff_dt2 = _expand_exact_date_window(str(args.dt1), str(args.dt2), _infer_temporal_window_days(meta))
    api = make_pycmap_api(ctx["cmap_api_key"], base_url=args.base_url)

    def _space_time_bbox(lat1: float, lat2: float, lon1: float, lon2: float):
        # NOTE: CMAP longitudes are in [-180, 180]. If the requested bbox crosses the antimeridian
        # (lon1 > lon2), split into two requests and concatenate.
        if lon1 is not None and lon2 is not None and float(lon1) > float(lon2):
            df1 = api.space_time(
                table,
                variable,
                str(eff_dt1),
                str(eff_dt2),
                lat1,
                lat2,
                float(lon1),
                180.0,
                args.depth1,
                args.depth2,
                servers=args.servers,
            )
            df2 = api.space_time(
                table,
                variable,
                str(eff_dt1),
                str(eff_dt2),
                lat1,
                lat2,
                -180.0,
                float(lon2),
                args.depth1,
                args.depth2,
                servers=args.servers,
            )
            return pd.concat([df1, df2], ignore_index=True)

        return api.space_time(
            table,
            variable,
            str(eff_dt1),
            str(eff_dt2),
            lat1,
            lat2,
            lon1,
            lon2,
            args.depth1,
            args.depth2,
            servers=args.servers,
        )

    df = _space_time_bbox(float(args.lat1), float(args.lat2), float(args.lon1), float(args.lon2))
    if df is None or df.empty:
        raise ToolInputError(
            "No data were returned for the requested map bounds and time window.",
            code="empty_plot_data",
            details={
                "table": table,
                "variable": variable,
                "requested_dt1": str(args.dt1),
                "requested_dt2": str(args.dt2),
                "effective_dt1": str(eff_dt1),
                "effective_dt2": str(eff_dt2),
            },
            suggestions={
                "next": "Try a broader time window or confirm the dataset temporal resolution.",
            },
        )

    export = _export_df(
        df,
        ctx["thread_id"],
        prefix=f"space_time_{table}_{variable}",
        fmt=args.format,
    )

    lat = args.lat_column
    lon = args.lon_column
    val = args.value_column or variable

    uid = uuid.uuid4().hex[:10]
    fname = f"plot_map_{uid}.png"
    out_png = _allocate_local_path(ctx["thread_id"], fname)

    ok = viz.save_cartopy_map_png(
        df,
        lat=lat,
        lon=lon,
        val=val,
        out_png=out_png,
        title=f"{table}:{val}",
        bbox=(float(args.lat1), float(args.lat2), float(args.lon1), float(args.lon2)),
        projection=args.projection,
        central_longitude=args.central_longitude,
        central_latitude=args.central_latitude,
        method=args.method,
    )
    render_mode = "cartopy" if ok else "scatter"
    if not ok:
        viz.save_scatter_map_png(df, lat=lat, lon=lon, val=val, out_png=out_png, title=f"{table}:{val}")

    plot_artifact = _publish_file(
        out_png,
        thread_id=ctx["thread_id"],
        filename=fname,
        artifact_type="png",
        content_type="image/png",
    )

    out = {
        "plot": plot_artifact,
        "render_mode": render_mode,
        "data_artifact": export["artifact"],
        "pycmap_code": (
            "import pycmap\n"
            "import pandas as pd\n"
            f"api = pycmap.API(token=API_KEY, baseURL='{args.base_url}')\n"
            + (
                (
                    f"df1 = api.space_time('{table}','{variable}','{eff_dt1}','{eff_dt2}',{args.lat1},{args.lat2},{args.lon1},180.0,{args.depth1},{args.depth2})\n"
                    f"df2 = api.space_time('{table}','{variable}','{eff_dt1}','{eff_dt2}',{args.lat1},{args.lat2},-180.0,{args.lon2},{args.depth1},{args.depth2})\n"
                    "df = pd.concat([df1, df2], ignore_index=True)\n"
                )
                if (args.lon1 is not None and args.lon2 is not None and float(args.lon1) > float(args.lon2))
                else f"df = api.space_time('{table}','{variable}','{eff_dt1}','{eff_dt2}',{args.lat1},{args.lat2},{args.lon1},{args.lon2},{args.depth1},{args.depth2})\n"
            )
        ),
        "effective_dt1": str(eff_dt1),
        "effective_dt2": str(eff_dt2),
    }

    if resolved and resolved.get('original') != resolved.get('resolved'):
        out['resolved'] = resolved
    return out
