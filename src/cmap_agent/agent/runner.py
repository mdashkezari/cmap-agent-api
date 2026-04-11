from __future__ import annotations

import json
import re
import difflib
import os
from urllib.parse import urlsplit, urlunsplit
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from cmap_agent.llm.types import LLMMessage
from cmap_agent.tools.catalog_tools import (
    CatalogSearchArgs,
    CatalogSearchKBFArgs,
    CatalogSearchVariablesArgs,
    catalog_search,
    catalog_search_kb_first,
    catalog_search_variables,
    _fetch_datasets_by_tables,
    _bbox_overlaps,
)
from cmap_agent.llm.base import LLMClient
from cmap_agent.tools.registry import ToolRegistry
from cmap_agent.tools.limits import LIMITS


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentToolCallPlan(BaseModel):
    type: str = "tool_call"
    tool_calls: list[ToolCall]


class AgentFinal(BaseModel):
    type: str = "final"
    assistant_message: str
    code: str | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)


def _try_parse_json(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    # Strip fenced code blocks if present
    if text.startswith("```"):
        # remove first fence line and last fence if exists
        parts = text.split("\n")
        parts = parts[1:] if parts else parts
        if parts and parts[-1].strip().startswith("```"):
            parts = parts[:-1]
        text = "\n".join(parts).strip()
    try:
        return json.loads(text)
    except Exception:
        return None


_COLOCALIZE_TOL_KEYS = ("dt_tol_days", "lat_tol_deg", "lon_tol_deg", "depth_tol_m")


def _sanitize_colocalize_arguments(user_message: str, args: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Remove model-guessed tolerances from cmap.colocalize calls.

    The colocalize tool can infer sensible defaults from catalog metadata. In practice,
    the LLM sometimes injects generic tolerances (e.g., 1°) even when the user did not
    ask for any matching window. This sanitization keeps tool behavior stable and
    prevents incorrect overrides.

    Policy:
      - If the user did not explicitly specify a tolerance (via a numeric window + units),
        drop the corresponding tolerance fields from each target.
      - This is per-dimension (time/spatial/depth) so users can override only one.
    """

    msg = (user_message or "").lower()

    # Only treat tolerances as "user-specified" when the message *explicitly* frames them
    # as a matching window. This avoids false positives when users paste catalog metadata
    # (e.g., lists of spatial/temporal resolution strings).
    tol_ctx = r"(?:within|tolerance|tol\b|window|radius|\u00b1|±|plus\s*minus)"

    dt_unit = r"(?:day|days|week|weeks|month|months|hour|hours|hr|hrs|minute|minutes|min|second|seconds|sec|secs|s)"
    dt_pat1 = re.compile(rf"{tol_ctx}[^\d]{{0,20}}\b\d+(?:\.\d+)?\b\s*{dt_unit}\b", re.I)
    dt_pat2 = re.compile(rf"\b\d+(?:\.\d+)?\b\s*{dt_unit}\b[^\w]{{0,20}}{tol_ctx}", re.I)
    keep_dt = bool(dt_pat1.search(msg) or dt_pat2.search(msg))

    spatial_unit = r"(?:deg|degree|degrees|\u00b0|°|km|kilometer|kilometers|arc\s*-?second|arcsecond|arcseconds)"
    spatial_pat1 = re.compile(rf"{tol_ctx}[^\d]{{0,20}}\b\d+(?:\.\d+)?\b\s*{spatial_unit}\b", re.I)
    spatial_pat2 = re.compile(rf"\b\d+(?:\.\d+)?\b\s*{spatial_unit}\b[^\w]{{0,20}}{tol_ctx}", re.I)
    keep_spatial = bool(spatial_pat1.search(msg) or spatial_pat2.search(msg))

    depth_pat1 = re.compile(rf"depth[\s\S]{{0,40}}{tol_ctx}[^\d]{{0,20}}\b\d+(?:\.\d+)?\b\s*(?:m|meter|meters)\b", re.I)
    depth_pat2 = re.compile(rf"{tol_ctx}[\s\S]{{0,40}}depth[^\d]{{0,20}}\b\d+(?:\.\d+)?\b\s*(?:m|meter|meters)\b", re.I)
    depth_pat3 = re.compile(rf"\b\d+(?:\.\d+)?\b\s*(?:m|meter|meters)\b[^\w]{{0,20}}depth[^\w]{{0,20}}{tol_ctx}", re.I)
    keep_depth = bool(depth_pat1.search(msg) or depth_pat2.search(msg) or depth_pat3.search(msg))

    changed = False
    out = dict(args or {})
    targets = out.get("targets")
    if not isinstance(targets, list):
        return out, False

    new_targets: list[dict[str, Any]] = []
    for t in targets:
        if not isinstance(t, dict):
            new_targets.append(t)
            continue
        tt = dict(t)
        # Remove any guessed tolerances unless the user explicitly requested them.
        if not keep_dt:
            for k in ("dt_tol_days",):
                if k in tt:
                    del tt[k]
                    changed = True
        if not keep_spatial:
            for k in ("lat_tol_deg", "lon_tol_deg"):
                if k in tt:
                    del tt[k]
                    changed = True
        if not keep_depth:
            for k in ("depth_tol_m",):
                if k in tt:
                    del tt[k]
                    changed = True
        new_targets.append(tt)

    out["targets"] = new_targets
    return out, changed


def _normalize_artifact(a: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize artifact dicts coming from different layers (tool vs artifact-store).

    - Ensure both `uri` and `url` exist when either is present.
    - Canonicalize `type` to stable high-level categories (image/data) when we can infer it.
      This prevents duplicate artifacts such as {type:'png', ...} and {type:'image', ...}
      referring to the same underlying file.
    """
    out = dict(a or {})

    # Some model responses (and a few older tool contracts) return lightweight
    # placeholders like {"plot_url": "..."} or {"data_artifact_url": "..."}
    # rather than a proper artifact object. Normalize those placeholders into
    # the standard artifact shape so we can dedupe consistently.
    plot_url = out.get("plot_url")
    if isinstance(plot_url, str) and plot_url.strip():
        u = plot_url.strip()
        return {"type": "image", "uri": u, "url": u, "description": out.get("description") or "Plot"}
    data_url = out.get("data_artifact_url") or out.get("data_url")
    if isinstance(data_url, str) and data_url.strip():
        u = data_url.strip()
        return {"type": "data", "uri": u, "url": u, "description": out.get("description") or "Data"}

    if "uri" not in out and "url" in out:
        out["uri"] = out["url"]
    if "url" not in out and "uri" in out:
        out["url"] = out["uri"]

    t = str(out.get("type") or "").strip().lower()
    content_type = str(out.get("content_type") or "").strip().lower()

    # Infer extension from filename or URL path (ignoring query params).
    candidate = str(out.get("filename") or out.get("name") or "")
    if not candidate:
        candidate = str(out.get("uri") or out.get("url") or "")
    ext = ""
    if candidate:
        try:
            path = urlsplit(candidate).path if "://" in candidate else candidate
        except Exception:
            path = candidate
        path = path.split("?", 1)[0].split("#", 1)[0]
        _, _ext = os.path.splitext(path)
        ext = _ext.lstrip(".").lower()

    # Keep already-canonical types unchanged.
    if t in {"image", "data"}:
        return out

    image_ext = {"png", "jpg", "jpeg", "gif", "webp", "svg", "tif", "tiff", "bmp"}
    data_ext = {"parquet", "csv", "json", "nc", "h5", "hdf5", "feather", "arrow"}

    is_image = content_type.startswith("image/") or t in image_ext or ext in image_ext
    is_data = (
        t in data_ext
        or ext in data_ext
        or content_type in {"application/x-parquet", "application/parquet", "application/json"}
        or content_type.startswith("text/")
    )

    if is_image:
        if ext or t:
            out.setdefault("format", ext or t)
        out["type"] = "image"
    elif is_data:
        if ext or t:
            out.setdefault("format", ext or t)
        out["type"] = "data"

    return out


def _strip_url_query(u: str) -> str:
    """Return URL/path without query/fragment, so presigned URLs dedupe correctly."""
    u = str(u or "").strip()
    if not u:
        return ""
    try:
        p = urlsplit(u)
        if p.scheme and p.netloc:
            return urlunsplit((p.scheme, p.netloc, p.path, "", ""))
    except Exception:
        pass
    return u.split("?", 1)[0].split("#", 1)[0]


def _artifact_key(a: Dict[str, Any]) -> tuple[str, str]:
    a = _normalize_artifact(a)
    uri = _strip_url_query(a.get("uri") or a.get("url") or "")
    return (str(a.get("type") or ""), uri)
def _merge_artifacts(primary: list[dict[str, Any]], secondary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge artifacts, keeping order from primary and filling missing entries from secondary."""
    merged: list[dict[str, Any]] = []
    index: dict[tuple[str, str], dict[str, Any]] = {}

    for a in primary:
        na = _normalize_artifact(a)
        k = _artifact_key(na)
        if k not in index:
            index[k] = na
            merged.append(na)
        else:
            # Prefer the first occurrence but fill missing fields.
            base = index[k]
            for kk, vv in na.items():
                if kk not in base or base[kk] in (None, ""):
                    base[kk] = vv

    for a in secondary:
        na = _normalize_artifact(a)
        k = _artifact_key(na)
        if k in index:
            base = index[k]
            for kk, vv in na.items():
                if kk not in base or base[kk] in (None, ""):
                    base[kk] = vv
            continue
        index[k] = na
        merged.append(na)

    return merged


def _coerce_to_plan_or_final(
    obj: dict[str, Any],
    registry: ToolRegistry,
) -> tuple[str, dict[str, Any] | None, AgentToolCallPlan | None]:
    """Normalize model JSON outputs.

    The agent prompt asks the model to return either:
      - {"type": "tool_call", "tool_calls": [{"name": "...", "arguments": {...}}]}
      - {"type": "final", ...}

    In practice, some models occasionally return *direct tool call* JSON like:
      - {"type": "catalog.search", "query": "BULA", "limit": 1}

    This helper converts those direct calls into a tool_call plan so the runner
    can still execute tools and continue.

    Returns: (kind, final_obj, plan)
      kind: 'final' | 'tool_call' | 'other'
    """
    t = (obj.get("type") or "").strip()
    if not t:
        return "other", None, None

    if t == "final":
        return "final", obj, None

    if t == "tool_call":
        try:
            return "tool_call", None, AgentToolCallPlan(**obj)
        except ValidationError:
            return "other", None, None

    # Direct tool call: type == tool name
    if registry.has(t):
        # Some models wrap args in an "arguments" field; support both.
        args = obj.get("arguments") if isinstance(obj.get("arguments"), dict) else None
        if args is None:
            args = {k: v for k, v in obj.items() if k != "type"}
        plan = AgentToolCallPlan(type="tool_call", tool_calls=[ToolCall(name=t, arguments=args or {})])
        return "tool_call", None, plan

    return "other", None, None


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert objects into JSON-serializable equivalents."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        # JSON cannot encode Decimal; float is typically fine for catalog metadata.
        return float(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    # Fallback: stringify unknown objects (e.g., UUID, numpy scalars, etc.)
    try:
        return str(obj)
    except Exception:
        return repr(obj)


def _compact_dataset_summary_item(ds: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "table",
        "dataset_id",
        "short_name",
        "title",
        "description",
        "keywords",
        "source",
        "spatial_resolution",
        "temporal_resolution",
        "time_coverage_start",
        "time_coverage_end",
        "lat_min",
        "lat_max",
        "lon_min",
        "lon_max",
        "depth_min",
        "depth_max",
        "updated_at",
    ]
    out: dict[str, Any] = {k: ds.get(k) for k in keep if ds.get(k) is not None}

    if isinstance(out.get("description"), str) and len(out["description"]) > 800:
        out["description"] = out["description"][:800]

    vars_ = ds.get("variables")
    if isinstance(vars_, list):
        compact_vars = []
        for v in vars_[:15]:
            if isinstance(v, dict):
                vv = {}
                for kk in ("variable", "long_name", "unit"):
                    if v.get(kk) is not None:
                        vv[kk] = v.get(kk)
                compact_vars.append(vv)
            else:
                compact_vars.append(v)
        out["variables"] = compact_vars

    return out


def _compact_colocalize_resolved(resolved: dict[str, Any]) -> dict[str, Any]:
    """Compact the cmap.colocalize `resolved` block for LLM consumption."""
    out: dict[str, Any] = {}

    src = resolved.get("source") if isinstance(resolved, dict) else None
    if isinstance(src, dict):
        out["source"] = {
            "table": src.get("table"),
            "rows": src.get("rows"),
            "columns": (src.get("columns") or [])[:12] if isinstance(src.get("columns"), list) else None,
            "custom_csv": src.get("custom_csv"),
            "custom_parquet": src.get("custom_parquet"),
        }

    targets = resolved.get("targets") if isinstance(resolved, dict) else None
    if isinstance(targets, list):
        compact_targets: list[dict[str, Any]] = []
        for t in targets[:6]:
            if not isinstance(t, dict):
                continue
            tol = t.get("tolerances") if isinstance(t.get("tolerances"), dict) else {}
            dt = tol.get("dt") if isinstance(tol.get("dt"), dict) else {}
            compact_targets.append(
                {
                    "table": t.get("table"),
                    "variables": (t.get("variables") or [])[:12] if isinstance(t.get("variables"), list) else t.get("variables"),
                    "is_climatology": t.get("is_climatology"),
                    "tolerances": {
                        "dt": {"value": dt.get("value"), "units": dt.get("units")},
                        "lat_tol_deg": tol.get("lat_tol_deg"),
                        "lon_tol_deg": tol.get("lon_tol_deg"),
                        "depth_tol_m": tol.get("depth_tol_m"),
                    },
                    "provided": t.get("provided"),
                    "inferred": t.get("inferred"),
                }
            )
        out["targets"] = compact_targets
        out["num_targets"] = int(len(targets))

    return out


def _tool_result_for_llm(name: str, result: dict[str, Any]) -> dict[str, Any]:
    """Create a compact, JSON-safe view of a tool result.

    This compact view is what we feed back to the LLM (and what we keep in tool_trace previews),
    so it must be:
      - informative enough for the LLM to stop repeating the same tool call
      - small enough to stay within prompt limits
      - JSON serializable
    """

    compact: dict[str, Any] = {"tool": name}

    # Special-case: catalog.dataset_summary is the core "summarize a dataset" tool.
    # If we don't include the match descriptions, the LLM will keep calling it.
    if name == "catalog.dataset_summary":
        if "total_matches" in result:
            compact["total_matches"] = result.get("total_matches")
        if "truncated" in result:
            compact["truncated"] = result.get("truncated")

        selected = result.get("selected")
        if isinstance(selected, dict):
            compact["selected"] = _compact_dataset_summary_item(selected)

        matches = result.get("matches")
        if isinstance(matches, list):
            compact["matches"] = [
                _compact_dataset_summary_item(m) for m in matches[:10] if isinstance(m, dict)
            ]

        # Some deployments may return an additional free-form note.
        if isinstance(result.get("note"), str):
            compact["note"] = result.get("note")[:400]

        return _to_jsonable(compact)

    # Special-case: cmap.colocalize should always feed back the resolved tables/vars/tolerances,
    # otherwise the LLM tends to retry or fails to explain what happened.
    if name == "cmap.colocalize":
        if "status" in result:
            compact["status"] = result.get("status")
        if "rows" in result:
            compact["rows"] = result.get("rows")
        if "columns" in result:
            compact["columns"] = (result.get("columns") or [])[:50]
        if "artifact" in result and isinstance(result.get("artifact"), dict):
            compact["artifact_url"] = result["artifact"].get("url")
            compact["artifact_type"] = result["artifact"].get("type")
            compact["artifact_backend"] = result["artifact"].get("backend")
        if "preview" in result:
            try:
                compact["preview"] = (result.get("preview") or [])[:20]
            except Exception:
                pass

        resolved = result.get("resolved")
        if isinstance(resolved, dict):
            compact["resolved"] = _compact_colocalize_resolved(resolved)

        warn = result.get("warnings")
        if isinstance(warn, list):
            compact["warnings"] = [str(x)[:240] for x in warn[:10]]

        return _to_jsonable(compact)

    # Special-case: catalog search tools should return a selected item + alternates so the LLM
    # can present "chosen" vs "also found" consistently.
    if name in {"catalog.search", "catalog.search_variables", "catalog.search_kb_first"}:
        qv = result.get("query")
        if isinstance(qv, (str, dict)):
            compact["query"] = qv
        sel = result.get("selected")
        if isinstance(sel, dict):
            compact["selected"] = sel
        alts = result.get("alternates")
        if isinstance(alts, list):
            compact["alternates"] = alts[:5]
        # Keep a small slice of the full result list too.
        res = result.get("results")
        if isinstance(res, list):
            compact["results"] = res[:10]
        return _to_jsonable(compact)

    # Generic compacting
    if "count" in result:
        compact["count"] = result["count"]
    if "rows" in result:
        compact["rows"] = result["rows"]
    if "columns" in result:
        compact["columns"] = (result["columns"] or [])[:50]
    if "artifact" in result and isinstance(result.get("artifact"), dict):
        compact["artifact_url"] = result["artifact"].get("url")
        compact["artifact_type"] = result["artifact"].get("type")
    if "plot" in result and isinstance(result.get("plot"), dict):
        compact["plot_url"] = result["plot"].get("url")
    if "data_artifact" in result and isinstance(result.get("data_artifact"), dict):
        compact["data_artifact_url"] = result["data_artifact"].get("url")
        compact["data_artifact_type"] = result["data_artifact"].get("type")
    if "preview" in result:
        try:
            compact["preview"] = (result["preview"] or [])[:20]
        except Exception:
            pass
    if "results" in result:
        try:
            compact["results"] = (result["results"] or [])[:10]
        except Exception:
            pass
    if "metadata" in result:
        try:
            compact["metadata"] = (result["metadata"] or [])[:5]
        except Exception:
            pass
    if "variables" in result:
        try:
            compact["variables"] = (result["variables"] or [])[:15]
        except Exception:
            pass


    if "resolved" in result and isinstance(result.get("resolved"), dict):
        compact["resolved"] = result.get("resolved")
    return _to_jsonable(compact)


def _request_requires_tools(user_message: str) -> bool:
    """Heuristic: decide whether we should REQUIRE at least one tool call.

    We enforce this to avoid LLMs returning a 'plan' in plain text instead of
    actually executing tools for data/plot/catalog requests.
    """
    um = (user_message or "").lower()

    # Strong signals: mapping/plotting or explicit data export/retrieval
    keywords = [
        "map",
        "plot",
        "graph",
        "chart",
        "figure",
        "heatmap",
        "download",
        "export",
        "csv",
        "parquet",
        "subset",
        "retrieve",
        "query",
    ]
    if any(k in um for k in keywords):
        return True

    # Aggregations almost always require tools.
    if any(k in um for k in ["average", "averaged", "mean", "avg", "median", "anomaly", "climatology"]):
        if any(v in um for v in ["temp", "temperature", "sst", "salin", "wind", "precip", "rain", "chlor", "oxygen"]):
            return True

    # Catalog operations
    if "how many" in um and "dataset" in um:
        return True
    if "summar" in um and "dataset" in um:
        return True

    # Time window + variable-like word -> prefer tools.
    months = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "sept",
        "sep",
        "october",
        "november",
        "december",
    ]
    vars_ = ["precip", "rain", "sst", "chlor", "salin", "wind", "temp", "temperature", "oxygen"]
    if any(m in um for m in months) and any(v in um for v in vars_):
        return True

    # Bounding boxes / coordinates are a strong signal for geospatial queries.
    if any(s in um for s in ["lat", "lon", "bbox", "(", ")"]):
        if any(v in um for v in vars_):
            return True

    return False




_REGION_HINTS = [
    "north atlantic", "south atlantic", "north pacific", "south pacific",
    "southern ocean", "arctic", "indian ocean", "mediterranean", "gulf of mexico",
]

_REGION_BBOXES = {
    "global": {"lat1": -90.0, "lat2": 90.0, "lon1": -180.0, "lon2": 180.0},
    "southern ocean": {"lat1": -90.0, "lat2": -60.0, "lon1": -180.0, "lon2": 180.0},
    "arctic ocean": {"lat1": 66.0, "lat2": 90.0, "lon1": -180.0, "lon2": 180.0},
    "arctic": {"lat1": 66.0, "lat2": 90.0, "lon1": -180.0, "lon2": 180.0},
    "north atlantic ocean": {"lat1": 0.0, "lat2": 60.0, "lon1": -76.0, "lon2": -6.0},
    "north atlantic": {"lat1": 0.0, "lat2": 60.0, "lon1": -76.0, "lon2": -6.0},
    "south atlantic ocean": {"lat1": -60.0, "lat2": 0.0, "lon1": -68.0, "lon2": 20.0},
    "south atlantic": {"lat1": -60.0, "lat2": 0.0, "lon1": -68.0, "lon2": 20.0},
    "north pacific ocean": {"lat1": 0.0, "lat2": 66.0, "lon1": 120.0, "lon2": -98.0},
    "north pacific": {"lat1": 0.0, "lat2": 66.0, "lon1": 120.0, "lon2": -98.0},
    "south pacific ocean": {"lat1": -60.0, "lat2": 0.0, "lon1": 147.0, "lon2": -68.0},
    "south pacific": {"lat1": -60.0, "lat2": 0.0, "lon1": 147.0, "lon2": -68.0},
    "indian ocean": {"lat1": -60.0, "lat2": 30.0, "lon1": 20.0, "lon2": 120.0},
    "mediterranean sea": {"lat1": 31.0, "lat2": 45.0, "lon1": -1.0, "lon2": 36.0},
    "mediterranean": {"lat1": 31.0, "lat2": 45.0, "lon1": -1.0, "lon2": 36.0},
    "gulf of mexico": {"lat1": 20.0, "lat2": 30.0, "lon1": -97.0, "lon2": -83.0},
}

_MONTH_TO_NUM = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "sept": 9, "sep": 9,
    "october": 10, "november": 11, "december": 12,
}


def _extract_bounds_from_message(user_message: str) -> dict[str, Any]:
    text = str(user_message or "")
    lower = text.lower()
    out: dict[str, Any] = {}

    for region, bbox in _REGION_BBOXES.items():
        if region in lower:
            out.update(bbox)
            break

    # ISO date first.
    iso = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", lower)
    if iso:
        out["dt1"] = iso.group(1)
        out["dt2"] = iso.group(1)
        return out

    # Month day year, with optional comma.
    month_pat = r"(january|february|march|april|may|june|july|august|september|sept|sep|october|november|december)"
    mdy = re.search(rf"\b{month_pat}\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,)?\s+(\d{{4}})\b", lower)
    if mdy:
        mon = _MONTH_TO_NUM.get(mdy.group(1), 0)
        day = int(mdy.group(2))
        year = int(mdy.group(3))
        if mon:
            out["dt1"] = f"{year:04d}-{mon:02d}-{day:02d}"
            out["dt2"] = out["dt1"]
            return out

    # Month year only -> whole month.
    my = re.search(rf"\b{month_pat}\s+(\d{{4}})\b", lower)
    if my:
        mon = _MONTH_TO_NUM.get(my.group(1), 0)
        year = int(my.group(2))
        if mon:
            from calendar import monthrange
            last = monthrange(year, mon)[1]
            out["dt1"] = f"{year:04d}-{mon:02d}-01"
            out["dt2"] = f"{year:04d}-{mon:02d}-{last:02d}"
    return out


def _user_explicitly_named_dataset(user_message: str) -> bool:
    um = (user_message or "").lower()
    if re.search(r"\btbl[a-z0-9_]+\b", um):
        return True
    # Quoted/standalone CMAP-style table-like identifiers are the main stable signal.
    return False


def _surface_requested(user_message: str) -> bool:
    um = (user_message or "").lower()
    return "surface" in um or "near surface" in um or "upper ocean" in um


def _time_series_requested(user_message: str) -> bool:
    s = (user_message or "").lower()
    return any(tok in s for tok in [
        "time series", "timeseries", "over time", "through time", "trend",
        "monthly series", "daily series", "yearly series", "annual cycle",
    ])


def _single_date_request(args: dict[str, Any]) -> bool:
    if not isinstance(args, dict):
        return False
    d1 = str(args.get("dt1") or "")[:10]
    d2 = str(args.get("dt2") or "")[:10]
    return bool(d1 and d1 == d2)


def _has_roi(args: dict[str, Any]) -> bool:
    return isinstance(args, dict) and all(args.get(k) is not None for k in ("lat1", "lat2", "lon1", "lon2"))


def _request_prefers_map(user_message: str, args: dict[str, Any]) -> bool:
    s = (user_message or "").lower()
    if _time_series_requested(s):
        return False
    if "map" in s or "spatial" in s:
        return True
    # For regional single-date field requests, a generic 'plot' should default to a map.
    if _has_roi(args) and _single_date_request(args):
        if any(tok in s for tok in ["plot", "show", "graph", "chart", "surface", "over "]):
            return True
    return False


def _extract_recent_request_message(conversation: list[dict[str, Any]]) -> str | None:
    msgs = list(conversation or [])
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if str(m.get("role") or "") != "user":
            continue
        content = str(m.get("content") or "").strip()
        lower = content.lower()
        if not content:
            continue
        # Skip short acknowledgements/corrections; keep the last substantive data request.
        wc = len(re.findall(r"\w+", lower))
        if wc <= 8 and any(tok in lower for tok in ["ok", "okay", "go ahead", "use ", "make a map", "time series", "no,", "no "]):
            continue
        if any(tok in lower for tok in ["plot", "map", "show", "graph", "chart", "subset", "colocalize", "download", "can you", "could you", "would you"]):
            return content
    return None


def _extract_recent_dataset_table(conversation: list[dict[str, Any]]) -> str | None:
    msgs = list(conversation or [])
    pat = re.compile(r"\b(tbl[a-zA-Z0-9_]+)\b")
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if str(m.get("role") or "") != "assistant":
            continue
        content = str(m.get("content") or "")
        hits = pat.findall(content)
        if hits:
            return hits[0]
    return None



def _extract_region_name(text_: str) -> str | None:
    lower = str(text_ or '').lower()
    for region in sorted(_REGION_BBOXES.keys(), key=len, reverse=True):
        if region in lower:
            return region
    # Small typo-tolerant fallback for common basin names.
    toks = re.findall(r"[a-z]+", lower)
    joined = " ".join(toks)
    # Look for phrases like 'north atlantic' with mild misspellings in the cardinal word.
    for basin in ("atlantic", "pacific"):
        if basin in joined:
            words = joined.split()
            for i, w in enumerate(words[:-1]):
                if words[i + 1] != basin:
                    continue
                if difflib.SequenceMatcher(None, w, 'north').ratio() >= 0.72:
                    candidate = f'north {basin}'
                    if candidate in _REGION_BBOXES:
                        return candidate
                if difflib.SequenceMatcher(None, w, 'south').ratio() >= 0.72:
                    candidate = f'south {basin}'
                    if candidate in _REGION_BBOXES:
                        return candidate
    return None


def _month_name_from_date_str(dt: str | None) -> str | None:
    s = _parse_date_safe_local(dt)
    if not s:
        return None
    try:
        import datetime as _dt
        d = _dt.date.fromisoformat(s)
        return d.strftime('%B')
    except Exception:
        return None


def _looks_climatology_request(text_: str) -> bool:
    s = str(text_ or '').lower()
    return 'climatology' in s or 'climatological' in s


def _request_state_from_text(text_: str) -> dict[str, Any]:
    text = str(text_ or '').strip()
    lower = text.lower()
    state: dict[str, Any] = {}
    fam = _field_family_from_text(lower)
    if fam:
        state['field_family'] = fam
    if _surface_requested(lower):
        state['surface'] = True
    if _looks_climatology_request(lower):
        state['climatology'] = True
    if 'satellite' in lower:
        state['sensor'] = 'satellite'
    region_name = _extract_region_name(lower)
    if region_name:
        state['region_name'] = region_name
        state.update(dict(_REGION_BBOXES.get(region_name) or {}))
    bounds = _extract_bounds_from_message(text)
    state.update({k: v for k, v in bounds.items() if v is not None})
    if _time_series_requested(lower):
        state['action'] = 'time_series'
    elif 'map' in lower:
        state['action'] = 'map'
    elif 'plot' in lower and (state.get('region_name') or all(state.get(k) is not None for k in ('lat1', 'lat2', 'lon1', 'lon2'))):
        state['action'] = 'map'
    return state


def _apply_followup_to_state(base: dict[str, Any], text_: str) -> dict[str, Any]:
    out = dict(base or {})
    text = str(text_ or '').strip()
    lower = text.lower()
    new = _request_state_from_text(text)
    # Explicit follow-up toggles / corrections.
    if _looks_climatology_request(lower) or 'not climatology' in lower:
        out['climatology'] = True
    if 'surface' in lower:
        out['surface'] = True
    if 'satellite' in lower:
        out['sensor'] = 'satellite'
    if new.get('field_family'):
        out['field_family'] = new['field_family']
    if new.get('region_name'):
        out['region_name'] = new['region_name']
        for k in ('lat1', 'lat2', 'lon1', 'lon2'):
            out[k] = new.get(k)
    for k in ('dt1', 'dt2'):
        if new.get(k) is not None:
            out[k] = new.get(k)
    if new.get('action'):
        out['action'] = new['action']
    return out


def _field_phrase_from_family(fam: str | None, *, surface: bool = False, sensor: str | None = None) -> str:
    fam = str(fam or '').lower().strip()
    phrase = {
        'chlorophyll': 'chlorophyll',
        'nitrate': 'dissolved nitrate',
        'nitrite': 'nitrite',
        'phosphate': 'phosphate',
        'silicate': 'silicate',
        'oxygen': 'dissolved oxygen',
        'salinity': 'salinity',
        'sst': 'sea surface temperature',
        'wind': 'wind',
        'precipitation': 'precipitation',
    }.get(fam, fam or 'data')
    if sensor == 'satellite' and 'satellite' not in phrase:
        phrase = f'satellite {phrase}'
    if surface and 'surface' not in phrase and 'sea surface' not in phrase:
        phrase = f'surface {phrase}'
    return phrase.strip()


def _build_request_message_from_state(state: dict[str, Any]) -> str | None:
    if not state:
        return None
    field_phrase = _field_phrase_from_family(state.get('field_family'), surface=bool(state.get('surface')), sensor=state.get('sensor'))
    region = state.get('region_name')
    action = state.get('action') or ('map' if region or all(state.get(k) is not None for k in ('lat1','lat2','lon1','lon2')) else 'plot')
    clim = bool(state.get('climatology'))
    # For climatology follow-ups, carry the month from an earlier exact date rather than the exact date.
    date_phrase = None
    if clim:
        month_name = _month_name_from_date_str(state.get('dt1'))
        if month_name:
            date_phrase = month_name
    if not date_phrase and state.get('dt1') and state.get('dt2') and str(state.get('dt1')) == str(state.get('dt2')):
        date_phrase = str(state.get('dt1'))
    elif state.get('dt1') and state.get('dt2'):
        date_phrase = f"{state.get('dt1')} to {state.get('dt2')}"
    parts = [f"can you make a {action} of"]
    if clim:
        parts.append('climatology of')
    parts.append(field_phrase)
    if region:
        parts.append(f"over the {region}")
    if date_phrase:
        parts.append(f"in {date_phrase}")
    return " ".join(parts).replace('  ', ' ').strip()


def _build_request_state_from_conversation(conversation: list[dict[str, Any]]) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for m in list(conversation or []):
        if str(m.get('role') or '') != 'user':
            continue
        content = str(m.get('content') or '').strip()
        if not content:
            continue
        lower = content.lower()
        wc = len(re.findall(r"\w+", lower))
        # Ignore bare confirmations; they do not define/modify the request semantics.
        if wc <= 6 and _is_affirmative_short_reply(lower):
            continue
        # Treat follow-up refinements as edits to the existing state.
        if state and (
            _looks_climatology_request(lower)
            or 'how about' in lower
            or 'what about' in lower
            or 'i meant' in lower
            or 'instead' in lower
            or 'but those are not climatology' in lower
            or _extract_region_name(lower) is not None
            or _field_family_from_text(lower) is not None
            or 'surface' in lower
        ):
            state = _apply_followup_to_state(state, content)
            continue
        # Start/update the base request on substantive data requests.
        if any(tok in lower for tok in ['plot', 'map', 'show', 'graph', 'chart', 'can you', 'could you', 'would you']):
            state = _apply_followup_to_state(_request_state_from_text(content), content)
            continue
        # As a fallback, allow compact referential edits to modify the active state.
        if state and wc <= 20:
            state = _apply_followup_to_state(state, content)
    return state


def _rewrite_followup_request(user_message: str, conversation: list[dict[str, Any]]) -> str | None:
    lower = str(user_message or '').strip().lower()
    if not lower or _user_explicitly_named_dataset(lower):
        return None
    if _is_affirmative_short_reply(lower):
        return None
    state = _build_request_state_from_conversation(conversation)
    if not state:
        return None
    merged = _apply_followup_to_state(state, user_message)
    rewritten = _build_request_message_from_state(merged)
    if not rewritten:
        return None
    # Only rewrite when the current turn looks referential or incomplete by itself.
    looks_followup = (
        _looks_climatology_request(lower)
        or 'how about' in lower
        or 'what about' in lower
        or 'i meant' in lower
        or 'instead' in lower
        or 'but those are not climatology' in lower
        or (_extract_region_name(lower) is not None and _field_family_from_text(lower) is None)
        or (len(re.findall(r"\w+", lower)) <= 12 and _field_family_from_text(lower) is not None)
    )
    return rewritten if looks_followup else None

def _followup_action_correction(user_message: str, conversation: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    s = str(user_message or "").strip().lower()
    if not s:
        return None, None
    wc = len(re.findall(r"\w+", s))
    if wc > 10:
        return None, None
    action = None
    if "map" in s and not _time_series_requested(s):
        action = "map"
    elif "time series" in s or "timeseries" in s or "trend" in s:
        action = "time_series"
    if not action:
        return None, None
    req = _extract_recent_request_message(conversation)
    table = _extract_recent_dataset_table(conversation)
    return action, f"{req} Use dataset {table}. Create a map, not a time series." if action == "map" and req and table else (f"{req} Use dataset {table}. Create a time series, not a map." if action == "time_series" and req and table else None)


def _maybe_apply_surface_defaults(user_message: str, args: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    if not isinstance(args, dict):
        return args, False
    if not _surface_requested(user_message):
        return args, False
    out = dict(args)
    changed = False
    d1 = out.get("depth1")
    d2 = out.get("depth2")
    # Clamp obviously invalid / overly broad surface defaults to a shallow, nonnegative slice.
    if d1 is None or d2 is None:
        out["depth1"] = 0
        out["depth2"] = 5
        return out, True
    try:
        fd1 = float(d1)
        fd2 = float(d2)
    except Exception:
        out["depth1"] = 0
        out["depth2"] = 5
        return out, True
    if fd1 < 0 or fd2 < 0 or fd2 > 50 or fd1 < -1 or (fd2 - fd1) > 50:
        out["depth1"] = 0
        out["depth2"] = 5
        changed = True
    return out, changed


def _extract_catalog_query(user_message: str) -> str:
    s = (user_message or "").strip().lower()
    if not s:
        return ""
    s = s.replace("chlorophyl", "chlorophyll")
    # Remove common leading request verbs.
    s = re.sub(r"^(please\s+)?(can\s+you|could\s+you|would\s+you|show\s+me|make|plot|map|graph|chart)\s+", "", s)
    s = re.sub(r"^(a|an|the)\s+", "", s)
    s = re.sub(r"\b(of|for)\b", " ", s)
    # Remove common region/date phrases while keeping core data intent words.
    for region in _REGION_HINTS:
        s = re.sub(rf"\b(over|in|across)\s+the\s+{re.escape(region)}\b", " ", s)
        s = re.sub(rf"\b(over|in|across)\s+{re.escape(region)}\b", " ", s)
    month_names = r"january|february|march|april|may|june|july|august|september|sept|sep|october|november|december"
    s = re.sub(rf"\b(on|for|during|in)\s+({month_names})\b(?:\s+\d{{1,2}}(?:,)?\s*\d{{4}}|\s+\d{{4}})?", " ", s)
    s = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", s)
    s = re.sub(r"[^a-z0-9_ ]+", " ", s)
    toks = [t for t in s.split() if t not in {"can","you","make","plot","map","graph","chart","show","me","the","a","an","over","in","across","of","for","on","during"}]
    # Keep compact intent phrase; fallback to original lowercased text if over-stripped.
    out = " ".join(toks).strip()
    return out or (user_message or "").strip()


def _latest_catalog_query_and_bounds(user_message: str, tool_trace: list[dict[str, Any]], pending_args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    query = ""
    bounds: dict[str, Any] = {}
    for item in reversed(tool_trace or []):
        tool_name = str(item.get("tool") or "")
        if tool_name not in {"catalog.search_kb_first", "catalog.search", "catalog.search_variables"}:
            continue
        args = item.get("arguments") or {}
        if isinstance(args, dict):
            q = args.get("query")
            if isinstance(q, str) and q.strip() and not query:
                query = q.strip()
            for k in ("lat1", "lat2", "lon1", "lon2", "dt1", "dt2"):
                if k in args and k not in bounds and args.get(k) is not None:
                    bounds[k] = args.get(k)
        if query:
            break
    if not query:
        query = _extract_catalog_query(user_message)
    inferred = _extract_bounds_from_message(user_message)
    for k, v in inferred.items():
        if bounds.get(k) is None:
            bounds[k] = v
    for k in ("lat1", "lat2", "lon1", "lon2", "dt1", "dt2"):
        if k in pending_args and pending_args.get(k) is not None:
            bounds[k] = pending_args.get(k)
    return query, bounds


def _merge_candidates(primary: list[dict[str, Any]], secondary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for src in (primary or []) + (secondary or []):
        if not isinstance(src, dict):
            continue
        t = str(src.get("table") or "").strip()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(src)
    return out


def _candidate_time_text(r: dict[str, Any]) -> str | None:
    tmin = r.get("time_min") or r.get("TimeMin")
    tmax = r.get("time_max") or r.get("TimeMax")
    if tmin and tmax:
        return f"{tmin} to {tmax}"
    return None


def _candidate_spatial_text(r: dict[str, Any]) -> str | None:
    lat_min = r.get("lat_min")
    lat_max = r.get("lat_max")
    lon_min = r.get("lon_min")
    lon_max = r.get("lon_max")
    if lat_min is None or lat_max is None or lon_min is None or lon_max is None:
        return None
    return f"lat {lat_min} to {lat_max}; lon {lon_min} to {lon_max}"


def _candidate_resolution_text(r: dict[str, Any], key: str) -> str | None:
    v = r.get(key)
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _parse_date_safe_local(s: Any) -> str | None:
    s = str(s or "").strip()
    if not s:
        return None
    m = re.match(r"(\d{4}-\d{2}-\d{2})", s)
    return m.group(1) if m else None


def _candidate_within_time(r: dict[str, Any], bounds: dict[str, Any]) -> bool:
    q1 = _parse_date_safe_local(bounds.get("dt1"))
    q2 = _parse_date_safe_local(bounds.get("dt2")) or q1
    if not q1:
        return True
    t1 = _parse_date_safe_local(r.get("time_min") or r.get("TimeMin"))
    t2 = _parse_date_safe_local(r.get("time_max") or r.get("TimeMax"))
    if not t1 or not t2:
        return True
    return t1 <= q1 <= t2 and t1 <= q2 <= t2


def _candidate_within_space(r: dict[str, Any], bounds: dict[str, Any]) -> bool:
    if not all(bounds.get(k) is not None for k in ("lat1", "lat2", "lon1", "lon2")):
        return True
    lat_min = r.get("lat_min")
    lat_max = r.get("lat_max")
    lon_min = r.get("lon_min")
    lon_max = r.get("lon_max")
    if lat_min is None or lat_max is None or lon_min is None or lon_max is None:
        return True
    try:
        return _bbox_overlaps(lat_min, lat_max, lon_min, lon_max, float(bounds["lat1"]), float(bounds["lat2"]), float(bounds["lon1"]), float(bounds["lon2"]))
    except Exception:
        return True


def _presentable_candidates(rows: list[dict[str, Any]], bounds: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        if not _candidate_within_time(r, bounds):
            continue
        if not _candidate_within_space(r, bounds):
            continue
        out.append(r)
    return out


def _field_family_from_text(text_: str) -> str | None:
    s = (text_ or "").lower()
    fams = {
        "chlorophyll": ["chlorophyll", "chlorophyl", "chlor_a", "chl", "chla"],
        "nitrate": ["nitrate", "no3"],
        "nitrite": ["nitrite", "no2"],
        "phosphate": ["phosphate", "po4"],
        "silicate": ["silicate", "silicic", "si"],
        "oxygen": ["oxygen", "o2", "dissolved oxygen"],
        "salinity": ["salinity", "sss"],
        "sst": ["sea surface temperature", "sst", "temperature"],
        "wind": ["wind", "stress"],
        "precipitation": ["precip", "precipitation", "rain", "rainfall", "tp"],
    }
    for fam, terms in fams.items():
        if any(t in s for t in terms):
            return fam
    return None


def _maybe_resolve_variable_for_table(query_variable: str, table: str, ctx: dict[str, Any]) -> str | None:
    qv = str(query_variable or "").strip()
    table = str(table or "").strip()
    if not qv or not table:
        return None
    try:
        res = catalog_search_variables(CatalogSearchVariablesArgs(query=qv, table_hint=table, limit=5), ctx)
    except Exception:
        return None
    rows = [r for r in (res.get("results") or []) if isinstance(r, dict)]
    if not rows:
        return None
    fam = _field_family_from_text(qv)
    for r in rows:
        v = str(r.get("variable") or "")
        ln = str(r.get("long_name") or "")
        blob = f"{v} {ln}"
        if fam and _field_family_from_text(blob) == fam:
            return v
    return str(rows[0].get("variable") or "") or None


def _deterministic_resolve_candidates(user_message: str, pending_args: dict[str, Any], tool_trace: list[dict[str, Any]], ctx: dict[str, Any]) -> dict[str, Any]:
    query, bounds = _latest_catalog_query_and_bounds(user_message, tool_trace, pending_args)
    results: list[dict[str, Any]] = []
    selected = None
    alternates: list[dict[str, Any]] = []

    # 1) KB-first search with explicit ROI/time if we have them.
    try:
        kb_args = CatalogSearchKBFArgs(
            query=query,
            lat1=bounds.get("lat1"), lat2=bounds.get("lat2"),
            lon1=bounds.get("lon1"), lon2=bounds.get("lon2"),
            dt1=bounds.get("dt1"), dt2=bounds.get("dt2"),
            limit=10,
        )
        kb_res = catalog_search_kb_first(kb_args, ctx)
    except Exception:
        kb_res = {"results": []}

    if isinstance(kb_res.get("results"), list):
        results = list(kb_res.get("results") or [])
    if isinstance(kb_res.get("selected"), dict):
        selected = kb_res.get("selected")
    if isinstance(kb_res.get("alternates"), list):
        alternates = list(kb_res.get("alternates") or [])

    # 2) SQL metadata fallback / augmentation.
    try:
        sql_res = catalog_search(CatalogSearchArgs(query=query, limit=10), ctx)
    except Exception:
        sql_res = {"results": []}
    sql_results = list(sql_res.get("results") or []) if isinstance(sql_res.get("results"), list) else []

    # 3) Variable-driven augmentation: find tables with matching variables and fetch metadata.
    var_tables: list[str] = []
    try:
        var_res = catalog_search_variables(CatalogSearchVariablesArgs(query=query, limit=15), ctx)
        for row in var_res.get("results") or []:
            if isinstance(row, dict):
                t = str(row.get("table") or "").strip()
                if t and t not in var_tables:
                    var_tables.append(t)
    except Exception:
        var_tables = []

    var_meta: list[dict[str, Any]] = []
    if var_tables:
        try:
            store = ctx.get("store")
            meta_rows = _fetch_datasets_by_tables(store, var_tables) if store is not None else []
        except Exception:
            meta_rows = []
        # Apply ROI overlap if available.
        if all(bounds.get(k) is not None for k in ("lat1", "lat2", "lon1", "lon2")):
            for r in meta_rows:
                try:
                    if _bbox_overlaps(r.get("lat_min"), r.get("lat_max"), r.get("lon_min"), r.get("lon_max"), float(bounds["lat1"]), float(bounds["lat2"]), float(bounds["lon1"]), float(bounds["lon2"])):
                        var_meta.append(r)
                except Exception:
                    pass
        else:
            var_meta = meta_rows

    merged = _merge_candidates(results, sql_results)
    merged = _merge_candidates(merged, var_meta)

    # Deterministic post-ranking of merged candidates using query/time/map hints.
    try:
        from cmap_agent.tools.catalog_tools import _post_rank_catalog_results
        merged = _post_rank_catalog_results(merged, query=query, dt1=bounds.get("dt1"), dt2=bounds.get("dt2"))
    except Exception:
        pass

    # If the active request is explicitly about climatology, keep climatology-style datasets when possible.
    if _looks_climatology_request(user_message):
        clim_rows = []
        for r in merged:
            if not isinstance(r, dict):
                continue
            blob = " ".join(str(r.get(k) or '') for k in ('table', 'name', 'title', 'description', 'temporal_resolution')).lower()
            if 'climatology' in blob:
                clim_rows.append(r)
        if clim_rows:
            merged = clim_rows

    # Do not present datasets that are explicitly outside the requested time/space bounds.
    ranked = _presentable_candidates(merged, bounds)
    if not ranked:
        ranked = merged

    # Derive selected/alternates from the filtered ranked list to keep behavior stable.
    if ranked:
        selected = ranked[0]
        base_table = str((selected or {}).get("table") or "")
        alternates = [r for r in ranked if str(r.get("table") or "") != base_table][:5]

    return {
        "query": {"q": query, **bounds},
        "selected": selected,
        "alternates": alternates,
        "results": ranked[:10],
    }




def _extract_pending_confirmation(conversation: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Parse the last deterministic confirmation message from prior conversation.

    Returns a dict with keys: request_message, selected, candidates.
    """
    if not conversation:
        return None
    msgs = list(conversation)
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if str(m.get("role") or "") != "assistant":
            continue
        content = str(m.get("content") or "")
        if "Best candidate:" not in content or "Should this best candidate be used" not in content:
            continue
        prev_user = ""
        for j in range(i - 1, -1, -1):
            if str(msgs[j].get("role") or "") == "user":
                prev_user = str(msgs[j].get("content") or "")
                break
        lines = [ln.rstrip() for ln in content.splitlines()]
        selected = None
        candidates: list[dict[str, Any]] = []
        mode = None
        pat = re.compile(r"^-\s*(.+?)\s*\(`([^`]+)`\)\s*$")
        for ln in lines:
            stripped = ln.strip()
            if stripped == "Best candidate:":
                mode = "best"
                continue
            if stripped == "Other possible candidates:":
                mode = "alts"
                continue
            m2 = pat.match(stripped)
            if not m2:
                continue
            item = {"name": m2.group(1).strip(), "table": m2.group(2).strip()}
            if mode == "best" and selected is None:
                selected = item
                candidates.append(item)
            elif mode == "alts":
                candidates.append(item)
        if selected:
            return {"request_message": prev_user, "selected": selected, "candidates": candidates}
    return None


def _is_affirmative_short_reply(text_: str) -> bool:
    s = str(text_ or "").strip().lower()
    if not s:
        return False
    if _looks_like_new_request_or_question(s):
        return False
    wc = len(re.findall(r"\w+", s))
    if wc > 8:
        return False
    affirm = ["yes", "yeah", "yep", "ok", "okay", "go ahead", "sure", "please do", "do it", "use that", "that works", "sounds good"]
    return any(tok in s for tok in affirm)


def _extract_pending_assistant_proposal(conversation: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Parse the most recent assistant-authored proposal/question that names a dataset.

    This catches follow-ups like: 'Would you like me to create a map using this climatology
    dataset for March?' which are not in the deterministic Best-candidate format.

    Returns a dict with keys: request_message, selected, action.
    """
    if not conversation:
        return None
    msgs = list(conversation)
    tbl_pat = re.compile(r"\b(tbl[a-zA-Z0-9_]+)\b")
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if str(m.get("role") or "") != "assistant":
            continue
        content = str(m.get("content") or "").strip()
        lower = content.lower()
        if not content:
            continue
        if "Best candidate:" in content and "Should this best candidate be used" in content:
            continue
        if not ("?" in content or "would you like me" in lower or "should i" in lower or "shall i" in lower or "do you want me" in lower):
            continue
        hits = tbl_pat.findall(content)
        if not hits:
            continue
        table = hits[0]
        action = None
        if "create a map" in lower or "make a map" in lower or "using this climatology dataset" in lower or "map using" in lower:
            action = "map"
        elif "time series" in lower or "timeseries" in lower or "trend" in lower:
            action = "time_series"
        prev_request = _extract_recent_request_message(msgs[:i]) or ""
        return {
            "request_message": prev_request,
            "selected": {"table": table, "name": table},
            "action": action,
            "assistant_message": content,
        }
    return None


def _resolve_assistant_proposal_reply(user_message: str, pending: dict[str, Any] | None) -> dict[str, Any] | None:
    if not pending or not isinstance(pending.get("selected"), dict):
        return None
    msg = str(user_message or "").strip()
    if not msg:
        return None
    lower = msg.lower()
    sel = pending.get("selected") or {}
    table = str(sel.get("table") or "").strip().lower()
    if table and table in lower:
        return sel
    if _is_affirmative_short_reply(lower):
        return sel
    return None


def _looks_like_new_request_or_question(text_: str) -> bool:
    s = str(text_ or "").strip().lower()
    if not s:
        return False
    if "?" in s:
        return True
    # New requests tend to mention data-query specifics; confirmation replies usually do not.
    if any(tok in s for tok in ["plot", "map", "graph", "chart", "subset", "download", "colocalize", "lat", "lon", "bbox"]):
        return True
    if re.search(r"\d{4}-\d{2}-\d{2}", s):
        return True
    if any(m in s for m in ["january","february","march","april","may","june","july","august","september","october","november","december"]):
        return True
    return False


def _resolve_confirmation_reply(user_message: str, pending: dict[str, Any] | None) -> dict[str, Any] | None:
    if not pending or not isinstance(pending.get("selected"), dict):
        return None
    msg = str(user_message or "").strip()
    if not msg:
        return None
    lower = msg.lower()
    candidates = [c for c in (pending.get("candidates") or []) if isinstance(c, dict)]
    # 1) Explicit candidate/table mention always wins.
    tbl_match = re.search(r"tbl[a-z0-9_]+", lower)
    if tbl_match:
        tbl = tbl_match.group(0)
        for c in candidates:
            if str(c.get("table") or "").lower() == tbl:
                return c
    for c in candidates:
        name = str(c.get("name") or "").strip().lower()
        table = str(c.get("table") or "").strip().lower()
        if name and name in lower:
            return c
        if table and table in lower:
            return c
    # 2) If the reply is short and does not look like a new request/question, accept the best candidate.
    token_count = len(re.findall(r"\w+", lower))
    if token_count <= 8 and not _looks_like_new_request_or_question(lower):
        return pending.get("selected")
    return None

def _build_confirmation_message(user_message: str, resolution: dict[str, Any]) -> str:
    sel = resolution.get("selected") or {}
    alts = list(resolution.get("alternates") or [])
    q = dict(resolution.get("query") or {})
    if not isinstance(sel, dict) or not sel:
        return "I could not confidently resolve the dataset yet. Could you specify the dataset you want to use?"

    def _fmt_candidate(r: dict[str, Any]) -> list[str]:
        name = r.get('name') or r.get('title') or r.get('table')
        return [
            f"- {name} (`{r.get('table')}`)",
            f"  - Temporal coverage: {_candidate_time_text(r) or '?'}",
            f"  - Spatial coverage: {_candidate_spatial_text(r) or '?'}",
            f"  - Temporal resolution: {_candidate_resolution_text(r, 'temporal_resolution') or '?'}",
            f"  - Spatial resolution: {_candidate_resolution_text(r, 'spatial_resolution') or '?'}",
        ]

    lines = [
        "I found a best CMAP dataset candidate for this request, but the dataset was not explicitly specified.",
        "Best candidate:",
        *_fmt_candidate(sel),
    ]
    if alts:
        lines.append("Other possible candidates:")
        for a in alts[:5]:
            lines.extend(_fmt_candidate(a))
    if q.get('dt1') or q.get('lat1') is not None or q.get('lon1') is not None:
        lines.append("Datasets explicitly outside the requested temporal or spatial coverage are omitted from this list.")
    lines.append("Should this best candidate be used, or would another candidate be preferred?")
    return "\n".join(lines)


def _should_request_dataset_confirmation(user_message: str, call_name: str, raw_args: dict[str, Any]) -> bool:
    if _user_explicitly_named_dataset(user_message):
        return False
    if not isinstance(raw_args, dict):
        return False
    if call_name.startswith("catalog."):
        return False
    # Require confirmation before executing any data/plot tool that has a resolved table.
    return call_name.startswith("cmap.") or call_name.startswith("viz.")


def execute_plan(
    *,
    llm: LLMClient,
    registry: ToolRegistry,
    system_prompt: str,
    conversation: list[dict[str, Any]],
    user_message: str,
    ctx: dict[str, Any],
    max_tool_calls: int = LIMITS.max_tool_calls,
) -> tuple[AgentFinal, list[dict[str, Any]]]:
    """Run one agent turn, with iterative tool-calling until a final answer.

    Returns: (final_response, tool_trace)
    """
    messages: list[LLMMessage] = [LLMMessage(role="system", content=system_prompt)]

    pending_confirmation = _extract_pending_confirmation(conversation)
    pending_proposal = _extract_pending_assistant_proposal(conversation)
    confirmed_candidate = _resolve_confirmation_reply(user_message, pending_confirmation)
    confirmed_proposal = _resolve_assistant_proposal_reply(user_message, pending_proposal)
    active_user_message = user_message
    confirmed_dataset_table = None
    contextual_rewrite = _rewrite_followup_request(user_message, conversation)
    followup_action, followup_rewritten = _followup_action_correction(user_message, conversation)
    if followup_rewritten:
        active_user_message = followup_rewritten
        tbl = _extract_recent_dataset_table(conversation)
        if tbl:
            confirmed_dataset_table = tbl
        messages.append(
            LLMMessage(
                role="system",
                content=(
                    "The user is correcting the visualization/action for the previously confirmed dataset and request. "
                    "Do not restart dataset discovery. Reuse the previously used dataset and request context, and only change the requested action."
                ),
            )
        )
    elif isinstance(confirmed_proposal, dict) and str(confirmed_proposal.get("table") or "").strip():
        confirmed_dataset_table = str(confirmed_proposal.get("table") or "").strip()
        prev_request = str((pending_proposal or {}).get("request_message") or "").strip()
        action = str((pending_proposal or {}).get("action") or "").strip()
        if prev_request:
            active_user_message = f"{prev_request} Use dataset {confirmed_dataset_table}."
        else:
            active_user_message = f"Use dataset {confirmed_dataset_table}."
        if action == "map":
            active_user_message += " Create a map."
        elif action == "time_series":
            active_user_message += " Create a time series."
        messages.append(
            LLMMessage(
                role="system",
                content=(
                    "The user is affirming the assistant's most recent dataset proposal/question and has now confirmed the dataset "
                    f"`{confirmed_dataset_table}`. Treat that dataset as explicitly specified for this turn. "
                    "Do not restart dataset discovery. Reuse the most recent request context and proceed directly to variable resolution and execution."
                ),
            )
        )
    elif isinstance(confirmed_candidate, dict) and str(confirmed_candidate.get("table") or "").strip():
        confirmed_dataset_table = str(confirmed_candidate.get("table") or "").strip()
        prev_request = str((pending_confirmation or {}).get("request_message") or "").strip()
        if prev_request:
            active_user_message = f"{prev_request} Use dataset {confirmed_dataset_table}."
        else:
            active_user_message = f"Use dataset {confirmed_dataset_table}."
        messages.append(
            LLMMessage(
                role="system",
                content=(
                    "The user is responding to a pending dataset-confirmation step and has now confirmed the dataset "
                    f"`{confirmed_dataset_table}`. Treat that dataset as explicitly specified for this turn. "
                    "Do not run another catalog dataset search unless it is strictly necessary to resolve a variable name. "
                    "Prefer proceeding directly to variable resolution and the requested data/plot tool."
                ),
            )
        )
    elif contextual_rewrite:
        active_user_message = contextual_rewrite
        messages.append(
            LLMMessage(
                role="system",
                content=(
                    "The current user turn is a follow-up refinement of the existing request. "
                    "Preserve the prior field, region, date, depth, and action unless the user changed them explicitly. "
                    "Do not restart discovery from scratch on unrelated datasets."
                ),
            )
        )

    # Prior conversation (as plain chat messages)
    for m in conversation:
        role = m.get("role", "user")
        content = m.get("content") or ""
        if content.strip():
            messages.append(LLMMessage(role=role, content=content))

    # Current user turn
    messages.append(LLMMessage(role="user", content=active_user_message))

    tool_trace: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    pycmap_code_snippets: list[str] = []

    remaining = max(0, int(max_tool_calls))

    # If the model returns non-JSON, we'll ask it to retry a couple times.
    invalid_json_retries = 0
    # If the user request clearly needs tools (data/catalog/map), don't allow
    # a plan-only final answer.
    force_tool_retries = 0
    requires_tools = _request_requires_tools(active_user_message)

    # Cache identical tool calls within a single /chat request to avoid
    # wasting tool budget (and to prevent accidental loops).
    tool_cache: dict[tuple[str, str], dict[str, Any]] = {}

    # If tools are disabled, give the model a clear instruction up-front.
    if remaining <= 0:
        messages.append(
            LLMMessage(
                role="system",
                content=(
                    "Tool calling is DISABLED for this request (max_tool_calls=0). "
                    "You MUST respond with JSON type='final' and provide your best answer "
                    "using only the provided conversation and RAG context."
                ),
            )
        )

    def _finalize_from_text(text: str) -> tuple[AgentFinal, list[dict[str, Any]]]:
        final = AgentFinal(assistant_message=text, code=None, artifacts=[])
        if not final.artifacts and artifacts:
            final.artifacts = artifacts
        if (final.code is None or not final.code.strip()) and pycmap_code_snippets:
            final.code = "\n\n".join(pycmap_code_snippets)
        return final, tool_trace

    while True:
        resp = llm.complete(messages)
        obj = _try_parse_json(resp.content)

        # If model didn't return JSON, ask it to retry (strict JSON only).
        if not isinstance(obj, dict) or "type" not in obj:
            if invalid_json_retries < 2:
                invalid_json_retries += 1
                messages.append(
                    LLMMessage(
                        role="user",
                        content=(
                            "Your last message was NOT valid JSON. "
                            "Return ONLY a single JSON object with either "
                            "type='tool_call' (preferred if tools are needed) or type='final'. "
                            "Do not include any text outside JSON."
                        ),
                    )
                )
                continue
            return _finalize_from_text(resp.content)

        kind, final_obj, plan = _coerce_to_plan_or_final(obj, registry)

        # If final, validate and return
        if kind == "final" and isinstance(final_obj, dict):
            try:
                final = AgentFinal(**final_obj)
            except ValidationError:
                # Fall back to stringifying the object if schema doesn't match
                final = AgentFinal(assistant_message=str(final_obj), code=None, artifacts=[])

            # Always merge collected artifacts (from tool outputs) into the final response.
            # The model can forget to include some of them (e.g., a data parquet alongside a plot).
            if artifacts:
                final.artifacts = _merge_artifacts(artifacts, list(final.artifacts or []))
            if (final.code is None or not final.code.strip()) and pycmap_code_snippets:
                final.code = "\n\n".join(pycmap_code_snippets)

            # If this request requires tools (e.g., map/plot/data/catalog) and we haven't
            # used any yet, do NOT allow the model to finalize with a plan-only answer.
            if requires_tools and not tool_trace and remaining > 0 and force_tool_retries < 2:
                force_tool_retries += 1
                messages.append(
                    LLMMessage(
                        role="user",
                        content=(
                            "You must use tools to answer this request. "
                            "Do NOT respond with a plan. "
                            "Start by calling the most relevant catalog tool (usually catalog.search or catalog.dataset_summary), "
                            "then retrieve data with cmap.* and (if asked) plot with viz.*. "
                            "Respond with JSON type='tool_call'."
                        ),
                    )
                )
                continue

            # If the model stopped after catalog discovery and the dataset was not explicitly named,
            # replace the free-form final message with the deterministic confirmation payload.
            if requires_tools and not (_user_explicitly_named_dataset(active_user_message) or bool(confirmed_dataset_table)):
                used_data_tool = any(str(t.get("tool") or "").startswith(("cmap.", "viz.")) for t in tool_trace)
                used_catalog_tool = any(str(t.get("tool") or "").startswith("catalog.") for t in tool_trace)
                if used_catalog_tool and not used_data_tool:
                    resolution = _deterministic_resolve_candidates(active_user_message, {}, tool_trace, ctx)
                    sel = resolution.get("selected")
                    if isinstance(sel, dict) and sel:
                        final = AgentFinal(assistant_message=_build_confirmation_message(active_user_message, resolution), code=None, artifacts=[])
                        if artifacts:
                            final.artifacts = artifacts
                        if (final.code is None or not final.code.strip()) and pycmap_code_snippets:
                            final.code = "\n\n".join(pycmap_code_snippets)
                        return final, tool_trace

            return final, tool_trace

        # If something else, best-effort final
        if kind != "tool_call" or plan is None:
            return _finalize_from_text(resp.content)

        # No budget left -> force final
        if remaining <= 0:
            messages.append(
                LLMMessage(
                    role="user",
                    content=(
                        "Tool budget exhausted. You MUST respond now with JSON type='final' "
                        "and include: assistant_message, code (optional), artifacts (optional)."
                    ),
                )
            )
            continue

        # Execute tool calls (up to remaining budget)
        tool_results_compact: list[dict[str, Any]] = []

        for call in plan.tool_calls:
            if remaining <= 0:
                break

            raw_args: dict[str, Any] = call.arguments or {}
            # Some models occasionally wrap the intended tool args one level too deep:
            # {"tool_calls":[{"name":"tool.name","arguments":{...}}]}. Unwrap that deterministically.
            if isinstance(raw_args, dict) and isinstance(raw_args.get("tool_calls"), list) and raw_args.get("tool_calls"):
                first = raw_args.get("tool_calls")[0]
                if isinstance(first, dict) and isinstance(first.get("arguments"), dict):
                    raw_args = dict(first.get("arguments") or {})
            exec_args: dict[str, Any] = raw_args
            trace_item: dict[str, Any] = {"tool": call.name, "status": "ok", "arguments": raw_args}

            # Normalize surface requests before tool execution.
            if call.name.startswith("cmap.") or call.name.startswith("viz."):
                if confirmed_dataset_table and (not isinstance(exec_args, dict) or str(exec_args.get("table") or "").strip().lower() != confirmed_dataset_table.lower()):
                    exec_args = dict(exec_args or {})
                    exec_args["table"] = confirmed_dataset_table
                    trace_item["original_arguments"] = raw_args
                    trace_item["arguments"] = exec_args
                    trace_item["arg_sanitized"] = True
                surfaced, changed_surface = _maybe_apply_surface_defaults(active_user_message, exec_args)
                if changed_surface:
                    exec_args = surfaced
                    trace_item.setdefault("original_arguments", raw_args)
                    trace_item["arguments"] = surfaced
                    trace_item["arg_sanitized"] = True
                # If a table is already chosen, resolve generic user-facing variable names (e.g. nitrate -> no3)
                # before execution so confirmed dataset choices carry forward cleanly.
                if isinstance(exec_args, dict) and exec_args.get("table") and exec_args.get("variable"):
                    resolved_var = _maybe_resolve_variable_for_table(str(exec_args.get("variable") or ""), str(exec_args.get("table") or ""), ctx)
                    if resolved_var and resolved_var != exec_args.get("variable"):
                        exec_args = dict(exec_args)
                        exec_args["variable"] = resolved_var
                        if "original_arguments" not in trace_item:
                            trace_item["original_arguments"] = raw_args
                        trace_item["arguments"] = exec_args
                        trace_item["arg_sanitized"] = True

            # For regional single-date field requests, a generic "plot" should default to a map.
            exec_call_name = call.name
            if exec_call_name in {"cmap.time_series", "viz.plot_timeseries"} and _request_prefers_map(active_user_message, exec_args):
                exec_call_name = "viz.plot_map"
                trace_item["tool"] = exec_call_name
                trace_item["rewritten_from"] = call.name
                if isinstance(exec_args, dict) and "interval" in exec_args:
                    exec_args = {k: v for k, v in exec_args.items() if k != "interval"}
                    trace_item.setdefault("original_arguments", raw_args)
                    trace_item["arguments"] = exec_args
                    trace_item["arg_sanitized"] = True

            # Deterministic dataset-resolution guard: if the user did not explicitly name a dataset,
            # do not execute cmap/viz tools yet. Resolve a best candidate + alternates and ask for confirmation.
            if _should_request_dataset_confirmation(active_user_message, exec_call_name, exec_args) and not bool(confirmed_dataset_table):
                resolution = _deterministic_resolve_candidates(active_user_message, exec_args, tool_trace, ctx)
                sel = resolution.get("selected")
                if isinstance(sel, dict) and sel:
                    trace_item["status"] = "blocked"
                    trace_item["result_preview"] = {"tool": call.name, "resolution": resolution}
                    tool_trace.append(trace_item)
                    final = AgentFinal(assistant_message=_build_confirmation_message(active_user_message, resolution), code=None, artifacts=[])
                    if artifacts:
                        final.artifacts = artifacts
                    if (final.code is None or not final.code.strip()) and pycmap_code_snippets:
                        final.code = "\n\n".join(pycmap_code_snippets)
                    return final, tool_trace

            # Guardrail: if the model invents tolerances that the user didn't specify,
            # strip them so the tool can infer from catalog metadata.
            if call.name == "cmap.colocalize":
                sanitized, changed = _sanitize_colocalize_arguments(active_user_message, raw_args)
                if changed:
                    exec_args = sanitized
                    trace_item["original_arguments"] = raw_args
                    trace_item["arguments"] = sanitized
                    trace_item["arg_sanitized"] = True

            # Deduplicate identical tool calls in a single run (models sometimes retry).
            args_key = json.dumps(exec_args, sort_keys=True, default=str)
            cache_key = (exec_call_name, args_key)
            if cache_key in tool_cache:
                compact = tool_cache[cache_key]
                tool_results_compact.append(compact)
                trace_item["status"] = "cached"
                trace_item["result_preview"] = compact
                tool_trace.append(trace_item)
                # Do NOT decrement remaining; we didn't execute a tool.
                continue

            try:
                tool = registry.get(exec_call_name)
                args = tool.args_model(**exec_args)
                result = tool.fn(args, ctx)

                # Gather artifacts & code
                if isinstance(result, dict):
                    if "artifact" in result:
                        artifacts.append(_normalize_artifact(result["artifact"]))
                    if "plot" in result:
                        artifacts.append(_normalize_artifact(result["plot"]))
                    if "data_artifact" in result:
                        artifacts.append(_normalize_artifact(result["data_artifact"]))
                    if "pycmap_code" in result and result["pycmap_code"]:
                        pycmap_code_snippets.append(str(result["pycmap_code"]))
                    compact = _tool_result_for_llm(call.name, result)
                else:
                    compact = _tool_result_for_llm(call.name, {"result": result})

                tool_results_compact.append(compact)
                trace_item["result_preview"] = compact
                tool_cache[cache_key] = compact

            except SystemExit as e:
                # Some third-party libraries (e.g., pycmap) call sys.exit() on validation errors.
                # Treat this as a tool error so the agent can recover instead of crashing the server.
                msg = str(e) or "SystemExit"
                trace_item["status"] = "error"
                trace_item["error"] = msg
                compact = {"tool": call.name, "error": {"code": "system_exit", "message": msg}}
                tool_results_compact.append(compact)
                trace_item["result_preview"] = compact
                tool_cache[cache_key] = compact

            except Exception as e:
                trace_item["status"] = "error"
                # If a tool raises a structured input error, preserve the full payload
                # so the model can recover (e.g., unknown table/variable suggestions).
                payload = None
                if hasattr(e, "to_dict") and callable(getattr(e, "to_dict")):
                    try:
                        payload = e.to_dict()  # type: ignore[attr-defined]
                    except Exception:
                        payload = None
                trace_item["error"] = payload or str(e)
                compact = {"tool": call.name, "error": payload or {"code": "exception", "message": str(e)}}
                tool_results_compact.append(compact)
                trace_item["result_preview"] = compact
                tool_cache[cache_key] = compact

            tool_trace.append(trace_item)
            remaining -= 1

        # Feed tool results back to the model and ask it to either finalize or request more tools
        messages.append(
            LLMMessage(
                role="user",
                content=(
                    "TOOL_RESULTS (compact):\n"
                    + json.dumps(tool_results_compact, indent=2, default=str)
                    + "\n\nNow respond in JSON with EITHER:\n"
                    "1) type='final' and fields: assistant_message, code (optional), artifacts (optional)\n"
                    "OR\n"
                    "2) type='tool_call' and tool_calls[] if you still need more tools.\n"
                ),
            )
        )