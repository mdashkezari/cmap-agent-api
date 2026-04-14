"""Agent execution loop.

Each call to ``execute_plan`` runs one user turn:

1. Intent is extracted via a lightweight LLM call (``extract_intent``).
2. ThreadState is updated from the intent.
3. The main tool-calling loop runs until the model returns a ``type=final``
   response or the tool budget is exhausted.
4. Dataset-confirmation and argument sanitization are applied deterministically
   before each tool execution.

State that was previously reconstructed by regex-scanning the conversation
history is now read directly from ``ThreadState`` (passed in via ``ctx``).
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, Field, ValidationError

from cmap_agent.agent.intent import UserIntent, extract_intent
from cmap_agent.agent.state import ThreadState
from cmap_agent.llm.base import LLMClient
from cmap_agent.llm.types import LLMMessage
from cmap_agent.tools.catalog_tools import (
    CatalogSearchArgs,
    CatalogSearchKBFArgs,
    CatalogSearchVariablesArgs,
    _bbox_overlaps,
    _fetch_datasets_by_tables,
    catalog_search,
    catalog_search_kb_first,
    catalog_search_variables,
)
from cmap_agent.tools.limits import LIMITS
from cmap_agent.tools.registry import ToolRegistry
from cmap_agent.utils import to_jsonable

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models for the JSON protocol between runner and LLM
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _try_parse_json(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        parts = text.split("\n")
        parts = parts[1:] if parts else parts
        if parts and parts[-1].strip().startswith("```"):
            parts = parts[:-1]
        text = "\n".join(parts).strip()
    try:
        return json.loads(text)
    except Exception:
        return None


def _coerce_to_plan_or_final(
    obj: dict[str, Any],
    registry: ToolRegistry,
) -> tuple[str, dict[str, Any] | None, AgentToolCallPlan | None]:
    """Normalise model JSON output into (kind, final_obj, plan).

    Handles the common case where the model returns a direct tool-call dict
    (``{"type": "catalog.search", ...}``) instead of the wrapper format.
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
    if registry.has(t):
        args = obj.get("arguments") if isinstance(obj.get("arguments"), dict) else None
        if args is None:
            args = {k: v for k, v in obj.items() if k != "type"}
        plan = AgentToolCallPlan(
            type="tool_call",
            tool_calls=[ToolCall(name=t, arguments=args or {})],
        )
        return "tool_call", None, plan
    return "other", None, None


# ---------------------------------------------------------------------------
# Artifact normalisation
# ---------------------------------------------------------------------------

def _strip_url_query(u: str) -> str:
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


def _normalize_artifact(a: dict[str, Any]) -> dict[str, Any]:
    """Normalise artifact dicts from different sources into a canonical shape."""
    out = dict(a or {})

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


def _artifact_key(a: dict[str, Any]) -> tuple[str, str]:
    a = _normalize_artifact(a)
    uri = _strip_url_query(a.get("uri") or a.get("url") or "")
    return (str(a.get("type") or ""), uri)


def _merge_artifacts(
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for a in primary:
        na = _normalize_artifact(a)
        k = _artifact_key(na)
        if k not in index:
            index[k] = na
            merged.append(na)
        else:
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


# ---------------------------------------------------------------------------
# Tool result compaction (what gets fed back to the LLM)
# ---------------------------------------------------------------------------

def _compact_dataset_summary_item(ds: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "table", "dataset_id", "short_name", "title", "description", "keywords",
        "source", "spatial_resolution", "temporal_resolution",
        "time_coverage_start", "time_coverage_end",
        "lat_min", "lat_max", "lon_min", "lon_max",
        "depth_min", "depth_max", "updated_at",
    ]
    out: dict[str, Any] = {k: ds.get(k) for k in keep if ds.get(k) is not None}
    if isinstance(out.get("description"), str) and len(out["description"]) > 800:
        out["description"] = out["description"][:800]
    vars_ = ds.get("variables")
    if isinstance(vars_, list):
        compact_vars = []
        for v in vars_[:15]:
            if isinstance(v, dict):
                vv = {kk: v.get(kk) for kk in ("variable", "long_name", "unit") if v.get(kk) is not None}
                compact_vars.append(vv)
            else:
                compact_vars.append(v)
        out["variables"] = compact_vars
    return out


def _compact_colocalize_resolved(resolved: dict[str, Any]) -> dict[str, Any]:
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
        compact_targets = []
        for t in targets[:6]:
            if not isinstance(t, dict):
                continue
            tol = t.get("tolerances") if isinstance(t.get("tolerances"), dict) else {}
            dt = tol.get("dt") if isinstance(tol.get("dt"), dict) else {}
            compact_targets.append({
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
            })
        out["targets"] = compact_targets
        out["num_targets"] = int(len(targets))
    return out


def _tool_result_for_llm(name: str, result: dict[str, Any]) -> dict[str, Any]:
    """Build a compact, JSON-safe view of a tool result for LLM consumption."""
    compact: dict[str, Any] = {"tool": name}

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
            compact["matches"] = [_compact_dataset_summary_item(m) for m in matches[:10] if isinstance(m, dict)]
        if isinstance(result.get("note"), str):
            compact["note"] = result.get("note")[:400]
        return to_jsonable(compact)

    if name == "cmap.colocalize":
        for key in ("status", "rows", "columns"):
            if key in result:
                val = result.get(key)
                compact[key] = (val or [])[:50] if key == "columns" else val
        if isinstance(result.get("artifact"), dict):
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
        return to_jsonable(compact)

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
        res = result.get("results")
        if isinstance(res, list):
            compact["results"] = res[:10]
        return to_jsonable(compact)

    # Generic compaction
    for key in ("count", "rows"):
        if key in result:
            compact[key] = result[key]
    if "columns" in result:
        compact["columns"] = (result["columns"] or [])[:50]
    if isinstance(result.get("artifact"), dict):
        compact["artifact_url"] = result["artifact"].get("url")
        compact["artifact_type"] = result["artifact"].get("type")
    if isinstance(result.get("plot"), dict):
        compact["plot_url"] = result["plot"].get("url")
    if isinstance(result.get("data_artifact"), dict):
        compact["data_artifact_url"] = result["data_artifact"].get("url")
        compact["data_artifact_type"] = result["data_artifact"].get("type")
    for key in ("preview", "results", "metadata", "variables"):
        if key in result:
            try:
                limit = 20 if key == "preview" else (10 if key in ("results", "metadata") else 15)
                compact[key] = (result[key] or [])[:limit]
            except Exception:
                pass
    if isinstance(result.get("resolved"), dict):
        compact["resolved"] = result.get("resolved")
    if result.get("substitution_warning"):
        compact["substitution_warning"] = result["substitution_warning"]
    return to_jsonable(compact)


# ---------------------------------------------------------------------------
# Argument sanitization
# ---------------------------------------------------------------------------

_COLOCALIZE_TOL_KEYS = ("dt_tol_days", "lat_tol_deg", "lon_tol_deg", "depth_tol_m")


def _sanitize_colocalize_arguments(
    user_message: str, args: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    """Strip model-invented colocalization tolerances not requested by the user."""
    msg = (user_message or "").lower()
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
    depth_pat3 = re.compile(rf"\b\d+(?:\.\d+)?\b\s*(?:m|meter|meters)\b[^\w]{{0,40}}depth[^\w]{{0,20}}{tol_ctx}", re.I)
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
        if not keep_dt and "dt_tol_days" in tt:
            del tt["dt_tol_days"]
            changed = True
        if not keep_spatial:
            for k in ("lat_tol_deg", "lon_tol_deg"):
                if k in tt:
                    del tt[k]
                    changed = True
        if not keep_depth and "depth_tol_m" in tt:
            del tt["depth_tol_m"]
            changed = True
        new_targets.append(tt)

    out["targets"] = new_targets
    return out, changed


def _maybe_apply_surface_defaults(
    intent: UserIntent, args: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    """Clamp depth bounds to a shallow surface slice when the intent requests it."""
    if not isinstance(args, dict) or not intent.surface_only:
        return args, False
    out = dict(args)
    d1 = out.get("depth1")
    d2 = out.get("depth2")
    if d1 is None or d2 is None:
        out["depth1"] = 0
        out["depth2"] = 5
        return out, True
    try:
        fd1, fd2 = float(d1), float(d2)
    except Exception:
        out["depth1"] = 0
        out["depth2"] = 5
        return out, True
    if fd1 < 0 or fd2 < 0 or fd2 > 50 or (fd2 - fd1) > 50:
        out["depth1"] = 0
        out["depth2"] = 5
        return out, True
    return args, False


def _maybe_resolve_variable_for_table(
    query_variable: str, table: str, ctx: dict[str, Any]
) -> str | None:
    """Resolve a human-readable variable name to the exact catalog VarName."""
    qv = str(query_variable or "").strip()
    table = str(table or "").strip()
    if not qv or not table:
        return None
    try:
        res = catalog_search_variables(
            CatalogSearchVariablesArgs(query=qv, table_hint=table, limit=5), ctx
        )
    except Exception:
        return None
    rows = [r for r in (res.get("results") or []) if isinstance(r, dict)]
    if not rows:
        return None
    # Return the first result — variable search is already ranked by relevance
    return str(rows[0].get("variable") or "") or None


# ---------------------------------------------------------------------------
# Dataset confirmation flow
# ---------------------------------------------------------------------------

def _user_explicitly_named_dataset(user_message: str) -> bool:
    """Return True when the user typed a literal tblXXX table name."""
    um = (user_message or "").lower()
    return bool(re.search(r"\btbl[a-z0-9_]+\b", um))



def _should_request_dataset_confirmation(
    user_message: str, call_name: str, raw_args: dict[str, Any],
    intent: "UserIntent | None" = None,
) -> bool:
    if _user_explicitly_named_dataset(user_message):
        return False
    if not isinstance(raw_args, dict):
        return False
    if call_name.startswith("catalog."):
        return False
    return call_name.startswith("cmap.") or call_name.startswith("viz.")


def _request_prefers_map(intent: UserIntent, args: dict[str, Any]) -> bool:
    if intent.action == "time_series":
        return False
    if intent.action == "map":
        return True
    has_roi = intent.has_bounds()
    dt1 = str(args.get("dt1") or "")[:10]
    dt2 = str(args.get("dt2") or "")[:10]
    single_date = bool(dt1 and dt1 == dt2)
    return has_roi and single_date


def _parse_date_safe(s: Any) -> str | None:
    s = str(s or "").strip()
    if not s:
        return None
    m = re.match(r"(\d{4}-\d{2}-\d{2})", s)
    return m.group(1) if m else None


def _candidate_within_time(r: dict[str, Any], dt1: str | None, dt2: str | None) -> bool:
    q1 = _parse_date_safe(dt1)
    q2 = _parse_date_safe(dt2) or q1
    if not q1:
        return True
    t1 = _parse_date_safe(r.get("time_min") or r.get("TimeMin"))
    t2 = _parse_date_safe(r.get("time_max") or r.get("TimeMax"))
    if not t1 or not t2:
        return True
    return t1 <= q1 <= t2 and t1 <= q2 <= t2


def _candidate_within_space(r: dict[str, Any], intent: UserIntent) -> bool:
    if not intent.has_bounds():
        return True
    lat_min, lat_max = r.get("lat_min"), r.get("lat_max")
    lon_min, lon_max = r.get("lon_min"), r.get("lon_max")
    if any(v is None for v in (lat_min, lat_max, lon_min, lon_max)):
        return True
    try:
        return _bbox_overlaps(
            lat_min, lat_max, lon_min, lon_max,
            float(intent.lat1), float(intent.lat2),
            float(intent.lon1), float(intent.lon2),
        )
    except Exception:
        return True


def _bare_query(search_query: str) -> str:
    """Strip make/sensor words to get the core variable name for discovery."""
    q = (search_query or "").lower()
    for drop in ("satellite", "model", "assimilation", "observation", "in-situ", "insitu"):
        q = q.replace(drop, "").strip()
    return q.strip() or search_query


def _extract_catalog_results_from_trace(
    tool_trace: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pull dataset results from prior catalog tool calls in the tool trace.

    When the model already called a catalog tool that returned good results
    (e.g. catalog.search_kb_first returned MODIS CHL), those results are
    reused directly rather than running fresh searches.  This prevents the
    deterministic resolution step from picking a different (worse) dataset
    than the one the catalog tool already found.
    """
    catalog_tool_names = {
        "catalog.search_kb_first",
        "catalog.search",
        # catalog.search_variables intentionally excluded — it returns variable-level
        # matches that often come from cruise datasets (e.g. a variable named "Chlorophyll"
        # in a CTD cruise), not the gridded satellite products the user usually wants.
        # catalog.dataset_summary intentionally excluded — it returns datasets
        # matching a program/institution query (e.g. "Armbrust group") which are not
        # relevant candidates for a data request about a specific variable.
    }
    all_results: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in reversed(tool_trace or []):  # most recent first
        tool_name = str(item.get("tool") or "")
        if tool_name not in catalog_tool_names:
            continue
        preview = item.get("result_preview") or {}
        # Collect selected + alternates + results from the preview
        for key in ("results", "alternates"):
            for r in (preview.get(key) or []):
                if not isinstance(r, dict):
                    continue
                t = str(r.get("table") or "").strip()
                if t and t not in seen:
                    seen.add(t)
                    all_results.append(r)
        sel = preview.get("selected")
        if isinstance(sel, dict):
            t = str(sel.get("table") or "").strip()
            if t and t not in seen:
                seen.add(t)
                all_results.insert(0, sel)  # selected goes first

    return all_results


def _deterministic_resolve_candidates(
    intent: UserIntent,
    pending_args: dict[str, Any],
    tool_trace: list[dict[str, Any]],
    ctx: dict[str, Any],
) -> dict[str, Any]:
    """Build a confirmed candidate list from prior tool trace + fresh searches.

    Strategy:
    1) Reuse results already returned by catalog tools in this turn's tool_trace
       — they are the most relevant results the model already found.
    2) Augment with fresh KB search (bare variable name, no ROI/filters).
    3) Augment with fresh SQL LIKE search (bare variable name).
    4) Merge, rank, filter to ROI/time, select best.
    """
    # Strip make/sensor words — discovery by variable topic only
    bare_q = _bare_query(intent.search_query or "")

    # 1) Reuse results from prior catalog tool calls (trace + state cache)
    trace_results = _extract_catalog_results_from_trace(tool_trace)

    # Also pull from thread_state cache (cross-turn persistence)
    thread_state = ctx.get("thread_state")
    state_results = list(thread_state.last_catalog_results or []) if thread_state else []

    # Combine: trace first (same-turn, most current), then state (cross-turn)
    seen_pre: set[str] = set()
    prior_results: list[dict[str, Any]] = []
    for r in trace_results + state_results:
        t = str(r.get("table") or "").strip()
        if t and t not in seen_pre:
            seen_pre.add(t)
            prior_results.append(r)

    # 2) KB semantic search — only when insufficient prior results.
    # Skip fresh KB search when trace/state already has good candidates.
    # A fresh KB search may return unrelated datasets (e.g. DOM data matching
    # "chlorophyll" in its description) that would corrupt the ranking.
    if len(prior_results) >= 3:
        kb_results = []
    else:
        try:
            kb_args = CatalogSearchKBFArgs(
                query=bare_q,
                lat1=None, lat2=None, lon1=None, lon2=None,
                dt1=None, dt2=None,
                make=None, sensor=None,
                limit=15,
            )
            kb_res = catalog_search_kb_first(kb_args, ctx)
            kb_results = list(kb_res.get("results") or []) if isinstance(kb_res.get("results"), list) else []
        except Exception:
            kb_results = []

    results = prior_results  # prior results carry correct kb_scores

    # 3) SQL LIKE search — bare variable name, no filters
    try:
        sql_res = catalog_search(CatalogSearchArgs(query=bare_q, limit=30), ctx)
        sql_results = list(sql_res.get("results") or [])
    except Exception:
        sql_results = []

    # 3) Variable-driven augmentation
    var_tables: list[str] = []
    try:
        var_res = catalog_search_variables(
            CatalogSearchVariablesArgs(query=bare_q, limit=20), ctx
        )
        for row in var_res.get("results") or []:
            if isinstance(row, dict):
                t = str(row.get("table") or "").strip()
                if t and t not in var_tables:
                    var_tables.append(t)
    except Exception:
        pass

    var_meta: list[dict[str, Any]] = []
    if var_tables:
        try:
            store = ctx.get("store")
            # No ROI filtering at discovery — ROI applied post-discovery
            meta_rows = _fetch_datasets_by_tables(store, var_tables[:20]) if store else []
            var_meta = meta_rows
        except Exception:
            pass

    # Merge all sources — dedup by table name
    # Order: trace results first (most relevant — already found by model),
    # then fresh KB results, then SQL, then variable-derived metadata
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for src in (results, kb_results, sql_results, var_meta):
        for r in src:
            if not isinstance(r, dict):
                continue
            t = str(r.get("table") or "").strip()
            if not t or t in seen:
                continue
            seen.add(t)
            merged.append(r)

    # Post-rank using improved scoring (gridded preference, kb_score weight)
    try:
        from cmap_agent.tools.catalog_tools import _post_rank_catalog_results
        merged = _post_rank_catalog_results(
            merged, query=bare_q, dt1=intent.dt1, dt2=intent.dt2,
            lat1=intent.lat1, lat2=intent.lat2, lon1=intent.lon1, lon2=intent.lon2,
        )
    except Exception:
        pass

    # Apply ROI and time filters AFTER ranking
    ranked = [
        r for r in merged
        if _candidate_within_time(r, intent.dt1, intent.dt2)
        and _candidate_within_space(r, intent)
    ]
    if not ranked:
        ranked = merged  # fallback: unfiltered ranked list

    selected = ranked[0] if ranked else None
    base_table = str((selected or {}).get("table") or "")
    alternates = [r for r in ranked if str(r.get("table") or "") != base_table][:5]

    return {
        "query": {"q": bare_q, "lat1": intent.lat1, "lat2": intent.lat2,
                  "lon1": intent.lon1, "lon2": intent.lon2,
                  "dt1": intent.dt1, "dt2": intent.dt2},
        "selected": selected,
        "alternates": alternates,
        "results": ranked[:10],
    }


def _candidate_time_text(r: dict[str, Any]) -> str | None:
    tmin = r.get("time_min") or r.get("TimeMin")
    tmax = r.get("time_max") or r.get("TimeMax")
    return f"{tmin} to {tmax}" if tmin and tmax else None


def _candidate_spatial_text(r: dict[str, Any]) -> str | None:
    lat_min, lat_max = r.get("lat_min"), r.get("lat_max")
    lon_min, lon_max = r.get("lon_min"), r.get("lon_max")
    if any(v is None for v in (lat_min, lat_max, lon_min, lon_max)):
        return None
    return f"lat {lat_min} to {lat_max}; lon {lon_min} to {lon_max}"


def _build_confirmation_message(resolution: dict[str, Any]) -> str:
    sel = resolution.get("selected") or {}
    alts = list(resolution.get("alternates") or [])
    q = dict(resolution.get("query") or {})
    if not isinstance(sel, dict) or not sel:
        return "A dataset could not be confidently resolved. Please specify the dataset to use."

    def _fmt(r: dict[str, Any]) -> list[str]:
        name = r.get("name") or r.get("title") or r.get("table")
        return [
            f"- {name} (`{r.get('table')}`)",
            f"  - Temporal coverage: {_candidate_time_text(r) or '?'}",
            f"  - Spatial coverage: {_candidate_spatial_text(r) or '?'}",
            f"  - Temporal resolution: {r.get('temporal_resolution') or '?'}",
            f"  - Spatial resolution: {r.get('spatial_resolution') or '?'}",
        ]

    lines = [
        "A best CMAP dataset candidate was found for this request, but the dataset was not explicitly specified.",
        "Best candidate:",
        *_fmt(sel),
    ]
    if alts:
        lines.append("Other possible candidates:")
        for a in alts[:5]:
            lines.extend(_fmt(a))
    if any(q.get(k) is not None for k in ("dt1", "lat1", "lon1")):
        lines.append("Datasets outside the requested temporal or spatial coverage are omitted from this list.")
    lines.append("Should this best candidate be used, or would another candidate be preferred?")
    return "\n".join(lines)


def _is_affirmative_short_reply(text: str) -> bool:
    s = str(text or "").strip().lower()
    if not s or len(re.findall(r"\w+", s)) > 8:
        return False
    return any(tok in s for tok in [
        "yes", "yeah", "yep", "ok", "okay", "go ahead", "sure",
        "please do", "do it", "use that", "that works", "sounds good",
    ])


def _resolve_pending_confirmation(
    user_message: str,
    thread_state: ThreadState,
) -> dict[str, Any] | None:
    """Check whether the user message is confirming a pending dataset selection.

    Returns the selected candidate dict if confirmed, None otherwise.
    """
    pending = thread_state.pending_confirmation
    if not pending or not isinstance(pending.get("selected"), dict):
        return None

    msg = str(user_message or "").strip().lower()
    candidates = [c for c in (pending.get("candidates") or []) if isinstance(c, dict)]

    # Explicit table name mention
    tbl_match = re.search(r"\btbl[a-z0-9_]+\b", msg)
    if tbl_match:
        tbl = tbl_match.group(0)
        for c in candidates:
            if str(c.get("table") or "").lower() == tbl:
                return c

    # Named candidate match
    for c in candidates:
        name = str(c.get("name") or "").strip().lower()
        table = str(c.get("table") or "").strip().lower()
        if (name and name in msg) or (table and table in msg):
            return c

    # Short affirmative reply → accept the top candidate
    token_count = len(re.findall(r"\w+", msg))
    if token_count <= 8 and _is_affirmative_short_reply(msg):
        return pending.get("selected")
    return None


# ---------------------------------------------------------------------------
# Main execution loop
# ---------------------------------------------------------------------------

def execute_plan(
    *,
    llm: LLMClient,
    registry: ToolRegistry,
    system_prompt: str,
    conversation: list[dict[str, Any]],
    user_message: str,
    ctx: dict[str, Any],
    max_tool_calls: int = LIMITS.max_tool_calls,
) -> tuple[AgentFinal, list[dict[str, Any]], ThreadState]:
    """Run one agent turn, executing tools iteratively until a final answer.

    Returns: (final_response, tool_trace, updated_thread_state)

    The returned ThreadState must be persisted by the caller (server/app.py).
    """
    # ---- Load thread state ------------------------------------------------
    thread_state: ThreadState = ctx.get("thread_state") or ThreadState()

    # ---- Extract structured intent from user message ----------------------
    intent = extract_intent(llm, user_message, thread_state)

    # ---- Check for pending confirmation reply -----------------------------
    confirmed_candidate = _resolve_pending_confirmation(user_message, thread_state)
    confirmed_dataset_table: str | None = None
    active_user_message = user_message

    if confirmed_candidate and str(confirmed_candidate.get("table") or "").strip():
        confirmed_dataset_table = str(confirmed_candidate.get("table") or "").strip()
        prev_request = str((thread_state.pending_confirmation or {}).get("request_message") or "").strip()
        # Tell the model to resolve the variable name before calling any data tool.
        # This prevents it from guessing variable names (e.g. "chlorophyll_concentration"
        # instead of the actual "chlor_a"), which causes colocalization errors.
        # Build the injected message. For colocalize requests with an uploaded file,
        # be explicit that the source is the file — not a CMAP dataset — to prevent
        # the model from confusing prior CMAP datasets in context with the source.
        _has_uploaded_file = bool(
            thread_state.pending_confirmation and
            "source_artifact" in str(thread_state.pending_confirmation or {})
        )
        _source_note = (
            " The source data is the user-uploaded file — do NOT use any CMAP table as source."
            if _has_uploaded_file else ""
        )
        active_user_message = (
            f"{prev_request} Use dataset {confirmed_dataset_table}."
            f"{_source_note}"
            f" First call catalog.list_variables with table='{confirmed_dataset_table}'"
            " to get the exact variable name. The full variable list will be returned —"
            " scan all long_name fields to identify the correct variable matching the"
            " user's request. Do not stop at the first page; all variables are present"
            " in the response. If the variable is not found by scanning long_name fields,"
            f" try catalog.search_variables with the user's variable keyword and"
            f" table_hint='{confirmed_dataset_table}' to locate it. Then proceed."
            if prev_request
            else f"Use dataset {confirmed_dataset_table}."
            f"{_source_note}"
            f" First call catalog.list_variables with table='{confirmed_dataset_table}'"
            " to get the exact variable name. The full variable list will be returned —"
            " scan all long_name fields to identify the correct variable matching the"
            " user's request. Do not stop at the first page; all variables are present"
            " in the response. If the variable is not found by scanning long_name fields,"
            f" try catalog.search_variables with the user's variable keyword and"
            f" table_hint='{confirmed_dataset_table}' to locate it. Then proceed."
        )
        thread_state.confirmed_table = confirmed_dataset_table
        thread_state.pending_confirmation = None
        # Clear cross-turn catalog results so they don't pollute candidate
        # resolution on the execution turn (e.g. Armbrust datasets from a
        # prior summary turn appearing as SST colocalize candidates).
        thread_state.last_catalog_results = []
    elif not intent.is_followup:
        # New request — update state from intent, clear prior confirmation
        thread_state.update_from_intent(intent)
    else:
        # Follow-up — merge intent into existing state without clearing
        thread_state.update_from_intent(intent)
        if thread_state.confirmed_table:
            # Only carry forward the confirmed table if the new request plausibly
            # refers to the same dataset (same search query topic). If the user has
            # switched topics (e.g. from SST to chlorophyll), clear it so the
            # confirmation guard fires correctly for the new dataset.
            prior_query = (thread_state.last_catalog_results[0].get("name") or "").lower()                 if thread_state.last_catalog_results else ""
            new_query = (intent.search_query or "").lower()
            # Check if ANY word from the new query appears in the confirmed table name
            confirmed_name = thread_state.confirmed_table.lower()
            query_words = [w for w in new_query.split() if len(w) > 3]
            topic_match = any(w in confirmed_name or w in prior_query for w in query_words)
            if topic_match:
                confirmed_dataset_table = thread_state.confirmed_table
            else:
                # Topic changed — clear confirmed table so confirmation fires again
                thread_state.confirmed_table = None
                thread_state.pending_confirmation = None

    # Expose confirmed table in ctx so cmap_tools can use it
    ctx["confirmed_table"] = confirmed_dataset_table

    # ---- Build message list -----------------------------------------------
    messages: list[LLMMessage] = [LLMMessage(role="system", content=system_prompt)]

    if confirmed_dataset_table:
        messages.append(LLMMessage(
            role="system",
            content=(
                f"The user has confirmed dataset `{confirmed_dataset_table}`. "
                "Treat it as explicitly specified. Do not restart dataset discovery. "
                "Proceed directly to variable resolution and execution."
            ),
        ))
    elif intent.is_followup and thread_state.confirmed_table:
        messages.append(LLMMessage(
            role="system",
            content=(
                "This is a follow-up refinement of the existing request. "
                "Preserve the prior field, region, date, depth, and action unless "
                "the user changed them explicitly."
            ),
        ))

    # Prior conversation history
    for m in conversation:
        role = m.get("role", "user")
        content = m.get("content") or ""
        if content.strip():
            messages.append(LLMMessage(role=role, content=content))

    messages.append(LLMMessage(role="user", content=active_user_message))

    # ---- Tool loop bookkeeping --------------------------------------------
    tool_trace: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    pycmap_code_snippets: list[str] = []
    remaining = max(0, int(max_tool_calls))
    invalid_json_retries = 0
    force_tool_retries = 0
    # Action signals from intent: if intent.action is not "chat", expect tools
    requires_tools = intent.action not in (None, "chat")
    tool_cache: dict[tuple[str, str], dict[str, Any]] = {}

    if remaining <= 0:
        messages.append(LLMMessage(
            role="system",
            content=(
                "Tool calling is DISABLED for this request (max_tool_calls=0). "
                "Respond with JSON type='final' using only the provided context."
            ),
        ))

    def _finalize(text: str) -> tuple[AgentFinal, list[dict[str, Any]], ThreadState]:
        final = AgentFinal(assistant_message=text, code=None, artifacts=[])
        if not final.artifacts and artifacts:
            final.artifacts = artifacts
        if not (final.code or "").strip() and pycmap_code_snippets:
            final.code = "\n\n".join(pycmap_code_snippets)
        return final, tool_trace, thread_state

    # ---- Main loop --------------------------------------------------------
    while True:
        resp = llm.complete(messages)
        obj = _try_parse_json(resp.content)

        if not isinstance(obj, dict) or "type" not in obj:
            if invalid_json_retries < 2:
                invalid_json_retries += 1
                messages.append(LLMMessage(
                    role="user",
                    content=(
                        "The last message was NOT valid JSON. "
                        "Return ONLY a single JSON object with either "
                        "type='tool_call' or type='final'. "
                        "No text outside JSON."
                    ),
                ))
                continue
            return _finalize(resp.content)

        kind, final_obj, plan = _coerce_to_plan_or_final(obj, registry)

        # Compute tool usage flags once, used in multiple checks below
        _used_data_tool_check = any(
            str(t.get("tool") or "").startswith(("cmap.", "viz."))
            for t in tool_trace
        )

        if kind == "final" and isinstance(final_obj, dict):
            try:
                final = AgentFinal(**final_obj)
            except ValidationError as ve:
                log.warning("AgentFinal ValidationError (using fallback): %s", ve)
                am = final_obj.get("assistant_message") or ""
                if not isinstance(am, str):
                    am = str(am)
                final = AgentFinal(assistant_message=am, code=None, artifacts=[])
            except Exception as e:
                log.warning("AgentFinal construction error (using fallback): %s", e)
                final = AgentFinal(assistant_message=str(final_obj), code=None, artifacts=[])

            if artifacts:
                final.artifacts = _merge_artifacts(artifacts, list(final.artifacts or []))
            if not (final.code or "").strip() and pycmap_code_snippets:
                final.code = "\n\n".join(pycmap_code_snippets)

            # If catalog tools returned results but the model still says "no results",
            # override with a targeted retry listing what was actually found
            if (
                tool_trace
                and remaining > 0
                and force_tool_retries < 2
                and not _used_data_tool_check  # defined below; compute early
            ):
                _early_trace_results = _extract_catalog_results_from_trace(tool_trace)
                _early_cat_used = any(
                    str(t.get("tool") or "").startswith("catalog.")
                    for t in tool_trace
                )
                if _early_cat_used and _early_trace_results:
                    # Model claimed no results but catalog already found datasets
                    force_tool_retries += 1
                    top = _early_trace_results[0]
                    top_name = top.get("name") or top.get("table")
                    top_table = top.get("table", "")
                    force_content = (
                        f"The catalog search DID return results — the top dataset is "
                        f"'{top_name}' (`{top_table}`). Do NOT say 'no results found'. "
                        f"Call catalog.list_variables with table='{top_table}' to get "
                        "the variable name, then call viz.plot_map with the user's "
                        "spatial bounds and date. Respond with JSON type='tool_call'."
                    )
                    messages.append(LLMMessage(role="user", content=force_content))
                    continue

            if requires_tools and intent.action != "summarize" and not tool_trace and remaining > 0 and force_tool_retries < 2:
                force_tool_retries += 1
                bare_q0 = _bare_query(intent.search_query or "") or (intent.search_query or "").strip() or "sea surface temperature"
                messages.append(LLMMessage(
                    role="user",
                    content=(
                        "Tools must be used to answer this request. "
                        f"Call catalog.search with query='{bare_q0}' — "
                        "bare variable name only, no region, no sensor, no make. "
                        "Then call catalog.list_variables to get the exact variable name. "
                        "Then call viz.plot_map with the spatial bounds and date. "
                        "Respond with JSON type='tool_call'."
                    ),
                ))
                continue

            # If the model used catalog tools but found no results, push back once
            # rather than accepting a "no results" final answer.
            _used_catalog_tool_check = any(
                str(t.get("tool") or "").startswith("catalog.")
                for t in tool_trace
            )
            if (
                requires_tools
                and not confirmed_dataset_table
                and not _used_data_tool_check
                and _used_catalog_tool_check
                and tool_trace
                and force_tool_retries < 2
                and remaining > 0
            ):
                # Model used catalog tools but still finalised without calling a data tool.
                # Push back once with a direct retry instruction.
                force_tool_retries += 1
                bare_q_retry = _bare_query(intent.search_query or "") or (intent.search_query or "").strip() or "sea surface temperature"
                messages.append(LLMMessage(
                    role="user",
                    content=(
                        "The catalog search returned no results. Do NOT finalise yet. "
                        f"Call catalog.search_kb_first with query='{bare_q_retry}' "
                        "(no sensor, no make, no ROI). "
                        "Then call catalog.list_variables and viz.plot_map. "
                        "Respond with JSON type='tool_call'."
                    ),
                ))
                continue

            # If the model stopped after catalog-only calls, trigger dataset confirmation
            if requires_tools and not confirmed_dataset_table and not _user_explicitly_named_dataset(active_user_message) and intent.action != "summarize":
                used_data_tool = any(
                    str(t.get("tool") or "").startswith(("cmap.", "viz."))
                    for t in tool_trace
                )
                used_catalog_tool = any(
                    str(t.get("tool") or "").startswith("catalog.")
                    for t in tool_trace
                )
                if used_catalog_tool and not used_data_tool:
                    resolution = _deterministic_resolve_candidates(intent, {}, tool_trace, ctx)
                    sel = resolution.get("selected")
                    if isinstance(sel, dict) and sel:
                        # Always confirm — user must explicitly approve dataset choice.
                        thread_state.pending_confirmation = {
                            "selected": sel,
                            "candidates": [sel] + list(resolution.get("alternates") or []),
                            "request_message": active_user_message,
                        }
                        final = AgentFinal(
                            assistant_message=_build_confirmation_message(resolution),
                            code=None,
                            artifacts=artifacts or [],
                        )
                        if pycmap_code_snippets:
                            final.code = "\n\n".join(pycmap_code_snippets)
                        return final, tool_trace, thread_state

            return final, tool_trace, thread_state

        if kind != "tool_call" or plan is None:
            return _finalize(resp.content)

        if remaining <= 0:
            messages.append(LLMMessage(
                role="user",
                content=(
                    "Tool budget exhausted. Respond now with JSON type='final' "
                    "including assistant_message, code (optional), artifacts (optional)."
                ),
            ))
            continue

        # ---- Execute tool calls -------------------------------------------
        tool_results_compact: list[dict[str, Any]] = []

        for call in plan.tool_calls:
            if remaining <= 0:
                break

            raw_args: dict[str, Any] = call.arguments or {}
            # Unwrap accidental double-nesting from some models
            if isinstance(raw_args, dict) and isinstance(raw_args.get("tool_calls"), list) and raw_args["tool_calls"]:
                first = raw_args["tool_calls"][0]
                if isinstance(first, dict) and isinstance(first.get("arguments"), dict):
                    raw_args = dict(first["arguments"])

            exec_args = raw_args
            trace_item: dict[str, Any] = {"tool": call.name, "status": "ok", "arguments": raw_args}

            if call.name.startswith(("cmap.", "viz.")):
                # For file-based colocalize: if source_artifact is present, remove any
                # source_table the model may have hallucinated from prior context (e.g.
                # listing Armbrust datasets then confusing one as the colocalize source).
                if call.name == "cmap.colocalize" and isinstance((exec_args or {}).get("source_artifact"), dict):
                    if exec_args.get("source_table"):
                        exec_args = dict(exec_args)
                        del exec_args["source_table"]
                        trace_item["original_arguments"] = raw_args
                        trace_item["arguments"] = exec_args
                        trace_item["arg_sanitized"] = True

                # Enforce confirmed table
                if confirmed_dataset_table and str((exec_args or {}).get("table") or "").strip().lower() != confirmed_dataset_table.lower():
                    exec_args = dict(exec_args or {})
                    exec_args["table"] = confirmed_dataset_table
                    trace_item["original_arguments"] = raw_args
                    trace_item["arguments"] = exec_args
                    trace_item["arg_sanitized"] = True

                # Apply surface depth defaults from intent
                surfaced, changed_surface = _maybe_apply_surface_defaults(intent, exec_args)
                if changed_surface:
                    exec_args = surfaced
                    trace_item.setdefault("original_arguments", raw_args)
                    trace_item["arguments"] = exec_args
                    trace_item["arg_sanitized"] = True

                # Resolve human variable names to catalog VarName
                if isinstance(exec_args, dict) and exec_args.get("table") and exec_args.get("variable"):
                    try:
                        resolved_var = _maybe_resolve_variable_for_table(
                            str(exec_args.get("variable") or ""),
                            str(exec_args.get("table") or ""),
                            ctx,
                        )
                        if resolved_var and resolved_var != exec_args.get("variable"):
                            exec_args = dict(exec_args)
                            exec_args["variable"] = resolved_var
                            trace_item.setdefault("original_arguments", raw_args)
                            trace_item["arguments"] = exec_args
                            trace_item["arg_sanitized"] = True
                    except Exception as _var_resolve_err:
                        log.warning(
                            "Variable resolution failed for %s/%s: %s",
                            exec_args.get("table"), exec_args.get("variable"),
                            _var_resolve_err,
                        )

            # Redirect time-series calls to map when intent signals a map
            exec_call_name = call.name
            if exec_call_name in {"cmap.time_series", "viz.plot_timeseries"} and _request_prefers_map(intent, exec_args):
                exec_call_name = "viz.plot_map"
                trace_item["tool"] = exec_call_name
                trace_item["rewritten_from"] = call.name
                if isinstance(exec_args, dict) and "interval" in exec_args:
                    exec_args = {k: v for k, v in exec_args.items() if k != "interval"}
                    trace_item.setdefault("original_arguments", raw_args)
                    trace_item["arguments"] = exec_args
                    trace_item["arg_sanitized"] = True

            # query_metadata intercept: if catalog.query_metadata already returned rows
            # in this turn, block follow-up catalog search tools — the SQL result is
            # definitive and the model should respond from it directly.
            _qm_success = any(
                t.get("tool") == "catalog.query_metadata"
                and t.get("status") == "ok"
                and (t.get("result_preview") or {}).get("rows")
                for t in tool_trace
            )
            if (
                _qm_success
                and exec_call_name.startswith(("catalog.search", "catalog.list", "catalog.dataset"))
            ):
                trace_item["status"] = "blocked"
                trace_item["blocked_reason"] = "query_metadata_answered"
                tool_trace.append(trace_item)
                messages.append(LLMMessage(
                    role="user",
                    content=(
                        "catalog.query_metadata already returned results for this request. "
                        "Do NOT call additional catalog search tools — the SQL query result "
                        "is the definitive answer. Respond with JSON type='final' and present "
                        "the query_metadata rows directly as your answer."
                    ),
                ))
                continue

            # Summarize intercept: if the request is informational, block data/viz tools
            # and push the model to answer in prose from catalog results only.
            # Exception: if confirmed_dataset_table is set, the user is confirming a
            # dataset selection (not asking for a summary) — never block in that case.
            if (
                intent.action == "summarize"
                and exec_call_name.startswith(("cmap.", "viz."))
                and not confirmed_dataset_table
            ):
                trace_item["status"] = "blocked"
                trace_item["blocked_reason"] = "summarize_action"
                tool_trace.append(trace_item)
                messages.append(LLMMessage(
                    role="user",
                    content=(
                        "This is an informational/summary request — do NOT call data retrieval "
                        "or plotting tools. The catalog search results already contain the "
                        "dataset information needed. Respond with JSON type='final' and provide "
                        "a comprehensive prose summary of the datasets found, including their "
                        "contents, temporal/spatial coverage, key variables, and any notes on "
                        "units and quality flags."
                    ),
                ))
                continue

            # Dataset confirmation guard
            if _should_request_dataset_confirmation(active_user_message, exec_call_name, exec_args, intent=intent) and not confirmed_dataset_table:
                resolution = _deterministic_resolve_candidates(intent, exec_args, tool_trace, ctx)
                sel = resolution.get("selected")
                if isinstance(sel, dict) and sel:
                    # Always confirm when user hasn't explicitly named the dataset —
                    # accuracy in scientific workflows requires explicit user approval.
                    thread_state.pending_confirmation = {
                        "selected": sel,
                        "candidates": [sel] + list(resolution.get("alternates") or []),
                        "request_message": active_user_message,
                    }
                    trace_item["status"] = "blocked"
                    trace_item["result_preview"] = {"tool": call.name, "resolution": resolution}
                    tool_trace.append(trace_item)
                    final = AgentFinal(
                        assistant_message=_build_confirmation_message(resolution),
                        code=None,
                        artifacts=artifacts or [],
                    )
                    if pycmap_code_snippets:
                        final.code = "\n\n".join(pycmap_code_snippets)
                    return final, tool_trace, thread_state

            # Strip unspecified colocalization tolerances
            if call.name == "cmap.colocalize":
                sanitized, changed = _sanitize_colocalize_arguments(active_user_message, raw_args)
                if changed:
                    exec_args = sanitized
                    trace_item["original_arguments"] = raw_args
                    trace_item["arguments"] = sanitized
                    trace_item["arg_sanitized"] = True

            # Dedup identical calls within a turn
            args_key = json.dumps(exec_args, sort_keys=True, default=str)
            cache_key = (exec_call_name, args_key)
            if cache_key in tool_cache:
                compact = tool_cache[cache_key]
                tool_results_compact.append(compact)
                trace_item["status"] = "cached"
                trace_item["result_preview"] = compact
                tool_trace.append(trace_item)
                continue

            try:
                tool = registry.get(exec_call_name)
                args = tool.args_model(**exec_args)
                result = tool.fn(args, ctx)

                if isinstance(result, dict):
                    for art_key in ("artifact", "plot", "data_artifact"):
                        if art_key in result:
                            artifacts.append(_normalize_artifact(result[art_key]))
                    if result.get("pycmap_code"):
                        pycmap_code_snippets.append(str(result["pycmap_code"]))

                    # Track confirmed table from successful data tool execution
                    if exec_call_name.startswith(("cmap.", "viz.")):
                        tbl = str((exec_args or {}).get("table") or "").strip()
                        if tbl:
                            thread_state.confirmed_table = tbl
                            confirmed_dataset_table = tbl
                            ctx["confirmed_table"] = tbl

                compact = _tool_result_for_llm(call.name, result if isinstance(result, dict) else {"result": result})
                tool_results_compact.append(compact)
                trace_item["result_preview"] = compact
                # Only cache non-empty catalog results
                _is_empty_cat = (
                    exec_call_name.startswith("catalog.")
                    and isinstance(result, dict)
                    and not result.get("results")
                    and not result.get("selected")
                )
                if not _is_empty_cat:
                    tool_cache[cache_key] = compact
                # Persist good catalog results to thread_state for cross-turn reuse
                if isinstance(result, dict) and exec_call_name.startswith("catalog."):
                    _cat_hits: list = []
                    for _k in ("results", "alternates"):
                        for _r in (result.get(_k) or []):
                            if isinstance(_r, dict) and _r.get("table"):
                                _cat_hits.append(_r)
                    _sel_r = result.get("selected")
                    if isinstance(_sel_r, dict) and _sel_r.get("table"):
                        _cat_hits.insert(0, _sel_r)
                    if _cat_hits and thread_state:
                        thread_state.last_catalog_results = _cat_hits[:20]

            except SystemExit as e:
                msg = str(e) or "SystemExit"
                trace_item["status"] = "error"
                trace_item["error"] = msg
                compact = {"tool": call.name, "error": {"code": "system_exit", "message": msg}}
                tool_results_compact.append(compact)
                trace_item["result_preview"] = compact
                tool_cache[cache_key] = compact

            except Exception as e:
                trace_item["status"] = "error"
                payload = None
                if hasattr(e, "to_dict"):
                    try:
                        payload = e.to_dict()
                    except Exception:
                        pass
                trace_item["error"] = payload or str(e)
                compact = {"tool": call.name, "error": payload or {"code": "exception", "message": str(e)}}
                tool_results_compact.append(compact)
                trace_item["result_preview"] = compact
                tool_cache[cache_key] = compact

            tool_trace.append(trace_item)
            remaining -= 1

        # Build a mandatory addendum when catalog tools returned non-empty results
        # AND the request is a data/map action (not a summarize/chat request).
        # This prevents the model from ignoring results and saying "no results found."
        catalog_found_addendum = ""
        _is_data_action = intent.action in ("map", "time_series", "download", "colocalize")
        if _is_data_action:
            for _compact in tool_results_compact:
                if not isinstance(_compact, dict):
                    continue
                _tool_name = str(_compact.get("tool") or "")
                if not _tool_name.startswith("catalog."):
                    continue
                _sel = _compact.get("selected")
                _results = _compact.get("results") or []
                if isinstance(_sel, dict) and _sel.get("table"):
                    _top = _sel
                elif _results:
                    _top = _results[0] if isinstance(_results[0], dict) else None
                else:
                    _top = None
                if _top and _top.get("table"):
                    _top_table = _top["table"]
                    _top_name = _top.get("name") or _top.get("title") or _top_table
                    catalog_found_addendum = (
                        f"\n\nIMPORTANT: The catalog search returned results. "
                        f"The top dataset is '{_top_name}' (`{_top_table}`). "
                        f"Do NOT say 'no results found'. "
                        f"Call catalog.list_variables with table='{_top_table}' to get "
                        "the variable name, then call viz.plot_map with the user's bounds and date."
                    )
                    break

        messages.append(LLMMessage(
            role="user",
            content=(
                "TOOL_RESULTS (compact):\n"
                + json.dumps(tool_results_compact, indent=2, default=str)
                + catalog_found_addendum
                + "\n\nRespond in JSON with EITHER:\n"
                "1) type='final' + fields: assistant_message, code (optional), artifacts (optional)\n"
                "OR\n"
                "2) type='tool_call' + tool_calls[] if more tools are needed.\n"
            ),
        ))
