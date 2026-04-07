from __future__ import annotations

import json
import re
import os
from urllib.parse import urlsplit, urlunsplit
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from cmap_agent.llm.types import LLMMessage
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

    # Prior conversation (as plain chat messages)
    for m in conversation:
        role = m.get("role", "user")
        content = m.get("content") or ""
        if content.strip():
            messages.append(LLMMessage(role=role, content=content))

    # Current user turn
    messages.append(LLMMessage(role="user", content=user_message))

    tool_trace: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    pycmap_code_snippets: list[str] = []

    remaining = max(0, int(max_tool_calls))

    # If the model returns non-JSON, we'll ask it to retry a couple times.
    invalid_json_retries = 0
    # If the user request clearly needs tools (data/catalog/map), don't allow
    # a plan-only final answer.
    force_tool_retries = 0
    requires_tools = _request_requires_tools(user_message)

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
            exec_args: dict[str, Any] = raw_args
            trace_item: dict[str, Any] = {"tool": call.name, "status": "ok", "arguments": raw_args}

            # Guardrail: if the model invents tolerances that the user didn't specify,
            # strip them so the tool can infer from catalog metadata.
            if call.name == "cmap.colocalize":
                sanitized, changed = _sanitize_colocalize_arguments(user_message, raw_args)
                if changed:
                    exec_args = sanitized
                    trace_item["original_arguments"] = raw_args
                    trace_item["arguments"] = sanitized
                    trace_item["arg_sanitized"] = True

            # Deduplicate identical tool calls in a single run (models sometimes retry).
            args_key = json.dumps(exec_args, sort_keys=True, default=str)
            cache_key = (call.name, args_key)
            if cache_key in tool_cache:
                compact = tool_cache[cache_key]
                tool_results_compact.append(compact)
                trace_item["status"] = "cached"
                trace_item["result_preview"] = compact
                tool_trace.append(trace_item)
                # Do NOT decrement remaining; we didn't execute a tool.
                continue

            try:
                tool = registry.get(call.name)
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