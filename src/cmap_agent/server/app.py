from __future__ import annotations

import logging
import os
import json
import traceback
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security.api_key import APIKeyHeader
from starlette.responses import JSONResponse

from cmap_agent.agent.state import ThreadState
from cmap_agent.config.settings import settings
from cmap_agent.utils import to_jsonable as _to_jsonable_shared
from cmap_agent.storage.sqlserver import SQLServerStore
from cmap_agent.tools.default_registry import build_default_registry
from cmap_agent.agent.context import build_system_prompt
from cmap_agent.agent.runner import execute_plan, AgentFinal

log = logging.getLogger(__name__)
from cmap_agent.llm.openai_client import OpenAIClient
from cmap_agent.server.models import (
    ChatRequest,
    ChatResponse,
    ThreadItem,
    ThreadListResponse,
    ThreadMessageItem,
    ThreadMessagesResponse,
)
from cmap_agent.rag.retrieval import retrieve_context
from cmap_agent.server.auth import (
    ApiKeyAuthenticator,
    is_public_path,
    require_api_key,
    resolve_auth_mode,
)
from cmap_agent.server.rate_limit import FixedWindowRateLimiter

app = FastAPI(title="CMAP Agent", version="0.2.0")

# --- OpenAPI auth support (Swagger "Authorize") ---
#
# The runtime auth enforcement is handled by the middleware below.
# We additionally declare an API key security scheme so the Swagger UI
# can send the `X-API-Key` header for interactive testing.
_api_key_header_name = str(getattr(settings, "CMAP_AGENT_AUTH_APIKEY_HEADER", "X-API-Key") or "X-API-Key")
_api_key_scheme = APIKeyHeader(name=_api_key_header_name, auto_error=False)


def _derive_thread_title(user_message: str, max_len: int = 80) -> str:
    """Cheap deterministic title: first line, trimmed and capped."""
    if not user_message:
        return "New chat"
    s = " ".join(user_message.strip().split())
    # Remove wrapping quotes (common when users paste dataset names).
    if len(s) >= 2 and ((s[0] == s[-1]) and s[0] in ('"', "'")):
        s = s[1:-1].strip()
    if not s:
        return "New chat"
    if len(s) <= max_len:
        return s
    return s[: max_len - 1].rstrip() + "…"


def _derive_thread_summary(user_message: str, assistant_message: str, max_len: int = 280) -> str:
    """A lightweight rolling summary for UI listing (no LLM calls)."""
    parts: list[str] = []
    if user_message:
        parts.append("User: " + " ".join(user_message.strip().split()))
    if assistant_message:
        parts.append("Assistant: " + " ".join(assistant_message.strip().split()))
    s = " | ".join(parts) if parts else ""
    if len(s) <= max_len:
        return s
    return s[: max_len - 1].rstrip() + "…"


def _build_version_payload() -> dict[str, str]:
    """Return build + runtime metadata for deployment verification."""

    commit = (
        os.getenv("CMAP_AGENT_GIT_SHA")
        or os.getenv("GIT_SHA")
        or os.getenv("COMMIT_SHA")
        or "unknown"
    )
    build_time = (
        os.getenv("CMAP_AGENT_BUILD_TIME")
        or os.getenv("BUILD_TIME")
        or os.getenv("BUILD_TIMESTAMP")
        or "unknown"
    )

    # Runtime environment info (useful for debugging container vs local drift)
    import sys as _sys
    import platform as _platform

    def _v(mod_name: str) -> str:
        try:
            mod = __import__(mod_name)
            return str(getattr(mod, "__version__", "unknown"))
        except Exception:
            return "unavailable"

    payload: dict[str, str] = {
        "commit": str(commit),
        "build_time": str(build_time),
        "python": str(_sys.version.split()[0]),
        "platform": str(_platform.platform()),
        "pandas": _v("pandas"),
        "pyarrow": _v("pyarrow"),
        "pycmap": _v("pycmap"),
        "cartopy": _v("cartopy"),
        "pyproj": _v("pyproj"),
        "kb_collection": str(os.getenv("CMAP_AGENT_KB_COLLECTION", "")),
    }
    return payload



# _to_jsonable is imported from cmap_agent.utils as _to_jsonable_shared
_to_jsonable = _to_jsonable_shared



_backend_raw = getattr(settings, "CMAP_AGENT_ARTIFACT_BACKEND", "local")
_backend = str(_backend_raw or "local").strip().strip('"').strip("'").strip().lower()

# Serve artifacts (parquet/csv/png) for local dev usage.
# In production we prefer S3 + pre-signed URLs, so /artifacts is intentionally not mounted.
if _backend == "local":
    os.makedirs(settings.CMAP_AGENT_ARTIFACT_DIR, exist_ok=True)
    app.mount("/artifacts", StaticFiles(directory=settings.CMAP_AGENT_ARTIFACT_DIR), name="artifacts")


def _sanitize_public(obj: Any) -> Any:
    """Remove internal-only fields (local paths, *_local_path) from response payloads."""

    backend = (_backend or "local").lower().strip()

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            ks = str(k)
            # always drop internal helpers
            if ks.startswith('_') or ks.endswith('_local_path'):
                continue
            # drop local filesystem paths when backend is S3
            if backend == 's3' and ks in {'path', 'local_path', 'fpath'}:
                continue
            out[ks] = _sanitize_public(v)
        return out
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_public(v) for v in obj]
    return obj

# Serve artifacts (parquet/csv/png) for local dev usage
# In production we prefer S3 + pre-signed URLs, so /artifacts may be unused.

_store: SQLServerStore | None = None
_authenticator: ApiKeyAuthenticator | None = None
_rate_limiter: FixedWindowRateLimiter | None = None

def get_store() -> SQLServerStore:
    global _store
    if _store is None:
        _store = SQLServerStore.from_env()
    return _store


def get_authenticator() -> ApiKeyAuthenticator:
    global _authenticator
    if _authenticator is None:
        _authenticator = ApiKeyAuthenticator(get_store())
    return _authenticator


def get_rate_limiter() -> FixedWindowRateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        limit = int(getattr(settings, "CMAP_AGENT_RATE_LIMIT_RPM", 60) or 60)
        window = int(getattr(settings, "CMAP_AGENT_RATE_LIMIT_WINDOW_SECONDS", 60) or 60)
        _rate_limiter = FixedWindowRateLimiter(limit=limit, window_seconds=window)
    return _rate_limiter


@app.middleware("http")
async def auth_and_rate_limit(request: Request, call_next):
    """Authenticate and apply basic per-user rate limiting.

    - Public paths remain unauthenticated: /health, /version, and docs by default.
    - If auth mode is enabled, require X-API-Key and resolve User_ID via SQL.
    - If rate limiting is enabled, limit requests per minute per User_ID.

    Note: In-process rate limiting is consistent only within a single running task.
    If scale to multiple ECS tasks, move the limiter to a shared store (Redis).
    """

    mode = resolve_auth_mode()
    protect_docs = bool(getattr(settings, "CMAP_AGENT_AUTH_PROTECT_DOCS", False))
    if is_public_path(request.url.path, protect_docs=protect_docs):
        return await call_next(request)

    # Auth
    if mode in {"apikey", "apikey_or_jwt", "auto"}:
        try:
            res = require_api_key(request, get_authenticator())
            request.state.user_id = res.user_id
            request.state.cmap_api_key = res.api_key
        except HTTPException as e:
            return JSONResponse(status_code=e.status_code, content={"detail": e.detail})
    elif mode in {"off", "none", ""}:
        pass
    else:
        # Unknown mode (or jwt-only, not yet implemented)
        return JSONResponse(status_code=500, content={"detail": f"Unsupported auth mode: {mode}"})

    # Rate limit (per user if available)
    if bool(getattr(settings, "CMAP_AGENT_RATE_LIMIT_ENABLED", True)):
        uid = getattr(request.state, "user_id", None)
        if uid is not None:
            decision = get_rate_limiter().check(str(uid))
            if not decision.allowed:
                headers = {}
                if decision.retry_after_seconds:
                    headers["Retry-After"] = str(decision.retry_after_seconds)
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded"},
                    headers=headers,
                )

    return await call_next(request)

def make_llm(provider: str, model: str):
    provider = (provider or "openai").lower().strip()
    if provider == "openai":
        return OpenAIClient(model=model)
    if provider == "anthropic":
        # Lazy import so Anthropic isn't required unless used
        from cmap_agent.llm.anthropic_client import AnthropicClient
        return AnthropicClient(model=model)
    raise ValueError(f"Unsupported provider: {provider}")

# --- CORS ---
# Enable browser-based frontends (e.g., https://simonscmap.ai) to call this API.
# Configure via settings.CMAP_AGENT_CORS_* env vars.
#
# IMPORTANT: CORS middleware must be added *after* the auth/rate-limit middleware
# (declared via @app.middleware("http")) so that browser preflight (OPTIONS) requests
# are handled before any API-key enforcement.
def _split_csv(value: str) -> list[str]:
    items: list[str] = []
    for raw in (value or "").split(","):
        v = (raw or "").strip()
        if not v:
            continue
        # Be tolerant to accidental quoting in env vars (common in ECS console).
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
            v = v[1:-1].strip()
        v = v.strip().strip('"').strip("'").strip()
        if v:
            items.append(v)
    return items


if bool(getattr(settings, "CMAP_AGENT_CORS_ENABLED", True)):
    origins = _split_csv(getattr(settings, "CMAP_AGENT_CORS_ALLOW_ORIGINS", ""))
    methods = _split_csv(getattr(settings, "CMAP_AGENT_CORS_ALLOW_METHODS", "GET,POST,OPTIONS"))
    headers_raw = str(getattr(settings, "CMAP_AGENT_CORS_ALLOW_HEADERS", "*") or "*").strip()
    headers = ["*"] if headers_raw == "*" else _split_csv(headers_raw)
    allow_credentials = bool(getattr(settings, "CMAP_AGENT_CORS_ALLOW_CREDENTIALS", False))
    max_age = int(getattr(settings, "CMAP_AGENT_CORS_MAX_AGE_SECONDS", 600) or 600)

    # If no origins were provided, keep CORS effectively disabled.
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=allow_credentials,
            allow_methods=methods or ["GET", "POST", "OPTIONS"],
            allow_headers=headers or ["*"],
            max_age=max_age,
        )

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/version")
def version():
    """Return build metadata (commit + build time) for deployment verification."""
    return _build_version_payload()



@app.post("/files/presign_upload")
def presign_upload(
    request: Request,
    body: dict,
    # When auth is OFF (local/dev), allow explicitly selecting the user_id.
    user_id: int = 0,
    _api_key: str | None = Security(_api_key_scheme),
):
    """
    Request a presigned S3 URL for uploading an input file directly from the browser.

    Typical frontend flow:
      1) POST /files/presign_upload  -> receive {upload_url, artifact}
      2) PUT file bytes to upload_url (from the browser)
      3) POST /chat and reference the returned artifact as a tool input (e.g., cmap.colocalize source_artifact)

    Notes:
      - This endpoint requires CMAP_AGENT_ARTIFACT_BACKEND='s3'.
      - Bucket CORS must allow the frontend origin for PUT/OPTIONS.
    """
    from cmap_agent.config.settings import settings
    import uuid as _uuid
    import os as _os
    import re as _re

    if str(settings.CMAP_AGENT_ARTIFACT_BACKEND).lower() != "s3":
        raise HTTPException(status_code=400, detail="presign_upload requires CMAP_AGENT_ARTIFACT_BACKEND=s3")

    bucket = settings.CMAP_AGENT_ARTIFACT_S3_BUCKET
    if not bucket:
        raise HTTPException(status_code=500, detail="CMAP_AGENT_ARTIFACT_S3_BUCKET is not set")

    # Resolve user_id under auth
    mode = resolve_auth_mode()
    if mode in {"apikey", "apikey_or_jwt", "auto"}:
        uid = int(getattr(request.state, "user_id", 0) or 0)
    else:
        uid = int(user_id or 0)

    filename = str((body or {}).get("filename") or "").strip()
    content_type = str((body or {}).get("content_type") or "").strip() or None
    size_bytes = (body or {}).get("size_bytes", None)
    try:
        size_bytes = int(size_bytes) if size_bytes is not None else None
    except Exception:
        size_bytes = None
    thread_hint = str((body or {}).get("thread_id") or "").strip() or None

    if not filename:
        raise HTTPException(status_code=422, detail="filename is required")

    # Basic sanitize (keep base name only)
    filename = _os.path.basename(filename)
    filename = _re.sub(r"[^A-Za-z0-9._-]+", "_", filename)

    allowed_exts = [x.strip().lower() for x in str(settings.CMAP_AGENT_UPLOAD_ALLOWED_EXTS or "csv,parquet").split(",") if x.strip()]
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if allowed_exts and ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"Unsupported file type .{ext}. Allowed: {', '.join(allowed_exts)}")

    max_mb = int(getattr(settings, "CMAP_AGENT_UPLOAD_MAX_MB", 200) or 200)
    if size_bytes is not None and size_bytes > max_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large. Max is {max_mb} MB")

    # Build key under prefix/uploads/...
    prefix = (settings.CMAP_AGENT_ARTIFACT_S3_PREFIX or "artifacts").strip("/")
    upload_id = _uuid.uuid4().hex
    if thread_hint:
        key = f"{prefix}/{thread_hint}/uploads/u{uid}/{upload_id}/{filename}"
    else:
        key = f"{prefix}/uploads/u{uid}/{upload_id}/{filename}"

    # Presign PUT
    import boto3  # type: ignore
    s3 = boto3.client("s3", region_name=(settings.CMAP_AGENT_ARTIFACT_S3_REGION or None))
    params = {"Bucket": bucket, "Key": key}
    # If we sign content-type, the browser MUST send the same header on PUT.
    if content_type:
        params["ContentType"] = content_type

    ttl = int(getattr(settings, "CMAP_AGENT_UPLOAD_PRESIGN_TTL_SECONDS", 3600) or 3600)
    upload_url = s3.generate_presigned_url("put_object", Params=params, ExpiresIn=ttl)

    artifact = {
        "type": "file",
        "filename": filename,
        "backend": "s3",
        "content_type": content_type,
        "s3_bucket": bucket,
        "s3_key": key,
        "s3_uri": f"s3://{bucket}/{key}",
        # No download URL yet; downloads are produced as needed via existing artifact publishing.
    }

    return {
        "status": "ok",
        "upload": {"method": "PUT", "url": upload_url, "headers": ({"Content-Type": content_type} if content_type else {})},
        "artifact": artifact,
        "limits": {"max_mb": max_mb},
    }


@app.get("/threads", response_model=ThreadListResponse)
def list_threads(
    request: Request,
    limit: int = 50,
    offset: int = 0,
    # When auth is OFF (local/dev), allow explicitly selecting the user_id.
    user_id: int = 0,
    _api_key: str | None = Security(_api_key_scheme),
):
    store = get_store()

    auth_user_id = getattr(request.state, "user_id", None)
    if auth_user_id is not None:
        uid = int(auth_user_id)
    else:
        # In auth-OFF mode, the caller can specify user_id for convenience.
        uid = int(user_id or 0)

    rows = store.list_threads(user_id=uid, limit=limit, offset=offset)

    # Backfill missing titles lazily so existing Threads rows become usable.
    items: list[ThreadItem] = []
    for r in rows:
        title = (r.get("Title") or "").strip()
        if not title:
            title = _derive_thread_title(str(r.get("FirstUserMessage") or ""))
            try:
                store.set_thread_title(thread_id=str(r.get("ThreadId")), user_id=uid, title=title)
            except Exception:
                # Title backfill should not break listing.
                pass

        items.append(
            ThreadItem(
                thread_id=str(r.get("ThreadId")),
                title=title,
                created_at=r.get("CreatedAt"),
                updated_at=r.get("UpdatedAt"),
                last_role=r.get("LastRole"),
                last_message=r.get("LastMessage"),
                summary=r.get("LatestSummary"),
            )
        )

    return ThreadListResponse(threads=items)


@app.get("/threads/{thread_id}/messages", response_model=ThreadMessagesResponse)
def get_thread_messages(
    thread_id: str,
    request: Request,
    limit: int = 200,
    offset: int = 0,
    user_id: int = 0,
    _api_key: str | None = Security(_api_key_scheme),
):
    store = get_store()

    auth_user_id = getattr(request.state, "user_id", None)
    if auth_user_id is not None:
        uid = int(auth_user_id)
    else:
        uid = int(user_id or 0)

    rows = store.list_thread_messages(thread_id=thread_id, user_id=uid, limit=limit, offset=offset)
    messages = [
        ThreadMessageItem(
            message_id=int(r.get("MessageId")),
            role=str(r.get("Role") or ""),
            content=str(r.get("Content") or ""),
            created_at=r.get("CreatedAt"),
        )
        for r in rows
    ]

    title = store.get_thread_title(thread_id=thread_id, user_id=uid) or ""
    title = title.strip() or "New chat"

    return ThreadMessagesResponse(thread_id=thread_id, title=title, messages=messages)


@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    request: Request,
    _api_key: str | None = Security(_api_key_scheme),
):
    try:
        store = get_store()
        reg = build_default_registry()
        sys_prompt = build_system_prompt(reg)

        auth_user_id = getattr(request.state, "user_id", None)
        auth_api_key = getattr(request.state, "cmap_api_key", None)
        if auth_user_id is not None:
            # If client sent a user_id, it must match the authenticated key (unless 0/default).
            if req.user_id not in (0, None) and int(req.user_id) != int(auth_user_id):
                raise HTTPException(status_code=403, detail="user_id does not match API key")
            user_id = int(auth_user_id)
            cmap_api_key = str(auth_api_key) if auth_api_key else None
        else:
            user_id = int(req.user_id or 0)
            cmap_api_key = store.load_cmap_api_key(user_id)

        # Thread + persistence
        # Ensure Threads.Title is populated (used by the frontend chat list).
        if req.thread_id:
            thread_id = str(req.thread_id)
            existing_title = store.get_thread_title(thread_id=thread_id, user_id=user_id)
            if not (existing_title or "").strip():
                store.set_thread_title(
                    thread_id=thread_id,
                    user_id=user_id,
                    title=_derive_thread_title(req.message),
                )
        else:
            thread_id = store.create_thread(
                user_id=user_id,
                title=_derive_thread_title(req.message),
            )
        user_msg_id = store.add_message(thread_id=thread_id, user_id=user_id, role="user", content=req.message)

        # Conversation history (keep it compact)
        msgs = store.get_recent_messages(thread_id=thread_id, limit=30)
        conversation = []
        for m in msgs:
            role = (m.get("Role") or m.get("role") or "").lower()
            content = m.get("Content") or m.get("content") or ""
            if role in {"system","user","assistant"} and content:
                # exclude the just-added user message to avoid duplication
                if role == "user" and content == req.message:
                    continue
                conversation.append({"role": role, "content": content})

        # Optional rolling summary (future: update summaries in background / periodically)
        summary = store.get_latest_summary(thread_id=thread_id) or ""
        if summary.strip():
            sys_prompt = sys_prompt + "\n\nConversation summary (memory):\n" + summary.strip()

        # RAG retrieval
        rag_context = ""
        try:
            rag_context, _hits = retrieve_context(req.message, k=settings.CMAP_AGENT_KB_TOP_K)
        except Exception:
            rag_context = ""

        if rag_context.strip():
            sys_prompt = (
                sys_prompt
                + "\n\nRAG context (from CMAP knowledge base). Use this as primary context; cite with [KB:<id>].\n"
                + rag_context
            )

        # Tool-call context
        ctx = {
            "thread_id": thread_id,
            "user_id": user_id,
            "cmap_api_key": cmap_api_key,
            "store": store,
        }

        llm = make_llm(req.llm.provider, req.llm.model)
        # IMPORTANT: max_tool_calls can be 0 to explicitly disable tool calling.
        # Using `or` would treat 0 as falsy and accidentally fall back to the default.
        max_calls = (
            settings.CMAP_AGENT_MAX_TOOL_CALLS
            if req.options.max_tool_calls is None
            else req.options.max_tool_calls
        )

        # Load persistent thread state (NULL column treated as blank state)
        thread_state = ThreadState.from_json(store.get_thread_state(thread_id))
        ctx["thread_state"] = thread_state

        result = execute_plan(
            llm=llm,
            registry=reg,
            system_prompt=sys_prompt,
            conversation=conversation,
            user_message=req.message,
            ctx=ctx,
            max_tool_calls=max_calls,
        )

        # execute_plan returns (AgentFinal, tool_trace, ThreadState).
        # Guard defensively so a future signature change never silently
        # produces a 500 with a confusing attribute error.
        if not isinstance(result, tuple) or len(result) != 3:
            raise RuntimeError(
                f"execute_plan returned unexpected type: {type(result)}"
            )
        final, tool_trace, updated_state = result

        if not isinstance(final, AgentFinal):
            log.error(
                "execute_plan final is not AgentFinal but %s: %r",
                type(final), final,
            )
            raise RuntimeError(
                f"execute_plan returned non-AgentFinal result: {type(final)}"
            )

        # Persist updated state (best effort — never fails the response)
        try:
            store.set_thread_state(thread_id, updated_state.to_json())
        except Exception:
            pass

        # Persist assistant message
        _assistant_msg_id = store.add_message(
            thread_id=thread_id,
            user_id=user_id,
            role="assistant",
            content=final.assistant_message,
            model_provider=req.llm.provider,
            model_name=req.llm.model,
        )

        # Keep ThreadSummaries populated for UI listing / quick previews.
        # This is intentionally deterministic (no additional LLM calls).
        try:
            store.add_summary(
                thread_id=thread_id,
                user_id=user_id,
                summary_type="rolling",
                summary_text=_derive_thread_summary(req.message, final.assistant_message),
                summary_json={
                    "last_user": req.message[:4000],
                    "last_assistant": final.assistant_message[:4000],
                },
            )
        except Exception:
            pass

        # Persist tool trace (best effort).
        # NOTE: ToolRuns is row-based (1 row per tool call). We store compact previews.
        try:
            for t in tool_trace or []:
                if not isinstance(t, dict):
                    continue
                tool_name = str(t.get("tool") or "")
                if not tool_name:
                    continue
                tool_args_json = json.dumps(_to_jsonable(_sanitize_public(t.get("arguments") or {})), default=str)
                tool_result_preview = json.dumps(_to_jsonable(_sanitize_public(t.get("result_preview"))), default=str)
                status = str(t.get("status") or "ok")
                err = t.get("error")
                store.add_tool_run(
                    thread_id=thread_id,
                    user_id=user_id,
                    message_id=user_msg_id,
                    tool_name=tool_name,
                    tool_args_json=tool_args_json,
                    tool_result_json=None,
                    tool_result_preview=tool_result_preview,
                    status=status,
                    error_message=str(err) if err else None,
                )
        except Exception:
            pass

        return ChatResponse(
            thread_id=thread_id,
            assistant_message=final.assistant_message,
            code=final.code if req.options.return_code else None,
            artifacts=_to_jsonable(_sanitize_public(final.artifacts)),
            tool_trace=_to_jsonable(_sanitize_public(tool_trace)),
        )
    except SystemExit as e:
        # Defensive: sys.exit() should never take down the API worker.
        raise HTTPException(status_code=500, detail=f"SystemExit: {str(e) or 'SystemExit'}")
    except HTTPException:
        raise
    except Exception as e:
        log.error("Unhandled exception in /chat:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
