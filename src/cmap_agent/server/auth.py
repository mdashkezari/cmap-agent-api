from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Optional

from fastapi import HTTPException, Request

from cmap_agent.config.settings import settings
from cmap_agent.storage.sqlserver import SQLServerStore


_IDENT_RE = re.compile(r"^[A-Za-z0-9_\.]+$")


def _norm(s: object | None) -> str:
    """Normalize env-provided strings (strip whitespace + quotes)."""

    if s is None:
        return ""
    return str(s).strip().strip('"').strip("'").strip()


def resolve_auth_mode() -> str:
    """Resolve effective auth mode.

    settings.CMAP_AGENT_AUTH_MODE supports:
      - off
      - apikey
      - jwt (reserved)
      - apikey_or_jwt (reserved)
      - auto: enable apikey when running on ECS, otherwise off
    """

    mode = _norm(getattr(settings, "CMAP_AGENT_AUTH_MODE", "auto")).lower()
    if mode == "auto":
        # ECS injects one of these; prefer metadata URI.
        if os.getenv("ECS_CONTAINER_METADATA_URI_V4") or os.getenv("AWS_EXECUTION_ENV", "").startswith("AWS_ECS"):
            return "apikey"
        return "off"
    return mode or "off"


def _validate_ident(name: str, kind: str) -> str:
    if not name or not _IDENT_RE.match(name):
        raise RuntimeError(f"Invalid SQL identifier for {kind}: {name!r}")
    return name


@dataclass
class ApiKeyAuthResult:
    user_id: int
    api_key: str


class ApiKeyAuthenticator:
    """Validate X-API-Key against tblAPI_keys and cache lookups."""

    def __init__(self, store: SQLServerStore):
        self.store = store
        self.cache_ttl = int(getattr(settings, "CMAP_AGENT_AUTH_CACHE_TTL_SECONDS", 600) or 600)
        self._cache: dict[str, tuple[int, float]] = {}  # api_key -> (user_id, expires_epoch)

        self.table = _validate_ident(_norm(settings.CMAP_AGENT_AUTH_APIKEY_TABLE), "apikey table")
        self.key_col = _validate_ident(_norm(settings.CMAP_AGENT_AUTH_APIKEY_COLUMN), "apikey column")
        self.user_col = _validate_ident(_norm(settings.CMAP_AGENT_AUTH_USERID_COLUMN), "userid column")

    def _cache_get(self, api_key: str) -> Optional[int]:
        v = self._cache.get(api_key)
        if not v:
            return None
        user_id, exp = v
        if time.time() >= exp:
            self._cache.pop(api_key, None)
            return None
        return int(user_id)

    def _cache_set(self, api_key: str, user_id: int) -> None:
        self._cache[api_key] = (int(user_id), time.time() + self.cache_ttl)

    def resolve_user_id(self, api_key: str) -> Optional[int]:
        api_key = _norm(api_key)
        if not api_key:
            return None
        cached = self._cache_get(api_key)
        if cached is not None:
            return cached

        user_id = self.store.resolve_user_id_by_api_key(
            api_key=api_key,
            table=self.table,
            api_key_column=self.key_col,
            user_id_column=self.user_col,
        )
        if user_id is None:
            return None
        self._cache_set(api_key, int(user_id))
        return int(user_id)


def is_public_path(path: str, protect_docs: bool) -> bool:
    """Return True if request path is public (no auth required)."""

    if path in {"/health", "/version"}:
        return True
    if not protect_docs and (path.startswith("/docs") or path.startswith("/redoc") or path == "/openapi.json"):
        return True
    # Local artifacts are public when mounted (only in local backend).
    if path.startswith("/artifacts"):
        return True
    return False


def require_api_key(request: Request, authenticator: ApiKeyAuthenticator) -> ApiKeyAuthResult:
    header = _norm(getattr(settings, "CMAP_AGENT_AUTH_APIKEY_HEADER", "X-API-Key")) or "X-API-Key"
    api_key = request.headers.get(header)
    api_key = _norm(api_key)
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key")

    user_id = authenticator.resolve_user_id(api_key)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return ApiKeyAuthResult(user_id=int(user_id), api_key=api_key)
