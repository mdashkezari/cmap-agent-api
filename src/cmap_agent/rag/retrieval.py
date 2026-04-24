"""retrieval — backend-agnostic KB retrieval.

Provides ``retrieve_context()`` and ``get_kb()`` which select the configured
knowledge base backend (ChromaDB or Qdrant) based on the
``CMAP_AGENT_KB_BACKEND`` setting.
"""
from __future__ import annotations

import logging
from typing import Any, Protocol

from cmap_agent.config.settings import settings
from cmap_agent.rag.format import format_kb_context

log = logging.getLogger(__name__)


class KBBackend(Protocol):
    """Minimal interface shared by ChromaKB and QdrantKB."""

    def query(
        self,
        query_text: str,
        k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...

    def upsert(
        self,
        *,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None: ...

    def all_ids(self, batch_size: int = ...) -> list[str]: ...

    def delete_ids(self, ids: list[str]) -> None: ...


def get_kb(**overrides: Any) -> KBBackend:
    """Return a KB instance for the configured backend.

    Keyword arguments are forwarded to the backend constructor and can
    override connection parameters (e.g. ``url``, ``api_key``,
    ``persist_dir``, ``collection``).
    """
    backend = settings.CMAP_AGENT_KB_BACKEND.lower().strip()

    if backend == "qdrant":
        from cmap_agent.rag.qdrant_kb import QdrantKB
        return QdrantKB(
            url=overrides.get("url"),
            api_key=overrides.get("api_key"),
            collection=overrides.get("collection"),
        )

    # Default: chroma
    from cmap_agent.rag.chroma_kb import ChromaKB
    return ChromaKB(
        persist_dir=overrides.get("persist_dir"),
        collection=overrides.get("collection"),
    )


def retrieve_context(query: str, *, k: int = 8) -> tuple[str, list[dict]]:
    """Retrieve KB context for a user query.

    Returns (formatted_context_string, raw_hits).
    """
    kb = get_kb()
    hits = kb.query(query, k=k)
    return format_kb_context(hits), hits
