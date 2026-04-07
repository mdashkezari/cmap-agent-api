from __future__ import annotations
from cmap_agent.rag.chroma_kb import ChromaKB
from cmap_agent.rag.format import format_kb_context

def retrieve_context(query: str, *, k: int = 8) -> tuple[str, list[dict]]:
    kb = ChromaKB()
    hits = kb.query(query, k=k)
    return format_kb_context(hits), hits
