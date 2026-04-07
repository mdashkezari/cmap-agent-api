from __future__ import annotations
from pydantic import BaseModel, Field
from cmap_agent.rag.chroma_kb import ChromaKB

class KBSearchArgs(BaseModel):
    query: str = Field(..., description="Semantic search query")
    limit: int = Field(8, ge=1, le=25, description="Number of KB chunks to return")

def kb_search(args: KBSearchArgs, ctx: dict) -> dict:
    kb = ChromaKB()
    hits = kb.query(args.query, k=args.limit)
    # Return compact results
    results=[]
    for h in hits:
        meta=h.get("metadata") or {}
        results.append({
            "id": h.get("id"),
            "doc_type": meta.get("doc_type"),
            "title": meta.get("title") or meta.get("dataset_name") or meta.get("name"),
            "table": meta.get("table"),
            "dataset_id": meta.get("dataset_id"),
            "distance": h.get("distance"),
        })
    return {"results": results}
