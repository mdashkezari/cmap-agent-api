from __future__ import annotations
from typing import Any, Iterable

def format_kb_context(hits: list[dict[str, Any]]) -> str:
    # Provide compact context with stable IDs for citation.
    blocks=[]
    for h in hits:
        meta=h.get("metadata") or {}
        title = meta.get("title") or meta.get("dataset_name") or meta.get("name") or ""
        dtype = meta.get("doc_type","")
        src = meta.get("source","kb")
        blocks.append(
            f"[KB:{h.get('id')}] type={dtype} title={title} source={src}\n"
            f"{(h.get('text') or '')[:2000]}"
        )
    return "\n\n".join(blocks).strip()
