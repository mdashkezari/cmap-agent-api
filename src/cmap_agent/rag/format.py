"""format — render KB hits into the prompt context string.

Hits arrive in RRF-fused rank order from the KB backend.  The renderer
surfaces that rank to the LLM via ``rank=N`` in the block header so that
the downstream prompt rule about preferring higher-ranked evidence has
a concrete signal to act on.

Per-doc-type text caps
----------------------
Reference-bank chunks are already chunked at 2000 chars during KB sync, so
a 2000-char cap costs nothing.  Catalog and variable docs are chunked at
7000 chars, so a 2000-char cap here would silently discard 5000 chars of
retrieved context — bad for dataset-description or variable-listing
questions.  The cap is therefore keyed off ``doc_type``, and an unknown
``doc_type`` falls back to the conservative 2000-char default.
"""
from __future__ import annotations
from typing import Any

# doc_type → max chars per hit when rendering into the prompt.
# Keep in sync with settings.CMAP_AGENT_KB_REFBANK_CHUNK_SIZE and
# settings.CMAP_AGENT_KB_CATALOG_CHUNK_SIZE.
_CAP_BY_DOCTYPE: dict[str, int] = {
    "paper_chunk":        2000,
    "dataset":            7000,
    "variable":           7000,
    "dataset_reference":  7000,
}
_DEFAULT_CAP = 2000


def _cap_for(meta: dict[str, Any]) -> int:
    return _CAP_BY_DOCTYPE.get(str(meta.get("doc_type", "")), _DEFAULT_CAP)


def format_kb_context(hits: list[dict[str, Any]]) -> str:
    """Render KB hits as prompt blocks with stable IDs and explicit rank.

    Each block is rendered as:

        [KB:<id> rank=N] type=<doc_type> title=<title> source=<source>
        <text>

    where N is the 1-based position in the input list (which is the
    RRF-fused rank returned by the KB backend) and the text cap depends on
    the hit's ``doc_type`` (see ``_CAP_BY_DOCTYPE``).
    """
    blocks = []
    for idx, h in enumerate(hits, start=1):
        meta = h.get("metadata") or {}
        title = meta.get("title") or meta.get("dataset_name") or meta.get("name") or ""
        dtype = meta.get("doc_type", "")
        src = meta.get("source", "kb")
        text = (h.get("text") or "")[: _cap_for(meta)]
        blocks.append(
            f"[KB:{h.get('id')} rank={idx}] type={dtype} title={title} source={src}\n"
            f"{text}"
        )
    return "\n\n".join(blocks).strip()
