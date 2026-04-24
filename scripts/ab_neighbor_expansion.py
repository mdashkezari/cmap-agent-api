#!/usr/bin/env python3
"""ab_neighbor_expansion.py — measure hit-rate after neighbour expansion.

Non-destructive diagnostic, companion to ``ab_refbank_chunksize.py``.

What it tests
-------------
An idea: when a paper chunk ranks in the top few hits, also pull its
±1 index neighbours from the same paper into the result list.  The
hypothesis is that a chunk which ranks well often neighbours a chunk
containing the actual answer sentence, so fetching adjacent chunks may
rescue answer-bearing content into the window the LLM sees.

This script simulates that transformation without touching production
retrieval, and invokes the general evaluation harness
(``eval_retrieval.py``'s query set) to score it.  A positive result is
a hit-rate increase across the whole query set — not a single rescued
query.  A rescue on one query that does not move the whole-set rate is
noise, not signal.

Usage
-----
    python scripts/ab_neighbor_expansion.py

Requires whatever collections are listed in ``CONFIGURATIONS`` to exist
in Qdrant.  Use ``ab_refbank_chunksize.py`` first to populate them.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cmap_agent.rag.qdrant_kb import QdrantKB
from cmap_agent.config.settings import settings


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIGURATIONS: list[tuple[str, int]] = [
    ("cmap_kb_v1",       2000),
    ("cmap_kb_ab_4000",  4000),
    ("cmap_kb_ab_7000",  7000),
]

SCAN_K = 40       # initial scan depth from Qdrant
PROBE_K = 8       # only expand neighbours for hits in this prefix
NEIGHBOR_RADIUS = 1  # ±1 chunk index


# ---------------------------------------------------------------------------
# Chunk-ID parsing
# ---------------------------------------------------------------------------
#
# Reference-bank doc IDs follow two shapes:
#
#   refbank:<short>:<stem>:<hash>              (the first chunk of the doc)
#   refbank:<short>:<stem>:<hash>#chunkN       (subsequent chunks)
#
# Given any such ID the neighbour at offset ±d is obtained by transforming
# the `#chunkN` suffix:
#
#    N=1 is represented by the base ID with no suffix.
#    N=2,3,... are represented as "#chunkN".
#
# ``_neighbors`` returns the list of candidate neighbour IDs, skipping
# neighbours that would have index < 1 (no such chunk can exist).

_CHUNK_SUFFIX = re.compile(r"^(.*?)(?:#chunk(\d+))?$")


def _split_chunk_id(doc_id: str) -> tuple[str, int] | None:
    """Return (base_id, chunk_index) for a refbank doc_id, or None."""
    if not doc_id.startswith("refbank:"):
        return None
    m = _CHUNK_SUFFIX.match(doc_id)
    if not m:
        return None
    base, idx = m.group(1), m.group(2)
    return base, int(idx) if idx else 1


def _rebuild_chunk_id(base: str, idx: int) -> str:
    return base if idx == 1 else f"{base}#chunk{idx}"


def _neighbor_ids(doc_id: str, radius: int = NEIGHBOR_RADIUS) -> list[str]:
    parsed = _split_chunk_id(doc_id)
    if parsed is None:
        return []
    base, idx = parsed
    out: list[str] = []
    for d in range(1, radius + 1):
        for sign in (-1, 1):
            cand_idx = idx + sign * d
            if cand_idx < 1:
                continue
            out.append(_rebuild_chunk_id(base, cand_idx))
    return out


# ---------------------------------------------------------------------------
# Qdrant helpers
# ---------------------------------------------------------------------------

def _collection_exists(kb: QdrantKB) -> bool:
    try:
        kb.client.get_collection(kb.collection_name)
        return True
    except Exception:
        return False


def _fetch_by_doc_ids(kb: QdrantKB, doc_ids: list[str]) -> dict[str, dict]:
    """Fetch payloads for the given doc IDs via scroll + filter.

    Returns a mapping of doc_id → hit dict (with text and metadata).  Doc
    IDs that are not present in the collection are simply absent from the
    returned mapping.
    """
    out: dict[str, dict] = {}
    if not doc_ids:
        return out
    # Qdrant `MatchAny` expects an 'any' list.
    from qdrant_client.models import Filter, FieldCondition, MatchAny
    try:
        points, _ = kb.client.scroll(
            collection_name=kb.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="_doc_id",
                        match=MatchAny(any=doc_ids),
                    )
                ]
            ),
            limit=len(doc_ids) * 2,  # safety margin
            with_payload=True,
        )
    except Exception as exc:
        print(f"    [warn] scroll failed: {exc}")
        return out

    for pt in points:
        payload = pt.payload or {}
        did = payload.get("_doc_id", str(pt.id))
        out[did] = {
            "id": did,
            "text": payload.get("_text", ""),
            "metadata": {k: v for k, v in payload.items()
                         if k not in ("_text", "_doc_id")},
        }
    return out


# ---------------------------------------------------------------------------
# Expansion + rank
# ---------------------------------------------------------------------------

def _expand_with_neighbors(
    kb: "QdrantKB",
    hits: list[dict],
    probe_k: int = PROBE_K,
    radius: int = NEIGHBOR_RADIUS,
) -> list[dict]:
    """Return a new ordering with ±``radius`` neighbours interleaved.

    For each of the first ``probe_k`` hits, the neighbour doc IDs are
    computed by parsing the chunk-index suffix and ±``radius``-stepping
    around it, and all missing neighbours are fetched in a single scroll
    call.  Each fetched neighbour is inserted immediately after its
    anchor hit in the returned list.  Duplicates are deduplicated so a
    chunk that legitimately ranks high keeps its original slot and its
    neighbours do not appear twice.

    The expansion is capped at the top ``probe_k`` hits because the
    hypothesis is that the answer-bearing chunk sits next to a
    high-ranked chunk.  Expanding deeper just pollutes the window with
    adjacency noise from chunks that were never strong matches to
    begin with.
    """
    # Collect every neighbour ID we want to fetch, then fetch in one call.
    wanted: set[str] = set()
    for h in hits[:probe_k]:
        for nid in _neighbor_ids(h.get("id", ""), radius=radius):
            wanted.add(nid)
    # Exclude IDs that are already in the original hit list.
    already = {h.get("id") for h in hits}
    wanted -= already
    fetched = _fetch_by_doc_ids(kb, sorted(wanted))

    # Build the expanded ordering.  Each anchor hit is followed by its
    # available neighbours; later hits (outside the probe prefix) are
    # appended unchanged.
    seen: set[str] = set()
    out: list[dict] = []
    for i, h in enumerate(hits):
        hid = h.get("id", "")
        if hid in seen:
            continue
        seen.add(hid)
        out.append(h)
        if i >= probe_k:
            continue
        for nid in _neighbor_ids(hid, radius=radius):
            if nid in seen:
                continue
            if nid in fetched:
                out.append(fetched[nid])
                seen.add(nid)
    return out


def _best_rank(hits: list[dict], any_of: list[str]) -> int | None:
    """1-based rank of the first hit whose text contains any target needle.

    Uses the same unicode+whitespace normalizer as ``eval_retrieval.py``
    so that A/B scores are directly comparable to the harness baseline.
    Without the shared normalizer this script would occasionally score
    differently from the main eval for the same ``(collection, query,
    any_of)`` triple, which would be misleading.
    """
    from eval_text_match import normalize_for_match
    needles = [normalize_for_match(s) for s in any_of if s]
    needles = [n for n in needles if n]
    if not needles:
        return None
    for i, h in enumerate(hits, start=1):
        text = normalize_for_match(h.get("text") or "")
        if any(n in text for n in needles):
            return i
    return None


def _verdict(rank: int | None, top_k: int, scan_k: int) -> str:
    if rank is None:
        return f"miss@{scan_k}"
    marker = "+" if rank <= top_k else " "
    return f"[{marker}] rank {rank:>2}"


def main() -> int:
    from cmap_agent.config.settings import settings

    # Import the harness loader so queries come from the canonical JSON.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import eval_retrieval as ev

    print("=" * 70)
    print("Neighbour-expansion A/B (non-destructive)")
    print("=" * 70)
    print(f"Qdrant URL:      {settings.QDRANT_URL}")
    print(f"Scan depth:      {SCAN_K}")
    print(f"Probe prefix:    top {PROBE_K} hits are expanded")
    print(f"Radius:          \u00b1{NEIGHBOR_RADIUS} chunk index")

    queries = ev._load_queries(ev.DEFAULT_QUERIES)
    if not queries:
        print("No queries in retrieval_eval_queries.json; nothing to score.")
        return 0
    print(f"Queries:         {len(queries)}")

    top_k = 12

    for name, size in CONFIGURATIONS:
        kb = QdrantKB(collection=name)
        if not _collection_exists(kb):
            print(f"\n[{size:>4}] {name}: missing — skipped")
            continue

        print(f"\n--- {name} (chunk size {size}) ---")
        base_hits_in_top = 0
        exp_hits_in_top = 0
        base_misses = 0
        exp_misses = 0

        for q in queries:
            base = kb.query(q["question"], k=SCAN_K)
            expanded = _expand_with_neighbors(kb, base)

            r_base = _best_rank(base, q["any_of"])
            r_exp = _best_rank(expanded, q["any_of"])

            if r_base is None:
                base_misses += 1
            elif r_base <= top_k:
                base_hits_in_top += 1
            if r_exp is None:
                exp_misses += 1
            elif r_exp <= top_k:
                exp_hits_in_top += 1

            delta = ""
            if r_base is not None and r_exp is not None:
                # Delta = base - exp, so positive means the rank number
                # got smaller under expansion, i.e. the target moved UP in
                # the result list.  This matches the reader's intuition
                # that "positive delta = improvement".
                delta = f"  \u0394={r_base - r_exp:+d}"
            print(
                f"    {q['id']:<30} "
                f"base: {_verdict(r_base, top_k, SCAN_K):<14} "
                f"exp: {_verdict(r_exp, top_k, SCAN_K):<14}"
                f"{delta}"
            )

        n = len(queries)
        print(
            f"  summary: hit-rate@{top_k}  "
            f"base {base_hits_in_top}/{n}  ->  expanded {exp_hits_in_top}/{n}  "
            f"| misses base {base_misses} -> expanded {exp_misses}"
        )

    print()
    print("=" * 70)
    print("Interpretation:")
    print("  \u0394 positive = target moved UP in the result list (improvement)")
    print("  \u0394 negative = target moved DOWN (expansion pushed it deeper)")
    print("  Ship if expanded hit-rate beats base across the query set and")
    print("  no single query regresses meaningfully.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
