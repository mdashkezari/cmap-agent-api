#!/usr/bin/env python3
"""ab_refbank_chunksize.py — measure retrieval hit-rate vs refbank chunk size.

This script is NON-DESTRUCTIVE.  It does not modify the production
``cmap_kb_v1`` collection.  It creates two additional Qdrant collections
alongside it at different reference-bank chunk sizes, then invokes the
general retrieval evaluation harness (``eval_retrieval.py``) against
each, so hit-rate can be compared across chunk sizes on the same query
set.

    cmap_kb_v1       — production (unchanged; chunk size = current setting)
    cmap_kb_ab_4000  — reference bank re-chunked at 4000 chars
    cmap_kb_ab_7000  — reference bank re-chunked at 7000 chars

Only the reference bank is re-ingested for the A/B collections — the
catalog and variable docs are irrelevant for this question and skipping
them keeps the sync to around 10-15 minutes per collection.

The queries evaluated are whatever is in ``retrieval_eval_queries.json``
— this script has no hardcoded questions of its own.  To add or replace
queries, edit that file; the A/B measurement will reflect the change on
the next run.

Usage
-----
    python scripts/ab_refbank_chunksize.py

Requires:
    OPENAI_API_KEY (for dense embeddings)
    a running Qdrant at $QDRANT_URL or http://localhost:6333
    the reference bank present at notrack/reference_bank/
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cmap_agent.config.settings import settings
from cmap_agent.rag.qdrant_kb import QdrantKB
from cmap_agent.sync.kb_sync import (
    _ingest_reference_bank,
    _default_bank_dir,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# (collection_name, chunk_size_chars, human_label).
CONFIGURATIONS: list[tuple[str, int, str]] = [
    ("cmap_kb_v1",       2000, "production (current)"),
    ("cmap_kb_ab_4000",  4000, "A/B candidate"),
    ("cmap_kb_ab_7000",  7000, "A/B candidate"),
]


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def _collection_exists(kb: QdrantKB) -> bool:
    try:
        kb.client.get_collection(kb.collection_name)
        return True
    except Exception:
        return False


def _sync_ab_collection(name: str, chunk_size: int) -> None:
    """Sync the reference bank into *name* at *chunk_size* chars per chunk.

    Idempotent: if the collection already exists, no re-sync is performed.
    To force a re-sync, delete the collection via Qdrant first.
    """
    kb = QdrantKB(collection=name)
    if _collection_exists(kb):
        print(f"  [skip] {name} already exists")
        return

    print(f"  [sync] {name} @ {chunk_size} chars/chunk ...")
    kb.ensure_collection()

    # Temporarily override the per-call chunk-size setting so that
    # _ingest_reference_bank produces chunks at the target size.
    orig = settings.CMAP_AGENT_KB_REFBANK_CHUNK_SIZE
    settings.CMAP_AGENT_KB_REFBANK_CHUNK_SIZE = chunk_size
    try:
        ids: list[str] = []
        texts: list[str] = []
        metas: list[dict] = []
        # Empty dataset_rows / short_name_to_table are acceptable — the
        # retrieval A/B does not depend on table_name or dataset_id metadata.
        n = _ingest_reference_bank(
            _default_bank_dir(),
            {},                       # dataset_rows
            {},                       # short_name_to_table
            ids, texts, metas,
        )
        print(f"         {n} chunks prepared, upserting ...")
        kb.upsert(ids=ids, texts=texts, metadatas=metas)
        print(f"         done.")
    finally:
        settings.CMAP_AGENT_KB_REFBANK_CHUNK_SIZE = orig


# ---------------------------------------------------------------------------
# Query helpers — delegate to the general eval harness
# ---------------------------------------------------------------------------

def _run_queries_via_harness() -> None:
    """Import eval_retrieval and run it against each A/B collection.

    The A/B scripts do not duplicate the query runner — they only set up
    different index configurations and ask the harness to report on each.
    """
    print("\n" + "=" * 70)
    print("Hit-rate by chunk size (queries from retrieval_eval_queries.json)")
    print("=" * 70)

    # Import here so script load does not require OpenAI keys just to read
    # the schema.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import eval_retrieval as ev

    queries = ev._load_queries(ev.DEFAULT_QUERIES)
    if not queries:
        print("No queries defined in retrieval_eval_queries.json; nothing to score.")
        return
    print(f"Loaded {len(queries)} queries.\n")

    for name, size, label in CONFIGURATIONS:
        kb = QdrantKB(collection=name)
        if not _collection_exists(kb):
            print(f"[{size:>4}] {name}: missing — skipped")
            continue
        print(f"\n--- {name} (chunk size {size}, {label}) ---")
        summary = ev._run(
            collection=name,
            queries=queries,
            top_k=12,
            scan_k=40,
            verbose=True,
        )
        ev._report(summary)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("A/B refbank chunk-size test (non-destructive)")
    print("=" * 70)
    print(f"Qdrant URL:  {settings.QDRANT_URL}")
    print(f"Bank dir:    {_default_bank_dir()}")
    print(f"Scan depth:  {SCAN_K} hits per query")
    print()

    print("[setup]")
    for name, size, label in CONFIGURATIONS:
        if name == "cmap_kb_v1":
            # Production collection — never touch it.
            kb = QdrantKB(collection=name)
            if _collection_exists(kb):
                print(f"  [ok]   {name} (production, left unchanged)")
            else:
                print(f"  [warn] {name} not found — production queries will be skipped")
            continue
        _sync_ab_collection(name, size)

    _run_queries_via_harness()

    print()
    print("=" * 70)
    print("Done.")
    print("To re-run a single chunk size from scratch, first delete its")
    print("collection:  curl -X DELETE $QDRANT_URL/collections/cmap_kb_ab_4000")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
