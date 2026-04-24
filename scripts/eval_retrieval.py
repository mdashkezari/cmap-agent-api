#!/usr/bin/env python3
"""eval_retrieval.py — general-purpose retrieval evaluation harness.

This script measures how well the current retrieval pipeline surfaces
answer-bearing chunks for a user-maintained set of natural-language
queries.  It is deliberately paper-agnostic: the queries live in
``retrieval_eval_queries.json`` and can be grown over time as real
user questions are added to the corpus of things the agent should
handle.

The metric is **hit rate at TOP_K**: the fraction of queries for which
at least one chunk within the top ``TOP_K`` retrieved hits contains any
of the query's target substrings.  A secondary metric is **mean rank**
of the best matching chunk (rank ``None`` for queries that miss entirely
is excluded from the mean, but the miss count is reported separately).

Results are broken down by question shape (``numeric_threshold``,
``sequence_literal``, etc.) so that regressions in one class are visible
even when the overall rate is unchanged.

Any retrieval intervention — chunk size, neighbour expansion, query
rewriting, sentence-level re-ranking — must be measured against this
harness to justify a production change.  Interventions that move the
hit rate up without regressing any shape class are good candidates.
Interventions that rescue one query but don't help the whole set are not.

Usage
-----
    # Run against the production collection (default).
    python scripts/eval_retrieval.py

    # Run against a different collection.
    python scripts/eval_retrieval.py --collection cmap_kb_ab_7000

    # Narrower or wider scan depth.
    python scripts/eval_retrieval.py --top-k 12 --scan-k 40

    # Path to a different queries file.
    python scripts/eval_retrieval.py --queries path/to/file.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


DEFAULT_QUERIES = _ROOT / "scripts" / "retrieval_eval_queries.json"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_queries(path: Path) -> list[dict]:
    """Parse the queries JSON and perform light validation."""
    if not path.exists():
        raise SystemExit(f"Queries file not found: {path}")
    raw = json.loads(path.read_text())
    queries = raw.get("queries") or []
    if not isinstance(queries, list):
        raise SystemExit(f"'queries' must be a list; got {type(queries).__name__}")

    seen_ids: set[str] = set()
    cleaned: list[dict] = []
    empty_ids: list[str] = []
    for i, q in enumerate(queries):
        if not isinstance(q, dict):
            raise SystemExit(f"Entry {i} is not an object")
        for key in ("id", "question", "any_of"):
            if key not in q:
                raise SystemExit(f"Entry {i} missing required key: {key}")
        qid = str(q["id"])
        if qid in seen_ids:
            raise SystemExit(f"Duplicate query id: {qid}")
        seen_ids.add(qid)
        any_of = q["any_of"]
        if not isinstance(any_of, list):
            raise SystemExit(f"Entry {qid!r} has non-list any_of")
        # Empty any_of is acceptable — the entry will score as an automatic
        # miss.  This is the right behaviour for answers that are too
        # prose-heavy for the substring extractor; rejecting them would
        # silently inflate the hit rate by dropping the hardest cases.
        if not any_of:
            empty_ids.append(qid)
        cleaned.append({
            "id": qid,
            "question": str(q["question"]),
            "any_of": [str(s) for s in any_of],
            "short_name": q.get("short_name", ""),
            "shape": q.get("shape", "unspecified"),
            "notes": q.get("notes", ""),
        })
    if empty_ids:
        print(
            f"[warn] {len(empty_ids)} entr{'y' if len(empty_ids)==1 else 'ies'} "
            f"with no candidate substrings (will score as automatic misses): "
            f"{', '.join(empty_ids)}"
        )
    return cleaned


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _best_rank(hits: list[dict], any_of: list[str]) -> int | None:
    """Return the 1-based rank of the first hit whose text contains any target.

    Text comparison uses ``eval_text_match.normalize_for_match`` on both
    sides to forgive PDF-extraction artefacts: non-breaking spaces, two
    different mu codepoints, en-dash vs hyphen, whitespace runs.  Without
    this, substrings that are conceptually present in the PDF (e.g.
    ``0.2 µm``) can fail to match because the chunk text renders them as
    ``0.2\u00a0\u03bcm`` or similar.
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


def _run(
    collection: str,
    queries: list[dict],
    top_k: int,
    scan_k: int,
    verbose: bool = False,
    backend: str = "qdrant",
) -> dict:
    """Execute all queries against *collection* and return a summary dict.

    ``backend`` selects the KB implementation: ``qdrant`` (default) or
    ``chroma``.  Both implementations expose a compatible ``query(text, k)``
    that returns ``list[{id, text, distance, metadata}]``.  A/B comparisons
    between backends use the same harness JSON so any difference in hit
    rate is attributable to the backend, not to measurement drift.
    """
    if backend == "qdrant":
        from cmap_agent.rag.qdrant_kb import QdrantKB
        kb = QdrantKB(collection=collection)
        try:
            kb.client.get_collection(collection)
        except Exception:
            raise SystemExit(
                f"Qdrant collection '{collection}' not found. "
                f"Rebuild with: cmap-agent-sync-kb --target qdrant --rebuild"
            )
    elif backend == "chroma":
        from cmap_agent.rag.chroma_kb import ChromaKB
        kb = ChromaKB(collection=collection)
        # ChromaKB raises on first query if the collection is missing;
        # probe with a tiny query so the error is clear and early.
        try:
            kb.query("ping", k=1)
        except Exception as exc:
            raise SystemExit(
                f"Chroma collection '{collection}' not queryable: {exc}\n"
                f"Rebuild with: cmap-agent-sync-kb --target chroma --rebuild"
            )
    else:
        raise SystemExit(f"Unknown backend: {backend!r} (use 'qdrant' or 'chroma')")

    per_query: list[dict] = []
    for q in queries:
        hits = kb.query(q["question"], k=scan_k)
        rank = _best_rank(hits, q["any_of"])
        in_top_k = rank is not None and rank <= top_k
        per_query.append({
            "id": q["id"],
            "shape": q["shape"],
            "short_name": q["short_name"],
            "rank": rank,
            "in_top_k": in_top_k,
        })
        if verbose:
            status = (
                f"rank {rank:>3}" if rank is not None else f"miss@{scan_k}"
            )
            marker = "+" if in_top_k else " "
            print(f"  [{marker}] {q['id']:<30} {q['shape']:<20} {status}")

    return {
        "collection": collection,
        "backend": backend,
        "top_k": top_k,
        "scan_k": scan_k,
        "total": len(per_query),
        "per_query": per_query,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _report(summary: dict) -> None:
    per = summary["per_query"]
    total = summary["total"]
    top_k = summary["top_k"]

    in_top = sum(1 for r in per if r["in_top_k"])
    misses = sum(1 for r in per if r["rank"] is None)
    ranks_found = [r["rank"] for r in per if r["rank"] is not None]

    print()
    print("=" * 70)
    print(f"Backend:    {summary.get('backend', 'qdrant')}")
    print(f"Collection: {summary['collection']}")
    print(f"Queries:    {total}")
    print(f"Scan depth: {summary['scan_k']}")
    print(f"Top-K:      {top_k}")
    print("=" * 70)

    if total == 0:
        print("No queries to report on.")
        return

    pct_top = 100.0 * in_top / total
    pct_miss = 100.0 * misses / total
    print(f"\nHit rate @ TOP {top_k}:   {in_top}/{total}  ({pct_top:.1f}%)")
    print(f"Complete misses:     {misses}/{total}  ({pct_miss:.1f}%)")
    if ranks_found:
        print(f"Mean rank (non-miss):  {statistics.mean(ranks_found):.1f}")
        print(f"Median rank:           {int(statistics.median(ranks_found))}")

    # Breakdown by shape.
    shapes: dict[str, list[dict]] = {}
    for r in per:
        shapes.setdefault(r["shape"], []).append(r)
    if len(shapes) > 1:
        print("\nBreakdown by question shape:")
        for shape, group in sorted(shapes.items()):
            gtop = sum(1 for g in group if g["in_top_k"])
            gmiss = sum(1 for g in group if g["rank"] is None)
            print(
                f"  {shape:<22} "
                f"top-{top_k}: {gtop}/{len(group)}  "
                f"miss: {gmiss}/{len(group)}"
            )

    # Breakdown by short_name if available.
    papers: dict[str, list[dict]] = {}
    for r in per:
        sn = r["short_name"] or "(unknown)"
        papers.setdefault(sn, []).append(r)
    if len(papers) > 1:
        print("\nBreakdown by source paper:")
        for sn, group in sorted(papers.items()):
            gtop = sum(1 for g in group if g["in_top_k"])
            gmiss = sum(1 for g in group if g["rank"] is None)
            print(
                f"  {sn:<22} "
                f"top-{top_k}: {gtop}/{len(group)}  "
                f"miss: {gmiss}/{len(group)}"
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    from cmap_agent.config.settings import settings

    ap = argparse.ArgumentParser(
        description="Measure retrieval hit rate against a query set."
    )
    ap.add_argument(
        "--backend",
        choices=("qdrant", "chroma"),
        default=settings.CMAP_AGENT_KB_BACKEND.lower().strip(),
        help="KB backend to query (default: whatever CMAP_AGENT_KB_BACKEND is set to).",
    )
    ap.add_argument(
        "--collection",
        default=settings.CMAP_AGENT_KB_COLLECTION,
        help="Collection name to query (default: production).",
    )
    ap.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERIES,
        help=f"Path to queries JSON (default: {DEFAULT_QUERIES}).",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Top-K threshold for the hit-rate metric (default: 12).",
    )
    ap.add_argument(
        "--scan-k",
        type=int,
        default=40,
        help="Scan depth per query (default: 40).",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Skip per-query output; show summary only.",
    )
    args = ap.parse_args()

    queries = _load_queries(args.queries)
    if not queries:
        print(f"No queries in {args.queries}. Nothing to do.")
        return 0

    print(f"Loaded {len(queries)} queries from {args.queries}")
    print(f"Backend: {args.backend}  Collection: {args.collection}")
    if not args.quiet:
        print("\nPer-query results:")
    summary = _run(
        backend=args.backend,
        collection=args.collection,
        queries=queries,
        top_k=args.top_k,
        scan_k=args.scan_k,
        verbose=not args.quiet,
    )
    _report(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
