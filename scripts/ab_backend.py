#!/usr/bin/env python3
"""ab_backend.py — A/B retrieval comparison: Chroma vs Qdrant.

Runs the retrieval-evaluation harness against each backend twice and
reports the hit rates side-by-side.  Two runs per backend captures the
variance that HNSW random-walk and float non-determinism introduce at
score ties — a single run can be lucky.

This script does NOT rebuild the KBs.  Before running it, both backends
must be populated with the SAME source data at the SAME chunk sizes.
The reproducible recipe is:

    # 1. Replace reference bank if needed (cleanup, paywall removal, etc.)
    # 2. Rebuild BOTH backends with current settings:
    cmap-agent-sync-kb --target chroma --rebuild
    cmap-agent-sync-kb --target qdrant --rebuild

    # 3. (Optional) verify substrings against the new reference bank:
    python scripts/verify_eval_targets.py

    # 4. Run the A/B:
    python scripts/ab_backend.py

Only with identical inputs on both sides is any hit-rate difference
attributable to the backend.

Decision rules (based on net harness hit-rate delta averaged across runs)
-----------------------------------------------------------------------
  Qdrant wins by >= 3 pp        ->  Migration justified.  Cloud cost OK.
  Qdrant wins by 1-2 pp         ->  Marginal.  Weigh against Qdrant Cloud cost.
  Within +/- 1 pp               ->  Effectively tied.  Revert to Chroma
                                    (simpler, cheaper, no measured loss).
  Chroma wins                   ->  Revert immediately.

Shape-level breakdown matters too: if Qdrant wins overall but loses
badly on sequence_literal, that hybrid-search-for-literal-tokens
trade-off is worth knowing even when the aggregate is net-positive.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT / "scripts"))


def _run_once(backend: str, collection: str, queries: list[dict],
              top_k: int, scan_k: int) -> dict:
    """Invoke the harness once and return its summary dict."""
    import eval_retrieval as ev
    return ev._run(
        backend=backend,
        collection=collection,
        queries=queries,
        top_k=top_k,
        scan_k=scan_k,
        verbose=False,
    )


def _by_shape(per_query: list[dict], top_k: int) -> dict[str, tuple[int, int]]:
    """Return {shape: (hits_in_top_k, total)}."""
    shapes: dict[str, list[dict]] = {}
    for r in per_query:
        shapes.setdefault(r["shape"], []).append(r)
    return {
        s: (sum(1 for r in rows if r["in_top_k"]), len(rows))
        for s, rows in shapes.items()
    }


def _summarise(runs: list[dict]) -> dict:
    """Aggregate a list of run summary dicts into a single report."""
    totals = [len(r["per_query"]) for r in runs]
    hits_per_run = [sum(1 for q in r["per_query"] if q["in_top_k"]) for r in runs]
    miss_per_run = [sum(1 for q in r["per_query"] if q["rank"] is None) for r in runs]
    return {
        "backend": runs[0]["backend"],
        "collection": runs[0]["collection"],
        "top_k": runs[0]["top_k"],
        "scan_k": runs[0]["scan_k"],
        "runs": len(runs),
        "total_queries": totals[0],
        "hits_per_run": hits_per_run,
        "hits_mean": statistics.mean(hits_per_run),
        "hits_stdev": statistics.stdev(hits_per_run) if len(hits_per_run) > 1 else 0.0,
        "misses_per_run": miss_per_run,
        "per_query_runs": [r["per_query"] for r in runs],
    }


def _print_side_by_side(summ_a: dict, summ_b: dict) -> None:
    """Print a two-column comparison."""
    ba, bb = summ_a["backend"], summ_b["backend"]
    n = summ_a["total_queries"]
    top_k = summ_a["top_k"]

    hra = summ_a["hits_mean"] / n * 100.0
    hrb = summ_b["hits_mean"] / n * 100.0

    print()
    print("=" * 72)
    print(f"Backend A/B  (TOP_K={top_k}, {summ_a['runs']} runs per backend)")
    print("=" * 72)
    header = f"{'':<26} {ba:<22} {bb:<22}"
    print(header)
    print("-" * 72)
    print(
        f"{'Queries scored':<26} {n:<22} {n:<22}"
    )
    print(
        f"{'Hits / run':<26} "
        f"{str(summ_a['hits_per_run']):<22} "
        f"{str(summ_b['hits_per_run']):<22}"
    )
    print(
        f"{'Mean hits':<26} "
        f"{summ_a['hits_mean']:<22.1f} "
        f"{summ_b['hits_mean']:<22.1f}"
    )
    print(
        f"{'Mean hit rate':<26} "
        f"{hra:<21.1f}% {hrb:<21.1f}%"
    )
    print(
        f"{'Stdev (hits, across runs)':<26} "
        f"{summ_a['hits_stdev']:<22.2f} "
        f"{summ_b['hits_stdev']:<22.2f}"
    )
    print(
        f"{'Complete misses / run':<26} "
        f"{str(summ_a['misses_per_run']):<22} "
        f"{str(summ_b['misses_per_run']):<22}"
    )

    # Delta.  The label is a practical-effect heuristic, not a statistical
    # significance test.  With disagreement counts this low (typically <10),
    # an exact McNemar test on our observed deltas gives p-values well above
    # conventional thresholds.  The project uses these labels for the
    # practical question "is the effect big enough to act on?" — for the
    # statistical question, a McNemar calculation on the per-query
    # disagreements (printed below) is the right tool.
    delta_pp = hrb - hra
    print()
    print(f"Delta (B - A):  {delta_pp:+.1f} pp")
    if abs(delta_pp) < 1.0:
        verdict = "practically tied"
    elif 1.0 <= abs(delta_pp) < 3.0:
        verdict = "small practical effect"
    else:
        verdict = "meaningful practical effect"
    print(f"Practical signal: {verdict}  (not a statistical significance test; see note below)")

    # Shape breakdown using run 0 of each backend (runs should be very similar).
    sa = _by_shape(summ_a["per_query_runs"][0], top_k)
    sb = _by_shape(summ_b["per_query_runs"][0], top_k)
    all_shapes = sorted(set(sa) | set(sb))
    print()
    print(f"Shape breakdown (run 0 of each backend, TOP_K={top_k})")
    print(f"{'shape':<22} {ba:<15} {bb:<15}")
    print("-" * 72)
    for shape in all_shapes:
        ah, at = sa.get(shape, (0, 0))
        bh, bt = sb.get(shape, (0, 0))
        print(f"  {shape:<20} {ah}/{at:<13} {bh}/{bt:<13}")

    # Per-query disagreements (run 0 of each).
    a0 = {q["id"]: q for q in summ_a["per_query_runs"][0]}
    b0 = {q["id"]: q for q in summ_b["per_query_runs"][0]}
    disagree_a_only = []   # A hits, B misses
    disagree_b_only = []   # B hits, A misses
    for qid in sorted(set(a0) & set(b0)):
        aq = a0[qid]
        bq = b0[qid]
        if aq["in_top_k"] and not bq["in_top_k"]:
            disagree_a_only.append((qid, aq["rank"], bq["rank"], aq["shape"]))
        elif bq["in_top_k"] and not aq["in_top_k"]:
            disagree_b_only.append((qid, aq["rank"], bq["rank"], aq["shape"]))
    if disagree_a_only:
        print()
        print(f"Queries where {ba} HITS but {bb} MISSES:")
        for qid, ar, br, sh in disagree_a_only:
            print(f"  {qid:<14} {ba} rank={ar}  {bb} rank={br}  shape={sh}")
    if disagree_b_only:
        print()
        print(f"Queries where {bb} HITS but {ba} MISSES:")
        for qid, ar, br, sh in disagree_b_only:
            print(f"  {qid:<14} {ba} rank={ar}  {bb} rank={br}  shape={sh}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--queries",
        type=Path,
        default=_ROOT / "scripts" / "retrieval_eval_queries.json",
    )
    ap.add_argument(
        "--collection",
        default=None,
        help="Collection name (both backends must use the same name). "
             "Defaults to CMAP_AGENT_KB_COLLECTION.",
    )
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--scan-k", type=int, default=40)
    ap.add_argument(
        "--runs",
        type=int,
        default=2,
        help="Runs per backend (default: 2). Averages reduce HNSW / tie-break noise.",
    )
    args = ap.parse_args()

    from cmap_agent.config.settings import settings
    import eval_retrieval as ev

    collection = args.collection or settings.CMAP_AGENT_KB_COLLECTION
    queries = ev._load_queries(args.queries)
    if not queries:
        print(f"No queries in {args.queries}.")
        return 0
    print(f"[setup] {len(queries)} queries, TOP_K={args.top_k}, "
          f"runs per backend={args.runs}, collection={collection!r}")

    chroma_runs = []
    print("\n=== Chroma ===")
    for i in range(args.runs):
        print(f"  run {i+1}/{args.runs} ...")
        chroma_runs.append(
            _run_once("chroma", collection, queries, args.top_k, args.scan_k)
        )
    summ_chroma = _summarise(chroma_runs)

    qdrant_runs = []
    print("\n=== Qdrant ===")
    for i in range(args.runs):
        print(f"  run {i+1}/{args.runs} ...")
        qdrant_runs.append(
            _run_once("qdrant", collection, queries, args.top_k, args.scan_k)
        )
    summ_qdrant = _summarise(qdrant_runs)

    _print_side_by_side(summ_chroma, summ_qdrant)

    print()
    print("=" * 72)
    print("Practical-effect decision rules (see header comment for full table):")
    print("  |delta| < 1 pp   -> practically tied, revert to Chroma")
    print("                      (simpler, cheaper; no measured gain)")
    print("  1-2 pp          -> small practical effect, weigh against Qdrant")
    print("                      Cloud monthly cost")
    print("  >= 3 pp         -> meaningful practical effect, migration justified")
    print()
    print("These labels are heuristic, not a statistical significance test.")
    print("A McNemar test on the per-query disagreements above is the right tool")
    print("for the statistical question; this script is for the practical one.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
