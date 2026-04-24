"""Catalog-search evaluation harness (v219+).

This is the catalog-search analog of ``scripts/retrieval_eval_queries.json`` /
``scripts/eval_retrieval.py``.  The retrieval harness measures reference-bank
retrieval quality.  This harness measures *dataset-discovery* quality —
whether a natural-language user query like "satellite POC over the south
pacific" surfaces the right CMAP dataset table at the top of
``catalog.search_kb_first``.

Rationale
---------
The reviewer's v216 assessment flagged ``catalog_tools.py`` as
architecturally fragile and required a measurement harness as the
precondition for any refactor (finding 3.4).  Without a harness,
changes to ranking logic can improve one query while silently
regressing several others.  This harness exists so any future change
to ``_post_rank_catalog_results``, ``_deterministic_resolve_candidates``,
or ``_bare_query`` can be validated against a whole-set baseline, not
by anecdote.

Entries
-------
Each entry in ``scripts/catalog_eval/queries.json`` is of the form::

  {
    "id": "CS001",
    "query": "satellite chlorophyll over the North Atlantic",
    "expected_any": [
      "tblModis_CHL", "tblModis_CHL_cl1", "tblModis_CHL_NRT",
      "tblCHL_REP"
    ],
    "must_not_top": [
      "tblPisces_Forecast", "tblPisces_Forecast_cl1"
    ],
    "shape": "satellite_variable_with_region",
    "notes": "Free-text notes on why this query class matters"
  }

Scoring
-------
For each query the harness calls ``catalog_search_kb_first`` and reports:

- **rank_of_expected**: 1-based rank of the first ``expected_any`` table in
  the merged result set, or ``None`` if none is present.
- **top1_ok**: True if rank_of_expected == 1.
- **top3_ok**: True if rank_of_expected <= 3.
- **mustnot_violation**: True if any ``must_not_top`` table appears at rank 1.

Summary metrics
---------------

- top1 hit rate across all entries
- top3 hit rate
- must-not-top violation count (more serious than a top3 miss — means the
  discovery suggested a genuinely wrong dataset)

Usage
-----

::

    python scripts/eval_catalog_search.py
    python scripts/eval_catalog_search.py --queries scripts/catalog_eval/queries.json

The harness requires a live SQL Server connection (catalog cache) and a
running Qdrant (KB backend).  Environment setup is the same as the main
agent.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any


_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


DEFAULT_QUERIES = _ROOT / "scripts" / "catalog_eval" / "queries.json"


def _load_queries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Queries file not found: {path}")
    data = json.loads(path.read_text())
    entries = data.get("queries") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        raise ValueError(f"Expected a list of queries in {path}")
    return entries


def _score_entry(
    entry: dict[str, Any],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score a single entry against its result list."""
    expected = [t.strip() for t in entry.get("expected_any") or [] if t.strip()]
    must_not_top = [t.strip() for t in entry.get("must_not_top") or [] if t.strip()]

    # Normalize the table-name field — different result sources use slightly
    # different keys.  Try common variants.
    def _table(r: dict[str, Any]) -> str:
        for k in ("table", "table_name", "Table_Name"):
            v = r.get(k)
            if v:
                return str(v).strip()
        return ""

    tables = [_table(r) for r in results]
    rank_of_expected: int | None = None
    for i, t in enumerate(tables, start=1):
        if t in expected:
            rank_of_expected = i
            break

    top1_table = tables[0] if tables else ""
    mustnot_violation = top1_table in must_not_top

    return {
        "id": entry.get("id"),
        "query": entry.get("query"),
        "shape": entry.get("shape") or "",
        "rank_of_expected": rank_of_expected,
        "top1_ok": rank_of_expected == 1,
        "top3_ok": rank_of_expected is not None and rank_of_expected <= 3,
        "top10_ok": rank_of_expected is not None and rank_of_expected <= 10,
        "top1_table": top1_table,
        "mustnot_violation": mustnot_violation,
        "total_results": len(tables),
    }


def _run_query(query: str, limit: int, ctx: dict) -> list[dict[str, Any]]:
    """Deprecated: kept for backwards compatibility.  See _run_query_on_path."""
    return _run_query_on_path(query, limit, ctx, path="kb_first")


def _run_query_on_path(
    query: str,
    limit: int,
    ctx: dict,
    path: str,
) -> list[dict[str, Any]]:
    """Run one query against one of the two catalog entry points.

    ``path="kb_first"``: calls ``catalog_search_kb_first`` (dense KB
    retrieval + post-ranking).  This is the path the agent's
    deterministic resolver (``_deterministic_resolve_candidates``) uses
    and the path the agent's system prompt often routes to.

    ``path="plain"``: calls ``catalog_search`` (SQL-LIKE path + dedup
    ranking).  This is what the agent's LLM tool-calling layer hits
    when it invokes ``catalog.search`` directly.

    Running both paths matters because they have different ranking
    internals.  A fix that only lands in one path can still leave the
    production agent broken.  v221 learned this the hard way: the
    variable-availability gate in ``_post_rank_catalog_results`` made
    the harness look fixed on ``kb_first`` while end-to-end tests
    showed the same failure continued on ``plain``.
    """
    if path == "kb_first":
        from cmap_agent.tools.catalog_tools import (
            catalog_search_kb_first,
            CatalogSearchKBFArgs,
        )
        args = CatalogSearchKBFArgs(
            query=query,
            lat1=None, lat2=None, lon1=None, lon2=None,
            dt1=None, dt2=None,
            make=None, sensor=None,
            limit=limit,
        )
        res = catalog_search_kb_first(args, ctx)
    elif path == "plain":
        from cmap_agent.tools.catalog_tools import (
            catalog_search,
            CatalogSearchArgs,
        )
        args = CatalogSearchArgs(query=query, limit=limit)
        res = catalog_search(args, ctx)
    else:
        raise ValueError(f"Unknown path: {path!r}")
    if not isinstance(res, dict):
        return []
    out = res.get("results")
    return list(out) if isinstance(out, list) else []


def _build_ctx() -> dict[str, Any]:
    """Wire up the same store + ctx the agent uses.

    ``SQLServerStore.from_env()`` is the constructor the FastAPI server
    uses — it reads the same CMAP_SQLSERVER_* env vars and creates an
    engine pointed at the same database.  Using it here means the harness
    queries exactly the catalog the running agent sees.
    """
    from cmap_agent.storage.sqlserver import SQLServerStore
    store = SQLServerStore.from_env()
    return {"store": store}


def _report(scored: list[dict[str, Any]], top_k_report: int) -> None:
    total = len(scored)
    if total == 0:
        print("No queries scored.")
        return

    top1 = sum(1 for s in scored if s["top1_ok"])
    top3 = sum(1 for s in scored if s["top3_ok"])
    top10 = sum(1 for s in scored if s["top10_ok"])
    mnv = sum(1 for s in scored if s["mustnot_violation"])

    print()
    print("=" * 72)
    print(f"Catalog-search eval  ({total} queries)")
    print("=" * 72)
    print(f"  Top-1 hit:          {top1}/{total}  ({100*top1/total:.1f}%)")
    print(f"  Top-3 hit:          {top3}/{total}  ({100*top3/total:.1f}%)")
    print(f"  Top-10 hit:         {top10}/{total}  ({100*top10/total:.1f}%)")
    print(f"  Must-not-top hits:  {mnv}/{total}  "
          f"(wrong-dataset suggestions)")

    ranks = [s["rank_of_expected"] for s in scored
             if s["rank_of_expected"] is not None]
    if ranks:
        print(f"  Mean rank (found):  {statistics.mean(ranks):.2f}")
        print(f"  Median rank:        {int(statistics.median(ranks))}")

    # Shape breakdown
    shapes: dict[str, list[dict[str, Any]]] = {}
    for s in scored:
        shapes.setdefault(s["shape"] or "(unshaped)", []).append(s)
    if len(shapes) > 1:
        print()
        print("Shape breakdown:")
        for sh, group in sorted(shapes.items()):
            g1 = sum(1 for g in group if g["top1_ok"])
            g3 = sum(1 for g in group if g["top3_ok"])
            gmn = sum(1 for g in group if g["mustnot_violation"])
            print(f"  {sh:<34} top1 {g1}/{len(group)}  "
                  f"top3 {g3}/{len(group)}  mustnot {gmn}/{len(group)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    ap.add_argument("--limit", type=int, default=10,
                    help="Limit passed to the catalog tools")
    ap.add_argument("--top-k-report", type=int, default=3,
                    help="Report cutoff used in verbose per-query display")
    ap.add_argument("--quiet", action="store_true",
                    help="Skip per-query lines")
    ap.add_argument("--path", choices=("both", "kb_first", "plain"),
                    default="both",
                    help="Which catalog entry point(s) to measure. "
                         "'both' (default) runs each query twice and "
                         "reports side-by-side.  Use 'kb_first' or "
                         "'plain' to isolate one path.")
    args = ap.parse_args()

    entries = _load_queries(args.queries)
    print(f"Loaded {len(entries)} queries from {args.queries}")
    print(f"Path: {args.path}")

    ctx = _build_ctx()

    paths = ("kb_first", "plain") if args.path == "both" else (args.path,)
    scored_by_path: dict[str, list[dict[str, Any]]] = {}
    for p in paths:
        scored: list[dict[str, Any]] = []
        if not args.quiet:
            print()
            print(f"Per-query results  (path: {p}):")
        for e in entries:
            q = e.get("query") or ""
            results = _run_query_on_path(q, args.limit, ctx, path=p)
            sc = _score_entry(e, results)
            scored.append(sc)
            if not args.quiet:
                rank_str = (
                    f"rank {sc['rank_of_expected']}"
                    if sc["rank_of_expected"] is not None
                    else "miss"
                )
                viol = "  [MUSTNOT VIOLATION]" if sc["mustnot_violation"] else ""
                marker = (
                    "+" if sc["top1_ok"] else ("~" if sc["top3_ok"] else " ")
                )
                print(
                    f"  [{marker}] {sc['id']:<8} {rank_str:<10} "
                    f"top1={sc['top1_table']!r}{viol}"
                )
        scored_by_path[p] = scored

    # Report: one summary per path, then a combined side-by-side table
    # when both were run.
    for p, scored in scored_by_path.items():
        print()
        print(f">>> Path: {p}")
        _report(scored, args.top_k_report)

    if len(scored_by_path) > 1:
        print()
        print("=" * 72)
        print("Side-by-side (per-query) — rank on each path")
        print("=" * 72)
        # Index each path's results by id
        by_id: dict[str, dict[str, dict[str, Any]]] = {}
        for p, scored in scored_by_path.items():
            for sc in scored:
                by_id.setdefault(sc["id"], {})[p] = sc
        for qid in sorted(by_id.keys()):
            row = by_id[qid]
            kb = row.get("kb_first") or {}
            pl = row.get("plain") or {}
            kb_r = kb.get("rank_of_expected")
            pl_r = pl.get("rank_of_expected")
            kb_s = str(kb_r) if kb_r is not None else "miss"
            pl_s = str(pl_r) if pl_r is not None else "miss"
            diverged = kb_r != pl_r
            mark = " <-- diverged" if diverged else ""
            print(f"  {qid:<8}  kb_first:{kb_s:<6}  plain:{pl_s:<6}{mark}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
