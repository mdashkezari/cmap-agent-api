"""diagnose_retrieval.py — inspect what the KB retrieves for a given query.

Prints the top-K hits for a natural-language query against the
currently configured KB backend (Qdrant by default, controlled by
``CMAP_AGENT_KB_BACKEND``).  Useful for comparing cloud vs local
retrieval behaviour, debugging why a particular answer is wrong,
and confirming the expected chunk is in the top-K.

This formalises the ad-hoc ``python3 -c`` snippets that were used
during the v225 Phase C cutover to diagnose the GRUMP AMMBI
archaeal-primers answer and the "5000" sequencing-depth threshold.

Usage
-----

    # Basic — show top-32 hits for a query (matches production top-K)
    python scripts/diagnose_retrieval.py "minimum sequencing depth"

    # Mark any hit whose text contains one or more substrings
    python scripts/diagnose_retrieval.py \\
        "archaeal primers AMMBI GRUMP" \\
        --mark "A2F" --mark "Arch21f" --mark "515Y"

    # More hits, longer previews
    python scripts/diagnose_retrieval.py "query text" -k 20 --preview 200

    # Force a specific backend (ignores CMAP_AGENT_KB_BACKEND)
    python scripts/diagnose_retrieval.py "query" --backend qdrant

Exit codes
----------
0 — ran successfully (regardless of retrieval quality)
1 — import or query error
2 — usage error (empty query)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _get_kb(backend: str | None):
    """Load the KB using the project's factory.  Honors
    CMAP_AGENT_KB_BACKEND env var; --backend overrides it."""
    if backend:
        os.environ["CMAP_AGENT_KB_BACKEND"] = backend
    from cmap_agent.rag.retrieval import get_kb
    return get_kb()


def _short(text: str, n: int) -> str:
    """Single-line preview up to ``n`` chars."""
    t = str(text or "").replace("\n", " ")
    if len(t) <= n:
        return t
    return t[:n] + "..."


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("query", help="Natural-language query to retrieve for")
    ap.add_argument("-k", "--top-k", type=int, default=32,
                    help="Number of hits to fetch (default: 32, matches "
                         "production CMAP_AGENT_KB_TOP_K)")
    ap.add_argument("--preview", type=int, default=120,
                    help="Max chars of each chunk to print (default: 120)")
    ap.add_argument("--mark", action="append", default=[],
                    help="Mark hits whose text contains this substring. "
                         "Repeatable: --mark A2F --mark 515Y")
    ap.add_argument("--backend", choices=["chroma", "qdrant"], default=None,
                    help="Override KB backend (default: from CMAP_AGENT_KB_BACKEND)")
    ap.add_argument("--show-ids", action="store_true",
                    help="Also print each hit's doc ID")
    args = ap.parse_args()

    if not args.query.strip():
        print("ERROR: empty query", file=sys.stderr)
        return 2

    try:
        kb = _get_kb(args.backend)
    except Exception as e:
        print(f"ERROR: failed to load KB backend: {e}", file=sys.stderr)
        return 1

    backend_name = os.environ.get("CMAP_AGENT_KB_BACKEND", "qdrant")
    print(f"Query:   {args.query!r}")
    print(f"Backend: {backend_name}  top_k={args.top_k}")
    if args.mark:
        print(f"Marks:   {args.mark}")
    print()

    try:
        hits = kb.query(args.query, k=args.top_k)
    except Exception as e:
        print(f"ERROR: query failed: {e}", file=sys.stderr)
        return 1

    if not hits:
        print("(no hits returned)")
        return 0

    # Compute max mark-label width for aligned output
    mark_w = max((len(m) for m in args.mark), default=0)

    for i, h in enumerate(hits, start=1):
        text = str(h.get("text") or "")
        marks_found = [m for m in args.mark if m in text]
        mark_col = ""
        if args.mark:
            if marks_found:
                mark_col = f"[{','.join(marks_found):{mark_w + 3}}]"
            else:
                mark_col = " " * (mark_w + 5)

        doc_id = ""
        if args.show_ids:
            did = str(h.get("id") or "")
            if not did:
                meta = h.get("metadata") or {}
                did = str(meta.get("_doc_id") or "")
            doc_id = f"  id={did[:60]}"

        print(f"#{i:>2} {mark_col} {_short(text, args.preview)}{doc_id}")

    if args.mark:
        print()
        for m in args.mark:
            ranks = [i for i, h in enumerate(hits, start=1)
                     if m in str(h.get("text") or "")]
            if ranks:
                print(f"  {m!r} appears at rank(s): {ranks}")
            else:
                print(f"  {m!r} NOT in top-{args.top_k}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
