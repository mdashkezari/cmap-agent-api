#!/usr/bin/env python3
"""verify_eval_targets.py — round-trip eval substrings against their PDFs.

Problem being addressed
-----------------------
``build_eval_queries.py`` extracts candidate ``any_of`` substrings from
the human-written answer prose.  Those substrings then drive the retrieval
hit-rate metric.  But an answer writer might paraphrase the paper, so a
substring that is distinctive in the answer may not appear verbatim in
the PDF text at all.  When that happens the harness scores the query as
a miss regardless of how well retrieval actually worked — the failure is
in the metric, not in retrieval.

This tool reads each eval entry's ``ref_file`` from the reference bank,
re-uses the same PDF text extractor as the KB sync pipeline
(``cmap_agent.sync.kb_sync._extract_text_from_file``, which applies the
same header filter and number-break fix that produce what Qdrant stores),
and tests each ``any_of`` substring against that text.  Substrings that
fail to appear are flagged.  The tool writes a report plus an optional
"pruned" queries JSON where failing substrings are removed and entries
that end up with no survivors are left with empty ``any_of`` (the harness
will score them as honest automatic misses, with a load-time warning).

The tool does NOT invent replacement substrings from the PDF.  Doing so
would reintroduce the hardcoded-to-content pattern we are trying to avoid.
Entries that lose all their substrings are a signal that the reference
answer paraphrases the paper and needs a human to pick a verbatim anchor,
or that the harness should score that entry differently.

Usage
-----
    python scripts/verify_eval_targets.py                    # report only
    python scripts/verify_eval_targets.py --write-pruned \\
           --out scripts/retrieval_eval_queries_pruned.json  # produce pruned
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

DEFAULT_QUERIES = _ROOT / "scripts" / "retrieval_eval_queries.json"


# Shared text normalizer used by both the verifier and the eval harness.
# See ``scripts/eval_text_match.py`` for the full rationale.
sys.path.insert(0, str(_ROOT / "scripts"))
from eval_text_match import normalize_for_match  # noqa: E402


# ---------------------------------------------------------------------------
# Reference bank resolution
# ---------------------------------------------------------------------------

def _bank_root() -> Path:
    from cmap_agent.sync.kb_sync import _default_bank_dir
    return _default_bank_dir()


def _resolve_ref_path(short_name: str, ref_file: str, bank: Path) -> Path | None:
    """Return the absolute path to *ref_file* under bank/<short_name>/.

    ``ref_file`` is the filename recorded in the prompt/answer pairs.  If
    that exact filename is missing, a loose match on the filename stem is
    attempted so that small naming drifts (extra spaces, etc.) don't
    disqualify an entry.
    """
    if not ref_file or not short_name:
        return None
    dataset_dir = bank / short_name
    if not dataset_dir.is_dir():
        return None

    exact = dataset_dir / ref_file
    if exact.is_file():
        return exact

    # Loose match on stem.
    target_stem = Path(ref_file).stem.lower()
    for f in dataset_dir.iterdir():
        if f.is_file() and f.stem.lower() == target_stem:
            return f
    return None


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _load_pdf_text(path: Path) -> str | None:
    """Delegate to the same extractor the KB sync uses."""
    from cmap_agent.sync.kb_sync import _extract_text_from_file
    return _extract_text_from_file(path)


def _verify_entry(entry: dict, bank: Path, text_cache: dict[str, str | None]) -> dict:
    """Return a result dict describing which substrings survived."""
    ref_file = entry.get("ref_file") or ""
    short_name = entry.get("short_name") or ""
    any_of = entry.get("any_of") or []

    path = _resolve_ref_path(short_name, ref_file, bank)
    if path is None:
        return {
            "id": entry["id"],
            "status": "ref_missing",
            "ref_file": ref_file,
            "short_name": short_name,
            "kept": [],
            "dropped": list(any_of),
        }

    key = str(path)
    if key not in text_cache:
        text_cache[key] = _load_pdf_text(path)
    text = text_cache[key]

    if not text:
        return {
            "id": entry["id"],
            "status": "ref_unreadable",
            "ref_file": ref_file,
            "short_name": short_name,
            "kept": [],
            "dropped": list(any_of),
        }

    lowered = normalize_for_match(text)
    kept: list[str] = []
    dropped: list[str] = []
    for s in any_of:
        if normalize_for_match(s) in lowered:
            kept.append(s)
        else:
            dropped.append(s)

    return {
        "id": entry["id"],
        "status": "ok",
        "ref_file": ref_file,
        "short_name": short_name,
        "kept": kept,
        "dropped": dropped,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERIES,
        help=f"Path to queries JSON (default: {DEFAULT_QUERIES}).",
    )
    ap.add_argument(
        "--write-pruned",
        action="store_true",
        help="Write a pruned queries file with non-matching substrings removed.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Destination for the pruned file (implies --write-pruned).",
    )
    args = ap.parse_args()

    raw = json.loads(args.queries.read_text())
    queries = raw.get("queries") or []
    if not queries:
        print(f"[warn] no queries in {args.queries}")
        return 0

    bank = _bank_root()
    if not bank.exists():
        print(f"[error] reference bank not found at {bank}")
        return 1
    print(f"[ok] reference bank: {bank}")
    print(f"[ok] queries: {len(queries)} entries from {args.queries}\n")

    text_cache: dict[str, str | None] = {}
    results = [_verify_entry(q, bank, text_cache) for q in queries]

    # Summary counts.
    status_counts: dict[str, int] = {}
    total_kept = 0
    total_dropped = 0
    entries_fully_dropped: list[str] = []
    entries_partially_dropped: list[dict] = []
    entries_ref_missing: list[str] = []
    entries_ref_unreadable: list[str] = []

    for r in results:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
        total_kept += len(r["kept"])
        total_dropped += len(r["dropped"])
        if r["status"] == "ref_missing":
            entries_ref_missing.append(r["id"])
        elif r["status"] == "ref_unreadable":
            entries_ref_unreadable.append(r["id"])
        elif r["status"] == "ok":
            if r["dropped"] and not r["kept"]:
                entries_fully_dropped.append(r["id"])
            elif r["dropped"]:
                entries_partially_dropped.append(r)

    print("Summary")
    print("-------")
    print(f"  entries checked:       {len(results)}")
    for k, v in sorted(status_counts.items()):
        print(f"    {k:<22} {v}")
    print(f"  substrings kept:       {total_kept}")
    print(f"  substrings dropped:    {total_dropped}")
    print(f"  fully dropped entries: {len(entries_fully_dropped)}")
    print(f"  partially dropped:     {len(entries_partially_dropped)}")

    if entries_ref_missing:
        print("\nRef file missing (entry will not score):")
        for pid in entries_ref_missing:
            print(f"  {pid}")
    if entries_ref_unreadable:
        print("\nRef file unreadable by PDF extractor:")
        for pid in entries_ref_unreadable:
            print(f"  {pid}")
    if entries_fully_dropped:
        print("\nFully dropped (every substring missed the PDF text):")
        for pid in entries_fully_dropped:
            # Find the first dropped substring for a hint of what was tried.
            r = next(rr for rr in results if rr["id"] == pid)
            sample = r["dropped"][:3]
            print(f"  {pid}   tried: {sample}")
    if entries_partially_dropped:
        print("\nPartially dropped (some substrings survive, some don't):")
        for r in entries_partially_dropped[:20]:
            print(f"  {r['id']}")
            for s in r["dropped"][:5]:
                print(f"    DROP  {s!r}")
            for s in r["kept"][:3]:
                print(f"    keep  {s!r}")

    # Write the pruned file if requested.
    if args.write_pruned or args.out:
        out_path = args.out or args.queries.with_name(
            args.queries.stem + "_pruned" + args.queries.suffix
        )
        pruned_queries = []
        r_by_id = {r["id"]: r for r in results}
        for q in queries:
            new_q = dict(q)
            r = r_by_id.get(q["id"])
            if r and r["status"] == "ok":
                new_q["any_of"] = r["kept"]
                new_q["verified"] = True
            else:
                # ref missing / unreadable — leave any_of as-is and mark
                # unverified so callers can see which entries weren't checked.
                new_q["verified"] = False
            pruned_queries.append(new_q)

        doc = dict(raw)
        doc["queries"] = pruned_queries
        out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
        print(f"\n[ok] wrote pruned file to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
