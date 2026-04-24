#!/usr/bin/env python3
"""test_pdf_number_fix.py — unit tests for v202/v203 KB-sync text fixes.

Covers:
  1. ``_fix_pdf_number_breaks`` — merges line-break-garbled numbers
     and, critically, leaves genuine numeric ranges untouched.
  2. ``format_kb_context``     — renders hits with ``rank=N`` annotations
     so the LLM has a signal for disambiguation.

These tests intentionally avoid any external services (no PDF, no
Qdrant, no OpenAI) and are safe to run in any environment.

Usage
-----
    python scripts/test_pdf_number_fix.py
"""
from __future__ import annotations

import sys
from pathlib import Path


# Ensure the local src/ is importable when run from a checkout root.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

# (input, expected_output, description).
# Cases where expected == input verify that the fix leaves text unchanged.
NUMBER_BREAK_CASES: list[tuple[str, str, str]] = [
    # --- true positives: line-break artefacts that should be repaired ---
    # These are isolated patterns with NO range-indicator context before them,
    # so the context guard does not apply and the prefix-repeat heuristic fires.
    (
        "256 – 256, 299, 695",
        "256,299,695",
        "three-digit prefix repeated, no range context",
    ),
    (
        "depth threshold of 5 – 5, 000",
        "depth threshold of 5,000",
        "four-digit threshold split at the dash, no range context",
    ),
    (
        "depth threshold of 5 -\n5, 000",
        "depth threshold of 5,000",
        "newline between dash and rest still repaired via \\s",
    ),
    (
        "a cutoff of 10 – 10, 000",
        "a cutoff of 10,000",
        "two-digit prefix repeated, no range context",
    ),
    (
        "filtered to 5 - 5, 000 reads",
        "filtered to 5,000 reads",
        "fix preserves surrounding text, no range context",
    ),

    # --- v205: range-indicator context must prevent the merge ---
    # The GRUMP paper's actual sentence — v203 destroyed this into "1,250,359".
    (
        "sequencing depth was calculated, which ranged from 1 – 1, 250, 359",
        "sequencing depth was calculated, which ranged from 1 – 1, 250, 359",
        "'ranged from' guards a same-prefix range that v203 would have merged",
    ),
    (
        "total reads ranged from 1 – 1, 250, 359 across samples",
        "total reads ranged from 1 – 1, 250, 359 across samples",
        "'ranged from' in any position guards the range",
    ),
    (
        "values ranging between 100 – 100, 000",
        "values ranging between 100 – 100, 000",
        "'between' signals a range even with repeated prefix",
    ),
    (
        "counts spanning 1 – 1, 000, 000 observations",
        "counts spanning 1 – 1, 000, 000 observations",
        "'spanning' signals a range",
    ),
    (
        "abundances vary from 1 to 1 – 1, 250, 359",
        "abundances vary from 1 to 1 – 1, 250, 359",
        "'vary from ... to' pattern guards the range",
    ),

    # --- false-positive regressions from v202 (kept) ---
    # Different-leading-digit ranges: the v203 prefix-repeat guard already
    # protects these, with or without a range indicator.
    (
        "sequencing depths ranged from 1 – 5, 000 reads",
        "sequencing depths ranged from 1 – 5, 000 reads",
        "genuine range 1 to 5,000 must not be merged (v202 bug)",
    ),
    (
        "a range of 2 – 10, 000 samples",
        "a range of 2 – 10, 000 samples",
        "genuine range 2 to 10,000 must not be merged",
    ),
    (
        "values from 100 – 200, 000",
        "values from 100 – 200, 000",
        "genuine range 100 to 200,000 must not be merged",
    ),

    # --- clean cases that should pass through untouched ---
    (
        "depth threshold of 5000",
        "depth threshold of 5000",
        "isolated clean number is unchanged",
    ),
    (
        "depth threshold of 5 – 5000",
        "depth threshold of 5 – 5000",
        "no comma separator — pattern does not match",
    ),
    (
        "n = 1, 000 samples",
        "n = 1, 000 samples",
        "bare comma-grouped number without dash prefix is unchanged",
    ),
    (
        "subsample to 5000 reads retained",
        "subsample to 5000 reads retained",
        "methods-section number with no artefact is unchanged",
    ),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_number_break_cases() -> tuple[int, int]:
    from cmap_agent.sync.text_fixes import fix_pdf_number_breaks

    passed = 0
    failed = 0
    print("== _fix_pdf_number_breaks ==")
    for src, expected, desc in NUMBER_BREAK_CASES:
        got = fix_pdf_number_breaks(src)
        ok = got == expected
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {desc}")
        if not ok:
            print(f"         input:    {src!r}")
            print(f"         expected: {expected!r}")
            print(f"         got:      {got!r}")
            failed += 1
        else:
            passed += 1
    return passed, failed


def _run_format_rank_cases() -> tuple[int, int]:
    from cmap_agent.rag.format import format_kb_context

    passed = 0
    failed = 0
    print("\n== format_kb_context rank annotation ==")

    hits = [
        {
            "id": "refbank:GRUMP:paper:abc#chunk12",
            "text": "samples were filtered to a minimum sequencing depth of 5,000 reads",
            "metadata": {"doc_type": "paper_chunk", "title": "GRUMP — paper", "source": "reference_bank"},
        },
        {
            "id": "refbank:GRUMP:paper:abc#chunk26",
            "text": "a rarefaction threshold of 1,000 was applied for alpha-diversity",
            "metadata": {"doc_type": "paper_chunk", "title": "GRUMP — paper", "source": "reference_bank"},
        },
    ]
    out = format_kb_context(hits)

    # Assertion 1: rank=1 precedes rank=2.
    r1_idx = out.find("rank=1")
    r2_idx = out.find("rank=2")
    if r1_idx == -1 or r2_idx == -1:
        print("  [FAIL] rank annotations missing from output")
        print(f"         output:\n{out}")
        failed += 1
    elif r1_idx >= r2_idx:
        print("  [FAIL] rank=1 does not precede rank=2")
        failed += 1
    else:
        print("  [PASS] rank=1 precedes rank=2")
        passed += 1

    # Assertion 2: header format preserved.
    if "[KB:refbank:GRUMP:paper:abc#chunk12 rank=1]" in out:
        print("  [PASS] rank=1 block header well-formed")
        passed += 1
    else:
        print("  [FAIL] rank=1 block header not found in expected form")
        print(f"         output:\n{out}")
        failed += 1

    # Assertion 3: doc text is preserved (not truncated for these short inputs).
    if "minimum sequencing depth of 5,000" in out and "rarefaction threshold of 1,000" in out:
        print("  [PASS] both chunk texts preserved in output")
        passed += 1
    else:
        print("  [FAIL] chunk text missing from output")
        failed += 1

    return passed, failed


def _run_format_cap_cases() -> tuple[int, int]:
    """Per-doc-type cap behaviour introduced in v204."""
    from cmap_agent.rag.format import format_kb_context

    passed = 0
    failed = 0
    print("\n== format_kb_context per-doc-type cap ==")

    long_text = "A" * 6500  # 6500 chars — between refbank cap (2000) and catalog cap (7000)

    # paper_chunk → 2000-char cap → text should be truncated to 2000 chars.
    out_pc = format_kb_context([
        {"id": "pc1", "text": long_text,
         "metadata": {"doc_type": "paper_chunk", "title": "t", "source": "s"}}
    ])
    # Count 'A's in output — should be exactly 2000.
    a_count_pc = out_pc.count("A")
    if a_count_pc == 2000:
        print(f"  [PASS] paper_chunk capped at 2000 chars (got {a_count_pc})")
        passed += 1
    else:
        print(f"  [FAIL] paper_chunk expected 2000 chars, got {a_count_pc}")
        failed += 1

    # dataset → 7000-char cap → text shorter than cap should pass through fully.
    out_ds = format_kb_context([
        {"id": "ds1", "text": long_text,
         "metadata": {"doc_type": "dataset", "title": "t", "source": "s"}}
    ])
    a_count_ds = out_ds.count("A")
    if a_count_ds == 6500:
        print(f"  [PASS] dataset preserves 6500-char body (cap 7000, got {a_count_ds})")
        passed += 1
    else:
        print(f"  [FAIL] dataset expected 6500 chars, got {a_count_ds}")
        failed += 1

    # variable → 7000-char cap
    out_var = format_kb_context([
        {"id": "v1", "text": long_text,
         "metadata": {"doc_type": "variable", "title": "t", "source": "s"}}
    ])
    if out_var.count("A") == 6500:
        print("  [PASS] variable preserves 6500-char body (cap 7000)")
        passed += 1
    else:
        print(f"  [FAIL] variable expected 6500 chars, got {out_var.count('A')}")
        failed += 1

    # Unknown doc_type → conservative 2000-char default.
    out_unknown = format_kb_context([
        {"id": "u1", "text": long_text,
         "metadata": {"doc_type": "something_new", "title": "t", "source": "s"}}
    ])
    if out_unknown.count("A") == 2000:
        print("  [PASS] unknown doc_type falls back to 2000-char default")
        passed += 1
    else:
        print(f"  [FAIL] unknown doc_type expected 2000 chars, got {out_unknown.count('A')}")
        failed += 1

    return passed, failed


def main() -> int:
    total_passed = 0
    total_failed = 0

    p, f = _run_number_break_cases()
    total_passed += p
    total_failed += f

    p, f = _run_format_rank_cases()
    total_passed += p
    total_failed += f

    p, f = _run_format_cap_cases()
    total_passed += p
    total_failed += f

    print(f"\nSummary: {total_passed} passed, {total_failed} failed")
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
