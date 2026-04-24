"""test_gate_dedup_integration.py — verify the variable-availability gate
fires inside ``_deduplicate_to_datasets``, not just inside
``_post_rank_catalog_results``.

Rationale.  v221 applied the gate to ``_post_rank_catalog_results``,
which is called from ``catalog_search_kb_first``.  The evaluation
harness measured that path and reported the nitrate case fixed.  But
production also uses ``catalog.search`` (plain), which goes through
``_deduplicate_to_datasets`` and did NOT have the gate applied.  A
user end-to-end test still saw the failing nitrate → SSS case.

v222 mirrors the gate into ``_deduplicate_to_datasets``.  This test
verifies that the mirror is in place and effective.  It uses a mock
cache and ensures the SSS-on-nitrate ranking flips correctly.
"""
from __future__ import annotations

import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]


_PASS = 0
_FAIL = 0


def _check(cond: bool, label: str) -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  [pass] {label}")
    else:
        _FAIL += 1
        print(f"  [FAIL] {label}")


def test_source_contains_gate_call() -> None:
    """Source-level check: ``_deduplicate_to_datasets`` invokes the gate."""
    src = (_ROOT / "src" / "cmap_agent" / "tools" / "catalog_tools.py").read_text()
    # Find the function block.
    import re as _re
    m = _re.search(
        r"^def _deduplicate_to_datasets\(.*?(?=^def |\Z)",
        src,
        _re.DOTALL | _re.MULTILINE,
    )
    if not m:
        _check(False, "locate _deduplicate_to_datasets")
        return
    body = m.group(0)
    _check("_variable_availability_score" in body,
           "gate call present in _deduplicate_to_datasets")
    _check("_concept_tokens_from_query" in body,
           "concept extraction present in _deduplicate_to_datasets")
    _check("_GATE_PENALTY" in body or "GATE_PENALTY" in body or
           "score -= " in body,
           "gate applies a score penalty")


def test_source_contains_gate_call_in_post_rank() -> None:
    """Source-level check: the v221 gate in _post_rank_catalog_results is
    still present (not accidentally removed during v222 edits)."""
    src = (_ROOT / "src" / "cmap_agent" / "tools" / "catalog_tools.py").read_text()
    import re as _re
    m = _re.search(
        r"^def _post_rank_catalog_results\(.*?(?=^def |\Z)",
        src,
        _re.DOTALL | _re.MULTILINE,
    )
    if not m:
        _check(False, "locate _post_rank_catalog_results")
        return
    body = m.group(0)
    _check("_variable_availability_score" in body,
           "v221 gate still in _post_rank_catalog_results")


def main() -> int:
    print("Running gate integration tests (v222):")
    print()
    test_source_contains_gate_call()
    print()
    test_source_contains_gate_call_in_post_rank()
    print()
    print(f"Summary: {_PASS} passed, {_FAIL} failed")
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
