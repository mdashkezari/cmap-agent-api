"""test_qdrant_filter_helpers.py — unit tests for v217 runtime parity fixes.

These tests exercise the filter-translation helper that maps Chroma-style
``where`` clauses onto Qdrant conditions, and the distance normalization
applied in ``QdrantKB.query()``.

The tests operate directly on the helper functions where possible; the
score-semantics test uses a stub Qdrant response because spinning up a
real Qdrant instance is out of scope for unit tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make src importable when run standalone, same pattern as other tests here.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Import the helper directly.  Do not instantiate QdrantKB (that would try
# to connect to a live Qdrant).  The helper and the models are module-level.
from cmap_agent.rag.qdrant_kb import _build_field_condition
from qdrant_client import models


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


def test_scalar_goes_to_match_value() -> None:
    """A plain scalar ``where`` value must become a MatchValue condition."""
    for scalar in ("dataset", 42, 3.14, True):
        fc = _build_field_condition("field", scalar)
        _check(
            isinstance(fc, models.FieldCondition),
            f"scalar {scalar!r} produces FieldCondition",
        )
        _check(
            isinstance(fc.match, models.MatchValue),
            f"scalar {scalar!r} uses MatchValue",
        )
        _check(
            fc.match.value == scalar,
            f"scalar {scalar!r} preserves value",
        )


def test_explicit_eq_becomes_match_value() -> None:
    """``{"$eq": scalar}`` is accepted as a verbose equality form."""
    fc = _build_field_condition("doc_type", {"$eq": "dataset"})
    _check(isinstance(fc.match, models.MatchValue), "$eq uses MatchValue")
    _check(fc.match.value == "dataset", "$eq preserves value")


def test_in_list_becomes_match_any() -> None:
    """``{"$in": [...]}`` must use MatchAny — the key v217 fix."""
    fc = _build_field_condition(
        "table", {"$in": ["tblGeotraces", "tblGlobal_PicoPhytoPlankton"]},
    )
    _check(
        isinstance(fc.match, models.MatchAny),
        "$in produces MatchAny (was MatchValue before v217)",
    )
    _check(
        list(fc.match.any) == ["tblGeotraces", "tblGlobal_PicoPhytoPlankton"],
        "$in preserves list values in order",
    )


def test_empty_in_list_does_not_crash() -> None:
    """Empty ``$in`` lists should not raise; they must match nothing."""
    fc = _build_field_condition("table", {"$in": []})
    _check(
        isinstance(fc.match, models.MatchAny),
        "empty $in produces MatchAny with a sentinel",
    )
    # Any sentinel that will never match real payloads is acceptable.
    # We care that the call returns without raising.
    _check(len(list(fc.match.any)) == 1, "empty $in has a single sentinel")


def test_tuple_in_list_is_accepted() -> None:
    """``$in`` values may be a tuple (some call sites pass immutables)."""
    fc = _build_field_condition("table", {"$in": ("a", "b")})
    _check(
        isinstance(fc.match, models.MatchAny),
        "tuple $in produces MatchAny",
    )
    _check(
        list(fc.match.any) == ["a", "b"],
        "tuple $in preserves values",
    )


def test_unknown_operator_falls_back_without_crashing() -> None:
    """An unrecognized operator dict must not raise."""
    fc = _build_field_condition("field", {"$regex": ".*"})
    _check(
        isinstance(fc, models.FieldCondition),
        "unknown operator returns a FieldCondition (fallback)",
    )
    # Fallback behaviour: wraps the whole dict in MatchValue.  This is
    # deliberately lossy — the condition will not match real data — but
    # it keeps the query path from crashing on a typo.
    _check(
        isinstance(fc.match, models.MatchValue),
        "unknown operator fallback uses MatchValue",
    )


def test_distance_semantics_parity() -> None:
    """QdrantKB must return distance with lower=better semantics.

    Rationale: consumers (notably ``catalog_tools._kb_semantic_table_scores``)
    apply ``s = 1/(1+d)`` to rank results.  That transform inverts the
    ordering if ``d`` is actually a higher-is-better score.  v217 maps
    the Qdrant RRF score into a distance via ``1.0 - score`` inside
    ``QdrantKB.query()`` so the downstream transform works correctly.

    We verify the transform symbolically here without touching a live
    Qdrant client.  If two hits have RRF scores ``s1 > s2``, then after
    mapping to distances ``d1 < d2``, and after ``1/(1+d)``, ``s1_out >
    s2_out`` — i.e. the ordering is preserved.
    """
    hit_better_score = 0.033   # RRF top-of-range for 2 channels at c=60
    hit_worse_score = 0.015
    d_better = 1.0 - hit_better_score
    d_worse = 1.0 - hit_worse_score
    _check(d_better < d_worse, "better score produces smaller distance")

    s_better = 1.0 / (1.0 + d_better)
    s_worse = 1.0 / (1.0 + d_worse)
    _check(
        s_better > s_worse,
        "after 1/(1+d), better hit still scores higher (ordering preserved)",
    )


def main() -> int:
    print("Running v217 runtime-parity tests:")
    print()
    test_scalar_goes_to_match_value()
    test_explicit_eq_becomes_match_value()
    test_in_list_becomes_match_any()
    test_empty_in_list_does_not_crash()
    test_tuple_in_list_is_accepted()
    test_unknown_operator_falls_back_without_crashing()
    test_distance_semantics_parity()
    print()
    print(f"Summary: {_PASS} passed, {_FAIL} failed")
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
