"""test_variable_availability_gate.py — unit tests for v221 gate helpers.

Confirms:
- ``_concept_tokens_from_query`` preserves concept words and drops
  stopwords / modality words.
- ``_variable_availability_score`` returns 0 when no variable matches
  any concept, 1 otherwise, and 1 (no-op) when it lacks data to gate on.
"""
from __future__ import annotations

import re as _re
import sys
from pathlib import Path
from typing import Any


_ROOT = Path(__file__).resolve().parents[1]


def _extract_function_source(path: Path, name: str) -> str:
    """Pull a single top-level function definition out of a source file.

    Matches ``def <name>(`` at column 0, copies through to (but not
    including) the next top-level definition.  Used so we don't have to
    import the full ``catalog_tools`` module (which pulls SQLAlchemy,
    pydantic, and other heavy deps the test runner may not have).
    """
    src = path.read_text()
    m = _re.search(
        rf"^def {name}\(.*?(?=^def |\Z)",
        src,
        _re.DOTALL | _re.MULTILINE,
    )
    if not m:
        raise RuntimeError(f"Could not locate function {name!r} in {path}")
    return m.group(0)


def _load_gate_helpers():
    """Import the two helpers with a minimal fake cache module."""
    path = _ROOT / "src" / "cmap_agent" / "tools" / "catalog_tools.py"
    # We need _CONCEPT_STOPWORDS (module constant) + the two functions.
    # Grab the stopword block too.
    src = path.read_text()
    m = _re.search(
        r"_CONCEPT_STOPWORDS: frozenset\[str\] = frozenset\(\{.*?\}\)",
        src,
        _re.DOTALL,
    )
    if not m:
        raise RuntimeError("Could not locate _CONCEPT_STOPWORDS")
    stopwords_src = m.group(0)

    concept_src = _extract_function_source(path, "_concept_tokens_from_query")
    avail_src = _extract_function_source(path, "_variable_availability_score")

    # Fake the _catalog_cache the availability function refers to.
    class _FakeCache:
        rows: list[dict[str, Any]] = []
    ns: dict = {"_catalog_cache": _FakeCache()}
    # frozenset needs to be available in ns
    ns["frozenset"] = frozenset
    exec(stopwords_src, ns)
    exec(concept_src, ns)
    exec(avail_src, ns)
    return ns["_concept_tokens_from_query"], ns["_variable_availability_score"], ns["_catalog_cache"]


_concept_tokens_from_query, _variable_availability_score, _fake_cache = _load_gate_helpers()


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


def test_concept_includes_family_terms() -> None:
    """A query hitting the 'nutrients' family should produce nitrate/phosphate/etc."""
    tokens = _concept_tokens_from_query("surface dissolved nitrate south atlantic")
    _check("nitrate" in tokens, f"nitrate in concept: {tokens}")
    _check("phosphate" in tokens,
           f"family expansion brings phosphate: {tokens}")
    _check("nutrient" in tokens,
           f"family expansion brings nutrient: {tokens}")


def test_concept_drops_stopwords_and_regions() -> None:
    tokens = _concept_tokens_from_query("surface dissolved nitrate south atlantic")
    for dropped in ("south", "atlantic", "the", "over", "in"):
        _check(dropped not in tokens,
               f"regional/stopword {dropped!r} dropped: {tokens}")


def test_concept_drops_modality_words() -> None:
    """'satellite' / 'model' etc. are routing cues, not variable concepts."""
    tokens = _concept_tokens_from_query("satellite chlorophyll north atlantic")
    for dropped in ("satellite", "model", "climatology", "monthly", "daily"):
        _check(dropped not in tokens,
               f"modality {dropped!r} dropped: {tokens}")
    _check("chlorophyll" in tokens, f"chlorophyll in concept: {tokens}")


def test_concept_empty_query_is_empty_set() -> None:
    _check(_concept_tokens_from_query("") == set(), "empty query -> empty set")
    _check(_concept_tokens_from_query(None) == set(), "None query -> empty set")


def test_concept_raw_tokens_fallback() -> None:
    """A concept not in the family dict still survives via the raw-tokens fallback."""
    tokens = _concept_tokens_from_query("bathymetry coastal")
    _check("bathymetry" in tokens, f"raw token bathymetry: {tokens}")
    _check("coastal" in tokens, f"raw token coastal: {tokens}")


def test_availability_returns_1_when_concept_empty() -> None:
    """No concept -> cannot gate -> always 1.0 (no-op)."""
    r = {"table": "tblSSS_NRT"}
    _check(_variable_availability_score(r, set()) == 1.0,
           "empty concept -> 1.0")


def test_availability_returns_1_when_cache_empty() -> None:
    """Cache has no data -> cannot gate -> 1.0."""
    _fake_cache.rows = []
    r = {"table": "tblSSS_NRT"}
    _check(_variable_availability_score(r, {"nitrate"}) == 1.0,
           "empty cache -> 1.0")


def test_availability_returns_0_when_no_variable_matches() -> None:
    """SSS has variable 'sss' / 'Sea Surface Salinity' — no nitrate match."""
    _fake_cache.rows = [
        {"table_name": "tblSSS_NRT", "variable": "sss",
         "long_name": "Sea Surface Salinity"},
    ]
    r = {"table": "tblSSS_NRT"}
    _check(_variable_availability_score(r, {"nitrate", "phosphate"}) == 0.0,
           "SSS has no nitrate variable -> 0.0 (gate fires)")


def test_availability_returns_1_when_variable_matches() -> None:
    """Darwin_Nutrient has 'nitrate' — concept 'nitrate' must match."""
    _fake_cache.rows = [
        {"table_name": "tblDarwin_Nutrient_Climatology",
         "variable": "NO3_darwin_clim",
         "long_name": "Nitrate concentration (climatology)"},
        {"table_name": "tblDarwin_Nutrient_Climatology",
         "variable": "PO4_darwin_clim",
         "long_name": "Phosphate concentration (climatology)"},
    ]
    r = {"table": "tblDarwin_Nutrient_Climatology"}
    _check(_variable_availability_score(r, {"nitrate"}) == 1.0,
           "Darwin has nitrate variable -> 1.0 (no gate)")


def test_availability_matches_on_variable_name_or_long_name() -> None:
    """MODIS POC's variable is 'POC', concept should match via short name."""
    _fake_cache.rows = [
        {"table_name": "tblModis_POC", "variable": "POC",
         "long_name": "Particulate Organic Carbon"},
    ]
    r = {"table": "tblModis_POC"}
    _check(_variable_availability_score(r, {"poc", "carbon"}) == 1.0,
           "POC matches on short variable name")


def test_availability_case_insensitive() -> None:
    """Query 'NITRATE' should still match a long_name 'Nitrate ...'"""
    _fake_cache.rows = [
        {"table_name": "tblX", "variable": "no3",
         "long_name": "Nitrate concentration"},
    ]
    r = {"table": "tblX"}
    _check(_variable_availability_score(r, {"nitrate"}) == 1.0,
           "case-insensitive match")


def test_availability_is_1_for_unknown_table() -> None:
    """Table not present in cache -> cannot gate -> 1.0 (don't penalize)."""
    _fake_cache.rows = [
        {"table_name": "tblKnown", "variable": "sst", "long_name": "SST"},
    ]
    r = {"table": "tblUnknown"}
    _check(_variable_availability_score(r, {"nitrate"}) == 1.0,
           "unknown table -> 1.0 (no penalty)")


def test_availability_regression_nitrate_case() -> None:
    """End-to-end: the CS004 failing case."""
    _fake_cache.rows = [
        # SSS has zero nitrate-related variables
        {"table_name": "tblSSS_NRT", "variable": "sss",
         "long_name": "Sea Surface Salinity"},
        # Darwin has nitrate
        {"table_name": "tblDarwin_Nutrient_Climatology",
         "variable": "NO3_darwin_clim",
         "long_name": "Nitrate"},
        # Pisces has nitrate
        {"table_name": "tblPisces_Forecast", "variable": "no3",
         "long_name": "Nitrate concentration"},
    ]
    concept = _concept_tokens_from_query("surface dissolved nitrate south atlantic")
    _check(_variable_availability_score({"table": "tblSSS_NRT"}, concept) == 0.0,
           "regression: SSS gated for nitrate query")
    _check(_variable_availability_score(
        {"table": "tblDarwin_Nutrient_Climatology"}, concept) == 1.0,
           "regression: Darwin not gated for nitrate query")
    _check(_variable_availability_score(
        {"table": "tblPisces_Forecast"}, concept) == 1.0,
           "regression: Pisces not gated for nitrate query")


def main() -> int:
    print("Running variable-availability gate tests (v221):")
    print()
    test_concept_includes_family_terms()
    print()
    test_concept_drops_stopwords_and_regions()
    print()
    test_concept_drops_modality_words()
    print()
    test_concept_empty_query_is_empty_set()
    print()
    test_concept_raw_tokens_fallback()
    print()
    test_availability_returns_1_when_concept_empty()
    test_availability_returns_1_when_cache_empty()
    test_availability_returns_0_when_no_variable_matches()
    test_availability_returns_1_when_variable_matches()
    test_availability_matches_on_variable_name_or_long_name()
    test_availability_case_insensitive()
    test_availability_is_1_for_unknown_table()
    print()
    test_availability_regression_nitrate_case()
    print()
    print(f"Summary: {_PASS} passed, {_FAIL} failed")
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
