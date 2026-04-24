"""test_bare_query.py — unit tests for _bare_query (v219).

Confirms the post-v219 behaviour: intent-bearing qualifiers are
preserved, only pure filler is stripped.  See the ``_bare_query``
docstring in ``runner.py`` for the full rationale.
"""
from __future__ import annotations

import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# Import the function.  runner.py has many other imports at module level
# (pydantic, sqlalchemy, openai client, ...) that may not be present in
# every test environment.  Isolate just the function we want by parsing
# the source and exec'ing the def — same pattern as other scripts here
# that need to test a pure helper without the full project deps.
def _import_bare_query():
    import re as _re
    path = _ROOT / "src" / "cmap_agent" / "agent" / "runner.py"
    src = path.read_text()
    # Extract the function definition block.
    m = _re.search(
        r"^def _bare_query\(.*?(?=^def |\Z)",
        src,
        _re.DOTALL | _re.MULTILINE,
    )
    if not m:
        raise RuntimeError("Could not locate _bare_query in runner.py")
    ns: dict = {}
    exec(m.group(0), ns)
    return ns["_bare_query"]


_bare_query = _import_bare_query()


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


def test_preserves_intent_qualifiers() -> None:
    """Qualifiers that reflect user intent must survive."""
    for qualifier in (
        "satellite", "model", "observation", "in-situ",
        "climatology", "monthly", "daily", "seasonal",
        "MODIS", "SeaWiFS", "MiSeq", "VIIRS",
        "North Atlantic", "south pacific", "southern ocean",
    ):
        query = f"{qualifier.lower()} chlorophyll"
        out = _bare_query(query)
        _check(
            qualifier.lower() in out.lower(),
            f"preserves {qualifier!r} in {query!r} -> {out!r}",
        )


def test_strips_pure_filler() -> None:
    """Filler words must be stripped."""
    cases = [
        ("cmap chlorophyll", "chlorophyll"),
        ("chlorophyll data", "chlorophyll"),
        ("chlorophyll datasets", "chlorophyll"),
        ("please show me chlorophyll", "chlorophyll"),
        ("can you show chlorophyll", "chlorophyll"),
        ("give me nitrate", "nitrate"),
        ("i want chlorophyll", "chlorophyll"),
    ]
    for inp, expected in cases:
        out = _bare_query(inp)
        _check(out == expected, f"{inp!r} -> {out!r} (expected {expected!r})")


def test_preserves_when_nothing_to_strip() -> None:
    """A clean query passes through unchanged (after lowercasing)."""
    for q in (
        "chlorophyll",
        "satellite sst",
        "nitrate climatology in the north atlantic",
    ):
        out = _bare_query(q)
        _check(out == q.lower(), f"{q!r} -> {out!r}")


def test_empty_or_blank_query_is_stable() -> None:
    """Empty / whitespace input returns the original (no crash)."""
    for inp in ("", "   ", None):
        # _bare_query accepts None via ``(search_query or "")``
        out = _bare_query(inp)
        # Acceptable outputs: empty string or the original input passed
        # through.  The guard is: must not raise.
        _check(isinstance(out, str), f"empty input {inp!r} -> {out!r} (str)")


def test_regression_intent_examples() -> None:
    """Regression: the three user-reported failure cases now preserve intent."""
    cases = [
        # Case 1: surface dissolved nitrate South Atlantic
        ("surface dissolved nitrate south atlantic",
         ["surface", "nitrate", "south", "atlantic"]),
        # Case 2: satellite POC South Pacific
        ("satellite poc south pacific",
         ["satellite", "poc", "south", "pacific"]),
        # Case 3: satellite chlorophyll North Atlantic
        ("satellite chlorophyll north atlantic",
         ["satellite", "chlorophyll", "north", "atlantic"]),
    ]
    for query, required_tokens in cases:
        out = _bare_query(query).lower()
        for tok in required_tokens:
            _check(
                tok in out,
                f"regression: token {tok!r} survives in {query!r} -> {out!r}",
            )


def test_does_not_strip_substrings_of_real_words() -> None:
    """'data' as filler must not strip 'datasets' partial, nor the 'data'
    inside longer tokens like 'database'.  Word-bounded stripping required.
    """
    out = _bare_query("metadata database")
    # Neither 'metadata' nor 'database' should be damaged.
    _check("metadata" in out and "database" in out,
           f"does not break substrings: 'metadata database' -> {out!r}")


def main() -> int:
    print("Running _bare_query tests (v219):")
    print()
    test_preserves_intent_qualifiers()
    print()
    test_strips_pure_filler()
    print()
    test_preserves_when_nothing_to_strip()
    print()
    test_empty_or_blank_query_is_stable()
    print()
    test_regression_intent_examples()
    print()
    test_does_not_strip_substrings_of_real_words()
    print()
    print(f"Summary: {_PASS} passed, {_FAIL} failed")
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
