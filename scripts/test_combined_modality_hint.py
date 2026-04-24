"""test_combined_modality_hint.py — unit tests for v224.

Covers the three-source modality resolution in
``_combined_modality_hint``:

1. Structured ``intent.sensor`` / ``intent.make`` fields
2. Raw user message text
3. ``intent.search_query`` (after intent-prompt stripping)

And the ambiguity policy: conflicting sources produce ``{}``.
"""
from __future__ import annotations

import re as _re
import sys
from pathlib import Path
from types import SimpleNamespace


_ROOT = Path(__file__).resolve().parents[1]


def _load_helper():
    """Pull ``_combined_modality_hint`` out of runner.py without
    importing the full runner (which depends on OpenAI client etc.).

    The helper calls ``_modality_intent_from_query`` from
    catalog_tools via a lazy import — we pre-populate the lazy-import
    target with a stub that mirrors the real behaviour.
    """
    path = _ROOT / "src" / "cmap_agent" / "agent" / "runner.py"
    src = path.read_text()
    m = _re.search(
        r"^def _combined_modality_hint\(.*?(?=^def |\Z)",
        src,
        _re.DOTALL | _re.MULTILINE,
    )
    if not m:
        raise RuntimeError("could not locate _combined_modality_hint")

    # Build a fake cmap_agent.tools.catalog_tools module so the lazy
    # import inside the helper resolves to the real thing.
    import importlib.util
    catalog_path = _ROOT / "src" / "cmap_agent" / "tools" / "catalog_tools.py"
    # Extract just _MODALITY_CUES + _modality_intent_from_query source
    csrc = catalog_path.read_text()
    cues_m = _re.search(
        r"_MODALITY_CUES:.*?\)\n",
        csrc,
        _re.DOTALL,
    )
    intent_fn_m = _re.search(
        r"^def _modality_intent_from_query\(.*?(?=^def |\Z)",
        csrc,
        _re.DOTALL | _re.MULTILINE,
    )
    if not cues_m or not intent_fn_m:
        raise RuntimeError("could not locate catalog_tools pieces")

    fake_module_ns: dict = {}
    exec(cues_m.group(0), fake_module_ns)
    exec(intent_fn_m.group(0), fake_module_ns)

    # Stitch into sys.modules so the helper's lazy import works.
    import types as _types
    fake_mod = _types.ModuleType("cmap_agent.tools.catalog_tools")
    for k, v in fake_module_ns.items():
        setattr(fake_mod, k, v)
    # Also create the package parents if absent.
    for pkg in ("cmap_agent", "cmap_agent.tools"):
        if pkg not in sys.modules:
            sys.modules[pkg] = _types.ModuleType(pkg)
    sys.modules["cmap_agent.tools.catalog_tools"] = fake_mod

    ns: dict = {}
    exec(m.group(0), ns)
    return ns["_combined_modality_hint"]


_combined_modality_hint = _load_helper()


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


def _intent(**kwargs) -> SimpleNamespace:
    """Fake UserIntent with sensible defaults."""
    defaults = {"sensor": None, "make": None, "search_query": ""}
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------
# Source 1: structured intent fields
# ---------------------------------------------------------------------

def test_intent_sensor_field_used() -> None:
    """intent.sensor='Satellite' + empty message/search_query -> sensor:satellite"""
    i = _intent(sensor="Satellite")
    _check(_combined_modality_hint(i, "") == {"sensor": "satellite"},
           "intent.sensor=Satellite -> sensor:satellite")


def test_intent_make_field_used() -> None:
    i = _intent(make="Model")
    _check(_combined_modality_hint(i, "") == {"make": "model"},
           "intent.make=Model -> make:model")


def test_intent_in_situ_sensor_normalised() -> None:
    """'in-Situ' -> 'in-situ' (lowercase)."""
    i = _intent(sensor="in-Situ")
    _check(_combined_modality_hint(i, "") == {"sensor": "in-situ"},
           "intent.sensor='in-Situ' -> sensor:in-situ")


# ---------------------------------------------------------------------
# Source 2: raw user message
# ---------------------------------------------------------------------

def test_user_message_used_when_intent_empty() -> None:
    """intent has no sensor/make but message mentions 'satellite'."""
    i = _intent(search_query="sst")
    _check(_combined_modality_hint(i, "colocalize with satellite sst") == {"sensor": "satellite"},
           "user_message satellite -> sensor:satellite")


def test_user_message_in_situ_phrase() -> None:
    i = _intent(search_query="chlorophyll")
    _check(_combined_modality_hint(i, "in-situ chlorophyll from cruises") == {"sensor": "in-situ"},
           "user_message 'in-situ' -> sensor:in-situ")


def test_user_message_model() -> None:
    i = _intent(search_query="nitrate climatology")
    _check(_combined_modality_hint(i, "model nitrate climatology") == {"make": "model"},
           "user_message 'model' -> make:model")


# ---------------------------------------------------------------------
# Source 3: search_query fallback
# ---------------------------------------------------------------------

def test_search_query_fallback() -> None:
    """Neither intent fields nor user_message set, but search_query
    somehow retained the qualifier (LLM didn't follow the strip rule)."""
    i = _intent(search_query="satellite chlorophyll")
    _check(_combined_modality_hint(i, "") == {"sensor": "satellite"},
           "search_query fallback picks up satellite")


# ---------------------------------------------------------------------
# Ambiguity: disagreement returns empty
# ---------------------------------------------------------------------

def test_conflict_returns_empty() -> None:
    """intent.sensor says satellite but user message says model."""
    i = _intent(sensor="Satellite")
    _check(_combined_modality_hint(i, "colocalize with model output") == {},
           "sensor=Satellite + message 'model' conflict -> {}")


def test_multi_modality_message_returns_empty() -> None:
    """User message itself is ambiguous - two modalities mentioned."""
    i = _intent(search_query="chlorophyll")
    # "satellite vs model" triggers both cue groups in
    # _modality_intent_from_query, which already returns {} for that.
    # _combined_modality_hint should propagate that emptiness.
    _check(_combined_modality_hint(i, "satellite vs model chlorophyll") == {},
           "ambiguous message -> {}")


def test_consistent_sources_collapse_to_one() -> None:
    """intent.sensor and user_message both say satellite -> one hint."""
    i = _intent(sensor="Satellite")
    _check(_combined_modality_hint(i, "satellite chlorophyll map") == {"sensor": "satellite"},
           "consistent sources collapse to one hint")


# ---------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------

def test_all_empty_returns_empty() -> None:
    i = _intent()
    _check(_combined_modality_hint(i, "") == {},
           "all empty -> {}")


def test_none_values_handled() -> None:
    i = _intent(sensor=None, make=None, search_query=None)
    _check(_combined_modality_hint(i, None) == {},
           "None values safe -> {}")


def test_empty_intent_fields_ignored() -> None:
    """Empty strings in intent fields should be treated as not-set."""
    i = _intent(sensor="", make="", search_query="satellite sst")
    _check(_combined_modality_hint(i, "") == {"sensor": "satellite"},
           "empty intent strings fall through to search_query")


# ---------------------------------------------------------------------
# Regression: the colocalize test case from user report
# ---------------------------------------------------------------------

def test_regression_colocalize_satellite_sst() -> None:
    """User: 'colocalize this dataset with satellite sst'
    intent.py strips 'satellite' from search_query but should have
    populated intent.sensor=Satellite."""
    # Path A: intent.sensor is populated correctly by intent extractor
    i = _intent(sensor="Satellite", search_query="sst")
    _check(_combined_modality_hint(i, "colocalize this dataset with satellite sst") == {"sensor": "satellite"},
           "regression: colocalize+satellite sst with populated intent.sensor")

    # Path B: intent.sensor was NOT populated (extractor missed it),
    # but the user message still contains 'satellite'
    i2 = _intent(sensor=None, search_query="sst")
    _check(_combined_modality_hint(i2, "colocalize this dataset with satellite sst") == {"sensor": "satellite"},
           "regression: colocalize+satellite sst falls back to user_message")


def main() -> int:
    print("Running _combined_modality_hint tests (v224):")
    print()
    test_intent_sensor_field_used()
    test_intent_make_field_used()
    test_intent_in_situ_sensor_normalised()
    print()
    test_user_message_used_when_intent_empty()
    test_user_message_in_situ_phrase()
    test_user_message_model()
    print()
    test_search_query_fallback()
    print()
    test_conflict_returns_empty()
    test_multi_modality_message_returns_empty()
    test_consistent_sources_collapse_to_one()
    print()
    test_all_empty_returns_empty()
    test_none_values_handled()
    test_empty_intent_fields_ignored()
    print()
    test_regression_colocalize_satellite_sst()
    print()
    print(f"Summary: {_PASS} passed, {_FAIL} failed")
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
