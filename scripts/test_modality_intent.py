"""test_modality_intent.py — unit tests for v223 modality-intent helpers.

Covers:
- ``_modality_intent_from_query`` extracts the intended modality
  from queries using a small phrase-token table, returns {} for
  ambiguous/absent intent.
- ``_modality_score_adjustment`` returns +3.0 on match, -4.0 on
  recognised-opposite, 0.0 when neutral / unknown.
"""
from __future__ import annotations

import re as _re
import sys
from pathlib import Path
from typing import Any


_ROOT = Path(__file__).resolve().parents[1]


def _extract_function_source(path: Path, name: str) -> str:
    src = path.read_text()
    m = _re.search(
        rf"^def {name}\(.*?(?=^def |\Z)",
        src,
        _re.DOTALL | _re.MULTILINE,
    )
    if not m:
        raise RuntimeError(f"Could not locate function {name!r} in {path}")
    return m.group(0)


def _load_modality_helpers():
    """Pull the two helpers + their shared phrase table into a fresh
    namespace, without importing the full catalog_tools module (which
    has heavy deps).
    """
    path = _ROOT / "src" / "cmap_agent" / "tools" / "catalog_tools.py"
    src = path.read_text()
    m = _re.search(
        r"_MODALITY_CUES:.*?\)\n",
        src,
        _re.DOTALL,
    )
    if not m:
        raise RuntimeError("Could not locate _MODALITY_CUES in catalog_tools.py")
    cues_src = m.group(0)

    intent_src = _extract_function_source(path, "_modality_intent_from_query")
    adj_src = _extract_function_source(path, "_modality_score_adjustment")

    ns: dict = {}
    exec(cues_src, ns)
    exec(intent_src, ns)
    exec(adj_src, ns)
    return ns["_modality_intent_from_query"], ns["_modality_score_adjustment"]


_modality_intent_from_query, _modality_score_adjustment = _load_modality_helpers()


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


# -------------------------------------------------------------------
# _modality_intent_from_query
# -------------------------------------------------------------------

def test_intent_satellite() -> None:
    for q in ("satellite chlorophyll",
              "show me satellite sst",
              "satellite POC south pacific"):
        i = _modality_intent_from_query(q)
        _check(i == {"sensor": "satellite"},
               f"{q!r} -> {i}")


def test_intent_in_situ_variants() -> None:
    for q in ("in-situ chlorophyll",
              "in situ nitrate from cruises",
              "insitu data",
              "from cruise stations",
              "cruises in the atlantic"):
        i = _modality_intent_from_query(q)
        _check(i == {"sensor": "in-situ"},
               f"{q!r} -> {i}")


def test_intent_model() -> None:
    for q in ("model nitrate climatology",
              "modeled chlorophyll",
              "modelled sst",
              "simulation output",
              "simulated nitrate"):
        i = _modality_intent_from_query(q)
        _check(i == {"make": "model"},
               f"{q!r} -> {i}")


def test_intent_observation() -> None:
    for q in ("observational chlorophyll",
              "observed nitrate",
              "measured oxygen"):
        i = _modality_intent_from_query(q)
        _check(i == {"make": "observation"},
               f"{q!r} -> {i}")


def test_intent_assimilation() -> None:
    for q in ("assimilation nitrate", "reanalysis wind"):
        i = _modality_intent_from_query(q)
        _check(i == {"make": "assimilation"},
               f"{q!r} -> {i}")


def test_intent_empty_on_ambiguous() -> None:
    """When the query mentions multiple modalities, return empty."""
    for q in ("satellite chlorophyll model comparison",
              "model vs satellite nitrate",
              "in-situ and satellite chlorophyll"):
        i = _modality_intent_from_query(q)
        _check(i == {}, f"ambiguous {q!r} -> {i}")


def test_intent_empty_on_no_modality() -> None:
    for q in ("nitrate in the north atlantic",
              "chlorophyll",
              "",
              None):
        i = _modality_intent_from_query(q or "")
        _check(i == {}, f"no modality {q!r} -> {i}")


# -------------------------------------------------------------------
# _modality_score_adjustment
# -------------------------------------------------------------------

def test_adjustment_boost_on_match() -> None:
    r = {"sensor": "Satellite"}
    _check(_modality_score_adjustment(r, {"sensor": "satellite"}) == 3.0,
           "boost on sensor match")
    r = {"make": "Model"}
    _check(_modality_score_adjustment(r, {"make": "model"}) == 3.0,
           "boost on make match")


def test_adjustment_penalty_on_recognised_opposite() -> None:
    # Satellite intent vs in-situ candidate → penalty
    r = {"sensor": "In-Situ"}
    _check(_modality_score_adjustment(r, {"sensor": "satellite"}) == -4.0,
           "penalty satellite intent, in-situ candidate")
    # In-situ intent vs satellite candidate → penalty
    r = {"sensor": "Satellite"}
    _check(_modality_score_adjustment(r, {"sensor": "in-situ"}) == -4.0,
           "penalty in-situ intent, satellite candidate")
    # Model intent vs observation candidate → penalty
    r = {"make": "Observation"}
    _check(_modality_score_adjustment(r, {"make": "model"}) == -4.0,
           "penalty model intent, observation candidate")


def test_adjustment_neutral_on_unknown_value() -> None:
    """Candidate with sensor='Blend' when intent is satellite: neither
    matches nor is a recognised opposite — should be neutral."""
    r = {"sensor": "Blend"}
    _check(_modality_score_adjustment(r, {"sensor": "satellite"}) == 0.0,
           "neutral on Blend sensor for satellite intent")
    r = {"sensor": "Uncategorized"}
    _check(_modality_score_adjustment(r, {"sensor": "satellite"}) == 0.0,
           "neutral on Uncategorized sensor")


def test_adjustment_neutral_on_empty_field() -> None:
    r = {"sensor": ""}
    _check(_modality_score_adjustment(r, {"sensor": "satellite"}) == 0.0,
           "neutral on empty sensor field")
    r: dict = {}
    _check(_modality_score_adjustment(r, {"sensor": "satellite"}) == 0.0,
           "neutral on missing sensor field")


def test_adjustment_neutral_on_empty_intent() -> None:
    r = {"sensor": "Satellite", "make": "Observation"}
    _check(_modality_score_adjustment(r, {}) == 0.0,
           "neutral on empty intent")


def test_regression_cs008_cs009_cases() -> None:
    """Regression: the two harness failures this bundle targets."""
    # CS008: "model nitrate climatology" → intent make:model
    intent = _modality_intent_from_query("model nitrate climatology")
    _check(intent == {"make": "model"}, f"CS008 intent: {intent}")
    # WOA is observational
    woa = {"make": "Observation", "sensor": "In-Situ"}
    _check(_modality_score_adjustment(woa, intent) == -4.0,
           "CS008: WOA (observation) penalised under model intent")
    # Darwin is a model
    darwin = {"make": "Model", "sensor": "Blend"}
    _check(_modality_score_adjustment(darwin, intent) == 3.0,
           "CS008: Darwin (model) boosted under model intent")

    # CS009: "in-situ chlorophyll from cruises" → intent sensor:in-situ
    intent2 = _modality_intent_from_query("in-situ chlorophyll from cruises")
    _check(intent2 == {"sensor": "in-situ"}, f"CS009 intent: {intent2}")
    # tblCHL_REP is satellite
    chl_rep = {"make": "Observation", "sensor": "Satellite"}
    _check(_modality_score_adjustment(chl_rep, intent2) == -4.0,
           "CS009: CHL_REP (satellite) penalised under in-situ intent")
    # A hypothetical in-situ chl dataset
    insitu_chl = {"make": "Observation", "sensor": "In-Situ"}
    _check(_modality_score_adjustment(insitu_chl, intent2) == 3.0,
           "CS009: in-situ candidate boosted under in-situ intent")


def main() -> int:
    print("Running modality-intent tests (v223):")
    print()
    test_intent_satellite()
    print()
    test_intent_in_situ_variants()
    print()
    test_intent_model()
    print()
    test_intent_observation()
    print()
    test_intent_assimilation()
    print()
    test_intent_empty_on_ambiguous()
    print()
    test_intent_empty_on_no_modality()
    print()
    test_adjustment_boost_on_match()
    test_adjustment_penalty_on_recognised_opposite()
    test_adjustment_neutral_on_unknown_value()
    test_adjustment_neutral_on_empty_field()
    test_adjustment_neutral_on_empty_intent()
    print()
    test_regression_cs008_cs009_cases()
    print()
    print(f"Summary: {_PASS} passed, {_FAIL} failed")
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
