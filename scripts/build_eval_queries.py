#!/usr/bin/env python3
"""build_eval_queries.py — merge prompt/answer files into harness schema.

Takes one or more ``(prompts, answers)`` pairs that follow the upstream
evaluation format (prompt files are a list of {p_id, dataset, ref, prompt};
answer files are a list of {p_id, answer}) and produces the
``retrieval_eval_queries.json`` file the harness consumes.

For each entry the tool:

  * Joins prompt and answer on ``p_id``.
  * Extracts candidate target substrings from the answer: named entities,
    multi-word capitalised phrases, numeric-with-unit expressions, code
    identifiers, and DOI/accession-like tokens.  These are substrings that
    (when they survive) should appear verbatim in the source PDF text —
    not prose sentences, because those rarely round-trip through PDF
    extraction.
  * Marks the resulting entry as ``verified=false`` so the harness and
    future tooling know the substrings have not yet been round-tripped
    through the actual KB text.

A later step (verify_eval_targets.py, not provided here) can open each
PDF, look up which candidate substrings actually appear in the extracted
text, drop those that do not, and flip ``verified=true`` on the survivors.

Usage
-----
    python scripts/build_eval_queries.py \
        --pair prompts_eval.json answers.json \
        --pair prompts_eval2.json answers2.json \
        --out scripts/retrieval_eval_queries.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Shape inference
# ---------------------------------------------------------------------------

_NUMERIC_CUE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|m|km|µm|mm|cm|°C|nm|bp|L|ml|pM|nM)\b", re.IGNORECASE)
_SEQUENCE_CUE = re.compile(r"\b[ACGTYRMKSWN]{12,}\b")
_DOI_CUE = re.compile(r"10\.\d{4,9}/")
_CITE_CUE = re.compile(r"\b(cite|citation|acknowledg|reference)", re.IGNORECASE)
_COMPARE_CUE = re.compile(r"\b(vs|versus|compared|compare|differ|difference)\b", re.IGNORECASE)
_METHOD_CUE = re.compile(r"\b(method|protocol|pipeline|software|model|primer|kit|instrument|filter)\b", re.IGNORECASE)
_COUNT_CUE = re.compile(r"\b\d{2,}\b")


def infer_shape(prompt: str, answer: str) -> str:
    """Return a coarse tag describing the question shape."""
    combined = f"{prompt}\n{answer}"
    if _SEQUENCE_CUE.search(combined):
        return "sequence_literal"
    if _DOI_CUE.search(combined):
        return "citation_attribution"
    if _CITE_CUE.search(prompt):
        return "citation_attribution"
    if _NUMERIC_CUE.search(combined):
        return "numeric_threshold"
    if _COMPARE_CUE.search(prompt):
        return "comparison"
    if _METHOD_CUE.search(prompt):
        return "method_description"
    if _COUNT_CUE.search(answer):
        return "numeric_count"
    return "descriptive"


# ---------------------------------------------------------------------------
# Substring candidate extraction
# ---------------------------------------------------------------------------

# Regexes are ordered by specificity; more specific patterns produce stronger
# candidates.  All extracted substrings retain their original casing so the
# harness's case-insensitive match works but published entities read
# naturally when inspected by a human.
_PAT_SEQUENCE = re.compile(r"\b[ACGTYRMKSWN]{12,}\b")
_PAT_SCI_NAME = re.compile(r"\b[A-Z][a-z]{2,}\s+[a-z][a-z]{2,}\b")
_PAT_CAPPED_PHRASE = re.compile(r"\b(?:[A-Z][a-zA-Z]{2,}\s+){1,5}[A-Z][a-zA-Z]{2,}\b")
_PAT_ACRONYM_WITH_VER = re.compile(r"\b[A-Z]{2,}[A-Za-z0-9\-\._]*(?:\s+v\.?\s*\d+(?:\.\d+)?|\s*-?\s*v\d+)?\b")
_PAT_NUM_UNIT = re.compile(
    r"\b(?:approximately\s+|about\s+|over\s+|more than\s+)?"
    r"(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+(?:\.[0-9]+)?)"
    # Unit / counting noun.  `\b` does not anchor after punctuation-unit
    # characters like ``%``, so units are grouped into two alternations and
    # the optional word-boundary is only required for alphabetic units.
    r"\s*(?:"
    r"%|°C|°"   # punctuation-shaped units — no trailing \b
    r"|(?:µm|μm|nm|mm|cm|km|m|bp|kb|L|ml|mL|pM|nM|µM|μM|"
    r"days|hours|minutes|seconds|years|months|weeks|"
    r"samples|records|strains|sites|cruises|transcripts|profiles|ASVs|"
    r"depths|stations|time points|time-points|tracers|scaffolds|"
    r"subclades|filters|libraries|cells)\b"
    r")",
    re.IGNORECASE,
)
_PAT_RANGE = re.compile(
    r"\b[0-9]{2,4}\s*[\u2013\-]\s*[0-9]{2,4}\b"  # e.g. 1992-2011, 2003-2020
)
_PAT_IDENT_CODE = re.compile(r"\b[A-Z]{1,6}[0-9]{2,6}\b")  # P744, GA02, 515Y, etc.

# Bare large integers (>=1000, or >=100 with thousands separators).  These
# are distinctive enough to be useful answer anchors even without an
# attached unit: 5000, 1,250,359, 136603, etc.  Smaller integers are too
# common to be useful substring targets.
_PAT_BARE_LARGE_INT = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:[1-9][0-9]{0,2}(?:,[0-9]{3})+"        # 1,000 / 1,250,359
    r"|[1-9][0-9]{3,})"                         # 1000 / 136603
    r"(?![A-Za-z0-9])"
)

_STOP_PHRASES = {
    "The authors", "The study", "The dataset", "The paper", "The collection",
    "The description", "The analysis", "This paper", "This study",
    "North Atlantic", "South Atlantic", "North Pacific", "South Pacific",
    "Atlantic Ocean", "Pacific Ocean", "Indian Ocean", "Southern Ocean",
    "Arctic Ocean", "DNA extraction", "DNA extract", "PCR amplification",
    "Open Access", "Data Availability", "Author contributions",
    "Code availability", "Competing interests", "Background Summary",
}

# Determiners / auxiliary leads that make a candidate prose-starter-shaped
# rather than a distinctive entity name.  These phrases would match in
# almost any chunk of any paper.
_PROSE_LEAD_TOKENS = {
    "the", "this", "that", "these", "those", "a", "an",
    "because", "although", "while", "since", "after", "before",
    "when", "where", "which",
    "for", "in", "on", "at", "by", "of", "from", "to", "with",
    "is", "was", "are", "were", "has", "have", "had", "be", "been",
    "can", "could", "should", "would", "may", "might", "will",
    "all", "some", "most", "many", "few",
}

# Common verb tokens.  If any of the first three tokens of a candidate
# is a verb, the candidate is almost certainly a prose fragment
# ("Samples were", "Water molecules contribute", "Scattering had less",
# "Users are expected").  Real named entities — "Southern Ocean",
# "Station ALOHA", "Lagrangian trajectories", "SAR11 Clade I" — do not
# contain these verb forms in their first few words.
_PROSE_VERB_TOKENS = {
    # to-be
    "is", "was", "are", "were", "be", "been", "being",
    # to-have
    "has", "have", "had", "having",
    # to-do
    "do", "does", "did", "done",
    # modals
    "can", "could", "should", "would", "may", "might", "will", "shall",
    # generic action verbs often starting methods/results sentences
    "found", "finds", "report", "reports", "reported", "reporting",
    "describes", "described", "describing",
    "says", "said", "saying",
    "shows", "showed", "showing", "shown",
    "includes", "included", "including",
    "contains", "contained", "containing",
    "uses", "used", "using",
    "extends", "extended", "extending",
    "yields", "yielded", "yielding",
    "grows", "grew", "grown", "growing",
    "contribute", "contributes", "contributed", "contributing",
    "needed", "needs", "need",
    "expected", "expects", "expect",
    "investigated", "investigates", "investigate",
    "corrected", "correct", "corrects",
    "proposed", "proposes", "propose",
    "consist", "consists", "consisted", "consisting",
    "argue", "argues", "argued",
    "exhibit", "exhibits", "exhibited",
    "amplify", "amplifies", "amplified",
}


def _looks_like_prose_starter(s: str) -> bool:
    """True if *s* looks like a sentence fragment rather than a named entity.

    Two signals:
      * The first token is one of ``_PROSE_LEAD_TOKENS`` — "The study",
        "For the detected", "Because the paper".
      * Any of the first three tokens is one of ``_PROSE_VERB_TOKENS`` —
        "Samples were collected", "Water molecules contribute",
        "Users are expected".

    Real named entities ("Station ALOHA", "Lagrangian coherent structure",
    "SAR11 Clade I", "Odontella aurita") contain neither pattern in their
    first three words.
    """
    parts = s.split()
    if not parts:
        return True
    if parts[0].lower() in _PROSE_LEAD_TOKENS:
        return True
    for tok in parts[:3]:
        if tok.lower() in _PROSE_VERB_TOKENS:
            return True
    return False


def _clean(s: str) -> str:
    return s.strip(" ,.;:()[]\"'").replace("\u2019", "'")


def _looks_noisy(s: str) -> bool:
    if len(s) < 4:
        return True
    if s.lower() in {"the", "a", "an", "and", "of", "for", "in", "on", "to"}:
        return True
    if s in _STOP_PHRASES:
        return True
    # Skip generic words written with a capital.
    if s.istitle() and len(s.split()) == 1 and s.lower() in {
        "yes", "no", "several", "reduced", "elevated", "higher", "lower",
        "bacteria", "archaea", "eukaryotes", "surface", "samples",
    }:
        return True
    return False


def extract_candidates(answer: str) -> list[str]:
    """Pull distinctive substrings out of *answer*.

    Returned items are ordered by specificity: exotic strings (nucleotide
    sequences, numeric+unit, named species) come first, looser candidates
    (generic capped phrases) come last.  Duplicates are removed while
    preserving order.
    """
    candidates: list[str] = []

    # Highest specificity: IUPAC nucleotide strings.
    for m in _PAT_SEQUENCE.finditer(answer):
        candidates.append(m.group(0))

    # Numeric + unit.
    for m in _PAT_NUM_UNIT.finditer(answer):
        t = _clean(m.group(0))
        if t:
            candidates.append(t)

    # Year ranges / numeric ranges.
    for m in _PAT_RANGE.finditer(answer):
        t = _clean(m.group(0))
        if t:
            candidates.append(t)

    # Identifier codes like P744, GA02, 515Y.
    for m in _PAT_IDENT_CODE.finditer(answer):
        t = _clean(m.group(0))
        if t and t not in _STOP_PHRASES:
            candidates.append(t)

    # Bare large integers — distinctive numeric answers without a unit.
    for m in _PAT_BARE_LARGE_INT.finditer(answer):
        t = _clean(m.group(0))
        if t:
            candidates.append(t)

    # Scientific binomials (Genus species).
    for m in _PAT_SCI_NAME.finditer(answer):
        t = _clean(m.group(0))
        if t and not _looks_noisy(t) and not _looks_like_prose_starter(t):
            candidates.append(t)

    # Multi-word capitalised phrases — weaker signal but often catch proper
    # nouns like institution names and programme titles.
    for m in _PAT_CAPPED_PHRASE.finditer(answer):
        t = _clean(m.group(0))
        if t and not _looks_noisy(t) and not _looks_like_prose_starter(t):
            candidates.append(t)

    # Acronyms with optional version suffix.
    for m in _PAT_ACRONYM_WITH_VER.finditer(answer):
        t = _clean(m.group(0))
        if (len(t) >= 3 and not _looks_noisy(t)
                and not _looks_like_prose_starter(t)):
            candidates.append(t)

    # De-duplicate while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(c)

    # Cap: too many candidates dilute the any_of signal.  Keep the most
    # specific (earlier) entries.
    return out[:10]


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def _ref_to_short_name(dataset: str) -> str:
    """Use the dataset name supplied in the prompt file as-is."""
    return dataset or ""


def merge_pair(prompts: list[dict], answers: list[dict]) -> list[dict]:
    """Join prompts and answers by ``p_id`` and build harness entries.

    Each answer dict may optionally carry a ``targets`` list of
    verbatim substrings that should be used as ``any_of`` exactly as
    written, bypassing the regex extractor.  This is the escape hatch
    for answers whose distinctive content is lowercase single words
    (e.g. "Pedinellaceae", "allometric", "oligotrophic") or short
    phrases that the general regex patterns cannot safely detect
    without introducing false positives across the whole set.

    If ``targets`` is present and non-empty, the extractor is skipped.
    If it's absent or empty, candidates are extracted from the prose.
    """
    a_by_id = {a["p_id"]: a for a in answers}
    out: list[dict] = []
    unmatched: list[str] = []
    for p in prompts:
        pid = p["p_id"]
        if pid not in a_by_id:
            unmatched.append(pid)
            continue
        ans_rec = a_by_id[pid]
        answer = ans_rec["answer"]
        explicit_targets = ans_rec.get("targets") or []
        if explicit_targets:
            any_of = [str(t) for t in explicit_targets if str(t).strip()]
            note = ("Hand-curated targets from answer file "
                    "(extractor bypassed).")
        else:
            any_of = extract_candidates(answer)
            note = ("Auto-generated from prompt+answer merge. "
                    "Substrings are candidate, not PDF-verified.")
        out.append({
            "id": pid,
            "question": p["prompt"],
            "any_of": any_of,
            "short_name": _ref_to_short_name(p.get("dataset", "")),
            # Answer files may set an explicit "shape" field to override the
            # regex-based inference.  Useful when the auto-inference picks
            # a tag that doesn't match the conceptual class of the question
            # (e.g. "what MiSeq version?" gets tagged numeric_threshold
            # because "2 x 300" matches the numeric pattern, but the user
            # is really asking a sequence_literal question).
            "shape": ans_rec.get("shape") or infer_shape(p["prompt"], answer),
            "notes": note,
            "ref_file": p.get("ref", ""),
            "verified": False,
            "answer_prose": answer,
        })
    if unmatched:
        print(f"  [warn] {len(unmatched)} prompts had no matching answer: {unmatched[:5]}...")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--pair",
        nargs=2,
        metavar=("PROMPTS", "ANSWERS"),
        action="append",
        required=True,
        help="A (prompts.json, answers.json) pair.  May be repeated.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Destination JSON file (harness schema).",
    )
    args = ap.parse_args()

    merged: list[dict] = []
    seen_ids: set[str] = set()
    for prompts_path, answers_path in args.pair:
        print(f"[load] {prompts_path} + {answers_path}")
        prompts = json.loads(Path(prompts_path).read_text())
        answers = json.loads(Path(answers_path).read_text())
        pair_entries = merge_pair(prompts, answers)
        for entry in pair_entries:
            if entry["id"] in seen_ids:
                print(f"  [skip] duplicate id {entry['id']!r}")
                continue
            seen_ids.add(entry["id"])
            merged.append(entry)

    doc = {
        "_comment": [
            "Reference prompt bank for retrieval evaluation.",
            "",
            "Each entry has:",
            "  id            unique slug",
            "  question      natural-language user query",
            "  any_of        candidate substrings; a retrieved chunk counts as a",
            "                hit if it contains any of them (case-insensitive,",
            "                with unicode+whitespace normalization applied)",
            "  short_name    dataset short name (for grouping by source paper)",
            "  shape         coarse question-shape tag",
            "  ref_file      the PDF / txt filename in the reference bank",
            "  verified      False until the any_of substrings have been",
            "                round-tripped through the PDF extraction",
            "  notes         how any_of was populated",
            "  answer_prose  the human reference answer (not used for scoring,",
            "                kept so future tooling can regenerate substrings)",
            "",
            "Answer files may optionally carry a 'targets' list on each entry.",
            "When present and non-empty, those strings are used verbatim as",
            "any_of and the regex extractor is bypassed.  Use this for answers",
            "whose distinctive content is lowercase single words or short",
            "phrases the general patterns cannot safely detect.",
            "",
            "Rebuild with:",
            "  python scripts/build_eval_queries.py \\",
            "    --pair scripts/eval_prompts/prompts_eval.json "
            "scripts/eval_prompts/answers.json \\",
            "    --pair scripts/eval_prompts/prompts_eval2.json "
            "scripts/eval_prompts/answers2.json \\",
            "    --pair scripts/eval_prompts/prompts_eval_manual.json "
            "scripts/eval_prompts/answers_manual.json \\",
            "    --pair scripts/eval_prompts/prompts_eval_hand.json "
            "scripts/eval_prompts/answers_hand.json \\",
            "    --out scripts/retrieval_eval_queries.json",
        ],
        "queries": merged,
    }
    args.out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")

    # Summary.
    from collections import Counter
    by_shape = Counter(e["shape"] for e in merged)
    by_ds = Counter(e["short_name"] for e in merged)
    print()
    print(f"[ok] wrote {len(merged)} entries to {args.out}")
    print(f"     {sum(1 for e in merged if e['any_of'])} entries have at least one candidate substring")
    print(f"     {sum(1 for e in merged if not e['any_of'])} entries have none (answer was too prose-heavy)")
    print()
    print("By question shape:")
    for shape, n in by_shape.most_common():
        print(f"  {shape:<22} {n}")
    print()
    print("By dataset short name (top 10):")
    for ds, n in by_ds.most_common(10):
        print(f"  {ds:<40} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
