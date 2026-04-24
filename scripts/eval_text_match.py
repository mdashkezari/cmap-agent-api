"""eval_text_match — shared text normalizer for eval tooling.

PDF text extraction introduces systematic artefacts that break literal
substring matching even when the paper genuinely contains the target
phrase.  The normalizer here folds the common variants into a single
canonical form so that "conceptually the same" matches across the
evaluation harness (``eval_retrieval.py``) and the target verifier
(``verify_eval_targets.py``).

Normalization rules
-------------------
 * Collapse invisible / zero-width characters to a single space
   (NBSP U+00A0, narrow NBSP U+202F, ZWSP U+200B, ZWJ U+200D,
   ZWNJ U+200C, BOM U+FEFF).
 * Unify the two micro-sign codepoints (U+00B5 and U+03BC) to ``u`` so
   that ``µm`` and ``μm`` compare equal.
 * Unify all dash-shaped characters (hyphen, NB hyphen, figure dash,
   en dash, em dash, minus sign) to ASCII ``-``.
 * Collapse whitespace runs to a single space.
 * Lowercase.

These transformations are applied identically to the needle (substring
to find) and the haystack (PDF / chunk text), so comparing the two is
robust to typesetting and extraction variants without introducing false
positives that would not survive the same normalization on both sides.
"""
from __future__ import annotations


_MU_FORMS = "\u00b5\u03bc"
_DASH_FORMS = "\u2010\u2011\u2012\u2013\u2014\u2212"
_INVISIBLE = "\u00a0\u200b\u200c\u200d\ufeff\u202f"


def normalize_for_match(s: str) -> str:
    """Return a canonical form of *s* suitable for substring comparison."""
    if not s:
        return ""
    for ch in _INVISIBLE:
        s = s.replace(ch, " ")
    for ch in _MU_FORMS:
        s = s.replace(ch, "u")
    for ch in _DASH_FORMS:
        s = s.replace(ch, "-")
    s = " ".join(s.split())
    return s.lower()
