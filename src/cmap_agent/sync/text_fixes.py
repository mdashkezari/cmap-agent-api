"""text_fixes — pure string helpers used by the KB sync pipeline.

These functions intentionally have no external dependencies so that they
can be unit-tested in any environment (no SQL Server driver, no OpenAI
key, no Qdrant, no PyMuPDF).
"""
from __future__ import annotations

import re as _re


# Pattern: digit(s), space, dash/en-dash, space, digits with comma separators.
# The same raw text is produced by two distinct phenomena:
#
#   (a) A MuPDF line-break artefact, where a large number like 1,250,359 has
#       been split mid-digit across a soft line break and re-emerges as
#       "1 – 1, 250, 359".
#   (b) A genuine numeric range written in the source, such as
#       "depths ranged from 1 – 1,250,359 reads per sample".
#
# The two are syntactically identical.  v203 distinguished them only on
# whether the leading digit group repeated at the start of the rest — a
# heuristic that fails for exactly the case the GRUMP paper uses, where
# the range happens to start at 1 and end at a number beginning with 1.
# v205 adds a contextual guard: if the preceding window contains a range
# indicator, the text is treated as a genuine range and returned untouched.
_NUM_BREAK = _re.compile(r"(\d{1,3})\s+[–\-]\s+(\d{1,3}(?:,\s*\d{3})+)")

# Window of preceding characters inspected for range-indicator context.
# 40 chars is wide enough to catch "which ranged from X – Y" phrasings yet
# narrow enough that unrelated earlier words do not trigger it.
_CONTEXT_WINDOW = 40

# Words that signal the number pair is a genuine numeric range.  Word
# boundaries are enforced so "rangefinder" does not match "range".
_RANGE_INDICATORS = _re.compile(
    r"\b(?:"
    r"ranged|range|ranging|ranges|"
    r"between|"
    r"spanning|spans|spanned|"
    r"varied|varying|varies|vary|"
    r"from(?:\s+\w+){0,6}\s+to"        # "from X ... to" up to 6 filler words
    r")\b",
    _re.IGNORECASE,
)


def fix_pdf_number_breaks(text: str) -> str:
    """Repair numbers garbled by soft line-breaks in two-column PDF layouts.

    MuPDF sometimes extracts a large number like 1,250,359 as "1 – 1, 250, 359"
    or "1 - 1, 250, 359" when a line break falls inside the number and an
    en-dash or hyphen appears at the break point.

    This function merges such occurrences back into a single formatted
    integer.  It leaves genuine numeric ranges alone by applying two
    guards, evaluated in order:

      1. Context guard: if a range-indicator word ("ranged", "between",
         "from X to Y", etc.) appears in the ``_CONTEXT_WINDOW`` chars
         preceding the match, the text is treated as a genuine range and
         returned verbatim.  This is the decisive signal in practice —
         methods sections that talk about measurement ranges almost
         always introduce them with one of these words.
      2. Prefix-repeat guard (v203): when no range indicator is present
         and the leading digit group does not repeat at the start of the
         rest, the text is again returned verbatim.  This catches
         unambiguous non-artefact cases such as "2 – 10,000".

    Ambiguity residual: a numeric range introduced with NO range-indicator
    word whose bounds happen to share a leading digit group (e.g. "1 –
    1,250,359" written with no preceding "ranged from") will still be
    merged.  In practice, technical writing virtually always introduces
    such ranges with context words, so this residual is tolerable.
    """

    def _merge(m: "_re.Match") -> str:
        # Context guard — look back at the window of text preceding the match.
        start = m.start()
        window_start = max(0, start - _CONTEXT_WINDOW)
        preceding = text[window_start:start]
        if _RANGE_INDICATORS.search(preceding):
            return m.group(0)

        # Prefix-repeat guard (v203 heuristic, retained as a second line of
        # defence for cases without a range indicator).
        prefix = m.group(1)
        rest_compact = m.group(2).replace(" ", "").replace(",", "")
        if not rest_compact.startswith(prefix):
            return m.group(0)

        try:
            n = int(rest_compact)
        except ValueError:
            return m.group(0)
        return f"{n:,}"

    return _NUM_BREAK.sub(_merge, text)
