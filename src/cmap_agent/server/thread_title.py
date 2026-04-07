from __future__ import annotations

import re


_WS_RE = re.compile(r"\s+")


def make_thread_title(message: str | None, max_len: int = 120) -> str:
    """Create a compact thread title from the first user prompt."""

    if not message:
        return "New chat"

    s = str(message).strip()
    if not s:
        return "New chat"

    # Collapse whitespace/newlines.
    s = _WS_RE.sub(" ", s)

    if len(s) <= max_len:
        return s

    # Prefer to cut at a word boundary.
    cut = s[: max_len + 1]
    i = cut.rfind(" ")
    if i >= int(max_len * 0.6):
        s = s[:i]
    else:
        s = s[:max_len]

    return s.rstrip() + "..."
