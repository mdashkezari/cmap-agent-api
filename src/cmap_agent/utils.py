"""Shared utility helpers used across the cmap_agent package."""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any


def to_jsonable(obj: Any) -> Any:
    """Recursively convert objects into JSON-serializable equivalents.

    Handles datetime, Decimal, bytes, and arbitrary objects via str() fallback.
    Used at API boundaries to prevent serialization failures from tool outputs.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    try:
        return str(obj)
    except Exception:
        return repr(obj)
