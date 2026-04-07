from __future__ import annotations
from dataclasses import dataclass
from cmap_agent.config.settings import settings

@dataclass(frozen=True)
class Limits:
    max_inline_rows: int = settings.CMAP_AGENT_MAX_INLINE_ROWS
    max_export_rows: int = settings.CMAP_AGENT_MAX_EXPORT_ROWS
    max_tool_calls: int = settings.CMAP_AGENT_MAX_TOOL_CALLS

LIMITS = Limits()
