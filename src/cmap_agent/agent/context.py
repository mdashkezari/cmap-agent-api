from __future__ import annotations

from cmap_agent.agent.prompts import system_prompt
from cmap_agent.tools.registry import ToolRegistry

def build_system_prompt(registry: ToolRegistry) -> str:
    return system_prompt(registry)
