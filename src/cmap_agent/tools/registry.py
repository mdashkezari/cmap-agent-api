from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Type

from pydantic import BaseModel

@dataclass
class Tool:
    name: str
    description: str
    args_model: Type[BaseModel]
    fn: Callable[[BaseModel, dict[str, Any]], dict[str, Any]]

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def list_for_prompt(self) -> list[dict[str, Any]]:
        out = []
        for t in self._tools.values():
            out.append({
                "name": t.name,
                "description": t.description,
                "args_schema": t.args_model.model_json_schema(),
            })
        return out
