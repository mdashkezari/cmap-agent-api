from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class KBChunk:
    id: str
    text: str
    metadata: dict[str, Any]
