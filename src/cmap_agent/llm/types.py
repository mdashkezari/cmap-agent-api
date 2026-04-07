from __future__ import annotations
from pydantic import BaseModel

class LLMMessage(BaseModel):
    role: str  # system|user|assistant
    content: str

class LLMResponse(BaseModel):
    content: str
    model: str | None = None
    provider: str | None = None
    usage: dict | None = None
