from __future__ import annotations
from abc import ABC, abstractmethod
from cmap_agent.llm.types import LLMMessage, LLMResponse

class LLMClient(ABC):
    @abstractmethod
    def complete(self, messages: list[LLMMessage]) -> LLMResponse:
        raise NotImplementedError
