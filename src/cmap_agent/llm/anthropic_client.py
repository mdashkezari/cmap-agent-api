from __future__ import annotations

from anthropic import Anthropic
from cmap_agent.config.settings import settings
from cmap_agent.llm.base import LLMClient
from cmap_agent.llm.types import LLMMessage, LLMResponse

class AnthropicClient(LLMClient):
    def __init__(self, model: str):
        if not settings.ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY not set.")
        self.client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.model = model

    def complete(self, messages: list[LLMMessage]) -> LLMResponse:
        # Anthropic expects system separately; collapse here:
        system = ""
        msgs = []
        for m in messages:
            if m.role == "system":
                system += m.content + "\n"
            else:
                msgs.append({"role": m.role, "content": m.content})
        resp = self.client.messages.create(
            model=self.model,
            system=system.strip() or None,
            messages=msgs,
            max_tokens=1200,
            temperature=0.2,
        )
        text = ""
        for block in resp.content:
            if block.type == "text":
                text += block.text
        usage = {"input_tokens": getattr(resp.usage, "input_tokens", None), "output_tokens": getattr(resp.usage, "output_tokens", None)}
        return LLMResponse(content=text, model=self.model, provider="anthropic", usage=usage)
