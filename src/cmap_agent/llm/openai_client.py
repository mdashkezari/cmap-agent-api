from __future__ import annotations

from openai import OpenAI
from cmap_agent.config.settings import settings
from cmap_agent.llm.base import LLMClient
from cmap_agent.llm.types import LLMMessage, LLMResponse

class OpenAIClient(LLMClient):
    def __init__(self, model: str):
        if not settings.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set.")
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = model
        # Most OpenAI chat models support JSON mode via response_format.
        # We try it first (to reduce "plan" responses), but fall back gracefully
        # if the selected model/account doesn't support it.
        self._json_mode_supported = True

    def complete(self, messages: list[LLMMessage]) -> LLMResponse:
        payload = dict(
            model=self.model,
            messages=[m.model_dump() for m in messages],
            temperature=0.2,
        )

        if self._json_mode_supported:
            payload["response_format"] = {"type": "json_object"}

        try:
            resp = self.client.chat.completions.create(**payload)
        except Exception:
            # If JSON mode isn't supported, retry once without it.
            if self._json_mode_supported:
                self._json_mode_supported = False
                payload.pop("response_format", None)
                resp = self.client.chat.completions.create(**payload)
            else:
                raise
        msg = resp.choices[0].message.content or ""
        usage = resp.usage.model_dump() if resp.usage else None
        return LLMResponse(content=msg, model=self.model, provider="openai", usage=usage)
