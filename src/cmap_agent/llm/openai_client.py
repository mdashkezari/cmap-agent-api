"""OpenAI LLM client.

JSON mode state is tracked per-call (not per-instance) so that a failed
JSON-mode attempt in one call (e.g. intent extraction) does not permanently
disable JSON mode for subsequent calls within the same request.
"""
from __future__ import annotations

from openai import OpenAI
from cmap_agent.config.settings import settings
from cmap_agent.llm.base import LLMClient
from cmap_agent.llm.types import LLMMessage, LLMResponse


class OpenAIClient(LLMClient):
    def __init__(self, model: str):
        if not settings.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set.")
        self.client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL or None,
            organization=settings.OPENAI_ORG or None,
        )
        self.model = model

    def complete(self, messages: list[LLMMessage]) -> LLMResponse:
        """Send messages to the OpenAI API.

        JSON mode is attempted first and silently disabled for this call only
        when the model or account does not support it.  The instance-level
        flag has been removed so that one failed call does not affect others.
        """
        payload = dict(
            model=self.model,
            messages=[m.model_dump() for m in messages],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        try:
            resp = self.client.chat.completions.create(**payload)
        except Exception:
            # JSON mode unsupported for this model/account — retry without it.
            payload.pop("response_format", None)
            resp = self.client.chat.completions.create(**payload)

        msg = resp.choices[0].message.content or ""
        usage = resp.usage.model_dump() if resp.usage else None
        return LLMResponse(content=msg, model=self.model, provider="openai", usage=usage)
