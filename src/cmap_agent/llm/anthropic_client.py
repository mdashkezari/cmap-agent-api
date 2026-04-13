"""Anthropic Claude LLM client.

Key differences from the OpenAI client:
- The Anthropic API accepts exactly one top-level ``system`` string.
  All subsequent role="system" messages injected mid-loop by the runner are
  converted to role="user" with a ``[System note]:`` prefix so that steering
  instructions remain visible to the model in conversation order.
- ``max_tokens`` is set to 8192 to accommodate verbose JSON final responses
  that include assistant_message, pycmap code snippets, and artifact URLs.
"""
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
        """Send messages to the Anthropic API.

        The first role="system" message becomes the Anthropic ``system``
        parameter.  Any additional role="system" messages are converted to
        role="user" with a ``[System note]:`` prefix so that mid-loop steering
        instructions injected by the runner remain in conversation order rather
        than being collapsed into the initial system prompt.
        """
        system = ""
        msgs: list[dict] = []
        first_system_consumed = False

        for m in messages:
            if m.role == "system" and not first_system_consumed:
                system = m.content
                first_system_consumed = True
            elif m.role == "system":
                # Mid-loop steering injection — surface as a user turn so it
                # appears in the correct position in the conversation.
                msgs.append({"role": "user", "content": f"[System note]: {m.content}"})
            else:
                msgs.append({"role": m.role, "content": m.content})

        resp = self.client.messages.create(
            model=self.model,
            system=system.strip() or None,
            messages=msgs,
            max_tokens=8192,
            temperature=0.0,
        )
        text = ""
        for block in resp.content:
            if block.type == "text":
                text += block.text
        usage = {
            "input_tokens": getattr(resp.usage, "input_tokens", None),
            "output_tokens": getattr(resp.usage, "output_tokens", None),
        }
        return LLMResponse(content=text, model=self.model, provider="anthropic", usage=usage)
