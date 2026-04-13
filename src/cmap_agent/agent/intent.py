"""LLM-based structured intent extraction.

A single lightweight LLM call at the start of each /chat turn extracts a
UserIntent from the raw user message and current thread state.  This replaces
the prior approach of regex-scanning the message for keywords, region names,
date patterns, and make/sensor literals.

The extraction uses a fresh, isolated API call so that any JSON-mode fallback
or retry within intent extraction cannot affect the shared LLM client state
used by the main agent loop.

Estimated cost: ~200 input tokens + ~80 output tokens per request.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field, field_validator

log = logging.getLogger(__name__)

_VALID_MAKES = ["Observation", "Model", "Assimilation"]
_VALID_SENSORS = ["in-Situ", "Satellite"]

_EXTRACTION_SYSTEM = """\
Extract structured intent from an oceanographic data query.

Respond with ONLY a JSON object — no prose, no markdown fences.

Schema:
{
  "search_query": "<key scientific terms for dataset/variable search, 2-8 words>",
  "lat1": <southern latitude float or null>,
  "lat2": <northern latitude float or null>,
  "lon1": <western longitude float or null>,
  "lon2": <eastern longitude float or null>,
  "dt1": "<YYYY-MM-DD or null>",
  "dt2": "<YYYY-MM-DD or null>",
  "make": "<Observation|Model|Assimilation or null>",
  "sensor": "<in-Situ|Satellite or null>",
  "action": "<map|time_series|download|summarize|colocalize|chat or null>",
  "is_followup": <true if this refines the prior request, false if new topic>,
  "surface_only": <true if user wants near-surface/surface data only>
}

Rules:
- search_query: distil the scientific variable or dataset topic only; omit verbs,
  regions, dates, and sensor/make qualifiers. Use the most specific scientific term
  the user provided — do not substitute or expand to synonyms. If the user says
  "nutrients", use "nutrients". If they say "iron", use "iron". If they say "POC",
  use "POC". The KB and SQL searches will handle semantic matching.
- Spatial: use null when no region is mentioned. For named regions use these bboxes
  (lat1, lat2, lon1, lon2):
  global=(-90,90,-180,180), southern_ocean=(-90,-60,-180,180),
  north_atlantic=(0,60,-76,-6), south_atlantic=(-60,0,-68,20),
  north_pacific=(0,66,120,-98), south_pacific=(-60,0,147,-68),
  indian_ocean=(-60,30,20,120), mediterranean=(31,45,-1,36),
  gulf_of_mexico=(20,30,-97,-83), arctic=(66,90,-180,180).
- Dates: resolve "last January" / "summer 2019" / "Q1 2022" to ISO date ranges.
  Return null when no date is mentioned.
- make: set only when the user explicitly requests a data type (model, satellite, etc.).
  Ignore mentions of "model" in other contexts.
- sensor: same rule as make.
- action: "chat" when the message is conversational with no data request.
  "summarize" when the user asks to list, describe, summarize, or explore datasets
  (e.g. "what BATS datasets exist", "summarize chlorophyll datasets", "what nutrient
  datasets cover this region"). "map" when a plot/map is requested.
- search_query for "summarize" actions: use the dataset name or acronym directly
  (e.g. "BATS" for Bermuda Atlantic Time-series, "HOT" for Hawaii Ocean Time-series)
  rather than variable names, so catalog.dataset_summary can find all matching datasets.
- is_followup: true when the message clearly refers to or modifies the prior request.
"""


class UserIntent(BaseModel):
    """Structured intent extracted from a single user message."""

    search_query: str = Field(default="")

    @field_validator("search_query", mode="before")
    @classmethod
    def _coerce_search_query(cls, v: object) -> str:
        """Coerce None to empty string — LLM sometimes returns null for short replies."""
        return "" if v is None else str(v)
    lat1: float | None = None
    lat2: float | None = None
    lon1: float | None = None
    lon2: float | None = None
    dt1: str | None = None
    dt2: str | None = None
    make: str | None = None
    sensor: str | None = None
    action: str | None = None
    is_followup: bool = False
    surface_only: bool = False

    def has_bounds(self) -> bool:
        return all(v is not None for v in (self.lat1, self.lat2, self.lon1, self.lon2))

    def bounds_dict(self) -> dict[str, float] | None:
        if not self.has_bounds():
            return None
        return {
            "lat1": self.lat1, "lat2": self.lat2,
            "lon1": self.lon1, "lon2": self.lon2,
        }


def _call_llm_for_intent(llm: Any, messages: list) -> str:
    """Make a single isolated LLM call for intent extraction.

    For OpenAI clients, a fresh per-call JSON-mode attempt is made without
    touching any shared instance state.  For other clients, the normal
    complete() path is used.
    """
    from cmap_agent.llm.openai_client import OpenAIClient

    if isinstance(llm, OpenAIClient):
        # Call the OpenAI API directly so the result cannot affect the shared
        # client instance's state (e.g. response_format fallback flags).
        payload = dict(
            model=llm.model,
            messages=[m.model_dump() for m in messages],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        try:
            resp = llm.client.chat.completions.create(**payload)
        except Exception:
            payload.pop("response_format", None)
            resp = llm.client.chat.completions.create(**payload)
        return resp.choices[0].message.content or ""

    # Anthropic and any other provider — use normal complete()
    from cmap_agent.llm.types import LLMMessage as _LLMMessage
    resp = llm.complete(messages)
    return resp.content or ""


def extract_intent(
    llm: Any,
    user_message: str,
    thread_state: Any,
) -> UserIntent:
    """Extract a UserIntent from the user message via a single LLM call.

    Falls back to a minimal UserIntent on any error so the agent can always
    continue — intent extraction failure is never fatal.
    """
    from cmap_agent.llm.types import LLMMessage

    state_lines: list[str] = []
    if thread_state is not None:
        if thread_state.confirmed_table:
            state_lines.append(f"Prior confirmed dataset: {thread_state.confirmed_table}")
        if thread_state.has_bounds():
            state_lines.append(
                f"Prior bounds: lat {thread_state.lat1}..{thread_state.lat2}, "
                f"lon {thread_state.lon1}..{thread_state.lon2}"
            )
        if thread_state.dt1:
            state_lines.append(
                f"Prior date range: {thread_state.dt1} to {thread_state.dt2 or thread_state.dt1}"
            )
        if thread_state.last_action:
            state_lines.append(f"Prior action: {thread_state.last_action}")

    state_block = ""
    if state_lines:
        state_block = (
            "\n\nPrior conversation state (for resolving follow-ups):\n"
            + "\n".join(state_lines)
        )

    messages = [
        LLMMessage(role="system", content=_EXTRACTION_SYSTEM),
        LLMMessage(role="user", content=f"User message: {user_message}{state_block}"),
    ]

    try:
        raw = _call_llm_for_intent(llm, messages).strip()
        # Strip markdown fences if the model adds them despite the instruction.
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = lines[1:] if lines else lines
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            raw = "\n".join(lines).strip()
        obj = json.loads(raw)
        intent = UserIntent.model_validate(obj)
        if intent.make and intent.make not in _VALID_MAKES:
            intent.make = None
        if intent.sensor and intent.sensor not in _VALID_SENSORS:
            intent.sensor = None
        return intent
    except Exception as exc:
        log.warning("Intent extraction failed (%s); using minimal intent.", exc)
        return UserIntent(search_query=user_message[:120])
