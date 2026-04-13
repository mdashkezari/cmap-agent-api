"""Persistent per-thread agent state.

ThreadState is stored as a JSON blob in agent.Threads.AgentState and loaded
at the start of each /chat request.  A NULL column is treated as a fresh
ThreadState(), so existing threads continue working without any data migration.

Fields are intentionally flat and JSON-serializable so the storage layer
requires no schema knowledge beyond NVARCHAR(MAX).
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel


class ThreadState(BaseModel):
    """Structured state carried across turns for a single conversation thread.

    Replaces the prior approach of reconstructing intent by regex-scanning
    the full conversation history on every request.
    """

    # Dataset confirmed by the user (table name, e.g. "tblSST_AVHRR_OI_NRT")
    confirmed_table: str | None = None

    # Variable confirmed for the confirmed table (e.g. "sst")
    confirmed_variable: str | None = None

    # Named region the user last requested (e.g. "north atlantic")
    region_name: str | None = None

    # Spatial bounds (degrees).  All four must be non-None to be usable.
    lat1: float | None = None
    lat2: float | None = None
    lon1: float | None = None
    lon2: float | None = None

    # Temporal bounds (ISO date strings, e.g. "2020-01-01")
    dt1: str | None = None
    dt2: str | None = None

    # Dataset make / sensor filter last inferred from user intent
    make: str | None = None    # e.g. "Observation", "Model", "Assimilation"
    sensor: str | None = None  # e.g. "in-Situ", "Satellite"

    # Pending dataset-confirmation step waiting for user reply.
    # Stored as a plain dict so it survives JSON round-trips without importing
    # runner internals into this module.
    pending_confirmation: dict[str, Any] | None = None

    # Last action type performed (used to carry action context into follow-ups)
    last_action: str | None = None  # "map" | "time_series" | "download" | ...

    # Whether the user requested surface/near-surface data
    surface: bool = False



    # Best catalog candidates from the most recent successful catalog search.
    # Persisted across turns so the confirmation guard can use them without
    # re-running searches that may return different (worse) results.
    last_catalog_results: list[dict] | None = None

    # ------------------------------------------------------------------ #
    # Serialization helpers                                                #
    # ------------------------------------------------------------------ #

    def to_json(self) -> str:
        """Serialize to a JSON string suitable for NVARCHAR(MAX) storage."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, raw: str | None) -> "ThreadState":
        """Deserialize from a JSON string.  Returns a blank state on None or parse error."""
        if not raw:
            return cls()
        try:
            return cls.model_validate_json(raw)
        except Exception:
            return cls()

    def has_bounds(self) -> bool:
        """Return True when all four spatial bounds are populated."""
        return all(v is not None for v in (self.lat1, self.lat2, self.lon1, self.lon2))

    def bounds_dict(self) -> dict[str, float] | None:
        """Return spatial bounds as a plain dict, or None when incomplete."""
        if not self.has_bounds():
            return None
        return {"lat1": self.lat1, "lat2": self.lat2, "lon1": self.lon1, "lon2": self.lon2}

    def update_from_intent(self, intent: "UserIntent") -> None:  # type: ignore[name-defined]
        """Merge a UserIntent into this state, preserving fields the intent left None."""
        from cmap_agent.agent.intent import UserIntent  # local import avoids circular deps

        if not isinstance(intent, UserIntent):
            return
        if intent.lat1 is not None:
            self.lat1 = intent.lat1
        if intent.lat2 is not None:
            self.lat2 = intent.lat2
        if intent.lon1 is not None:
            self.lon1 = intent.lon1
        if intent.lon2 is not None:
            self.lon2 = intent.lon2
        if intent.dt1 is not None:
            self.dt1 = intent.dt1
        if intent.dt2 is not None:
            self.dt2 = intent.dt2
        if intent.make is not None:
            self.make = intent.make
        if intent.sensor is not None:
            self.sensor = intent.sensor
        if intent.action is not None:
            self.last_action = intent.action
        if intent.surface_only:
            self.surface = True
        # A new substantive request resets the confirmed dataset unless the
        # intent explicitly refers to the same table.
        if not intent.is_followup:
            self.confirmed_table = None
            self.confirmed_variable = None
            self.pending_confirmation = None
