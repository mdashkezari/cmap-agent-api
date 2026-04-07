from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class RateLimitDecision:
    allowed: bool
    retry_after_seconds: int | None = None


class FixedWindowRateLimiter:
    """Very small in-process fixed-window rate limiter.

    This is intentionally simple: it is good enough for a single ECS task.
    If scale out to multiple tasks, use a shared limiter store
    (Redis/ElastiCache) for global consistency.
    """

    def __init__(self, limit: int, window_seconds: int = 60):
        self.limit = int(limit)
        self.window_seconds = int(window_seconds)
        self._state: dict[str, tuple[float, int]] = {}  # key -> (window_start_epoch, count)

    def check(self, key: str) -> RateLimitDecision:
        now = time.time()
        window_start, count = self._state.get(key, (now, 0))

        # Reset window
        if now - window_start >= self.window_seconds:
            window_start, count = now, 0

        if count >= self.limit:
            retry_after = max(1, int(self.window_seconds - (now - window_start)))
            return RateLimitDecision(allowed=False, retry_after_seconds=retry_after)

        self._state[key] = (window_start, count + 1)
        return RateLimitDecision(allowed=True)
