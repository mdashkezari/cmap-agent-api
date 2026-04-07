from __future__ import annotations

from typing import Any


class ToolInputError(Exception):
    """A structured error intended to be shown to the LLM.

    Use this when a tool fails due to user/agent input (e.g., unknown table/variable),
    and include suggestions that help the agent recover.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_input",
        details: dict[str, Any] | None = None,
        suggestions: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code or "invalid_input")
        self.message = str(message or "")
        self.details = details or {}
        self.suggestions = suggestions or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "suggestions": self.suggestions,
        }
