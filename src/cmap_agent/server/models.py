from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Literal
from uuid import UUID
from datetime import datetime

class LLMConfig(BaseModel):
    provider: Literal["openai","anthropic"] = "openai"
    model: str = "gpt-4.1-mini"

class ChatOptions(BaseModel):
    return_code: bool = True
    max_tool_calls: int | None = None

class ChatRequest(BaseModel):
    # user_id is optional when authenticating via X-API-Key / JWT.
    # For backward compatibility, defaults to 0.
    user_id: int = 0
    thread_id: UUID | None = None
    message: str
    llm: LLMConfig = Field(default_factory=LLMConfig)
    options: ChatOptions = Field(default_factory=ChatOptions)

class ChatResponse(BaseModel):
    thread_id: str
    assistant_message: str
    code: str | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)


class ThreadItem(BaseModel):
    thread_id: UUID
    title: str
    created_at: datetime
    updated_at: datetime
    last_message: str | None = None
    last_role: str | None = None
    summary: str | None = None


class ThreadListResponse(BaseModel):
    threads: list[ThreadItem] = Field(default_factory=list)


class ThreadMessageItem(BaseModel):
    message_id: int
    role: str
    content: str
    created_at: datetime


class ThreadMessagesResponse(BaseModel):
    thread_id: UUID
    title: str | None = None
    messages: list[ThreadMessageItem] = Field(default_factory=list)
