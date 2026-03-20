"""Shared Pydantic models."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SessionStatus(str, Enum):
    THINKING = "thinking"
    EXECUTING = "executing"
    STOPPED = "stopped"
    PAUSED = "paused"  # step mode: waiting for user confirm


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def to_openai(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role.value}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls:
            d["tool_calls"] = [tc.to_openai() for tc in self.tool_calls]
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            d["name"] = self.name
        return d


class ToolCall(BaseModel):
    id: str
    function_name: str
    arguments: str  # JSON string

    def to_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.function_name,
                "arguments": self.arguments,
            },
        }


class ToolResult(BaseModel):
    output: str
    error: str = ""
    return_code: int = 0
    truncated: bool = False


class SessionMeta(BaseModel):
    id: str
    title: str = "New Session"
    status: SessionStatus = SessionStatus.STOPPED
    model: str = ""
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    token_used: int = 0
