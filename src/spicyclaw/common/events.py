"""Gateway <-> UI event protocol."""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field


class ServerEvent(BaseModel):
    type: Literal[
        "chunk",
        "tool_call",
        "tool_output",
        "tool_end",
        "status",
        "session_update",
        "error",
        "system",
    ]
    session_id: str
    data: dict[str, Any] = {}
    ts: float = Field(default_factory=time.time)


class ClientEvent(BaseModel):
    type: Literal["message", "confirm", "abort", "command"]
    session_id: str
    data: dict[str, Any] = {}
