"""Stop tool — signals the workloop to stop."""

from __future__ import annotations

from typing import Any

from spicyclaw.common.types import ToolResult
from spicyclaw.gateway.tools.base import Tool


class StopTool(Tool):
    name = "stop"
    description = "Stop execution. Call this when the task is complete or when you need user input."
    parameters = {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Why you are stopping (e.g. 'task complete', 'need user input')",
            },
        },
        "required": ["reason"],
    }

    async def execute(self, arguments: dict[str, Any], **kwargs: Any) -> ToolResult:
        reason = arguments.get("reason", "no reason given")
        return ToolResult(output=f"Stopped: {reason}")
