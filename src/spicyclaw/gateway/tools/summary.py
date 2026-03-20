"""Summary tool — used by context compression to store summaries."""

from __future__ import annotations

from typing import Any

from spicyclaw.common.types import ToolResult
from spicyclaw.gateway.tools.base import Tool


class SummaryTool(Tool):
    name = "summary"
    description = "Submit a summary of completed work. Used during context compression."
    parameters = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The summary text covering what was accomplished",
            },
        },
        "required": ["content"],
    }

    async def execute(self, arguments: dict[str, Any], **kwargs: Any) -> ToolResult:
        content = arguments.get("content", "")
        if not content.strip():
            return ToolResult(output="", error="Empty summary", return_code=1)
        return ToolResult(output=f"Summary recorded: {content[:200]}...")
