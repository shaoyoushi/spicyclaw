"""Tool base class, registry, and common parameters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any

from spicyclaw.common.types import ToolResult

# Every tool call must include these parameters for progress tracking
COMMON_PARAMS = {
    "work_node": {
        "type": "string",
        "description": "当前正在执行的工作节点编号，如 '2.3.1'",
    },
    "next_step": {
        "type": "string",
        "description": "下一步打算做什么",
    },
}

COMMON_REQUIRED = ["work_node", "next_step"]


class Tool(ABC):
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for tool-specific params

    def to_openai_tool(self) -> dict[str, Any]:
        """Generate OpenAI tools format with common params injected."""
        params = deepcopy(self.parameters)
        params.setdefault("properties", {}).update(COMMON_PARAMS)
        required = list(params.get("required", []))
        for r in COMMON_REQUIRED:
            if r not in required:
                required.append(r)
        params["required"] = required
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": params,
            },
        }

    @abstractmethod
    async def execute(self, arguments: dict[str, Any], **kwargs: Any) -> ToolResult:
        """Execute the tool. Common params (work_node, next_step) are
        already extracted by the workloop before calling this."""
        ...


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def to_openai_tools(self) -> list[dict[str, Any]]:
        return [t.to_openai_tool() for t in self._tools.values()]

    @property
    def names(self) -> list[str]:
        return list(self._tools.keys())
