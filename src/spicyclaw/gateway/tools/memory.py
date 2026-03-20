"""Memory tool — read/write session-level memory files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from spicyclaw.common.types import ToolResult
from spicyclaw.gateway.tools.base import Tool

logger = logging.getLogger(__name__)


class MemoryReadTool(Tool):
    name = "memory_read"
    description = "Read a memory file from the session's memory directory."
    parameters = {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "Name of the memory file to read (e.g. 'notes.md')",
            },
        },
        "required": ["filename"],
    }

    async def execute(self, arguments: dict[str, Any], **kwargs: Any) -> ToolResult:
        filename = arguments.get("filename", "")
        cwd: Path | None = kwargs.get("cwd")
        if not cwd or not filename:
            return ToolResult(output="", error="Missing filename or cwd", return_code=1)

        memory_dir = cwd / "memory"
        filepath = memory_dir / filename

        # Prevent path traversal
        try:
            filepath.resolve().relative_to(memory_dir.resolve())
        except ValueError:
            return ToolResult(output="", error="Invalid filename", return_code=1)

        if not filepath.exists():
            # List available files
            available = [f.name for f in memory_dir.iterdir() if f.is_file()] if memory_dir.exists() else []
            return ToolResult(
                output=f"Available: {', '.join(available) or 'none'}",
                error=f"File not found: {filename}",
                return_code=1,
            )

        content = filepath.read_text(encoding="utf-8")
        return ToolResult(output=content)


class MemoryWriteTool(Tool):
    name = "memory_write"
    description = "Write content to a memory file in the session's memory directory."
    parameters = {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "Name of the memory file to write (e.g. 'notes.md')",
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file",
            },
        },
        "required": ["filename", "content"],
    }

    async def execute(self, arguments: dict[str, Any], **kwargs: Any) -> ToolResult:
        filename = arguments.get("filename", "")
        content = arguments.get("content", "")
        cwd: Path | None = kwargs.get("cwd")
        if not cwd or not filename:
            return ToolResult(output="", error="Missing filename or cwd", return_code=1)

        memory_dir = cwd / "memory"
        memory_dir.mkdir(exist_ok=True)
        filepath = memory_dir / filename

        # Prevent path traversal
        try:
            filepath.resolve().relative_to(memory_dir.resolve())
        except ValueError:
            return ToolResult(output="", error="Invalid filename", return_code=1)

        filepath.write_text(content, encoding="utf-8")
        logger.info("Memory written: %s (%d bytes)", filename, len(content))
        return ToolResult(output=f"Written {len(content)} bytes to {filename}")
