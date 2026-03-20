"""Shell execution tool."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from spicyclaw.common.types import ToolResult
from spicyclaw.gateway.tools.base import Tool

logger = logging.getLogger(__name__)


class ShellTool(Tool):
    name = "shell"
    description = "Execute a shell command and return stdout/stderr."
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute",
            },
        },
        "required": ["command"],
    }

    def __init__(self, timeout: float = 120.0, max_output: int = 10000) -> None:
        self.timeout = timeout
        self.max_output = max_output

    async def execute(self, arguments: dict[str, Any], **kwargs: Any) -> ToolResult:
        command = arguments.get("command", "")
        cwd: Path | None = kwargs.get("cwd")

        if not command.strip():
            return ToolResult(output="", error="Empty command", return_code=1)

        logger.info("Shell exec: %s", command[:200])

        try:
            proc = await asyncio.create_subprocess_exec(
                "bash", "-c", command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd) if cwd else None,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return ToolResult(
                    output="",
                    error=f"Command timed out after {self.timeout}s",
                    return_code=-1,
                )

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            rc = proc.returncode or 0

            truncated = False
            if len(stdout) > self.max_output:
                stdout = stdout[: self.max_output] + f"\n... [truncated, total {len(stdout_bytes)} bytes]"
                truncated = True
            if len(stderr) > self.max_output:
                stderr = stderr[: self.max_output] + f"\n... [truncated, total {len(stderr_bytes)} bytes]"
                truncated = True

            return ToolResult(
                output=stdout,
                error=stderr,
                return_code=rc,
                truncated=truncated,
            )
        except Exception as e:
            logger.exception("Shell exec failed")
            return ToolResult(output="", error=str(e), return_code=1)
