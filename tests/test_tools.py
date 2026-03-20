"""Tests for tool base, shell, and stop tools."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from spicyclaw.gateway.tools.base import COMMON_REQUIRED, Tool, ToolRegistry
from spicyclaw.gateway.tools.shell import ShellTool
from spicyclaw.gateway.tools.stop import StopTool


class TestToolBase:
    def test_to_openai_tool_injects_common_params(self):
        tool = StopTool()
        spec = tool.to_openai_tool()
        assert spec["type"] == "function"
        fn = spec["function"]
        assert fn["name"] == "stop"
        props = fn["parameters"]["properties"]
        assert "work_node" in props
        assert "next_step" in props
        assert "reason" in props
        required = fn["parameters"]["required"]
        for r in COMMON_REQUIRED:
            assert r in required
        assert "reason" in required

    def test_to_openai_tool_no_duplicate_required(self):
        tool = ShellTool()
        spec = tool.to_openai_tool()
        required = spec["function"]["parameters"]["required"]
        assert len(required) == len(set(required))

    def test_to_openai_tool_does_not_mutate_class(self):
        tool = ShellTool()
        _ = tool.to_openai_tool()
        # Original parameters should not have common params
        assert "work_node" not in tool.parameters.get("properties", {})


class TestToolRegistry:
    def test_register_and_get(self):
        reg = ToolRegistry()
        reg.register(ShellTool())
        assert reg.get("shell") is not None
        assert reg.get("nonexistent") is None

    def test_names(self):
        reg = ToolRegistry()
        reg.register(ShellTool())
        reg.register(StopTool())
        assert sorted(reg.names) == ["shell", "stop"]

    def test_to_openai_tools(self):
        reg = ToolRegistry()
        reg.register(ShellTool())
        reg.register(StopTool())
        tools = reg.to_openai_tools()
        assert len(tools) == 2
        names = {t["function"]["name"] for t in tools}
        assert names == {"shell", "stop"}


@pytest.mark.asyncio
class TestShellTool:
    async def test_simple_command(self):
        tool = ShellTool()
        result = await tool.execute({"command": "echo hello"})
        assert result.output.strip() == "hello"
        assert result.return_code == 0
        assert result.error == ""

    async def test_command_with_cwd(self, tmp_path: Path):
        tool = ShellTool()
        result = await tool.execute({"command": "pwd"}, cwd=tmp_path)
        assert str(tmp_path) in result.output

    async def test_command_stderr(self):
        tool = ShellTool()
        result = await tool.execute({"command": "echo err >&2"})
        assert "err" in result.error
        assert result.return_code == 0

    async def test_command_failure(self):
        tool = ShellTool()
        result = await tool.execute({"command": "exit 42"})
        assert result.return_code == 42

    async def test_empty_command(self):
        tool = ShellTool()
        result = await tool.execute({"command": ""})
        assert result.return_code == 1
        assert "Empty" in result.error

    async def test_timeout(self):
        tool = ShellTool(timeout=0.5)
        result = await tool.execute({"command": "sleep 10"})
        assert result.return_code == -1
        assert "timed out" in result.error

    async def test_output_truncation(self):
        tool = ShellTool(max_output=50)
        result = await tool.execute({"command": "python3 -c 'print(\"A\" * 200)'"})
        assert result.truncated is True
        assert "truncated" in result.output

    async def test_pipe_command(self):
        tool = ShellTool()
        result = await tool.execute({"command": "echo 'line1\nline2\nline3' | wc -l"})
        assert result.return_code == 0
        assert result.output.strip() == "3"


@pytest.mark.asyncio
class TestStopTool:
    async def test_stop_returns_reason(self):
        tool = StopTool()
        result = await tool.execute({"reason": "task complete"})
        assert "task complete" in result.output
        assert result.return_code == 0

    async def test_stop_default_reason(self):
        tool = StopTool()
        result = await tool.execute({})
        assert "no reason given" in result.output
