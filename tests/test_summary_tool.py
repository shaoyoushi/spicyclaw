"""Tests for the summary tool."""

from __future__ import annotations

import pytest

from spicyclaw.gateway.tools.summary import SummaryTool


class TestSummaryTool:
    @pytest.fixture
    def tool(self):
        return SummaryTool()

    def test_tool_name(self, tool):
        assert tool.name == "summary"

    def test_openai_schema(self, tool):
        schema = tool.to_openai_tool()
        assert schema["function"]["name"] == "summary"
        props = schema["function"]["parameters"]["properties"]
        assert "content" in props
        assert "work_node" in props
        assert "next_step" in props

    @pytest.mark.asyncio
    async def test_execute_with_content(self, tool):
        result = await tool.execute({"content": "Task completed successfully"})
        assert result.return_code == 0
        assert "Summary recorded" in result.output

    @pytest.mark.asyncio
    async def test_execute_empty_content(self, tool):
        result = await tool.execute({"content": "   "})
        assert result.return_code == 1
        assert "Empty summary" in result.error

    @pytest.mark.asyncio
    async def test_execute_no_content(self, tool):
        result = await tool.execute({})
        assert result.return_code == 1
