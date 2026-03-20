"""Tests for memory read/write tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from spicyclaw.gateway.tools.memory import MemoryReadTool, MemoryWriteTool


class TestMemoryWriteTool:
    @pytest.fixture
    def tool(self):
        return MemoryWriteTool()

    @pytest.mark.asyncio
    async def test_write_creates_file(self, tool, tmp_path):
        (tmp_path / "memory").mkdir()
        result = await tool.execute(
            {"filename": "notes.md", "content": "hello world"},
            cwd=tmp_path,
        )
        assert result.return_code == 0
        assert "11 bytes" in result.output
        assert (tmp_path / "memory" / "notes.md").read_text() == "hello world"

    @pytest.mark.asyncio
    async def test_write_creates_memory_dir(self, tool, tmp_path):
        result = await tool.execute(
            {"filename": "test.txt", "content": "data"},
            cwd=tmp_path,
        )
        assert result.return_code == 0
        assert (tmp_path / "memory" / "test.txt").exists()

    @pytest.mark.asyncio
    async def test_write_no_cwd(self, tool):
        result = await tool.execute({"filename": "x.md", "content": "y"})
        assert result.return_code == 1

    @pytest.mark.asyncio
    async def test_write_path_traversal(self, tool, tmp_path):
        (tmp_path / "memory").mkdir()
        result = await tool.execute(
            {"filename": "../escape.txt", "content": "evil"},
            cwd=tmp_path,
        )
        assert result.return_code == 1
        assert "Invalid" in result.error

    @pytest.mark.asyncio
    async def test_openai_schema(self, tool):
        schema = tool.to_openai_tool()
        assert schema["function"]["name"] == "memory_write"
        props = schema["function"]["parameters"]["properties"]
        assert "filename" in props
        assert "content" in props


class TestMemoryReadTool:
    @pytest.fixture
    def tool(self):
        return MemoryReadTool()

    @pytest.mark.asyncio
    async def test_read_existing_file(self, tool, tmp_path):
        mem = tmp_path / "memory"
        mem.mkdir()
        (mem / "notes.md").write_text("hello")
        result = await tool.execute({"filename": "notes.md"}, cwd=tmp_path)
        assert result.return_code == 0
        assert result.output == "hello"

    @pytest.mark.asyncio
    async def test_read_nonexistent(self, tool, tmp_path):
        mem = tmp_path / "memory"
        mem.mkdir()
        (mem / "other.md").write_text("x")
        result = await tool.execute({"filename": "missing.md"}, cwd=tmp_path)
        assert result.return_code == 1
        assert "not found" in result.error
        assert "other.md" in result.output

    @pytest.mark.asyncio
    async def test_read_no_cwd(self, tool):
        result = await tool.execute({"filename": "x.md"})
        assert result.return_code == 1

    @pytest.mark.asyncio
    async def test_read_path_traversal(self, tool, tmp_path):
        (tmp_path / "memory").mkdir()
        result = await tool.execute({"filename": "../../etc/passwd"}, cwd=tmp_path)
        assert result.return_code == 1
        assert "Invalid" in result.error

    @pytest.mark.asyncio
    async def test_read_empty_memory_dir(self, tool, tmp_path):
        (tmp_path / "memory").mkdir()
        result = await tool.execute({"filename": "x.md"}, cwd=tmp_path)
        assert result.return_code == 1
        assert "none" in result.output
