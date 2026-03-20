"""Integration tests for Phase 4 features against real LLM API.

Requires: http://192.168.100.100:8200/v1 with model llama_cpp_model
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from spicyclaw.common.types import Message, Role, ToolCall
from spicyclaw.config import Settings
from spicyclaw.gateway.context import ContextManager
from spicyclaw.gateway.llm_client import LLMClient
from spicyclaw.gateway.session import SessionManager
from spicyclaw.gateway.tools.base import ToolRegistry
from spicyclaw.gateway.tools.memory import MemoryReadTool, MemoryWriteTool
from spicyclaw.gateway.tools.shell import ShellTool
from spicyclaw.gateway.tools.stop import StopTool
from spicyclaw.gateway.tools.summary import SummaryTool
from spicyclaw.gateway.workloop import run_workloop


@pytest.fixture
def real_settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        logs_dir=tmp_path / "logs",
        roles_dir=tmp_path / "roles",
        skills_dir=tmp_path / "skills",
        api_base_url="http://192.168.100.100:8200/v1",
        api_key="sk-test03",
        model="llama_cpp_model",
        max_tokens=4096,
        max_steps=10,
        shell_timeout=30,
        idle_timeout=60,
        _env_file=None,  # type: ignore[call-arg]
    )


@pytest.fixture
def real_tool_registry(real_settings: Settings) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ShellTool(timeout=real_settings.shell_timeout, max_output=5000))
    reg.register(StopTool())
    reg.register(SummaryTool())
    reg.register(MemoryReadTool())
    reg.register(MemoryWriteTool())
    return reg


@pytest.mark.asyncio
class TestIntegrationPhase4:
    async def test_memory_tools_in_workloop(self, real_settings, real_tool_registry, tmp_path):
        """Test that memory tools are registered and available."""
        names = real_tool_registry.names
        assert "shell" in names
        assert "stop" in names
        assert "summary" in names
        assert "memory_read" in names
        assert "memory_write" in names

        tools = real_tool_registry.to_openai_tools()
        tool_names = [t["function"]["name"] for t in tools]
        assert "memory_read" in tool_names
        assert "memory_write" in tool_names

    async def test_memory_read_write_roundtrip(self, tmp_path):
        """Test memory tools directly."""
        write_tool = MemoryWriteTool()
        read_tool = MemoryReadTool()

        result = await write_tool.execute(
            {"filename": "test.md", "content": "hello from integration test"},
            cwd=tmp_path,
        )
        assert result.return_code == 0

        result = await read_tool.execute({"filename": "test.md"}, cwd=tmp_path)
        assert result.return_code == 0
        assert result.output == "hello from integration test"

    async def test_context_compact_with_real_llm(self, real_settings, tmp_path):
        """Test context compression using the real LLM."""
        mgr = SessionManager(real_settings)
        mgr.init()
        session = mgr.create()

        session.context = [
            Message(role=Role.SYSTEM, content="You are a helpful assistant."),
            Message(role=Role.USER, content="What is 2+2?"),
            Message(role=Role.ASSISTANT, content="2+2 equals 4."),
            Message(role=Role.USER, content="What is 3+3?"),
            Message(role=Role.ASSISTANT, content="3+3 equals 6."),
            Message(role=Role.USER, content="What is 4+4?"),
            Message(role=Role.ASSISTANT, content="4+4 equals 8."),
            Message(role=Role.USER, content="Thanks!"),
            Message(role=Role.ASSISTANT, content="You're welcome!"),
        ]

        real_settings_compact = Settings(
            data_dir=real_settings.data_dir,
            logs_dir=real_settings.logs_dir,
            roles_dir=real_settings.roles_dir,
            skills_dir=real_settings.skills_dir,
            api_base_url=real_settings.api_base_url,
            api_key=real_settings.api_key,
            model=real_settings.model,
            max_tokens=4096,
            compact_keep_rounds=1,
            idle_timeout=60,
            _env_file=None,  # type: ignore[call-arg]
        )

        llm = LLMClient(real_settings_compact)
        try:
            ctx_mgr = ContextManager(session, real_settings_compact)
            summary = await ctx_mgr.full_compact(llm)

            assert summary is not None
            assert len(summary) > 0
            print(f"Compact summary: {summary}")

            ctx = session.context
            assert ctx[0].role == Role.SYSTEM
            assert "[Context Summary]" in ctx[1].content
            user_msgs = [m for m in ctx if m.role == Role.USER]
            assert len(user_msgs) == 1
            assert user_msgs[0].content == "Thanks!"
        finally:
            await llm.close()

    async def test_workloop_with_all_tools(self, real_settings, real_tool_registry):
        """Test workloop with new tools registered."""
        mgr = SessionManager(real_settings)
        mgr.init()
        session = mgr.create()
        session.add_message(Message(
            role=Role.USER,
            content="Use the shell tool to run 'echo hello' and then stop with reason 'done'.",
        ))

        llm = LLMClient(real_settings)
        try:
            await asyncio.wait_for(
                run_workloop(session, llm, real_tool_registry, real_settings),
                timeout=120,
            )

            assert session.status.value == "stopped"
            assert len(session.context) > 2
            tool_msgs = [m for m in session.context if m.role == Role.TOOL]
            assert len(tool_msgs) > 0
            print(f"Tool messages: {len(tool_msgs)}")
            for m in tool_msgs:
                print(f"  [{m.name}] {m.content[:100]}")
        finally:
            await llm.close()

    async def test_work_node_compact_with_real_llm(self, real_settings, tmp_path):
        """Test work node compression using the real LLM."""
        mgr = SessionManager(real_settings)
        mgr.init()
        session = mgr.create()

        tc1 = ToolCall(
            id="tc1",
            function_name="shell",
            arguments=json.dumps({"command": "echo task1", "work_node": "1.1", "next_step": "next"}),
        )
        tc2 = ToolCall(
            id="tc2",
            function_name="shell",
            arguments=json.dumps({"command": "echo task2", "work_node": "2.1", "next_step": "next"}),
        )

        session.context = [
            Message(role=Role.SYSTEM, content="You are a helpful assistant."),
            Message(role=Role.USER, content="Do task 1"),
            Message(role=Role.ASSISTANT, tool_calls=[tc1]),
            Message(role=Role.TOOL, content="task1", tool_call_id="tc1", name="shell"),
            Message(role=Role.USER, content="Do task 2"),
            Message(role=Role.ASSISTANT, tool_calls=[tc2]),
            Message(role=Role.TOOL, content="task2", tool_call_id="tc2", name="shell"),
        ]

        real_settings_compact = Settings(
            data_dir=real_settings.data_dir,
            logs_dir=real_settings.logs_dir,
            roles_dir=real_settings.roles_dir,
            skills_dir=real_settings.skills_dir,
            api_base_url=real_settings.api_base_url,
            api_key=real_settings.api_key,
            model=real_settings.model,
            max_tokens=4096,
            compact_keep_rounds=1,
            idle_timeout=60,
            _env_file=None,  # type: ignore[call-arg]
        )

        llm = LLMClient(real_settings_compact)
        try:
            ctx_mgr = ContextManager(session, real_settings_compact)
            summary = await ctx_mgr.compact_work_nodes(llm, ["1.1"])

            assert summary is not None
            print(f"Work node summary: {summary}")

            ctx = session.context
            assert any("[Work Node Summary: 1.1]" in (m.content or "") for m in ctx)
            # Node 2.1 should still be there
            tool_msgs = [m for m in ctx if m.role == Role.TOOL]
            assert any("task2" in m.content for m in tool_msgs)
        finally:
            await llm.close()

    async def test_idle_timeout_health_probe(self, real_settings):
        """Test that idle_timeout health probe works with real LLM."""
        llm = LLMClient(real_settings)
        try:
            # The real LLM should be healthy
            is_healthy = await llm._probe_health()
            assert is_healthy is True
            assert llm.healthy is True
        finally:
            await llm.close()
