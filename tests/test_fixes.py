"""Tests for bugfix batch: workspace separation, command during run,
work_node validation, forced compact, timestamps, etc."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from spicyclaw.common.events import ServerEvent
from spicyclaw.common.i18n import set_lang, t
from spicyclaw.common.types import Message, Role, SessionMeta, SessionStatus, ToolCall, ToolResult
from spicyclaw.config import Settings
from spicyclaw.gateway.context import ContextManager
from spicyclaw.gateway.llm_client import LLMResponse
from spicyclaw.gateway.routes import _handle_command
from spicyclaw.gateway.session import Session, SessionManager
from spicyclaw.gateway.tools.base import ToolRegistry
from spicyclaw.gateway.tools.memory import MemoryReadTool, MemoryWriteTool
from spicyclaw.gateway.tools.shell import ShellTool
from spicyclaw.gateway.tools.stop import StopTool
from spicyclaw.gateway.workloop import run_workloop


def make_llm_response(
    content: str = "",
    tool_calls: list[ToolCall] | None = None,
    usage_tokens: int = 100,
) -> LLMResponse:
    r = LLMResponse()
    r.content = content
    r.tool_calls = tool_calls or []
    r.usage_tokens = usage_tokens
    return r


async def mock_stream_chat_factory(responses: list[LLMResponse]):
    call_count = 0

    async def stream_chat(messages, tools=None):
        nonlocal call_count
        if call_count < len(responses):
            resp = responses[call_count]
            call_count += 1
        else:
            resp = make_llm_response(content="Done.")
        if resp.content:
            yield "chunk", resp
        yield "done", resp

    return stream_chat


@pytest.fixture(autouse=True)
def reset_lang():
    set_lang("en")
    yield
    set_lang("en")


# ── Workspace separation ──


class TestWorkspaceSeparation:
    def test_session_has_workspace(self, session: Session):
        """Session.workspace points to session_dir/workspace."""
        assert session.workspace == session.dir / "workspace"
        assert session.workspace.exists()

    @pytest.mark.asyncio
    async def test_shell_runs_in_workspace(self, session: Session, tmp_settings: Settings):
        """Shell tool cwd is workspace, not session dir."""
        session.add_message(Message(role=Role.USER, content="test"))

        shell_call = ToolCall(
            id="tc_1",
            function_name="shell",
            arguments=json.dumps({
                "command": "pwd",
                "work_node": "1",
                "next_step": "check",
            }),
        )
        stop_call = ToolCall(
            id="tc_2",
            function_name="stop",
            arguments=json.dumps({
                "reason": "done",
                "work_node": "1",
                "next_step": "none",
            }),
        )
        resp1 = make_llm_response(tool_calls=[shell_call])
        resp2 = make_llm_response(tool_calls=[stop_call])

        mock_llm = AsyncMock()
        mock_llm.stream_chat = await mock_stream_chat_factory([resp1, resp2])

        registry = ToolRegistry()
        registry.register(ShellTool())
        registry.register(StopTool())

        await run_workloop(session, mock_llm, registry, tmp_settings)

        tool_msgs = [m for m in session.context if m.role == Role.TOOL and m.name == "shell"]
        assert len(tool_msgs) >= 1
        # pwd output should end with /workspace
        assert tool_msgs[0].content.strip().endswith("/workspace")

    @pytest.mark.asyncio
    async def test_task_md_read_from_workspace(self, session: Session, tmp_settings: Settings):
        """/task reads TASK.md from workspace dir."""
        session.workspace.mkdir(parents=True, exist_ok=True)
        (session.workspace / "TASK.md").write_text("# Test Task\nDetails")
        result = await _handle_command(session, "task", "", None, tmp_settings)
        assert "Test Task" in result["message"]

    @pytest.mark.asyncio
    async def test_plan_json_read_from_workspace(self, session: Session, tmp_settings: Settings):
        """/plan reads PLAN.json from workspace dir."""
        session.workspace.mkdir(parents=True, exist_ok=True)
        (session.workspace / "PLAN.json").write_text('{"nodes": [{"id": "1"}]}')
        result = await _handle_command(session, "plan", "", None, tmp_settings)
        assert "nodes" in result["message"]

    @pytest.mark.asyncio
    async def test_system_prompt_uses_workspace(self, session: Session, tmp_settings: Settings):
        """System prompt working directory should reference workspace."""
        session.add_message(Message(role=Role.USER, content="test"))
        resp = make_llm_response(content="ok")

        mock_llm = AsyncMock()
        mock_llm.stream_chat = await mock_stream_chat_factory([resp])

        registry = ToolRegistry()
        registry.register(ShellTool())
        registry.register(StopTool())

        await run_workloop(session, mock_llm, registry, tmp_settings)

        sys_msg = session.context[0]
        assert sys_msg.role == Role.SYSTEM
        assert "workspace" in sys_msg.content


# ── Memory tool uses session_dir ──


class TestMemoryToolSessionDir:
    @pytest.mark.asyncio
    async def test_memory_write_uses_session_dir(self, tmp_path: Path):
        tool = MemoryWriteTool()
        result = await tool.execute(
            {"filename": "note.md", "content": "test"},
            session_dir=tmp_path,
        )
        assert result.return_code == 0
        assert (tmp_path / "memory" / "note.md").exists()

    @pytest.mark.asyncio
    async def test_memory_read_uses_session_dir(self, tmp_path: Path):
        mem = tmp_path / "memory"
        mem.mkdir()
        (mem / "note.md").write_text("hello")
        tool = MemoryReadTool()
        result = await tool.execute({"filename": "note.md"}, session_dir=tmp_path)
        assert result.return_code == 0
        assert result.output == "hello"

    @pytest.mark.asyncio
    async def test_memory_write_without_session_dir_fails(self):
        tool = MemoryWriteTool()
        result = await tool.execute({"filename": "x.md", "content": "y"})
        assert result.return_code == 1

    @pytest.mark.asyncio
    async def test_memory_read_without_session_dir_fails(self):
        tool = MemoryReadTool()
        result = await tool.execute({"filename": "x.md"})
        assert result.return_code == 1


# ── Work_node / next_step validation ──


@pytest.mark.asyncio
class TestWorkNodeValidation:
    async def test_missing_work_node_returns_error(self, session: Session, tmp_settings: Settings):
        """Tool call without work_node returns error message."""
        session.add_message(Message(role=Role.USER, content="test"))

        bad_call = ToolCall(
            id="tc_1",
            function_name="shell",
            arguments=json.dumps({
                "command": "echo hi",
                # Missing work_node and next_step
            }),
        )
        resp1 = make_llm_response(tool_calls=[bad_call])
        resp2 = make_llm_response(content="ok")

        mock_llm = AsyncMock()
        mock_llm.stream_chat = await mock_stream_chat_factory([resp1, resp2])

        registry = ToolRegistry()
        registry.register(ShellTool())
        registry.register(StopTool())

        await run_workloop(session, mock_llm, registry, tmp_settings)

        tool_msgs = [m for m in session.context if m.role == Role.TOOL]
        assert len(tool_msgs) >= 1
        assert "Missing required parameters" in tool_msgs[0].content
        assert "work_node" in tool_msgs[0].content
        assert "next_step" in tool_msgs[0].content

    async def test_missing_only_next_step_returns_error(self, session: Session, tmp_settings: Settings):
        """Tool call with work_node but no next_step returns error."""
        session.add_message(Message(role=Role.USER, content="test"))

        bad_call = ToolCall(
            id="tc_1",
            function_name="shell",
            arguments=json.dumps({
                "command": "echo hi",
                "work_node": "1.1",
                # Missing next_step
            }),
        )
        resp1 = make_llm_response(tool_calls=[bad_call])
        resp2 = make_llm_response(content="ok")

        mock_llm = AsyncMock()
        mock_llm.stream_chat = await mock_stream_chat_factory([resp1, resp2])

        registry = ToolRegistry()
        registry.register(ShellTool())
        registry.register(StopTool())

        await run_workloop(session, mock_llm, registry, tmp_settings)

        tool_msgs = [m for m in session.context if m.role == Role.TOOL]
        assert any("next_step" in m.content for m in tool_msgs)

    async def test_valid_work_node_proceeds_normally(self, session: Session, tmp_settings: Settings):
        """Tool call with both work_node and next_step executes normally."""
        session.add_message(Message(role=Role.USER, content="test"))

        good_call = ToolCall(
            id="tc_1",
            function_name="shell",
            arguments=json.dumps({
                "command": "echo success",
                "work_node": "1.1",
                "next_step": "verify output",
            }),
        )
        resp1 = make_llm_response(tool_calls=[good_call])
        resp2 = make_llm_response(content="done")

        mock_llm = AsyncMock()
        mock_llm.stream_chat = await mock_stream_chat_factory([resp1, resp2])

        registry = ToolRegistry()
        registry.register(ShellTool())
        registry.register(StopTool())

        await run_workloop(session, mock_llm, registry, tmp_settings)

        tool_msgs = [m for m in session.context if m.role == Role.TOOL]
        assert any("success" in m.content for m in tool_msgs)


# ── Commands during workloop ──


@pytest.mark.asyncio
class TestCommandsDuringRun:
    async def test_stop_command_during_run(self, session: Session, tmp_settings: Settings):
        """/stop should work while workloop is running."""
        result = await _handle_command(session, "stop", "", None, tmp_settings)
        assert session.abort_event.is_set()

    async def test_status_command_always_works(self, session: Session, tmp_settings: Settings):
        """/status should always return info."""
        result = await _handle_command(session, "status", "", None, tmp_settings)
        assert session.id in result["message"]

    async def test_yolo_command_during_run(self, session: Session, tmp_settings: Settings):
        """/yolo should toggle mode even during run."""
        session.step_mode = True
        result = await _handle_command(session, "yolo", "", None, tmp_settings)
        assert not session.step_mode


# ── Forced compact ──


@pytest.mark.asyncio
class TestForcedCompact:
    async def test_manual_compact_forces_compression(self, session: Session, tmp_settings: Settings):
        """Manual /compact should compress even when context is small."""
        # Add enough messages to have something to compress
        session.context.append(Message(role=Role.SYSTEM, content="system"))
        for i in range(6):
            session.context.append(Message(role=Role.USER, content=f"message {i}"))
            session.context.append(Message(role=Role.ASSISTANT, content=f"reply {i}"))

        # Token usage is low — auto-compact would NOT trigger
        session.meta.token_used = 100

        mock_llm = AsyncMock()
        summary_resp = LLMResponse()
        summary_resp.content = "Summary of conversation"
        mock_llm.chat = AsyncMock(return_value=summary_resp)

        ctx_mgr = ContextManager(session, tmp_settings)
        # Without force, should_compact is False
        assert not ctx_mgr.should_compact

        # Force compact should still work
        result = await ctx_mgr.full_compact(mock_llm, force=True)
        assert result is not None
        assert "Summary" in result

    async def test_auto_compact_respects_threshold(self, session: Session, tmp_settings: Settings):
        """Auto-compact only triggers when threshold exceeded."""
        ctx_mgr = ContextManager(session, tmp_settings)
        session.meta.token_used = 100
        assert not ctx_mgr.should_compact

        session.meta.token_used = int(tmp_settings.max_tokens * 0.85)
        assert ctx_mgr.should_compact


# ── Timestamps ──


class TestTimestamps:
    def test_message_has_timestamp(self):
        """Message model includes ts field."""
        msg = Message(role=Role.USER, content="hello")
        assert msg.ts is not None
        assert msg.ts > 0
        # Should be close to current time
        assert abs(msg.ts - time.time()) < 5

    def test_message_timestamp_serializes(self):
        """ts field survives serialization/deserialization."""
        msg = Message(role=Role.USER, content="hi")
        data = msg.model_dump()
        assert "ts" in data
        restored = Message.model_validate(data)
        assert restored.ts == msg.ts

    def test_server_event_has_timestamp(self):
        """ServerEvent includes ts."""
        event = ServerEvent(type="chunk", session_id="test", data={"text": "hi"})
        assert event.ts is not None
        assert event.ts > 0


# ── i18n new key ──


class TestI18nNewKeys:
    def test_msg_queued_en(self):
        set_lang("en")
        assert "received" in t("msg_queued").lower()

    def test_msg_queued_zh(self):
        set_lang("zh")
        assert "收到" in t("msg_queued")


# ── TASK.md title from workspace ──


@pytest.mark.asyncio
class TestTaskTitleFromWorkspace:
    async def test_title_updated_from_workspace_task_md(self, session: Session, tmp_settings: Settings):
        """Workloop reads TASK.md from workspace/ not session dir."""
        session.add_message(Message(role=Role.USER, content="create task"))

        # Shell will create TASK.md in workspace
        shell_call = ToolCall(
            id="tc_1",
            function_name="shell",
            arguments=json.dumps({
                "command": f"echo '# Workspace Task' > {session.workspace}/TASK.md",
                "work_node": "1",
                "next_step": "stop",
            }),
        )
        stop_call = ToolCall(
            id="tc_2",
            function_name="stop",
            arguments=json.dumps({
                "reason": "done",
                "work_node": "2",
                "next_step": "none",
            }),
        )
        resp1 = make_llm_response(tool_calls=[shell_call])
        resp2 = make_llm_response(tool_calls=[stop_call])

        mock_llm = AsyncMock()
        mock_llm.stream_chat = await mock_stream_chat_factory([resp1, resp2])

        registry = ToolRegistry()
        registry.register(ShellTool())
        registry.register(StopTool())

        await run_workloop(session, mock_llm, registry, tmp_settings)

        assert session.meta.title == "Workspace Task"

    async def test_task_md_in_session_dir_not_read(self, session: Session, tmp_settings: Settings):
        """TASK.md in session dir (not workspace) should NOT be read."""
        (session.dir / "TASK.md").write_text("# Wrong Location")
        result = await _handle_command(session, "task", "", None, tmp_settings)
        assert "TASK.md" in result["message"]  # "not found" message
