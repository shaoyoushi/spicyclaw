"""Tests for the workloop."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spicyclaw.common.types import Message, Role, SessionStatus, ToolCall, ToolResult
from spicyclaw.config import Settings
from spicyclaw.gateway.llm_client import LLMResponse
from spicyclaw.gateway.session import Session, SessionManager
from spicyclaw.gateway.tools.base import ToolRegistry
from spicyclaw.gateway.tools.shell import ShellTool
from spicyclaw.gateway.tools.stop import StopTool
from spicyclaw.gateway.workloop import RepeatTracker, run_workloop


def make_llm_response(
    content: str = "",
    tool_calls: list[ToolCall] | None = None,
    usage_tokens: int = 100,
    finish_reason: str = "stop",
) -> LLMResponse:
    r = LLMResponse()
    r.content = content
    r.tool_calls = tool_calls or []
    r.usage_tokens = usage_tokens
    r.finish_reason = finish_reason
    return r


async def mock_stream_chat_factory(responses: list[LLMResponse]):
    """Create a mock stream_chat that yields from a list of pre-built responses."""
    call_count = 0

    async def stream_chat(messages, tools=None):
        nonlocal call_count
        if call_count < len(responses):
            resp = responses[call_count]
            call_count += 1
        else:
            resp = make_llm_response(content="I'm done, no more responses.")
        # Simulate streaming: yield chunk then done
        if resp.content:
            yield "chunk", resp
        yield "done", resp

    return stream_chat


class TestRepeatTracker:
    def test_no_repeat(self):
        tracker = RepeatTracker(max_errors=3, max_outputs=5)
        r1 = ToolResult(output="out1")
        r2 = ToolResult(output="out2")
        assert tracker.check("shell", {"cmd": "ls"}, r1) is None
        assert tracker.check("shell", {"cmd": "ls"}, r2) is None

    def test_error_repeat_triggers(self):
        tracker = RepeatTracker(max_errors=3, max_outputs=10)
        r = ToolResult(output="", error="fail", return_code=1)
        args = {"cmd": "bad"}
        assert tracker.check("shell", args, r) is None
        assert tracker.check("shell", args, r) is None
        result = tracker.check("shell", args, r)
        assert result is not None
        assert "3 times" in result

    def test_output_repeat_triggers(self):
        tracker = RepeatTracker(max_errors=5, max_outputs=3)
        r = ToolResult(output="same output")
        args = {"cmd": "echo x"}
        assert tracker.check("shell", args, r) is None
        assert tracker.check("shell", args, r) is None
        result = tracker.check("shell", args, r)
        assert result is not None
        assert "3 times" in result

    def test_different_commands_dont_trigger(self):
        tracker = RepeatTracker(max_errors=2, max_outputs=2)
        r = ToolResult(output="same")
        assert tracker.check("shell", {"cmd": "a"}, r) is None
        assert tracker.check("shell", {"cmd": "b"}, r) is None
        assert tracker.check("shell", {"cmd": "a"}, r) is None

    def test_interleaved_resets_count(self):
        tracker = RepeatTracker(max_errors=3, max_outputs=3)
        r1 = ToolResult(output="out1")
        r2 = ToolResult(output="out2")
        args = {"cmd": "x"}
        tracker.check("shell", args, r1)
        tracker.check("shell", args, r1)
        tracker.check("shell", args, r2)  # breaks the streak
        assert tracker.check("shell", args, r1) is None  # reset


@pytest.mark.asyncio
class TestWorkloop:
    async def test_content_only_response_stops(
        self, session: Session, tool_registry: ToolRegistry, tmp_settings: Settings
    ):
        """When LLM returns content without tool_calls, workloop stops."""
        session.add_message(Message(role=Role.USER, content="hello"))
        resp = make_llm_response(content="Hello! How can I help?")

        mock_llm = AsyncMock()
        stream_fn = await mock_stream_chat_factory([resp])
        mock_llm.stream_chat = stream_fn

        await run_workloop(session, mock_llm, tool_registry, tmp_settings)

        assert session.status == SessionStatus.STOPPED
        # Should have: system + user + assistant
        roles = [m.role for m in session.context]
        assert roles[0] == Role.SYSTEM
        assert roles[1] == Role.USER
        assert roles[2] == Role.ASSISTANT
        assert session.context[2].content == "Hello! How can I help?"

    async def test_stop_tool_ends_loop(
        self, session: Session, tool_registry: ToolRegistry, tmp_settings: Settings
    ):
        """When model calls stop tool, workloop ends."""
        session.add_message(Message(role=Role.USER, content="do something"))

        stop_call = ToolCall(
            id="tc_1",
            function_name="stop",
            arguments=json.dumps({
                "reason": "task complete",
                "work_node": "1",
                "next_step": "none",
            }),
        )
        resp = make_llm_response(tool_calls=[stop_call])

        mock_llm = AsyncMock()
        stream_fn = await mock_stream_chat_factory([resp])
        mock_llm.stream_chat = stream_fn

        await run_workloop(session, mock_llm, tool_registry, tmp_settings)

        assert session.status == SessionStatus.STOPPED
        # Context: system + user + assistant(tool_call) + tool(result)
        tool_msgs = [m for m in session.context if m.role == Role.TOOL]
        assert len(tool_msgs) == 1
        assert "task complete" in tool_msgs[0].content

    async def test_shell_tool_execution(
        self, session: Session, tool_registry: ToolRegistry, tmp_settings: Settings
    ):
        """Shell tool executes and result goes back to context."""
        session.add_message(Message(role=Role.USER, content="list files"))

        shell_call = ToolCall(
            id="tc_1",
            function_name="shell",
            arguments=json.dumps({
                "command": "echo test_output",
                "work_node": "1.1",
                "next_step": "check output",
            }),
        )
        resp1 = make_llm_response(tool_calls=[shell_call])
        resp2 = make_llm_response(content="Done listing files.")

        mock_llm = AsyncMock()
        stream_fn = await mock_stream_chat_factory([resp1, resp2])
        mock_llm.stream_chat = stream_fn

        await run_workloop(session, mock_llm, tool_registry, tmp_settings)

        tool_msgs = [m for m in session.context if m.role == Role.TOOL]
        assert len(tool_msgs) == 1
        assert "test_output" in tool_msgs[0].content

    async def test_unknown_tool_returns_error(
        self, session: Session, tool_registry: ToolRegistry, tmp_settings: Settings
    ):
        """Unknown tool name produces an error tool message."""
        session.add_message(Message(role=Role.USER, content="test"))

        bad_call = ToolCall(
            id="tc_1",
            function_name="nonexistent_tool",
            arguments=json.dumps({
                "work_node": "1",
                "next_step": "n/a",
            }),
        )
        resp1 = make_llm_response(tool_calls=[bad_call])
        resp2 = make_llm_response(content="Ok.")

        mock_llm = AsyncMock()
        stream_fn = await mock_stream_chat_factory([resp1, resp2])
        mock_llm.stream_chat = stream_fn

        await run_workloop(session, mock_llm, tool_registry, tmp_settings)

        tool_msgs = [m for m in session.context if m.role == Role.TOOL]
        assert len(tool_msgs) == 1
        assert "Unknown tool" in tool_msgs[0].content

    async def test_invalid_json_args(
        self, session: Session, tool_registry: ToolRegistry, tmp_settings: Settings
    ):
        """Invalid JSON in tool arguments produces an error message."""
        session.add_message(Message(role=Role.USER, content="test"))

        bad_call = ToolCall(
            id="tc_1",
            function_name="shell",
            arguments="not valid json{{{",
        )
        resp1 = make_llm_response(tool_calls=[bad_call])
        resp2 = make_llm_response(content="Ok, I'll fix that.")

        mock_llm = AsyncMock()
        stream_fn = await mock_stream_chat_factory([resp1, resp2])
        mock_llm.stream_chat = stream_fn

        await run_workloop(session, mock_llm, tool_registry, tmp_settings)

        tool_msgs = [m for m in session.context if m.role == Role.TOOL]
        assert len(tool_msgs) == 1
        assert "Invalid JSON" in tool_msgs[0].content

    async def test_abort_stops_loop(
        self, session: Session, tool_registry: ToolRegistry, tmp_settings: Settings
    ):
        """Setting abort_event stops the workloop."""
        session.add_message(Message(role=Role.USER, content="long task"))
        session.abort_event.set()

        resp = make_llm_response(content="Starting...")

        mock_llm = AsyncMock()
        stream_fn = await mock_stream_chat_factory([resp])
        mock_llm.stream_chat = stream_fn

        await run_workloop(session, mock_llm, tool_registry, tmp_settings)

        assert session.status == SessionStatus.STOPPED

    async def test_max_steps_protection(
        self, session: Session, tool_registry: ToolRegistry, tmp_settings: Settings
    ):
        """Workloop stops after max_steps."""
        tmp_settings.max_steps = 2
        session.add_message(Message(role=Role.USER, content="loop forever"))

        shell_call = ToolCall(
            id="tc_1",
            function_name="shell",
            arguments=json.dumps({
                "command": "echo hi",
                "work_node": "1",
                "next_step": "continue",
            }),
        )
        # Return shell calls indefinitely
        responses = [make_llm_response(tool_calls=[shell_call]) for _ in range(10)]

        mock_llm = AsyncMock()
        stream_fn = await mock_stream_chat_factory(responses)
        mock_llm.stream_chat = stream_fn

        await run_workloop(session, mock_llm, tool_registry, tmp_settings)

        assert session.status == SessionStatus.STOPPED
        # Should have stopped after 2 steps
        tool_msgs = [m for m in session.context if m.role == Role.TOOL]
        assert len(tool_msgs) == 2

    async def test_system_prompt_injected(
        self, session: Session, tool_registry: ToolRegistry, tmp_settings: Settings
    ):
        """System prompt is injected at the start of context."""
        session.add_message(Message(role=Role.USER, content="test"))
        resp = make_llm_response(content="ok")

        mock_llm = AsyncMock()
        stream_fn = await mock_stream_chat_factory([resp])
        mock_llm.stream_chat = stream_fn

        await run_workloop(session, mock_llm, tool_registry, tmp_settings)

        assert session.context[0].role == Role.SYSTEM
        assert "SpicyClaw" in session.context[0].content

    async def test_task_md_updates_title(
        self, session: Session, tool_registry: ToolRegistry, tmp_settings: Settings
    ):
        """Creating TASK.md updates session title."""
        session.add_message(Message(role=Role.USER, content="create task file"))

        # Model creates TASK.md via shell
        shell_call = ToolCall(
            id="tc_1",
            function_name="shell",
            arguments=json.dumps({
                "command": f"echo '# My Cool Task\nDetails here' > {session.workspace}/TASK.md",
                "work_node": "1",
                "next_step": "create plan",
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
        stream_fn = await mock_stream_chat_factory([resp1, resp2])
        mock_llm.stream_chat = stream_fn

        await run_workloop(session, mock_llm, tool_registry, tmp_settings)

        assert session.meta.title == "My Cool Task"

    async def test_multiple_tool_calls_in_one_response(
        self, session: Session, tool_registry: ToolRegistry, tmp_settings: Settings
    ):
        """Model returns multiple tool calls in one response."""
        session.add_message(Message(role=Role.USER, content="do two things"))

        call1 = ToolCall(
            id="tc_1",
            function_name="shell",
            arguments=json.dumps({
                "command": "echo first",
                "work_node": "1.1",
                "next_step": "second command",
            }),
        )
        call2 = ToolCall(
            id="tc_2",
            function_name="shell",
            arguments=json.dumps({
                "command": "echo second",
                "work_node": "1.2",
                "next_step": "done",
            }),
        )
        resp1 = make_llm_response(tool_calls=[call1, call2])
        resp2 = make_llm_response(content="Both done.")

        mock_llm = AsyncMock()
        stream_fn = await mock_stream_chat_factory([resp1, resp2])
        mock_llm.stream_chat = stream_fn

        await run_workloop(session, mock_llm, tool_registry, tmp_settings)

        tool_msgs = [m for m in session.context if m.role == Role.TOOL]
        assert len(tool_msgs) == 2
        assert "first" in tool_msgs[0].content
        assert "second" in tool_msgs[1].content
