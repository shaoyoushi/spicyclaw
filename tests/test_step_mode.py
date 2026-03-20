"""Tests for step mode (confirm flow) in the workloop."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from spicyclaw.common.types import Message, Role, ToolCall
from spicyclaw.config import Settings
from spicyclaw.gateway.llm_client import LLMResponse
from spicyclaw.gateway.session import Session, SessionManager
from spicyclaw.gateway.tools.base import ToolRegistry
from spicyclaw.gateway.tools.shell import ShellTool
from spicyclaw.gateway.tools.stop import StopTool
from spicyclaw.gateway.workloop import run_workloop


@pytest.fixture
def step_settings(tmp_path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        logs_dir=tmp_path / "logs",
        roles_dir=tmp_path / "roles",
        skills_dir=tmp_path / "skills",
        api_base_url="http://fake:9999/v1",
        api_key="test",
        model="test",
        yolo=False,  # Enable step mode checking
        _env_file=None,  # type: ignore[call-arg]
    )


@pytest.fixture
def step_session(step_settings) -> Session:
    mgr = SessionManager(step_settings)
    mgr.init()
    s = mgr.create()
    s.step_mode = True  # Enable step mode
    return s


@pytest.fixture
def tool_registry(step_settings):
    reg = ToolRegistry()
    reg.register(ShellTool(timeout=5, max_output=1000))
    reg.register(StopTool())
    return reg


@pytest.mark.asyncio
class TestStepMode:
    async def test_step_mode_pauses_then_confirms(
        self, step_session, tool_registry, step_settings
    ):
        """In step mode, workloop pauses and resumes on confirm."""
        step_session.add_message(Message(role=Role.USER, content="test"))

        stop_call = ToolCall(
            id="tc_1",
            function_name="stop",
            arguments=json.dumps({
                "reason": "done",
                "work_node": "1",
                "next_step": "none",
            }),
        )

        async def stream_chat(messages, tools=None):
            resp = LLMResponse()
            resp.tool_calls = [stop_call]
            resp.usage_tokens = 100
            yield "done", resp

        mock_llm = AsyncMock()
        mock_llm.stream_chat = stream_chat

        # Schedule confirm after a short delay
        async def confirm_later():
            await asyncio.sleep(0.3)
            step_session.confirm_event.set()

        events = []
        mock_ws = AsyncMock()
        mock_ws.send_text = AsyncMock(side_effect=lambda t: events.append(json.loads(t)))
        step_session.subscribers.add(mock_ws)

        confirm_task = asyncio.create_task(confirm_later())
        await run_workloop(step_session, mock_llm, tool_registry, step_settings)
        await confirm_task

        # Should have a paused status event
        paused_events = [e for e in events if e["type"] == "status" and e["data"].get("status") == "paused"]
        assert len(paused_events) >= 1

    async def test_step_mode_abort_during_pause(
        self, step_session, tool_registry, step_settings
    ):
        """Aborting during step mode pause stops the workloop."""
        step_session.add_message(Message(role=Role.USER, content="test"))

        shell_call = ToolCall(
            id="tc_1",
            function_name="shell",
            arguments=json.dumps({
                "command": "echo hi",
                "work_node": "1",
                "next_step": "done",
            }),
        )

        async def stream_chat(messages, tools=None):
            resp = LLMResponse()
            resp.tool_calls = [shell_call]
            resp.usage_tokens = 100
            yield "done", resp

        mock_llm = AsyncMock()
        mock_llm.stream_chat = stream_chat

        # Abort during pause
        async def abort_later():
            await asyncio.sleep(0.3)
            step_session.abort_event.set()

        abort_task = asyncio.create_task(abort_later())
        await run_workloop(step_session, mock_llm, tool_registry, step_settings)
        await abort_task

        # Should have stopped without executing the shell command
        # The shell command should NOT appear in tool results
        tool_msgs = [m for m in step_session.context if m.role == Role.TOOL]
        assert len(tool_msgs) == 0

    async def test_yolo_mode_no_pause(self, step_session, tool_registry, step_settings):
        """When step_mode is False, no pausing occurs even with yolo=False."""
        step_session.step_mode = False
        step_session.add_message(Message(role=Role.USER, content="test"))

        stop_call = ToolCall(
            id="tc_1",
            function_name="stop",
            arguments=json.dumps({
                "reason": "done",
                "work_node": "1",
                "next_step": "none",
            }),
        )

        async def stream_chat(messages, tools=None):
            resp = LLMResponse()
            resp.tool_calls = [stop_call]
            resp.usage_tokens = 100
            yield "done", resp

        mock_llm = AsyncMock()
        mock_llm.stream_chat = stream_chat

        events = []
        mock_ws = AsyncMock()
        mock_ws.send_text = AsyncMock(side_effect=lambda t: events.append(json.loads(t)))
        step_session.subscribers.add(mock_ws)

        await run_workloop(step_session, mock_llm, tool_registry, step_settings)

        # Should NOT have any paused status events
        paused_events = [e for e in events if e["type"] == "status" and e["data"].get("status") == "paused"]
        assert len(paused_events) == 0
