"""Tests for work node compression."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from spicyclaw.common.types import Message, Role, ToolCall
from spicyclaw.config import Settings
from spicyclaw.gateway.context import (
    ContextManager,
    _extract_work_nodes,
    _extract_work_nodes_tail,
    _get_message_work_node,
)
from spicyclaw.gateway.llm_client import LLMResponse
from spicyclaw.gateway.session import Session, SessionManager


@pytest.fixture
def wn_settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        logs_dir=tmp_path / "logs",
        roles_dir=tmp_path / "roles",
        skills_dir=tmp_path / "skills",
        api_base_url="http://fake:9999/v1",
        api_key="test",
        model="test",
        max_tokens=1000,
        compact_keep_rounds=1,
        _env_file=None,  # type: ignore[call-arg]
    )


@pytest.fixture
def wn_session(wn_settings: Settings) -> Session:
    mgr = SessionManager(wn_settings)
    mgr.init()
    return mgr.create()


def _make_tc(tool_id: str, name: str, work_node: str) -> ToolCall:
    return ToolCall(
        id=tool_id,
        function_name=name,
        arguments=json.dumps({"command": "echo hi", "work_node": work_node, "next_step": "next"}),
    )


class TestExtractWorkNodes:
    def test_extract_from_tool_calls(self):
        ctx = [
            Message(role=Role.ASSISTANT, tool_calls=[_make_tc("t1", "shell", "1.1")]),
            Message(role=Role.TOOL, content="ok", tool_call_id="t1", name="shell"),
            Message(role=Role.ASSISTANT, tool_calls=[_make_tc("t2", "shell", "2.1")]),
            Message(role=Role.TOOL, content="ok", tool_call_id="t2", name="shell"),
        ]
        nodes = _extract_work_nodes(ctx)
        assert nodes == {"1.1", "2.1"}

    def test_extract_empty_context(self):
        assert _extract_work_nodes([]) == set()

    def test_extract_no_tool_calls(self):
        ctx = [
            Message(role=Role.USER, content="hello"),
            Message(role=Role.ASSISTANT, content="hi"),
        ]
        assert _extract_work_nodes(ctx) == set()


class TestExtractWorkNodesTail:
    def test_tail_keeps_recent(self):
        ctx = [
            Message(role=Role.SYSTEM, content="sys"),
            Message(role=Role.USER, content="old task"),
            Message(role=Role.ASSISTANT, tool_calls=[_make_tc("t1", "shell", "1.1")]),
            Message(role=Role.TOOL, content="ok", tool_call_id="t1", name="shell"),
            Message(role=Role.USER, content="recent task"),
            Message(role=Role.ASSISTANT, tool_calls=[_make_tc("t2", "shell", "2.1")]),
            Message(role=Role.TOOL, content="ok", tool_call_id="t2", name="shell"),
        ]
        recent = _extract_work_nodes_tail(ctx, keep_rounds=1)
        assert "2.1" in recent
        # 1.1 should NOT be in recent since it's before the keep boundary
        assert "1.1" not in recent


class TestGetMessageWorkNode:
    def test_assistant_with_tool_calls(self):
        msg = Message(role=Role.ASSISTANT, tool_calls=[_make_tc("t1", "shell", "3.2")])
        ctx = [msg]
        assert _get_message_work_node(msg, ctx, 0) == "3.2"

    def test_tool_result_maps_to_assistant(self):
        assistant = Message(role=Role.ASSISTANT, tool_calls=[_make_tc("t1", "shell", "1.1")])
        tool = Message(role=Role.TOOL, content="output", tool_call_id="t1", name="shell")
        ctx = [assistant, tool]
        assert _get_message_work_node(tool, ctx, 1) == "1.1"

    def test_user_message_no_node(self):
        msg = Message(role=Role.USER, content="hi")
        assert _get_message_work_node(msg, [msg], 0) is None


class TestWorkNodeCompact:
    @pytest.mark.asyncio
    async def test_compact_specific_nodes(self, wn_session, wn_settings):
        """Compact messages for specific work nodes."""
        wn_session.context = [
            Message(role=Role.SYSTEM, content="system"),
            # Node 1.1
            Message(role=Role.USER, content="task 1"),
            Message(role=Role.ASSISTANT, tool_calls=[_make_tc("t1", "shell", "1.1")]),
            Message(role=Role.TOOL, content="result1", tool_call_id="t1", name="shell"),
            # Node 2.1
            Message(role=Role.USER, content="task 2"),
            Message(role=Role.ASSISTANT, tool_calls=[_make_tc("t2", "shell", "2.1")]),
            Message(role=Role.TOOL, content="result2", tool_call_id="t2", name="shell"),
        ]

        mock_response = LLMResponse()
        mock_response.content = "Summary of node 1.1 work"
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value=mock_response)

        ctx_mgr = ContextManager(wn_session, wn_settings)
        summary = await ctx_mgr.compact_work_nodes(mock_llm, ["1.1"])

        assert summary == "Summary of node 1.1 work"
        # Node 1.1 messages should be replaced by summary
        ctx = wn_session.context
        assert ctx[0].role == Role.SYSTEM
        assert "[Work Node Summary: 1.1]" in ctx[1].content
        # Node 2.1 messages should still be there
        tool_msgs = [m for m in ctx if m.role == Role.TOOL]
        assert any("result2" in m.content for m in tool_msgs)

    @pytest.mark.asyncio
    async def test_compact_no_matching_nodes(self, wn_session, wn_settings):
        wn_session.context = [
            Message(role=Role.SYSTEM, content="system"),
            Message(role=Role.USER, content="hi"),
        ]
        mock_llm = AsyncMock()
        ctx_mgr = ContextManager(wn_session, wn_settings)
        result = await ctx_mgr.compact_work_nodes(mock_llm, ["99.99"])
        assert result is None

    @pytest.mark.asyncio
    async def test_compact_auto_nodes(self, wn_session, wn_settings):
        """Auto compact should compress old nodes, keep recent."""
        wn_session.context = [
            Message(role=Role.SYSTEM, content="system"),
            # Old node
            Message(role=Role.USER, content="old"),
            Message(role=Role.ASSISTANT, tool_calls=[_make_tc("t1", "shell", "1.1")]),
            Message(role=Role.TOOL, content="old_result", tool_call_id="t1", name="shell"),
            # Recent node
            Message(role=Role.USER, content="recent"),
            Message(role=Role.ASSISTANT, tool_calls=[_make_tc("t2", "shell", "2.1")]),
            Message(role=Role.TOOL, content="recent_result", tool_call_id="t2", name="shell"),
        ]

        mock_response = LLMResponse()
        mock_response.content = "Summary of old work"
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value=mock_response)

        ctx_mgr = ContextManager(wn_session, wn_settings)
        summary = await ctx_mgr.compact_work_nodes(mock_llm)

        assert summary is not None
        # Recent node should be preserved
        ctx = wn_session.context
        tool_msgs = [m for m in ctx if m.role == Role.TOOL]
        assert any("recent_result" in m.content for m in tool_msgs)
