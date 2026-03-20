"""Tests for context compression."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from spicyclaw.common.types import Message, Role
from spicyclaw.config import Settings
from spicyclaw.gateway.context import ContextManager, _messages_to_text
from spicyclaw.gateway.llm_client import LLMResponse
from spicyclaw.gateway.session import Session, SessionManager


@pytest.fixture
def compact_settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        logs_dir=tmp_path / "logs",
        roles_dir=tmp_path / "roles",
        skills_dir=tmp_path / "skills",
        api_base_url="http://fake:9999/v1",
        api_key="test",
        model="test",
        max_tokens=1000,
        full_compact_ratio=0.8,
        compact_keep_rounds=2,
        _env_file=None,  # type: ignore[call-arg]
    )


@pytest.fixture
def compact_session(compact_settings: Settings) -> Session:
    mgr = SessionManager(compact_settings)
    mgr.init()
    return mgr.create()


class TestContextManager:
    def test_usage_ratio(self, compact_session, compact_settings):
        ctx_mgr = ContextManager(compact_session, compact_settings)
        ctx_mgr.update_tokens(400)
        assert ctx_mgr.usage_ratio == 0.4
        assert not ctx_mgr.should_compact

    def test_should_compact_at_threshold(self, compact_session, compact_settings):
        ctx_mgr = ContextManager(compact_session, compact_settings)
        ctx_mgr.update_tokens(800)
        assert ctx_mgr.should_compact

    @pytest.mark.asyncio
    async def test_full_compact_too_short(self, compact_session, compact_settings):
        ctx_mgr = ContextManager(compact_session, compact_settings)
        compact_session.context = [
            Message(role=Role.SYSTEM, content="sys"),
            Message(role=Role.USER, content="hi"),
        ]
        mock_llm = AsyncMock()
        result = await ctx_mgr.full_compact(mock_llm)
        assert result is None

    @pytest.mark.asyncio
    async def test_full_compact_success(self, compact_session, compact_settings):
        """Compress middle messages, keep system + recent rounds."""
        # Build a context with system + old messages + 2 recent rounds
        compact_session.context = [
            Message(role=Role.SYSTEM, content="system prompt"),
            Message(role=Role.USER, content="old task 1"),
            Message(role=Role.ASSISTANT, content="old response 1"),
            Message(role=Role.USER, content="old task 2"),
            Message(role=Role.ASSISTANT, content="old response 2"),
            Message(role=Role.USER, content="old task 3"),
            Message(role=Role.ASSISTANT, content="old response 3"),
            # Recent round 1
            Message(role=Role.USER, content="recent 1"),
            Message(role=Role.ASSISTANT, content="recent reply 1"),
            # Recent round 2
            Message(role=Role.USER, content="recent 2"),
            Message(role=Role.ASSISTANT, content="recent reply 2"),
        ]

        # Mock LLM to return a summary
        mock_response = LLMResponse()
        mock_response.content = "Summary of old tasks 1-3"

        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value=mock_response)

        ctx_mgr = ContextManager(compact_session, compact_settings)
        result = await ctx_mgr.full_compact(mock_llm)

        assert result == "Summary of old tasks 1-3"
        # Context should be: system + summary + recent rounds
        ctx = compact_session.context
        assert ctx[0].role == Role.SYSTEM
        assert ctx[0].content == "system prompt"
        assert "[Context Summary]" in ctx[1].content
        assert "Summary of old tasks" in ctx[1].content
        # Recent rounds preserved
        user_msgs = [m for m in ctx if m.role == Role.USER]
        assert len(user_msgs) == 2
        assert user_msgs[0].content == "recent 1"
        assert user_msgs[1].content == "recent 2"

    @pytest.mark.asyncio
    async def test_full_compact_llm_failure(self, compact_session, compact_settings):
        compact_session.context = [
            Message(role=Role.SYSTEM, content="sys"),
            Message(role=Role.USER, content="old 1"),
            Message(role=Role.ASSISTANT, content="resp 1"),
            Message(role=Role.USER, content="old 2"),
            Message(role=Role.ASSISTANT, content="resp 2"),
            Message(role=Role.USER, content="recent"),
            Message(role=Role.ASSISTANT, content="recent reply"),
        ]

        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(side_effect=Exception("LLM down"))

        ctx_mgr = ContextManager(compact_session, compact_settings)
        result = await ctx_mgr.full_compact(mock_llm)
        assert result is None
        # Context should be unchanged
        assert len(compact_session.context) == 7


class TestMessagesToText:
    def test_basic(self):
        msgs = [
            Message(role=Role.USER, content="hello"),
            Message(role=Role.ASSISTANT, content="world"),
        ]
        text = _messages_to_text(msgs)
        assert "[USER] hello" in text
        assert "[ASSISTANT] world" in text

    def test_tool_message(self):
        msgs = [
            Message(role=Role.TOOL, content="output here", name="shell", tool_call_id="tc1"),
        ]
        text = _messages_to_text(msgs)
        assert "[TOOL/shell]" in text
