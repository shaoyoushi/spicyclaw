"""Tests for session recovery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spicyclaw.common.types import Message, Role, SessionMeta, ToolCall
from spicyclaw.config import Settings
from spicyclaw.gateway.session import Session, SessionManager


@pytest.fixture
def recovery_settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        logs_dir=tmp_path / "logs",
        roles_dir=tmp_path / "roles",
        skills_dir=tmp_path / "skills",
        api_base_url="http://fake:9999/v1",
        api_key="test",
        model="test",
        _env_file=None,  # type: ignore[call-arg]
    )


class TestSessionRecovery:
    def test_no_recoverable_empty(self, recovery_settings):
        mgr = SessionManager(recovery_settings)
        mgr.init()
        assert mgr.get_recoverable() == []

    def test_no_recoverable_clean_session(self, recovery_settings):
        mgr = SessionManager(recovery_settings)
        mgr.init()
        session = mgr.create()
        session.add_message(Message(role=Role.USER, content="hello"))
        session.add_message(Message(role=Role.ASSISTANT, content="hi"))
        assert mgr.get_recoverable() == []

    def test_recoverable_pending_tool(self, recovery_settings):
        """Session with tool_calls but no matching tool results is recoverable."""
        mgr = SessionManager(recovery_settings)
        mgr.init()
        session = mgr.create()

        tc = ToolCall(
            id="tc1",
            function_name="shell",
            arguments=json.dumps({"command": "echo hi"}),
        )
        session.add_message(Message(role=Role.USER, content="task"))
        session.add_message(Message(
            role=Role.ASSISTANT,
            tool_calls=[tc],
        ))
        # No tool result message — indicates interrupted execution

        recoverable = mgr.get_recoverable()
        assert len(recoverable) == 1
        assert recoverable[0].id == session.id

    def test_not_recoverable_complete_tool(self, recovery_settings):
        """Session with matching tool results is NOT recoverable."""
        mgr = SessionManager(recovery_settings)
        mgr.init()
        session = mgr.create()

        tc = ToolCall(
            id="tc1",
            function_name="shell",
            arguments=json.dumps({"command": "echo hi"}),
        )
        session.add_message(Message(role=Role.USER, content="task"))
        session.add_message(Message(role=Role.ASSISTANT, tool_calls=[tc]))
        session.add_message(Message(
            role=Role.TOOL,
            content="hi",
            tool_call_id="tc1",
            name="shell",
        ))

        assert mgr.get_recoverable() == []

    def test_recoverable_persists_across_restart(self, recovery_settings):
        """Test that recovery works after simulating restart."""
        mgr = SessionManager(recovery_settings)
        mgr.init()
        session = mgr.create()

        tc = ToolCall(
            id="tc1",
            function_name="shell",
            arguments=json.dumps({"command": "echo hi"}),
        )
        session.add_message(Message(role=Role.USER, content="task"))
        session.add_message(Message(role=Role.ASSISTANT, tool_calls=[tc]))
        session.save_context()
        session.save_meta()

        # Simulate restart
        mgr2 = SessionManager(recovery_settings)
        mgr2.init()
        recoverable = mgr2.get_recoverable()
        assert len(recoverable) == 1

    def test_role_name_attribute(self, recovery_settings):
        mgr = SessionManager(recovery_settings)
        mgr.init()
        session = mgr.create()
        assert session.role_name is None
        session.role_name = "programmer"
        assert session.role_name == "programmer"
