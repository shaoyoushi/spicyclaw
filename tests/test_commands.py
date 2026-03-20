"""Tests for user command handling."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from spicyclaw.common.i18n import set_lang
from spicyclaw.common.types import Message, Role, SessionMeta
from spicyclaw.config import Settings
from spicyclaw.gateway.llm_client import LLMResponse
from spicyclaw.gateway.roles import RoleManager
from spicyclaw.gateway.routes import _handle_command
from spicyclaw.gateway.session import Session, SessionManager


@pytest.fixture
def cmd_settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        logs_dir=tmp_path / "logs",
        roles_dir=tmp_path / "roles",
        skills_dir=tmp_path / "skills",
        api_base_url="http://fake:9999/v1",
        api_key="test",
        model="test-model",
        max_tokens=32768,
        _env_file=None,  # type: ignore[call-arg]
    )


@pytest.fixture
def cmd_session(cmd_settings: Settings) -> Session:
    mgr = SessionManager(cmd_settings)
    mgr.init()
    return mgr.create()


@pytest.fixture(autouse=True)
def reset_lang():
    set_lang("en")
    yield
    set_lang("en")


@pytest.mark.asyncio
class TestCommands:
    async def test_help(self, cmd_session, cmd_settings):
        result = await _handle_command(cmd_session, "help", "", None, cmd_settings)
        assert "/help" in result["message"]
        assert "/yolo" in result["message"]
        assert "/step" in result["message"]
        assert "/resume" in result["message"]

    async def test_yolo(self, cmd_session, cmd_settings):
        cmd_session.step_mode = True
        result = await _handle_command(cmd_session, "yolo", "", None, cmd_settings)
        assert "YOLO" in result["message"]
        assert not cmd_session.step_mode

    async def test_step(self, cmd_session, cmd_settings):
        result = await _handle_command(cmd_session, "step", "", None, cmd_settings)
        assert "Step" in result["message"]
        assert cmd_session.step_mode

    async def test_stop(self, cmd_session, cmd_settings):
        result = await _handle_command(cmd_session, "stop", "", None, cmd_settings)
        assert cmd_session.abort_event.is_set()

    async def test_status(self, cmd_session, cmd_settings):
        result = await _handle_command(cmd_session, "status", "", None, cmd_settings)
        msg = result["message"]
        assert cmd_session.id in msg
        assert "stopped" in msg
        assert "test-model" in msg
        assert "Role:" in msg

    async def test_task_no_file(self, cmd_session, cmd_settings):
        result = await _handle_command(cmd_session, "task", "", None, cmd_settings)
        assert "TASK.md" in result["message"]

    async def test_task_with_file(self, cmd_session, cmd_settings):
        cmd_session.workspace.mkdir(parents=True, exist_ok=True)
        (cmd_session.workspace / "TASK.md").write_text("# My Task\nDo things")
        result = await _handle_command(cmd_session, "task", "", None, cmd_settings)
        assert "My Task" in result["message"]

    async def test_plan_no_file(self, cmd_session, cmd_settings):
        result = await _handle_command(cmd_session, "plan", "", None, cmd_settings)
        assert "PLAN.json" in result["message"]

    async def test_plan_with_file(self, cmd_session, cmd_settings):
        cmd_session.workspace.mkdir(parents=True, exist_ok=True)
        (cmd_session.workspace / "PLAN.json").write_text('{"nodes": []}')
        result = await _handle_command(cmd_session, "plan", "", None, cmd_settings)
        assert "nodes" in result["message"]

    async def test_compact_empty(self, cmd_session, cmd_settings):
        mock_llm = AsyncMock()
        result = await _handle_command(cmd_session, "compact", "", mock_llm, cmd_settings)
        assert "compact" in result["message"].lower() or "压缩" in result["message"]

    async def test_unknown_command(self, cmd_session, cmd_settings):
        result = await _handle_command(cmd_session, "foobar", "", None, cmd_settings)
        assert "/foobar" in result["message"]

    async def test_settings(self, cmd_session, cmd_settings):
        result = await _handle_command(cmd_session, "settings", "", None, cmd_settings)
        msg = result["message"]
        assert "test-model" in msg
        assert "32768" in msg

    async def test_session_info(self, cmd_session, cmd_settings):
        result = await _handle_command(cmd_session, "session", "", None, cmd_settings)
        assert cmd_session.id in result["message"]

    async def test_session_set_role(self, cmd_session, cmd_settings, tmp_path):
        roles_dir = tmp_path / "roles"
        roles_dir.mkdir(exist_ok=True)
        (roles_dir / "dev.yaml").write_text("name: dev\nsystem_prompt: You are a dev.\n")
        role_mgr = RoleManager()
        role_mgr.load_dir(roles_dir)

        # Set the role
        result = await _handle_command(
            cmd_session, "session", "dev", None, cmd_settings, role_mgr
        )
        assert "dev" in result["message"]
        assert cmd_session.role_name == "dev"

    async def test_session_set_unknown_role(self, cmd_session, cmd_settings):
        role_mgr = RoleManager()
        result = await _handle_command(
            cmd_session, "session", "nonexistent", None, cmd_settings, role_mgr
        )
        assert "not found" in result["message"].lower() or "未找到" in result["message"]

    async def test_resume_empty(self, cmd_session, cmd_settings):
        result = await _handle_command(cmd_session, "resume", "", None, cmd_settings)
        assert "resume" in result["message"].lower() or "恢复" in result["message"]

    async def test_resume_with_context(self, cmd_session, cmd_settings):
        cmd_session.context = [Message(role=Role.USER, content="hi")]
        result = await _handle_command(cmd_session, "resume", "", None, cmd_settings)
        assert result.get("_action") == "resume"

    async def test_chinese_commands(self, cmd_session, cmd_settings):
        set_lang("zh")
        result = await _handle_command(cmd_session, "help", "", None, cmd_settings)
        assert "显示帮助" in result["message"]

        result = await _handle_command(cmd_session, "yolo", "", None, cmd_settings)
        assert "YOLO" in result["message"]
