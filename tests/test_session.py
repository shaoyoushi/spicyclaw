"""Tests for session management."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spicyclaw.common.types import Message, Role, SessionStatus
from spicyclaw.config import Settings
from spicyclaw.gateway.session import Session, SessionManager


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        logs_dir=tmp_path / "logs",
        _env_file=None,  # type: ignore[call-arg]
    )


@pytest.fixture
def session_mgr(settings: Settings) -> SessionManager:
    mgr = SessionManager(settings)
    mgr.init()
    return mgr


class TestSessionManager:
    def test_create_session(self, session_mgr: SessionManager):
        session = session_mgr.create(model="test-model")
        assert len(session.id) == 12
        assert session.meta.model == "test-model"
        assert session.dir.exists()
        assert (session.dir / "session.json").exists()
        assert (session.dir / "memory").is_dir()
        assert session.workspace.exists()
        assert session.workspace == session.dir / "workspace"

    def test_create_uses_default_model(self, session_mgr: SessionManager):
        session = session_mgr.create()
        assert session.meta.model == session_mgr.settings.model

    def test_get_session(self, session_mgr: SessionManager):
        session = session_mgr.create()
        found = session_mgr.get(session.id)
        assert found is session

    def test_get_nonexistent(self, session_mgr: SessionManager):
        assert session_mgr.get("nonexistent") is None

    def test_list_all(self, session_mgr: SessionManager):
        session_mgr.create()
        session_mgr.create()
        assert len(session_mgr.list_all()) == 2

    def test_persistence_reload(self, settings: Settings):
        # Create and populate
        mgr1 = SessionManager(settings)
        mgr1.init()
        s = mgr1.create(model="m1")
        s.add_message(Message(role=Role.USER, content="hello"))
        s.save_context()
        sid = s.id

        # Reload
        mgr2 = SessionManager(settings)
        mgr2.init()
        s2 = mgr2.get(sid)
        assert s2 is not None
        assert s2.meta.model == "m1"
        assert s2.meta.status == SessionStatus.STOPPED
        assert len(s2.context) == 1
        assert s2.context[0].content == "hello"

    def test_load_skips_non_directory(self, settings: Settings):
        mgr = SessionManager(settings)
        mgr.init()
        # Create a stray file in sessions dir
        (settings.sessions_dir / "stray.txt").write_text("junk")
        # Reload should not crash
        mgr2 = SessionManager(settings)
        mgr2.init()
        assert len(mgr2.sessions) == 0


class TestSession:
    def test_add_message_appends_to_context(self, session_mgr: SessionManager):
        session = session_mgr.create()
        msg = Message(role=Role.USER, content="hi")
        session.add_message(msg)
        assert len(session.context) == 1
        assert session.context[0].content == "hi"

    def test_add_message_writes_history(self, session_mgr: SessionManager):
        session = session_mgr.create()
        session.add_message(Message(role=Role.USER, content="one"))
        session.add_message(Message(role=Role.ASSISTANT, content="two"))

        history = (session.dir / "history.jsonl").read_text()
        lines = [l for l in history.strip().split("\n") if l]
        assert len(lines) == 2
        assert json.loads(lines[0])["content"] == "one"
        assert json.loads(lines[1])["content"] == "two"

    def test_status_setter_updates_timestamp(self, session_mgr: SessionManager):
        session = session_mgr.create()
        old_ts = session.meta.updated_at
        import time
        time.sleep(0.01)
        session.status = SessionStatus.THINKING
        assert session.status == SessionStatus.THINKING
        assert session.meta.updated_at > old_ts

    def test_save_and_load_context(self, session_mgr: SessionManager):
        session = session_mgr.create()
        session.add_message(Message(role=Role.USER, content="test"))
        session.save_context()

        # Clear and reload
        session.context.clear()
        session.load_context()
        assert len(session.context) == 1
        assert session.context[0].content == "test"

    def test_save_meta(self, session_mgr: SessionManager):
        session = session_mgr.create()
        session.meta.title = "My Task"
        session.save_meta()

        data = json.loads((session.dir / "session.json").read_text())
        assert data["title"] == "My Task"
