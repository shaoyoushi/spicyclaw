"""Tests for context manager."""

from __future__ import annotations

from pathlib import Path

import pytest

from spicyclaw.config import Settings
from spicyclaw.gateway.context import ContextManager
from spicyclaw.gateway.session import Session, SessionManager


@pytest.fixture
def ctx_mgr(session: Session, tmp_settings: Settings) -> ContextManager:
    return ContextManager(session, tmp_settings)


class TestContextManager:
    def test_update_tokens(self, ctx_mgr: ContextManager, session: Session):
        ctx_mgr.update_tokens(1000)
        assert session.meta.token_used == 1000

    def test_usage_ratio(self, ctx_mgr: ContextManager):
        ctx_mgr.update_tokens(16384)
        # default max_tokens=32768
        assert abs(ctx_mgr.usage_ratio - 0.5) < 0.01

    def test_should_compact_false(self, ctx_mgr: ContextManager):
        ctx_mgr.update_tokens(1000)
        assert ctx_mgr.should_compact is False

    def test_should_compact_true(self, ctx_mgr: ContextManager, tmp_settings: Settings):
        threshold = int(tmp_settings.max_tokens * tmp_settings.full_compact_ratio)
        ctx_mgr.update_tokens(threshold + 1)
        assert ctx_mgr.should_compact is True

    def test_usage_ratio_zero_max(self, session: Session):
        settings = Settings(
            max_tokens=0,
            data_dir=session.dir.parent.parent,
            _env_file=None,  # type: ignore[call-arg]
        )
        ctx = ContextManager(session, settings)
        assert ctx.usage_ratio == 0.0
