"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from spicyclaw.config import Settings
from spicyclaw.gateway.server import create_app
from spicyclaw.gateway.session import Session, SessionManager
from spicyclaw.gateway.tools.base import ToolRegistry
from spicyclaw.gateway.tools.shell import ShellTool
from spicyclaw.gateway.tools.stop import StopTool


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    """Settings with temp directories, no .env file loading."""
    return Settings(
        data_dir=tmp_path / "data",
        logs_dir=tmp_path / "logs",
        roles_dir=tmp_path / "roles",
        skills_dir=tmp_path / "skills",
        api_base_url="http://fake-llm:9999/v1",
        api_key="test-key",
        model="test-model",
        _env_file=None,  # type: ignore[call-arg]
    )


@pytest.fixture
def app(tmp_settings: Settings):
    return create_app(tmp_settings)


@pytest_asyncio.fixture
async def client(app) -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def tool_registry(tmp_settings: Settings) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ShellTool(
        timeout=tmp_settings.shell_timeout,
        max_output=tmp_settings.shell_max_output,
    ))
    registry.register(StopTool())
    return registry


@pytest.fixture
def session_mgr(tmp_settings: Settings) -> SessionManager:
    mgr = SessionManager(tmp_settings)
    mgr.init()
    return mgr


@pytest.fixture
def session(session_mgr: SessionManager) -> Session:
    return session_mgr.create(model="test-model")
