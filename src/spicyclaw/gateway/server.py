"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from spicyclaw.config import Settings
from spicyclaw.gateway.llm_client import LLMClient
from spicyclaw.gateway.roles import RoleManager
from spicyclaw.gateway.routes import setup_routes
from spicyclaw.gateway.session import SessionManager
from spicyclaw.gateway.skills import SkillManager
from spicyclaw.gateway.tools.base import ToolRegistry
from spicyclaw.gateway.tools.memory import MemoryReadTool, MemoryWriteTool
from spicyclaw.gateway.tools.shell import ShellTool
from spicyclaw.gateway.tools.stop import StopTool
from spicyclaw.gateway.tools.summary import SummaryTool
from spicyclaw.ui.web.router import get_static_files, router as ui_router

logger = logging.getLogger(__name__)


def create_tool_registry(settings: Settings) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ShellTool(
        timeout=settings.shell_timeout,
        max_output=settings.shell_max_output,
    ))
    registry.register(StopTool())
    registry.register(SummaryTool())
    registry.register(MemoryReadTool())
    registry.register(MemoryWriteTool())

    # Load skills as tools
    skill_mgr = SkillManager()
    skill_mgr.load_dir(settings.skills_dir)
    for skill_tool in skill_mgr.to_tools():
        registry.register(skill_tool)

    return registry


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    session_mgr = SessionManager(settings)
    llm = LLMClient(settings)
    tool_registry = create_tool_registry(settings)

    # Load roles
    role_mgr = RoleManager()
    role_mgr.load_dir(settings.roles_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        settings.logs_dir.mkdir(parents=True, exist_ok=True)
        session_mgr.init()

        # Log recoverable sessions (don't auto-resume — user decides)
        recoverable = session_mgr.get_recoverable()
        if recoverable:
            logger.info(
                "Found %d recoverable session(s): %s",
                len(recoverable),
                [s.id for s in recoverable],
            )

        logger.info(
            "SpicyClaw started — model=%s, api=%s, tools=%s, roles=%s",
            settings.model,
            settings.api_base_url,
            tool_registry.names,
            role_mgr.list_names(),
        )
        yield
        await llm.close()
        logger.info("SpicyClaw stopped")

    app = FastAPI(title="SpicyClaw", version="0.1.0", lifespan=lifespan)

    # API routes
    api_router = setup_routes(session_mgr, llm, tool_registry, settings, role_mgr)
    app.include_router(api_router)

    # Web UI
    app.include_router(ui_router)
    app.mount("/static", get_static_files(), name="static")

    # Store refs for tests
    app.state.settings = settings
    app.state.session_mgr = session_mgr
    app.state.llm = llm
    app.state.tool_registry = tool_registry
    app.state.role_mgr = role_mgr

    return app
