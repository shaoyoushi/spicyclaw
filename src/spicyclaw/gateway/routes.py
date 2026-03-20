"""HTTP + WebSocket endpoints."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from spicyclaw.common.events import ClientEvent, ServerEvent
from spicyclaw.common.i18n import t
from spicyclaw.common.types import Message, Role, SessionMeta
from spicyclaw.config import Settings
from spicyclaw.gateway.context import ContextManager
from spicyclaw.gateway.llm_client import LLMClient
from spicyclaw.gateway.roles import RoleManager
from spicyclaw.gateway.session import SessionManager
from spicyclaw.gateway.tools.base import ToolRegistry
from spicyclaw.gateway.workloop import run_workloop

logger = logging.getLogger(__name__)


class CreateSessionRequest(BaseModel):
    model: str = ""


class SendMessageRequest(BaseModel):
    content: str


def _start_workloop(session, llm, tool_registry, settings) -> bool:
    """Start workloop if not already running. Returns True if started."""
    if session.workloop_task and not session.workloop_task.done():
        return False
    session.abort_event.clear()
    session.workloop_task = asyncio.create_task(
        run_workloop(session, llm, tool_registry, settings),
        name=f"workloop-{session.id}",
    )
    return True


async def _handle_command(
    session,
    command: str,
    args: str,
    llm: LLMClient,
    settings: Settings,
    role_mgr: RoleManager | None = None,
) -> dict:
    """Handle a user command. Returns a response dict."""
    cmd = command.lower().strip()

    if cmd == "help":
        return {"message": t("cmd_help")}

    elif cmd == "yolo":
        session.step_mode = False
        return {"message": t("switched_yolo")}

    elif cmd == "step":
        session.step_mode = True
        return {"message": t("switched_step")}

    elif cmd == "stop":
        session.abort_event.set()
        return {"message": t("aborting")}

    elif cmd == "compact":
        ctx_mgr = ContextManager(session, settings)
        if args.strip():
            node_ids = [n.strip() for n in args.split(",") if n.strip()]
            summary = await ctx_mgr.compact_work_nodes(llm, node_ids)
            if summary:
                return {"message": t("compact_nodes_summary", nodes=node_ids, summary=summary[:200])}
            return {"message": t("compact_nodes_nothing", nodes=node_ids)}
        else:
            # Manual compact: force compress regardless of context size
            summary = await ctx_mgr.full_compact(llm, force=True)
            if summary:
                return {"message": t("compact_summary", summary=summary[:200])}
            return {"message": t("compact_nothing")}

    elif cmd == "status":
        role_name = getattr(session, "role_name", None) or "default"
        return {
            "message": (
                f"Session: {session.id}\n"
                f"Status: {session.status.value}\n"
                f"Model: {session.meta.model}\n"
                f"Title: {session.meta.title}\n"
                f"Tokens: {session.meta.token_used}/{settings.max_tokens}\n"
                f"Mode: {'Step' if session.step_mode else 'YOLO'}\n"
                f"Role: {role_name}\n"
                f"Context messages: {len(session.context)}"
            )
        }

    elif cmd == "task":
        task_file = session.workspace / "TASK.md"
        if task_file.exists():
            return {"message": task_file.read_text(encoding="utf-8")[:2000]}
        return {"message": t("no_task")}

    elif cmd == "plan":
        plan_file = session.workspace / "PLAN.json"
        if plan_file.exists():
            return {"message": plan_file.read_text(encoding="utf-8")[:2000]}
        return {"message": t("no_plan")}

    elif cmd == "session":
        if not args.strip():
            role_name = getattr(session, "role_name", None) or "default"
            available = role_mgr.list_names() if role_mgr else []
            return {
                "message": (
                    f"Session ID: {session.id}\n"
                    f"Title: {session.meta.title}\n"
                    f"Current role: {role_name}\n"
                    f"Available roles: {', '.join(available) or 'none'}"
                )
            }
        # Set role
        role_name = args.strip()
        if role_mgr is None:
            return {"message": t("role_mgr_unavail")}
        role = role_mgr.get(role_name)
        if role is None:
            available = role_mgr.list_names()
            return {"message": t("role_not_found", role=role_name, available=", ".join(available) or "none")}
        session.role_name = role_name
        # Update system prompt if context has one
        if session.context and session.context[0].role == Role.USER:
            # No system prompt yet — insert
            from spicyclaw.gateway.workloop import SYSTEM_PROMPT
            base = SYSTEM_PROMPT.format(work_dir=session.workspace)
            session.context.insert(0, Message(
                role=Role.SYSTEM,
                content=f"{role.system_prompt}\n\n{base}" if role.system_prompt else base,
            ))
        elif session.context and session.context[0].role == Role.SYSTEM:
            from spicyclaw.gateway.workloop import SYSTEM_PROMPT
            base = SYSTEM_PROMPT.format(work_dir=session.workspace)
            session.context[0] = Message(
                role=Role.SYSTEM,
                content=f"{role.system_prompt}\n\n{base}" if role.system_prompt else base,
            )
        return {"message": t("role_set", role=role_name)}

    elif cmd == "settings":
        return {
            "message": (
                f"Model: {settings.model}\n"
                f"API: {settings.api_base_url}\n"
                f"Max tokens: {settings.max_tokens}\n"
                f"Max steps: {settings.max_steps}\n"
                f"YOLO: {settings.yolo}\n"
                f"Shell timeout: {settings.shell_timeout}s\n"
                f"Idle timeout: {settings.idle_timeout}s\n"
                f"Compact ratio: {settings.full_compact_ratio}\n"
                f"Keep rounds: {settings.compact_keep_rounds}"
            )
        }

    elif cmd == "resume":
        if session.workloop_task and not session.workloop_task.done():
            return {"message": t("resume_running")}
        if not session.context:
            return {"message": t("resume_empty")}
        return {"message": t("resuming"), "_action": "resume"}

    else:
        return {"message": t("unknown_cmd", cmd=cmd)}


def setup_routes(
    session_mgr: SessionManager,
    llm: LLMClient,
    tool_registry: ToolRegistry,
    settings: Settings,
    role_mgr: RoleManager | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.post("/sessions", response_model=SessionMeta)
    async def create_session(req: CreateSessionRequest | None = None):
        req = req or CreateSessionRequest()
        session = session_mgr.create(model=req.model)
        return session.meta

    @router.get("/sessions", response_model=list[SessionMeta])
    async def list_sessions():
        return session_mgr.list_all()

    @router.get("/sessions/{session_id}", response_model=SessionMeta)
    async def get_session(session_id: str):
        session = session_mgr.get(session_id)
        if not session:
            raise HTTPException(404, "Session not found")
        return session.meta

    @router.post("/sessions/{session_id}/message")
    async def send_message(session_id: str, req: SendMessageRequest):
        """Send a message and start the workloop (HTTP API)."""
        session = session_mgr.get(session_id)
        if not session:
            raise HTTPException(404, "Session not found")

        if session.workloop_task and not session.workloop_task.done():
            raise HTTPException(409, "Workloop already running")

        user_msg = Message(role=Role.USER, content=req.content)
        session.add_message(user_msg)

        _start_workloop(session, llm, tool_registry, settings)
        return {"status": "started", "session_id": session_id}

    @router.post("/sessions/{session_id}/abort")
    async def abort_session(session_id: str):
        session = session_mgr.get(session_id)
        if not session:
            raise HTTPException(404, "Session not found")
        session.abort_event.set()
        return {"status": "aborting"}

    @router.get("/sessions/{session_id}/context")
    async def get_context(session_id: str):
        session = session_mgr.get(session_id)
        if not session:
            raise HTTPException(404, "Session not found")
        return [m.model_dump() for m in session.context]

    @router.websocket("/sessions/{session_id}/ws")
    async def session_websocket(websocket: WebSocket, session_id: str):
        session = session_mgr.get(session_id)
        if not session:
            await websocket.close(code=4004, reason="Session not found")
            return

        await websocket.accept()
        session.subscribers.add(websocket)
        logger.info("WebSocket connected for session %s", session_id)

        try:
            # Send current status on connect
            await websocket.send_text(
                ServerEvent(
                    type="status",
                    session_id=session_id,
                    data={"status": session.status.value},
                ).model_dump_json()
            )

            while True:
                raw = await websocket.receive_text()
                try:
                    event = ClientEvent.model_validate_json(raw)
                except Exception:
                    await websocket.send_text(
                        ServerEvent(
                            type="error",
                            session_id=session_id,
                            data={"message": "Invalid event format"},
                        ).model_dump_json()
                    )
                    continue

                if event.type == "message":
                    content = event.data.get("content", "").strip()
                    if not content:
                        continue
                    if session.workloop_task and not session.workloop_task.done():
                        # Allow sending messages while workloop is running —
                        # add to context so the agent sees it on next LLM call
                        msg = Message(role=Role.USER, content=content)
                        session.add_message(msg)
                        await websocket.send_text(
                            ServerEvent(
                                type="system",
                                session_id=session_id,
                                data={"message": t("msg_queued")},
                            ).model_dump_json()
                        )
                        continue
                    msg = Message(role=Role.USER, content=content)
                    session.add_message(msg)
                    _start_workloop(session, llm, tool_registry, settings)

                elif event.type == "abort":
                    session.abort_event.set()

                elif event.type == "confirm":
                    session.confirm_event.set()

                elif event.type == "command":
                    cmd = event.data.get("command", "")
                    cmd_args = event.data.get("args", "")
                    result = await _handle_command(
                        session, cmd, cmd_args, llm, settings, role_mgr
                    )
                    # Handle special actions
                    action = result.pop("_action", None)
                    await websocket.send_text(
                        ServerEvent(
                            type="system",
                            session_id=session_id,
                            data=result,
                        ).model_dump_json()
                    )
                    if action == "resume":
                        _start_workloop(session, llm, tool_registry, settings)

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected for session %s", session_id)
        finally:
            session.subscribers.discard(websocket)

    return router
