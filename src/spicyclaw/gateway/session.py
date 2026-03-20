"""Session management and persistence."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from spicyclaw.common.events import ServerEvent
from spicyclaw.common.types import Message, Role, SessionMeta, SessionStatus
from spicyclaw.config import Settings

logger = logging.getLogger(__name__)


class Session:
    def __init__(self, meta: SessionMeta, base_dir: Path) -> None:
        self.meta = meta
        self.dir = base_dir / meta.id
        self.context: list[Message] = []
        self.subscribers: set[Any] = set()  # WebSocket connections
        self.workloop_task: asyncio.Task[None] | None = None
        self.abort_event = asyncio.Event()
        self.confirm_event = asyncio.Event()
        self.step_mode: bool = False
        self.role_name: str | None = None

    @property
    def id(self) -> str:
        return self.meta.id

    @property
    def status(self) -> SessionStatus:
        return self.meta.status

    @status.setter
    def status(self, value: SessionStatus) -> None:
        self.meta.status = value
        self.meta.updated_at = time.time()

    def add_message(self, msg: Message) -> None:
        self.context.append(msg)
        self._append_history(msg)

    def _append_history(self, msg: Message) -> None:
        history_file = self.dir / "history.jsonl"
        with open(history_file, "a", encoding="utf-8") as f:
            f.write(msg.model_dump_json() + "\n")

    def save_context(self) -> None:
        context_file = self.dir / "context.json"
        data = [m.model_dump() for m in self.context]
        context_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def save_meta(self) -> None:
        meta_file = self.dir / "session.json"
        meta_file.write_text(self.meta.model_dump_json(indent=2), encoding="utf-8")

    def load_context(self) -> None:
        context_file = self.dir / "context.json"
        if context_file.exists():
            data = json.loads(context_file.read_text(encoding="utf-8"))
            self.context = [Message.model_validate(m) for m in data]

    async def broadcast(self, event: ServerEvent) -> None:
        dead: list[Any] = []
        for ws in self.subscribers:
            try:
                await ws.send_text(event.model_dump_json())
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.subscribers.discard(ws)


class SessionManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.sessions: dict[str, Session] = {}
        self._base_dir = settings.sessions_dir

    def init(self) -> None:
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._load_existing()

    def _load_existing(self) -> None:
        for d in sorted(d for d in self._base_dir.iterdir() if d.is_dir()):
            meta_file = d / "session.json"
            if not meta_file.exists():
                continue
            try:
                meta = SessionMeta.model_validate_json(meta_file.read_text(encoding="utf-8"))
                meta.status = SessionStatus.STOPPED
                session = Session(meta, self._base_dir)
                session.load_context()
                self.sessions[meta.id] = session
                logger.info("Loaded session %s: %s", meta.id, meta.title)
            except Exception:
                logger.exception("Failed to load session from %s", d)

    def create(self, model: str = "") -> Session:
        sid = uuid.uuid4().hex[:12]
        meta = SessionMeta(id=sid, model=model or self.settings.model)
        session = Session(meta, self._base_dir)
        session.dir.mkdir(parents=True, exist_ok=True)
        (session.dir / "memory").mkdir(exist_ok=True)
        session.save_meta()
        self.sessions[sid] = session
        logger.info("Created session %s", sid)
        return session

    def get(self, session_id: str) -> Session | None:
        return self.sessions.get(session_id)

    def list_all(self) -> list[SessionMeta]:
        return [s.meta for s in self.sessions.values()]

    def get_recoverable(self) -> list[Session]:
        """Return sessions that were interrupted (have context but status is stopped).

        A session is recoverable if it has context messages and was likely
        interrupted mid-execution (has pending tool calls without results).
        """
        recoverable: list[Session] = []
        for session in self.sessions.values():
            if not session.context:
                continue
            # Check if last assistant message has tool_calls without matching results
            last_assistant = None
            for msg in reversed(session.context):
                if msg.role == Role.ASSISTANT and msg.tool_calls:
                    last_assistant = msg
                    break
                if msg.role == Role.ASSISTANT:
                    break
            if last_assistant and last_assistant.tool_calls:
                # Check if all tool_calls have results
                tc_ids = {tc.id for tc in last_assistant.tool_calls}
                result_ids = {
                    msg.tool_call_id
                    for msg in session.context
                    if msg.role == Role.TOOL and msg.tool_call_id
                }
                if not tc_ids.issubset(result_ids):
                    recoverable.append(session)
        return recoverable
