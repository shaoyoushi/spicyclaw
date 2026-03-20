"""Tests for WebSocket event handling."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from starlette.testclient import TestClient

from spicyclaw.common.events import ClientEvent
from spicyclaw.common.types import Message, Role, SessionStatus, ToolCall
from spicyclaw.config import Settings
from spicyclaw.gateway.llm_client import LLMResponse
from spicyclaw.gateway.session import Session
from spicyclaw.gateway.tools.base import ToolRegistry
from spicyclaw.gateway.workloop import run_workloop


class TestWebSocket:
    def test_ws_connect_receives_status(self, app):
        """WebSocket connection receives initial status event."""
        client = TestClient(app)
        # Create a session first
        resp = client.post("/api/sessions", json={})
        sid = resp.json()["id"]

        with client.websocket_connect(f"/api/sessions/{sid}/ws") as ws:
            data = ws.receive_json()
            assert data["type"] == "status"
            assert data["session_id"] == sid
            assert data["data"]["status"] == "stopped"

    def test_ws_invalid_session(self, app):
        """WebSocket rejects connection for invalid session."""
        client = TestClient(app)
        with pytest.raises(Exception):
            with client.websocket_connect("/api/sessions/nonexistent/ws") as ws:
                pass

    def test_ws_invalid_event_format(self, app):
        """Invalid JSON event returns error."""
        client = TestClient(app)
        resp = client.post("/api/sessions", json={})
        sid = resp.json()["id"]

        with client.websocket_connect(f"/api/sessions/{sid}/ws") as ws:
            _ = ws.receive_json()  # initial status
            ws.send_text("not valid json{{{")
            data = ws.receive_json()
            assert data["type"] == "error"
            assert "Invalid" in data["data"]["message"]

    def test_ws_empty_message_ignored(self, app):
        """Empty message content is silently ignored."""
        client = TestClient(app)
        resp = client.post("/api/sessions", json={})
        sid = resp.json()["id"]

        with client.websocket_connect(f"/api/sessions/{sid}/ws") as ws:
            _ = ws.receive_json()  # initial status
            ws.send_json({
                "type": "message",
                "session_id": sid,
                "data": {"content": "   "},
            })
            # No response expected for empty message; send abort to verify
            # the connection is still alive
            ws.send_json({
                "type": "abort",
                "session_id": sid,
                "data": {},
            })
            # Connection should still be alive (no error response)


class TestWebSocketWorkloop:
    def test_ws_message_starts_workloop(self, app):
        """Sending a message via WebSocket starts the workloop."""
        client = TestClient(app)
        resp = client.post("/api/sessions", json={})
        sid = resp.json()["id"]

        with client.websocket_connect(f"/api/sessions/{sid}/ws") as ws:
            _ = ws.receive_json()  # initial status

            # The workloop will fail because fake LLM is not reachable,
            # but we can verify the message was added and an error event comes back
            ws.send_json({
                "type": "message",
                "session_id": sid,
                "data": {"content": "hello agent"},
            })

            # Should receive events (error due to fake LLM, then status:stopped)
            events = []
            for _ in range(10):
                try:
                    data = ws.receive_json(mode="text")
                    events.append(data)
                    if data["type"] == "status" and data["data"].get("status") == "stopped":
                        break
                except Exception:
                    break

            event_types = [e["type"] for e in events]
            # Should end with status:stopped
            assert "status" in event_types

        # Verify message was added to context
        ctx_resp = client.get(f"/api/sessions/{sid}/context")
        ctx = ctx_resp.json()
        user_msgs = [m for m in ctx if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0]["content"] == "hello agent"

    def test_ws_abort_event(self, app):
        """Abort event sets the session's abort flag."""
        client = TestClient(app)
        resp = client.post("/api/sessions", json={})
        sid = resp.json()["id"]

        session = app.state.session_mgr.get(sid)

        with client.websocket_connect(f"/api/sessions/{sid}/ws") as ws:
            _ = ws.receive_json()
            ws.send_json({
                "type": "abort",
                "session_id": sid,
                "data": {},
            })

        assert session.abort_event.is_set()


@pytest.mark.asyncio
class TestWorkloopBroadcast:
    async def test_title_update_broadcasts_session_update(
        self, session: Session, tool_registry: ToolRegistry, tmp_settings: Settings
    ):
        """TASK.md creation triggers session_update broadcast."""
        session.add_message(Message(role=Role.USER, content="test"))

        # Create TASK.md via shell, then stop
        shell_call = ToolCall(
            id="tc_1",
            function_name="shell",
            arguments=json.dumps({
                "command": f"echo '# My Title' > {session.dir}/TASK.md",
                "work_node": "1",
                "next_step": "stop",
            }),
        )
        stop_call = ToolCall(
            id="tc_2",
            function_name="stop",
            arguments=json.dumps({
                "reason": "done",
                "work_node": "2",
                "next_step": "none",
            }),
        )

        async def stream_chat(messages, tools=None):
            nonlocal call_count
            if call_count == 0:
                call_count += 1
                resp = LLMResponse()
                resp.tool_calls = [shell_call]
                resp.usage_tokens = 100
                yield "done", resp
            else:
                resp = LLMResponse()
                resp.tool_calls = [stop_call]
                resp.usage_tokens = 200
                yield "done", resp

        call_count = 0
        mock_llm = AsyncMock()
        mock_llm.stream_chat = stream_chat

        # Track broadcasted events
        events = []
        mock_ws = AsyncMock()
        mock_ws.send_text = AsyncMock(side_effect=lambda t: events.append(json.loads(t)))
        session.subscribers.add(mock_ws)

        await run_workloop(session, mock_llm, tool_registry, tmp_settings)

        session_updates = [e for e in events if e["type"] == "session_update"]
        assert len(session_updates) == 1
        assert session_updates[0]["data"]["title"] == "My Title"
