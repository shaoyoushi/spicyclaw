"""Tests for common types."""

from spicyclaw.common.types import Message, Role, SessionMeta, SessionStatus, ToolCall, ToolResult
from spicyclaw.common.events import ServerEvent, ClientEvent


class TestMessage:
    def test_user_message_to_openai(self):
        msg = Message(role=Role.USER, content="hello")
        d = msg.to_openai()
        assert d == {"role": "user", "content": "hello"}

    def test_assistant_message_with_tool_calls(self):
        tc = ToolCall(id="tc_1", function_name="shell", arguments='{"cmd": "ls"}')
        msg = Message(role=Role.ASSISTANT, content="Let me check.", tool_calls=[tc])
        d = msg.to_openai()
        assert d["role"] == "assistant"
        assert d["content"] == "Let me check."
        assert len(d["tool_calls"]) == 1
        assert d["tool_calls"][0]["function"]["name"] == "shell"

    def test_tool_message_to_openai(self):
        msg = Message(role=Role.TOOL, content="file1.txt\nfile2.txt", tool_call_id="tc_1", name="shell")
        d = msg.to_openai()
        assert d["role"] == "tool"
        assert d["tool_call_id"] == "tc_1"
        assert d["name"] == "shell"

    def test_minimal_message_omits_none_fields(self):
        msg = Message(role=Role.SYSTEM, content="You are an agent.")
        d = msg.to_openai()
        assert "tool_calls" not in d
        assert "tool_call_id" not in d
        assert "name" not in d

    def test_message_roundtrip_json(self):
        msg = Message(role=Role.USER, content="test")
        restored = Message.model_validate_json(msg.model_dump_json())
        assert restored.role == Role.USER
        assert restored.content == "test"


class TestToolCall:
    def test_to_openai(self):
        tc = ToolCall(id="call_abc", function_name="stop", arguments='{}')
        d = tc.to_openai()
        assert d["id"] == "call_abc"
        assert d["type"] == "function"
        assert d["function"]["name"] == "stop"
        assert d["function"]["arguments"] == "{}"


class TestToolResult:
    def test_defaults(self):
        r = ToolResult(output="ok")
        assert r.error == ""
        assert r.return_code == 0
        assert r.truncated is False


class TestSessionMeta:
    def test_defaults(self):
        meta = SessionMeta(id="abc123")
        assert meta.title == "New Session"
        assert meta.status == SessionStatus.STOPPED
        assert meta.token_used == 0
        assert meta.created_at > 0

    def test_json_roundtrip(self):
        meta = SessionMeta(id="x", title="Test", model="gpt-4")
        restored = SessionMeta.model_validate_json(meta.model_dump_json())
        assert restored.id == "x"
        assert restored.title == "Test"


class TestEvents:
    def test_server_event(self):
        ev = ServerEvent(type="chunk", session_id="s1", data={"text": "hi"})
        assert ev.type == "chunk"
        assert ev.ts > 0
        d = ev.model_dump()
        assert d["session_id"] == "s1"

    def test_client_event(self):
        ev = ClientEvent(type="message", session_id="s1", data={"content": "hello"})
        assert ev.type == "message"
