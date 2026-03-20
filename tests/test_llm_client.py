"""Tests for LLM client."""

from __future__ import annotations

import json

import pytest

from spicyclaw.gateway.llm_client import LLMResponse


class TestLLMResponse:
    def test_content_accumulation(self):
        r = LLMResponse()
        r.content = "hello "
        r.content += "world"
        assert r.content == "hello world"

    def test_tool_call_delta_single(self):
        r = LLMResponse()
        # First delta: id + function name
        r._feed_tool_call_delta(0, {
            "id": "call_1",
            "function": {"name": "shell", "arguments": '{"cm'}
        })
        # Second delta: more arguments
        r._feed_tool_call_delta(0, {
            "function": {"arguments": 'd": "ls"}'}
        })
        r._finalize_tool_calls()

        assert len(r.tool_calls) == 1
        tc = r.tool_calls[0]
        assert tc.id == "call_1"
        assert tc.function_name == "shell"
        assert json.loads(tc.arguments) == {"cmd": "ls"}

    def test_tool_call_delta_multiple(self):
        r = LLMResponse()
        r._feed_tool_call_delta(0, {
            "id": "call_1",
            "function": {"name": "shell", "arguments": "{}"}
        })
        r._feed_tool_call_delta(1, {
            "id": "call_2",
            "function": {"name": "stop", "arguments": "{}"}
        })
        r._finalize_tool_calls()

        assert len(r.tool_calls) == 2
        assert r.tool_calls[0].function_name == "shell"
        assert r.tool_calls[1].function_name == "stop"

    def test_finalize_clears_pending(self):
        r = LLMResponse()
        r._feed_tool_call_delta(0, {
            "id": "call_1",
            "function": {"name": "shell", "arguments": "{}"}
        })
        r._finalize_tool_calls()
        assert len(r._pending_calls) == 0

        # Calling finalize again should not duplicate
        r._finalize_tool_calls()
        assert len(r.tool_calls) == 1

    def test_empty_finalize(self):
        r = LLMResponse()
        r._finalize_tool_calls()
        assert len(r.tool_calls) == 0
