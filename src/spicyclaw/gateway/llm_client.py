"""OpenAI-compatible streaming LLM client with idle_timeout and health probing."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator

import httpx

from spicyclaw.common.types import Message, ToolCall
from spicyclaw.config import Settings

logger = logging.getLogger(__name__)


class LLMResponse:
    """Accumulated result from a streaming LLM call."""

    def __init__(self) -> None:
        self.content: str = ""
        self.tool_calls: list[ToolCall] = []
        self.usage_tokens: int = 0
        self.finish_reason: str | None = None
        self._pending_calls: dict[int, dict[str, str]] = {}

    def _feed_tool_call_delta(self, index: int, delta: dict[str, Any]) -> None:
        if index not in self._pending_calls:
            fn = delta.get("function", {})
            self._pending_calls[index] = {
                "id": delta.get("id", ""),
                "name": fn.get("name", ""),
                "arguments": fn.get("arguments", ""),
            }
        else:
            entry = self._pending_calls[index]
            if "id" in delta and delta["id"]:
                entry["id"] = delta["id"]
            fn = delta.get("function", {})
            if fn.get("name"):
                entry["name"] += fn["name"]
            if fn.get("arguments"):
                entry["arguments"] += fn["arguments"]

    def _finalize_tool_calls(self) -> None:
        for _idx in sorted(self._pending_calls):
            entry = self._pending_calls[_idx]
            self.tool_calls.append(
                ToolCall(
                    id=entry["id"],
                    function_name=entry["name"],
                    arguments=entry["arguments"],
                )
            )
        self._pending_calls.clear()


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.api_base_url,
            headers={
                "Authorization": f"Bearer {settings.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(settings.request_timeout, connect=10.0),
        )
        self._healthy = True
        self._health_check_task: asyncio.Task | None = None

    @property
    def healthy(self) -> bool:
        return self._healthy

    async def close(self) -> None:
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        await self._client.aclose()

    async def _probe_health(self) -> bool:
        """Send a GET to /v1/models with 5s timeout to check if LLM is alive."""
        try:
            resp = await self._client.get("/models", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def _run_health_loop(self) -> None:
        """Background loop: poll health until recovered, then stop."""
        interval = self.settings.health_check_interval
        logger.warning("LLM marked unhealthy, starting health check loop (every %.0fs)", interval)
        while True:
            await asyncio.sleep(interval)
            if await self._probe_health():
                self._healthy = True
                logger.info("LLM health recovered")
                return

    def _start_health_check(self) -> None:
        """Start background health polling if not already running."""
        if self._health_check_task and not self._health_check_task.done():
            return
        self._health_check_task = asyncio.create_task(
            self._run_health_loop(), name="llm-health-check"
        )

    async def _wait_for_healthy(self) -> None:
        """Block until the LLM is healthy again."""
        if self._healthy:
            return
        logger.info("Waiting for LLM to become healthy...")
        while not self._healthy:
            await asyncio.sleep(1.0)

    async def stream_chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[tuple[str, LLMResponse]]:
        """Stream a chat completion with idle_timeout support.

        Yields ("chunk", response) for content deltas and ("done", response) when finished.

        Idle timeout logic:
        - If no chunk received within idle_timeout seconds, probe /v1/models
        - If probe succeeds: model is still thinking, continue waiting
        - If probe fails: mark unhealthy, start background health polling, raise
        """
        # Wait for healthy before sending
        await self._wait_for_healthy()

        body: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [m.to_openai() for m in messages],
            "stream": True,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        response = LLMResponse()
        idle_timeout = self.settings.idle_timeout

        async with self._client.stream("POST", "/chat/completions", json=body) as http:
            http.raise_for_status()

            last_chunk_time = time.monotonic()

            async for line in self._iter_lines_with_idle(http, idle_timeout):
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break

                last_chunk_time = time.monotonic()

                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    logger.warning("Malformed SSE chunk: %s", payload[:200])
                    continue

                choice = chunk.get("choices", [{}])[0]
                delta = choice.get("delta", {})

                # Content
                if delta.get("content"):
                    response.content += delta["content"]
                    yield "chunk", response

                # Tool calls
                if delta.get("tool_calls"):
                    for tc_delta in delta["tool_calls"]:
                        idx = tc_delta.get("index", 0)
                        response._feed_tool_call_delta(idx, tc_delta)

                # Finish reason
                if choice.get("finish_reason"):
                    response.finish_reason = choice["finish_reason"]

                # Usage (some providers send it in the final chunk)
                if chunk.get("usage"):
                    response.usage_tokens = chunk["usage"].get("total_tokens", 0)

        response._finalize_tool_calls()

        # Fallback token estimate
        if response.usage_tokens == 0:
            text_len = sum(len(m.content or "") for m in messages) + len(response.content)
            response.usage_tokens = text_len // 4

        yield "done", response

    async def _iter_lines_with_idle(
        self, http_response: httpx.Response, idle_timeout: float
    ) -> AsyncIterator[str]:
        """Iterate over SSE lines with idle timeout detection.

        When idle_timeout is exceeded without receiving data:
        1. Probe /v1/models (5s timeout)
        2. Success → model is thinking, reset timer and continue
        3. Failure → mark unhealthy, start health check loop, raise
        """
        buffer = ""
        aiter = http_response.aiter_bytes()

        while True:
            try:
                chunk_bytes = await asyncio.wait_for(
                    aiter.__anext__(), timeout=idle_timeout
                )
                text = chunk_bytes.decode("utf-8", errors="replace")
                buffer += text
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.rstrip("\r")
                    yield line
            except asyncio.TimeoutError:
                # Idle timeout — probe health
                logger.debug("No data for %.0fs, probing LLM health...", idle_timeout)
                if await self._probe_health():
                    logger.debug("LLM probe OK — model is still thinking")
                    continue  # Reset and keep waiting
                else:
                    self._healthy = False
                    self._start_health_check()
                    raise httpx.ReadTimeout(
                        f"LLM idle for {idle_timeout}s and health probe failed"
                    )
            except StopAsyncIteration:
                # Yield any remaining buffered content
                if buffer.strip():
                    yield buffer
                return

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Non-streaming convenience wrapper."""
        response: LLMResponse | None = None
        async for _event, response in self.stream_chat(messages, tools):
            pass
        assert response is not None
        return response
