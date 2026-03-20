"""Tests for LLM client idle_timeout and health check."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from spicyclaw.config import Settings
from spicyclaw.gateway.llm_client import LLMClient, LLMResponse


@pytest.fixture
def llm_settings(tmp_path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        logs_dir=tmp_path / "logs",
        roles_dir=tmp_path / "roles",
        skills_dir=tmp_path / "skills",
        api_base_url="http://fake:9999/v1",
        api_key="test",
        model="test",
        idle_timeout=2.0,
        health_check_interval=1.0,
        _env_file=None,  # type: ignore[call-arg]
    )


class TestLLMClientHealth:
    def test_initial_healthy(self, llm_settings):
        client = LLMClient(llm_settings)
        assert client.healthy is True

    @pytest.mark.asyncio
    async def test_probe_health_failure(self, llm_settings):
        client = LLMClient(llm_settings)
        # Without a real server, probe should fail
        result = await client._probe_health()
        assert result is False
        await client.close()

    @pytest.mark.asyncio
    async def test_health_recovery_loop(self, llm_settings):
        """Test that health check loop recovers when probe succeeds."""
        client = LLMClient(llm_settings)
        client._healthy = False

        call_count = 0

        async def mock_probe():
            nonlocal call_count
            call_count += 1
            return call_count >= 2  # Recover on second probe

        client._probe_health = mock_probe

        await client._run_health_loop()
        assert client._healthy is True
        assert call_count >= 2
        await client.close()

    @pytest.mark.asyncio
    async def test_wait_for_healthy_when_healthy(self, llm_settings):
        client = LLMClient(llm_settings)
        # Should return immediately when healthy
        await client._wait_for_healthy()
        await client.close()

    @pytest.mark.asyncio
    async def test_wait_for_healthy_blocks(self, llm_settings):
        client = LLMClient(llm_settings)
        client._healthy = False

        async def restore():
            await asyncio.sleep(0.2)
            client._healthy = True

        task = asyncio.create_task(restore())
        await client._wait_for_healthy()
        assert client._healthy is True
        await task
        await client.close()

    @pytest.mark.asyncio
    async def test_close_cancels_health_task(self, llm_settings):
        client = LLMClient(llm_settings)
        client._healthy = False

        # Create a long-running health check
        async def never_recover():
            while True:
                await asyncio.sleep(100)

        client._probe_health = AsyncMock(return_value=False)
        client._health_check_task = asyncio.create_task(never_recover())

        await client.close()
        assert client._health_check_task.cancelled() or client._health_check_task.done()
