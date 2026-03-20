"""Tests for API routes."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
class TestSessionRoutes:
    async def test_create_session(self, client: AsyncClient):
        resp = await client.post("/api/sessions", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert data["title"] == "New Session"
        assert data["status"] == "stopped"

    async def test_create_session_with_model(self, client: AsyncClient):
        resp = await client.post("/api/sessions", json={"model": "custom-model"})
        assert resp.status_code == 200
        assert resp.json()["model"] == "custom-model"

    async def test_create_session_no_body(self, client: AsyncClient):
        resp = await client.post("/api/sessions")
        assert resp.status_code == 200
        assert "id" in resp.json()

    async def test_list_sessions_empty(self, client: AsyncClient):
        resp = await client.get("/api/sessions")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_list_sessions_after_create(self, client: AsyncClient):
        await client.post("/api/sessions", json={})
        await client.post("/api/sessions", json={})
        resp = await client.get("/api/sessions")
        assert len(resp.json()) == 2

    async def test_get_session(self, client: AsyncClient):
        create_resp = await client.post("/api/sessions", json={})
        sid = create_resp.json()["id"]
        resp = await client.get(f"/api/sessions/{sid}")
        assert resp.status_code == 200
        assert resp.json()["id"] == sid

    async def test_get_session_not_found(self, client: AsyncClient):
        resp = await client.get("/api/sessions/nonexistent")
        assert resp.status_code == 404

    async def test_get_context_empty(self, client: AsyncClient):
        create_resp = await client.post("/api/sessions", json={})
        sid = create_resp.json()["id"]
        resp = await client.get(f"/api/sessions/{sid}/context")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_get_context_not_found(self, client: AsyncClient):
        resp = await client.get("/api/sessions/nonexistent/context")
        assert resp.status_code == 404

    async def test_send_message_session_not_found(self, client: AsyncClient):
        resp = await client.post(
            "/api/sessions/nonexistent/message",
            json={"content": "hello"},
        )
        assert resp.status_code == 404


@pytest.mark.asyncio
class TestConfig:
    async def test_settings_isolated(self, client: AsyncClient, app):
        """Ensure test settings use temp dirs, not real ones."""
        settings = app.state.settings
        assert "fake-llm" in settings.api_base_url
        assert settings.api_key == "test-key"
