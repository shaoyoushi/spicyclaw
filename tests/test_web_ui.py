"""Tests for Web UI router and static file serving."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
class TestWebUIRouter:
    async def test_index_returns_html(self, client: AsyncClient):
        resp = await client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "SpicyClaw" in resp.text
        assert "vue.global.prod.js" in resp.text

    async def test_static_css(self, client: AsyncClient):
        resp = await client.get("/static/style.css")
        assert resp.status_code == 200
        assert "text/css" in resp.headers["content-type"]
        assert "--bg-primary" in resp.text

    async def test_static_js(self, client: AsyncClient):
        resp = await client.get("/static/app.js")
        assert resp.status_code == 200
        assert "createApp" in resp.text

    async def test_static_vue_vendor(self, client: AsyncClient):
        resp = await client.get("/static/vendor/vue.global.prod.js")
        assert resp.status_code == 200

    async def test_static_404(self, client: AsyncClient):
        resp = await client.get("/static/nonexistent.js")
        assert resp.status_code == 404

    async def test_api_still_works(self, client: AsyncClient):
        """Ensure API routes still work alongside UI routes."""
        resp = await client.get("/api/sessions")
        assert resp.status_code == 200
        assert resp.json() == []
