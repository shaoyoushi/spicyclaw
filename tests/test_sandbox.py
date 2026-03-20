"""Tests for Docker sandbox module (without Docker dependency)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from spicyclaw.gateway.sandbox import (
    CONTAINER_PREFIX,
    Sandbox,
    SandboxError,
    SandboxManager,
)


class TestSandbox:
    def test_container_id(self):
        mock_client = MagicMock()
        sb = Sandbox("test-container-123", mock_client)
        assert sb.container_id == "test-container-123"

    def test_exec_success(self):
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.exec_run.return_value = MagicMock(
            output=(b"hello\n", b""),
            exit_code=0,
        )
        mock_client.containers.get.return_value = mock_container

        sb = Sandbox("cid", mock_client)
        stdout, stderr, rc = sb.exec("echo hello")
        assert stdout == "hello\n"
        assert stderr == ""
        assert rc == 0

    def test_exec_with_stderr(self):
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.exec_run.return_value = MagicMock(
            output=(b"", b"error msg"),
            exit_code=1,
        )
        mock_client.containers.get.return_value = mock_container

        sb = Sandbox("cid", mock_client)
        stdout, stderr, rc = sb.exec("bad cmd")
        assert stderr == "error msg"
        assert rc == 1

    def test_exec_container_not_found(self):
        mock_client = MagicMock()
        # Simulate NotFound by raising an exception
        mock_client.containers.get.side_effect = Exception("not found")

        sb = Sandbox("cid", mock_client)
        with pytest.raises(SandboxError, match="Exec failed"):
            sb.exec("test")

    def test_destroy(self):
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_client.containers.get.return_value = mock_container

        sb = Sandbox("cid", mock_client)
        sb.destroy()
        mock_container.stop.assert_called_once()
        mock_container.remove.assert_called_once()


class TestSandboxManager:
    def test_available_no_docker(self):
        mgr = SandboxManager()
        # Without mock, HAS_DOCKER may or may not be True
        # but available should not crash
        result = mgr.available
        assert isinstance(result, bool)

    def test_get_nonexistent(self):
        mgr = SandboxManager()
        assert mgr.get("nonexistent") is None

    def test_destroy_nonexistent(self):
        mgr = SandboxManager()
        mgr.destroy("nonexistent")  # Should not raise

    def test_destroy_all_empty(self):
        mgr = SandboxManager()
        mgr.destroy_all()  # Should not raise

    def test_container_prefix(self):
        assert CONTAINER_PREFIX == "spicyclaw-"
