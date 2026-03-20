"""Docker container sandbox for secure command execution.

Requires the optional 'docker' package: pip install spicyclaw[sandbox]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import docker  # type: ignore[import-untyped]
    from docker.errors import DockerException, NotFound  # type: ignore[import-untyped]
    HAS_DOCKER = True
except ImportError:
    HAS_DOCKER = False

    class NotFound(Exception):  # type: ignore[no-redef]
        """Stub for when docker is not installed."""

    class DockerException(Exception):  # type: ignore[no-redef]
        """Stub for when docker is not installed."""


CONTAINER_PREFIX = "spicyclaw-"
DEFAULT_IMAGE = "ubuntu:22.04"


class SandboxError(Exception):
    """Raised when a sandbox operation fails."""


class Sandbox:
    """Manages a Docker container for sandboxed command execution."""

    def __init__(self, container_id: str, client: Any) -> None:
        self._container_id = container_id
        self._client = client

    @property
    def container_id(self) -> str:
        return self._container_id

    def exec(
        self, command: str, timeout: float = 120.0, workdir: str = "/workspace"
    ) -> tuple[str, str, int]:
        """Execute a command in the container.

        Returns (stdout, stderr, return_code).
        """
        try:
            container = self._client.containers.get(self._container_id)
            exec_result = container.exec_run(
                ["bash", "-c", command],
                workdir=workdir,
                demux=True,
            )
            stdout = (exec_result.output[0] or b"").decode("utf-8", errors="replace") if exec_result.output else ""
            stderr = (exec_result.output[1] or b"").decode("utf-8", errors="replace") if exec_result.output else ""
            rc = exec_result.exit_code or 0
            return stdout, stderr, rc
        except NotFound:
            raise SandboxError(f"Container {self._container_id} not found")
        except Exception as e:
            raise SandboxError(f"Exec failed: {e}") from e

    def destroy(self) -> None:
        """Stop and remove the container."""
        try:
            container = self._client.containers.get(self._container_id)
            container.stop(timeout=5)
            container.remove(force=True)
            logger.info("Destroyed sandbox container %s", self._container_id[:12])
        except NotFound:
            pass
        except Exception:
            logger.exception("Failed to destroy container %s", self._container_id[:12])


class SandboxManager:
    """Creates and manages sandbox containers."""

    def __init__(self) -> None:
        self._client: Any = None
        self._sandboxes: dict[str, Sandbox] = {}

    @property
    def available(self) -> bool:
        """Check if Docker is available."""
        if not HAS_DOCKER:
            return False
        try:
            self._ensure_client()
            self._client.ping()
            return True
        except Exception:
            return False

    def _ensure_client(self) -> None:
        if self._client is None:
            if not HAS_DOCKER:
                raise SandboxError("Docker SDK not installed. Install with: pip install spicyclaw[sandbox]")
            self._client = docker.from_env()

    def create(
        self,
        session_id: str,
        workspace_dir: Path,
        image: str = DEFAULT_IMAGE,
    ) -> Sandbox:
        """Create a new sandbox container for a session.

        The session's workspace directory is bind-mounted to /workspace.
        """
        self._ensure_client()

        container_name = f"{CONTAINER_PREFIX}{session_id}"

        # Remove existing container with same name
        try:
            old = self._client.containers.get(container_name)
            old.remove(force=True)
        except (NotFound, Exception):
            pass

        try:
            container = self._client.containers.run(
                image,
                command="sleep infinity",
                name=container_name,
                volumes={
                    str(workspace_dir.resolve()): {
                        "bind": "/workspace",
                        "mode": "rw",
                    }
                },
                detach=True,
                network_mode="bridge",
                mem_limit="512m",
                cpu_period=100000,
                cpu_quota=50000,  # 50% CPU
            )
            sandbox = Sandbox(container.id, self._client)
            self._sandboxes[session_id] = sandbox
            logger.info(
                "Created sandbox %s for session %s (image=%s)",
                container.id[:12], session_id, image,
            )
            return sandbox
        except Exception as e:
            raise SandboxError(f"Failed to create container: {e}") from e

    def get(self, session_id: str) -> Sandbox | None:
        return self._sandboxes.get(session_id)

    def destroy(self, session_id: str) -> None:
        sandbox = self._sandboxes.pop(session_id, None)
        if sandbox:
            sandbox.destroy()

    def cleanup_stale(self) -> int:
        """Find and remove leftover spicyclaw-* containers from previous runs.

        Returns the number of containers cleaned up.
        """
        self._ensure_client()
        count = 0
        try:
            containers = self._client.containers.list(
                all=True,
                filters={"name": CONTAINER_PREFIX},
            )
            for container in containers:
                try:
                    container.remove(force=True)
                    count += 1
                    logger.info("Cleaned up stale container: %s", container.name)
                except Exception:
                    logger.warning("Failed to clean up container: %s", container.name)
        except Exception:
            logger.exception("Failed to list stale containers")
        return count

    def destroy_all(self) -> None:
        """Destroy all managed sandboxes."""
        for session_id in list(self._sandboxes.keys()):
            self.destroy(session_id)
