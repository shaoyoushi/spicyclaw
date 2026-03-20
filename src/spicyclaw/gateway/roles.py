"""Agent role loading from YAML files."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import yaml  # type: ignore[import-untyped]
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass
class Role:
    name: str
    description: str = ""
    system_prompt: str = ""
    tools: list[str] = field(default_factory=list)


class RoleManager:
    def __init__(self) -> None:
        self._roles: dict[str, Role] = {}

    def load_dir(self, roles_dir: Path) -> None:
        """Load all .yaml/.yml role files from the directory."""
        if not roles_dir.exists():
            logger.debug("Roles directory not found: %s", roles_dir)
            return

        if not HAS_YAML:
            logger.warning("PyYAML not installed — skipping role loading")
            return

        for f in sorted(roles_dir.iterdir()):
            if f.suffix not in (".yaml", ".yml"):
                continue
            try:
                data = yaml.safe_load(f.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    continue
                role = Role(
                    name=data.get("name", f.stem),
                    description=data.get("description", ""),
                    system_prompt=data.get("system_prompt", ""),
                    tools=data.get("tools", []),
                )
                self._roles[role.name] = role
                logger.info("Loaded role: %s", role.name)
            except Exception:
                logger.exception("Failed to load role from %s", f)

    def get(self, name: str) -> Role | None:
        return self._roles.get(name)

    def list_names(self) -> list[str]:
        return list(self._roles.keys())
