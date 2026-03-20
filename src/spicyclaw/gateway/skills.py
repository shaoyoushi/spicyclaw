"""Claude Code Skills format loader — markdown files with YAML frontmatter."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spicyclaw.common.types import ToolResult
from spicyclaw.gateway.tools.base import Tool

logger = logging.getLogger(__name__)

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class SkillDef:
    name: str
    description: str = ""
    tools: list[str] = field(default_factory=list)
    prompt: str = ""


class SkillTool(Tool):
    """A skill registered as a callable tool. When invoked, it injects the
    skill prompt into context to guide the model's behavior."""

    def __init__(self, skill: SkillDef) -> None:
        self.skill = skill
        self.name = f"skill_{skill.name}"
        self.description = skill.description or f"Activate skill: {skill.name}"
        self.parameters = {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "Input or query for this skill",
                },
            },
            "required": ["input"],
        }

    async def execute(self, arguments: dict[str, Any], **kwargs: Any) -> ToolResult:
        user_input = arguments.get("input", "")
        prompt = self.skill.prompt.replace("{input}", user_input)
        return ToolResult(output=f"[Skill: {self.skill.name}]\n{prompt}")


class SkillManager:
    def __init__(self) -> None:
        self._skills: dict[str, SkillDef] = {}

    def load_dir(self, skills_dir: Path) -> None:
        """Load all .md skill files from the directory."""
        if not skills_dir.exists():
            logger.debug("Skills directory not found: %s", skills_dir)
            return

        for f in sorted(skills_dir.iterdir()):
            if f.suffix != ".md":
                continue
            try:
                text = f.read_text(encoding="utf-8")
                skill = _parse_skill(f.stem, text)
                if skill:
                    self._skills[skill.name] = skill
                    logger.info("Loaded skill: %s", skill.name)
            except Exception:
                logger.exception("Failed to load skill from %s", f)

    def get(self, name: str) -> SkillDef | None:
        return self._skills.get(name)

    def list_names(self) -> list[str]:
        return list(self._skills.keys())

    def to_tools(self) -> list[SkillTool]:
        return [SkillTool(s) for s in self._skills.values()]


def _parse_skill(default_name: str, text: str) -> SkillDef | None:
    """Parse a markdown skill file with optional YAML frontmatter."""
    match = FRONTMATTER_RE.match(text)
    if not match:
        # No frontmatter — treat entire file as prompt
        return SkillDef(name=default_name, prompt=text.strip())

    frontmatter_text = match.group(1)
    prompt = text[match.end():].strip()

    # Simple YAML-like parsing (avoid yaml dependency for skills)
    meta: dict[str, Any] = {}
    for line in frontmatter_text.split("\n"):
        line = line.strip()
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            if val.startswith("[") and val.endswith("]"):
                # Simple list parsing
                items = [v.strip().strip("'\"") for v in val[1:-1].split(",") if v.strip()]
                meta[key] = items
            else:
                meta[key] = val

    return SkillDef(
        name=meta.get("name", default_name),
        description=meta.get("description", ""),
        tools=meta.get("tools", []),
        prompt=prompt,
    )
