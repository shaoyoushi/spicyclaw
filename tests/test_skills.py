"""Tests for skill loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from spicyclaw.gateway.skills import SkillManager, SkillTool, _parse_skill


class TestParseSkill:
    def test_no_frontmatter(self):
        skill = _parse_skill("test", "Just a prompt text.")
        assert skill is not None
        assert skill.name == "test"
        assert skill.prompt == "Just a prompt text."

    def test_with_frontmatter(self):
        text = (
            "---\n"
            "name: web-search\n"
            "description: Search the web\n"
            "tools: [shell, stop]\n"
            "---\n"
            "Use curl to search."
        )
        skill = _parse_skill("default", text)
        assert skill is not None
        assert skill.name == "web-search"
        assert skill.description == "Search the web"
        assert skill.tools == ["shell", "stop"]
        assert skill.prompt == "Use curl to search."

    def test_empty_file(self):
        skill = _parse_skill("empty", "")
        assert skill is not None
        assert skill.prompt == ""


class TestSkillManager:
    def test_load_dir(self, tmp_path):
        (tmp_path / "greet.md").write_text(
            "---\nname: greet\ndescription: Greet user\n---\nSay hello."
        )
        mgr = SkillManager()
        mgr.load_dir(tmp_path)
        assert "greet" in mgr.list_names()
        assert mgr.get("greet").prompt == "Say hello."

    def test_load_nonexistent_dir(self, tmp_path):
        mgr = SkillManager()
        mgr.load_dir(tmp_path / "nope")
        assert mgr.list_names() == []

    def test_ignores_non_md(self, tmp_path):
        (tmp_path / "test.txt").write_text("nope")
        (tmp_path / "test.md").write_text("ok")
        mgr = SkillManager()
        mgr.load_dir(tmp_path)
        assert mgr.list_names() == ["test"]

    def test_to_tools(self, tmp_path):
        (tmp_path / "s1.md").write_text("---\nname: s1\n---\nprompt1")
        mgr = SkillManager()
        mgr.load_dir(tmp_path)
        tools = mgr.to_tools()
        assert len(tools) == 1
        assert isinstance(tools[0], SkillTool)
        assert tools[0].name == "skill_s1"


class TestSkillTool:
    @pytest.mark.asyncio
    async def test_execute(self, tmp_path):
        (tmp_path / "search.md").write_text(
            "---\nname: search\ndescription: search web\n---\n"
            "Search for: {input}"
        )
        mgr = SkillManager()
        mgr.load_dir(tmp_path)
        tools = mgr.to_tools()
        tool = tools[0]

        result = await tool.execute({"input": "hello world"})
        assert result.return_code == 0
        assert "Search for: hello world" in result.output
        assert "[Skill: search]" in result.output

    def test_openai_schema(self, tmp_path):
        (tmp_path / "test.md").write_text("---\nname: test\n---\nprompt")
        mgr = SkillManager()
        mgr.load_dir(tmp_path)
        tools = mgr.to_tools()
        schema = tools[0].to_openai_tool()
        assert schema["function"]["name"] == "skill_test"
        assert "input" in schema["function"]["parameters"]["properties"]
