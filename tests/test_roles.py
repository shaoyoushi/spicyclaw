"""Tests for role loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from spicyclaw.gateway.roles import RoleManager


class TestRoleManager:
    def test_load_empty_dir(self, tmp_path):
        mgr = RoleManager()
        mgr.load_dir(tmp_path)
        assert mgr.list_names() == []

    def test_load_nonexistent_dir(self, tmp_path):
        mgr = RoleManager()
        mgr.load_dir(tmp_path / "nope")
        assert mgr.list_names() == []

    def test_load_yaml_role(self, tmp_path):
        role_file = tmp_path / "programmer.yaml"
        role_file.write_text(
            "name: programmer\n"
            "description: A programmer role\n"
            "system_prompt: You are a programmer.\n"
            "tools:\n  - shell\n  - stop\n"
        )
        mgr = RoleManager()
        mgr.load_dir(tmp_path)
        assert "programmer" in mgr.list_names()
        role = mgr.get("programmer")
        assert role is not None
        assert role.description == "A programmer role"
        assert role.system_prompt == "You are a programmer."
        assert role.tools == ["shell", "stop"]

    def test_load_yml_extension(self, tmp_path):
        (tmp_path / "sysadmin.yml").write_text("name: sysadmin\ndescription: sys admin\n")
        mgr = RoleManager()
        mgr.load_dir(tmp_path)
        assert "sysadmin" in mgr.list_names()

    def test_ignores_non_yaml(self, tmp_path):
        (tmp_path / "readme.md").write_text("# Roles")
        (tmp_path / "test.yaml").write_text("name: test\n")
        mgr = RoleManager()
        mgr.load_dir(tmp_path)
        assert mgr.list_names() == ["test"]

    def test_get_nonexistent(self, tmp_path):
        mgr = RoleManager()
        mgr.load_dir(tmp_path)
        assert mgr.get("nope") is None

    def test_invalid_yaml_skipped(self, tmp_path):
        (tmp_path / "bad.yaml").write_text(":::invalid\n  yaml: [")
        (tmp_path / "good.yaml").write_text("name: good\n")
        mgr = RoleManager()
        mgr.load_dir(tmp_path)
        assert mgr.list_names() == ["good"]
