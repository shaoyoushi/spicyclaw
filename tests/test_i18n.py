"""Tests for i18n module."""

from __future__ import annotations

from spicyclaw.common.i18n import set_lang, t


class TestI18n:
    def test_default_english(self):
        set_lang("en")
        assert t("aborting") == "Aborting..."

    def test_chinese(self):
        set_lang("zh")
        assert t("aborting") == "正在中止..."
        set_lang("en")  # Reset

    def test_unknown_key(self):
        assert t("nonexistent_key") == "nonexistent_key"

    def test_formatting(self):
        set_lang("en")
        result = t("max_steps", max_steps=100)
        assert "100" in result

    def test_formatting_chinese(self):
        set_lang("zh")
        result = t("max_steps", max_steps=500)
        assert "500" in result
        set_lang("en")  # Reset

    def test_invalid_lang_fallback(self):
        set_lang("fr")  # Not supported
        assert t("aborting") == "Aborting..."  # Falls back to English

    def test_help_text(self):
        set_lang("en")
        help_text = t("cmd_help")
        assert "/help" in help_text
        assert "/yolo" in help_text

    def test_help_text_chinese(self):
        set_lang("zh")
        help_text = t("cmd_help")
        assert "/help" in help_text
        assert "显示帮助" in help_text
        set_lang("en")  # Reset
