"""Simple internationalization — Chinese/English string tables."""

from __future__ import annotations

from typing import Any

# Message keys → (English, Chinese)
_STRINGS: dict[str, tuple[str, str]] = {
    # Workloop
    "aborted": ("Task aborted by user", "用户已中止任务"),
    "waiting_input": ("Agent stopped — waiting for input", "代理已停止 — 等待用户输入"),
    "format_errors": ("Too many format errors, stopping", "格式错误过多，已停止"),
    "max_steps": ("Protection: max steps ({max_steps}) reached", "保护机制：已达到最大步数 ({max_steps})"),
    "step_confirm": ("Step mode: waiting for confirmation to execute", "步进模式：等待确认执行"),
    "auto_compact": ("Auto-compacting context...", "正在自动压缩上下文..."),
    "compact_done": ("Context compressed successfully", "上下文压缩成功"),
    "protection": ("Protection: {detail}", "保护机制：{detail}"),

    # Commands
    "cmd_help": (
        "Available commands:\n"
        "/help — show this help\n"
        "/yolo — switch to YOLO mode (auto-execute)\n"
        "/step — switch to step mode (confirm each action)\n"
        "/stop — abort current execution\n"
        "/compact [node_ids] — compress context window\n"
        "/status — show session status\n"
        "/task — show current task\n"
        "/plan — show current plan\n"
        "/session [role] — show session info or set role\n"
        "/settings — show current settings\n"
        "/resume — resume interrupted workloop",
        "可用命令：\n"
        "/help — 显示帮助\n"
        "/yolo — 切换到YOLO模式（自动执行）\n"
        "/step — 切换到步进模式（每步确认）\n"
        "/stop — 中止当前执行\n"
        "/compact [节点ID] — 压缩上下文窗口\n"
        "/status — 显示会话状态\n"
        "/task — 显示当前任务\n"
        "/plan — 显示当前计划\n"
        "/session [角色] — 显示会话信息或设置角色\n"
        "/settings — 显示当前设置\n"
        "/resume — 恢复中断的工作循环",
    ),
    "switched_yolo": ("Switched to YOLO mode — auto-executing all tool calls", "已切换到YOLO模式 — 自动执行所有工具调用"),
    "switched_step": ("Switched to Step mode — will confirm before each execution", "已切换到步进模式 — 每次执行前需确认"),
    "aborting": ("Aborting...", "正在中止..."),
    "compact_nothing": ("Nothing to compact (context too short)", "无需压缩（上下文太短）"),
    "compact_nodes_nothing": ("Nothing to compact for nodes: {nodes}", "指定节点无需压缩：{nodes}"),
    "compact_summary": ("Context compressed. Summary: {summary}...", "上下文已压缩。摘要：{summary}..."),
    "compact_nodes_summary": ("Work nodes {nodes} compressed. Summary: {summary}...", "工作节点 {nodes} 已压缩。摘要：{summary}..."),
    "no_task": ("No TASK.md found", "未找到 TASK.md"),
    "no_plan": ("No PLAN.json found", "未找到 PLAN.json"),
    "unknown_cmd": ("Unknown command: /{cmd}. Type /help for available commands.", "未知命令：/{cmd}。输入 /help 查看可用命令。"),
    "role_set": ("Role set to '{role}'", "角色已设置为 '{role}'"),
    "role_not_found": ("Role '{role}' not found. Available: {available}", "角色 '{role}' 未找到。可用角色：{available}"),
    "role_mgr_unavail": ("Role manager not available", "角色管理器不可用"),
    "resume_running": ("Workloop is already running", "工作循环已在运行中"),
    "resume_empty": ("No context to resume — nothing to do", "没有上下文可恢复"),
    "resuming": ("Resuming workloop...", "正在恢复工作循环..."),
    "busy": ("Agent is busy, please wait", "代理正忙，请稍候"),
}

_current_lang = "en"


def set_lang(lang: str) -> None:
    """Set the current language ('en' or 'zh')."""
    global _current_lang
    _current_lang = lang if lang in ("en", "zh") else "en"


def t(key: str, **kwargs: Any) -> str:
    """Translate a message key to the current language with optional formatting."""
    pair = _STRINGS.get(key)
    if pair is None:
        return key
    idx = 1 if _current_lang == "zh" else 0
    text = pair[idx]
    if kwargs:
        text = text.format(**kwargs)
    return text
