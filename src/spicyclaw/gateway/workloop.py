"""Core Agent work loop."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any

from spicyclaw.common.events import ServerEvent
from spicyclaw.common.i18n import t
from spicyclaw.common.types import Message, Role, SessionStatus, ToolResult
from spicyclaw.config import Settings
from spicyclaw.gateway.context import ContextManager
from spicyclaw.gateway.llm_client import LLMClient, LLMResponse
from spicyclaw.gateway.session import Session
from spicyclaw.gateway.tools.base import ToolRegistry

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are SpicyClaw, an AI agent that completes tasks by executing shell commands.

## Available Tools
- **shell**: Execute a shell command. Parameter: `command` (string).
- **stop**: Stop execution. Parameter: `reason` (string). Call this when the task is complete or you need user input.

## Rules
1. Every tool call MUST include `work_node` (current task node ID, e.g. "1.2") and `next_step` (brief description of what you plan to do AFTER the current tool call finishes — NOT the current step).
2. When starting a new task:
   a. First, use the shell tool to create `TASK.md` — first line is a short title, rest describes the task.
   b. Then use the shell tool to create `PLAN.json` — a WBS plan, format: `{{"nodes": [{{"id": "1", "title": "...", "children": [...]}}, ...]}}`
      Use `jq` for precise reading and writing of PLAN.json rather than rewriting the entire file each time.
   c. Execute the plan step by step, using work_node IDs from your plan.
3. Call `stop` with reason "task complete" when done, or "need user input" when you need clarification.
4. Always check command output before proceeding. Fix errors before moving on.
5. Working directory: {work_dir}
   All your files (TASK.md, PLAN.json, source code, etc.) should be created in this directory.
   Do NOT create files outside this directory unless explicitly asked.
6. The memory_read and memory_write tools are for persistent session memory ONLY — do NOT use them to create task files like TASK.md or PLAN.json. Use shell commands for that.
"""


class RepeatTracker:
    """Detects repeated tool calls with identical results."""

    def __init__(self, max_errors: int, max_outputs: int) -> None:
        self.max_errors = max_errors
        self.max_outputs = max_outputs
        self._history: list[tuple[str, str, bool]] = []

    def check(self, tool_name: str, args: dict[str, Any], result: ToolResult) -> str | None:
        """Returns a warning message if repeat threshold exceeded, None otherwise."""
        key = f"{tool_name}:{_hash_dict(args)}"
        out_hash = _hash_str(result.output + result.error)
        is_error = result.return_code != 0

        entry = (key, out_hash, is_error)
        self._history.append(entry)

        count = 0
        for past in reversed(self._history):
            if past == entry:
                count += 1
            else:
                break

        if is_error and count >= self.max_errors:
            return f"Same command failed {count} times consecutively with identical error"
        if not is_error and count >= self.max_outputs:
            return f"Same command produced identical output {count} times consecutively"
        return None


def _hash_dict(d: dict[str, Any]) -> str:
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]


def _hash_str(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:8]


async def run_workloop(
    session: Session,
    llm: LLMClient,
    tool_registry: ToolRegistry,
    settings: Settings,
) -> None:
    """Run the agent work loop until stop, abort, or protection triggers."""
    ctx_mgr = ContextManager(session, settings)
    repeat_tracker = RepeatTracker(settings.max_repeat_errors, settings.max_repeat_outputs)
    step = 0
    format_error_count = 0
    max_format_errors = 5

    # Ensure workspace exists
    session.workspace.mkdir(parents=True, exist_ok=True)

    # Ensure system prompt is present
    if not session.context or session.context[0].role != Role.SYSTEM:
        system_msg = Message(
            role=Role.SYSTEM,
            content=SYSTEM_PROMPT.format(work_dir=session.workspace),
        )
        session.context.insert(0, system_msg)

    try:
        while step < settings.max_steps:
            if session.abort_event.is_set():
                await _broadcast_system(session, t("aborted"))
                break

            session.status = SessionStatus.THINKING

            tools_def = tool_registry.to_openai_tools()

            # Stream LLM response, broadcast content chunks
            response = LLMResponse()
            last_content_len = 0
            async for event, response in llm.stream_chat(session.context, tools_def):
                if session.abort_event.is_set():
                    break
                if event == "chunk":
                    delta = response.content[last_content_len:]
                    last_content_len = len(response.content)
                    if delta:
                        await session.broadcast(ServerEvent(
                            type="chunk",
                            session_id=session.id,
                            data={"text": delta},
                        ))

            if session.abort_event.is_set():
                await _broadcast_system(session, t("aborted"))
                break

            # Update token tracking
            ctx_mgr.update_tokens(response.usage_tokens)
            ctx_mgr.check_and_warn()

            # No tool calls — model is replying with text only
            if not response.tool_calls:
                assistant_msg = Message(
                    role=Role.ASSISTANT, content=response.content or ""
                )
                session.add_message(assistant_msg)
                await _broadcast_system(session, t("waiting_input"))
                break

            # Add assistant message with tool calls
            assistant_msg = Message(
                role=Role.ASSISTANT,
                content=response.content or None,
                tool_calls=response.tool_calls,
            )
            session.add_message(assistant_msg)

            # Step mode: pause before execution
            if session.step_mode and not settings.yolo:
                session.status = SessionStatus.PAUSED
                await session.broadcast(ServerEvent(
                    type="status",
                    session_id=session.id,
                    data={"status": "paused"},
                ))
                await _broadcast_system(session, t("step_confirm"))
                # Wait for confirm or abort
                session.confirm_event.clear()
                while not session.confirm_event.is_set():
                    if session.abort_event.is_set():
                        break
                    await asyncio.sleep(0.1)

                if session.abort_event.is_set():
                    await _broadcast_system(session, t("aborted"))
                    break

            # Execute each tool call
            should_stop = False
            for tc in response.tool_calls:
                # Parse arguments
                try:
                    args = json.loads(tc.arguments)
                except json.JSONDecodeError as e:
                    format_error_count += 1
                    error_msg = f"Invalid JSON in tool arguments: {e}"
                    tool_msg = Message(
                        role=Role.TOOL,
                        content=error_msg,
                        tool_call_id=tc.id,
                        name=tc.function_name,
                    )
                    session.add_message(tool_msg)
                    if format_error_count >= max_format_errors:
                        await _broadcast_system(session, t("format_errors"))
                        should_stop = True
                    continue

                format_error_count = 0

                # Extract and validate common params
                work_node = args.pop("work_node", "")
                next_step = args.pop("next_step", "")

                if not work_node or not next_step:
                    missing = []
                    if not work_node:
                        missing.append("work_node")
                    if not next_step:
                        missing.append("next_step")
                    error_msg = (
                        f"ERROR: Missing required parameters: {', '.join(missing)}. "
                        f"Every tool call MUST include 'work_node' (current task node ID) "
                        f"and 'next_step' (what you plan to do after this call). "
                        f"Please retry with these parameters."
                    )
                    tool_msg = Message(
                        role=Role.TOOL,
                        content=error_msg,
                        tool_call_id=tc.id,
                        name=tc.function_name,
                    )
                    session.add_message(tool_msg)
                    await session.broadcast(ServerEvent(
                        type="tool_end",
                        session_id=session.id,
                        data={
                            "tool_call_id": tc.id,
                            "output": "",
                            "error": error_msg,
                            "return_code": 1,
                            "truncated": False,
                        },
                    ))
                    continue

                # Broadcast tool_call event
                await session.broadcast(ServerEvent(
                    type="tool_call",
                    session_id=session.id,
                    data={
                        "tool_call_id": tc.id,
                        "name": tc.function_name,
                        "arguments": args,
                        "work_node": work_node,
                        "next_step": next_step,
                    },
                ))

                # Execute — shell tool cwd is workspace, not session dir
                tool = tool_registry.get(tc.function_name)
                if tool is None:
                    result = ToolResult(
                        output="", error=f"Unknown tool: {tc.function_name}", return_code=1
                    )
                else:
                    session.status = SessionStatus.EXECUTING
                    result = await tool.execute(args, cwd=session.workspace, session_dir=session.dir)

                # Broadcast tool result
                await session.broadcast(ServerEvent(
                    type="tool_end",
                    session_id=session.id,
                    data={
                        "tool_call_id": tc.id,
                        "output": result.output[:2000],
                        "error": result.error[:2000],
                        "return_code": result.return_code,
                        "truncated": result.truncated,
                    },
                ))

                # Build tool result message for context
                content = result.output
                if result.error:
                    content += f"\n[STDERR]\n{result.error}"
                tool_msg = Message(
                    role=Role.TOOL,
                    content=content,
                    tool_call_id=tc.id,
                    name=tc.function_name,
                )
                session.add_message(tool_msg)

                # Check for stop tool
                if tc.function_name == "stop":
                    should_stop = True

                # Repeat detection
                if tool is not None:
                    warning = repeat_tracker.check(tc.function_name, args, result)
                    if warning:
                        await _broadcast_system(session, t("protection", detail=warning))
                        should_stop = True

            step += 1

            # Check for TASK.md and update title
            await _try_update_title(session)

            # Save state periodically
            session.save_context()
            session.save_meta()

            # Auto-compact if threshold exceeded
            if ctx_mgr.should_compact:
                await _broadcast_system(session, t("auto_compact"))
                summary = await ctx_mgr.full_compact(llm)
                if summary:
                    await _broadcast_system(session, t("compact_done"))

            if should_stop:
                break

        else:
            # max_steps exceeded
            await _broadcast_system(
                session, t("max_steps", max_steps=settings.max_steps)
            )

    except Exception as e:
        logger.exception("Workloop error in session %s", session.id)
        await session.broadcast(ServerEvent(
            type="error",
            session_id=session.id,
            data={"message": str(e)},
        ))
    finally:
        session.status = SessionStatus.STOPPED
        session.save_context()
        session.save_meta()
        await session.broadcast(ServerEvent(
            type="status",
            session_id=session.id,
            data={"status": "stopped"},
        ))


async def _try_update_title(session: Session) -> None:
    """Read TASK.md first line as session title if not yet set."""
    if session.meta.title != "New Session":
        return
    task_file = session.workspace / "TASK.md"
    if not task_file.exists():
        return
    try:
        first_line = task_file.read_text(encoding="utf-8").split("\n", 1)[0].strip()
        if first_line.startswith("#"):
            first_line = first_line.lstrip("# ").strip()
        if first_line:
            session.meta.title = first_line[:100]
            logger.info("Session %s title: %s", session.id, session.meta.title)
            await session.broadcast(ServerEvent(
                type="session_update",
                session_id=session.id,
                data={"title": session.meta.title},
            ))
    except Exception:
        pass


async def _broadcast_system(session: Session, message: str) -> None:
    """Send a system message to subscribers and add to context."""
    logger.info("Session %s: %s", session.id, message)
    await session.broadcast(ServerEvent(
        type="system",
        session_id=session.id,
        data={"message": message},
    ))
