"""Context manager — token tracking and context compression."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from spicyclaw.common.types import Message, Role
from spicyclaw.config import Settings

if TYPE_CHECKING:
    from spicyclaw.gateway.llm_client import LLMClient
    from spicyclaw.gateway.session import Session

logger = logging.getLogger(__name__)

COMPACT_PROMPT = """\
Summarize the following conversation segment concisely. \
Preserve key facts: what tasks were attempted, what commands ran, \
what succeeded, what failed, and any important state changes. \
Be factual and brief — this summary replaces the original messages in the context window."""


class ContextManager:
    def __init__(self, session: Session, settings: Settings) -> None:
        self.session = session
        self.settings = settings

    def update_tokens(self, usage_tokens: int) -> None:
        self.session.meta.token_used = usage_tokens

    @property
    def usage_ratio(self) -> float:
        if self.settings.max_tokens <= 0:
            return 0.0
        return self.session.meta.token_used / self.settings.max_tokens

    @property
    def should_compact(self) -> bool:
        return self.usage_ratio >= self.settings.full_compact_ratio

    def check_and_warn(self) -> None:
        """Log a warning if context is getting large."""
        if self.should_compact:
            logger.warning(
                "Session %s context usage %.0f%% (%d/%d tokens) — compact recommended",
                self.session.id,
                self.usage_ratio * 100,
                self.session.meta.token_used,
                self.settings.max_tokens,
            )

    async def full_compact(self, llm: LLMClient) -> str | None:
        """Compress the middle portion of context, keeping system prompt and recent rounds.

        Returns the summary text on success, None on failure.
        """
        ctx = self.session.context
        keep_rounds = self.settings.compact_keep_rounds

        # Need at least system + some messages + recent to compress
        if len(ctx) < 3:
            return None

        # Find boundaries: keep system prompt (index 0) and last N rounds
        # A "round" is roughly: user msg + assistant msg + tool msgs
        keep_tail = 0
        rounds_found = 0
        for i in range(len(ctx) - 1, 0, -1):
            keep_tail += 1
            if ctx[i].role == Role.USER:
                rounds_found += 1
                if rounds_found >= keep_rounds:
                    break

        if keep_tail >= len(ctx) - 1:
            # Not enough to compress
            return None

        # Split: [system] [middle...] [tail...]
        system_msg = ctx[0] if ctx[0].role == Role.SYSTEM else None
        start_idx = 1 if system_msg else 0
        split_point = len(ctx) - keep_tail

        if split_point <= start_idx:
            return None

        middle = ctx[start_idx:split_point]
        tail = ctx[split_point:]

        # Build summary request
        middle_text = _messages_to_text(middle)
        summary = await _get_summary(llm, middle_text, self.settings)
        if not summary:
            return None

        # Replace context: system + summary + tail
        summary_msg = Message(
            role=Role.ASSISTANT,
            content=f"[Context Summary]\n{summary}",
        )

        new_ctx: list[Message] = []
        if system_msg:
            new_ctx.append(system_msg)
        new_ctx.append(summary_msg)
        new_ctx.extend(tail)

        compressed_count = len(middle)
        self.session.context = new_ctx
        self.session.save_context()

        logger.info(
            "Session %s compacted: %d messages → summary + %d tail",
            self.session.id,
            compressed_count,
            len(tail),
        )
        return summary

    async def compact_work_nodes(
        self, llm: LLMClient, node_ids: list[str] | None = None
    ) -> str | None:
        """Compress messages belonging to specific work nodes.

        If node_ids is None, compresses all completed work nodes
        (those not referenced in recent messages).

        Returns the summary text on success, None on failure.
        """
        ctx = self.session.context

        # Collect all work_nodes from tool_calls in context
        all_nodes = _extract_work_nodes(ctx)
        if not all_nodes:
            return None

        # Determine which nodes to compress
        if node_ids is None:
            # Find recent nodes (last 2 rounds) to keep
            recent_nodes = _extract_work_nodes_tail(ctx, self.settings.compact_keep_rounds)
            target_nodes = all_nodes - recent_nodes
        else:
            target_nodes = set(node_ids)

        if not target_nodes:
            return None

        # Partition context into: to_compress and to_keep
        system_msg = ctx[0] if ctx and ctx[0].role == Role.SYSTEM else None
        start_idx = 1 if system_msg else 0

        to_compress: list[Message] = []
        to_keep: list[Message] = []

        # We need to keep messages in order. Walk through and decide per-message.
        i = start_idx
        while i < len(ctx):
            msg = ctx[i]
            node = _get_message_work_node(msg, ctx, i)
            if node and node in target_nodes:
                to_compress.append(msg)
            else:
                to_keep.append(msg)
            i += 1

        if not to_compress:
            return None

        # Summarize compressed messages
        text = _messages_to_text(to_compress)
        summary = await _get_summary(llm, text, self.settings)
        if not summary:
            return None

        summary_msg = Message(
            role=Role.ASSISTANT,
            content=f"[Work Node Summary: {', '.join(sorted(target_nodes))}]\n{summary}",
        )

        new_ctx: list[Message] = []
        if system_msg:
            new_ctx.append(system_msg)
        new_ctx.append(summary_msg)
        new_ctx.extend(to_keep)

        self.session.context = new_ctx
        self.session.save_context()

        logger.info(
            "Session %s work node compact: nodes=%s, %d messages compressed",
            self.session.id,
            sorted(target_nodes),
            len(to_compress),
        )
        return summary


def _extract_work_nodes(ctx: list[Message]) -> set[str]:
    """Extract all work_node values from tool_calls in context."""
    nodes: set[str] = set()
    for msg in ctx:
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.arguments)
                    wn = args.get("work_node", "")
                    if wn:
                        nodes.add(wn)
                except (json.JSONDecodeError, AttributeError):
                    pass
    return nodes


def _extract_work_nodes_tail(ctx: list[Message], keep_rounds: int) -> set[str]:
    """Extract work_nodes from the last N rounds of context."""
    # First pass: find the boundary index
    boundary = 0
    rounds_found = 0
    for i in range(len(ctx) - 1, -1, -1):
        if ctx[i].role == Role.USER:
            rounds_found += 1
            if rounds_found >= keep_rounds:
                boundary = i
                break

    # Second pass: extract work_nodes only from messages after the boundary
    return _extract_work_nodes(ctx[boundary:])


def _get_message_work_node(
    msg: Message, ctx: list[Message], idx: int
) -> str | None:
    """Determine the work_node associated with a message.

    For assistant messages with tool_calls, extract from arguments.
    For tool result messages, find the matching assistant tool_call.
    """
    if msg.tool_calls:
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.arguments)
                wn = args.get("work_node", "")
                if wn:
                    return wn
            except (json.JSONDecodeError, AttributeError):
                pass
    if msg.role == Role.TOOL and msg.tool_call_id:
        # Search backwards for matching assistant message
        for j in range(idx - 1, -1, -1):
            prev = ctx[j]
            if prev.tool_calls:
                for tc in prev.tool_calls:
                    if tc.id == msg.tool_call_id:
                        try:
                            args = json.loads(tc.arguments)
                            return args.get("work_node", "")
                        except (json.JSONDecodeError, AttributeError):
                            pass
                break  # Stop at first assistant with tool_calls
    return None


def _messages_to_text(messages: list[Message]) -> str:
    """Convert messages to a readable text block for summarization."""
    parts: list[str] = []
    for m in messages:
        role = m.role.value.upper()
        if m.tool_calls:
            calls = ", ".join(tc.function_name for tc in m.tool_calls)
            parts.append(f"[{role}] (tool calls: {calls})")
        if m.content:
            text = m.content[:2000]
            if m.name:
                parts.append(f"[{role}/{m.name}] {text}")
            else:
                parts.append(f"[{role}] {text}")
    return "\n".join(parts)


async def _get_summary(llm: LLMClient, text: str, settings: Settings) -> str | None:
    """Call LLM to summarize a block of conversation text."""
    messages = [
        Message(role=Role.SYSTEM, content=COMPACT_PROMPT),
        Message(role=Role.USER, content=text),
    ]
    try:
        response = await llm.chat(messages)
        return response.content.strip() if response.content else None
    except Exception:
        logger.exception("Context compression LLM call failed")
        return None
