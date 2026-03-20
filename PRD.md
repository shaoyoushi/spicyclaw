# SpicyClaw — Product Requirements Document

> **Version**: 0.2.0
> **Last Updated**: 2026-03-20
> **Status**: v1 Implemented

---

## 1. Overview

**SpicyClaw** is a lightweight, self-hosted AI Agent platform. It connects to any OpenAI-compatible LLM API and autonomously completes tasks by executing shell commands in a structured work loop. Users interact through a real-time web interface, watching the agent think, plan, and execute.

### Core Value Proposition

- **Zero-dependency frontend** — Vue 3 CDN, no build step, no Node.js required
- **Any LLM backend** — works with Ollama, vLLM, llama.cpp, OpenAI, or any OpenAI-compatible API
- **Structured execution** — every tool call carries a `work_node` ID and `next_step` description for transparent progress tracking
- **Self-managing context** — automatic token tracking and compression to stay within model limits
- **Extensible** — roles (YAML), skills (Markdown), and tools (Python) can be added without modifying core code

---

## 2. Architecture

### 2.1 High-Level Design

Single-process Python application. FastAPI serves both the API gateway and the web UI. All communication between the UI and the agent flows through WebSocket events.

```
┌─────────────────────────────────────────────────┐
│                  SpicyClaw Process               │
│                                                  │
│  ┌───────────┐     WebSocket      ┌───────────┐ │
│  │  Web UI   │◄──────────────────►│  Gateway  │ │
│  │ (Vue 3)   │   ServerEvent /    │ (FastAPI) │ │
│  │           │   ClientEvent      │           │ │
│  └───────────┘                    └─────┬─────┘ │
│                                         │       │
│                     ┌───────────────────┼───┐   │
│                     │    Work Loop      │   │   │
│                     │                   ▼   │   │
│                     │  LLM ──► Tools ──►│   │   │
│                     │   ▲               │   │   │
│                     │   └───── Context ──┘   │   │
│                     └───────────────────────┘   │
│                                                  │
│  ┌──────────────────────────────────────────┐   │
│  │  Storage: data/sessions/{id}/            │   │
│  │  session.json · context.json · history   │   │
│  │  workspace/ · memory/                    │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

### 2.2 Tech Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Async framework | FastAPI (Starlette) | Native async, built-in WebSocket, auto OpenAPI docs |
| LLM HTTP client | httpx (async) | Streaming support, timeout control, connection pooling |
| Process execution | asyncio.create_subprocess_exec | Native async, zero-dependency |
| Container sandbox | docker SDK (optional) | Isolated command execution |
| Configuration | pydantic-settings | Unified CLI / env / .env handling |
| Storage | JSON files | Minimal, one directory per session |
| Frontend | Vue 3 CDN + CSS | No build step, reactive data-driven UI |
| Testing | pytest + pytest-asyncio | Standard Python testing |

**Requirements**: Python ≥ 3.11

### 2.3 Directory Structure

```
spicyclaw/
├── pyproject.toml
├── src/spicyclaw/
│   ├── __main__.py              # CLI entry point
│   ├── config.py                # Pydantic Settings
│   ├── common/
│   │   ├── types.py             # Shared data models
│   │   ├── events.py            # Event protocol definitions
│   │   └── i18n.py              # Internationalization (en/zh)
│   ├── gateway/
│   │   ├── server.py            # FastAPI app factory
│   │   ├── routes.py            # HTTP + WebSocket endpoints
│   │   ├── session.py           # Session + SessionManager
│   │   ├── workloop.py          # Core agent work loop
│   │   ├── llm_client.py        # OpenAI-compatible streaming client
│   │   ├── context.py           # Context compression engine
│   │   ├── roles.py             # YAML role loader
│   │   ├── skills.py            # Markdown skill loader
│   │   ├── sandbox.py           # Docker container sandbox
│   │   └── tools/
│   │       ├── base.py          # Tool ABC + registry + common params
│   │       ├── shell.py         # Shell command execution
│   │       ├── stop.py          # Workloop stop signal
│   │       ├── memory.py        # Session memory read/write
│   │       └── summary.py       # Context summary recording
│   └── ui/web/
│       ├── router.py            # Static file serving
│       └── static/
│           ├── index.html       # Vue 3 SPA
│           ├── app.js           # Application logic
│           ├── style.css        # Dark theme styles
│           └── vendor/vue.global.prod.js
├── roles/                       # Role definitions (YAML)
├── skills/                      # Skill definitions (Markdown)
├── data/sessions/               # Runtime session data
├── logs/                        # Application logs
└── tests/                       # Test suite
```

---

## 3. Data Models

### 3.1 Session Status

| Status | Description |
|--------|-------------|
| `thinking` | Waiting for LLM response |
| `executing` | Running a tool (shell command, etc.) |
| `stopped` | Idle, waiting for user input |
| `paused` | Step mode — waiting for user confirmation |

### 3.2 Message Roles

| Role | Description |
|------|-------------|
| `system` | System prompt (injected by agent) |
| `user` | User input |
| `assistant` | LLM response (may include tool_calls) |
| `tool` | Tool execution result |

### 3.3 Core Models

- **Message**: role, content, tool_calls[], tool_call_id, name, ts (Unix timestamp). Serializable to OpenAI format.
- **ToolCall**: id, function_name, arguments (JSON string).
- **ToolResult**: output, error, return_code, truncated.
- **SessionMeta**: id, title, status, model, created_at, updated_at, token_used.

---

## 4. Tool System

### 4.1 Common Parameters

Every tool call must include two mandatory parameters, automatically injected into all tool schemas:

| Parameter | Type | Description |
|-----------|------|-------------|
| `work_node` | string | Current task node ID from the WBS plan (e.g., "2.3.1") |
| `next_step` | string | Brief description of what the agent plans to do AFTER the current tool call finishes (not the current step) |

These parameters are extracted by the work loop before passing remaining arguments to the tool implementation. They serve for progress tracking and UI display.

**Validation**: If either `work_node` or `next_step` is missing from a tool call, the work loop returns an error to the model requiring it to retry with proper parameters. The tool is NOT executed.

### 4.2 Built-in Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `shell` | `command` (string) | Execute a bash command in the session's `workspace/` directory. Supports timeout (default 120s) and output truncation (default 10,000 chars). |
| `stop` | `reason` (string) | Signal the work loop to stop. Reasons: "task complete", "need user input", etc. |
| `memory_read` | `filename` (string) | Read a file from the session's `memory/` directory. For persistent session notes ONLY — not for task files. Lists available files on not-found. Path traversal protection. |
| `memory_write` | `filename`, `content` (strings) | Write content to a memory file. For persistent session notes ONLY — not for TASK.md or PLAN.json. |
| `summary` | `content` (string) | Record a summary (used during context compression). |

### 4.3 Skill Tools

Skills are loaded from `.md` files in the `skills/` directory and registered as tools named `skill_{name}`. Each skill has:

- **Frontmatter** (optional YAML): name, description, tools list
- **Body**: prompt text with `{input}` placeholder

When invoked, the skill prompt is returned as tool output, guiding the LLM to follow the skill's instructions.

**Example** (`skills/web-search.md`):

```markdown
---
name: web-search
description: Search the web using curl
tools: [shell]
---

Use the shell tool to run: curl -s "https://api.example.com/search?q={input}&format=json"
Parse the JSON response and summarize the results.
```

---

## 5. Work Loop

### 5.1 Core Flow

```
User sends message
    │
    ▼
Build messages: [system_prompt, ...context, user_msg]
    │
    ▼
Call LLM (with tool definitions) ──stream──► broadcast content chunks to UI
    │
    ▼
Parse response
    ├─ Has tool_calls → validate work_node/next_step → execute tools → collect results
    │   ├─ Missing work_node/next_step → return error to model, require retry
    │   ├─ YOLO mode → auto-continue loop
    │   └─ Step mode → pause, wait for user confirm (asyncio.Event)
    ├─ Content only (no tool_calls) → treat as conversational reply, wait for user
    └─ JSON parse error → tell model to fix (max 5 consecutive errors)
```

### 5.2 Task Initialization

When the agent starts a new task, the system prompt instructs it to:

1. Use the **shell tool** to create `TASK.md` in the workspace — first line is a short title (becomes session title), rest describes the task
2. Use the **shell tool** to create `PLAN.json` in the workspace — WBS structure: `{"nodes": [{"id": "1", "title": "...", "children": [...]}]}`. Use `jq` for precise reading and writing rather than rewriting the entire file each time.
3. Execute the plan step by step, using `work_node` IDs from the plan

### 5.3 Protection Mechanisms

| Protection | Threshold | Action |
|------------|-----------|--------|
| Max steps | 1000 (configurable) | Pause with warning |
| Repeated error | 5 consecutive identical errors | Pause with warning |
| Repeated output | 10 consecutive identical outputs | Pause with warning |
| Format errors | 5 consecutive JSON parse failures | Stop |
| Token usage | 80% of max_tokens | Auto-compact context |
| Missing common params | work_node or next_step absent | Return error to model |

### 5.4 Execution Modes

| Mode | Behavior | Activation |
|------|----------|------------|
| **YOLO** (default) | Auto-execute all tool calls | `/yolo` command, or `yolo=True` in config |
| **Step** | Pause before each execution batch, require user confirmation | `/step` command |

---

## 6. Context Management

### 6.1 Token Tracking

After each LLM response, the context manager updates token usage from the API's `usage.total_tokens` field. If the API doesn't provide usage data, a fallback estimate of `text_length / 4` is used.

### 6.2 Full Compression

**Auto-triggered** when `token_used / max_tokens ≥ full_compact_ratio` (default 0.8).

**Manually triggered** via `/compact` — this forces compression regardless of current context size.

1. Split context into: `[system_prompt] [middle messages...] [recent N rounds]`
2. Convert middle messages to text
3. Call LLM to generate a concise summary
4. Replace middle with a single `[Context Summary]` assistant message
5. Preserve system prompt and recent rounds (default: 4)

### 6.3 Work Node Compression

Triggered via `/compact 1.1,2.1` to compress specific work nodes.

1. Parse `work_node` from tool_call arguments across the context
2. Identify messages belonging to target node IDs
3. Summarize those messages via LLM
4. Replace with a `[Work Node Summary: ...]` assistant message
5. Keep all non-target messages intact

### 6.4 Persistence

| File | Format | Content |
|------|--------|---------|
| `context.json` | JSON array | Current active context window |
| `history.jsonl` | JSON Lines (append-only) | Complete message history (never truncated) |
| `session.json` | JSON object | Session metadata |

---

## 7. LLM Client

### 7.1 Streaming

The client uses httpx async streaming to parse SSE (Server-Sent Events). Content deltas are yielded as `("chunk", response)` events and broadcast to the UI in real-time. Tool call deltas are accumulated across chunks and finalized when the stream ends.

### 7.2 Idle Timeout & Health Probing

When no data is received within `idle_timeout` seconds (default 30):

1. Send GET `/v1/models` with 5-second timeout
2. **Probe succeeds** → model is still thinking, continue waiting
3. **Probe fails** → mark LLM as unhealthy, start background health polling every `health_check_interval` seconds (default 10)
4. When health recovers, mark healthy and resume normal operation
5. New requests block on `_wait_for_healthy()` until recovery

---

## 8. Session Management

### 8.1 Session Lifecycle

1. **Create**: `POST /api/sessions` → new UUID, directory structure, metadata
2. **Active**: user sends messages → work loop runs → tools execute
3. **Persist**: context and metadata saved after each work loop step
4. **Recover**: on startup, interrupted sessions (pending tool calls without results) are detected and can be resumed via `/resume`

### 8.2 Session Directory

```
data/sessions/{session-id}/
├── session.json      # Metadata (id, title, status, model, timestamps, token_used)
├── context.json      # Current context window
├── history.jsonl     # Full message history (append-only)
├── workspace/        # Agent working directory (model's cwd)
│   ├── TASK.md       # Agent-created task description (first line = title)
│   ├── PLAN.json     # Agent-created WBS plan
│   └── ...           # Any files the agent creates during task execution
└── memory/           # Session-level persistent memory files (via memory tools)
```

**Key separation**: The `workspace/` directory is the model's working directory — all shell commands execute here. TASK.md, PLAN.json, source code, and other task-related files go in workspace. Session infrastructure files (context, history, metadata) and memory files stay outside workspace. When using Docker sandbox, `workspace/` is bind-mounted to `/workspace` inside the container.

### 8.3 Recovery

On startup, `SessionManager.get_recoverable()` scans all sessions for those with pending tool calls (assistant message with `tool_calls` but no matching `tool` result messages). Users can resume via the `/resume` command.

---

## 9. Event Protocol

### 9.1 Server → Client (ServerEvent)

| Type | Data | Description |
|------|------|-------------|
| `chunk` | `{text}` | LLM streaming content delta |
| `tool_call` | `{tool_call_id, name, arguments, work_node, next_step}` | Model requests tool execution |
| `tool_end` | `{tool_call_id, output, error, return_code, truncated}` | Tool execution result |
| `status` | `{status}` | Session status change |
| `session_update` | `{title, ...}` | Session metadata change (e.g., title from TASK.md) |
| `error` | `{message}` | Error message |
| `system` | `{message}` | System notification (protection warnings, mode changes) |

All ServerEvents include `session_id` and `ts` (Unix timestamp).

### 9.2 Client → Server (ClientEvent)

| Type | Data | Description |
|------|------|-------------|
| `message` | `{content}` | User sends a message (queued if agent is running) |
| `confirm` | `{}` | User confirms execution in step mode |
| `abort` | `{}` | User stops current execution |
| `command` | `{command, args}` | User issues a slash command |

All ClientEvents include `session_id`.

---

## 10. Role System

Roles are defined in YAML files under the `roles/` directory.

**Format**:

```yaml
name: programmer
description: Software development focused agent
system_prompt: |
  You are an expert programmer. Focus on writing clean, well-tested code.
tools:
  - shell
  - stop
  - memory_read
  - memory_write
```

**Behavior**: When a role is set via `/session <role>`, its `system_prompt` is prepended to the default system prompt. The role can be changed mid-session.

---

## 11. Docker Sandbox (Optional)

Requires: `pip install spicyclaw[sandbox]` (installs `docker>=7.0`)

### 11.1 Container Management

- `create(session_id, workspace_dir, image)` → creates a container with the session's `workspace/` directory bind-mounted to `/workspace`
- `exec(command, timeout, workdir)` → runs a command inside the container, returns (stdout, stderr, return_code)
- `destroy()` → stops and removes the container
- `cleanup_stale()` → removes leftover `spicyclaw-*` containers from previous runs

### 11.2 Resource Limits

- Memory: 512MB
- CPU: 50% (cpu_quota=50000/cpu_period=100000)
- Network: bridge mode

---

## 12. User Commands

All commands are sent via WebSocket as `command` events. They can also be typed in the input box with a `/` prefix. **Commands are available at all times**, including while the agent is running.

| Command | Arguments | Description |
|---------|-----------|-------------|
| `/help` | — | Show available commands |
| `/yolo` | — | Switch to YOLO mode (auto-execute) |
| `/step` | — | Switch to Step mode (confirm each execution) |
| `/stop` | — | Abort current execution |
| `/compact` | `[node_ids]` | Force compress context. Optional: comma-separated work node IDs for selective compression. Manual compact ignores context size threshold. |
| `/status` | — | Show session status, tokens, mode, role, message count |
| `/task` | — | Display TASK.md content (from workspace/) |
| `/plan` | — | Display PLAN.json content (from workspace/) |
| `/session` | `[role_name]` | Show session info, or set active role |
| `/settings` | — | Show current configuration values |
| `/resume` | — | Resume an interrupted work loop |

---

## 13. Web UI

### 13.1 Layout

- **Sidebar** (260px): Session list with status dots (color-coded: blue=thinking, green=executing, yellow=paused, gray=stopped). New session button.
- **Main Area**: Chat view with auto-scroll. Message types:
  - **User**: right-aligned bubble with blue background, timestamp
  - **Assistant**: left-aligned with border, streaming cursor animation, timestamp
  - **Tool**: compact card showing tool name, work_node badge, return code (green/red), collapsible output, timestamp
  - **System/Error**: centered, muted/red text, timestamp

### 13.2 Input Area

- Textarea with auto-height (max 200px)
- Enter to send, Shift+Enter for newline
- `/command` detection — sent as command events instead of messages
- Input is **always enabled** — messages sent while the agent is running are queued and visible to the agent on its next LLM call
- Stop button during active execution
- Confirm + Abort buttons during paused (step mode) state

### 13.3 Timestamps

All messages and events display timestamps (HH:MM:SS format). Messages include a `ts` field (Unix timestamp) for precise recording. Timestamps are shown in the user's local timezone.

### 13.4 WebSocket

- Auto-connect on session switch
- 2-second reconnect on disconnect
- Re-fetches context on reconnect for state sync
- Connection status indicator

---

## 14. Internationalization

Supports English (`en`) and Simplified Chinese (`zh`). Set via:

- Config: `lang: "zh"`
- CLI: `--lang zh`
- Environment: `SPICYCLAW_LANG=zh`

All user-facing strings in the work loop and command system use the `t(key, **kwargs)` translation function. The Web UI labels remain in English (can be localized via the HTML template).

---

## 15. Configuration

All settings use the `SPICYCLAW_` environment variable prefix and can be set via `.env` file, environment variables, or CLI arguments.

| Setting | Default | Description |
|---------|---------|-------------|
| `api_base_url` | `http://localhost:11434/v1` | LLM API endpoint |
| `api_key` | `sk-placeholder` | API authentication key |
| `model` | `qwen2.5:14b` | Model identifier |
| `max_tokens` | `32768` | Context window limit for auto-compact |
| `request_timeout` | `1800` (30min) | Maximum request duration |
| `idle_timeout` | `30` | Seconds before idle health probe |
| `health_check_interval` | `10` | Health polling interval when unhealthy |
| `host` | `127.0.0.1` | Server bind address |
| `port` | `8000` | Server bind port |
| `max_steps` | `1000` | Work loop iteration limit |
| `yolo` | `true` | Default execution mode |
| `max_repeat_errors` | `5` | Consecutive identical error threshold |
| `max_repeat_outputs` | `10` | Consecutive identical output threshold |
| `full_compact_ratio` | `0.8` | Token usage ratio to trigger auto-compact |
| `compact_keep_rounds` | `4` | Conversation rounds to preserve during compression |
| `shell_timeout` | `120` | Shell command timeout (seconds) |
| `shell_max_output` | `10000` | Shell output truncation (characters) |
| `lang` | `en` | UI language (`en` or `zh`) |
| `data_dir` | `data` | Session storage path |
| `roles_dir` | `roles` | Role definitions path |
| `skills_dir` | `skills` | Skill definitions path |
| `logs_dir` | `logs` | Log output path |

---

## 16. API Endpoints

### HTTP

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/sessions` | Create a new session |
| `GET` | `/api/sessions` | List all sessions |
| `GET` | `/api/sessions/{id}` | Get session metadata |
| `POST` | `/api/sessions/{id}/message` | Send message and start work loop |
| `POST` | `/api/sessions/{id}/abort` | Abort current execution |
| `GET` | `/api/sessions/{id}/context` | Get full context (message history) |

### WebSocket

| Path | Description |
|------|-------------|
| `/api/sessions/{id}/ws` | Real-time bidirectional event stream |

### Static

| Path | Description |
|------|-------------|
| `/` | Web UI (index.html) |
| `/static/*` | CSS, JS, vendor assets |

---

## 17. Testing

### Test Categories

| Category | Files | Description |
|----------|-------|-------------|
| Unit tests | 17 files | Mock-based tests for all modules |
| Integration tests | 1 file | Real LLM API tests (context compression, workloop, health probe) |
| Total | 217 tests | 211 unit + 6 integration |

### Running Tests

```bash
# Unit tests only (fast, no external dependencies)
pytest tests/ --ignore=tests/test_integration_phase4.py

# Integration tests (requires LLM API access)
pytest tests/test_integration_phase4.py -v -s

# All tests
pytest tests/
```

---

## 18. Deployment

### Quick Start

```bash
# Install
pip install -e .

# Run with defaults (Ollama on localhost:11434)
spicyclaw

# Run with custom LLM
spicyclaw --api-base-url http://your-llm:8080/v1 --model your-model --api-key your-key

# Run in Chinese
spicyclaw --lang zh

# With Docker sandbox support
pip install -e ".[sandbox]"
```

### Environment Variables

```bash
export SPICYCLAW_API_BASE_URL=http://your-llm:8080/v1
export SPICYCLAW_MODEL=your-model
export SPICYCLAW_API_KEY=your-key
export SPICYCLAW_LANG=zh
```

Or create a `.env` file in the working directory.

---

## 19. Dependencies

### Required

```
fastapi >= 0.115
uvicorn[standard] >= 0.34
httpx >= 0.28
pydantic >= 2.0
pydantic-settings >= 2.0
```

### Optional

```
docker >= 7.0      # For container sandbox
pyyaml >= 6.0      # For role loading (auto-detected)
```

### Development

```
pytest
pytest-asyncio
httpx              # For test client
```
