# SpicyClaw — 产品需求文档

> **版本**: 0.1.0
> **最后更新**: 2026-03-20
> **状态**: v1 已实现

---

## 1. 概述

**SpicyClaw** 是一个轻量级、可自托管的 AI Agent 平台。它对接任何 OpenAI 兼容的 LLM API，通过结构化的工作循环自主执行 Shell 命令来完成任务。用户通过实时 Web 界面进行交互，可以观察 Agent 的思考、计划和执行过程。

### 核心价值

- **零依赖前端** — Vue 3 CDN 加载，无需构建步骤，无需 Node.js
- **任意 LLM 后端** — 支持 Ollama、vLLM、llama.cpp、OpenAI 或任何 OpenAI 兼容 API
- **结构化执行** — 每次工具调用都携带 `work_node` ID 和 `next_step` 描述，透明化进度追踪
- **自管理上下文** — 自动 token 追踪与压缩，确保不超出模型限制
- **可扩展** — 角色（YAML）、技能（Markdown）和工具（Python）均可在不修改核心代码的情况下添加

---

## 2. 架构

### 2.1 高层设计

单进程 Python 应用。FastAPI 同时提供 API 网关和 Web UI 服务。UI 与 Agent 之间的所有通信通过 WebSocket 事件进行。

```
┌─────────────────────────────────────────────────┐
│                  SpicyClaw 进程                  │
│                                                  │
│  ┌───────────┐     WebSocket      ┌───────────┐ │
│  │  Web UI   │◄──────────────────►│   网关    │ │
│  │ (Vue 3)   │   ServerEvent /    │ (FastAPI) │ │
│  │           │   ClientEvent      │           │ │
│  └───────────┘                    └─────┬─────┘ │
│                                         │       │
│                     ┌───────────────────┼───┐   │
│                     │    工作循环       │   │   │
│                     │                   ▼   │   │
│                     │  LLM ──► 工具 ──►│   │   │
│                     │   ▲               │   │   │
│                     │   └───── 上下文 ──┘   │   │
│                     └───────────────────────┘   │
│                                                  │
│  ┌──────────────────────────────────────────┐   │
│  │  存储: data/sessions/{id}/               │   │
│  │  session.json · context.json · history   │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

### 2.2 技术栈

| 层级 | 选择 | 理由 |
|------|------|------|
| 异步框架 | FastAPI (Starlette) | 原生 async、内置 WebSocket、自动 OpenAPI 文档 |
| LLM HTTP 客户端 | httpx (async) | 流式支持、超时控制、连接池 |
| 进程执行 | asyncio.create_subprocess_exec | 原生异步，零依赖 |
| 容器沙箱 | docker SDK（可选） | 隔离命令执行 |
| 配置 | pydantic-settings | 统一 CLI / env / .env 处理 |
| 存储 | JSON 文件 | 极简，每个会话一个目录 |
| 前端 | Vue 3 CDN + CSS | 无构建步骤，响应式数据驱动 UI |
| 测试 | pytest + pytest-asyncio | Python 标准测试方案 |

**要求**: Python ≥ 3.11

### 2.3 目录结构

```
spicyclaw/
├── pyproject.toml
├── src/spicyclaw/
│   ├── __main__.py              # CLI 入口
│   ├── config.py                # Pydantic Settings 配置
│   ├── common/
│   │   ├── types.py             # 共享数据模型
│   │   ├── events.py            # 事件协议定义
│   │   └── i18n.py              # 国际化 (en/zh)
│   ├── gateway/
│   │   ├── server.py            # FastAPI 应用工厂
│   │   ├── routes.py            # HTTP + WebSocket 端点
│   │   ├── session.py           # Session + SessionManager
│   │   ├── workloop.py          # 核心 Agent 工作循环
│   │   ├── llm_client.py        # OpenAI 兼容流式客户端
│   │   ├── context.py           # 上下文压缩引擎
│   │   ├── roles.py             # YAML 角色加载器
│   │   ├── skills.py            # Markdown 技能加载器
│   │   ├── sandbox.py           # Docker 容器沙箱
│   │   └── tools/
│   │       ├── base.py          # Tool 抽象基类 + 注册表 + 公共参数
│   │       ├── shell.py         # Shell 命令执行
│   │       ├── stop.py          # 工作循环停止信号
│   │       ├── memory.py        # 会话记忆读写
│   │       └── summary.py       # 上下文摘要记录
│   └── ui/web/
│       ├── router.py            # 静态文件服务
│       └── static/
│           ├── index.html       # Vue 3 SPA
│           ├── app.js           # 应用逻辑
│           ├── style.css        # 暗色主题样式
│           └── vendor/vue.global.prod.js
├── roles/                       # 角色定义（YAML）
├── skills/                      # 技能定义（Markdown）
├── data/sessions/               # 运行时会话数据
├── logs/                        # 应用日志
└── tests/                       # 测试套件
```

---

## 3. 数据模型

### 3.1 会话状态

| 状态 | 描述 |
|------|------|
| `thinking` | 等待 LLM 响应 |
| `executing` | 正在执行工具（Shell 命令等） |
| `stopped` | 空闲，等待用户输入 |
| `paused` | 单步模式 — 等待用户确认 |

### 3.2 消息角色

| 角色 | 描述 |
|------|------|
| `system` | 系统提示词（由 Agent 注入） |
| `user` | 用户输入 |
| `assistant` | LLM 响应（可能包含 tool_calls） |
| `tool` | 工具执行结果 |

### 3.3 核心模型

- **Message**: role, content, tool_calls[], tool_call_id, name。可序列化为 OpenAI 格式。
- **ToolCall**: id, function_name, arguments（JSON 字符串）。
- **ToolResult**: output, error, return_code, truncated。
- **SessionMeta**: id, title, status, model, created_at, updated_at, token_used。

---

## 4. 工具系统

### 4.1 公共参数

每次工具调用必须包含两个必选参数，自动注入到所有工具的 schema 中：

| 参数 | 类型 | 描述 |
|------|------|------|
| `work_node` | string | 当前 WBS 计划中的任务节点 ID（如 "2.3.1"） |
| `next_step` | string | Agent 下一步计划的简要描述 |

这些参数由工作循环在传递给工具实现之前提取，用于进度追踪和 UI 展示。

### 4.2 内置工具

| 工具 | 参数 | 描述 |
|------|------|------|
| `shell` | `command`（string） | 执行 bash 命令。支持超时（默认 120 秒）和输出截断（默认 10,000 字符）。 |
| `stop` | `reason`（string） | 通知工作循环停止。原因如："task complete"、"need user input" 等。 |
| `memory_read` | `filename`（string） | 从会话的 `memory/` 目录读取文件。文件不存在时列出可用文件。含路径遍历防护。 |
| `memory_write` | `filename`, `content`（strings） | 向记忆文件写入内容。必要时创建 memory 目录。 |
| `summary` | `content`（string） | 记录摘要（用于上下文压缩过程）。 |

### 4.3 技能工具

技能从 `skills/` 目录下的 `.md` 文件加载，注册为名为 `skill_{name}` 的工具。每个技能包含：

- **前置元数据**（可选 YAML）：name, description, tools 列表
- **正文**：带 `{input}` 占位符的提示词文本

调用时，技能提示词作为工具输出返回，引导 LLM 按照技能指示操作。

**示例**（`skills/web-search.md`）：

```markdown
---
name: web-search
description: 使用 curl 搜索网页
tools: [shell]
---

Use the shell tool to run: curl -s "https://api.example.com/search?q={input}&format=json"
Parse the JSON response and summarize the results.
```

---

## 5. 工作循环

### 5.1 核心流程

```
用户发送消息
    │
    ▼
构建消息: [system_prompt, ...context, user_msg]
    │
    ▼
调用 LLM（带工具定义）──stream──► 向 UI 广播内容片段
    │
    ▼
解析响应
    ├─ 有 tool_calls → 提取 work_node/next_step → 执行工具 → 收集结果
    │   ├─ YOLO 模式 → 自动继续循环
    │   └─ 单步模式 → 暂停，等待用户确认（asyncio.Event）
    ├─ 仅有 content（无 tool_calls）→ 视为对话回复，等待用户输入
    └─ JSON 解析错误 → 要求模型修正（最多连续 5 次错误）
```

### 5.2 任务初始化

当 Agent 开始新任务时，系统提示词指示它：

1. 创建 `TASK.md` — 第一行为简短标题（作为会话标题），其余描述任务内容
2. 创建 `PLAN.json` — WBS 结构：`{"nodes": [{"id": "1", "title": "...", "children": [...]}]}`
3. 按计划逐步执行，使用计划中的 `work_node` ID

### 5.3 保护机制

| 保护 | 阈值 | 动作 |
|------|------|------|
| 最大步数 | 1000（可配置） | 暂停并警告 |
| 重复错误 | 连续 5 次相同错误 | 暂停并警告 |
| 重复输出 | 连续 10 次相同输出 | 暂停并警告 |
| 格式错误 | 连续 5 次 JSON 解析失败 | 停止 |
| Token 用量 | 超过 max_tokens 的 80% | 自动压缩上下文 |

### 5.4 执行模式

| 模式 | 行为 | 激活方式 |
|------|------|----------|
| **YOLO**（默认） | 自动执行所有工具调用 | `/yolo` 命令，或配置 `yolo=True` |
| **单步** | 每次执行前暂停，需用户确认 | `/step` 命令 |

---

## 6. 上下文管理

### 6.1 Token 追踪

每次 LLM 响应后，上下文管理器从 API 的 `usage.total_tokens` 字段更新 token 用量。如果 API 未提供用量数据，则使用 `文本长度 / 4` 作为回退估算。

### 6.2 完整压缩

当 `token_used / max_tokens ≥ full_compact_ratio`（默认 0.8）时自动触发，或通过 `/compact` 手动触发。

1. 将上下文分为：`[系统提示词] [中间消息...] [最近 N 轮]`
2. 将中间消息转换为文本
3. 调用 LLM 生成简洁摘要
4. 用单条 `[Context Summary]` 助手消息替换中间部分
5. 保留系统提示词和最近轮次（默认：4 轮）

### 6.3 工作节点压缩

通过 `/compact 1.1,2.1` 触发，压缩指定工作节点。

1. 从上下文中解析工具调用参数中的 `work_node`
2. 识别属于目标节点 ID 的消息
3. 通过 LLM 生成这些消息的摘要
4. 用 `[Work Node Summary: ...]` 助手消息替换
5. 保持所有非目标消息不变

### 6.4 持久化

| 文件 | 格式 | 内容 |
|------|------|------|
| `context.json` | JSON 数组 | 当前活跃上下文窗口 |
| `history.jsonl` | JSON Lines（仅追加） | 完整消息历史（永不截断） |
| `session.json` | JSON 对象 | 会话元数据 |

---

## 7. LLM 客户端

### 7.1 流式传输

客户端使用 httpx 异步流式解析 SSE（Server-Sent Events）。内容增量以 `("chunk", response)` 事件产出，实时广播到 UI。工具调用增量在多个 chunk 中累积，流结束时最终化。

### 7.2 空闲超时与健康探测

当 `idle_timeout` 秒（默认 30 秒）内未收到数据时：

1. 发送 GET `/v1/models` 请求，5 秒超时
2. **探测成功** → 模型仍在思考，继续等待
3. **探测失败** → 标记 LLM 为不健康，每 `health_check_interval` 秒（默认 10 秒）进行后台健康轮询
4. 健康恢复后，标记为健康并恢复正常运行
5. 新请求在 `_wait_for_healthy()` 上阻塞直到恢复

---

## 8. 会话管理

### 8.1 会话生命周期

1. **创建**: `POST /api/sessions` → 新 UUID、目录、元数据
2. **活跃**: 用户发送消息 → 工作循环运行 → 工具执行
3. **持久化**: 每个工作循环步骤后保存上下文和元数据
4. **恢复**: 启动时检测中断的会话（有未完成的工具调用），可通过 `/resume` 恢复

### 8.2 会话目录

```
data/sessions/{session-id}/
├── session.json      # 元数据（id, title, status, model, timestamps, token_used）
├── context.json      # 当前上下文窗口
├── history.jsonl     # 完整消息历史（仅追加）
├── TASK.md           # Agent 创建的任务描述（第一行 = 标题）
├── PLAN.json         # Agent 创建的 WBS 计划
└── memory/           # 会话级持久记忆文件
```

### 8.3 恢复

启动时，`SessionManager.get_recoverable()` 扫描所有会话，查找有未完成工具调用的会话（助手消息包含 `tool_calls` 但没有对应的 `tool` 结果消息）。用户可通过 `/resume` 命令恢复。

---

## 9. 事件协议

### 9.1 服务端 → 客户端（ServerEvent）

| 类型 | 数据 | 描述 |
|------|------|------|
| `chunk` | `{text}` | LLM 流式内容增量 |
| `tool_call` | `{tool_call_id, name, arguments, work_node, next_step}` | 模型请求执行工具 |
| `tool_end` | `{tool_call_id, output, error, return_code, truncated}` | 工具执行结果 |
| `status` | `{status}` | 会话状态变更 |
| `session_update` | `{title, ...}` | 会话元数据变更（如从 TASK.md 获取的标题） |
| `error` | `{message}` | 错误信息 |
| `system` | `{message}` | 系统通知（保护警告、模式切换等） |

### 9.2 客户端 → 服务端（ClientEvent）

| 类型 | 数据 | 描述 |
|------|------|------|
| `message` | `{content}` | 用户发送消息 |
| `confirm` | `{}` | 用户在单步模式中确认执行 |
| `abort` | `{}` | 用户停止当前执行 |
| `command` | `{command, args}` | 用户发出斜杠命令 |

所有事件包含 `session_id` 和时间戳。

---

## 10. 角色系统

角色在 `roles/` 目录下的 YAML 文件中定义。

**格式**：

```yaml
name: programmer
description: 专注软件开发的 Agent
system_prompt: |
  You are an expert programmer. Focus on writing clean, well-tested code.
tools:
  - shell
  - stop
  - memory_read
  - memory_write
```

**行为**: 通过 `/session <role>` 设置角色后，其 `system_prompt` 被添加到默认系统提示词之前。角色可在会话中途更换。

---

## 11. Docker 沙箱（可选）

需要：`pip install spicyclaw[sandbox]`（安装 `docker>=7.0`）

### 11.1 容器管理

- `create(session_id, workspace_dir, image)` → 创建容器，将会话工作空间挂载到 `/workspace`
- `exec(command, timeout, workdir)` → 在容器内执行命令，返回 (stdout, stderr, return_code)
- `destroy()` → 停止并删除容器
- `cleanup_stale()` → 清理上次运行遗留的 `spicyclaw-*` 容器

### 11.2 资源限制

- 内存：512MB
- CPU：50%（cpu_quota=50000/cpu_period=100000）
- 网络：bridge 模式

---

## 12. 用户命令

所有命令通过 WebSocket 以 `command` 事件发送。也可在输入框中以 `/` 前缀输入。

| 命令 | 参数 | 描述 |
|------|------|------|
| `/help` | — | 显示可用命令 |
| `/yolo` | — | 切换到 YOLO 模式（自动执行） |
| `/step` | — | 切换到单步模式（每次执行需确认） |
| `/stop` | — | 中止当前执行 |
| `/compact` | `[node_ids]` | 压缩上下文。可选：逗号分隔的工作节点 ID 进行选择性压缩 |
| `/status` | — | 显示会话状态、token 用量、模式、角色、消息数 |
| `/task` | — | 显示 TASK.md 内容 |
| `/plan` | — | 显示 PLAN.json 内容 |
| `/session` | `[role_name]` | 显示会话信息，或设置活跃角色 |
| `/settings` | — | 显示当前配置值 |
| `/resume` | — | 恢复中断的工作循环 |

---

## 13. Web UI

### 13.1 布局

- **侧边栏**（260px）：会话列表，带状态指示点（颜色编码：蓝色=思考中，绿色=执行中，黄色=已暂停，灰色=已停止）。新建会话按钮。
- **主区域**：聊天视图，自动滚动。消息类型：
  - **用户**：右对齐气泡，蓝色背景
  - **助手**：左对齐，带边框，流式光标动画
  - **工具**：紧凑卡片，显示工具名、work_node 徽章、返回码（绿/红），可折叠输出
  - **系统/错误**：居中，灰色/红色文字

### 13.2 输入区域

- 自适应高度文本区域（最大 200px）
- Enter 发送，Shift+Enter 换行
- `/command` 检测 — 作为命令事件发送，而非普通消息
- 思考/执行状态下禁用输入
- 执行期间显示停止按钮
- 暂停（单步模式）时显示确认 + 中止按钮

### 13.3 WebSocket

- 切换会话时自动连接
- 断开后 2 秒重连
- 重连后重新获取上下文以同步状态
- 连接状态指示器

---

## 14. 国际化

支持英语（`en`）和简体中文（`zh`）。设置方式：

- 配置：`lang: "zh"`
- CLI：`--lang zh`
- 环境变量：`SPICYCLAW_LANG=zh`

工作循环和命令系统中的所有用户可见字符串使用 `t(key, **kwargs)` 翻译函数。Web UI 标签保持英语（可通过 HTML 模板本地化）。

---

## 15. 配置

所有设置使用 `SPICYCLAW_` 环境变量前缀，可通过 `.env` 文件、环境变量或 CLI 参数设置。

| 设置 | 默认值 | 描述 |
|------|--------|------|
| `api_base_url` | `http://localhost:11434/v1` | LLM API 端点 |
| `api_key` | `sk-placeholder` | API 认证密钥 |
| `model` | `qwen2.5:14b` | 模型标识符 |
| `max_tokens` | `32768` | 上下文窗口限制（用于自动压缩） |
| `request_timeout` | `1800`（30分钟） | 最大请求持续时间 |
| `idle_timeout` | `30` | 空闲健康探测前的等待秒数 |
| `health_check_interval` | `10` | 不健康状态下的健康轮询间隔 |
| `host` | `127.0.0.1` | 服务器绑定地址 |
| `port` | `8000` | 服务器绑定端口 |
| `max_steps` | `1000` | 工作循环迭代上限 |
| `yolo` | `true` | 默认执行模式 |
| `max_repeat_errors` | `5` | 连续相同错误阈值 |
| `max_repeat_outputs` | `10` | 连续相同输出阈值 |
| `full_compact_ratio` | `0.8` | 触发自动压缩的 token 用量比例 |
| `compact_keep_rounds` | `4` | 压缩时保留的对话轮数 |
| `shell_timeout` | `120` | Shell 命令超时（秒） |
| `shell_max_output` | `10000` | Shell 输出截断（字符数） |
| `lang` | `en` | UI 语言（`en` 或 `zh`） |
| `data_dir` | `data` | 会话存储路径 |
| `roles_dir` | `roles` | 角色定义路径 |
| `skills_dir` | `skills` | 技能定义路径 |
| `logs_dir` | `logs` | 日志输出路径 |

---

## 16. API 端点

### HTTP

| 方法 | 路径 | 描述 |
|------|------|------|
| `POST` | `/api/sessions` | 创建新会话 |
| `GET` | `/api/sessions` | 列出所有会话 |
| `GET` | `/api/sessions/{id}` | 获取会话元数据 |
| `POST` | `/api/sessions/{id}/message` | 发送消息并启动工作循环 |
| `POST` | `/api/sessions/{id}/abort` | 中止当前执行 |
| `GET` | `/api/sessions/{id}/context` | 获取完整上下文（消息历史） |

### WebSocket

| 路径 | 描述 |
|------|------|
| `/api/sessions/{id}/ws` | 实时双向事件流 |

### 静态资源

| 路径 | 描述 |
|------|------|
| `/` | Web UI（index.html） |
| `/static/*` | CSS、JS、第三方资源 |

---

## 17. 测试

### 测试分类

| 分类 | 文件数 | 描述 |
|------|--------|------|
| 单元测试 | 16 个文件 | 所有模块的 Mock 测试 |
| 集成测试 | 1 个文件 | 真实 LLM API 测试（上下文压缩、工作循环、健康探测） |
| 合计 | 193 个测试 | 187 单元 + 6 集成 |

### 运行测试

```bash
# 仅单元测试（快速，无外部依赖）
pytest tests/ --ignore=tests/test_integration_phase4.py

# 集成测试（需要 LLM API 访问）
pytest tests/test_integration_phase4.py -v -s

# 所有测试
pytest tests/
```

---

## 18. 部署

### 快速开始

```bash
# 安装
pip install -e .

# 使用默认配置运行（本地 Ollama localhost:11434）
spicyclaw

# 使用自定义 LLM
spicyclaw --api-base-url http://your-llm:8080/v1 --model your-model --api-key your-key

# 以中文运行
spicyclaw --lang zh

# 启用 Docker 沙箱支持
pip install -e ".[sandbox]"
```

### 环境变量

```bash
export SPICYCLAW_API_BASE_URL=http://your-llm:8080/v1
export SPICYCLAW_MODEL=your-model
export SPICYCLAW_API_KEY=your-key
export SPICYCLAW_LANG=zh
```

或在工作目录创建 `.env` 文件。

---

## 19. 依赖

### 必需

```
fastapi >= 0.115
uvicorn[standard] >= 0.34
httpx >= 0.28
pydantic >= 2.0
pydantic-settings >= 2.0
```

### 可选

```
docker >= 7.0      # 容器沙箱
pyyaml >= 6.0      # 角色加载（自动检测）
```

### 开发

```
pytest
pytest-asyncio
httpx              # 测试客户端
```
