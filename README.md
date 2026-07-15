# StudyLoop

[![CI](https://github.com/tx3457/study-loop/actions/workflows/ci.yml/badge.svg)](https://github.com/tx3457/study-loop/actions/workflows/ci.yml)

StudyLoop 是一个面向个人学习材料的自适应辅导系统：它把文档检索、练习生成、作答批改、学习画像更新和下一步教学决策连成反馈闭环，并提供受轮次上限约束的工具调用 Agent。

> 当前定位是可审阅的学习/作品集项目，不是生产级教学平台。运行真实检索与生成需要自行配置 OpenAI-compatible chat、structured-output 和 embedding 服务。

## 它解决什么问题

普通“上传文档后出题”的应用在一次生成后就结束，后续练习无法利用学生刚产生的表现。StudyLoop 将上一轮的成绩、薄弱点和掌握度作为下一轮 observation，让系统在升难度、补薄弱点、讲解、继续巩固、转学习路径或结束之间做出选择。

```text
学习材料
   |
   v
解析与切块 -> Hybrid Retrieval -> 生成练习 -> 用户作答 -> 批改
                                                       |
                                                       v
                              下一步教学动作 <- 更新学习画像
                                      |
                                      +----> 下一轮
```

项目的重点不是技术名词数量，而是三个可检查的闭环：

1. 检索结果进入题目生成与质量检查，而不是只建一个向量库。
2. 工具执行结果以 observation 回到模型上下文，模型可以据此继续调用其他工具或结束。
3. 批改结果写入 learner memory，并影响下一轮自适应动作。

## 哪些部分是真正的 Agent

使用 LangGraph 并不自动等于 Agent。仓库按实际控制权区分如下：

| 模块 | 模型拥有的决策权 | 准确分类 |
| --- | --- | --- |
| `/agent/autonomous` | 每轮动态选择业务工具、`ask_user` 或 `finalize`；工具 observation 回灌后再决策；最多 8 轮 | Tool-using Agent |
| `/agent/tutor/assist` | 使用同一工具循环，并通过 `interrupt` / `Command(resume=...)` 完成人机协作暂停恢复 | Tool-using Agent + HITL |
| `/agent/adaptive/*` | 模型基于成绩、画像和轨迹选择结构化教学动作，代码执行预定义分支 | Agentic Workflow |
| `/agent/run`、quiz/critic/reviser 图 | 节点顺序、阈值与回环由开发者预先编码 | LangGraph Workflow |
| 文档检索、学习画像 | 不包含自主行动循环 | RAG / Application Memory |

更完整的边界与状态流见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 关键设计与边界

### 1. 长文档不能直接塞进提示词

上传文件先经过格式白名单、大小限制、解析和自适应切块，再进入 Chroma 向量索引与 BM25 语料。查询阶段使用向量检索与 BM25 的 RRF 融合；query rewrite、HyDE、multi-query 和 cross-encoder reranker 都是可选阶段，可独立关闭。

这证明了检索链路存在，但不证明检索效果优于任何基线。公开仓库没有发布 Recall@K、MRR 或引用准确率，原因见 [docs/EVALUATION.md](docs/EVALUATION.md)。

### 2. 工具调用必须形成 observation 闭环

`ToolRegistry` 集中注册检索、出题、批改、画像更新和学习路径工具，并记录 timeout、retry 与 audit 配置。`tool_loop` 将模型给出的 JSON 参数交给白名单工具执行，再把结果追加为 tool message，下一轮模型能够据此改变动作。

当前 JSON schema 主要用于约束模型输出；它不是完整的服务端 Pydantic 参数校验。`ToolMetadata.permission` 也只是描述性元数据，不应被理解为已经实现的鉴权系统。

### 3. 运行状态与学习记忆分开管理

- LangGraph checkpointer 保存某个 `thread_id` 的图执行状态，用于 Tutor 路径的 `interrupt` / resume。
- Store-compatible memory 保存跨会话的学习画像、掌握度、薄弱点、偏好和学习事件。
- 未配置 PostgreSQL 时，学习记忆使用进程内 Store 与本地 JSON 快照；这只是单机 fallback。

独立 `/agent/autonomous` 端点的暂停会话仍保存在进程内，服务重启后不能恢复。只有 Tutor interrupt/resume 路径接入 SQLite checkpointer。

### 4. 失败必须有边界

- Agent 循环使用明确的最大轮数和 `finalize` 停止工具。
- 工具执行经过 allowlist、timeout、retry 和 audit。
- 文档上传限制扩展名与体积；PDF/OCR 设置页数和解析边界。
- prompt injection 与输出泄漏默认执行本地规则检测，可选启用额外模型检查。
- live MCP、云端 tracing 和重型 reranker 在示例配置中默认关闭。

这些措施是工程边界，不代表沙箱、完整权限系统或生产级安全保证。详情见 [SECURITY.md](SECURITY.md)。

## 技术架构

```text
React + Vite
      |
      v
FastAPI routers
      |
      +-- predefined LangGraph workflows
      |      quiz / critic / reviser / grader / planner
      |
      +-- bounded tool-use loops
      |      autonomous agent / tutor assistant
      |
      +-- adaptive learning workflow
             grade observation -> next teaching action

Shared services
  retrieval: Chroma + BM25 + RRF + optional rewrite/reranker
  tools:     ToolRegistry + timeout/retry/audit
  memory:    InMemoryStore or PostgreSQL Store
  state:     SQLite LangGraph checkpointer
  models:    OpenAI-compatible chat / structured output / embeddings
```

## 快速开始

### 1. 确定性 Agent Demo（不调用模型或外网）

要求 Python 3.11。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
python scripts/demo_react_tutor_agent.py
```

Demo 注入 scripted model policy 与 fake tool handlers，但走真实的 `assistant_agent -> tool_loop -> ToolRegistry` 控制链。它用于复现行动—观察—继续行动的轨迹，不代表真实模型任务成功率。

### 2. Docker Compose

```bash
cp .env.example .env
# 在本机编辑 .env 中的 provider 地址、模型名和 key
docker compose up --build
```

启动后：

- Web：`http://localhost:4001`
- API / OpenAPI：`http://localhost:8001/docs`
- PostgreSQL：`localhost:5434`

`.env.example` 中的数据库口令只是本地开发占位值；在共享环境运行前必须替换。

### 3. 本地开发

```bash
cp .env.example .env
python -m uvicorn main:app --reload --port 8001
```

另一个终端：

```bash
cd frontend
npm ci
npm run dev
```

准备好有效 provider 配置后，可以上传仓库内的合成示例材料：

```bash
curl -F "file=@examples/sample_document.md" http://localhost:8001/documents/upload
```

支持 PDF、DOCX、TXT、Markdown 和常见图片。扫描型 PDF 与图片 OCR 依赖 Poppler/Tesseract；Docker 镜像已安装英文和简体中文运行依赖。

## 测试与验证

```bash
python -m pytest -q

cd frontend
npm run lint
npm run build
```

GitHub Actions 会运行后端确定性测试、scripted Agent demo、前端 lint/build 和 Docker Compose 配置校验。测试默认使用不可访问的本地占位 provider，防止误调用收费接口。

公开版本刻意移除了本地向量库、上传语料、模型评测日志、原始轨迹和来源不清晰的面试题数据。当前能够公开验证的是控制流与工程契约，不是模型质量指标；请勿把单元测试通过数写成 Agent 成功率。

## 主要目录

```text
study-loop/
├── agents/                 LangGraph 节点、Tutor graph 与 assistant tool loop
├── routers/                FastAPI 端点与 HTTP 会话边界
├── services/               RAG、工具、记忆、checkpoint、guardrail、trace
├── models/                 Pydantic 请求、响应与状态模型
├── frontend/               React/Vite 前端
├── scripts/                不依赖真实模型的确定性 Agent demo
├── tests/                  mock/unit/control-flow tests
├── docs/                   架构边界与公开评测政策
├── examples/               可再分发的合成示例材料
└── docker-compose.yml      PostgreSQL + backend + frontend
```

## 已知限制

- 真实模型表现取决于 provider、模型、提示词、知识库与数据质量，当前无公开端到端质量指标。
- standalone autonomous session 是进程内状态，不支持服务重启恢复。
- Tutor supervisor 默认关闭，并仍包含 experimental worker，不作为已完成主能力宣传。
- 工具 schema 不等于完整服务端参数验证，工具 permission 元数据不等于鉴权。
- reranker 默认模型体积较大，首次启用会下载模型；轻量首次运行建议保持关闭。
- 这不是任意代码执行 Agent，也没有提供 OS shell、文件写入或浏览器控制工具。

## License

Source available for portfolio review. Copyright (c) 2026 tx3457. All rights reserved. See [LICENSE](LICENSE).
