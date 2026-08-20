# StudyLoop

[![CI](https://github.com/tx3457/study-loop/actions/workflows/ci.yml/badge.svg)](https://github.com/tx3457/study-loop/actions/workflows/ci.yml)

StudyLoop 是一个基于个人学习材料的 AI 自适应学习系统。它支持文档检索、学习路径生成、练习与批改、学习画像，以及带工具调用的学习 Agent。

## 主要功能

- 上传并解析 PDF、Word、Markdown、文本和图片
- 使用 Chroma 与 BM25 检索学习材料
- 生成学习路径、练习题和学习报告
- 自动批改答案并记录错题与掌握度
- 提供有轮次上限、工具白名单、人工确认和可选检索引用约束的 Autonomous Agent
- 保存学习记忆和工作流 checkpoint

### 功能边界

默认产品主线是 Web 中的文档、学习路径、Quiz、Adaptive、Dashboard 和
Autonomous。`/agent/tutor/*` supervisor 图属于 API-only Lab：默认不注册路由、
没有前端页面，而且纯讲解 tutor worker 尚未完成。只有显式设置
`MAS_SUPERVISOR_ENABLED=true` 并重启后端才会暴露这些实验接口；它不作为当前
简历的已完成功能。其他未进入 Web 主线的 `/agent/run`、`/agent/stream`、
`/eval/ab` 与 `/audit` 也应视为开发者接口，而不是独立产品入口。

Autonomous 选择文档后，Web 默认要求最终回复携带本轮检索得到的片段 ID；
严格模式会把检索限制在所选文档，并要求全部引用 ID 都属于本轮检索；引用缺失
或任一 ID 无效时都会安全拒答。这里校验的是引用来源与本轮检索的一致性，
不代表已经逐条验证回复中的事实是否被片段支持；
评测边界见 [`docs/EVALUATION.md`](docs/EVALUATION.md)。

“文档管理”中的删除采用 **仅删除材料** 语义：它会移除 StudyLoop 内的原文检索
索引，不会删除电脑上的原文件，也不会清除已经形成的学习历史、学习画像、错题、
测验或 Agent 会话工件；已保存会话中的题目或引用片段仍可能继续显示或被原会话使用，
因此不等同于隐私数据彻底清除。为避免旧学习记录错误关联到
同名新内容，已删除的文件名会作为墓碑保留，不能直接同名重传；需要重新上传时请先
重命名文件。完整级联清除与同名文档版本化属于后续能力。

## 环境要求

- Python 3.11
- Node.js 22.12+（22.x）
- Docker Compose v2（可选）
- Poppler 与 Tesseract（本地解析扫描 PDF 或图片时需要）

## 快速开始

### Docker Compose

```bash
cp .env.example .env
# 编辑 .env，配置模型服务并设置非空 POSTGRES_PASSWORD
docker compose up --build -d --wait
```

启动后访问：

- Web：<http://localhost:4001>
- API 文档：<http://localhost:8001/docs>

默认 Compose 仅将 Web 与 API 绑定到本机回环地址，PostgreSQL 不发布宿主端口。
启动命令会等待 PostgreSQL 与后端 `/health/ready` 的存储检查通过；该检查不访问模型服务。
后端以固定的非 root 用户运行；启动前的一次性初始化容器会修复旧 Chroma 卷的
目录属主，因此从早期 root 镜像升级时无需删除已有索引卷。
如需共享访问，应先在反向代理层增加认证与 TLS，不要直接将后端端口暴露到公网。

停止服务：

```bash
docker compose down
```

### 本地开发

启动后端：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env
# 编辑 .env，配置模型服务
python -m uvicorn main:app --reload --port 8001 --workers 1 --no-access-log
```

如需启用本地 Cross-Encoder 精排，改用
`pip install -r requirements-reranker.txt -r requirements-dev.txt`，再设置
`RERANKER_ENABLED=true`。默认安装和 Docker 镜像不包含大型模型运行时。

启动前端：

```bash
cd frontend
npm ci
npm run dev
```

前端默认运行在 <http://localhost:5173>。

## 模型配置

完整功能需要 OpenAI-compatible 的 Chat、Structured Output 和 Embeddings API。主要环境变量如下：

| 用途 | 环境变量 |
| --- | --- |
| Chat / Tool Calling | `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL` |
| Structured Output | `STRUCTURED_API_KEY`、`STRUCTURED_BASE_URL`、`STRUCTURED_MODEL` |
| Embeddings | `EMBEDDING_API_KEY`、`EMBEDDING_BASE_URL`、`LLM_EMBEDDING_MODEL` |

Web 前端会为 Autonomous 请求、Learning Path 创建、Quiz 创建与单题答案、Adaptive 创建与单轮提交生成
`Idempotency-Key`；同一内容重试时复用原 key，修改答案后生成新 key。Quiz 的题号以及
Adaptive 的轮次与状态版本也会随请求提交，服务端会拒绝过期覆盖。Web Quiz 与
Adaptive 都把完整私有题目、阶段性批改、画像写入标记和规范进度保存在服务端；浏览器
只保存恢复意图、最近一次安全快照和尚未确认的完整请求。因此刷新、切页、响应丢失或
后端重启后可以继续，未作答题目的答案、解析和来源不会返回浏览器。使用
Learning Path 生成结果也会保存为不可变资源，刷新时由资源 ID 恢复；新标签页没有浏览器
恢复指针时会读取默认用户最近创建的路径。阶段进度以服务端为准：每次只开放首个未完成
阶段，只有绑定该阶段的 Quiz 完成规范 AI 批改（或报告所复用的同一份规范批改）后才会
幂等解锁下一阶段；仅答完题目、刷新或放弃练习都不会提前推进。删除原材料时该路径
作为学习记录保留，但只能查看，不能再从已删除材料发起阶段练习。使用
`DATABASE_URL` 时，重试 receipt、Learning Path、Web Quiz、Adaptive 与人工确认暂停快照保存在
PostgreSQL；本地无数据库时默认使用 `IDEMPOTENCY_DB_PATH` 指定的 SQLite 文件，也可用
功能专属路径覆盖。Autonomous 与 Adaptive 默认保留 1 小时，Web Quiz 默认保留 24 小时。
Autonomous 在请求发出前保存完整待对账请求与 key：刷新、断网或响应丢失后会原样重放，
不会在结果不确定时解锁编辑并生成新 key；人工确认暂停可从页面执行服务端取消，正在执行
或已发生副作用的操作不会被假装取消。清洁的崩溃租约可安全接管；跨过业务工具执行边界后，
只有能由同一请求绑定的持久会话终态证明结果时才会自动对账，否则保持 fail-closed，需要
重新开始或人工调查。相关容量和 TTL 配置见
[`.env.example`](.env.example)。当前受支持且默认的 Docker/Compose 拓扑是一个 backend
worker 和一个 replica，请勿扩容：向量库使用内嵌的 Chroma `PersistentClient`，不能由多个
进程共同写同一目录。`DATABASE_URL` 让 receipt、会话和学习者记忆等状态存储具备跨进程协调能力，
但它本身并不会把内嵌 Chroma 变成多进程安全；设置 `WEB_CONCURRENCY>1` 也不会绕过该边界。
若未来要做多 worker/replica，必须先改成独立 Chroma 服务与 `HttpClient`，并同时配置
PostgreSQL；这只是必要条件，还要把进程内 BM25 缓存失效和文档写入协调改造成跨进程协议。
`MEMORY_STORE_SETUP_LOCK_TIMEOUT_SECONDS`、`QUIZ_SESSION_PG_LOCK_TIMEOUT_MS` 等参数控制
PostgreSQL 状态组件内部的初始化、连接、语句、并发锁与取消清理；它们并不表示完整应用已经支持
多 worker/replica。未配置 `DATABASE_URL` 时，学习者
记忆还会额外使用单进程内存和本地 JSON 快照兜底。

配置 `DATABASE_URL` 时，learner-memory 冷启动会先在独立的有界连接中完成并校验 Store schema，
再打开长期运行连接；运行期同步 Store I/O 会先在事件循环侧排队，同一时刻只允许一个后台线程任务
进入共享执行器，因此慢查询不会直接冻结 API 事件循环或用等待线程占满共享执行器。连接、锁、语句、排队和取消收尾预算均可在
[`.env.example`](.env.example) 中配置。这不是连接池或热恢复机制：PostgreSQL 进程重启后，仍应按
下方 readiness 说明重启 backend，以重新建立长期连接。

Structured Output 和 Embeddings 的专用 key 与地址必须成对配置；两者同时留空时才会整组回退到 `LLM_*`。完整配置见 [`.env.example`](.env.example)。

`.env` 是本项目的本地配置，不会从其他项目自动复制，也不会提交到 Git。配置真实 Provider 时：

1. 将 `.env.example` 复制为 `.env`。
2. 把 `LLM_*` 替换为真实的 OpenAI-compatible Chat API 地址、密钥和模型名。
3. 如果 Chat Provider 不支持 JSON Schema Structured Output，单独配置 `STRUCTURED_*`。
4. 如果 Chat Provider 不提供 Embeddings，单独配置 `EMBEDDING_*` 和 `LLM_EMBEDDING_MODEL`。

后端启动后可检查配置：

```bash
curl -i http://localhost:8001/health/live
curl -i http://localhost:8001/health/ready
curl -i http://localhost:8001/health/providers
```

`/health/live` 只检查进程存活，不访问存储或模型服务。`/health/ready` 检查当前进程使用的
Chroma collection 元数据是否可读；配置 `DATABASE_URL` 时还会执行有界的 PostgreSQL
`SELECT 1`，并读取应用实际持有的 learner-memory Store；未配置数据库时则校验 Web 主流程实际选择的
SQLite 文件以及本机 learner-memory 快照目录可读写。必需存储不可用时返回 503 和固定诊断码，
但不会返回数据库地址、Chroma 路径或原始异常。该检查证明存储连接、collection catalog 与本地状态
文件当前可访问，不等于完整上传、出题或写入流程的端到端测试。PostgreSQL 进程重启后应同时重启
backend，让其持有的 learner-memory 连接重新建立。`/health/providers` 调用 Provider 的 `models.list`，
不会发起 Chat、Structured Output 或 Embedding 请求；结果默认缓存 30 秒。全部模型在目录中可见时
返回 200，否则返回 503 和稳定的诊断码。目录可达只代表凭据、地址和模型可见性正常，不代表
Structured Output、Tool Calling 或 Embedding 能力已经实际验证。

通过 Web 容器探测时使用 `http://localhost:4001/api/health/ready`；裸的 Web 路径
`/health/ready` 属于前端路由，不是 API readiness 端点。

统一的 `llm_chat`、`llm_parse` 和 Embedding 重试链路默认各有 60 秒端到端预算（含退避等待），可通过 `PROVIDER_REQUEST_DEADLINE_SECONDS` 调整。HTTP 边界会返回稳定错误码：限流为 `provider_rate_limited`（429）、超时为 `provider_timeout`（504）、其他上游故障为 `provider_unavailable`（503），未配置则为 `provider_not_configured`（503）；前端不需要解析 Provider 原始异常。
每个 HTTP 响应都会带服务端生成或严格校验后的 `X-Request-ID`；浏览器的通用 API 与
流式错误提示会显示该请求编号，用于把用户看到的故障与服务端安全日志关联。未预期异常、流式
执行中断、评测失败、实验 Tutor 失败与 MCP 远端错误只公开固定错误码和文案，不会把
Provider 响应正文、持久化路径或原始异常文本返回给客户端。

首次接入真实 Provider 时，再显式运行一次能力检查：

```bash
python scripts/check_provider_capabilities.py
```

该命令会发送少量真实请求，验证普通 Chat、JSON Mode、生产所用的自动工具选择行为、Structured Output 和 Embedding，并可能产生少量费用。输出不会包含密钥、Provider 地址、响应正文或原始异常；全部通过时退出码为 0。它不会作为公开 HTTP 接口或默认 CI 步骤运行。

## 使用方法

1. 在“文档管理”上传学习材料。
2. 在“学习路径”中选择文档、生成计划，并从当前开放阶段开始练习。
3. 在“答题练习”中完成题目和 AI 批改，再返回路径继续下一阶段。
4. 在“学习报告”中查看掌握度、薄弱点和历史记录。
5. 按需使用 Autonomous Agent 或 Adaptive Learning。

示例材料位于 [`examples/sample_document.md`](examples/sample_document.md)。

支持 `.pdf`、`.docx`、`.txt`、`.md` 及常见图片格式，默认上传上限为 20 MB。

## 测试

```bash
# 后端高信号静态检查（语法、未定义名称、未使用导入/变量）
python -m ruff check .

# 后端确定性测试与核心运行时代码分支覆盖率
python -m pytest -q --cov --cov-config=pyproject.toml --cov-report=term-missing:skip-covered

# 使用临时 PostgreSQL 时会额外执行 receipt、Web Quiz、暂停会话与 memory 冷启动测试
TEST_DATABASE_URL=postgresql://user:password@127.0.0.1:5432/studyloop_test \
  python -m pytest -q \
    tests/test_idempotency_postgres.py \
    tests/test_quiz_sessions_postgres.py \
    tests/test_adaptive_sessions_postgres.py \
    tests/test_autonomous_sessions_postgres.py \
    tests/test_memory_postgres.py

# 确定性 tool-loop 演示与 BM25 评测校验（不代表实验 Tutor Web 功能）
python scripts/demo_react_tutor_agent.py
python evaluation/scifact_bm25/verify.py

# 前端检查
cd frontend
npm run lint
npm run build

# 浏览器 E2E（首次运行先执行 npx playwright install chromium）
npm run test:e2e
```

CI 将核心 Python 运行时代码的分支覆盖率回归下限设为 75%。该数字不统计
`tests/`、`scripts/`、`evaluation/`、浏览器 E2E 或容器 smoke，也不代表真实模型的
回答正确率、RAG 事实性或 Agent 成功率。未配置 `TEST_DATABASE_URL` 时，本地会跳过
PostgreSQL 专项用例；该变量只能指向可清空的临时测试库，不能使用开发或生产数据库。

浏览器 E2E 使用本地 Vite 和 Mock API，不会调用真实模型服务。

## 项目结构

```text
study-loop/
├── agents/          LangGraph Agent 与工作流
├── routers/         FastAPI 路由
├── services/        检索、模型、记忆和工具服务
├── models/          Pydantic 数据模型
├── frontend/        React 前端
├── tests/           后端测试
├── evaluation/      检索评测
├── examples/        示例材料
└── docs/            项目文档
```

## 文档

- [架构说明](docs/ARCHITECTURE.md)
- [评测说明](docs/EVALUATION.md)
- [安全说明](SECURITY.md)

## License

Copyright (c) 2026 tx3457. All rights reserved. See [LICENSE](LICENSE).
