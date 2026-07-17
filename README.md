# StudyLoop

[![CI](https://github.com/tx3457/study-loop/actions/workflows/ci.yml/badge.svg)](https://github.com/tx3457/study-loop/actions/workflows/ci.yml)

StudyLoop 是一个基于个人学习材料的 AI 自适应学习系统。它支持文档检索、学习路径生成、练习与批改、学习画像，以及带工具调用的学习 Agent。

## 主要功能

- 上传并解析 PDF、Word、Markdown、文本和图片
- 使用 Chroma 与 BM25 检索学习材料
- 生成学习路径、练习题和学习报告
- 自动批改答案并记录错题与掌握度
- 提供 Autonomous Agent 和带人工确认的 Tutor Agent
- 保存学习记忆和工作流 checkpoint

## 环境要求

- Python 3.11
- Node.js 20.19+ 或 22.12+
- Docker Compose v2（可选）
- Poppler 与 Tesseract（本地解析扫描 PDF 或图片时需要）

## 快速开始

### Docker Compose

```bash
cp .env.example .env
# 编辑 .env，配置模型服务
docker compose up --build -d
```

启动后访问：

- Web：<http://localhost:4001>
- API 文档：<http://localhost:8001/docs>

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
python -m uvicorn main:app --reload --port 8001
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

Autonomous Agent 会为同一次前端重试复用 `Idempotency-Key`。使用
`DATABASE_URL` 时 receipt 保存在 PostgreSQL；本地无数据库时保存在
`IDEMPOTENCY_DB_PATH` 指定的 SQLite 文件。

Structured Output 和 Embeddings 的专用 key 与地址必须成对配置；两者同时留空时才会整组回退到 `LLM_*`。完整配置见 [`.env.example`](.env.example)。

`.env` 是本项目的本地配置，不会从其他项目自动复制，也不会提交到 Git。配置真实 Provider 时：

1. 将 `.env.example` 复制为 `.env`。
2. 把 `LLM_*` 替换为真实的 OpenAI-compatible Chat API 地址、密钥和模型名。
3. 如果 Chat Provider 不支持 JSON Schema Structured Output，单独配置 `STRUCTURED_*`。
4. 如果 Chat Provider 不提供 Embeddings，单独配置 `EMBEDDING_*` 和 `LLM_EMBEDDING_MODEL`。

后端启动后可检查配置：

```bash
curl -i http://localhost:8001/health/live
curl -i http://localhost:8001/health/providers
```

`/health/live` 只检查进程存活，不访问外部服务。`/health/providers` 调用 Provider 的 `models.list`，不会发起 Chat、Structured Output 或 Embedding 请求；结果默认缓存 30 秒。全部模型在目录中可见时返回 200，否则返回 503 和稳定的诊断码。目录可达只代表凭据、地址和模型可见性正常，不代表 Structured Output、Tool Calling 或 Embedding 能力已经实际验证。

首次接入真实 Provider 时，再显式运行一次能力检查：

```bash
python scripts/check_provider_capabilities.py
```

该命令会发送少量真实请求，验证普通 Chat、JSON Mode、生产所用的自动工具选择行为、Structured Output 和 Embedding，并可能产生少量费用。输出不会包含密钥、Provider 地址、响应正文或原始异常；全部通过时退出码为 0。它不会作为公开 HTTP 接口或默认 CI 步骤运行。

## 使用方法

1. 在“文档管理”上传学习材料。
2. 在“学习路径”中选择文档并生成计划。
3. 在“答题练习”中完成题目并查看批改结果。
4. 在“学习报告”中查看掌握度、薄弱点和历史记录。
5. 按需使用 Autonomous Agent 或 Adaptive Learning。

示例材料位于 [`examples/sample_document.md`](examples/sample_document.md)。

支持 `.pdf`、`.docx`、`.txt`、`.md` 及常见图片格式，默认上传上限为 20 MB。

## 测试

```bash
# 后端测试
python -m pytest -q

# 使用临时 PostgreSQL 时会额外执行持久幂等跨连接测试
TEST_DATABASE_URL=postgresql://user:password@127.0.0.1:5432/studyloop_test \
  python -m pytest -q tests/test_idempotency_postgres.py

# Agent 演示与 BM25 评测校验
python scripts/demo_react_tutor_agent.py
python evaluation/scifact_bm25/verify.py

# 前端检查
cd frontend
npm run lint
npm run build

# 浏览器 E2E（首次运行先执行 npx playwright install chromium）
npm run test:e2e
```

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
