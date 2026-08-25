# StudyLoop

[![CI](https://github.com/tx3457/study-loop/actions/workflows/ci.yml/badge.svg)](https://github.com/tx3457/study-loop/actions/workflows/ci.yml)

**Document-grounded adaptive tutoring with bounded tool-use agents.**

StudyLoop 是一个基于个人学习材料的 AI 自适应学习系统：从文档检索与引用、学习路径、Quiz、AI 批改，到可中断和恢复的工具调用 Agent，形成完整的学习闭环。

## 产品演示

<p align="center">
  <a href="docs/assets/autonomous-grounded.png">
    <img src="docs/assets/autonomous-grounded.png" width="100%" alt="StudyLoop 自主学习 Agent 展示检索引用、最终回答和工具执行过程">
  </a>
</p>

<p align="center"><sub>选择学习材料后，Agent 调用受限业务工具，并把本轮检索证据与执行过程返回到界面。</sub></p>

<p align="center">
  <a href="docs/assets/hitl-resume.png"><img src="docs/assets/hitl-resume.png" width="49%" alt="StudyLoop 展示 Agent 中断、用户补充和恢复后的带引用回答"></a>
  <a href="docs/assets/learning-loop.png"><img src="docs/assets/learning-loop.png" width="49%" alt="StudyLoop 根据学习进度展示分阶段学习路径和解锁状态"></a>
</p>

<p align="center"><sub>左：HITL 中断、补充与恢复的完整流程；右：学习路径按完成状态依次解锁。截图使用仓库内的反向传播示例材料展示同一条学习流程。</sub></p>

## 核心亮点

- **文档驱动学习**：解析 PDF、DOCX、Markdown、文本和图片，通过 Chroma 与 BM25 检索材料；选择文档后可要求回答携带服务端登记的片段 ID。
- **有界工具 Agent**：模型只能选择注册表中的业务工具，工具结果会作为 observation 回到下一轮决策；执行最多 8 轮，并支持 HITL 暂停、补充和恢复。
- **自适应学习闭环**：学习路径、Quiz、AI 批改、错题与学习画像共同决定下一阶段或下一轮辅导动作。
- **可恢复的 Web 工作流**：Learning Path、Quiz、Adaptive 和 Autonomous 使用幂等请求、服务端会话与持久化快照处理刷新、断网、响应丢失和后端重启。
- **工程化交付**：提供 Docker Compose、PostgreSQL/SQLite 状态存储、Pytest、Playwright E2E、前端构建检查和容器 smoke test。

## 系统架构

<p align="center">
  <a href="docs/assets/architecture.svg">
    <img src="docs/assets/architecture.svg" width="100%" alt="StudyLoop 系统架构：React Web、FastAPI、文档流水线、有界工具 Agent、自适应学习闭环、检索数据面与持久化状态">
  </a>
</p>

主学习流程是：

1. 上传材料并完成解析、切分和检索建库。
2. 从材料生成学习路径或 Quiz，完成作答与 AI 批改。
3. 将掌握度、薄弱点和学习事件写入学习画像。
4. Adaptive workflow 或 Autonomous Agent 根据当前状态选择下一步；需要更多信息时通过 HITL 暂停。

Autonomous 是模型选择工具的有界 Agent；Adaptive 是模型选择结构化教学动作、由应用执行固定分支的 agentic workflow。详细执行边界见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

## Quick Start

推荐使用 Docker Compose 启动完整本地环境：

```bash
cp .env.example .env
# 配置模型服务，并设置非空 POSTGRES_PASSWORD
docker compose up --build -d --wait
```

启动后访问：

- Web：<http://localhost:4001>
- API 文档：<http://localhost:8001/docs>

默认拓扑只绑定本机回环地址，不发布 PostgreSQL 端口，并固定使用一个 backend worker。Provider 配置、健康检查和可选能力见 [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md)；本地开发与测试命令见 [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md)。

示例学习材料位于 [`examples/sample_document.md`](examples/sample_document.md)。

## 技术栈

| Layer | Technologies |
| --- | --- |
| Web | React 19, React Router, Vite |
| API | FastAPI, Pydantic, Uvicorn |
| Agent and workflows | LangGraph, bounded tool loop, HITL checkpoint/resume |
| Retrieval | Chroma, BM25, optional query rewriting / HyDE / reranking |
| State | PostgreSQL, SQLite, learner-memory snapshots |
| Delivery and quality | Docker Compose, GitHub Actions, Pytest, Playwright |

## 产品边界

Web 产品主线包括文档管理、学习路径、答题练习、自主 Agent、自适应辅导和学习报告。`/agent/tutor/*` supervisor 图是默认关闭的 API-only Lab，且纯讲解 worker 尚未完成，因此不作为当前产品或简历功能。

引用校验保证片段 ID 来自本轮、指定文档范围内的检索结果，但不等同于对回答中每一项事实完成语义核验。文档删除是“仅删除材料”，不会级联清除已经形成的学习历史和会话工件。完整边界见架构与安全文档。

## 文档

- [架构与执行边界](docs/ARCHITECTURE.md)
- [配置、Provider 与运行说明](docs/CONFIGURATION.md)
- [本地开发与验证](docs/DEVELOPMENT.md)
- [安全与数据处理](SECURITY.md)
- [产品与界面设计](DESIGN.md)

## License

Copyright (c) 2026 tx3457. All rights reserved. See [LICENSE](LICENSE).
