# LightRAG 知识库：启用、恢复与验证

V1 面向个人自托管：每个知识库独立组织多份文档，自动构图并支持人工纠错；自主
Agent 可以检索整个知识库，按需联网，并由用户主动收录网页快照。旧学习路径、Quiz
和自适应辅导仍使用单文档。功能默认关闭，无需手动迁移已有 `.env` 或数据卷。
升级时会话表自动新增归属字段；只从有效旧快照回填，无法确认归属的损坏快照需要重新开始。

## 启用

继续使用现有 `.env` 中的聊天与 Embedding 服务。Embedding 的 key/base URL 必须
成对配置，留空时一同继承聊天服务。必须设置实际向量维度，例如当前项目使用的
`BAAI/bge-large-zh-v1.5` 为 1024 维。

新建被 Git 忽略的 `.env.knowledge`，不要覆盖已有文件：

```bash
python - <<'PY'
import os
import secrets
body = (
    'KNOWLEDGE_BASES_ENABLED=true\n'
    f'KNOWLEDGE_SERVICE_TOKEN={secrets.token_urlsafe(32)}\n'
    f'KNOWLEDGE_POSTGRES_PASSWORD={secrets.token_urlsafe(32)}\n'
    'KNOWLEDGE_EMBEDDING_DIM=1024\n'
    'MCP_LIVE_ENABLED=false\n'
)
fd = os.open('.env.knowledge', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, 'w') as output:
    output.write(body)
PY

docker compose --env-file .env --env-file .env.knowledge \
  -f docker-compose.yml -f docker-compose.knowledge.yml up --build -d --wait
```

该扩展增加独立知识库服务和 PostgreSQL 16/pgvector 数据卷，不替换原 PostgreSQL 15
或 Chroma。新增服务不发布宿主机端口；前端仍通过原 API 访问。

启用联网时，将 `.env.knowledge` 的 `MCP_LIVE_ENABLED` 改为 `true` 并重新创建
backend。扩展镜像将 `duckduckgo-mcp-server==0.7.0` 安装在独立 Python 环境，避免
其 MCP/Starlette 版本改写主应用依赖。原生运行可设置 `MCP_DDG_COMMAND` 指向该
可执行文件，或使用已有 `uvx`。搜索连接失败会明确显示不可用，旧学习流程继续工作。

## 使用与边界

1. 打开“知识库”，创建一个领域空间并上传材料，等待后台任务结束。
   资料列表每页 50 份，可通过上一页、下一页管理后续资料。
2. “知识图谱”按名称搜索和选择实体，可重命名、合并、删除实体或关系，查看原文。
   用户纠错独立保存，后续导入和重建会重新应用。
3. “问知识库”进入自主 Agent。联网默认关闭；开启后搜索词固定来自用户输入，
   模型不能把检索到的私有正文拼进外部查询或任意抓取 URL。
4. 网页结果必须实际抓取成快照才能成为网页引用。点击“收录”保存同一快照，
   不会重新抓取成另一个版本；过期快照返回 410。

图谱最多显示 200 节点/400 关系，可搜索后查看局部子图。资料替换产生新版本；
旧材料导入是独立副本，不自动同步原文件。删除材料会撤销当前索引证据，但已结束
回答保留历史引用说明。引用 ID 校验不等同于逐项事实的语义验证。

联网抓取只接受公开 HTTP/HTTPS，逐跳校验并固定公网地址，不允许内网地址、脚本
执行或不受限正文。使用 Fake-IP DNS（例如把公网域名解析到 198.18.0.0/15）的
环境会被安全地拒绝；不要通过允许私网地址绕过它。

## 任务失败与配置变化

更新期间该知识库暂停检索，其他知识库和旧功能继续可用。失败后状态为待恢复，
可重试失败任务或点击“重建索引”；重建只使用当前资料和已保存纠错，不自动反复
消耗模型额度。模型、Embedding 或索引配置变化后，旧索引会标为待恢复，需显式
重建。暂停中的问答若遇到资料/版本变化，应重新开始。

有实体重命名或合并记录时，删除、替换资料会重建该库的派生索引，再重放纠错，
以完整撤销旧来源。同配置重建可复用抽取缓存，但会重新计算向量；模型、服务
地址或索引配置变化时会清理旧缓存。删除知识库会清空该 workspace 的索引及缓存；
历史业务记录和材料版本仍保留在存储及备份中，不等同于永久擦除。

V1 保持单服务进程、一个全局索引写入者；不要增加 Uvicorn workers。当前架构
通过暂停查询及重建补偿多阶段写入，没有宣称 LightRAG 与业务状态是一个原子事务。

## 离线备份与恢复

先停止 `knowledge-service`，再备份独立数据库及材料目录；不要删除 Docker volumes。
脚本记录 SHA-256、镜像摘要、索引配置哈希及模型信息，不保存 API 密钥。

```bash
python scripts/knowledge_backup.py backup \
  --container <knowledge-postgres容器名> --database studyloop_graph \
  --materials <挂载的知识库材料目录> --archive <仓库外的空备份目录> \
  --service-stopped --embedding-dimension 1024 \
  --embedding-model BAAI/bge-large-zh-v1.5 --llm-model <当前聊天模型>

python scripts/knowledge_backup.py restore \
  --container <目标知识库数据库容器名> --database <空目标数据库> \
  --materials <空目标材料目录> --archive <备份目录>
```

恢复拒绝非空数据库/目录，校验归档哈希和路径，暂存文件后用单事务恢复数据库，
成功才发布材料目录。数据库恢复失败会回滚并清理暂存文件；若最后目录发布失败，
数据库已完整恢复，脚本保留并打印验证过的暂存目录，应离线完成目录发布，不再
对同一个数据库重复恢复。重新配置原模型和维度后启动服务并验证来源与图谱。

## 验证入口

- `docs/LIGHTRAG_IMPLEMENTATION_PLAN.md`：已批准范围与决策。
- `docs/LIGHTRAG_TEST_SPEC.md`：验收标准。
- `docs/LIGHTRAG_STATUS.md`：执行过程、失败基线和当前完成情况。
- `docs/lightrag/verification/`：真实 HTTP、恢复演练和页面证据。
- `docs/lightrag/evaluation/`：冻结小语料实验，包含修复前失败记录。

主应用使用原来的 lint/pytest/Playwright 命令。知识库服务使用独立环境，避免根目录
测试 fixture 要求 Chroma：

```bash
TEST_GRAPH_SERVICE_DATABASE_URL=<可清空的临时PostgreSQL数据库> \
  python -m pytest -q --confcutdir=tests/graph_service tests/graph_service
TEST_LIGHTRAG_DATABASE_URL=<独立临时PostgreSQL数据库> \
  python scripts/verify_lightrag_contract.py
python scripts/check_compose_security.py
python scripts/check_knowledge_compose.py
python scripts/verify_knowledge_source_consistency.py \
  --container <知识库数据库容器名> --database studyloop_graph
```

来源一致性检查只读运行，需在知识库服务停止或空闲时执行。它会检查图节点、关系
和实体/关系向量中的片段引用是否仍属于当前有效资料；只应指向专用产品数据库，
不要混用包含未注册评测 workspace 的临时库。备份恢复后也应运行一次。

此次修复补全了知识库写操作的幂等目标标识。修复前生成的旧格式收据会返回 409；
应先核对原任务与资料状态，再决定是否创建新操作，避免自动换键重复提交。

上述图服务测试会清空其临时库内的业务表，不能指向用户数据库。真实模型评测必须
显式传入 `--allow-live-models`；默认仅验证冻结清单。替身测试、真实检索实验、词项
覆盖评分和逐事实语义正确性是不同的证据，不应混为一谈。
