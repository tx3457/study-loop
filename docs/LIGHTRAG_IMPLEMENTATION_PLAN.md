# LightRAG 多知识库与联网问答 V1

状态：V1 已实施并完成验收；最终结果和边界见 LIGHTRAG_STATUS.md。

## 目标和边界

- 单用户自托管，每库几十到几百份资料；医学、机器学习等知识库分别隔离。
- 多文档自动构图、实体重命名/合并、实体与关系删除、跨文档自主问答。
- 联网默认关闭，网页只能由用户主动将当时查看的快照收录入库。
- 保留单文档学习路径、Quiz、自适应辅导和既有学习历史；新资料不进入旧选择器。
- 不修改既有冻结评测，不预设 GraphRAG 的效果优于当前混合检索。

## 架构决策

采用独立内部知识库服务，以 LightRAG SDK 1.5.7 为引擎，一个进程/事件循环管理按
可信 owner 和知识库 UUID 分配的不可变 workspace。服务端派生 workspace，禁止
客户端指定以及环境变量覆盖各知识库范围。实例只在空闲时淘汰。

知识库服务持有资料、版本、纠错规则、后台任务、网页快照和索引。原始资料保存在
专用材料卷；元数据和 LightRAG 四类存储放在独立 PostgreSQL 16 + pgvector。
使用 PGKVStorage、PGDocStatusStorage、PGTableGraphStorage 和 PGVectorStorage；
无需 AGE。新数据库采用 pgvector/pgvector:0.8.6-pg16，实施时固定镜像摘要。
现有 PostgreSQL 15、Chroma 和后端依赖环境不变。新服务仅由现有后端通过内部
服务令牌访问，不暴露公共端口。功能开关默认关闭，故障不阻断旧功能。

取舍：官方 REST 每库一个实例更简单但增加容器配置；直接嵌入主后端耦合依赖和
故障。独立 SDK 适配服务增加维护成本，提供统一身份、来源、纠错和恢复边界。

## 资料与图谱

- 知识库、资料和不可变资料版本采用 UUID；文件名仅展示。
- 每份资料属于一个知识库；同库完全相同内容复用，同名不同内容可并存。
- 显式替换创建新版本；保存原始内容、解析文本、哈希和可验证的来源位置。
- 复用已有文件类型、20 MB 上传限制及解析失败检查。
- 旧文档显式复制解析快照，保留来源链接；后续不跨系统自动同步或级联删除。
- 标准 ainsert 使用资料版本 UUID 和不含敏感信息的来源 token；不用 deprecated
  ainsert_custom_chunks。检索 chunk ID 经 text_chunks.full_doc_id 映射回资料版本。
  未知来源拒绝引用；没有可验证页码时不生成页码。
- 图谱提供搜索、局部浏览和来源详情；最多返回 200 节点/400 条边并标明截断。
  前端 React/SVG 提供缩放、平移和可访问列表，不新增图形框架。

纠错使用应用持有的 canonical entity UUID 和有序事件：重命名保留别名，名称冲突
要求显式合并；合并形成无环别名映射；实体和实际 SDK 边身份的删除形成抑制规则。
重复执行幂等，已不存在的抑制对象视为满足规则。失去原文支持的实体对应规则休眠，
重新出现时重新应用；不因此创建无证据节点。歧义不猜测，转为待处理状态。
每次入库、删除或重建后重放纠错并更新向量，成功后才可查询。用户修改与原文分开
展示；图谱摘要不是独立的引用证据。

## 写入、恢复和接口

持久任务队列首版仅一个全局写入任务，数据库 advisory lock 确认单写入者。每库
读写互斥：写入前等待检索读者退出，先持久化 mutation epoch 并标记 updating。
修改索引、重放纠错、校验来源和刷新存储全部成功后增加 revision 并置 ready。
失败/退出标记 dirty，暂停查询和后续写入；显式重试或根据资料及纠错日志重建。
启动发现未完成任务并保留恢复状态，不无限自动调用模型。

问答固定 epoch/revision，检索、证据使用、最终发布和恢复时重新验证。失败写入
即使未增加 revision，也不能让旧问答继续发布。已结束回答保留历史来源说明，
删除资料后标注来源状态，不用于新的检索或暂停会话恢复。

公共 API 由主后端认证和代理：

| 路径 | 行为 |
| --- | --- |
| /knowledge-bases、/knowledge-bases/{id} | 创建、列表、详情、修改、删除 |
| /knowledge-bases/{id}/documents | 上传、列表、显式替换/删除/导入旧材料 |
| /knowledge-bases/{id}/graph、/corrections | 有界图查询、来源和纠错 |
| /knowledge-bases/{id}/web-import | snapshot_id 指定快照收录 |
| /knowledge-jobs/{id} | 状态和显式重试 |

耗时操作返回 202/job_id；写操作使用幂等键和 expected revision，冲突返回 409。

## Autonomous 和联网

- 请求增加 knowledge_base_id（与 document_id 互斥）、web_enabled=false。
- KB 模式独立工具注册表只开放知识库检索、可选联网及 ask/finalize；不开放出题、
  画像写入和任意文档选择。共享执行循环保留原注册表作为默认行为。
- aquery_data 只提供证据，StudyLoop 负责生成与强制引用校验。仅本轮取得的授权
  KB 片段和开启联网后的网页快照可引用；伪造或缺少引用时弃答。网页独立来源及
  仅网页回答明确标注；旧文档模式的严格引用规则不变。
- 保留旧 citations；新增 source_citations 判别联合，包含源 ID、版本、位置或
  URL/抓取时间/内容哈希，以及片段。
- 服务端快照 v4、浏览器恢复 v3；旧数据迁移为 KB 空/联网关闭，保留已有契约哈希。
  新会话记录有效工具契约、KB epoch/revision、资料版本和快照引用；变化要求重启。
- DDG MCP 搜索通过有类型的适配器调用；KB 模式不暴露原始 MCP 抓取旁路。HTTP
  正文抓取检查并固定公网 DNS 地址，逐跳验证，阻断内网/保留地址/凭据，限制
  10 秒、2 MiB、3 次跳转。只提取文本，不执行脚本，不直接渲染 HTML。
- 网页始终是非可信资料，不能改变工具权限；不自动把整份私有文档发送给搜索方。
- 网页快照绑定 owner/session，保存 URL、标题、正文、哈希、时间和过期状态。
  正文独立持久化，会话仅存 ID/哈希/有界摘录；至少保留会话有效期和七天。
  用户显式收录复制同一正文，过期 410，不静默重新抓取。

## 实施顺序

1. 真实 LightRAG/PG + 确定性模型替身合同验证：隔离、入库、来源、纠错和恢复。
   关键合同失败时记录 STATUS 和证据，调整计划，不扩大到完整 UI。
2. 知识库服务：版本、引擎适配、队列、纠错、图查询、健康、备份恢复。
3. 网关/Autonomous/引用与恢复协议，知识库前端及网页搜索/主动收录。
4. 单元、真实集成、浏览器、原流程回归和恢复演练，最后独立审查。

可在合同通过后按后端与前端独立职责使用原生 executor 子代理；最终 verifier
检查交付。当前 App 不调用需要 tmux 的 OMX team。

## 验收

详见 LIGHTRAG_TEST_SPEC.md。通过功能/隔离/引用/纠错/恢复验证后再启用。真实
语料评测固定题目、资料、模型，对比现有混合检索，单独报告质量、成本、延迟。

## 上游依据

- https://github.com/HKUDS/LightRAG/blob/v1.5.7/docs/ProgramingWithCore.md
- https://github.com/HKUDS/LightRAG/blob/v1.5.7/lightrag/utils_pipeline.py
- https://github.com/HKUDS/LightRAG/blob/v1.5.7/lightrag/pipeline.py
- https://github.com/pgvector/pgvector#docker

架构审查及独立 critic 在本会话中已依次批准，用户随后明确要求实施。
