# GraphRAG 的硅基流动模型配置

图谱抽取使用独立的 `KNOWLEDGE_LLM_*` 配置，问答可通过 `KNOWLEDGE_QA_MODEL`
选择同一连接上的另一模型，图谱向量使用 `KNOWLEDGE_EMBEDDING_*`。
旧文档检索的向量模型仍由 `LLM_EMBEDDING_MODEL` 控制，
避免更换图模型时用不同模型的向量查询已有 Chroma 集合。

## 调用协议

- Base URL：`https://api.siliconflow.cn/v1`；Bearer API key。
- 图谱抽取：`Pro/deepseek-ai/DeepSeek-V3.2`；知识库问答：
  `Qwen/Qwen3.8-27B`，均通过
  `/chat/completions`；`enable_thinking=false` 放入 OpenAI SDK 的 `extra_body`。
- 向量：`BAAI/bge-m3`，通过 `/embeddings`，输出1024维，模型输入上限8192tokens。
- 入库前用短文本和长文本验证模型可用性、维度，再验证 JSON 输出和工具调用。
  这些公开样例预检不包含用户课件。

官方依据：[Chat Completions](https://docs.siliconflow.cn/docs/api/chat-completions-post)、
[Embedding API](https://siliconflow.readme.io/reference/createembedding)、
[推理参数](https://docs.siliconflow.cn/docs/userguide/capabilities/reasoning)、
[Qwen 模型页](https://www.siliconflow.com/models/qwen3-5-35b-a3b)。
当前推荐采用显式关闭思考的 DeepSeek-V3.2 Pro 配置，temperature=0，其他可选采样参数留空。
[硅基流动模型说明](https://www.siliconflow.com/models/deepseek-v3-2)列出 JSON 模式和工具调用支持；
具体模型名称还应通过当前账号的 `/models` 及实际调用确认。

也支持 Qwen 的模型特定默认值：按
[Qwen3.5 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-35B-A3B#best-practices)，
`Qwen/Qwen3.5-35B-A3B` 默认使用 temperature=0.7、top_p=0.8、top_k=20、min_p=0、
presence_penalty=1.5。旧 `Qwen/Qwen3-30B-A3B-Instruct-2507` 仅默认前四项，不默认发送
presence_penalty。top_k/min_p通过 `extra_body` 发送。这些默认值不自动用于其他模型。
切换模型时应同步清空不适用的采样覆盖字段，不能保留上一模型的专用参数。

当前问答配置显式采用 [Qwen3.8 官方模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)
给出的非思考参数：0.7/0.8/20/0、presence_penalty=1.5。此模型没有隐式采样默认值，
因此应填写下列 `KNOWLEDGE_QA_*` 覆盖项。接口可调用和引用身份有效不保证答案完全
受材料支持；仍须按自己的知识库验证答案和延迟。

## 推荐配置

在被 Git 忽略的 `.env` 或 `.env.knowledge` 配置真实凭据，不把 API key 写入文档。
每个独立模型的 key、base URL、model 三项必须一起设置；可复用已有硅基流动凭据。
未启用独立配置时，思考参数默认留空，避免向旧模型发送不支持的扩展字段。

```dotenv
KNOWLEDGE_LLM_API_KEY=<硅基流动API key>
KNOWLEDGE_LLM_BASE_URL=https://api.siliconflow.cn/v1
KNOWLEDGE_LLM_MODEL=Pro/deepseek-ai/DeepSeek-V3.2
KNOWLEDGE_LLM_ENABLE_THINKING=false
KNOWLEDGE_LLM_TEMPERATURE=0
KNOWLEDGE_LLM_TOP_P=
KNOWLEDGE_LLM_TOP_K=
KNOWLEDGE_LLM_MIN_P=
KNOWLEDGE_LLM_PRESENCE_PENALTY=
KNOWLEDGE_LLM_MAX_OUTPUT_TOKENS=8192
KNOWLEDGE_LLM_MAX_ASYNC=2
KNOWLEDGE_EMBEDDING_API_KEY=<硅基流动API key>
KNOWLEDGE_EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
KNOWLEDGE_EMBEDDING_MODEL=BAAI/bge-m3
KNOWLEDGE_EMBEDDING_DIM=1024
KNOWLEDGE_EMBEDDING_MAX_INPUT_TOKENS=8192
KNOWLEDGE_EMBEDDING_TOKEN_BUDGET=4096
KNOWLEDGE_EMBEDDING_MAX_UTF8_BYTES=8000
KNOWLEDGE_CHUNK_TOKENS=800
KNOWLEDGE_CHUNK_OVERLAP_TOKENS=100
KNOWLEDGE_EXTRACT_MAX_RECORDS=40
KNOWLEDGE_EXTRACT_MAX_ENTITIES=20
KNOWLEDGE_EXTRACT_MAX_GLEANING=0
KNOWLEDGE_LLM_TIMEOUT_SECONDS=180
KNOWLEDGE_LLM_SDK_TIMEOUT_SECONDS=300
KNOWLEDGE_EMBEDDING_TIMEOUT_SECONDS=60
KNOWLEDGE_EMBEDDING_SDK_TIMEOUT_SECONDS=120
KNOWLEDGE_QUERY_TIMEOUT_SECONDS=90
KNOWLEDGE_MUTATION_TIMEOUT_SECONDS=900
KNOWLEDGE_QA_MAX_OUTPUT_TOKENS=4096
KNOWLEDGE_QA_MODEL=Qwen/Qwen3.8-27B
KNOWLEDGE_QA_ENABLE_THINKING=false
KNOWLEDGE_QA_TEMPERATURE=0.7
KNOWLEDGE_QA_TOP_P=0.8
KNOWLEDGE_QA_TOP_K=20
KNOWLEDGE_QA_MIN_P=0
KNOWLEDGE_QA_PRESENCE_PENALTY=1.5
KNOWLEDGE_QA_TIMEOUT_SECONDS=120
KNOWLEDGE_HTTP_TIMEOUT_SECONDS=100
KNOWLEDGE_TOOL_TIMEOUT_SECONDS=105
```

Compose 扩展已透传这些参数；本机直接运行时需将变量导入图服务进程环境。
主应用会读取项目 `.env`，模型配置在知识库请求首次使用时创建客户端。修改已运行
服务的模型配置后，需要重启后端进程才能使用新配置。

问答模型仍共享完整的 `KNOWLEDGE_LLM_API_KEY/BASE_URL/MODEL` 连接配置，不新增
凭据或端点，也不自动切换后备模型。问答专用生成参数优先；问答和抽取模型相同时，
未指定的问答参数继承 `KNOWLEDGE_LLM_*`；模型不同时，使用问答模型自己的默认值，
不继承抽取模型的采样覆盖。Qwen3 Instruct 的默认值为0.7/0.8/20/0并关闭思考；
上面的 Qwen3.8 示例则显式填写所需参数。
只更改问答模型或参数不会使图索引失效，但会使旧暂停问答会话需要重新开始。

## 输入、输出与超时边界

4096是SDK估算切块预算，8000是NFKC归一化后的UTF-8字节保护上限，都不是BGE原生
tokenizer的精确计数。图服务没有增加模型下载或tokenizer依赖。超出安全界限时明确
拒绝，不由provider回调静默裁掉原文。文档、实体描述、关系描述和查询均经过此保护；
SDK仍可对派生的向量检索描述做有界投影，完整描述和原始资料另外保存。

模型返回`finish_reason=length`时，图谱抽取拒绝这份部分结果，任务进入失败/dirty；
知识库问答也拒绝执行不完整工具调用或发布截断回答。输出上限是上限，SDK某次调用
要求更低上限时会保留较低值。

知识库问答遇到HTTP 200但无正文、无工具调用的空响应时，在原总截止时间内最多重试
一次；再次为空时返回模型服务错误，不把它伪装成“资料不足”。拒绝、内容过滤和截断
不触发这次重试。知识库问答先通过工具检索，再以 JSON 正文返回完整答案、引用 ID
和是否拒答；最终答案不再放入 `finalize` 工具参数。首次检索成功前必须调用工具，
只搜索网页摘要不满足此条件；成功检索但结果为空时允许明确拒答。无效 JSON、
尚未检索就结束及失败的工具调用不能被当作有效回答。

证据须在整次检索的格式、来源身份、数量和知识库版本校验成功后统一保存。
检索中途失败时不留下本次新证据，防止模型引用尚未实际收到的内容。

JSON 答案仍经过严格字段解析、真实引用和知识库版本检查；保留检索和 `ask_user`
能力，达到最大轮数时也使用相同的 JSON 答案校验。JSON 格式与引用身份有效本身
不证明内容正确，仍需逐题检查完整性和原文支持。控制协议升级会使旧暂停的知识库
会话失效，需要重新开始；普通文档会话不受这项升级影响。

所选知识库问答模型须同时支持工具调用和 `response_format={"type":"json_object"}`。
更换供应商或使用旧模型配置回退时，应先验证这两项能力；参见硅基流动的
[JSON 模式说明](https://docs.siliconflow.cn/docs/userguide/guides/json-mode)。

知识库检索向模型提供最多20个候选片段，与图服务的候选数量一致。已经提供过的证据
再次出现时只发送身份、元数据及 `already_observed=true`，完整正文仍在此前工具消息
及内部证据状态中。原文注入检测始终检查本次取回的完整文本；最终引用卡片保留摘要。
每次工具 JSON 最多65,536个UTF-8字节；本轮唯一证据的完整投影最多131,072字节、
64条。超限拒绝整批结果，不裁掉候选片段来凑预算。工具内部会抛出413异常；代理会
按工具失败处理，最终 HTTP 响应可能是503，而非直接透传413。这些是证据字节保护，
不是模型原生 tokenizer 的精确计数，也不是整个会话的 token 预算。

知识库每轮的控制说明合并到首条 system 消息，仅包含轮次、工具名和证据ID，
不把检索正文或模型生成的工具参数提升为系统指令。合并只用于当次模型请求，
不会写入暂停会话的原始消息；这也兼容要求 system 消息位于开头的模型。

每个片段提示模型最多提取40条实体/关系记录，其中实体最多20条，并关闭额外补抽。
这是图摘要的数量预算，完整原文和检索片段仍保留；图谱不保证穷尽所有实体与关系。
这些抽取参数也参与索引哈希，修改后必须重建。

HTTP和SDK预算分别设置。锁定SDK会将LLM SDK timeout再乘二作为worker执行上限，
所以300秒对应约600秒；900秒是整份资料的入库/重建上限。查询90秒、网关读取100秒、
知识工具105秒构成独立的交互截止时间。禁用OpenAI客户端的嵌套重试，避免无界叠加。

## 模型变更与复测

模型、provider地址、分块、输入/输出预算、采样参数及思考策略参与索引配置哈希。变更后旧索引
必须显式重建；即使新旧模型都是1024维，也不能复用原向量。重建从当前有效资料重建，
重放纠错，成功后才恢复查询。暂停问答的模型语义指纹变化时要求重新开始。

真实材料的原文、题目、索引和含正文输出必须留在仓库外。保留修复前失败记录，复测
使用相同资料与题目并另存新运行；不得把模型接口可调用或少量样例通过描述成质量结论。

`scripts/evaluate_knowledge.py` 保留原检索对比实验的全局模型配置，不读取这组独立模型
设置。验证当前知识库配置时应通过生产图服务（`Settings.from_env()`）和知识库 HTTP
接口，并记录实际请求的模型名称；旧实验结果不能用于证明本次模型切换已生效。
