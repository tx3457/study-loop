/**
 * API Client — StudyLoop 后端请求封装
 *
 * 开发环境：VITE_API_URL 未设置 → /api，由 Vite 反代到本地后端
 * Docker 环境：VITE_API_URL=/api → 由 Nginx 反代到后端
 */

const BASE_URL = import.meta.env.VITE_API_URL || '/api'
const REQUEST_ID_PATTERN = /^req_[0-9a-f]{32}$/
const TOKEN_KEY = 'studyloop.auth.token'

// 后端设置 STUDYLOOP_AUTH_TOKEN 后，所有业务端点都要求 Bearer 令牌。
// 没配置时后端是匿名模式，这里读不到令牌就什么都不加，行为不变。
export function readAuthToken() {
  try {
    return globalThis.localStorage?.getItem(TOKEN_KEY) || ''
  } catch {
    // 隐私模式下 localStorage 不可读：当作没配置，请求照发，由后端决定放不放行。
    return ''
  }
}

export function writeAuthToken(token) {
  try {
    const value = (token || '').trim()
    if (value) globalThis.localStorage?.setItem(TOKEN_KEY, value)
    else globalThis.localStorage?.removeItem(TOKEN_KEY)
  } catch {
    // 隐私模式下写不进去：本次会话内仍可请求，只是刷新后要重填。
  }
}

function withAuth(options = {}) {
  const token = readAuthToken()
  if (!token) return options
  return {
    ...options,
    headers: { ...(options.headers || {}), Authorization: `Bearer ${token}` },
  }
}

function responseRequestId(response, body = null) {
  const headerValue = response?.headers?.get?.('X-Request-ID')
  const bodyValue = body && typeof body === 'object' ? body.request_id : null
  return [headerValue, bodyValue].find(
    value => typeof value === 'string' && REQUEST_ID_PATTERN.test(value),
  ) || null
}

function apiError(response, body, fallback) {
  const rawMessage = body?.detail || body?.error || fallback
  const message = typeof rawMessage === 'string' ? rawMessage : fallback
  const requestId = responseRequestId(response, body)
  const error = new Error(
    requestId ? `${message}（请求编号：${requestId}）` : message,
  )
  error.status = response?.status
  error.code = body?.code
  error.reason = body?.reason
  error.requestId = requestId
  return error
}

export function createIdempotencyKey() {
  const cryptoApi = globalThis.crypto
  if (typeof cryptoApi?.randomUUID === 'function') {
    return cryptoApi.randomUUID()
  }
  if (typeof cryptoApi?.getRandomValues !== 'function') {
    throw new Error('当前浏览器无法生成安全请求标识，请升级浏览器后重试')
  }

  const bytes = cryptoApi.getRandomValues(new Uint8Array(16))
  bytes[6] = (bytes[6] & 0x0f) | 0x40
  bytes[8] = (bytes[8] & 0x3f) | 0x80
  const hex = Array.from(bytes, byte => byte.toString(16).padStart(2, '0'))
  return [
    hex.slice(0, 4).join(''),
    hex.slice(4, 6).join(''),
    hex.slice(6, 8).join(''),
    hex.slice(8, 10).join(''),
    hex.slice(10, 16).join(''),
  ].join('-')
}

export function isTerminalExecutionError(error) {
  return error?.status === 410
    || error?.code === 'side_effect_ambiguous'
    || (
      error?.code === 'idempotency_conflict'
      && error.reason !== 'in_progress'
    )
}

/**
 * 通用请求封装，自动处理错误
 */
async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, withAuth(options))

  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw apiError(res, body, `请求失败 (${res.status})`)
  }

  return res.json()
}

/** 上传文档（multipart/form-data） */
export async function uploadDocument(file) {
  const formData = new FormData()
  formData.append('file', file)

  return request('/documents/upload', {
    method: 'POST',
    body: formData,
  })
}

/**
 * 模型服务配置自检：只列模型目录，不生成、不嵌入，后端缓存 30 秒。
 * 配置有问题时后端用 503 返回同样结构的结果——那是答案而不是故障，
 * 所以 200 和 503 都读正文，其余状态才按错误处理。
 */
export async function getProviderHealth() {
  const res = await fetch(`${BASE_URL}/health/providers`, withAuth())
  if (res.status !== 200 && res.status !== 503) {
    const body = await res.json().catch(() => ({}))
    throw apiError(res, body, `请求失败 (${res.status})`)
  }
  return res.json()
}

/** 内置示例材料（纯文本），由页面包装成 File 后走普通上传流程。 */
export async function getSampleDocument() {
  return request('/documents/sample')
}

/** 获取文档列表 */
export async function getDocuments() {
  return request('/documents')
}

/** 删除文档 */
export async function deleteDocument(documentId) {
  return request(`/documents/${encodeURIComponent(documentId)}`, {
    method: 'DELETE',
  })
}

/* ═══════════════════════════════════════════════════════════════════
   Learning Path
   ═══════════════════════════════════════════════════════════════════ */

/** 生成 AI 学习路径 */
export async function generateLearningPath(documentId) {
  return request(`/learning-path/${encodeURIComponent(documentId)}`, {
    method: 'POST',
  })
}

/** 创建可跨刷新恢复的学习路径资源。 */
export async function createLearningPathResource({
  document_id,
  user_id = 'default_user',
  idempotency_key,
  signal,
}) {
  return request('/learning-paths', {
    method: 'POST',
    signal,
    headers: {
      'Content-Type': 'application/json',
      ...(idempotency_key ? { 'Idempotency-Key': idempotency_key } : {}),
    },
    body: JSON.stringify({ document_id, user_id }),
  })
}

/** 读取服务端持久化的学习路径资源。 */
export async function getLearningPathResource(
  learningPathId,
  { signal } = {},
) {
  return request(`/learning-paths/${encodeURIComponent(learningPathId)}`, {
    signal,
  })
}

/** 读取默认用户最近创建的学习路径；没有记录时返回 null。 */
export async function getCurrentLearningPathResource(
  documentId = null,
  { signal } = {},
) {
  const query = documentId
    ? `?${new URLSearchParams({ document_id: documentId })}`
    : ''
  return request(`/learning-paths/current${query}`, { signal })
}

/* ═══════════════════════════════════════════════════════════════════
   Quiz Session
   ═══════════════════════════════════════════════════════════════════ */

/** 开始答题会话 */
export async function startSession({
  document_id,
  description,
  count,
  difficulty,
  type,
  user_id,
  learning_path_source,
  idempotency_key,
  signal,
}) {
  return request('/session/start', {
    method: 'POST',
    signal,
    headers: {
      'Content-Type': 'application/json',
      ...(idempotency_key ? { 'Idempotency-Key': idempotency_key } : {}),
    },
    body: JSON.stringify({
      document_id,
      description,
      count,
      difficulty,
      type,
      user_id,
      ...(learning_path_source ? { learning_path_source } : {}),
    }),
  })
}

/** 获取浏览器安全的会话快照，用于刷新与跨页面恢复 */
export async function getSessionSnapshot(sessionId, { signal } = {}) {
  return request(`/session/${encodeURIComponent(sessionId)}`, { signal })
}

/** 提交单题答案 */
export async function submitAnswer(sessionId, {
  answer,
  question_index,
  idempotency_key,
  signal,
}) {
  return request(`/session/${encodeURIComponent(sessionId)}/answer`, {
    method: 'POST',
    signal,
    headers: {
      'Content-Type': 'application/json',
      ...(idempotency_key ? { 'Idempotency-Key': idempotency_key } : {}),
    },
    body: JSON.stringify({ answer, question_index }),
  })
}

/** 获取答题结果 */
export async function getSessionResult(sessionId, { signal } = {}) {
  return request(`/session/${encodeURIComponent(sessionId)}/result`, { signal })
}

/** AI 批改 */
export async function gradeSession(sessionId, { signal } = {}) {
  return request(`/session/${encodeURIComponent(sessionId)}/grade`, { method: 'POST', signal })
}

/* ═══════════════════════════════════════════════════════════════════
   Dashboard
   ═══════════════════════════════════════════════════════════════════ */

/** 用户画像（语义记忆：掌握度 + 薄弱知识点） */
export async function getUserProfile(userId = 'default_user') {
  return request(`/user/${encodeURIComponent(userId)}/profile`)
}

/** 学习历史（情节记忆：会话摘要列表） */
export async function getUserSessions(userId = 'default_user') {
  return request(`/user/${encodeURIComponent(userId)}/sessions`)
}

/** 模型用量：服务商返回的 token 数，按功能归类；不估算金额。 */
export async function getModelUsage() {
  return request('/usage')
}

/** 今天该复习的知识点。调度早就在跑，这里只是把它读出来给人看。 */
export async function getDueReviews(userId = 'default_user') {
  return request(`/user/${encodeURIComponent(userId)}/reviews/due`)
}

/** 错题本。身份由后端从 Authorization 头解析，这里不再自报 user_id。 */
export async function getWrongQuestions(documentId) {
  return request(`/wrong-questions/${encodeURIComponent(documentId)}`)
}

/** 用当前文档的持久错题创建一轮重练会话 */
export async function startWrongQuestionPractice(
  documentId,
  idempotencyKey,
  { signal } = {},
) {
  return request(
    `/wrong-questions/${encodeURIComponent(documentId)}/practice`,
    {
      method: 'POST',
      signal,
      headers: idempotencyKey ? { 'Idempotency-Key': idempotencyKey } : {},
    },
  )
}

/* ═══════════════════════════════════════════════════════════════════
   AI Grading & Report
   ═══════════════════════════════════════════════════════════════════ */

/** 学习评估报告 */
export async function generateReport(sessionId, { signal } = {}) {
  return request(`/session/${encodeURIComponent(sessionId)}/report`, { method: 'POST', signal })
}

/* ═══════════════════════════════════════════════════════════════════
   Autonomous Agent（bounded ReAct + HITL）
   ═══════════════════════════════════════════════════════════════════ */

/** 启动 autonomous agent（首次提问） */
export async function runAutonomous({
  query,
  user_id = 'default_user',
  document_id = null,
  knowledge_base_id = null,
  grounding_required = false,
  web_enabled = false,
  idempotency_key,
  signal,
}) {
  return request('/agent/autonomous', {
    method: 'POST',
    signal,
    headers: {
      'Content-Type': 'application/json',
      ...(idempotency_key ? { 'Idempotency-Key': idempotency_key } : {}),
    },
    body: JSON.stringify({
      query,
      user_id,
      document_id,
      knowledge_base_id,
      grounding_required,
      web_enabled,
    }),
  })
}

/** 续跑 autonomous agent（ask_user 后用户回答） */
export async function continueAutonomous({
  conversation_id,
  user_reply,
  idempotency_key,
  signal,
}) {
  return request('/agent/autonomous/continue', {
    method: 'POST',
    signal,
    headers: {
      'Content-Type': 'application/json',
      ...(idempotency_key ? { 'Idempotency-Key': idempotency_key } : {}),
    },
    body: JSON.stringify({ conversation_id, user_reply }),
  })
}

/** 放弃一个尚未续跑的 autonomous HITL 暂停会话。 */
export async function cancelAutonomous(conversationId, { signal } = {}) {
  return request(`/agent/autonomous/${encodeURIComponent(conversationId)}`, {
    method: 'DELETE',
    signal,
  })
}

/** 查询 audit trail（按 run_id） */
export async function getAudit(runId) {
  return request(`/audit/${encodeURIComponent(runId)}`)
}

/* ═══════════════════════════════════════════════════════════════════
   Knowledge bases
   ═══════════════════════════════════════════════════════════════════ */

const jsonMutation = (method, body, idempotencyKey = null) => ({
  method,
  headers: {
    'Content-Type': 'application/json',
    ...(idempotencyKey ? { 'Idempotency-Key': idempotencyKey } : {}),
  },
  body: JSON.stringify(body),
})

export const getKnowledgeCapabilities = ({ signal } = {}) =>
  request('/knowledge-bases/capabilities', { signal })

export const getKnowledgeBases = ({ signal } = {}) =>
  request('/knowledge-bases', { signal })

export const createKnowledgeBase = (body, idempotencyKey) =>
  request('/knowledge-bases', jsonMutation('POST', body, idempotencyKey))

export const getKnowledgeBase = (knowledgeBaseId, { signal } = {}) =>
  request(`/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}`, { signal })

export const updateKnowledgeBase = (knowledgeBaseId, body, idempotencyKey) =>
  request(
    `/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}`,
    jsonMutation('PATCH', body, idempotencyKey),
  )

export const deleteKnowledgeBase = (knowledgeBaseId, expectedRevision, idempotencyKey) =>
  request(
    `/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}`,
    jsonMutation('DELETE', { expected_revision: expectedRevision }, idempotencyKey),
  )

export const getKnowledgeDocuments = (
  knowledgeBaseId,
  { signal, limit = 50, offset = 0 } = {},
) => {
  const query = new URLSearchParams({ limit: String(limit), offset: String(offset) })
  return request(
    `/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}/documents?${query}`,
    { signal },
  )
}

export async function uploadKnowledgeDocument(knowledgeBaseId, file, expectedRevision, idempotencyKey) {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('expected_revision', String(expectedRevision))
  return request(`/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}/documents/upload`, {
    method: 'POST',
    headers: idempotencyKey ? { 'Idempotency-Key': idempotencyKey } : {},
    body: formData,
  })
}

export async function replaceKnowledgeDocument(
  knowledgeBaseId,
  documentId,
  file,
  expectedRevision,
  idempotencyKey,
) {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('expected_revision', String(expectedRevision))
  return request(
    `/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}/documents/${encodeURIComponent(documentId)}`,
    {
      method: 'PUT',
      headers: idempotencyKey ? { 'Idempotency-Key': idempotencyKey } : {},
      body: formData,
    },
  )
}

export const deleteKnowledgeDocument = (knowledgeBaseId, documentId, expectedRevision, idempotencyKey) =>
  request(
    `/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}/documents/${encodeURIComponent(documentId)}`,
    jsonMutation('DELETE', { expected_revision: expectedRevision }, idempotencyKey),
  )

export const importLegacyDocument = (knowledgeBaseId, legacyDocumentId, expectedRevision, idempotencyKey) =>
  request(
    `/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}/documents/import-legacy`,
    jsonMutation('POST', {
      legacy_document_id: legacyDocumentId,
      expected_revision: expectedRevision,
    }, idempotencyKey),
  )

export function getKnowledgeGraph(knowledgeBaseId, { search = '', focus = '', signal } = {}) {
  const query = new URLSearchParams({ limit_nodes: '200', limit_edges: '400' })
  if (search.trim()) query.set('search', search.trim())
  if (focus.trim()) query.set('focus', focus.trim())
  return request(`/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}/graph?${query}`, { signal })
}

export const getKnowledgeSource = (knowledgeBaseId, sourceVersionId, { signal } = {}) =>
  request(
    `/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}/sources/${encodeURIComponent(sourceVersionId)}`,
    { signal },
  )

export const createKnowledgeCorrection = (knowledgeBaseId, body, idempotencyKey) =>
  request(
    `/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}/corrections`,
    jsonMutation('POST', body, idempotencyKey),
  )

export const importWebSnapshot = (knowledgeBaseId, snapshotId, expectedRevision, idempotencyKey) =>
  request(
    `/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}/web-import`,
    jsonMutation('POST', {
      snapshot_id: snapshotId,
      expected_revision: expectedRevision,
    }, idempotencyKey),
  )

export const getKnowledgeJob = (jobId, { signal } = {}) =>
  request(`/knowledge-jobs/${encodeURIComponent(jobId)}`, { signal })

export const retryKnowledgeJob = (jobId, expectedRevision, idempotencyKey) =>
  request(
    `/knowledge-jobs/${encodeURIComponent(jobId)}/retry`,
    jsonMutation('POST', { expected_revision: expectedRevision }, idempotencyKey),
  )

export const rebuildKnowledgeBase = (knowledgeBaseId, expectedRevision, idempotencyKey) =>
  request(
    `/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}/rebuild`,
    jsonMutation('POST', { expected_revision: expectedRevision }, idempotencyKey),
  )

/* ═══════════════════════════════════════════════════════════════════
   Adaptive Learning Loop
   ═══════════════════════════════════════════════════════════════════ */

/** 开启自适应辅导会话（agent 决策开场 + 出第一轮题） */
export async function startAdaptive({
  user_id = 'default_user',
  document_id,
  goal,
  idempotency_key,
  signal,
}) {
  return request('/agent/adaptive/start', {
    method: 'POST',
    signal,
    headers: {
      'Content-Type': 'application/json',
      ...(idempotency_key ? { 'Idempotency-Key': idempotency_key } : {}),
    },
    body: JSON.stringify({ user_id, document_id, goal }),
  })
}

/** 读取服务端持久化的 Adaptive 安全快照。 */
export async function getAdaptiveSnapshot(
  adaptiveSessionId,
  { signal } = {},
) {
  return request(
    `/agent/adaptive/${encodeURIComponent(adaptiveSessionId)}`,
    { signal },
  )
}

/** 提交本轮作答，推进闭环（批改 → agent 决策下一步） */
export async function submitAdaptive({
  adaptive_session_id,
  answers,
  turn,
  revision,
  idempotency_key,
  signal,
}) {
  return request('/agent/adaptive/submit', {
    method: 'POST',
    signal,
    headers: {
      'Content-Type': 'application/json',
      ...(idempotency_key ? { 'Idempotency-Key': idempotency_key } : {}),
    },
    body: JSON.stringify({ adaptive_session_id, answers, turn, revision }),
  })
}

/* ═══════════════════════════════════════════════════════════════════
   SSE Streaming
   ═══════════════════════════════════════════════════════════════════ */

/**
 * 连接 Agent 流式端点，逐事件回调
 * @param {object} params - RunRequest 参数
 * @param {function} onEvent - 每个 SSE 事件的回调 ({type, node, label, result, detail})
 * @returns {Promise} resolve on done, reject on error
 */
export async function streamAgent(params, onEvent) {
  // SSE 走裸 fetch（需要 ReadableStream），不经过 request()，
  // 所以必须自己带上令牌——漏掉就会在启用认证后 401。
  const res = await fetch(`${BASE_URL}/agent/stream`, withAuth({
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  }))

  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw apiError(res, body, `Stream 请求失败 (${res.status})`)
  }

  const responseRequestIdValue = responseRequestId(res)
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() // 保留最后一个不完整行

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        let event
        try {
          event = JSON.parse(line.slice(6))
        } catch {
          continue
        }
        onEvent(event)
        if (event.type === 'error') {
          const requestId = (
            typeof event.request_id === 'string'
            && REQUEST_ID_PATTERN.test(event.request_id)
          ) ? event.request_id : responseRequestIdValue
          const message = typeof event.detail === 'string'
            ? event.detail
            : 'Agent 流式执行失败'
          const error = new Error(
            requestId ? `${message}（请求编号：${requestId}）` : message,
          )
          error.code = event.code
          error.requestId = requestId
          throw error
        }
      }
    }
  }
}
