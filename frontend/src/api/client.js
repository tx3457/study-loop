/**
 * API Client — StudyLoop 后端请求封装
 *
 * 开发环境：VITE_API_URL 未设置 → /api，由 Vite 反代到本地后端
 * Docker 环境：VITE_API_URL=/api → 由 Nginx 反代到后端
 */

const BASE_URL = import.meta.env.VITE_API_URL || '/api'
const REQUEST_ID_PATTERN = /^req_[0-9a-f]{32}$/

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
  const res = await fetch(`${BASE_URL}${path}`, options)

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
}) {
  return request('/learning-paths', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(idempotency_key ? { 'Idempotency-Key': idempotency_key } : {}),
    },
    body: JSON.stringify({ document_id, user_id }),
  })
}

/** 读取服务端持久化的学习路径资源。 */
export async function getLearningPathResource(learningPathId) {
  return request(`/learning-paths/${encodeURIComponent(learningPathId)}`)
}

/** 读取默认用户最近创建的学习路径；没有记录时返回 null。 */
export async function getCurrentLearningPathResource(documentId = null) {
  const query = documentId
    ? `?${new URLSearchParams({ document_id: documentId })}`
    : ''
  return request(`/learning-paths/current${query}`)
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
}) {
  return request('/session/start', {
    method: 'POST',
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
export async function getSessionSnapshot(sessionId) {
  return request(`/session/${encodeURIComponent(sessionId)}`)
}

/** 提交单题答案 */
export async function submitAnswer(sessionId, {
  answer,
  question_index,
  idempotency_key,
}) {
  return request(`/session/${sessionId}/answer`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(idempotency_key ? { 'Idempotency-Key': idempotency_key } : {}),
    },
    body: JSON.stringify({ answer, question_index }),
  })
}

/** 获取答题结果 */
export async function getSessionResult(sessionId) {
  return request(`/session/${sessionId}/result`)
}

/** AI 批改 */
export async function gradeSession(sessionId) {
  return request(`/session/${sessionId}/grade`, { method: 'POST' })
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

/** 错题本 */
export async function getWrongQuestions(documentId, userId = 'default_user') {
  const query = new URLSearchParams({ user_id: userId })
  return request(`/wrong-questions/${encodeURIComponent(documentId)}?${query}`)
}

/** 用当前文档的持久错题创建一轮重练会话 */
export async function startWrongQuestionPractice(
  documentId,
  userId = 'default_user',
  idempotencyKey,
) {
  const query = new URLSearchParams({ user_id: userId })
  return request(
    `/wrong-questions/${encodeURIComponent(documentId)}/practice?${query}`,
    {
      method: 'POST',
      headers: idempotencyKey ? { 'Idempotency-Key': idempotencyKey } : {},
    },
  )
}

/* ═══════════════════════════════════════════════════════════════════
   AI Grading & Report
   ═══════════════════════════════════════════════════════════════════ */

/** 学习评估报告 */
export async function generateReport(sessionId) {
  return request(`/session/${sessionId}/report`, { method: 'POST' })
}

/* ═══════════════════════════════════════════════════════════════════
   Autonomous Agent（bounded ReAct + HITL）
   ═══════════════════════════════════════════════════════════════════ */

/** 启动 autonomous agent（首次提问） */
export async function runAutonomous({
  query,
  user_id = 'default_user',
  document_id = null,
  grounding_required = false,
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
    body: JSON.stringify({ query, user_id, document_id, grounding_required }),
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
   Adaptive Learning Loop
   ═══════════════════════════════════════════════════════════════════ */

/** 开启自适应辅导会话（agent 决策开场 + 出第一轮题） */
export async function startAdaptive({
  user_id = 'default_user',
  document_id,
  goal,
  idempotency_key,
}) {
  return request('/agent/adaptive/start', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(idempotency_key ? { 'Idempotency-Key': idempotency_key } : {}),
    },
    body: JSON.stringify({ user_id, document_id, goal }),
  })
}

/** 读取服务端持久化的 Adaptive 安全快照。 */
export async function getAdaptiveSnapshot(adaptiveSessionId) {
  return request(
    `/agent/adaptive/${encodeURIComponent(adaptiveSessionId)}`,
  )
}

/** 提交本轮作答，推进闭环（批改 → agent 决策下一步） */
export async function submitAdaptive({
  adaptive_session_id,
  answers,
  turn,
  revision,
  idempotency_key,
}) {
  return request('/agent/adaptive/submit', {
    method: 'POST',
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
  const res = await fetch(`${BASE_URL}/agent/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  })

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
