/**
 * API Client — StudyLoop 后端请求封装
 *
 * 开发环境：VITE_API_URL 未设置 → /api，由 Vite 反代到本地后端
 * Docker 环境：VITE_API_URL=/api → 由 Nginx 反代到后端
 */

const BASE_URL = import.meta.env.VITE_API_URL || '/api'

/**
 * 通用请求封装，自动处理错误
 */
async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, options)

  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    const error = new Error(body.detail || body.error || `请求失败 (${res.status})`)
    error.status = res.status
    error.code = body.code
    error.reason = body.reason
    throw error
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

/* ═══════════════════════════════════════════════════════════════════
   Quiz Session
   ═══════════════════════════════════════════════════════════════════ */

/** 开始答题会话 */
export async function startSession({ document_id, description, count, difficulty, type, user_id }) {
  return request('/session/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ document_id, description, count, difficulty, type, user_id }),
  })
}

/** 提交单题答案 */
export async function submitAnswer(sessionId, answer) {
  return request(`/session/${sessionId}/answer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ answer }),
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
export async function getWrongQuestions(documentId) {
  return request(`/wrong-questions/${encodeURIComponent(documentId)}`)
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
  idempotency_key,
}) {
  return request('/agent/autonomous', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(idempotency_key ? { 'Idempotency-Key': idempotency_key } : {}),
    },
    body: JSON.stringify({ query, user_id, document_id }),
  })
}

/** 续跑 autonomous agent（ask_user 后用户回答） */
export async function continueAutonomous({
  conversation_id,
  user_reply,
  idempotency_key,
}) {
  return request('/agent/autonomous/continue', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(idempotency_key ? { 'Idempotency-Key': idempotency_key } : {}),
    },
    body: JSON.stringify({ conversation_id, user_reply }),
  })
}

/** 查询 audit trail（按 run_id） */
export async function getAudit(runId) {
  return request(`/audit/${encodeURIComponent(runId)}`)
}

/* ═══════════════════════════════════════════════════════════════════
   Adaptive Learning Loop（Direction A：Agent 驱动的自适应学习闭环）
   ═══════════════════════════════════════════════════════════════════ */

/** 开启自适应辅导会话（agent 决策开场 + 出第一轮题） */
export async function startAdaptive({ user_id = 'default_user', document_id, goal }) {
  return request('/agent/adaptive/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id, document_id, goal }),
  })
}

/** 提交本轮作答，推进闭环（批改 → agent 决策下一步） */
export async function submitAdaptive({ adaptive_session_id, answers }) {
  return request('/agent/adaptive/submit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ adaptive_session_id, answers }),
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
    throw new Error(body.detail || `Stream 请求失败 (${res.status})`)
  }

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
        try {
          const event = JSON.parse(line.slice(6))
          onEvent(event)
          if (event.type === 'error') throw new Error(event.detail)
        } catch (e) {
          if (e.message !== line.slice(6)) throw e // JSON parse error 忽略，业务 error 抛出
        }
      }
    }
  }
}
