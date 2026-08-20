/**
 * Autonomous Agent 页面
 *
 * ReAct + HITL 流程：
 *   - 用户输入学习目标
 *   - Agent 自主决定调工具 / finalize / 问用户
 *   - LLM 调 ask_user → 弹框让用户答 → 提交后调 /continue 续跑
 *
 * 状态机：
 *   idle → running → (awaiting → continuing)* → done | error
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import {
  createIdempotencyKey,
  runAutonomous,
  continueAutonomous,
  getDocuments,
  isTerminalExecutionError,
} from '../api/client'
import './Autonomous.css'

const DEFAULT_USER_ID = 'default_user'

const initialState = {
  phase: 'idle',     // idle | running | awaiting | continuing | done | error
  request: {
    query: '',
    user_id: DEFAULT_USER_ID,
    document_id: '',
    grounding_required: false,
  },
  response: null,
  error: null,
  retryBlocked: false,
}

const AWAITING_STORAGE_KEY = 'study-loop.autonomous.awaiting.v1'
const UUID_V4_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i

function clearAwaitingRecovery() {
  try {
    globalThis.sessionStorage?.removeItem(AWAITING_STORAGE_KEY)
  } catch {
    // Storage can be unavailable in hardened browser contexts.
  }
}

function readAwaitingRecovery() {
  try {
    const raw = globalThis.sessionStorage?.getItem(AWAITING_STORAGE_KEY)
    if (!raw) return null

    const value = JSON.parse(raw)
    const request = value?.request
    const valid = value?.version === 1
      && typeof value.conversation_id === 'string'
      && /^[A-Za-z0-9_-]{1,256}$/.test(value.conversation_id)
      && typeof value.user_question === 'string'
      && value.user_question.trim().length > 0
      && value.user_question.length <= 8000
      && typeof value.draft === 'string'
      && value.draft.length <= 8000
      && request && typeof request === 'object'
      && typeof request.query === 'string'
      && request.query.trim().length > 0
      && request.query.length <= 8000
      && request.user_id === DEFAULT_USER_ID
      && typeof request.document_id === 'string'
      && request.document_id.length <= 1024
      && (
        request.grounding_required == null
        || typeof request.grounding_required === 'boolean'
      )
      && (
        value.continue_idempotency_key == null
        || (
          typeof value.continue_idempotency_key === 'string'
          && UUID_V4_PATTERN.test(value.continue_idempotency_key)
        )
      )

    if (!valid) throw new Error('invalid autonomous recovery payload')
    return {
      version: 1,
      conversation_id: value.conversation_id,
      user_question: value.user_question,
      draft: value.draft,
      request: {
        query: request.query,
        user_id: DEFAULT_USER_ID,
        document_id: request.document_id,
        // Older v1 recovery records did not include this additive field.
        grounding_required: request.grounding_required === true,
      },
      continue_idempotency_key: value.continue_idempotency_key || null,
    }
  } catch {
    clearAwaitingRecovery()
    return null
  }
}

function writeAwaitingRecovery(value) {
  try {
    globalThis.sessionStorage?.setItem(AWAITING_STORAGE_KEY, JSON.stringify(value))
  } catch {
    // The dialog remains usable even when storage is unavailable or full.
  }
}

function FormattedAnswer({ text }) {
  return (
    <div className="final-answer">
      {text.split(/(\*\*[^*]+\*\*)/g).map((part, index) =>
        part.startsWith('**') && part.endsWith('**')
          ? <strong key={index}>{part.slice(2, -2)}</strong>
          : <span key={index}>{part}</span>
      )}
    </div>
  )
}

export default function Autonomous() {
  const [restoredAwaiting] = useState(() => readAwaitingRecovery())
  const [state, setState] = useState(() => restoredAwaiting ? {
    ...initialState,
    phase: 'awaiting',
    request: restoredAwaiting.request,
    response: {
      awaiting_user_input: true,
      conversation_id: restoredAwaiting.conversation_id,
      user_question: restoredAwaiting.user_question,
      rounds_used: 0,
      steps: [],
      tools_called: ['ask_user'],
      truncated: false,
      grounding_status: restoredAwaiting.request.grounding_required
        ? 'pending'
        : 'not_requested',
      grounding_required: restoredAwaiting.request.grounding_required,
      grounding_document_id: restoredAwaiting.request.document_id || null,
    },
  } : initialState)
  const [askReply, setAskReply] = useState(restoredAwaiting?.draft || '')
  const [documents, setDocuments] = useState([])
  const [documentsLoading, setDocumentsLoading] = useState(false)
  const [documentsError, setDocumentsError] = useState(null)
  const documentsRequestId = useRef(0)
  const lastConversationId = useRef(restoredAwaiting?.conversation_id || null)
  const startIdempotencyKey = useRef(null)
  const continueIdempotencyKey = useRef(
    restoredAwaiting?.continue_idempotency_key || null
  )
  const modalRef = useRef(null)
  const modalInputRef = useRef(null)
  const startButtonRef = useRef(null)
  const resetButtonRef = useRef(null)
  const previousFocusRef = useRef(null)
  const dialogOpen = (
    state.phase === 'awaiting' || state.phase === 'continuing'
  ) && Boolean(state.response?.user_question)
  const formLocked = ['running', 'awaiting', 'continuing'].includes(state.phase)

  const loadDocs = useCallback(async () => {
    const requestId = ++documentsRequestId.current
    setDocumentsLoading(true)
    setDocumentsError(null)
    try {
      const data = await getDocuments()
      if (requestId !== documentsRequestId.current) return
      setDocuments(data.documents || [])
    } catch (err) {
      if (requestId !== documentsRequestId.current) return
      setDocumentsError(err.message)
    } finally {
      if (requestId === documentsRequestId.current) setDocumentsLoading(false)
    }
  }, [])

  useEffect(() => {
    loadDocs()
    return () => {
      documentsRequestId.current += 1
    }
  }, [loadDocs])

  function persistCurrentAwaiting(
    draft = askReply,
    idempotencyKey = continueIdempotencyKey.current
  ) {
    const conversationId = lastConversationId.current
    const userQuestion = state.response?.user_question
    if (!conversationId || !userQuestion) return
    writeAwaitingRecovery({
      version: 1,
      conversation_id: conversationId,
      user_question: userQuestion,
      draft,
      request: state.request,
      continue_idempotency_key: idempotencyKey,
    })
  }

  useEffect(() => {
    if (!dialogOpen) return
    const conversationId = lastConversationId.current
    const userQuestion = state.response?.user_question
    if (!conversationId || !userQuestion) return
    writeAwaitingRecovery({
      version: 1,
      conversation_id: conversationId,
      user_question: userQuestion,
      draft: askReply,
      request: state.request,
      continue_idempotency_key: continueIdempotencyKey.current,
    })
  }, [askReply, dialogOpen, state.phase, state.request, state.response?.user_question])

  useEffect(() => {
    if (!dialogOpen) return undefined

    previousFocusRef.current = document.activeElement
    const previousOverflow = document.body.style.overflow
    const focusFrame = window.requestAnimationFrame(() => modalInputRef.current?.focus())
    const handleKeyDown = (event) => {
      if (event.key !== 'Tab') return

      const modal = modalRef.current
      if (!modal) return
      const focusable = [...modal.querySelectorAll(
        'textarea:not([disabled]), button:not([disabled]), [tabindex]:not([tabindex="-1"])'
      )]
      const first = focusable[0]
      const last = focusable.at(-1)

      if (!first || !last) return
      if (event.shiftKey && (document.activeElement === first || !modal.contains(document.activeElement))) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first.focus()
      }
    }

    document.body.style.overflow = 'hidden'
    document.addEventListener('keydown', handleKeyDown)
    return () => {
      window.cancelAnimationFrame(focusFrame)
      document.body.style.overflow = previousOverflow
      document.removeEventListener('keydown', handleKeyDown)
      window.requestAnimationFrame(() => {
        const previousFocus = previousFocusRef.current
        if (previousFocus?.isConnected && !previousFocus.disabled) previousFocus.focus()
      })
    }
  }, [dialogOpen])

  useEffect(() => {
    const shouldFocusReply = state.phase === 'continuing'
      || (state.phase === 'awaiting' && state.error)
    if (!shouldFocusReply) return undefined

    const focusFrame = window.requestAnimationFrame(() => modalInputRef.current?.focus())
    return () => window.cancelAnimationFrame(focusFrame)
  }, [state.error, state.phase])

  useEffect(() => {
    if (state.phase !== 'error') return undefined

    const focusFrame = window.requestAnimationFrame(() => {
      const target = state.retryBlocked
        ? resetButtonRef.current
        : startButtonRef.current
      target?.focus()
    })
    return () => window.cancelAnimationFrame(focusFrame)
  }, [state.phase, state.retryBlocked])

  /** 处理 agent 响应：分发到对应状态 */
  function _handleResponse(resp) {
    continueIdempotencyKey.current = null
    if (resp.awaiting_user_input) {
      lastConversationId.current = resp.conversation_id
      writeAwaitingRecovery({
        version: 1,
        conversation_id: resp.conversation_id,
        user_question: resp.user_question,
        draft: '',
        request: state.request,
        continue_idempotency_key: null,
      })
      setState(s => ({
        ...s,
        phase: 'awaiting',
        response: resp,
        error: null,
        retryBlocked: false,
      }))
      setAskReply('')
    } else {
      clearAwaitingRecovery()
      lastConversationId.current = null
      setState(s => ({
        ...s,
        phase: 'done',
        response: resp,
        error: null,
        retryBlocked: false,
      }))
    }
  }

  /** 首次提交 */
  async function handleStart(e) {
    e.preventDefault()
    if (!state.request.query.trim()) return
    try {
      const idempotencyKey = startIdempotencyKey.current || createIdempotencyKey()
      startIdempotencyKey.current = idempotencyKey
      clearAwaitingRecovery()
      setState(s => ({
        ...s,
        phase: 'running',
        response: null,
        error: null,
        retryBlocked: false,
      }))
      const resp = await runAutonomous({
        query: state.request.query,
        user_id: DEFAULT_USER_ID,
        document_id: state.request.document_id || null,
        grounding_required: state.request.grounding_required,
        idempotency_key: idempotencyKey,
      })
      startIdempotencyKey.current = null
      _handleResponse(resp)
    } catch (err) {
      const retryBlocked = isTerminalExecutionError(err)
      if (retryBlocked) {
        startIdempotencyKey.current = null
        clearAwaitingRecovery()
      }
      setState(s => ({
        ...s,
        phase: 'error',
        error: err.message,
        retryBlocked,
      }))
    }
  }

  /** 提交 ask_user 回答续跑 */
  async function handleContinue() {
    if (
      state.phase === 'continuing'
      || !askReply.trim()
      || !lastConversationId.current
    ) return
    const conversationId = lastConversationId.current
    const userReply = askReply.trim()
    try {
      const idempotencyKey = continueIdempotencyKey.current || createIdempotencyKey()
      continueIdempotencyKey.current = idempotencyKey
      persistCurrentAwaiting(userReply, idempotencyKey)
      setState(s => ({ ...s, phase: 'continuing', error: null }))
      const resp = await continueAutonomous({
        conversation_id: conversationId,
        user_reply: userReply,
        idempotency_key: idempotencyKey,
      })
      _handleResponse(resp)
    } catch (err) {
      if (err.status === 404 || isTerminalExecutionError(err)) {
        clearAwaitingRecovery()
        continueIdempotencyKey.current = null
        lastConversationId.current = null
        setState(s => ({
          ...s,
          phase: 'error',
          error: `回答提交失败：${err.message}`,
          retryBlocked: true,
        }))
      } else {
        setState(s => ({
          ...s,
          phase: 'awaiting',
          error: err.message,
          retryBlocked: false,
        }))
      }
    }
  }

  function handleReset() {
    clearAwaitingRecovery()
    setState(initialState)
    setAskReply('')
    lastConversationId.current = null
    startIdempotencyKey.current = null
    continueIdempotencyKey.current = null
  }

  const r = state.response
  const citations = Array.isArray(r?.citations)
    ? r.citations.filter(citation => (
      citation
      && typeof citation.chunk_id === 'string'
      && typeof citation.document_id === 'string'
      && typeof citation.snippet === 'string'
    ))
    : []
  const invalidCitationCount = Number.isInteger(r?.invalid_citation_count)
    ? r.invalid_citation_count
    : Array.isArray(r?.invalid_citation_ids) ? r.invalid_citation_ids.length : 0
  const effectiveGroundingRequired = typeof r?.grounding_required === 'boolean'
    ? r.grounding_required
    : state.request.grounding_required
  const groundingStatus = r?.grounding_status
    || (state.phase === 'awaiting' && effectiveGroundingRequired
      ? 'pending'
      : 'not_requested')
  const isAbstained = r?.abstained === true || groundingStatus === 'abstained'
  const isSafetyBlocked = ['input_safety_blocked', 'output_leak_blocked']
    .includes(r?.finalize_reason)

  return (
    <div className="autonomous-page">
      <div
        className="autonomous-content"
        aria-hidden={dialogOpen ? true : undefined}
        inert={dialogOpen}
      >
      <header className="page-header">
        <h1 className="page-title">自主学习 Agent</h1>
        <p className="page-desc">
          描述学习目标，Agent 会选择合适的工具，并在需要时向你确认信息；
          选择材料后可要求最终回答附上本轮实际检索到的片段。
        </p>
      </header>

      {/* ── 输入区 ──────────────────────────────────────────────────────── */}
      <form className="auto-form" onSubmit={handleStart}>
        <div className="form-row">
          <label htmlFor="autonomous-goal">你的学习目标</label>
          <textarea
            id="autonomous-goal"
            rows={3}
            value={state.request.query}
            onChange={e => {
              startIdempotencyKey.current = null
              setState(s => ({ ...s, request: { ...s.request, query: e.target.value } }))
            }}
            placeholder="例如：帮我规划学习 RAG 的路径，然后出 3 道选择题"
            disabled={formLocked}
          />
        </div>

        <div className="form-row">
          <div className="doc-input">
            <div className="field-label-row">
              <label htmlFor="autonomous-document">文档 ID（可选）</label>
              <button
                type="button"
                className="btn-link"
                onClick={loadDocs}
                disabled={documentsLoading}
                aria-busy={documentsLoading}
              >
                {documentsLoading ? '刷新中...' : '刷新文档'}
              </button>
            </div>
            <input
              id="autonomous-document"
              list="doc-list"
              value={state.request.document_id}
              onChange={e => {
                startIdempotencyKey.current = null
                const documentId = e.target.value
                setState(s => {
                  const hadDocument = Boolean(s.request.document_id.trim())
                  return {
                    ...s,
                    request: {
                      ...s.request,
                      document_id: documentId,
                      // A newly selected document is grounded by default. Keep
                      // later explicit checkbox choices while the user edits it.
                      grounding_required: !hadDocument && documentId.trim()
                        ? true
                        : documentId.trim() ? s.request.grounding_required : false,
                    },
                  }
                })
              }}
              placeholder="留空让 Agent 主动询问"
              disabled={formLocked}
            />
            <datalist id="doc-list">
              {documents.map(d => <option key={d} value={d} />)}
            </datalist>
            {documentsError && (
              <p className="field-error" role="alert">
                文档列表加载失败：{documentsError}
              </p>
            )}
          </div>
        </div>

        <div className="grounding-row">
          <label className="grounding-option" htmlFor="autonomous-grounding">
            <input
              id="autonomous-grounding"
              type="checkbox"
              checked={state.request.grounding_required}
              onChange={e => {
                startIdempotencyKey.current = null
                setState(s => ({
                  ...s,
                  request: {
                    ...s.request,
                    grounding_required: e.target.checked,
                  },
                }))
              }}
              disabled={formLocked || !state.request.document_id.trim()}
            />
            <span>
              <strong>要求可核验文档引用</strong>
              <small>
                选择文档后默认启用；服务端只接受本轮检索返回的片段 ID，
                找不到有效片段时会明确拒答。
              </small>
            </span>
          </label>
        </div>

        <div className="form-actions">
          {!state.retryBlocked && (
            <button
              ref={startButtonRef}
              type="submit"
              className="btn-primary"
              disabled={formLocked || !state.request.query.trim()}
            >
              {state.phase === 'running'
                ? '执行中...'
                : state.phase === 'error' ? '再次执行当前目标' : '开始执行'}
            </button>
          )}
          {(state.phase === 'done' || state.phase === 'error') && (
            <button
              ref={resetButtonRef}
              type="button"
              className="btn-ghost"
              onClick={handleReset}
            >
              清空并重新开始
            </button>
          )}
        </div>
      </form>

      {/* ── 执行进度 ────────────────────────────────────────────────────── */}
      {r && (
        <div className="auto-result">
          {/* Final answer */}
          {r.final_answer && (
            <section className={`result-section result-final ${isAbstained ? 'result-abstained' : ''}`}>
              <h3>最终回复</h3>
              <FormattedAnswer text={r.final_answer} />
              {groundingStatus === 'citation_ids_valid' && (
                <div className="grounding-banner grounding-valid" role="status">
                  <strong>引用 ID 已核验</strong>
                  <span>
                    下列片段来自本轮工具检索；这不等同于已对回复中的每项事实做语义核验。
                  </span>
                </div>
              )}
              {isAbstained && isSafetyBlocked && (
                <div className="grounding-banner grounding-abstained" role="status">
                  <strong>输出已被安全策略拦截</strong>
                  <span>本次执行没有向界面返回被识别为敏感或不安全的内容。</span>
                </div>
              )}
              {isAbstained && !isSafetyBlocked && (
                <div className="grounding-banner grounding-abstained" role="status">
                  <strong>证据不足，已安全拒答</strong>
                  <span>当前证据没有满足完整引用约束，未返回未经充分支持的答案。</span>
                </div>
              )}
              {invalidCitationCount > 0 && (
                <p className="grounding-note">
                  已丢弃 {invalidCitationCount} 个不属于本轮检索结果的引用 ID。
                </p>
              )}
              {r.finalize_reason && (
                <div className="final-meta">
                  完成说明：<code>{r.finalize_reason}</code>
                </div>
              )}
            </section>
          )}

          {!isAbstained && citations.length > 0 && (
            <section className="result-section citation-section" aria-labelledby="autonomous-citations-title">
              <h3 id="autonomous-citations-title">文档证据</h3>
              <ol className="citation-list">
                {citations.map((citation, index) => (
                  <li className="citation-item" key={`${citation.chunk_id}-${index}`}>
                    <div className="citation-head">
                      <strong>证据 {index + 1}</strong>
                      {Number.isInteger(citation.rank) && (
                        <span>检索排名 {citation.rank}</span>
                      )}
                    </div>
                    <blockquote className="citation-snippet">{citation.snippet}</blockquote>
                    <div className="citation-meta">
                      <span>文档 <code>{citation.document_id}</code></span>
                      {Number.isInteger(citation.chunk_index) && (
                        <span>片段序号 <code>{citation.chunk_index}</code></span>
                      )}
                      <span>片段 ID <code>{citation.chunk_id}</code></span>
                    </div>
                  </li>
                ))}
              </ol>
            </section>
          )}

          <details className="result-details" open={state.phase !== 'done'}>
            <summary>
              <span>执行详情</span>
              <span className="result-details-meta">{r.steps?.length || 0} 步 · {r.rounds_used} 轮</span>
            </summary>
            <div className="result-details-body">
              {r.plan && r.plan.length > 0 && (
                <section className="result-section">
                  <h3>执行计划</h3>
                  <ol className="plan-list">
                    {r.plan.map((step, i) => <li key={i}>{step}</li>)}
                  </ol>
                </section>
              )}

              {r.steps && r.steps.length > 0 && (
                <section className="result-section">
                  <h3>工具步骤</h3>
                  <ul className="step-list">
                    {r.steps.map((s, i) => (
                      <li
                        key={i}
                        className={`step-item ${s.blocked_reason ? 'step-blocked' : ''} ${s.tool_name === 'finalize' ? 'step-finalize' : ''} ${s.tool_name === 'ask_user' ? 'step-ask' : ''}`}
                      >
                        <div className="step-head">
                          <span className="step-round">R{s.round_index + 1}</span>
                          <span className="step-tool">{s.tool_name || '(无)'}</span>
                          {s.blocked_reason && <span className="step-tag tag-blocked">已阻止：{s.blocked_reason}</span>}
                          {s.tool_name === 'finalize' && <span className="step-tag tag-finalize">完成</span>}
                          {s.tool_name === 'ask_user' && <span className="step-tag tag-ask">等待回答</span>}
                        </div>
                        {s.tool_args && Object.keys(s.tool_args).length > 0 && (
                          <pre className="step-args">{JSON.stringify(s.tool_args, null, 2)}</pre>
                        )}
                        {s.observation_preview && <div className="step-obs">{s.observation_preview}</div>}
                      </li>
                    ))}
                  </ul>
                </section>
              )}

              <section className="result-section">
                <h3>运行信息</h3>
                <div className="meta-grid">
                  <div><span className="meta-key">轮次</span><span>{r.rounds_used}</span></div>
                  <div><span className="meta-key">是否截断</span><span>{r.truncated ? '是' : '否'}</span></div>
                  <div><span className="meta-key">工具调用</span><span>{(r.tools_called || []).join(', ') || '无'}</span></div>
                  <div>
                    <span className="meta-key">引用约束</span>
                    <span>{effectiveGroundingRequired ? '已要求' : '未要求'}</span>
                  </div>
                  {r.grounding_document_id && (
                    <div>
                      <span className="meta-key">检索范围</span>
                      <span>{r.grounding_document_id}</span>
                    </div>
                  )}
                </div>
              </section>
            </div>
          </details>
        </div>
      )}

      {/* ── 错误显示 ────────────────────────────────────────────────────── */}
      {state.phase === 'error' && (
        <div className="error-banner" role="alert">⚠️ {state.error}</div>
      )}
      </div>

      {/* ── HITL Modal：ask_user 弹框 ───────────────────────────────────── */}
      {dialogOpen && (
        <div className="modal-overlay" role="presentation">
          <div
            ref={modalRef}
            className="modal-card"
            role="dialog"
            aria-modal="true"
            aria-busy={state.phase === 'continuing'}
            aria-labelledby="autonomous-dialog-title"
            aria-describedby="autonomous-dialog-question"
          >
            <h2 id="autonomous-dialog-title" className="modal-header">Agent 想问你</h2>
            <div id="autonomous-dialog-question" className="modal-question">{r.user_question}</div>
            {groundingStatus === 'pending' && (
              <div className="grounding-pending" role="status">
                本轮已启用引用约束；最终回复会在续跑完成后校验检索片段 ID。
              </div>
            )}
            <textarea
              ref={modalInputRef}
              className="modal-input"
              rows={3}
              value={askReply}
              onChange={e => {
                continueIdempotencyKey.current = null
                const reply = e.target.value
                setAskReply(reply)
                persistCurrentAwaiting(reply, null)
              }}
              placeholder="输入你的回答..."
              aria-label="你的回答"
              readOnly={state.phase === 'continuing'}
              onKeyDown={e => {
                if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) handleContinue()
              }}
            />
            {state.error && (
              <div className="modal-error" role="alert">
                回答提交失败：{state.error}。你的回答已保留，可以重试。
              </div>
            )}
            <div className="modal-actions">
              <button
                type="button"
                className="btn-ghost"
                onClick={handleReset}
                disabled={state.phase === 'continuing'}
              >
                取消整个执行
              </button>
              <button
                type="button"
                className="btn-primary"
                onClick={handleContinue}
                disabled={state.phase === 'continuing' || !askReply.trim()}
              >
                {state.phase === 'continuing'
                  ? '提交中...'
                  : state.error ? '重试回答' : '回答 ⏎ (Ctrl+Enter)'}
              </button>
            </div>
            <div className="modal-meta">
              conversation_id: <code>{lastConversationId.current?.slice(0, 16)}...</code>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
