/**
 * Autonomous Agent 页面
 *
 * 演示真 ReAct + HITL：
 *   - 用户输入学习目标
 *   - Agent 自主决定调工具 / finalize / 问用户
 *   - LLM 调 ask_user → 弹框让用户答 → 提交后调 /continue 续跑
 *
 * 状态机：
 *   idle → running → (awaiting → continuing)* → done | error
 */
import { useEffect, useState, useRef } from 'react'
import {
  createIdempotencyKey,
  runAutonomous,
  continueAutonomous,
  getDocuments,
  isTerminalExecutionError,
} from '../api/client'
import './Autonomous.css'

const initialState = {
  phase: 'idle',     // idle | running | awaiting | continuing | done | error
  request: { query: '', user_id: 'default_user', document_id: '' },
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
      && typeof request.user_id === 'string'
      && request.user_id.length <= 256
      && typeof request.document_id === 'string'
      && request.document_id.length <= 1024
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
        user_id: request.user_id,
        document_id: request.document_id,
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
    },
  } : initialState)
  const [askReply, setAskReply] = useState(restoredAwaiting?.draft || '')
  const [documents, setDocuments] = useState([])
  const [documentsError, setDocumentsError] = useState(null)
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
        user_id: state.request.user_id || 'default_user',
        document_id: state.request.document_id || null,
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

  async function loadDocs() {
    setDocumentsError(null)
    try {
      const data = await getDocuments()
      setDocuments(data.documents || [])
    } catch (err) {
      setDocumentsError(err.message)
    }
  }

  const r = state.response

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
          描述学习目标，Agent 会选择合适的工具，并在需要时向你确认信息。
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

        <div className="form-row form-row-inline">
          <div>
            <label htmlFor="autonomous-user">用户 ID</label>
            <input
              id="autonomous-user"
              value={state.request.user_id}
              onChange={e => {
                startIdempotencyKey.current = null
                setState(s => ({ ...s, request: { ...s.request, user_id: e.target.value } }))
              }}
              disabled={formLocked}
            />
          </div>
          <div className="doc-input">
            <div className="field-label-row">
              <label htmlFor="autonomous-document">文档 ID（可选）</label>
              <button type="button" className="btn-link" onClick={loadDocs}>刷新文档</button>
            </div>
            <input
              id="autonomous-document"
              list="doc-list"
              value={state.request.document_id}
              onChange={e => {
                startIdempotencyKey.current = null
                setState(s => ({ ...s, request: { ...s.request, document_id: e.target.value } }))
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
            <section className="result-section result-final">
              <h3>最终回复</h3>
              <FormattedAnswer text={r.final_answer} />
              {r.finalize_reason && (
                <div className="final-meta">
                  完成说明：<code>{r.finalize_reason}</code>
                </div>
              )}
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
