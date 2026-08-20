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
  cancelAutonomous,
  createIdempotencyKey,
  runAutonomous,
  continueAutonomous,
  getDocuments,
  isTerminalExecutionError,
} from '../api/client'
import {
  autonomousRecoveryToken,
  clearAutonomousRecovery,
  createAutonomousAwaiting,
  createPendingAutonomousContinue,
  createPendingAutonomousStart,
  readAutonomousRecovery,
  releasePendingAutonomousContinue,
  replaceAutonomousRecovery,
  writeAutonomousRecovery,
} from '../state/autonomousRecovery'
import './Autonomous.css'

const DEFAULT_USER_ID = 'default_user'
const AUTONOMOUS_REQUEST_DEADLINE_MS = 120_000

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

class RecoverySupersededError extends Error {}

class InvalidAutonomousResponseError extends Error {
  constructor(message) {
    super(message)
    this.terminal = true
  }
}

function toFormRequest(request) {
  return {
    query: request?.query || '',
    user_id: DEFAULT_USER_ID,
    document_id: request?.document_id || '',
    grounding_required: request?.grounding_required === true,
  }
}

function responseFromRecovery(recovery) {
  if (!recovery || !['awaiting', 'pending_continue'].includes(recovery.kind)) {
    return null
  }
  return {
    awaiting_user_input: true,
    conversation_id: recovery.conversation_id,
    user_question: recovery.user_question,
    rounds_used: 0,
    steps: [],
    tools_called: ['ask_user'],
    truncated: false,
    grounding_status: recovery.request.grounding_required
      ? 'pending'
      : 'not_requested',
    grounding_required: recovery.request.grounding_required,
    grounding_document_id: recovery.request.document_id,
  }
}

function isPendingRecovery(recovery) {
  return ['pending_start', 'pending_continue'].includes(recovery?.kind)
}

function isTerminalRecoveryError(error) {
  return error?.terminal === true
    || error?.status === 404
    || isTerminalExecutionError(error)
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
  const [restoredRecovery] = useState(() => readAutonomousRecovery())
  const [recovery, setRecovery] = useState(restoredRecovery)
  const recoveryRef = useRef(restoredRecovery)
  const [recoveryAttempt, setRecoveryAttempt] = useState(0)
  const recoveryInFlight = useRef(false)
  const recoveryEpoch = useRef(0)
  const mountedRef = useRef(true)
  const activeRequestControllers = useRef(new Set())
  const [state, setState] = useState(() => restoredRecovery ? {
    ...initialState,
    phase: restoredRecovery.kind === 'pending_start'
      ? 'running'
      : restoredRecovery.kind === 'pending_continue' ? 'continuing' : 'awaiting',
    request: toFormRequest(restoredRecovery.request),
    response: responseFromRecovery(restoredRecovery),
  } : initialState)
  const [askReply, setAskReply] = useState(
    restoredRecovery?.kind === 'pending_continue'
      ? restoredRecovery.body.user_reply
      : restoredRecovery?.draft || '',
  )
  const [documents, setDocuments] = useState([])
  const [documentsLoading, setDocumentsLoading] = useState(false)
  const [documentsError, setDocumentsError] = useState(null)
  const documentsRequestId = useRef(0)
  const modalRef = useRef(null)
  const modalInputRef = useRef(null)
  const startButtonRef = useRef(null)
  const resetButtonRef = useRef(null)
  const previousFocusRef = useRef(null)
  const pendingStart = recovery?.kind === 'pending_start'
  const pendingContinue = recovery?.kind === 'pending_continue'
  const dialogOpen = ['awaiting', 'pending_continue'].includes(recovery?.kind)
    && Boolean(state.response?.user_question)
  const operationBusy = ['running', 'continuing', 'canceling'].includes(state.phase)
  const formLocked = operationBusy || Boolean(recovery)

  const adoptRecovery = useCallback((next) => {
    if (!mountedRef.current) return null
    recoveryRef.current = next
    setRecovery(next)
    return next
  }, [])

  const assertCurrentRecovery = useCallback((epoch, expectedToken) => {
    if (
      !mountedRef.current
      || epoch !== recoveryEpoch.current
      || autonomousRecoveryToken(recoveryRef.current) !== expectedToken
      || autonomousRecoveryToken(readAutonomousRecovery()) !== expectedToken
    ) {
      throw new RecoverySupersededError()
    }
  }, [])

  const runWithDeadline = useCallback(async (requestFactory) => {
    const controller = new AbortController()
    activeRequestControllers.current.add(controller)
    let timedOut = false
    const timer = window.setTimeout(() => {
      timedOut = true
      controller.abort()
    }, AUTONOMOUS_REQUEST_DEADLINE_MS)
    try {
      return await requestFactory(controller.signal)
    } catch (err) {
      if (timedOut) {
        const timeoutError = new Error(
          '等待服务端响应超时；请求可能仍在执行，请使用同一请求继续对账',
        )
        timeoutError.code = 'request_timeout'
        throw timeoutError
      }
      throw err
    } finally {
      window.clearTimeout(timer)
      activeRequestControllers.current.delete(controller)
    }
  }, [])

  const loadDocs = useCallback(async () => {
    const requestId = ++documentsRequestId.current
    setDocumentsLoading(true)
    setDocumentsError(null)
    try {
      const data = await getDocuments()
      if (!mountedRef.current || requestId !== documentsRequestId.current) return
      setDocuments(data.documents || [])
    } catch (err) {
      if (!mountedRef.current || requestId !== documentsRequestId.current) return
      setDocumentsError(err.message)
    } finally {
      if (mountedRef.current && requestId === documentsRequestId.current) {
        setDocumentsLoading(false)
      }
    }
  }, [])

  useEffect(() => {
    const requestControllers = activeRequestControllers.current
    mountedRef.current = true
    void loadDocs()
    return () => {
      mountedRef.current = false
      documentsRequestId.current += 1
      queueMicrotask(() => {
        if (!mountedRef.current) {
          requestControllers.forEach(controller => controller.abort())
          requestControllers.clear()
          recoveryEpoch.current += 1
          recoveryInFlight.current = false
        }
      })
    }
  }, [loadDocs])

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

  const acceptPendingResponse = useCallback((resp, record, epoch, expectedToken) => {
    assertCurrentRecovery(epoch, expectedToken)
    if (!resp || typeof resp !== 'object') {
      throw new InvalidAutonomousResponseError(
        '服务端返回的 Autonomous 响应无效，请重新开始',
      )
    }

    if (resp.awaiting_user_input) {
      const next = createAutonomousAwaiting(record.request, resp)
      if (!next) {
        throw new InvalidAutonomousResponseError(
          '服务端返回的暂停会话无效，请重新开始',
        )
      }
      const stored = replaceAutonomousRecovery(expectedToken, next)
      if (!stored) throw new RecoverySupersededError()
      adoptRecovery(stored)
      setAskReply('')
      setState(current => ({
        ...current,
        phase: 'awaiting',
        request: toFormRequest(stored.request),
        response: resp,
        error: null,
        retryBlocked: false,
      }))
      return
    }

    if (!clearAutonomousRecovery(expectedToken)) {
      throw new RecoverySupersededError()
    }
    adoptRecovery(null)
    setAskReply('')
    setState(current => ({
      ...current,
      phase: 'done',
      request: toFormRequest(record.request),
      response: resp,
      error: null,
      retryBlocked: false,
    }))
  }, [adoptRecovery, assertCurrentRecovery])

  const handlePendingError = useCallback((err, record, epoch, expectedToken) => {
    if (err instanceof RecoverySupersededError) return
    try {
      assertCurrentRecovery(epoch, expectedToken)
    } catch (superseded) {
      if (superseded instanceof RecoverySupersededError) return
      throw superseded
    }

    if ([413, 422].includes(err?.status)) {
      if (record.kind === 'pending_continue') {
        const awaiting = releasePendingAutonomousContinue(record)
        const stored = awaiting
          ? replaceAutonomousRecovery(expectedToken, awaiting)
          : null
        if (!stored) return
        adoptRecovery(stored)
        setAskReply(stored.draft)
        setState(current => ({
          ...current,
          phase: 'awaiting',
          response: current.response || responseFromRecovery(stored),
          error: err.message,
          retryBlocked: false,
        }))
        return
      }

      if (!clearAutonomousRecovery(expectedToken)) return
      adoptRecovery(null)
      setState(current => ({
        ...current,
        phase: 'error',
        error: err.message,
        retryBlocked: false,
      }))
      return
    }

    if (isTerminalRecoveryError(err)) {
      if (!clearAutonomousRecovery(expectedToken)) return
      adoptRecovery(null)
      setState(current => ({
        ...current,
        phase: 'error',
        error: record.kind === 'pending_continue'
          ? `回答提交失败：${err.message}`
          : err.message,
        retryBlocked: true,
      }))
      return
    }

    // An in-progress conflict, a transport failure, or a provider/proxy 5xx
    // cannot prove that the server stopped. Keep the exact body/key bound and
    // expose only an exact replay action.
    setState(current => ({
      ...current,
      phase: record.kind === 'pending_continue' ? 'awaiting' : 'error',
      response: current.response || responseFromRecovery(record),
      error: err.message,
      retryBlocked: false,
    }))
  }, [adoptRecovery, assertCurrentRecovery])

  useEffect(() => {
    void recoveryAttempt
    const record = recoveryRef.current
    if (!isPendingRecovery(record) || recoveryInFlight.current) return

    const expectedToken = autonomousRecoveryToken(record)
    const epoch = ++recoveryEpoch.current
    recoveryInFlight.current = true
    setState(current => ({
      ...current,
      phase: record.kind === 'pending_start' ? 'running' : 'continuing',
      request: toFormRequest(record.request),
      response: record.kind === 'pending_continue'
        ? current.response || responseFromRecovery(record)
        : null,
      error: null,
      retryBlocked: false,
    }))

    const pendingRequest = runWithDeadline(signal => (
      record.kind === 'pending_start'
        ? runAutonomous({
            ...record.request,
            idempotency_key: record.idempotency_key,
            signal,
          })
        : continueAutonomous({
            ...record.body,
            idempotency_key: record.idempotency_key,
            signal,
          })
    ))

    void pendingRequest
      .then(resp => acceptPendingResponse(resp, record, epoch, expectedToken))
      .catch(err => handlePendingError(err, record, epoch, expectedToken))
      .finally(() => {
        if (epoch === recoveryEpoch.current) recoveryInFlight.current = false
      })
  }, [acceptPendingResponse, handlePendingError, recoveryAttempt, runWithDeadline])

  function retryRecovery() {
    const current = recoveryRef.current
    if (!isPendingRecovery(current) || recoveryInFlight.current) return
    setState(previous => ({
      ...previous,
      phase: current.kind === 'pending_start' ? 'running' : 'continuing',
      error: null,
    }))
    setRecoveryAttempt(attempt => attempt + 1)
  }

  /** 首次提交：先同步保存 exact body/key，再由恢复 effect 发请求。 */
  function handleStart(event) {
    event.preventDefault()
    if (pendingStart) {
      retryRecovery()
      return
    }
    if (recoveryRef.current || !state.request.query.trim()) return

    const request = {
      query: state.request.query.trim(),
      user_id: DEFAULT_USER_ID,
      document_id: state.request.document_id.trim() || null,
      grounding_required: state.request.grounding_required,
    }
    let pending
    try {
      pending = createPendingAutonomousStart(request, createIdempotencyKey())
    } catch (err) {
      setState(current => ({ ...current, phase: 'error', error: err.message }))
      return
    }
    const stored = pending && writeAutonomousRecovery(pending)
    if (!stored) {
      setState(current => ({
        ...current,
        phase: 'error',
        error: '浏览器无法保存执行恢复信息，请检查存储权限后重试',
        retryBlocked: false,
      }))
      return
    }

    adoptRecovery(stored)
    setState(current => ({
      ...current,
      phase: 'running',
      request: toFormRequest(stored.request),
      response: null,
      error: null,
      retryBlocked: false,
    }))
    setRecoveryAttempt(attempt => attempt + 1)
  }

  /** 提交 ask_user 回答：同样先把不可变 body/key 落盘。 */
  function handleContinue() {
    if (pendingContinue) {
      retryRecovery()
      return
    }
    const awaiting = recoveryRef.current
    if (
      awaiting?.kind !== 'awaiting'
      || recoveryInFlight.current
      || !askReply.trim()
    ) return

    let pending
    try {
      pending = createPendingAutonomousContinue(
        awaiting,
        askReply.trim(),
        createIdempotencyKey(),
      )
    } catch (err) {
      setState(current => ({ ...current, error: err.message }))
      return
    }
    const expectedToken = autonomousRecoveryToken(awaiting)
    const stored = pending
      ? replaceAutonomousRecovery(expectedToken, pending)
      : null
    if (!stored) {
      setState(current => ({
        ...current,
        error: '浏览器无法保存待提交回答，请检查存储权限后重试',
      }))
      return
    }

    adoptRecovery(stored)
    setAskReply(stored.body.user_reply)
    setState(current => ({ ...current, phase: 'continuing', error: null }))
    setRecoveryAttempt(attempt => attempt + 1)
  }

  function handleReplyChange(reply) {
    const awaiting = recoveryRef.current
    if (awaiting?.kind !== 'awaiting') return
    setAskReply(reply)
    const next = createAutonomousAwaiting(awaiting.request, awaiting, reply)
    const stored = next && replaceAutonomousRecovery(
      autonomousRecoveryToken(awaiting),
      next,
    )
    if (stored) {
      adoptRecovery(stored)
    } else {
      setState(current => ({
        ...current,
        error: '浏览器无法保存回答草稿，请检查存储权限后重试',
      }))
    }
  }

  async function handleCancel() {
    const awaiting = recoveryRef.current
    if (awaiting?.kind !== 'awaiting' || recoveryInFlight.current) return
    const expectedToken = autonomousRecoveryToken(awaiting)
    const epoch = ++recoveryEpoch.current
    recoveryInFlight.current = true
    setState(current => ({ ...current, phase: 'canceling', error: null }))

    const finishConfirmedCancel = () => {
      assertCurrentRecovery(epoch, expectedToken)
      if (!clearAutonomousRecovery(expectedToken)) {
        throw new RecoverySupersededError()
      }
      adoptRecovery(null)
      setAskReply('')
      setState(initialState)
    }

    try {
      const result = await runWithDeadline(signal => cancelAutonomous(
        awaiting.conversation_id,
        { signal },
      ))
      assertCurrentRecovery(epoch, expectedToken)
      if (!['canceled', 'missing'].includes(result?.status)) {
        throw new Error('服务端未确认取消结果，请重试')
      }
      finishConfirmedCancel()
    } catch (err) {
      if (err instanceof RecoverySupersededError) return
      if (err?.status === 404) {
        try {
          finishConfirmedCancel()
        } catch (superseded) {
          if (!(superseded instanceof RecoverySupersededError)) throw superseded
        }
        return
      }
      try {
        assertCurrentRecovery(epoch, expectedToken)
      } catch (superseded) {
        if (superseded instanceof RecoverySupersededError) return
        throw superseded
      }
      setState(current => ({
        ...current,
        phase: 'awaiting',
        error: `取消失败：${err.message}`,
      }))
    } finally {
      if (epoch === recoveryEpoch.current) recoveryInFlight.current = false
    }
  }

  function handleReset() {
    if (recoveryRef.current) return
    recoveryEpoch.current += 1
    recoveryInFlight.current = false
    clearAutonomousRecovery()
    setState(initialState)
    setAskReply('')
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
            onChange={e => setState(s => ({
              ...s,
              request: { ...s.request, query: e.target.value },
            }))}
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
              disabled={operationBusy || (
                !pendingStart
                && (Boolean(recovery) || !state.request.query.trim())
              )}
            >
              {state.phase === 'running'
                ? '执行中...'
                : state.phase === 'error' ? '再次执行当前目标' : '开始执行'}
            </button>
          )}
          {!recovery && (state.phase === 'done' || state.phase === 'error') && (
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
            aria-busy={['continuing', 'canceling'].includes(state.phase)}
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
              onChange={e => handleReplyChange(e.target.value)}
              placeholder="输入你的回答..."
              aria-label="你的回答"
              readOnly={pendingContinue || operationBusy}
              onKeyDown={e => {
                if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) handleContinue()
              }}
            />
            {state.error && (
              <div className="modal-error" role="alert">
                {state.error.startsWith('取消失败：')
                  ? `${state.error}。会话与草稿仍已保留。`
                  : `回答提交失败：${state.error}。你的回答已保留，可以重试。`}
              </div>
            )}
            <div className="modal-actions">
              <button
                type="button"
                className="btn-ghost"
                onClick={handleCancel}
                disabled={operationBusy || pendingContinue}
              >
                {state.phase === 'canceling' ? '取消中...' : '取消整个执行'}
              </button>
              <button
                type="button"
                className="btn-primary"
                onClick={handleContinue}
                disabled={operationBusy || !askReply.trim()}
              >
                {state.phase === 'continuing'
                  ? '提交中...'
                  : state.error ? '重试回答' : '回答 ⏎ (Ctrl+Enter)'}
              </button>
            </div>
            <div className="modal-meta">
              conversation_id: <code>{recovery?.conversation_id?.slice(0, 16)}...</code>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
