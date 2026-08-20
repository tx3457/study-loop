/**
 * 可恢复的自适应学习闭环。
 *
 * 服务端快照是唯一进度真相；sessionStorage 只保留恢复意图、稳定请求键、
 * 最近一次安全快照，以及尚未确认的完整 submit 请求。
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  createIdempotencyKey,
  getAdaptiveSnapshot,
  getDocuments,
  startAdaptive,
  submitAdaptive,
} from '../api/client'
import {
  clearAdaptiveRecovery,
  createAdaptiveRecovery,
  normalizeAdaptiveSnapshot,
  readAdaptiveRecovery,
  sameAdaptiveIntent,
  writeAdaptiveRecovery,
} from '../state/adaptiveRecovery'
import './Adaptive.css'

const ACTION_META = {
  advance: { text: '升难度', cls: 'act-advance' },
  teach: { text: '讲解', cls: 'act-teach' },
  remediate: { text: '补薄弱点', cls: 'act-remediate' },
  continue: { text: '继续巩固', cls: 'act-continue' },
  switch_to_plan: { text: '转学习路径', cls: 'act-plan' },
  finish: { text: '结束', cls: 'act-finish' },
}

class RecoverySupersededError extends Error {}

function ActionBadge({ action }) {
  const meta = ACTION_META[action] || { text: action || '继续', cls: 'act-continue' }
  return <span className={'act-badge ' + meta.cls}>{meta.text}</span>
}

function isSessionGone(error) {
  return error?.status === 404
    || error?.status === 410
    || error?.code === 'adaptive_session_not_found'
    || error?.code === 'adaptive_session_expired'
}

function isBusyError(error) {
  return error?.status === 409 && (
    error?.reason === 'in_progress'
    || error?.code === 'adaptive_session_busy'
  )
}

function isStaleError(error) {
  return error?.status === 409 && (
    error?.reason === 'stale'
    || error?.reason === 'completed'
    || error?.code === 'adaptive_session_stale'
    || error?.code === 'adaptive_session_completed'
  )
}

function isRejectedSubmit(error) {
  return error?.status === 400 || error?.status === 422
}

function safeStages(path) {
  if (!path || !Array.isArray(path.stages)) return []
  return path.stages.filter(stage => (
    stage
    && typeof stage === 'object'
    && typeof stage.title === 'string'
    && stage.title.trim()
  ))
}

function LearningPathSummary({ path, fallbackDocumentId }) {
  const [launch] = useState(() => {
    try {
      return { id: createIdempotencyKey(), error: null }
    } catch (err) {
      return { id: null, error: err.message }
    }
  })
  const stages = safeStages(path)
  const documentId = typeof path?.document_id === 'string' && path.document_id.trim()
    ? path.document_id.trim()
    : fallbackDocumentId
  const firstStage = stages[0]
  const firstTopic = Array.isArray(firstStage?.topics)
    ? firstStage.topics.find(topic => typeof topic === 'string' && topic.trim())
    : null
  const query = new URLSearchParams()
  if (documentId) query.set('document_id', documentId)
  if (firstTopic || firstStage?.title) query.set('topic', firstTopic || firstStage.title)
  if (firstStage && documentId && launch.id) query.set('launch_id', launch.id)
  const quizHref = firstStage && documentId && launch.id
    ? '/quiz?' + query.toString()
    : '/learning-path'
  const totalMinutes = stages.reduce((sum, stage) => (
    Number.isFinite(stage.estimated_minutes)
      ? sum + Math.max(0, stage.estimated_minutes)
      : sum
  ), 0)

  return (
    <div className="learning-path" aria-label="推荐学习路径">
      <h4>{typeof path?.title === 'string' && path.title.trim() ? path.title : '推荐学习路径'}</h4>
      <p className="decision-meta">
        {stages.length > 0 ? '共 ' + stages.length + ' 个阶段' : '学习路径已生成'}
        {totalMinutes > 0 ? ' ｜ 预计 ' + totalMinutes + ' 分钟' : ''}
      </p>
      {stages.length > 0 && (
        <ol>
          {stages.map((stage, index) => (
            <li key={String(stage.stage ?? index) + '-' + stage.title} className="fb-item">
              <b>{stage.title}</b>
              {typeof stage.description === 'string' && stage.description.trim() && (
                <div className="decision-reason">{stage.description}</div>
              )}
              {Array.isArray(stage.topics) && stage.topics.length > 0 && (
                <div className="decision-targets">
                  主题：{stage.topics.filter(topic => typeof topic === 'string').join('、')}
                </div>
              )}
            </li>
          ))}
        </ol>
      )}
      {launch.error && firstStage ? (
        <p className="load-error-state" role="alert">{launch.error}</p>
      ) : (
        <Link className="btn-primary" to={quizHref}>
          {firstStage ? '从第一阶段开始练习' : '前往学习路径'}
        </Link>
      )}
    </div>
  )
}

export default function Adaptive() {
  const [recovery, setRecovery] = useState(() => readAdaptiveRecovery())
  const recoveryRef = useRef(recovery)
  const [recoveryReady, setRecoveryReady] = useState(() => recovery == null)
  const [recoveryAttempt, setRecoveryAttempt] = useState(0)
  const recoveryInFlight = useRef(false)
  const recoveryEpoch = useRef(0)
  const recoveryRetryTimer = useRef(null)
  const mountedRef = useRef(true)

  const [phase, setPhase] = useState(() => recovery ? 'recovering' : 'idle')
  const [req, setReq] = useState(() => recovery?.intent || {
    user_id: 'default_user',
    document_id: '',
    goal: '',
  })
  const [documents, setDocuments] = useState([])
  const [docsLoading, setDocsLoading] = useState(false)
  const [docsError, setDocsError] = useState(null)
  const [resp, setResp] = useState(() => recovery?.snapshot || null)
  const [answers, setAnswers] = useState(() => (
    recovery?.pending_submit?.body.answers
    || new Array(recovery?.snapshot?.questions?.length || 0).fill('')
  ))
  const [error, setError] = useState(null)
  const docsRequestId = useRef(0)

  const busy = ['starting', 'recovering', 'submitting'].includes(phase)
  const pendingLocked = Boolean(recovery?.pending_submit)
  const hasRecovery = Boolean(recovery)

  const persistRecovery = useCallback((value) => {
    if (!mountedRef.current) return null
    const stored = writeAdaptiveRecovery(value)
    if (!stored) return null
    recoveryRef.current = stored
    setRecovery(stored)
    return stored
  }, [])

  const discardRecovery = useCallback(() => {
    if (!mountedRef.current) return
    clearAdaptiveRecovery()
    recoveryRef.current = null
    setRecovery(null)
  }, [])

  const supersedeRequests = useCallback(() => {
    recoveryEpoch.current += 1
    recoveryInFlight.current = false
    clearTimeout(recoveryRetryTimer.current)
    recoveryRetryTimer.current = null
  }, [])

  const loadDocs = useCallback(async () => {
    const requestId = ++docsRequestId.current
    setDocsLoading(true)
    setDocsError(null)
    try {
      const data = await getDocuments()
      if (!mountedRef.current || requestId !== docsRequestId.current) return
      setDocuments(data.documents || [])
    } catch (err) {
      if (!mountedRef.current || requestId !== docsRequestId.current) return
      setDocsError(err.message)
    } finally {
      if (mountedRef.current && requestId === docsRequestId.current) {
        setDocsLoading(false)
      }
    }
  }, [])

  useEffect(() => {
    void loadDocs()
  }, [loadDocs])

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      docsRequestId.current += 1
      clearTimeout(recoveryRetryTimer.current)
      queueMicrotask(() => {
        if (!mountedRef.current) recoveryEpoch.current += 1
      })
    }
  }, [])

  const renderSnapshot = useCallback((snapshot, record, clearError = true) => {
    setReq(record.intent)
    setResp(snapshot)
    if (clearError) setError(null)
    if (snapshot.done) {
      setAnswers([])
      setPhase('done')
    } else if (snapshot.turn_type === 'teach') {
      setAnswers([])
      setPhase('reading')
    } else {
      setAnswers(
        record.pending_submit?.body.answers
        || new Array(snapshot.questions.length).fill(''),
      )
      setPhase('answering')
    }
  }, [])

  const storeSnapshot = useCallback((snapshot, record, pendingSubmit) => {
    const stored = persistRecovery({
      ...record,
      session: { adaptive_session_id: snapshot.adaptive_session_id },
      snapshot,
      pending_submit: pendingSubmit,
    })
    if (!stored) {
      throw new Error('浏览器无法保存自适应学习进度，请检查存储权限后重试')
    }
    renderSnapshot(snapshot, stored)
    setRecoveryReady(true)
    return stored
  }, [persistRecovery, renderSnapshot])

  const recoverAdaptive = useCallback(async (epoch) => {
    const assertCurrent = () => {
      if (!mountedRef.current || epoch !== recoveryEpoch.current) {
        throw new RecoverySupersededError()
      }
    }

    let record = recoveryRef.current
    if (!record) {
      setRecoveryReady(true)
      setPhase('idle')
      return
    }

    if (!record.session) {
      setPhase('starting')
      const started = normalizeAdaptiveSnapshot(await startAdaptive({
        ...record.intent,
        idempotency_key: record.start_idempotency_key,
      }))
      assertCurrent()
      if (!started) throw new Error('服务端返回的自适应学习快照无效，请稍后重试')
      storeSnapshot(started, record, null)
      return
    }

    setPhase('recovering')
    let snapshot = normalizeAdaptiveSnapshot(
      await getAdaptiveSnapshot(record.session.adaptive_session_id),
      record.session.adaptive_session_id,
    )
    assertCurrent()
    if (!snapshot) throw new Error('服务端返回的自适应学习快照无效，请稍后重试')

    record = persistRecovery({ ...record, snapshot })
    if (!record) throw new Error('浏览器无法更新自适应学习进度，请检查存储权限后重试')
    if (snapshot.busy) {
      const busyError = new Error('上一轮仍在处理中，稍后会自动恢复')
      busyError.status = 409
      busyError.code = 'adaptive_session_busy'
      busyError.reason = 'in_progress'
      throw busyError
    }

    const pending = record.pending_submit
    if (pending && (snapshot.done || snapshot.turn > pending.body.turn)) {
      storeSnapshot(snapshot, record, null)
      return
    }
    if (pending && snapshot.turn < pending.body.turn) {
      throw new Error('本地待提交轮次超出服务端进度，请重新开始')
    }
    if (!pending) {
      storeSnapshot(snapshot, record, null)
      return
    }

    renderSnapshot(snapshot, record)
    setPhase('submitting')
    const submitted = normalizeAdaptiveSnapshot(await submitAdaptive({
      ...pending.body,
      idempotency_key: pending.idempotency_key,
    }), record.session.adaptive_session_id)
    assertCurrent()
    if (!submitted) throw new Error('服务端返回的自适应学习快照无效，请稍后重试')
    snapshot = submitted
    storeSnapshot(snapshot, record, null)
  }, [persistRecovery, renderSnapshot, storeSnapshot])

  useEffect(() => {
    if (recoveryReady || recoveryInFlight.current || !recoveryRef.current) return
    void recoveryAttempt
    recoveryInFlight.current = true
    const epoch = ++recoveryEpoch.current

    void recoverAdaptive(epoch).catch(err => {
      if (err instanceof RecoverySupersededError || epoch !== recoveryEpoch.current) return
      const current = recoveryRef.current
      let retryDelay = null

      if (isSessionGone(err)) {
        discardRecovery()
        setResp(null)
        setAnswers([])
        setPhase('idle')
        setRecoveryReady(true)
      } else if (isStaleError(err) && current?.session) {
        // A stale response may arrive after the server already checkpointed
        // this exact pending request but before it advanced the public turn.
        // Keep the original key/body until GET proves the turn is done or has
        // advanced; otherwise a retry with a new key can never resume it.
        if (current.snapshot) renderSnapshot(current.snapshot, current, false)
        setPhase('recovering')
        setRecoveryReady(false)
        retryDelay = 0
      } else if (isRejectedSubmit(err) && current?.pending_submit) {
        const stored = persistRecovery({ ...current, pending_submit: null })
        if (stored) {
          setPhase('recovering')
          setRecoveryReady(false)
          retryDelay = 0
        } else {
          if (current.snapshot) renderSnapshot(current.snapshot, current, false)
          setRecoveryReady(true)
        }
      } else if (isBusyError(err)) {
        if (current?.snapshot) renderSnapshot(current.snapshot, current, false)
        setPhase('recovering')
        setRecoveryReady(false)
        retryDelay = 800
      } else {
        if (current?.snapshot) renderSnapshot(current.snapshot, current, false)
        else setPhase('error')
        setRecoveryReady(true)
      }
      setError(err.message)

      if (retryDelay != null) {
        clearTimeout(recoveryRetryTimer.current)
        recoveryRetryTimer.current = setTimeout(() => {
          setRecoveryAttempt(attempt => attempt + 1)
        }, retryDelay)
      }
    }).finally(() => {
      if (epoch === recoveryEpoch.current) recoveryInFlight.current = false
    })
  }, [
    discardRecovery,
    persistRecovery,
    recoverAdaptive,
    recoveryAttempt,
    recoveryReady,
    renderSnapshot,
  ])

  function retryRecovery() {
    if (!recoveryRef.current) return
    setError(null)
    setRecoveryReady(false)
    setRecoveryAttempt(attempt => attempt + 1)
  }

  function handleStart(event) {
    event.preventDefault()
    const intent = {
      user_id: 'default_user',
      document_id: req.document_id.trim(),
      goal: req.goal.trim(),
    }
    if (!intent.document_id || !intent.goal) return

    supersedeRequests()
    let next
    try {
      const existing = readAdaptiveRecovery()
      next = existing
        && !existing.session
        && sameAdaptiveIntent(existing.intent, intent)
        ? existing
        : createAdaptiveRecovery(intent, createIdempotencyKey())
    } catch (err) {
      setError(err.message)
      setPhase('idle')
      return
    }
    const stored = next && persistRecovery(next)
    if (!stored) {
      setError('浏览器无法保存恢复信息，请检查存储权限后重试')
      setPhase('error')
      return
    }

    setReq(stored.intent)
    setResp(null)
    setAnswers([])
    setError(null)
    setPhase('starting')
    setRecoveryReady(false)
    setRecoveryAttempt(attempt => attempt + 1)
  }

  function queueSubmit(nextAnswers) {
    let record = recoveryRef.current
    if (
      !record?.session
      || !record.snapshot
      || record.snapshot.done
      || record.pending_submit
    ) {
      if (record?.pending_submit) retryRecovery()
      return
    }

    const body = {
      adaptive_session_id: record.session.adaptive_session_id,
      answers: nextAnswers.map(answer => answer.trim()),
      turn: record.snapshot.turn,
      revision: record.snapshot.revision,
    }
    let idempotencyKey
    try {
      idempotencyKey = createIdempotencyKey()
    } catch (err) {
      setError(err.message)
      return
    }
    const pending = { idempotency_key: idempotencyKey, body }
    record = persistRecovery({ ...record, pending_submit: pending })
    if (!record) {
      setError('浏览器无法保存待提交答案，请检查存储权限后重试')
      return
    }

    setAnswers(body.answers)
    setError(null)
    setPhase('submitting')
    setRecoveryReady(false)
    setRecoveryAttempt(attempt => attempt + 1)
  }

  function handleSubmit() {
    if (pendingLocked) {
      retryRecovery()
      return
    }
    if (answers.length === 0 || answers.some(answer => !answer.trim())) return
    queueSubmit(answers)
  }

  function handleContinueLesson() {
    if (pendingLocked) retryRecovery()
    else queueSubmit([])
  }

  function setAnswer(index, value) {
    if (busy || pendingLocked) return
    setAnswers(previous => {
      const next = [...previous]
      next[index] = value
      return next
    })
  }

  function handleReset() {
    supersedeRequests()
    discardRecovery()
    setRecoveryReady(true)
    setReq({ user_id: 'default_user', document_id: '', goal: '' })
    setResp(null)
    setAnswers([])
    setError(null)
    setPhase('idle')
  }

  const r = resp
  const decision = r?.decision
  const formLocked = hasRecovery

  return (
    <div className="adaptive-page">
      <header className="page-header">
        <h1 className="page-title">自适应辅导</h1>
        <p className="page-desc">
          系统会根据每轮答题表现调整难度、补充薄弱点或生成新的学习路径。
        </p>
      </header>

      <form className="adp-form" onSubmit={handleStart}>
        <div className="form-row">
          <label htmlFor="adaptive-goal">学习目标</label>
          <textarea
            id="adaptive-goal"
            rows={2}
            value={req.goal}
            onChange={event => setReq(current => ({ ...current, goal: event.target.value }))}
            placeholder="例如：快速排序与归并排序"
            disabled={formLocked}
          />
        </div>
        <div className="form-row">
          <div className="doc-input">
            <div className="field-label-row">
              <label htmlFor="adaptive-document">文档 ID</label>
              <button
                type="button"
                className="btn-link"
                onClick={loadDocs}
                disabled={docsLoading}
                aria-busy={docsLoading}
              >
                {docsLoading ? '刷新中...' : '刷新文档'}
              </button>
            </div>
            <input
              id="adaptive-document"
              list="adp-doc-list"
              value={req.document_id}
              onChange={event => setReq(current => ({
                ...current,
                document_id: event.target.value,
              }))}
              placeholder="选择已建库的文档"
              disabled={formLocked}
            />
            <datalist id="adp-doc-list">
              {documents.map(document => <option key={document} value={document} />)}
            </datalist>
          </div>
        </div>
        <div className="form-actions">
          {!hasRecovery && (
            <button
              type="submit"
              className="btn-primary"
              disabled={!req.document_id.trim() || !req.goal.trim()}
            >
              开始自适应辅导
            </button>
          )}
          {hasRecovery && (
            <button type="button" className="btn-ghost" onClick={handleReset}>
              重新开始
            </button>
          )}
          {phase === 'starting' && <span className="hint" role="status">正在创建可恢复会话...</span>}
          {phase === 'recovering' && <span className="hint" role="status">正在同步学习进度...</span>}
        </div>
      </form>

      {error && (
        <div className="error-banner" role="alert">
          ⚠️ {error}
          {hasRecovery && recoveryReady && (
            <button type="button" className="btn-ghost" onClick={retryRecovery}>
              重试恢复
            </button>
          )}
        </div>
      )}
      {docsError && <div className="error-banner" role="alert">⚠️ {docsError}</div>}

      {r && r.last_report_score != null && (
        <div className="report-banner">
          上一轮得分 <b>{(r.last_report_score * 100).toFixed(0)}%</b>
          {r.mastery != null && (
            <> ｜ 当前掌握度 <b>{(r.mastery * 100).toFixed(0)}%</b></>
          )}
          {r.last_report_gaps.length > 0 && (
            <span className="gaps">盲点：{r.last_report_gaps.join('、')}</span>
          )}
        </div>
      )}

      {r?.last_report_feedback?.length > 0 && (
        <section className="feedback-section">
          <h3>🧾 上一轮逐题反馈</h3>
          {r.last_report_feedback.map((feedback, index) => (
            <div
              key={feedback.index ?? index}
              className={'fb-item ' + (feedback.is_correct ? 'fb-ok' : 'fb-bad')}
            >
              <div className="fb-head">
                <span className="fb-mark">{feedback.is_correct ? '✓' : '✗'}</span>
                <span className="fb-q">{(feedback.index ?? index) + 1}. {feedback.question}</span>
              </div>
              {!feedback.is_correct && (
                <div className="fb-body">
                  <div className="fb-line">
                    你的答案：<b>{feedback.your_answer}</b> ｜ 正确：<b>{feedback.correct_answer}</b>
                  </div>
                  {feedback.ai_feedback && <div className="fb-ai">{feedback.ai_feedback}</div>}
                  {feedback.knowledge_gap && (
                    <div className="fb-gap">盲点：{feedback.knowledge_gap}</div>
                  )}
                </div>
              )}
            </div>
          ))}
        </section>
      )}

      {decision && (
        <div className="decision-card">
          <div className="decision-head">
            <span className="decision-title">🧠 Agent 决策</span>
            <ActionBadge action={decision.action} />
            <span className="decision-meta">
              主题「{decision.topic}」 ｜ 难度 {(decision.difficulty_score ?? 0).toFixed(2)}
            </span>
          </div>
          {decision.reason && <div className="decision-reason">{decision.reason}</div>}
          {decision.target_weak_points?.length > 0 && (
            <div className="decision-targets">
              针对薄弱点：{decision.target_weak_points.join('、')}
            </div>
          )}
        </div>
      )}

      {r?.turn_type === 'teach' && !r.done && r.lesson && (
        <section className="lesson-section">
          <h3>📖 讲解：{decision?.topic}</h3>
          <div className="lesson-body">{r.lesson}</div>
          <button
            type="button"
            className="btn-primary"
            onClick={handleContinueLesson}
            disabled={busy}
          >
            {phase === 'submitting'
              ? '出题中...'
              : pendingLocked ? '重试本轮提交' : '我懂了，出题验证 ⏎'}
          </button>
        </section>
      )}

      {!r?.done && r?.questions?.length > 0 && (
        <section className="quiz-section">
          <h3>📝 第 {r.turn} 轮 · 共 {r.questions.length} 题</h3>
          {r.questions.map((question, index) => {
            const questionId = 'adaptive-question-' + r.turn + '-' + index
            return (
              <div className="quiz-item" key={question.index}>
                <div id={questionId} className="quiz-q">
                  {index + 1}. {question.question}
                </div>
                {question.options && question.options.length > 0 ? (
                  <div className="quiz-options" role="radiogroup" aria-labelledby={questionId}>
                    {question.options.map((option, optionIndex) => (
                      <label
                        key={optionIndex}
                        className={'opt ' + (answers[index] === option ? 'opt-sel' : '')}
                      >
                        <input
                          type="radio"
                          name={'q-' + r.turn + '-' + index}
                          checked={answers[index] === option}
                          onChange={() => setAnswer(index, option)}
                          disabled={busy || pendingLocked}
                        />
                        {option}
                      </label>
                    ))}
                  </div>
                ) : (
                  <input
                    id={'adaptive-answer-' + r.turn + '-' + index}
                    className="quiz-text"
                    value={answers[index] || ''}
                    onChange={event => setAnswer(index, event.target.value)}
                    placeholder="输入你的答案"
                    disabled={busy || pendingLocked}
                    aria-labelledby={questionId}
                  />
                )}
              </div>
            )
          })}
          <button
            type="button"
            className="btn-primary"
            onClick={handleSubmit}
            disabled={
              busy
              || (!pendingLocked && (
                answers.length === 0
                || answers.some(answer => !answer.trim())
              ))
            }
          >
            {phase === 'submitting'
              ? '批改 + 决策中...'
              : pendingLocked ? '重试本轮提交' : '提交本轮 ⏎'}
          </button>
        </section>
      )}

      {phase === 'done' && r && (
        <section className="done-section">
          <h3>🏁 辅导结束</h3>
          <div className="done-summary">{r.summary}</div>
          {r.learning_path && (
            <LearningPathSummary path={r.learning_path} fallbackDocumentId={req.document_id} />
          )}
        </section>
      )}

      {r?.trajectory?.length > 0 && (
        <section className="traj-section">
          <h3>📈 学习轨迹</h3>
          <table className="traj-table">
            <thead>
              <tr><th>轮</th><th>决策</th><th>主题</th><th>难度</th><th>得分</th><th>掌握度</th></tr>
            </thead>
            <tbody>
              {r.trajectory.map((turn, index) => (
                <tr key={turn.turn ?? index}>
                  <td>T{turn.turn}</td>
                  <td><ActionBadge action={turn.action} /></td>
                  <td>{turn.topic}</td>
                  <td>{(turn.difficulty_score ?? 0).toFixed(2)}</td>
                  <td>{turn.score != null ? (turn.score * 100).toFixed(0) + '%' : '—'}</td>
                  <td>{turn.mastery_after != null ? (turn.mastery_after * 100).toFixed(0) + '%' : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </div>
  )
}
