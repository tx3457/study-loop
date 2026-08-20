import { useState, useEffect, useRef, useCallback } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import DocumentPrerequisite from '../components/DocumentPrerequisite'
import {
  createIdempotencyKey,
  generateReport,
  getDocuments,
  getSessionResult,
  getSessionSnapshot,
  gradeSession,
  startSession,
  startWrongQuestionPractice,
  submitAnswer,
} from '../api/client'
import {
  clearQuizRecovery,
  createStandardQuizRecovery,
  readQuizRecovery,
  sameQuizIntent,
  writeQuizRecovery,
} from '../state/quizRecovery'
import './Quiz.css'

/* ═══════════════════════════════════════════════════════════════════
   Quiz Page — 答题界面

   状态机：setup → loading → answering ⇄ feedback → results → grading → report
   ═══════════════════════════════════════════════════════════════════ */

const DIFFICULTY_OPTIONS = [
  { value: 'easy',   label: '基础', color: '#059669' },
  { value: 'medium', label: '进阶', color: '#D97706' },
  { value: 'hard',   label: '挑战', color: '#DC2626' },
]

const TYPE_OPTIONS = [
  { value: 'choice',       label: '选择题' },
  { value: 'true_false',   label: '判断题' },
  { value: 'short_answer', label: '简答题' },
]

const DEFAULT_QUIZ_CONFIG = {
  document_id: '',
  description: '',
  count: 5,
  difficulty: 'medium',
  type: 'choice',
  user_id: 'default_user',
}

const LABELED_OPTION_PATTERN = /^([A-Za-z])(?:\s*[.．、:：)）]\s*|\s+)(.+)$/u
const QUIZ_LAUNCH_ID_PATTERN = /^[A-Za-z0-9._:-]{8,128}$/u

function readQuizPreset(search) {
  const params = new URLSearchParams(search)
  const documentId = (params.get('document_id') || '').trim()
  const topic = (params.get('topic') || '').trim()
  const rawLaunchId = (params.get('launch_id') || '').trim()
  return {
    documentId,
    topic,
    launchId: QUIZ_LAUNCH_ID_PATTERN.test(rawLaunchId) ? rawLaunchId : null,
    invalidLaunchId: Boolean(rawLaunchId) && !QUIZ_LAUNCH_ID_PATTERN.test(rawLaunchId),
    invalidBounds: documentId.length > 512 || topic.length > 4000,
    hasIntent: Boolean(documentId || topic),
  }
}

function presetConflictsWithRecovery(preset, recovery) {
  if (!preset.hasIntent || !recovery) return false
  if (
    recovery.intent.kind !== 'standard'
    || preset.invalidLaunchId
    || preset.invalidBounds
  ) return true

  if (preset.launchId) {
    return (
      !recovery.launch_id
      || preset.launchId !== recovery.launch_id
      || preset.documentId !== recovery.launch_preset?.document_id
      || preset.topic !== recovery.launch_preset?.topic
    )
  }

  const request = recovery.intent.request
  const expectedTopic = request.description === '全文' ? '' : request.description
  return preset.documentId !== request.document_id || preset.topic !== expectedTopic
}

function normalizeText(value) {
  return typeof value === 'string'
    ? value.trim().replace(/\s+/gu, ' ').toLocaleLowerCase()
    : ''
}

function splitOption(value) {
  const raw = typeof value === 'string' ? value.trim() : ''
  const match = raw.match(LABELED_OPTION_PATTERN)
  return {
    value: normalizeText(raw),
    label: match?.[1].toUpperCase() || null,
    text: normalizeText(match?.[2] || raw),
  }
}

function isCorrectOption(option, correctAnswer) {
  const candidate = splitOption(option)
  const expected = splitOption(correctAnswer)
  if (!expected.value) return false
  if (candidate.value === expected.value) return true
  if (candidate.label && /^[A-Za-z]$/.test(expected.value)) {
    return candidate.label === expected.value.toUpperCase()
  }
  if (candidate.label && candidate.text === expected.value) return true
  return Boolean(
    candidate.label
    && expected.label
    && candidate.label === expected.label
    && candidate.text === expected.text
  )
}

class RecoverySupersededError extends Error {}

function assertSessionSnapshot(
  snapshot,
  expectedSessionId,
  expectedDocumentId,
  expectedOrigin,
) {
  const valid = snapshot
    && snapshot.schema_version === 1
    && snapshot.session_id === expectedSessionId
    && snapshot.document_id === expectedDocumentId
    && snapshot.origin === expectedOrigin
    && Number.isInteger(snapshot.revision)
    && snapshot.revision >= 1
    && ['active', 'completed'].includes(snapshot.status)
    && Number.isInteger(snapshot.total)
    && snapshot.total >= 1
    && Number.isInteger(snapshot.answered_count)
    && snapshot.answered_count >= 0
    && snapshot.answered_count <= snapshot.total
    && Array.isArray(snapshot.questions)
    && snapshot.questions.length === snapshot.total
    && typeof snapshot.expires_at === 'number'
    && Number.isFinite(snapshot.expires_at)

  if (!valid) throw new Error('服务端返回的答题快照无效，请稍后重试')
  if (snapshot.status === 'active' && snapshot.answered_count >= snapshot.total) {
    throw new Error('服务端返回的答题进度不一致，请稍后重试')
  }
  if (snapshot.status === 'completed' && snapshot.answered_count !== snapshot.total) {
    throw new Error('服务端返回的答题结果不完整，请稍后重试')
  }
  return snapshot
}

function recoverySessionFromResponse(response) {
  if (!response?.session_id) {
    throw new Error('服务端没有返回可恢复的答题会话，请稍后重试')
  }
  return {
    session_id: response.session_id,
    // 兼容升级前的同源后端；新契约始终返回这两个字段。
    revision: Number.isInteger(response.revision) && response.revision >= 1
      ? response.revision
      : 1,
    expires_at: typeof response.expires_at === 'number' && Number.isFinite(response.expires_at)
      ? response.expires_at
      : Math.floor(Date.now() / 1000) + 3600,
  }
}

function shouldClearQuizSession(error) {
  return error?.status === 404
    || error?.status === 410
    || error?.code === 'quiz_session_corrupt'
    || error?.code === 'side_effect_ambiguous'
    || (
      error?.code === 'idempotency_conflict'
      && error?.reason === 'ambiguous'
    )
}

function isPayloadMismatch(error) {
  return error?.code === 'idempotency_conflict'
    && error?.reason === 'payload_mismatch'
}

function isRecoveryBusyError(error) {
  return error?.status === 409 && (
    error?.reason === 'in_progress'
    || error?.code === 'quiz_session_busy'
  )
}

export default function Quiz() {
  const location = useLocation()
  const navigate = useNavigate()
  const presetHydratedSearch = useRef(null)

  const [recovery, setRecovery] = useState(() => readQuizRecovery())
  const recoveryRef = useRef(recovery)
  const [recoveryReady, setRecoveryReady] = useState(() => recovery == null)
  const [recoveryAttempt, setRecoveryAttempt] = useState(0)
  const recoveryInFlight = useRef(false)
  const recoveryEpoch = useRef(0)
  const recoveryRetryTimer = useRef(null)
  const mountedRef = useRef(true)
  const incomingPreset = readQuizPreset(location.search)
  const presetConflict = presetConflictsWithRecovery(incomingPreset, recovery)

  /* ── 状态 ──────────────────────────────────────────────────────── */
  const [phase, setPhase] = useState(() => recovery ? 'loading' : 'setup') // setup | loading | answering | feedback | results | grading | report
  const [documents, setDocuments] = useState([])
  const [docsLoading, setDocsLoading] = useState(true)
  const [docsError, setDocsError] = useState(null)
  const [error, setError] = useState(null)

  // AI 批改 + 学习报告
  const [gradingReport, setGradingReport] = useState(null)
  const [learningReport, setLearningReport] = useState(null)
  const [grading, setGrading] = useState(false)
  const [reporting, setReporting] = useState(false)
  const gradingRequestEpoch = useRef(0)
  const reportRequestEpoch = useRef(0)
  const gradingInFlight = useRef(false)
  const reportingInFlight = useRef(false)

  // 配置
  const [config, setConfig] = useState(() => ({ ...DEFAULT_QUIZ_CONFIG }))

  // 答题状态
  const [sessionId, setSessionId] = useState(null)
  const [questions, setQuestions] = useState([])
  const [currentIdx, setCurrentIdx] = useState(0)
  const [selectedAnswer, setSelectedAnswer] = useState('')
  const [feedback, setFeedback] = useState(null)
  const [submitting, setSubmitting] = useState(false)
  const [advancing, setAdvancing] = useState(false)
  const advancingInFlight = useRef(false)

  // 结果
  const [result, setResult] = useState(null)

  // 计时器
  const [elapsed, setElapsed] = useState(0)
  const timerRef = useRef(null)

  const persistRecovery = useCallback((value) => {
    if (!mountedRef.current) return null
    const stored = writeQuizRecovery(value)
    if (!stored) return null
    recoveryRef.current = stored
    setRecovery(stored)
    return stored
  }, [])

  const discardRecovery = useCallback(() => {
    if (!mountedRef.current) return
    clearQuizRecovery()
    recoveryRef.current = null
    setRecovery(null)
  }, [])

  /* ── 加载文档列表 ──────────────────────────────────────────────── */
  const loadDocuments = useCallback(async () => {
    setDocsLoading(true)
    setDocsError(null)
    try {
      const data = await getDocuments()
      const nextDocuments = data.documents || []
      setDocuments(nextDocuments)
      setConfig(current => recoveryRef.current || nextDocuments.includes(current.document_id)
        ? current
        : { ...current, document_id: '' })
    } catch (err) {
      setDocsError(err.message)
    } finally {
      setDocsLoading(false)
    }
  }, [])

  useEffect(() => { loadDocuments() }, [loadDocuments])

  /* ── 计时器 ────────────────────────────────────────────────────── */
  const startTimer = useCallback(() => {
    clearInterval(timerRef.current)
    setElapsed(0)
    timerRef.current = setInterval(() => setElapsed(t => t + 1), 1000)
  }, [])

  const stopTimer = useCallback(() => {
    clearInterval(timerRef.current)
    timerRef.current = null
  }, [])

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      clearInterval(timerRef.current)
      clearTimeout(recoveryRetryTimer.current)
      // React StrictMode immediately mounts effects again after its
      // development-only cleanup. Defer epoch invalidation one microtask so
      // that simulated cleanup does not strand a valid recovery in flight,
      // while a real unmount still invalidates every late response.
      queueMicrotask(() => {
        if (mountedRef.current) return
        recoveryEpoch.current += 1
        gradingRequestEpoch.current += 1
        reportRequestEpoch.current += 1
      })
    }
  }, [])

  const hydrateSnapshot = useCallback((snapshot, record) => {
    const acknowledgedCount = Math.min(
      record.acknowledged_answer_count,
      snapshot.answered_count,
    )
    const hasUnacknowledgedFeedback = Boolean(
      snapshot.last_answer_result
      && snapshot.last_answer_index === snapshot.answered_count - 1
      && snapshot.answered_count > acknowledgedCount
    )
    const updatedRecord = {
      ...record,
      session: {
        session_id: snapshot.session_id,
        revision: snapshot.revision,
        expires_at: snapshot.expires_at,
      },
      acknowledged_answer_count: acknowledgedCount,
      pending_answer: null,
    }
    const stored = persistRecovery(updatedRecord)

    gradingRequestEpoch.current += 1
    reportRequestEpoch.current += 1
    gradingInFlight.current = false
    reportingInFlight.current = false
    setGrading(false)
    setReporting(false)
    setSubmitting(false)
    advancingInFlight.current = false
    setAdvancing(false)

    setSessionId(snapshot.session_id)
    setQuestions(snapshot.questions)
    setConfig(current => record.intent.kind === 'standard'
      ? record.intent.request
      : {
          ...current,
          document_id: record.intent.request.document_id,
          description: '错题重练',
          count: snapshot.total,
          user_id: record.intent.request.user_id,
        })
    setResult(snapshot.result)
    setGradingReport(snapshot.grading_report)
    setLearningReport(snapshot.learning_report)

    if (snapshot.learning_report) {
      stopTimer()
      setCurrentIdx(Math.max(snapshot.total - 1, 0))
      setSelectedAnswer('')
      setFeedback(null)
      setPhase('report')
    } else if (snapshot.grading_report) {
      stopTimer()
      setCurrentIdx(Math.max(snapshot.total - 1, 0))
      setSelectedAnswer('')
      setFeedback(null)
      setPhase('grading')
    } else if (hasUnacknowledgedFeedback) {
      startTimer()
      setCurrentIdx(snapshot.last_answer_index)
      setSelectedAnswer(snapshot.last_user_answer || '')
      setFeedback(snapshot.last_answer_result)
      setPhase('feedback')
    } else if (snapshot.status === 'completed') {
      stopTimer()
      setCurrentIdx(Math.max(snapshot.total - 1, 0))
      setSelectedAnswer('')
      setFeedback(null)
      setPhase('results')
    } else {
      startTimer()
      setCurrentIdx(snapshot.answered_count)
      setSelectedAnswer('')
      setFeedback(null)
      setPhase('answering')
    }

    setRecoveryReady(true)
    setError(stored ? null : '浏览器无法更新恢复进度，请不要刷新页面')
  }, [persistRecovery, startTimer, stopTimer])

  const recoverQuiz = useCallback(async (epoch) => {
    const assertCurrent = () => {
      if (epoch !== recoveryEpoch.current) throw new RecoverySupersededError()
    }

    let record = recoveryRef.current
    if (!record) {
      setRecoveryReady(true)
      setPhase('setup')
      return
    }

    if (!record.session) {
      const response = record.intent.kind === 'standard'
        ? await startSession({
            ...record.intent.request,
            idempotency_key: record.start_idempotency_key,
          })
        : await startWrongQuestionPractice(
            record.intent.request.document_id,
            record.intent.request.user_id,
            record.start_idempotency_key,
          )
      assertCurrent()
      record = persistRecovery({
        ...record,
        session: recoverySessionFromResponse(response),
      })
      if (!record) {
        throw new Error('浏览器无法保存答题会话，请检查存储权限后重试')
      }

      const initialSnapshot = assertSessionSnapshot({
        schema_version: 1,
        origin: record.intent.kind === 'standard' ? 'standard' : 'wrong_question',
        session_id: record.session.session_id,
        document_id: record.intent.request.document_id,
        revision: record.session.revision,
        status: 'active',
        total: response.total,
        answered_count: 0,
        questions: response.questions,
        last_answer_index: null,
        last_user_answer: null,
        last_answer_result: null,
        result: null,
        grading_report: null,
        learning_report: null,
        expires_at: record.session.expires_at,
        busy: false,
      },
      record.session.session_id,
      record.intent.request.document_id,
      record.intent.kind === 'standard' ? 'standard' : 'wrong_question')
      hydrateSnapshot(initialSnapshot, record)
      return
    }

    let snapshot = assertSessionSnapshot(
      await getSessionSnapshot(record.session.session_id),
      record.session.session_id,
      record.intent.request.document_id,
      record.intent.kind === 'standard' ? 'standard' : 'wrong_question',
    )
    assertCurrent()
    if (snapshot.busy) {
      const busyError = new Error('答题进度正在同步，稍后会自动重试')
      busyError.status = 409
      busyError.code = 'quiz_session_busy'
      busyError.reason = 'in_progress'
      throw busyError
    }

    const pendingAnswer = record.pending_answer
    if (pendingAnswer && snapshot.answered_count <= pendingAnswer.question_index) {
      if (
        snapshot.status !== 'active'
        || snapshot.answered_count !== pendingAnswer.question_index
      ) {
        throw new Error('待提交答案与服务端进度不一致，请重新加载')
      }

      await submitAnswer(record.session.session_id, {
        answer: pendingAnswer.answer,
        question_index: pendingAnswer.question_index,
        idempotency_key: pendingAnswer.idempotency_key,
      })
      assertCurrent()
      snapshot = assertSessionSnapshot(
        await getSessionSnapshot(record.session.session_id),
        record.session.session_id,
        record.intent.request.document_id,
        record.intent.kind === 'standard' ? 'standard' : 'wrong_question',
      )
      assertCurrent()
      if (snapshot.busy) {
        const busyError = new Error('答案正在同步，稍后会自动重试')
        busyError.status = 409
        busyError.code = 'quiz_session_busy'
        busyError.reason = 'in_progress'
        throw busyError
      }
    }

    if (pendingAnswer && snapshot.answered_count <= pendingAnswer.question_index) {
      throw new Error('答案尚未写入服务端，请重试恢复')
    }

    hydrateSnapshot(snapshot, record)
  }, [hydrateSnapshot, persistRecovery])

  useEffect(() => {
    if (
      presetConflict
      || recoveryReady
      || recoveryInFlight.current
      || !recoveryRef.current
    ) return
    void recoveryAttempt
    recoveryInFlight.current = true
    const epoch = ++recoveryEpoch.current

    void recoverQuiz(epoch).catch(err => {
      if (err instanceof RecoverySupersededError || epoch !== recoveryEpoch.current) return
      const current = recoveryRef.current
      let retryDelay = null
      if (shouldClearQuizSession(err) || (isPayloadMismatch(err) && !current?.session)) {
        discardRecovery()
        setRecoveryReady(true)
        setSessionId(null)
        setQuestions([])
        setPhase('setup')
      } else if (isPayloadMismatch(err) && current?.session) {
        const reset = persistRecovery({ ...current, pending_answer: null })
        setPhase('loading')
        if (reset) retryDelay = 0
      } else {
        setPhase('loading')
      }
      setError(err.message)

      if (isRecoveryBusyError(err)) {
        retryDelay = 800
      }
      if (retryDelay != null) {
        clearTimeout(recoveryRetryTimer.current)
        recoveryRetryTimer.current = setTimeout(
          () => setRecoveryAttempt(attempt => attempt + 1),
          retryDelay,
        )
      }
    }).finally(() => {
      if (epoch === recoveryEpoch.current) recoveryInFlight.current = false
    })
  }, [
    discardRecovery,
    persistRecovery,
    presetConflict,
    recoverQuiz,
    recoveryAttempt,
    recoveryReady,
  ])

  useEffect(() => {
    if (
      !recoveryReady
      || phase !== 'setup'
      || docsLoading
      || docsError
      || presetHydratedSearch.current === location.search
    ) {
      return
    }

    const preset = readQuizPreset(location.search)
    const { documentId, topic } = preset
    presetHydratedSearch.current = location.search

    if (!documentId && !topic) {
      setConfig(current => ({
        ...current,
        document_id: '',
        description: '',
      }))
      setError(null)
      return
    }

    if (preset.invalidLaunchId || preset.invalidBounds) {
      setConfig(current => ({
        ...current,
        document_id: '',
        description: '',
      }))
      setError('练习链接参数无效，请返回原页面重新选择练习')
      return
    }

    if (!documentId || !documents.includes(documentId)) {
      setConfig(current => ({
        ...current,
        document_id: '',
        description: topic,
      }))
      setError('链接中的学习文档不存在或已被删除，请重新选择文档。')
      return
    }

    setConfig(current => ({
      ...current,
      document_id: documentId,
      description: topic,
    }))
    setError(null)
  }, [docsError, docsLoading, documents, location.search, phase, recoveryReady])

  /* ── 开始答题 ──────────────────────────────────────────────────── */
  const handleStart = () => {
    if (!config.document_id) return
    gradingRequestEpoch.current += 1
    reportRequestEpoch.current += 1
    gradingInFlight.current = false
    reportingInFlight.current = false
    setPhase('loading')
    setError(null)
    setGradingReport(null)
    setLearningReport(null)

    let nextRecovery
    try {
      const startRequest = {
        ...config,
        document_id: config.document_id.trim(),
        description: config.description.trim() || '全文',
        user_id: config.user_id.trim(),
      }
      const intent = { kind: 'standard', request: startRequest }
      const existing = readQuizRecovery()
      const preset = readQuizPreset(location.search)
      nextRecovery = existing
        && !existing.session
        && sameQuizIntent(existing.intent, intent)
        ? existing
        : createStandardQuizRecovery(
            startRequest,
            createIdempotencyKey(),
            preset.launchId,
            preset.launchId
              ? { document_id: preset.documentId, topic: preset.topic }
              : null,
          )
    } catch (err) {
      setError(err.message)
      setPhase('setup')
      return
    }
    if (!nextRecovery || !persistRecovery(nextRecovery)) {
      setError('浏览器无法保存恢复进度，请检查存储权限后重试')
      setPhase('setup')
      return
    }

    setSessionId(null)
    setQuestions([])
    setCurrentIdx(0)
    setSelectedAnswer('')
    setFeedback(null)
    setResult(null)
    setRecoveryReady(false)
    setRecoveryAttempt(attempt => attempt + 1)
  }

  /* ── 提交答案 ──────────────────────────────────────────────────── */
  const handleSubmit = async () => {
    const answer = selectedAnswer.trim()
    if (!answer || submitting) return
    const questionIndex = currentIdx
    let record = recoveryRef.current
    if (!record?.session || record.session.session_id !== sessionId) {
      setError('当前答题会话无法恢复，请重新开始')
      return
    }
    const operationEpoch = recoveryEpoch.current

    const pendingAnswer = record.pending_answer || {
      idempotency_key: createIdempotencyKey(),
      question_index: questionIndex,
      answer,
    }
    if (
      pendingAnswer.question_index !== questionIndex
      || pendingAnswer.answer !== answer
    ) {
      setError('上一次答案仍待确认，请先重试原答案')
      return
    }
    record = persistRecovery({ ...record, pending_answer: pendingAnswer })
    if (!record) {
      setError('浏览器无法保存待提交答案，请检查存储权限后重试')
      return
    }

    setSubmitting(true)
    setError(null)

    try {
      const fb = await submitAnswer(sessionId, {
        answer,
        question_index: questionIndex,
        idempotency_key: pendingAnswer.idempotency_key,
      })
      if (
        operationEpoch !== recoveryEpoch.current
        || recoveryRef.current?.session?.session_id !== sessionId
      ) return
      persistRecovery({
        ...record,
        session: {
          session_id: sessionId,
          revision: Number.isInteger(fb.revision) && fb.revision >= 1
            ? fb.revision
            : record.session.revision,
          expires_at: typeof fb.expires_at === 'number' && Number.isFinite(fb.expires_at)
            ? fb.expires_at
            : record.session.expires_at,
        },
        pending_answer: null,
      })
      setFeedback(fb)
      setPhase('feedback')
    } catch (err) {
      if (operationEpoch !== recoveryEpoch.current) return
      if (shouldClearQuizSession(err)) {
        clearSession()
      } else if (isPayloadMismatch(err)) {
        persistRecovery({ ...record, pending_answer: null })
        setPhase('loading')
        setRecoveryReady(false)
        setRecoveryAttempt(attempt => attempt + 1)
      } else if (
        err.status === 409
        && (err.reason === 'stale' || err.code === 'quiz_session_stale' || !err.code)
      ) {
        setPhase('loading')
        setRecoveryReady(false)
        setRecoveryAttempt(attempt => attempt + 1)
      } else {
        setPhase('answering')
      }
      setError(err.message)
    } finally {
      if (operationEpoch === recoveryEpoch.current) setSubmitting(false)
    }
  }

  /* ── 下一题 / 查看结果 ──────────────────────────────────────────── */
  const handleNext = async () => {
    if (advancingInFlight.current) return
    advancingInFlight.current = true
    setAdvancing(true)
    const operationEpoch = recoveryEpoch.current
    const record = recoveryRef.current
    const acknowledgedRecord = record
      ? persistRecovery({
          ...record,
          acknowledged_answer_count: Math.max(
            record.acknowledged_answer_count,
            currentIdx + 1,
          ),
        }) || record
      : null
    if (feedback?.is_last) {
      stopTimer()
      try {
        const res = await getSessionResult(sessionId)
        if (
          operationEpoch !== recoveryEpoch.current
          || recoveryRef.current?.session?.session_id !== sessionId
        ) return
        if (acknowledgedRecord) {
          persistRecovery({
            ...acknowledgedRecord,
            session: {
              session_id: sessionId,
              revision: Number.isInteger(res.revision) && res.revision >= 1
                ? res.revision
                : acknowledgedRecord.session.revision,
              expires_at: typeof res.expires_at === 'number' && Number.isFinite(res.expires_at)
                ? res.expires_at
                : acknowledgedRecord.session.expires_at,
            },
          })
        }
        setResult(res)
        setPhase('results')
        if ((res.pending || 0) > 0) {
          void runGrade(sessionId)
        }
      } catch (err) {
        if (operationEpoch === recoveryEpoch.current) setError(err.message)
      } finally {
        if (operationEpoch === recoveryEpoch.current) {
          advancingInFlight.current = false
          setAdvancing(false)
        }
      }
    } else {
      setCurrentIdx(idx => idx + 1)
      setSelectedAnswer('')
      setFeedback(null)
      setPhase('answering')
      advancingInFlight.current = false
      setAdvancing(false)
    }
  }

  function clearSession() {
    stopTimer()
    clearTimeout(recoveryRetryTimer.current)
    recoveryEpoch.current += 1
    recoveryInFlight.current = false
    discardRecovery()
    setRecoveryReady(true)
    gradingRequestEpoch.current += 1
    reportRequestEpoch.current += 1
    gradingInFlight.current = false
    reportingInFlight.current = false
    setPhase('setup')
    setSessionId(null)
    setQuestions([])
    setCurrentIdx(0)
    setSelectedAnswer('')
    setFeedback(null)
    setSubmitting(false)
    advancingInFlight.current = false
    setAdvancing(false)
    setResult(null)
    setGradingReport(null)
    setLearningReport(null)
    setGrading(false)
    setReporting(false)
    setElapsed(0)
  }

  const continueCurrentQuiz = () => {
    presetHydratedSearch.current = null
    setError(null)
    navigate('/quiz', { replace: true })
  }

  const startIncomingQuiz = () => {
    if (!incomingPresetCanStart) {
      setError('新练习材料尚未通过校验，请先重新加载文档后再试')
      return
    }
    presetHydratedSearch.current = null
    clearSession()
    setConfig({ ...DEFAULT_QUIZ_CONFIG })
    setError(null)
  }

  /* ── 重新开始 ──────────────────────────────────────────────────── */
  const handleRestart = () => {
    clearSession()
    setError(null)
  }

  const handleAnswerChange = (answer) => {
    if (recoveryRef.current?.pending_answer) return
    setSelectedAnswer(answer)
  }

  /* ── AI 批改 ─────────────────────────────────────────────────────── */
  async function runGrade(targetSessionId) {
    if (!targetSessionId || gradingInFlight.current) return
    gradingInFlight.current = true
    const requestEpoch = ++gradingRequestEpoch.current
    setGrading(true)
    setError(null)
    try {
      const report = await gradeSession(targetSessionId)
      if (requestEpoch !== gradingRequestEpoch.current) return
      const record = recoveryRef.current
      if (record?.session?.session_id === targetSessionId) {
        persistRecovery({
          ...record,
          session: {
            session_id: targetSessionId,
            revision: report.revision,
            expires_at: report.expires_at,
          },
        })
      }
      setGradingReport(report)
      setPhase('grading')
    } catch (err) {
      if (requestEpoch !== gradingRequestEpoch.current) return
      if (shouldClearQuizSession(err)) clearSession()
      setError(err.message)
    } finally {
      if (requestEpoch === gradingRequestEpoch.current) {
        gradingInFlight.current = false
        setGrading(false)
      }
    }
  }

  const handleGrade = () => runGrade(sessionId)

  /* ── 学习报告 ───────────────────────────────────────────────────── */
  const handleReport = async () => {
    if (!sessionId || reportingInFlight.current) return
    reportingInFlight.current = true
    const targetSessionId = sessionId
    const requestEpoch = ++reportRequestEpoch.current
    setReporting(true)
    setError(null)
    try {
      const report = await generateReport(targetSessionId)
      if (requestEpoch !== reportRequestEpoch.current) return
      const record = recoveryRef.current
      if (record?.session?.session_id === targetSessionId) {
        persistRecovery({
          ...record,
          session: {
            session_id: targetSessionId,
            revision: report.revision,
            expires_at: report.expires_at,
          },
        })
      }
      setLearningReport(report)
      setPhase('report')
    } catch (err) {
      if (requestEpoch !== reportRequestEpoch.current) return
      if (shouldClearQuizSession(err)) clearSession()
      setError(err.message)
    } finally {
      if (requestEpoch === reportRequestEpoch.current) {
        reportingInFlight.current = false
        setReporting(false)
      }
    }
  }

  /* ── 工具函数 ──────────────────────────────────────────────────── */
  const formatTime = (s) => {
    const m = Math.floor(s / 60)
    const sec = s % 60
    return `${m.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`
  }

  const currentQ = questions[currentIdx]
  const currentQuestionType = currentQ?.type || config.type
  const resultPending = result?.pending || 0
  const resultIncorrect = result?.incorrect
    ?? Math.max((result?.total || 0) - (result?.correct || 0) - resultPending, 0)
  const resultHasFinalScore = result != null && resultPending === 0 && typeof result.score === 'number'
  const pendingAnswerLocked = Boolean(recovery?.pending_answer)
  const currentQuizLabel = recovery
    ? `${recovery.intent.request.document_id} / ${
        recovery.intent.kind === 'standard'
          ? recovery.intent.request.description
          : '错题重练'
      }`
    : ''
  const incomingQuizLabel = `${incomingPreset.documentId || '未指定材料'} / ${
    incomingPreset.topic || '全文'
  }`
  const incomingPresetCanStart = Boolean(
    incomingPreset.hasIntent
    && !incomingPreset.invalidLaunchId
    && !incomingPreset.invalidBounds
    && incomingPreset.documentId
    && !docsLoading
    && !docsError
    && documents.includes(incomingPreset.documentId)
  )
  const incomingPresetStatus = docsLoading
    ? '正在验证新练习材料…'
    : docsError
      ? '暂时无法验证新练习材料；当前进度尚未被删除。'
      : !incomingPreset.documentId || !documents.includes(incomingPreset.documentId)
        ? '新练习材料不存在或已被删除；当前进度尚未被删除。'
        : incomingPreset.invalidLaunchId || incomingPreset.invalidBounds
          ? '新练习链接无效；当前进度尚未被删除。'
          : null

  const retryRecovery = () => {
    setError(null)
    setPhase('loading')
    setRecoveryReady(false)
    setRecoveryAttempt(attempt => attempt + 1)
  }

  return (
    <div className="quiz-page">
      <header className="page-header">
        <h1 className="page-title">答题练习</h1>
        <p className="page-desc">选择文档和参数，AI 根据内容出题；客观题即时反馈，简答题进行语义批改。</p>
      </header>

      {error && (
        <div className="error-banner" role="alert">
          <span>&#9888;</span>
          <span>{error}</span>
          <button className="error-close" aria-label="关闭错误提示" onClick={() => setError(null)}>&times;</button>
        </div>
      )}

      {presetConflict && (
        <section
          className="quiz-intent-conflict"
          aria-labelledby="quiz-intent-conflict-title"
          aria-describedby="quiz-intent-conflict-desc"
        >
          <h2 id="quiz-intent-conflict-title">检测到另一项练习</h2>
          <p id="quiz-intent-conflict-desc">
            浏览器中还有可恢复的练习进度。请选择继续原练习，或明确放弃它并使用刚选择的新目标。
          </p>
          <dl>
            <div>
              <dt>当前进度</dt>
              <dd>{currentQuizLabel}</dd>
            </div>
            <div>
              <dt>新练习</dt>
              <dd>{incomingQuizLabel}</dd>
            </div>
          </dl>
          {incomingPresetStatus && (
            <div className="quiz-intent-validation" role="status">
              <span>{incomingPresetStatus}</span>
              {docsError && (
                <button type="button" className="state-action" onClick={loadDocuments}>
                  重新加载文档
                </button>
              )}
            </div>
          )}
          <div className="quiz-intent-actions">
            <button type="button" className="restart-btn" onClick={continueCurrentQuiz}>
              继续当前练习
            </button>
            <button
              type="button"
              className="start-btn"
              onClick={startIncomingQuiz}
              disabled={!incomingPresetCanStart}
            >
              放弃并开始新练习
            </button>
          </div>
        </section>
      )}

      {/* ══════════════ Setup Phase ══════════════ */}
      {!presetConflict && phase === 'setup' && docsError && (
        <div className="load-error-state" role="alert">
          <p className="state-title">无法加载文档列表</p>
          <p className="state-desc">{docsError}</p>
          <button type="button" className="state-action" onClick={loadDocuments}>
            重新加载文档
          </button>
        </div>
      )}

      {!presetConflict && phase === 'setup' && !docsError && !docsLoading && documents.length === 0 && (
        <DocumentPrerequisite description="开始答题前，需要先上传一份学习材料供系统检索和出题。" />
      )}

      {!presetConflict && phase === 'setup' && !docsError && (docsLoading || documents.length > 0) && (
        <div className="quiz-setup">
          {/* 文档选择 */}
          <div className="setup-field">
            <label className="field-label" htmlFor="quiz-document">学习文档</label>
            <select
              id="quiz-document"
              className="field-select"
              value={config.document_id}
              onChange={e => {
                setConfig(c => ({ ...c, document_id: e.target.value }))
                setError(null)
              }}
              disabled={docsLoading}
            >
              <option value="">{docsLoading ? '加载中...' : '-- 选择文档 --'}</option>
              {documents.map(d => <option key={d} value={d}>{d}</option>)}
            </select>
          </div>

          {/* 出题主题 */}
          <div className="setup-field">
            <label className="field-label" htmlFor="quiz-topic">出题主题 <span className="field-hint">（可选，留空则覆盖全文）</span></label>
            <input
              id="quiz-topic"
              className="field-input"
              type="text"
              placeholder="例如：第三章 向量检索"
              value={config.description}
              onChange={e => setConfig(c => ({ ...c, description: e.target.value }))}
              maxLength={4000}
            />
          </div>

          {/* 题数 */}
          <div className="setup-field">
            <span id="quiz-count-label" className="field-label">题目数量</span>
            <div className="count-selector" role="group" aria-labelledby="quiz-count-label">
              {[3, 5, 8, 10].map(n => (
                <button
                  type="button"
                  key={n}
                  className={`count-btn ${config.count === n ? 'active' : ''}`}
                  aria-pressed={config.count === n}
                  onClick={() => setConfig(c => ({ ...c, count: n }))}
                >
                  {n} 题
                </button>
              ))}
            </div>
          </div>

          {/* 难度 */}
          <div className="setup-field">
            <span id="quiz-difficulty-label" className="field-label">难度</span>
            <div className="difficulty-selector" role="group" aria-labelledby="quiz-difficulty-label">
              {DIFFICULTY_OPTIONS.map(d => (
                <button
                  type="button"
                  key={d.value}
                  className={`diff-btn ${config.difficulty === d.value ? 'active' : ''}`}
                  aria-pressed={config.difficulty === d.value}
                  style={{ '--diff-color': d.color }}
                  onClick={() => setConfig(c => ({ ...c, difficulty: d.value }))}
                >
                  {d.label}
                </button>
              ))}
            </div>
          </div>

          {/* 题型 */}
          <div className="setup-field">
            <span id="quiz-type-label" className="field-label">题型</span>
            <div className="type-selector" role="group" aria-labelledby="quiz-type-label">
              {TYPE_OPTIONS.map(t => (
                <button
                  type="button"
                  key={t.value}
                  className={`type-btn ${config.type === t.value ? 'active' : ''}`}
                  aria-pressed={config.type === t.value}
                  onClick={() => setConfig(c => ({ ...c, type: t.value }))}
                >
                  {t.label}
                </button>
              ))}
            </div>
          </div>

          <button
            type="button"
            className="start-btn"
            onClick={handleStart}
            disabled={docsLoading || !config.document_id}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="5 3 19 12 5 21 5 3"/>
            </svg>
            开始答题
          </button>
        </div>
      )}

      {/* ══════════════ Loading Phase ══════════════ */}
      {!presetConflict && phase === 'loading' && (
        <div className="quiz-loading" role="status" aria-live="polite">
          <div className="loading-spinner" />
          <p className="loading-text">
            {recovery?.session
              ? '正在恢复答题进度...'
              : recovery?.intent.kind === 'wrong_question'
                ? '正在准备错题重练...'
                : 'AI 正在出题...'}
          </p>
          <p className="loading-hint">
            {recovery?.session
              ? '正在与服务端核对最新进度，请稍候'
              : '正在检索文档并生成题目，请稍候'}
          </p>
          {error && recovery && (
            <div className="result-actions">
              <button type="button" className="state-action" onClick={retryRecovery}>
                重试恢复
              </button>
              <button type="button" className="restart-btn" onClick={handleRestart}>
                放弃本次进度
              </button>
            </div>
          )}
        </div>
      )}

      {/* ══════════════ Answering / Feedback Phase ══════════════ */}
      {!presetConflict && (phase === 'answering' || phase === 'feedback') && currentQ && (
        <div className="quiz-active">
          {/* 进度条 + 计时 */}
          <div className="quiz-toolbar">
            <div
              className="progress-bar-wrapper"
              role="progressbar"
              aria-label="答题进度"
              aria-valuemin="0"
              aria-valuemax={questions.length}
              aria-valuenow={currentIdx + (phase === 'feedback' ? 1 : 0)}
            >
              <div
                className="progress-bar-fill"
                style={{ width: `${((currentIdx + (phase === 'feedback' ? 1 : 0)) / questions.length) * 100}%` }}
              />
            </div>
            <div className="toolbar-info">
              <span className="toolbar-progress">
                {currentIdx + 1} / {questions.length}
              </span>
              <span className="toolbar-timer">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                {formatTime(elapsed)}
              </span>
            </div>
          </div>

          {/* 题目卡片 */}
          <div className="question-card" key={currentIdx}>
            <p id={`quiz-question-${currentIdx}`} className="question-text">{currentQ.question}</p>

            {/* 选项列表 */}
            {currentQuestionType !== 'short_answer'
              && currentQ.options
              && currentQ.options.length > 0 && (
              <div className="options-list" role="group" aria-labelledby={`quiz-question-${currentIdx}`}>
                {currentQ.options.map((opt, i) => {
                  const letter = String.fromCharCode(65 + i)
                  const isSelected = selectedAnswer === opt
                  const isCorrect = isCorrectOption(opt, feedback?.correct_answer)
                  let optClass = 'option-item'
                  if (phase === 'feedback') {
                    if (isCorrect) optClass += ' correct'
                    else if (isSelected && !feedback?.correct) optClass += ' wrong'
                    else optClass += ' disabled'
                  } else if (isSelected) {
                    optClass += ' selected'
                  }

                  return (
                    <button
                      type="button"
                      key={i}
                      className={optClass}
                      aria-pressed={isSelected}
                      onClick={() => phase === 'answering' && handleAnswerChange(opt)}
                      disabled={phase === 'feedback' || submitting || pendingAnswerLocked}
                    >
                      <span className="option-letter">{letter}</span>
                      <span className="option-text">{opt}</span>
                      {phase === 'feedback' && isCorrect && (
                        <span className="option-check">&#10003;</span>
                      )}
                      {phase === 'feedback' && isSelected && !feedback?.correct && !isCorrect && (
                        <span className="option-cross">&#10007;</span>
                      )}
                    </button>
                  )
                })}
              </div>
            )}

            {/* 简答题输入 */}
            {currentQuestionType === 'short_answer' && (
              <textarea
                className="short-answer-input"
                aria-labelledby={`quiz-question-${currentIdx}`}
                placeholder="请输入你的答案..."
                value={selectedAnswer}
                onChange={e => handleAnswerChange(e.target.value)}
                disabled={phase === 'feedback' || submitting || pendingAnswerLocked}
                maxLength={4000}
                rows={3}
              />
            )}

            {/* 反馈区域 */}
            {phase === 'feedback' && feedback && (
              <div
                className={`feedback-box ${feedback.evaluation_status === 'pending_ai'
                  ? 'pending'
                  : feedback.correct ? 'correct' : 'wrong'}`}
                aria-live="polite"
              >
                <div className="feedback-header">
                  <span className="feedback-icon">
                    {feedback.evaluation_status === 'pending_ai'
                      ? '✓ 答案已记录'
                      : feedback.correct ? '✓ 正确' : '✗ 错误'}
                  </span>
                </div>
                {feedback.evaluation_status === 'pending_ai' ? (
                  <p className="feedback-explain">简答题需要语义判断，完成本轮后会给出 AI 批改与讲解。</p>
                ) : (
                  <>
                    <p className="feedback-answer">
                      <strong>正确答案：</strong>{feedback.correct_answer}
                    </p>
                    <p className="feedback-explain">{feedback.explanation}</p>
                  </>
                )}
              </div>
            )}

            {/* 操作按钮 */}
            <div className="question-actions">
              {phase === 'answering' ? (
                <button
                  className="submit-btn"
                  onClick={handleSubmit}
                  disabled={!selectedAnswer.trim() || submitting}
                >
                  {submitting ? '提交中...' : '提交答案'}
                </button>
              ) : (
                <button className="next-btn" onClick={handleNext} disabled={advancing}>
                  {advancing
                    ? '处理中...'
                    : feedback?.is_last
                      ? feedback?.evaluation_status === 'pending_ai' ? '开始 AI 批改' : '查看结果'
                      : '下一题'}
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="9 18 15 12 9 6"/></svg>
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ══════════════ Results Phase ══════════════ */}
      {!presetConflict && phase === 'results' && result && (
        <div className="quiz-results">
          {/* 得分卡片 */}
          <div className="result-score-card">
            <div className="score-ring">
              <svg viewBox="0 0 120 120" className="score-svg">
                <circle cx="60" cy="60" r="52" className="ring-bg" />
                <circle
                  cx="60" cy="60" r="52"
                  className="ring-fill"
                  style={{
                    strokeDasharray: `${2 * Math.PI * 52}`,
                    strokeDashoffset: `${2 * Math.PI * 52 * (1 - (resultHasFinalScore ? result.score : 0))}`,
                  }}
                />
              </svg>
              <div className="score-text">
                <span className={`score-number ${resultHasFinalScore ? '' : 'pending'}`}>
                  {resultHasFinalScore ? Math.round(result.score * 100) : '待评'}
                </span>
                {resultHasFinalScore && <span className="score-unit">分</span>}
              </div>
            </div>
            <div className="score-details">
              <div className="score-detail-item">
                <span className="sd-value">{result.correct}</span>
                <span className="sd-label">正确</span>
              </div>
              <div className="score-detail-divider" />
              <div className="score-detail-item">
                <span className="sd-value">{resultIncorrect}</span>
                <span className="sd-label">错误</span>
              </div>
              {resultPending > 0 && (
                <>
                  <div className="score-detail-divider" />
                  <div className="score-detail-item">
                    <span className="sd-value">{resultPending}</span>
                    <span className="sd-label">AI 待批改</span>
                  </div>
                </>
              )}
              <div className="score-detail-divider" />
              <div className="score-detail-item">
                <span className="sd-value">{formatTime(elapsed)}</span>
                <span className="sd-label">用时</span>
              </div>
            </div>
          </div>

          <div className="result-actions">
            <button className="grade-btn" onClick={handleGrade} disabled={grading}>
              {grading ? (
                <><span className="btn-spinner" />批改中...</>
              ) : (
                <>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>
                  </svg>
                  {resultPending > 0 ? '重试 AI 批改' : 'AI 批改讲解'}
                </>
              )}
            </button>
            <button className="restart-btn" onClick={handleRestart}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>
              </svg>
              再来一轮
            </button>
          </div>
        </div>
      )}

      {/* ══════════════ Grading Phase ══════════════ */}
      {!presetConflict && phase === 'grading' && gradingReport && (
        <div className="quiz-grading">
          <div className="grading-header">
            <h2 className="grading-title">AI 批改报告</h2>
            <div className="grading-summary">
              <span className="gs-score">{Math.round(gradingReport.score * 100)} 分</span>
              <span className="gs-detail">{gradingReport.correct}/{gradingReport.total} 正确</span>
            </div>
          </div>

          <div className="grade-list">
            {gradingReport.grades.map((g, idx) => (
              <div
                key={idx}
                className={`grade-item ${g.is_correct ? 'correct' : 'wrong'}`}
                style={{ animationDelay: `${idx * 0.06}s` }}
              >
                <div className="gi-header">
                  <span className={`gi-badge ${g.is_correct ? 'correct' : 'wrong'}`}>
                    {g.is_correct ? '✓ 正确' : '✗ 错误'}
                  </span>
                  <span className="gi-index">第 {g.index + 1} 题</span>
                </div>

                <p className="gi-question">{g.question}</p>

                <div className="gi-answers">
                  <div className={`gi-answer ${g.is_correct ? 'correct' : 'wrong'}`}>
                    <span className="gi-label">你的答案</span>
                    <span>{g.user_answer}</span>
                  </div>
                  {!g.is_correct && (
                    <div className="gi-answer correct">
                      <span className="gi-label">
                        {questions[g.index]?.type === 'short_answer' ? '参考答案' : '正确答案'}
                      </span>
                      <span>{g.correct_answer}</span>
                    </div>
                  )}
                </div>

                {g.ai_feedback && (
                  <div className="gi-feedback">
                    <strong>AI 讲解：</strong>{g.ai_feedback}
                  </div>
                )}

                {g.knowledge_gap && (
                  <span className="gi-gap">
                    知识盲点：{g.knowledge_gap}
                  </span>
                )}
              </div>
            ))}
          </div>

          <div className="result-actions">
            <button className="report-btn" onClick={handleReport} disabled={reporting}>
              {reporting ? (
                <><span className="btn-spinner" />生成中...</>
              ) : (
                <>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
                  </svg>
                  学习评估报告
                </>
              )}
            </button>
            <button className="restart-btn" onClick={handleRestart}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>
              </svg>
              再来一轮
            </button>
          </div>
        </div>
      )}

      {/* ══════════════ Report Phase ══════════════ */}
      {!presetConflict && phase === 'report' && learningReport && (
        <div className="quiz-report">
          <div className="report-header-card">
            <h2 className="report-title">学习评估报告</h2>
            <p className="report-summary">{learningReport.summary}</p>
            <div className="report-score">
              总分：<strong>{Math.round(learningReport.overall_score * 100)}</strong> / 100
            </div>
          </div>

          {/* 知识点掌握度 */}
          {learningReport.topic_mastery.length > 0 && (
            <div className="report-section">
              <h3 className="rs-title">知识点掌握度</h3>
              <div className="mastery-bars">
                {learningReport.topic_mastery.map((tm, i) => (
                  <div key={i} className="mastery-item" style={{ animationDelay: `${i * 0.08}s` }}>
                    <div className="mi-header">
                      <span className="mi-topic">{tm.topic}</span>
                      <span className="mi-pct">{Math.round(tm.mastery_pct)}%</span>
                    </div>
                    <div className="mi-bar-bg">
                      <div
                        className={`mi-bar-fill ${tm.mastery_pct >= 70 ? 'good' : 'weak'}`}
                        style={{ width: `${Math.min(tm.mastery_pct, 100)}%` }}
                      />
                    </div>
                    <span className="mi-detail">{tm.correct_count}/{tm.question_count} 正确</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* 优势 / 薄弱 */}
          <div className="report-row">
            {learningReport.strengths.length > 0 && (
              <div className="report-section half">
                <h3 className="rs-title strengths">&#10003; 掌握良好</h3>
                <div className="tag-list">
                  {learningReport.strengths.map((s, i) => (
                    <span key={i} className="tag good">{s}</span>
                  ))}
                </div>
              </div>
            )}
            {learningReport.weaknesses.length > 0 && (
              <div className="report-section half">
                <h3 className="rs-title weaknesses">&#9888; 需要加强</h3>
                <div className="tag-list">
                  {learningReport.weaknesses.map((w, i) => (
                    <span key={i} className="tag weak">{w}</span>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* 建议 */}
          {learningReport.recommendations.length > 0 && (
            <div className="report-section">
              <h3 className="rs-title">&#128161; 学习建议</h3>
              <ol className="rec-list">
                {learningReport.recommendations.map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ol>
            </div>
          )}

          <button className="restart-btn center" onClick={handleRestart}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>
            </svg>
            再来一轮
          </button>
        </div>
      )}
    </div>
  )
}
