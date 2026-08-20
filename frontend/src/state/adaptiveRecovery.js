export const ADAPTIVE_RECOVERY_STORAGE_KEY = 'study-loop.adaptive.recovery.v1'

const SCHEMA_VERSION = 1
const DEFAULT_USER_ID = 'default_user'
const IDEMPOTENCY_KEY_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$/u
const QUESTION_TYPES = new Set(['choice', 'true_false', 'short_answer'])
const ACTIONS = new Set([
  'advance',
  'teach',
  'remediate',
  'continue',
  'switch_to_plan',
  'finish',
])
const TERMINATE_REASONS = new Set([
  'agent_finish',
  'mastery_reached',
  'max_turns',
  'switch_to_plan',
])

function isObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value)
}

function boundedText(value, maxLength) {
  return typeof value === 'string'
    && value.trim().length > 0
    && value.trim().length <= maxLength
}

function isIdempotencyKey(value) {
  return typeof value === 'string' && IDEMPOTENCY_KEY_PATTERN.test(value.trim())
}

function finiteOptionalScore(value) {
  return value == null || (
    typeof value === 'number'
    && Number.isFinite(value)
    && value >= 0
    && value <= 1
  )
}

function optionalText(value, maxLength) {
  return value == null || (
    typeof value === 'string'
    && value.length <= maxLength
  )
}

function jsonCopy(value) {
  try {
    const encoded = JSON.stringify(value)
    // The durable aggregate has a 2 MiB encoded payload limit. UTF-8 bytes are
    // never fewer than JSON string code units, so this cannot reject a payload
    // that the server Store accepted merely because of a narrower Web cap.
    if (encoded.length > 2_097_152) return null
    return JSON.parse(encoded)
  } catch {
    return null
  }
}

function normalizeIntent(value) {
  if (
    !isObject(value)
    || value.user_id !== DEFAULT_USER_ID
    || !boundedText(value.document_id, 512)
    || !boundedText(value.goal, 4000)
  ) {
    return null
  }
  return {
    user_id: DEFAULT_USER_ID,
    document_id: value.document_id.trim(),
    goal: value.goal.trim(),
  }
}

function normalizeQuestion(value, index) {
  if (
    !isObject(value)
    || value.index !== index
    || !boundedText(value.question, 4000)
    || !QUESTION_TYPES.has(value.type)
    || (
      value.options != null
      && (
        !Array.isArray(value.options)
        || value.options.length > 20
        || value.options.some(option => !boundedText(option, 4000))
      )
    )
  ) {
    return null
  }
  return {
    index,
    question: value.question.trim(),
    options: value.options == null
      ? null
      : value.options.map(option => option.trim()),
    type: value.type,
  }
}

function normalizeDecision(value) {
  if (
    !isObject(value)
    || !ACTIONS.has(value.action)
    || !boundedText(value.topic, 4000)
    || typeof value.difficulty_score !== 'number'
    || !Number.isFinite(value.difficulty_score)
    || value.difficulty_score < 0
    || value.difficulty_score > 1
    || !optionalText(value.reason, 8000)
    || !Array.isArray(value.target_weak_points)
    || value.target_weak_points.length > 100
    || value.target_weak_points.some(point => !boundedText(point, 2000))
  ) {
    return null
  }
  return {
    action: value.action,
    topic: value.topic.trim(),
    difficulty_score: value.difficulty_score,
    reason: value.reason || '',
    target_weak_points: value.target_weak_points.map(point => point.trim()),
  }
}

function normalizeFeedback(value) {
  if (
    !isObject(value)
    || !Number.isInteger(value.index)
    || value.index < 0
    || value.index > 9
    || !boundedText(value.question, 4000)
    || !boundedText(value.your_answer, 4000)
    || !boundedText(value.correct_answer, 4000)
    || typeof value.is_correct !== 'boolean'
    || !optionalText(value.ai_feedback, 8000)
    || !optionalText(value.knowledge_gap, 2000)
  ) {
    return null
  }
  return {
    index: value.index,
    question: value.question.trim(),
    your_answer: value.your_answer.trim(),
    correct_answer: value.correct_answer.trim(),
    is_correct: value.is_correct,
    ai_feedback: value.ai_feedback || null,
    knowledge_gap: value.knowledge_gap || null,
  }
}

function normalizeTrajectoryTurn(value) {
  if (
    !isObject(value)
    || !Number.isInteger(value.turn)
    || value.turn < 1
    || value.turn > 100
    || !ACTIONS.has(value.action)
    || !boundedText(value.topic, 4000)
    || typeof value.difficulty_score !== 'number'
    || !Number.isFinite(value.difficulty_score)
    || value.difficulty_score < 0
    || value.difficulty_score > 1
    || !optionalText(value.reason, 8000)
    || !finiteOptionalScore(value.score)
    || !finiteOptionalScore(value.mastery_after)
    || !Array.isArray(value.knowledge_gaps)
    || value.knowledge_gaps.length > 100
    || value.knowledge_gaps.some(gap => !boundedText(gap, 2000))
  ) {
    return null
  }
  return {
    turn: value.turn,
    action: value.action,
    topic: value.topic.trim(),
    difficulty_score: value.difficulty_score,
    reason: value.reason || '',
    score: value.score ?? null,
    mastery_after: value.mastery_after ?? null,
    knowledge_gaps: value.knowledge_gaps.map(gap => gap.trim()),
  }
}

export function normalizeAdaptiveSnapshot(value, expectedSessionId = null) {
  if (
    !isObject(value)
    || value.schema_version !== 1
    || !boundedText(value.adaptive_session_id, 128)
    || (
      expectedSessionId != null
      && value.adaptive_session_id !== expectedSessionId
    )
    || !Number.isInteger(value.turn)
    || value.turn < 1
    || value.turn > 100
    || typeof value.done !== 'boolean'
    || !['quiz', 'teach'].includes(value.turn_type)
    || !Array.isArray(value.questions)
    || value.questions.length > 10
    || !isObject(value.decision)
    || !Array.isArray(value.last_report_gaps)
    || value.last_report_gaps.length > 100
    || value.last_report_gaps.some(gap => !boundedText(gap, 2000))
    || !Array.isArray(value.last_report_feedback)
    || value.last_report_feedback.length > 10
    || !finiteOptionalScore(value.last_report_score)
    || !finiteOptionalScore(value.mastery)
    || !Array.isArray(value.trajectory)
    || value.trajectory.length === 0
    || value.trajectory.length > 100
    || (value.lesson != null && typeof value.lesson !== 'string')
    || typeof value.summary !== 'string'
    || value.summary.length > 40000
    || typeof value.terminate_reason !== 'string'
    || value.terminate_reason.length > 128
    || (value.learning_path != null && !isObject(value.learning_path))
    || !Number.isInteger(value.revision)
    || value.revision < 1
    || typeof value.expires_at !== 'number'
    || !Number.isFinite(value.expires_at)
    || value.expires_at <= 0
    || typeof value.busy !== 'boolean'
  ) {
    return null
  }

  const questions = value.questions.map(normalizeQuestion)
  if (questions.some(question => question == null)) return null
  const decision = normalizeDecision(value.decision)
  if (!decision) return null
  const feedback = value.last_report_feedback.map(normalizeFeedback)
  if (
    feedback.some(item => item == null)
    || new Set(feedback.map(item => item.index)).size !== feedback.length
  ) {
    return null
  }
  const trajectory = value.trajectory.map(normalizeTrajectoryTurn)
  if (
    trajectory.some(turn => turn == null)
    || trajectory.length !== value.turn
    || trajectory.some((turn, index) => turn.turn !== index + 1)
  ) {
    return null
  }
  const lesson = value.lesson == null ? null : value.lesson.trim()
  if (
    (value.done && (questions.length > 0 || lesson))
    || (
      !value.done
      && value.turn_type === 'quiz'
      && (questions.length === 0 || lesson)
    )
    || (
      !value.done
      && value.turn_type === 'teach'
      && (questions.length > 0 || !boundedText(lesson, 40000))
    )
  ) {
    return null
  }

  const learningPath = jsonCopy(value.learning_path)
  if (value.learning_path != null && !learningPath) return null
  if (
    (
      value.done
      && (
        !boundedText(value.summary, 40000)
        || !TERMINATE_REASONS.has(value.terminate_reason)
        || ((decision.action === 'finish') !== (value.terminate_reason === 'agent_finish'))
        || (
          (decision.action === 'switch_to_plan')
          !== (value.terminate_reason === 'switch_to_plan')
        )
        || ((learningPath != null) !== (value.terminate_reason === 'switch_to_plan'))
      )
    )
    || (
      !value.done
      && (
        value.summary !== ''
        || value.terminate_reason !== ''
        || learningPath != null
        || ['finish', 'switch_to_plan'].includes(decision.action)
      )
    )
  ) {
    return null
  }

  return {
    schema_version: 1,
    adaptive_session_id: value.adaptive_session_id.trim(),
    turn: value.turn,
    done: value.done,
    turn_type: value.turn_type,
    questions,
    lesson,
    decision,
    last_report_score: value.last_report_score ?? null,
    last_report_gaps: value.last_report_gaps.map(gap => gap.trim()),
    last_report_feedback: feedback,
    mastery: value.mastery ?? null,
    trajectory,
    summary: value.summary,
    terminate_reason: value.terminate_reason,
    learning_path: learningPath,
    revision: value.revision,
    expires_at: value.expires_at,
    busy: value.busy,
  }
}

function normalizeSession(value) {
  if (
    !isObject(value)
    || !boundedText(value.adaptive_session_id, 128)
  ) {
    return null
  }
  return { adaptive_session_id: value.adaptive_session_id.trim() }
}

function normalizePendingSubmit(value, sessionId) {
  if (value == null) return null
  if (
    !isObject(value)
    || !isIdempotencyKey(value.idempotency_key)
    || !isObject(value.body)
    || value.body.adaptive_session_id !== sessionId
    || !Number.isInteger(value.body.turn)
    || value.body.turn < 1
    || value.body.turn > 100
    || !Number.isInteger(value.body.revision)
    || value.body.revision < 1
    || !Array.isArray(value.body.answers)
    || value.body.answers.length > 10
    || value.body.answers.some(answer => !boundedText(answer, 4000))
  ) {
    return null
  }
  return {
    idempotency_key: value.idempotency_key.trim(),
    body: {
      adaptive_session_id: sessionId,
      answers: value.body.answers.map(answer => answer.trim()),
      turn: value.body.turn,
      revision: value.body.revision,
    },
  }
}

export function normalizeAdaptiveRecovery(value) {
  if (
    !isObject(value)
    || value.schema_version !== SCHEMA_VERSION
    || !isIdempotencyKey(value.start_idempotency_key)
  ) {
    return null
  }

  const intent = normalizeIntent(value.intent)
  if (!intent) return null

  if (value.session == null && value.snapshot == null) {
    if (value.pending_submit != null) return null
    return {
      schema_version: SCHEMA_VERSION,
      intent,
      start_idempotency_key: value.start_idempotency_key.trim(),
      session: null,
      snapshot: null,
      pending_submit: null,
    }
  }

  const session = normalizeSession(value.session)
  if (!session) return null
  const snapshot = normalizeAdaptiveSnapshot(
    value.snapshot,
    session.adaptive_session_id,
  )
  if (!snapshot) return null
  const pendingSubmit = normalizePendingSubmit(
    value.pending_submit,
    session.adaptive_session_id,
  )
  if (snapshot.done && pendingSubmit) return null

  return {
    schema_version: SCHEMA_VERSION,
    intent,
    start_idempotency_key: value.start_idempotency_key.trim(),
    session,
    snapshot,
    pending_submit: pendingSubmit,
  }
}

export function createAdaptiveRecovery(intent, idempotencyKey) {
  return normalizeAdaptiveRecovery({
    schema_version: SCHEMA_VERSION,
    intent: {
      user_id: DEFAULT_USER_ID,
      document_id: intent?.document_id,
      goal: intent?.goal,
    },
    start_idempotency_key: idempotencyKey,
    session: null,
    snapshot: null,
    pending_submit: null,
  })
}

export function sameAdaptiveIntent(left, right) {
  const normalizedLeft = normalizeIntent(left)
  const normalizedRight = normalizeIntent(right)
  return Boolean(
    normalizedLeft
    && normalizedRight
    && JSON.stringify(normalizedLeft) === JSON.stringify(normalizedRight)
  )
}

export function readAdaptiveRecovery() {
  try {
    const storage = globalThis.sessionStorage
    if (!storage) return null
    const raw = storage.getItem(ADAPTIVE_RECOVERY_STORAGE_KEY)
    if (!raw) return null
    const recovery = normalizeAdaptiveRecovery(JSON.parse(raw))
    if (recovery) return recovery
    storage.removeItem(ADAPTIVE_RECOVERY_STORAGE_KEY)
  } catch {
    try {
      globalThis.sessionStorage?.removeItem(ADAPTIVE_RECOVERY_STORAGE_KEY)
    } catch {
      // Storage can be unavailable in hardened browser contexts.
    }
  }
  return null
}

export function writeAdaptiveRecovery(value) {
  const recovery = normalizeAdaptiveRecovery(value)
  if (!recovery) return null
  try {
    const storage = globalThis.sessionStorage
    if (!storage) return null
    storage.setItem(
      ADAPTIVE_RECOVERY_STORAGE_KEY,
      JSON.stringify(recovery),
    )
    return recovery
  } catch {
    return null
  }
}

export function clearAdaptiveRecovery() {
  try {
    globalThis.sessionStorage?.removeItem(ADAPTIVE_RECOVERY_STORAGE_KEY)
  } catch {
    // Nothing else can be done when sessionStorage is unavailable.
  }
}
