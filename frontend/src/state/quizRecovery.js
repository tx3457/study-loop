export const QUIZ_RECOVERY_STORAGE_KEY = 'study-loop.quiz.recovery.v1'

const SCHEMA_VERSION = 1
const IDEMPOTENCY_KEY_PATTERN = /^[A-Za-z0-9._:-]{8,128}$/u

function isObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value)
}

function isNonEmptyString(value) {
  return typeof value === 'string' && value.trim().length > 0
}

function isBoundedText(value, maxLength) {
  return isNonEmptyString(value) && value.trim().length <= maxLength
}

function isIdempotencyKey(value) {
  return typeof value === 'string' && IDEMPOTENCY_KEY_PATTERN.test(value.trim())
}

function normalizeLaunchId(value) {
  if (value == null) return null
  return isIdempotencyKey(value) ? value.trim() : null
}

function normalizeLaunchPreset(value) {
  if (value == null) return null
  if (
    !isObject(value)
    || !isBoundedText(value.document_id, 512)
    || typeof value.topic !== 'string'
    || value.topic.trim().length > 4000
  ) {
    return null
  }
  return {
    document_id: value.document_id.trim(),
    topic: value.topic.trim(),
  }
}

function normalizeStandardRequest(value) {
  if (
    !isObject(value)
    || !isBoundedText(value.document_id, 512)
    || typeof value.description !== 'string'
    || value.description.trim().length > 4000
    || !Number.isInteger(value.count)
    || value.count < 1
    || value.count > 10
    || !['easy', 'medium', 'hard'].includes(value.difficulty)
    || !['choice', 'true_false', 'short_answer'].includes(value.type)
    || !isBoundedText(value.user_id, 128)
  ) {
    return null
  }

  return {
    document_id: value.document_id.trim(),
    description: value.description.trim(),
    count: value.count,
    difficulty: value.difficulty,
    type: value.type,
    user_id: value.user_id.trim(),
  }
}

function normalizeWrongQuestionRequest(value) {
  if (
    !isObject(value)
    || !isBoundedText(value.document_id, 512)
    || !isBoundedText(value.user_id, 128)
  ) {
    return null
  }

  return {
    document_id: value.document_id.trim(),
    user_id: value.user_id.trim(),
  }
}

function normalizeIntent(value) {
  if (!isObject(value)) return null

  if (value.kind === 'standard') {
    const request = normalizeStandardRequest(value.request)
    return request ? { kind: 'standard', request } : null
  }

  if (value.kind === 'wrong_question') {
    const request = normalizeWrongQuestionRequest(value.request)
    return request ? { kind: 'wrong_question', request } : null
  }

  return null
}

function normalizeSession(value) {
  if (value == null) return null
  if (
    !isObject(value)
    || !isBoundedText(value.session_id, 256)
    || !Number.isInteger(value.revision)
    || value.revision < 1
    || typeof value.expires_at !== 'number'
    || !Number.isFinite(value.expires_at)
  ) {
    return null
  }

  return {
    session_id: value.session_id.trim(),
    revision: value.revision,
    expires_at: value.expires_at,
  }
}

function normalizePendingAnswer(value) {
  if (value == null) return null
  if (
    !isObject(value)
    || !isIdempotencyKey(value.idempotency_key)
    || !Number.isInteger(value.question_index)
    || value.question_index < 0
    || !isBoundedText(value.answer, 4000)
  ) {
    return null
  }

  return {
    idempotency_key: value.idempotency_key.trim(),
    question_index: value.question_index,
    answer: value.answer.trim(),
  }
}

export function normalizeQuizRecovery(value) {
  if (!isObject(value) || value.schema_version !== SCHEMA_VERSION) return null

  const intent = normalizeIntent(value.intent)
  const session = normalizeSession(value.session)
  const pendingAnswer = normalizePendingAnswer(value.pending_answer)
  const launchId = normalizeLaunchId(value.launch_id)
  const launchPreset = normalizeLaunchPreset(value.launch_preset)
  const acknowledgedAnswerCount = value.acknowledged_answer_count

  if (
    !intent
    || !isIdempotencyKey(value.start_idempotency_key)
    || (value.launch_id != null && !launchId)
    || (value.launch_preset != null && !launchPreset)
    || Boolean(launchId) !== Boolean(launchPreset)
    || !Number.isInteger(acknowledgedAnswerCount)
    || acknowledgedAnswerCount < 0
    || (pendingAnswer && !session)
  ) {
    return null
  }

  return {
    schema_version: SCHEMA_VERSION,
    intent,
    launch_id: launchId,
    launch_preset: launchPreset,
    start_idempotency_key: value.start_idempotency_key.trim(),
    session,
    acknowledged_answer_count: acknowledgedAnswerCount,
    pending_answer: pendingAnswer,
  }
}

export function createStandardQuizRecovery(
  request,
  idempotencyKey,
  launchId = null,
  launchPreset = null,
) {
  return normalizeQuizRecovery({
    schema_version: SCHEMA_VERSION,
    intent: { kind: 'standard', request },
    launch_id: launchId,
    launch_preset: launchPreset,
    start_idempotency_key: idempotencyKey,
    session: null,
    acknowledged_answer_count: 0,
    pending_answer: null,
  })
}

export function createWrongQuestionQuizRecovery(
  documentId,
  idempotencyKey,
  userId = 'default_user',
) {
  return normalizeQuizRecovery({
    schema_version: SCHEMA_VERSION,
    intent: {
      kind: 'wrong_question',
      request: { document_id: documentId, user_id: userId },
    },
    launch_id: null,
    launch_preset: null,
    start_idempotency_key: idempotencyKey,
    session: null,
    acknowledged_answer_count: 0,
    pending_answer: null,
  })
}

export function sameQuizIntent(left, right) {
  const normalizedLeft = normalizeIntent(left)
  const normalizedRight = normalizeIntent(right)
  return Boolean(
    normalizedLeft
    && normalizedRight
    && JSON.stringify(normalizedLeft) === JSON.stringify(normalizedRight)
  )
}

export function readQuizRecovery() {
  try {
    const storage = globalThis.sessionStorage
    if (!storage) return null
    const raw = storage.getItem(QUIZ_RECOVERY_STORAGE_KEY)
    if (!raw) return null
    const recovery = normalizeQuizRecovery(JSON.parse(raw))
    if (recovery) return recovery
    storage.removeItem(QUIZ_RECOVERY_STORAGE_KEY)
  } catch {
    try {
      globalThis.sessionStorage?.removeItem(QUIZ_RECOVERY_STORAGE_KEY)
    } catch {
      // Storage can be unavailable in hardened browser contexts.
    }
  }
  return null
}

export function writeQuizRecovery(value) {
  const recovery = normalizeQuizRecovery(value)
  if (!recovery) return null
  try {
    const storage = globalThis.sessionStorage
    if (!storage) return null
    storage.setItem(
      QUIZ_RECOVERY_STORAGE_KEY,
      JSON.stringify(recovery),
    )
    return recovery
  } catch {
    return null
  }
}

export function clearQuizRecovery() {
  try {
    globalThis.sessionStorage?.removeItem(QUIZ_RECOVERY_STORAGE_KEY)
  } catch {
    // Nothing else can be done when sessionStorage is unavailable.
  }
}
