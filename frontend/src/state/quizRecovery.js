export const QUIZ_RECOVERY_STORAGE_KEY = 'study-loop.quiz.recovery.v1'

const SCHEMA_VERSION = 1
const IDEMPOTENCY_KEY_PATTERN = /^[A-Za-z0-9._:-]{8,128}$/u
const LEARNING_PATH_ID_PATTERN = /^lp_[0-9a-f]{32}$/u

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

export function normalizeLearningPathSource(value) {
  if (value == null) return null
  if (
    !isObject(value)
    || typeof value.learning_path_id !== 'string'
    || !LEARNING_PATH_ID_PATTERN.test(value.learning_path_id)
    || !Number.isInteger(value.stage_id)
    || value.stage_id < 1
    || value.stage_id > 12
  ) return null
  return {
    learning_path_id: value.learning_path_id,
    stage_id: value.stage_id,
  }
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
  const hasPathBinding = value.path_id != null || value.stage_id != null
  const source = normalizeLearningPathSource(hasPathBinding ? {
    learning_path_id: value.path_id,
    stage_id: value.stage_id,
  } : null)
  if (hasPathBinding && !source) return null
  return {
    document_id: value.document_id.trim(),
    topic: value.topic.trim(),
    ...(source ? {
      path_id: source.learning_path_id,
      stage_id: source.stage_id,
    } : {}),
  }
}

function normalizeStandardRequest(value) {
  const source = normalizeLearningPathSource(value?.learning_path_source)
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
    || (value.learning_path_source != null && !source)
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
    ...(source ? { learning_path_source: source } : {}),
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
  const source = intent?.kind === 'standard'
    ? intent.request.learning_path_source || null
    : null
  const presetSource = launchPreset?.path_id ? {
    learning_path_id: launchPreset.path_id,
    stage_id: launchPreset.stage_id,
  } : null

  if (
    !intent
    || !isIdempotencyKey(value.start_idempotency_key)
    || (value.launch_id != null && !launchId)
    || (value.launch_preset != null && !launchPreset)
    || Boolean(launchId) !== Boolean(launchPreset)
    || Boolean(source) !== Boolean(presetSource)
    || (source && (!launchId || !launchPreset))
    || (
      launchPreset
      && launchPreset.document_id !== intent.request.document_id
    )
    || (
      launchPreset
      && !(
        launchPreset.topic === intent.request.description
        || (
          launchPreset.topic === ''
          && intent.request.description === '全文'
        )
      )
    )
    || (
      source
      && (
        source.learning_path_id !== presetSource.learning_path_id
        || source.stage_id !== presetSource.stage_id
      )
    )
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
