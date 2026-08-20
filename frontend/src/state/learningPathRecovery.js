export const LEARNING_PATH_RECOVERY_STORAGE_KEY = 'study-loop.learning-path.recovery.v1'

const SCHEMA_VERSION = 1
const DEFAULT_USER_ID = 'default_user'
const IDEMPOTENCY_KEY_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$/u
const PATH_ID_PATTERN = /^lp_[0-9a-f]{32}$/u

function isObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value)
}

function textLength(value) {
  return Array.from(value.trim()).length
}

function boundedText(value, maxLength) {
  return typeof value === 'string'
    && textLength(value) > 0
    && textLength(value) <= maxLength
}

function normalizeIntent(value) {
  if (
    !isObject(value)
    || value.user_id !== DEFAULT_USER_ID
    || !boundedText(value.document_id, 512)
  ) {
    return null
  }
  return {
    user_id: DEFAULT_USER_ID,
    document_id: value.document_id.trim(),
  }
}

function normalizeStage(value, index) {
  if (
    !isObject(value)
    || value.stage !== index + 1
    || !boundedText(value.title, 200)
    || !Array.isArray(value.topics)
    || value.topics.length < 1
    || value.topics.length > 20
    || value.topics.some(topic => !boundedText(topic, 200))
    || !boundedText(value.description, 4000)
    || !Number.isInteger(value.estimated_minutes)
    || value.estimated_minutes < 1
    || value.estimated_minutes > 480
  ) {
    return null
  }
  const topics = value.topics.map(topic => topic.trim())
  if (new Set(topics.map(topic => topic.toLocaleLowerCase())).size !== topics.length) {
    return null
  }
  return {
    stage: value.stage,
    title: value.title.trim(),
    topics,
    description: value.description.trim(),
    estimated_minutes: value.estimated_minutes,
  }
}

function normalizePath(value) {
  if (
    !isObject(value)
    || !boundedText(value.document_id, 512)
    || !boundedText(value.title, 200)
    || !Number.isInteger(value.total_stages)
    || value.total_stages < 1
    || value.total_stages > 12
    || !Array.isArray(value.stages)
    || value.stages.length !== value.total_stages
  ) {
    return null
  }
  const stages = value.stages.map(normalizeStage)
  if (stages.some(stage => stage == null)) return null
  return {
    document_id: value.document_id.trim(),
    title: value.title.trim(),
    total_stages: value.total_stages,
    stages,
  }
}

export function normalizeLearningPathResource(value) {
  if (
    !isObject(value)
    || value.schema_version !== SCHEMA_VERSION
    || typeof value.learning_path_id !== 'string'
    || !PATH_ID_PATTERN.test(value.learning_path_id)
    || value.user_id !== DEFAULT_USER_ID
    || typeof value.created_at !== 'number'
    || !Number.isFinite(value.created_at)
    || value.created_at < 0
    || value.expires_at !== null
  ) {
    return null
  }
  const path = normalizePath(value.path)
  const progress = value.progress
  if (
    !path
    || !isObject(progress)
    || !Number.isInteger(progress.completed_through)
    || progress.completed_through < 0
    || progress.completed_through > path.total_stages
    || !Number.isInteger(progress.revision)
    || progress.revision !== progress.completed_through + 1
  ) return null
  return {
    schema_version: SCHEMA_VERSION,
    learning_path_id: value.learning_path_id,
    user_id: DEFAULT_USER_ID,
    path,
    progress: {
      completed_through: progress.completed_through,
      revision: progress.revision,
    },
    created_at: value.created_at,
    expires_at: null,
  }
}

export function isLearningPathId(value) {
  return typeof value === 'string' && PATH_ID_PATTERN.test(value)
}

export function normalizeLearningPathRecovery(value) {
  if (!isObject(value) || value.schema_version !== SCHEMA_VERSION) return null
  const intent = normalizeIntent(value.intent)
  const key = typeof value.start_idempotency_key === 'string'
    ? value.start_idempotency_key.trim()
    : ''
  const pathId = value.learning_path_id
  if (
    !intent
    || !IDEMPOTENCY_KEY_PATTERN.test(key)
    || (pathId != null && (typeof pathId !== 'string' || !PATH_ID_PATTERN.test(pathId)))
  ) {
    return null
  }
  return {
    schema_version: SCHEMA_VERSION,
    intent,
    start_idempotency_key: key,
    learning_path_id: pathId || null,
  }
}

export function createLearningPathRecovery(documentId, idempotencyKey) {
  return normalizeLearningPathRecovery({
    schema_version: SCHEMA_VERSION,
    intent: { user_id: DEFAULT_USER_ID, document_id: documentId },
    start_idempotency_key: idempotencyKey,
    learning_path_id: null,
  })
}

export function bindLearningPathRecovery(recovery, learningPathId) {
  return normalizeLearningPathRecovery({
    ...recovery,
    learning_path_id: learningPathId,
  })
}

export function readLearningPathRecovery() {
  try {
    const raw = globalThis.sessionStorage?.getItem(LEARNING_PATH_RECOVERY_STORAGE_KEY)
    if (!raw) return null
    const recovery = normalizeLearningPathRecovery(JSON.parse(raw))
    if (recovery) return recovery
    globalThis.sessionStorage?.removeItem(LEARNING_PATH_RECOVERY_STORAGE_KEY)
  } catch {
    try {
      globalThis.sessionStorage?.removeItem(LEARNING_PATH_RECOVERY_STORAGE_KEY)
    } catch {
      // Storage can be unavailable in hardened browser contexts.
    }
  }
  return null
}

export function writeLearningPathRecovery(value) {
  const recovery = normalizeLearningPathRecovery(value)
  if (!recovery) return false
  try {
    globalThis.sessionStorage?.setItem(
      LEARNING_PATH_RECOVERY_STORAGE_KEY,
      JSON.stringify(recovery),
    )
    return globalThis.sessionStorage?.getItem(LEARNING_PATH_RECOVERY_STORAGE_KEY)
      === JSON.stringify(recovery)
  } catch {
    return false
  }
}

export function clearLearningPathRecovery(expectedKey = null) {
  try {
    if (expectedKey) {
      const current = readLearningPathRecovery()
      if (current?.start_idempotency_key !== expectedKey) return false
    }
    globalThis.sessionStorage?.removeItem(LEARNING_PATH_RECOVERY_STORAGE_KEY)
    return true
  } catch {
    return false
  }
}
