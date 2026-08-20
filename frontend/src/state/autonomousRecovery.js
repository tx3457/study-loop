export const AUTONOMOUS_RECOVERY_STORAGE_KEY = 'study-loop.autonomous.recovery.v2'
export const LEGACY_AUTONOMOUS_AWAITING_STORAGE_KEY = 'study-loop.autonomous.awaiting.v1'

const SCHEMA_VERSION = 2
const DEFAULT_USER_ID = 'default_user'
const IDEMPOTENCY_KEY_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$/u
const CONVERSATION_ID_PATTERN = /^[A-Za-z0-9_-]{1,128}$/u
const RECOVERY_KINDS = new Set(['pending_start', 'awaiting', 'pending_continue'])

function isObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value)
}

function boundedText(value, maxLength, { allowBlank = false } = {}) {
  return typeof value === 'string'
    && value.length <= maxLength
    && (allowBlank || value.trim().length > 0)
}

function isIdempotencyKey(value) {
  return typeof value === 'string' && IDEMPOTENCY_KEY_PATTERN.test(value.trim())
}

function normalizeRequest(value) {
  if (
    !isObject(value)
    || !boundedText(value.query, 8000)
    || value.user_id !== DEFAULT_USER_ID
    || (
      value.document_id != null
      && !boundedText(value.document_id, 512)
    )
    || typeof value.grounding_required !== 'boolean'
    || (value.grounding_required && value.document_id == null)
  ) {
    return null
  }

  const documentId = value.document_id == null
    ? null
    : value.document_id.trim()
  if (value.grounding_required && !documentId) return null

  return {
    query: value.query.trim(),
    user_id: DEFAULT_USER_ID,
    document_id: documentId,
    grounding_required: value.grounding_required,
  }
}

function normalizeAwaitingFields(value) {
  if (
    !CONVERSATION_ID_PATTERN.test(value.conversation_id || '')
    || !boundedText(value.user_question, 4000)
    || !boundedText(value.draft, 8000, { allowBlank: true })
  ) {
    return null
  }
  return {
    conversation_id: value.conversation_id,
    user_question: value.user_question.trim(),
    draft: value.draft,
  }
}

export function normalizeAutonomousRecovery(value) {
  if (
    !isObject(value)
    || value.schema_version !== SCHEMA_VERSION
    || !RECOVERY_KINDS.has(value.kind)
  ) {
    return null
  }

  const request = normalizeRequest(value.request)
  if (!request) return null

  if (value.kind === 'pending_start') {
    if (!isIdempotencyKey(value.idempotency_key)) return null
    return {
      schema_version: SCHEMA_VERSION,
      kind: 'pending_start',
      request,
      idempotency_key: value.idempotency_key.trim(),
    }
  }

  const awaiting = normalizeAwaitingFields(value)
  if (!awaiting) return null

  if (value.kind === 'awaiting') {
    return {
      schema_version: SCHEMA_VERSION,
      kind: 'awaiting',
      request,
      ...awaiting,
    }
  }

  if (
    !isIdempotencyKey(value.idempotency_key)
    || !isObject(value.body)
    || value.body.conversation_id !== awaiting.conversation_id
    || !boundedText(value.body.user_reply, 8000)
  ) {
    return null
  }

  const userReply = value.body.user_reply.trim()
  return {
    schema_version: SCHEMA_VERSION,
    kind: 'pending_continue',
    request,
    ...awaiting,
    draft: userReply,
    idempotency_key: value.idempotency_key.trim(),
    body: {
      conversation_id: awaiting.conversation_id,
      user_reply: userReply,
    },
  }
}

export function createPendingAutonomousStart(request, idempotencyKey) {
  return normalizeAutonomousRecovery({
    schema_version: SCHEMA_VERSION,
    kind: 'pending_start',
    request,
    idempotency_key: idempotencyKey,
  })
}

export function createAutonomousAwaiting(request, response, draft = '') {
  return normalizeAutonomousRecovery({
    schema_version: SCHEMA_VERSION,
    kind: 'awaiting',
    request,
    conversation_id: response?.conversation_id,
    user_question: response?.user_question,
    draft,
  })
}

export function createPendingAutonomousContinue(awaiting, userReply, idempotencyKey) {
  if (normalizeAutonomousRecovery(awaiting)?.kind !== 'awaiting') return null
  return normalizeAutonomousRecovery({
    ...awaiting,
    kind: 'pending_continue',
    draft: userReply,
    idempotency_key: idempotencyKey,
    body: {
      conversation_id: awaiting.conversation_id,
      user_reply: userReply,
    },
  })
}

export function releasePendingAutonomousContinue(pending) {
  const normalized = normalizeAutonomousRecovery(pending)
  if (normalized?.kind !== 'pending_continue') return null
  return normalizeAutonomousRecovery({
    schema_version: SCHEMA_VERSION,
    kind: 'awaiting',
    request: normalized.request,
    conversation_id: normalized.conversation_id,
    user_question: normalized.user_question,
    draft: normalized.body.user_reply,
  })
}

export function autonomousRecoveryToken(value) {
  const recovery = normalizeAutonomousRecovery(value)
  if (!recovery) return null
  if (recovery.kind === 'pending_start') {
    return `pending_start:${recovery.idempotency_key}`
  }
  if (recovery.kind === 'pending_continue') {
    return `pending_continue:${recovery.idempotency_key}`
  }
  return `awaiting:${recovery.conversation_id}`
}

function readCurrentV2(storage) {
  const raw = storage.getItem(AUTONOMOUS_RECOVERY_STORAGE_KEY)
  if (!raw) return null
  try {
    return normalizeAutonomousRecovery(JSON.parse(raw))
  } catch {
    return null
  }
}

function migrateLegacy(value) {
  if (!isObject(value) || value.version !== 1 || !isObject(value.request)) {
    return null
  }
  const request = normalizeRequest({
    query: value.request.query,
    user_id: value.request.user_id,
    document_id: value.request.document_id?.trim() || null,
    grounding_required: value.request.grounding_required === true,
  })
  if (!request) return null

  const awaiting = createAutonomousAwaiting(request, value, value.draft)
  if (!awaiting) return null
  if (value.continue_idempotency_key == null) return awaiting
  return createPendingAutonomousContinue(
    awaiting,
    value.draft,
    value.continue_idempotency_key,
  )
}

export function readAutonomousRecovery() {
  try {
    const storage = globalThis.sessionStorage
    if (!storage) return null

    const rawV2 = storage.getItem(AUTONOMOUS_RECOVERY_STORAGE_KEY)
    if (rawV2) {
      const recovery = readCurrentV2(storage)
      if (recovery) {
        storage.removeItem(LEGACY_AUTONOMOUS_AWAITING_STORAGE_KEY)
        return recovery
      }
      storage.removeItem(AUTONOMOUS_RECOVERY_STORAGE_KEY)
    }

    const legacyRaw = storage.getItem(LEGACY_AUTONOMOUS_AWAITING_STORAGE_KEY)
    if (!legacyRaw) return null
    let migrated = null
    try {
      migrated = migrateLegacy(JSON.parse(legacyRaw))
    } catch {
      migrated = null
    }
    storage.removeItem(LEGACY_AUTONOMOUS_AWAITING_STORAGE_KEY)
    if (!migrated) return null
    storage.setItem(AUTONOMOUS_RECOVERY_STORAGE_KEY, JSON.stringify(migrated))
    return migrated
  } catch {
    try {
      globalThis.sessionStorage?.removeItem(AUTONOMOUS_RECOVERY_STORAGE_KEY)
      globalThis.sessionStorage?.removeItem(LEGACY_AUTONOMOUS_AWAITING_STORAGE_KEY)
    } catch {
      // Storage can be unavailable in hardened browser contexts.
    }
    return null
  }
}

export function writeAutonomousRecovery(value) {
  const recovery = normalizeAutonomousRecovery(value)
  if (!recovery) return null
  try {
    const storage = globalThis.sessionStorage
    if (!storage) return null
    storage.setItem(AUTONOMOUS_RECOVERY_STORAGE_KEY, JSON.stringify(recovery))
    storage.removeItem(LEGACY_AUTONOMOUS_AWAITING_STORAGE_KEY)
    return recovery
  } catch {
    return null
  }
}

export function replaceAutonomousRecovery(expectedToken, value) {
  const replacement = normalizeAutonomousRecovery(value)
  if (!expectedToken || !replacement) return null
  try {
    const storage = globalThis.sessionStorage
    if (!storage) return null
    const current = readCurrentV2(storage)
    if (autonomousRecoveryToken(current) !== expectedToken) return null
    storage.setItem(AUTONOMOUS_RECOVERY_STORAGE_KEY, JSON.stringify(replacement))
    storage.removeItem(LEGACY_AUTONOMOUS_AWAITING_STORAGE_KEY)
    return replacement
  } catch {
    return null
  }
}

export function clearAutonomousRecovery(expectedToken = null) {
  try {
    const storage = globalThis.sessionStorage
    if (!storage) return false
    if (expectedToken != null) {
      const current = readCurrentV2(storage)
      if (autonomousRecoveryToken(current) !== expectedToken) return false
    }
    storage.removeItem(AUTONOMOUS_RECOVERY_STORAGE_KEY)
    storage.removeItem(LEGACY_AUTONOMOUS_AWAITING_STORAGE_KEY)
    return true
  } catch {
    return false
  }
}
