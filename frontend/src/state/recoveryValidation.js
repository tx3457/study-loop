/**
 * Recovery 模块共用的校验规则。
 *
 * 这些规则必须落在和服务端一样的拒绝边界上。偏松，前端会放行一个必然被 400
 * 挡回来的请求；偏紧，会拦下服务端本来接受的输入。两种都表现为用户看不懂的
 * 失败，而且只在边界值上出现，平时测不出来。
 */

export const DEFAULT_USER_ID = 'default_user'

// 与 services/idempotency.py 的 _KEY_PATTERN 逐字对应。
export const IDEMPOTENCY_KEY_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$/u

export const LEARNING_PATH_ID_PATTERN = /^lp_[0-9a-f]{32}$/u

export function isObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value)
}

export function isIdempotencyKey(value) {
  return typeof value === 'string' && IDEMPOTENCY_KEY_PATTERN.test(value.trim())
}

/**
 * 服务端按 Unicode code point 限长——Pydantic 的 max_length 走 Python len()，
 * 数的是码点，且对未 trim 的原始串计数；非空则是 trim 之后才判断的。这里把两
 * 件事分开算，才和服务端落在同一条边界上。
 *
 * JS 的 .length 数的是 UTF-16 单元，一个 emoji 会被算成两个字符，用它限长会把
 * 服务端愿意接受的内容提前拒掉。
 */
export function boundedText(value, maxLength, { allowBlank = false } = {}) {
  if (typeof value !== 'string') return false
  if (Array.from(value).length > maxLength) return false
  return allowBlank || value.trim().length > 0
}

/** 可缺省的文本：null/undefined 合法，给了值就按同一条长度边界校验。 */
export function optionalText(value, maxLength) {
  return value == null || boundedText(value, maxLength, { allowBlank: true })
}
