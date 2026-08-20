const DEFAULT_RECOVERY_REQUEST_DEADLINE_MS = 120_000

const configuredDeadline = Number(
  import.meta.env.VITE_RECOVERY_REQUEST_DEADLINE_MS,
)

export const RECOVERY_REQUEST_DEADLINE_MS = (
  Number.isFinite(configuredDeadline) && configuredDeadline > 0
)
  ? configuredDeadline
  : DEFAULT_RECOVERY_REQUEST_DEADLINE_MS

function recoveryRequestTimeoutError() {
  const error = new Error(
    '等待服务端响应超时；请求结果尚未确定，请使用原请求重试恢复',
  )
  error.code = 'recovery_request_timeout'
  return error
}

/**
 * Start one browser-side recovery attempt with a bounded response deadline.
 * Aborting only stops this browser from waiting; it never means that the
 * server rolled the request back.
 */
export function startRecoveryRequest(requestFactory, activeControllers) {
  const controller = new AbortController()
  activeControllers?.add(controller)

  const operation = Promise.resolve().then(
    () => requestFactory(controller.signal),
  )
  let timer = null
  const deadline = new Promise((_, reject) => {
    timer = globalThis.setTimeout(() => {
      reject(recoveryRequestTimeoutError())
      controller.abort()
    }, RECOVERY_REQUEST_DEADLINE_MS)
  })
  const promise = Promise.race([operation, deadline]).finally(() => {
    globalThis.clearTimeout(timer)
    activeControllers?.delete(controller)
  })

  return {
    promise,
    abort: () => controller.abort(),
  }
}

export function runRecoveryRequest(requestFactory, activeControllers) {
  return startRecoveryRequest(requestFactory, activeControllers).promise
}

export function abortRecoveryRequests(activeControllers) {
  activeControllers.forEach(controller => controller.abort())
  activeControllers.clear()
}
