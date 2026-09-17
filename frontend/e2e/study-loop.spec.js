import { expect, test } from '@playwright/test'

const API_PREFIX = '/api'
const UUID_V4_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i
const AUTONOMOUS_SESSION_KEY = 'study-loop.autonomous.recovery.v2'
const LEGACY_AUTONOMOUS_SESSION_KEY = 'study-loop.autonomous.awaiting.v1'
const QUIZ_RECOVERY_KEY = 'study-loop.quiz.recovery.v1'
const ADAPTIVE_RECOVERY_KEY = 'study-loop.adaptive.recovery.v1'
const LEARNING_PATH_RECOVERY_KEY = 'study-loop.learning-path.recovery.v1'
const FUTURE_EXPIRES_AT = 4_102_444_800
const RECOVERY_DEADLINE_ADVANCE_MS = 120_001
const RECOVERY_TIMEOUT_MESSAGE = '等待服务端响应超时；请求结果尚未确定，请使用原请求重试恢复'

test.beforeEach(async ({ page }) => {
  await page.route('https://fonts.googleapis.com/**', route => route.fulfill({
    status: 200,
    contentType: 'text/css',
    body: '',
  }))
  await page.route('https://fonts.gstatic.com/**', route => route.fulfill({
    status: 204,
    body: '',
  }))
})

function trackBrowserProblems(page) {
  const problems = []

  page.on('console', message => {
    if (message.type() === 'error' || message.type() === 'warning') {
      problems.push(`${message.type()}: ${message.text()}`)
    }
  })
  page.on('pageerror', error => problems.push(`pageerror: ${error.message}`))

  return problems
}

async function mockApi(page, handler) {
  const unexpectedRequests = []

  await page.route(
    url => url.pathname.startsWith(`${API_PREFIX}/`),
    async route => {
      const request = route.request()
      const url = new URL(request.url())
      const path = url.pathname.slice(API_PREFIX.length)
      let response = await handler({ path, request })

      // Adaptive and Autonomous load their optional document choices on entry.
      // Tests focused on other behavior use an empty library as the neutral
      // baseline; document-loading tests override it in their own handler.
      if (!response && request.method() === 'GET' && path === '/documents') {
        response = { body: { documents: [] } }
      }
      if (
        !response
        && request.method() === 'GET'
        && path === '/learning-paths/current'
      ) {
        response = { body: null }
      }

      if (!response) {
        unexpectedRequests.push(`${request.method()} ${path}`)
        await route.fulfill({
          status: 500,
          contentType: 'application/json',
          body: JSON.stringify({ detail: `Unexpected E2E API request: ${request.method()} ${path}` }),
        })
        return
      }

      if (response.abort) {
        await route.abort(response.abort)
        return
      }

      await route.fulfill({
        status: response.status || 200,
        contentType: 'application/json',
        headers: response.headers,
        body: JSON.stringify(response.body),
      })
    }
  )

  return unexpectedRequests
}

function waitForAbortedApiRequest(page, method, path) {
  return page.waitForEvent('requestfailed', {
    predicate: request => (
      request.method() === method
      && new URL(request.url()).pathname === `${API_PREFIX}${path}`
    ),
    timeout: 6_000,
  }).catch(error => error)
}

function expectBrowserAbort(request) {
  expect(typeof request?.failure).toBe('function')
  expect(request.failure()?.errorText || '').toMatch(/abort|cancel/iu)
}

function quizRecovery({
  sessionId = null,
  pendingAnswer = null,
  acknowledgedAnswerCount = 0,
} = {}) {
  return {
    schema_version: 1,
    intent: {
      kind: 'standard',
      request: {
        document_id: 'notes.md',
        description: '首字母',
        count: 1,
        difficulty: 'medium',
        type: 'choice',
        user_id: 'default_user',
      },
    },
    launch_id: null,
    launch_preset: null,
    start_idempotency_key: '11111111-1111-4111-8111-111111111111',
    session: sessionId ? {
      session_id: sessionId,
      revision: 1,
      expires_at: FUTURE_EXPIRES_AT,
    } : null,
    acknowledged_answer_count: acknowledgedAnswerCount,
    pending_answer: pendingAnswer,
  }
}

function quizSnapshot({
  sessionId,
  status = 'active',
  answeredCount = status === 'completed' ? 1 : 0,
  revision = answeredCount + 1,
  lastAnswerResult = null,
  result = status === 'completed' ? {
    session_id: sessionId,
    document_id: 'notes.md',
    total: 1,
    correct: 1,
    incorrect: 0,
    pending: 0,
    score: 1,
    details: [],
    revision,
    expires_at: FUTURE_EXPIRES_AT,
  } : null,
  gradingReport = null,
  learningReport = null,
} = {}) {
  return {
    schema_version: 1,
    origin: 'standard',
    session_id: sessionId,
    document_id: 'notes.md',
    revision,
    status,
    total: 1,
    answered_count: answeredCount,
    questions: [{
      index: 0,
      question: '请选择首字母',
      options: ['Alpha', 'Beta'],
      type: 'choice',
    }],
    last_answer_index: lastAnswerResult ? 0 : null,
    last_user_answer: lastAnswerResult ? 'Alpha' : null,
    last_answer_result: lastAnswerResult,
    result,
    grading_report: gradingReport,
    learning_report: learningReport,
    learning_path_source: null,
    learning_path_completion: null,
    expires_at: FUTURE_EXPIRES_AT,
    busy: false,
  }
}

function adaptiveSnapshot(overrides = {}) {
  return {
    schema_version: 1,
    adaptive_session_id: 'adapt-fixture',
    turn: 1,
    done: false,
    turn_type: 'quiz',
    questions: [{
      index: 0,
      question: '请选择首字母',
      options: ['Alpha', 'Beta'],
      type: 'choice',
    }],
    lesson: null,
    decision: {
      action: 'continue',
      topic: '首字母',
      reason: '继续验证掌握情况',
      difficulty: 'medium',
      difficulty_score: 0.5,
      question_type: 'choice',
      count: 1,
      target_weak_points: [],
    },
    last_report_score: null,
    last_report_gaps: [],
    last_report_feedback: [],
    mastery: 0.4,
    trajectory: [{
      turn: 1,
      action: 'continue',
      topic: '首字母',
      difficulty_score: 0.5,
      score: null,
      mastery_after: 0.4,
      knowledge_gaps: [],
    }],
    summary: '',
    terminate_reason: '',
    learning_path: null,
    learning_path_id: null,
    revision: 1,
    expires_at: FUTURE_EXPIRES_AT,
    busy: false,
    ...overrides,
  }
}

function adaptiveRecovery(snapshot, pendingSubmit = null) {
  return {
    schema_version: 1,
    intent: {
      user_id: 'default_user',
      document_id: 'notes.md',
      goal: '掌握首字母',
    },
    start_idempotency_key: 'adaptive-start-key-1234',
    session: { adaptive_session_id: snapshot.adaptive_session_id },
    snapshot,
    pending_submit: pendingSubmit,
  }
}

function learningPathResource({
  id = 'lp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
  documentId = 'notes.md',
  title = '检索学习路径',
  stageTitle = '混合检索',
  topics = ['BM25', 'RRF'],
  completedThrough = 0,
} = {}) {
  return {
    schema_version: 1,
    learning_path_id: id,
    user_id: 'default_user',
    path: {
      document_id: documentId,
      title,
      total_stages: 1,
      stages: [{
        stage: 1,
        title: stageTitle,
        topics,
        description: '理解材料中的核心概念',
        estimated_minutes: 20,
      }],
    },
    progress: {
      completed_through: completedThrough,
      revision: completedThrough + 1,
    },
    created_at: 1_787_200_000,
    expires_at: null,
  }
}

function sequentialLearningPathResource(completedThrough = 0) {
  const resource = learningPathResource({
    title: '顺序学习路径',
    stageTitle: '基础概念',
    topics: ['基础'],
    completedThrough,
  })
  resource.path.total_stages = 2
  resource.path.stages.push({
    stage: 2,
    title: '综合应用',
    topics: ['应用'],
    description: '把基础概念应用到综合问题',
    estimated_minutes: 20,
  })
  return resource
}

test('mobile navigation traps focus and restores it on close', async ({ page }) => {
  const problems = trackBrowserProblems(page)
  await page.setViewportSize({ width: 390, height: 844 })
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: [] } }
    }
    return null
  })

  await page.goto('/documents')

  const viewport = await page.evaluate(() => ({
    width: window.innerWidth,
    scrollWidth: document.documentElement.scrollWidth,
  }))
  expect(viewport.scrollWidth).toBe(viewport.width)

  const menuButton = page.getByRole('button', { name: '打开导航' })
  await menuButton.click()

  const drawer = page.getByRole('dialog', { name: '主要导航' })
  const closeButton = drawer.getByRole('button', { name: '关闭导航' })
  await expect(drawer).toHaveAttribute('aria-modal', 'true')
  await expect(closeButton).toBeFocused()
  await expect(page.locator('.mobile-header')).toHaveJSProperty('inert', true)
  await expect(page.locator('main')).toHaveJSProperty('inert', true)

  await page.keyboard.press('Shift+Tab')
  await expect(drawer.getByRole('link', { name: '学习报告' })).toBeFocused()
  await page.keyboard.press('Tab')
  await expect(closeButton).toBeFocused()

  await page.keyboard.press('Escape')
  await expect(menuButton).toBeFocused()
  await expect(page.locator('.sidebar')).toHaveAttribute('aria-hidden', 'true')
  await expect(page.locator('.sidebar')).toHaveJSProperty('inert', true)
  expect(unexpectedRequests).toEqual([])
  expect(problems).toEqual([])
})

test('quiz setup exposes named controls and pressed states', async ({ page }) => {
  const problems = trackBrowserProblems(page)
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['sample_document.md'] } }
    }
    return null
  })

  await page.goto('/quiz')

  const documentSelect = page.getByRole('combobox', { name: '学习文档' })
  await expect(documentSelect).toBeVisible()
  await expect(documentSelect.getByRole('option', { name: 'sample_document.md' })).toHaveCount(1)
  await expect(page.getByRole('textbox', { name: /出题主题/ })).toBeVisible()

  const countGroup = page.getByRole('group', { name: '题目数量' })
  const fiveQuestions = countGroup.getByRole('button', { name: '5 题' })
  const eightQuestions = countGroup.getByRole('button', { name: '8 题' })
  await expect(fiveQuestions).toHaveAttribute('aria-pressed', 'true')
  await eightQuestions.click()
  await expect(fiveQuestions).toHaveAttribute('aria-pressed', 'false')
  await expect(eightQuestions).toHaveAttribute('aria-pressed', 'true')

  await expect(
    page.getByRole('group', { name: '难度' }).getByRole('button', { name: '进阶' })
  ).toHaveAttribute('aria-pressed', 'true')
  await expect(
    page.getByRole('group', { name: '题型' }).getByRole('button', { name: '选择题' })
  ).toHaveAttribute('aria-pressed', 'true')
  expect(unexpectedRequests).toEqual([])
  expect(problems).toEqual([])
})

test('learning flows replace unusable setup controls with an upload action', async ({ page }) => {
  const problems = trackBrowserProblems(page)
  await page.setViewportSize({ width: 390, height: 844 })
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: [] } }
    }
    return null
  })

  for (const { route, unavailableAction } of [
    { route: '/learning-path', unavailableAction: '生成学习路径' },
    { route: '/quiz', unavailableAction: '开始答题' },
  ]) {
    await page.goto(route)

    await expect(page.getByRole('heading', { name: '先上传学习材料' })).toBeVisible()
    await expect(page.getByRole('link', { name: '前往文档管理' })).toHaveAttribute(
      'href',
      '/documents'
    )
    await expect(page.getByRole('button', { name: unavailableAction })).toHaveCount(0)
  }

  const viewport = await page.evaluate(() => ({
    width: window.innerWidth,
    scrollWidth: document.documentElement.scrollWidth,
  }))
  expect(viewport.scrollWidth).toBe(viewport.width)

  expect(unexpectedRequests).toEqual([])
  expect(problems).toEqual([])
})

test('learning flow document failures stay recoverable and distinct from empty data', async ({ page }) => {
  let failDocuments = true
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return failDocuments
        ? { status: 503, body: { detail: '文档服务暂时不可用' } }
        : { body: { documents: [] } }
    }
    return null
  })

  for (const route of ['/learning-path', '/quiz']) {
    failDocuments = true
    await page.goto(route)

    await expect(page.getByRole('alert')).toContainText('无法加载文档列表')
    await expect(page.getByRole('heading', { name: '先上传学习材料' })).toHaveCount(0)

    failDocuments = false
    await page.getByRole('button', { name: '重新加载文档' }).click()
    await expect(page.getByRole('heading', { name: '先上传学习材料' })).toBeVisible()
    await expect(page.getByRole('alert')).toHaveCount(0)
  }

  expect(unexpectedRequests).toEqual([])
})

test('learning path keeps the last durable result until a replacement succeeds', async ({ page }) => {
  const replacement = learningPathResource({
    id: 'lp_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
    documentId: 'b.md',
    title: 'B 文档专属路径',
    stageTitle: '理解 B',
    topics: ['B'],
  })
  const replacementAttempts = []
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['a.md', 'b.md'] } }
    }
    if (request.method() === 'POST' && path === '/learning-paths') {
      const body = request.postDataJSON()
      if (body.document_id === 'a.md') {
        return {
          body: learningPathResource({
            id: 'lp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
            documentId: 'a.md',
            title: 'A 文档专属路径',
            stageTitle: '理解 A',
            topics: ['A'],
          }),
        }
      }
      if (body.document_id === 'b.md') {
        replacementAttempts.push({
          key: await request.headerValue('idempotency-key'),
          body,
        })
        return replacementAttempts.length === 1
          ? { status: 503, body: { detail: 'B 文档路径生成失败' } }
          : { body: replacement }
      }
    }
    return null
  })

  await page.goto('/learning-path')
  const documentSelect = page.getByRole('combobox', { name: '学习文档' })

  await documentSelect.selectOption('a.md')
  await page.getByRole('button', { name: '生成学习路径' }).click()
  await expect(page.getByRole('heading', { name: 'A 文档专属路径' })).toBeVisible()

  await documentSelect.selectOption('b.md')
  await expect(page.getByRole('heading', { name: 'A 文档专属路径' })).toBeVisible()

  await page.getByRole('button', { name: '生成学习路径' }).click()
  await expect(page.getByRole('alert')).toContainText('B 文档路径生成失败')
  await expect(page.getByRole('heading', { name: 'A 文档专属路径' })).toBeVisible()
  await expect(documentSelect).toHaveValue('b.md')
  await expect(documentSelect).toBeDisabled()
  const retry = page.locator('.lp-generate-btn')
  await expect(retry).toHaveText('重试恢复')
  await retry.click()
  await expect(page.getByRole('heading', { name: 'B 文档专属路径' })).toBeVisible()
  expect(replacementAttempts).toHaveLength(2)
  expect(replacementAttempts[1]).toEqual(replacementAttempts[0])
  expect(unexpectedRequests).toEqual([])
})

test('learning path replacement response loss recovers the new intent instead of the stale URL', async ({ page }) => {
  const first = learningPathResource({
    id: 'lp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
    documentId: 'a.md',
    title: 'A 路径',
    stageTitle: '学习 A',
    topics: ['A'],
  })
  const replacement = learningPathResource({
    id: 'lp_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
    documentId: 'b.md',
    title: 'B 路径',
    stageTitle: '学习 B',
    topics: ['B'],
  })
  const replacementAttempts = []
  let allowReplacementSuccess = false
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['a.md', 'b.md'] } }
    }
    if (request.method() === 'POST' && path === '/learning-paths') {
      const body = request.postDataJSON()
      if (body.document_id === 'a.md') return { body: first }
      replacementAttempts.push({
        key: request.headers()['idempotency-key'],
        body,
      })
      if (!allowReplacementSuccess) return { abort: 'failed' }
      return { body: replacement }
    }
    return null
  })

  await page.goto('/learning-path')
  const select = page.getByRole('combobox', { name: '学习文档' })
  await select.selectOption('a.md')
  await page.getByRole('button', { name: '生成学习路径' }).click()
  await expect(page.getByRole('heading', { name: 'A 路径' })).toBeVisible()

  await select.selectOption('b.md')
  await page.getByRole('button', { name: '生成学习路径' }).click()
  await expect(page.getByRole('alert')).toBeVisible()
  expect(new URL(page.url()).searchParams.get('path_id')).toBeNull()
  await expect(page.getByRole('heading', { name: 'A 路径' })).toBeVisible()

  const attemptsBeforeReload = replacementAttempts.length
  expect(attemptsBeforeReload).toBeGreaterThanOrEqual(1)
  allowReplacementSuccess = true
  await page.reload()
  await expect(page.getByRole('heading', { name: 'B 路径' })).toBeVisible()
  await expect(page).toHaveURL(new RegExp(`path_id=${replacement.learning_path_id}`))
  expect(replacementAttempts).toHaveLength(attemptsBeforeReload + 1)
  for (const attempt of replacementAttempts) {
    expect(attempt).toEqual(replacementAttempts[0])
  }
  expect(unexpectedRequests).toEqual([])
})

test('learning path browser navigation cancels an in-flight replacement and loads the URL target', async ({ page }) => {
  const first = learningPathResource({
    id: 'lp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
    documentId: 'a.md',
    title: 'A 路径',
  })
  const target = learningPathResource({
    id: 'lp_cccccccccccccccccccccccccccccccc',
    documentId: 'c.md',
    title: 'C 路径',
  })
  const lateReplacement = learningPathResource({
    id: 'lp_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
    documentId: 'b.md',
    title: 'B 路径',
  })
  let releaseReplacement
  let markReplacementStarted
  const replacementGate = new Promise(resolve => { releaseReplacement = resolve })
  const replacementStarted = new Promise(resolve => { markReplacementStarted = resolve })
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['a.md', 'b.md', 'c.md'] } }
    }
    if (
      request.method() === 'GET'
      && path === `/learning-paths/${first.learning_path_id}`
    ) {
      return { body: first }
    }
    if (
      request.method() === 'GET'
      && path === `/learning-paths/${target.learning_path_id}`
    ) {
      return { body: target }
    }
    if (request.method() === 'POST' && path === '/learning-paths') {
      markReplacementStarted()
      await replacementGate
      return { body: lateReplacement }
    }
    return null
  })

  try {
    await page.goto(`/learning-path?path_id=${first.learning_path_id}`)
    await expect(page.getByRole('heading', { name: 'A 路径' })).toBeVisible()
    await page.getByRole('combobox', { name: '学习文档' }).selectOption('b.md')
    const replacementAborted = waitForAbortedApiRequest(page, 'POST', '/learning-paths')
    await page.getByRole('button', { name: '生成学习路径' }).click()
    await replacementStarted

    await page.evaluate(pathId => {
      history.pushState({}, '', `/learning-path?path_id=${pathId}`)
      dispatchEvent(new PopStateEvent('popstate'))
    }, target.learning_path_id)
    await expect(page.getByRole('heading', { name: 'C 路径' })).toBeVisible()
    expectBrowserAbort(await replacementAborted)

    releaseReplacement()
    await expect(page.getByRole('heading', { name: 'C 路径' })).toBeVisible()
    await expect(page.getByRole('heading', { name: 'B 路径' })).toHaveCount(0)
    expect(unexpectedRequests).toEqual([])
  } finally {
    releaseReplacement()
  }
})

test('learning path create deadline preserves its receipt and retries the exact request', async ({ page }) => {
  await page.clock.install()
  const late = learningPathResource({
    id: 'lp_dddddddddddddddddddddddddddddddd',
    title: '不应采用的迟到路径',
  })
  const canonical = learningPathResource({
    id: 'lp_eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee',
    title: '超时后恢复的路径',
  })
  const queryTarget = learningPathResource({
    id: 'lp_ffffffffffffffffffffffffffffffff',
    documentId: 'other.md',
    title: '显式查询的其他路径',
  })
  const attempts = []
  let releaseFirst
  let markFirstStarted
  const firstGate = new Promise(resolve => { releaseFirst = resolve })
  const firstStarted = new Promise(resolve => { markFirstStarted = resolve })
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md', 'other.md'] } }
    }
    if (
      request.method() === 'GET'
      && path === `/learning-paths/${queryTarget.learning_path_id}`
    ) {
      return { body: queryTarget }
    }
    if (
      request.method() === 'GET'
      && path === `/learning-paths/${canonical.learning_path_id}`
    ) {
      // 恢复成功后页面把 URL 同步到 canonical，恢复 effect 因此再跑一轮。
      // 它是否真的发出这次 GET，取决于 resource 与 URL 在同一批渲染里的先后；
      // 两种时序读到的都是同一份资源，后面的断言在两种情况下都成立。
      return { body: canonical }
    }
    if (request.method() === 'POST' && path === '/learning-paths') {
      attempts.push({
        key: await request.headerValue('idempotency-key'),
        body: request.postDataJSON(),
        rawBody: request.postData(),
      })
      if (attempts.length === 1) {
        markFirstStarted()
        await firstGate
        return { body: late }
      }
      return { body: canonical }
    }
    return null
  })

  try {
    await page.goto('/learning-path')
    await page.getByRole('combobox', { name: '学习文档' }).selectOption('notes.md')
    const firstAborted = waitForAbortedApiRequest(page, 'POST', '/learning-paths')
    await page.getByRole('button', { name: '生成学习路径' }).click()
    await firstStarted
    await page.clock.fastForward(RECOVERY_DEADLINE_ADVANCE_MS)
    await expect(page.getByRole('alert')).toContainText(RECOVERY_TIMEOUT_MESSAGE)
    expectBrowserAbort(await firstAborted)
    const select = page.getByRole('combobox', { name: '学习文档' })
    await expect(select).toHaveValue('notes.md')
    await expect(select).toBeDisabled()
    await expect(page.locator('.error-retry')).toHaveCount(0)
    const mainRetry = page.locator('.lp-generate-btn')
    await expect(mainRetry).toHaveText('重试恢复')
    await expect(mainRetry).toBeEnabled()
    const pending = await page.evaluate(
      key => JSON.parse(sessionStorage.getItem(key)),
      LEARNING_PATH_RECOVERY_KEY,
    )
    expect(pending).toMatchObject({
      intent: { user_id: 'default_user', document_id: 'notes.md' },
      learning_path_id: null,
    })
    expect(pending.start_idempotency_key).toBe(attempts[0].key)

    await page.evaluate(pathId => {
      history.pushState({}, '', `/learning-path?path_id=${pathId}`)
      dispatchEvent(new PopStateEvent('popstate'))
    }, queryTarget.learning_path_id)
    await expect(page.getByRole('heading', { name: '显式查询的其他路径' })).toBeVisible()
    await expect(select).toHaveValue('notes.md')
    await expect(select).toBeDisabled()
    await expect(mainRetry).toHaveText('重试恢复')
    await expect(mainRetry).toBeEnabled()

    releaseFirst()
    await mainRetry.click()
    await expect(page.getByRole('heading', { name: '超时后恢复的路径' })).toBeVisible()
    await expect(page.getByRole('heading', { name: '不应采用的迟到路径' })).toHaveCount(0)
    await expect(page.getByRole('heading', { name: '显式查询的其他路径' })).toHaveCount(0)
    expect(attempts).toHaveLength(2)
    expect(attempts[0]).toEqual(attempts[1])
    expect(attempts[0].body).toEqual({
      document_id: 'notes.md',
      user_id: 'default_user',
    })
    expect(attempts[0].key).toMatch(UUID_V4_PATTERN)
    expect(
      await page.evaluate(
        key => JSON.parse(sessionStorage.getItem(key)),
        LEARNING_PATH_RECOVERY_KEY,
      ),
    ).toMatchObject({
      start_idempotency_key: attempts[0].key,
      learning_path_id: canonical.learning_path_id,
    })
    expect(unexpectedRequests).toEqual([])
  } finally {
    releaseFirst()
  }
})

test('learning path GET deadline keeps the bound recovery until an explicit retry', async ({ page }) => {
  await page.clock.install()
  const canonical = learningPathResource({ title: '权威路径快照' })
  const late = learningPathResource({ title: '迟到路径快照' })
  const recovery = {
    schema_version: 1,
    intent: { user_id: 'default_user', document_id: 'notes.md' },
    start_idempotency_key: 'learning-path-get-timeout-key',
    learning_path_id: canonical.learning_path_id,
  }
  await page.addInitScript(({ key, value }) => {
    sessionStorage.setItem(key, JSON.stringify(value))
  }, { key: LEARNING_PATH_RECOVERY_KEY, value: recovery })
  let reads = 0
  let releaseFirst
  let markFirstStarted
  const firstGate = new Promise(resolve => { releaseFirst = resolve })
  const firstStarted = new Promise(resolve => { markFirstStarted = resolve })
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md'] } }
    }
    if (
      request.method() === 'GET'
      && path === `/learning-paths/${canonical.learning_path_id}`
    ) {
      reads += 1
      if (reads === 1) {
        markFirstStarted()
        await firstGate
        return { body: late }
      }
      return { body: canonical }
    }
    return null
  })

  try {
    const firstAborted = waitForAbortedApiRequest(
      page,
      'GET',
      `/learning-paths/${canonical.learning_path_id}`,
    )
    await page.goto(`/learning-path?path_id=${canonical.learning_path_id}`)
    await firstStarted
    await page.clock.fastForward(RECOVERY_DEADLINE_ADVANCE_MS)
    await expect(page.getByRole('alert')).toContainText(RECOVERY_TIMEOUT_MESSAGE)
    expectBrowserAbort(await firstAborted)
    const retry = page.getByRole('button', { name: '重试恢复' })
    await expect(retry).toBeEnabled()
    expect(
      await page.evaluate(
        key => JSON.parse(sessionStorage.getItem(key)),
        LEARNING_PATH_RECOVERY_KEY,
      ),
    ).toEqual(recovery)

    releaseFirst()
    await retry.click()
    await expect(page.getByRole('heading', { name: '权威路径快照' })).toBeVisible()
    await expect(page.getByRole('heading', { name: '迟到路径快照' })).toHaveCount(0)
    expect(reads).toBe(2)
    expect(unexpectedRequests).toEqual([])
  } finally {
    releaseFirst()
  }
})

test('learning path stage opens a refresh-safe quiz preset without auto-starting', async ({ page }) => {
  const problems = trackBrowserProblems(page)
  const resource = learningPathResource()
  const startBodies = []
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md'] } }
    }
    if (request.method() === 'POST' && path === '/learning-paths') {
      return {
        body: resource,
      }
    }
    if (request.method() === 'POST' && path === '/session/start') {
      startBodies.push(await request.postDataJSON())
      return {
        body: {
          session_id: 'quiz-from-path',
          total: 1,
          questions: [{
            index: 0,
            question: 'RRF 如何融合多路排序？',
            options: ['A. 按倒数排名累加', 'B. 只保留单路结果'],
            type: 'choice',
          }],
          revision: 1,
          expires_at: FUTURE_EXPIRES_AT,
          learning_path_source: {
            learning_path_id: resource.learning_path_id,
            stage_id: 1,
          },
        },
      }
    }
    if (request.method() === 'GET' && path === '/session/quiz-from-path') {
      return {
        body: {
          schema_version: 1,
          origin: 'standard',
          session_id: 'quiz-from-path',
          document_id: 'notes.md',
          revision: 1,
          status: 'active',
          total: 1,
          answered_count: 0,
          questions: [{
            index: 0,
            question: 'RRF 如何融合多路排序？',
            options: ['A. 按倒数排名累加', 'B. 只保留单路结果'],
            type: 'choice',
          }],
          last_answer_index: null,
          last_user_answer: null,
          last_answer_result: null,
          result: null,
          grading_report: null,
          learning_report: null,
          learning_path_source: {
            learning_path_id: resource.learning_path_id,
            stage_id: 1,
          },
          learning_path_completion: null,
          expires_at: FUTURE_EXPIRES_AT,
          busy: false,
        },
      }
    }
    return null
  })

  await page.goto('/learning-path')
  await page.getByRole('combobox', { name: '学习文档' }).selectOption('notes.md')
  await page.getByRole('button', { name: '生成学习路径' }).click()
  await page.getByRole('button', { name: '练习阶段 1：混合检索' }).click()

  await expect(page).toHaveURL(/\/quiz\?/)
  let quizUrl = new URL(page.url())
  expect(quizUrl.searchParams.get('document_id')).toBe('notes.md')
  expect(quizUrl.searchParams.get('topic')).toBe('BM25、RRF')
  expect(quizUrl.searchParams.get('launch_id')).toMatch(UUID_V4_PATTERN)
  expect(quizUrl.searchParams.get('path_id')).toBe(resource.learning_path_id)
  expect(quizUrl.searchParams.get('stage_id')).toBe('1')
  await expect(page.getByRole('combobox', { name: '学习文档' })).toHaveValue('notes.md')
  await expect(page.getByRole('textbox', { name: /出题主题/ })).toHaveValue('BM25、RRF')
  expect(startBodies).toEqual([])

  await page.reload()

  quizUrl = new URL(page.url())
  expect(quizUrl.searchParams.get('document_id')).toBe('notes.md')
  expect(quizUrl.searchParams.get('topic')).toBe('BM25、RRF')
  expect(quizUrl.searchParams.get('path_id')).toBe(resource.learning_path_id)
  expect(quizUrl.searchParams.get('stage_id')).toBe('1')
  const launchId = quizUrl.searchParams.get('launch_id')
  expect(launchId).toMatch(UUID_V4_PATTERN)
  await expect(page.getByRole('combobox', { name: '学习文档' })).toHaveValue('notes.md')
  await expect(page.getByRole('textbox', { name: /出题主题/ })).toHaveValue('BM25、RRF')
  expect(startBodies).toEqual([])

  await page.getByRole('button', { name: '开始答题' }).click()
  await expect(page.getByText('RRF 如何融合多路排序？')).toBeVisible()
  expect(
    await page.evaluate(key => JSON.parse(sessionStorage.getItem(key)), QUIZ_RECOVERY_KEY)
  ).toMatchObject({
    launch_id: launchId,
    launch_preset: {
      document_id: 'notes.md',
      topic: 'BM25、RRF',
      path_id: resource.learning_path_id,
      stage_id: 1,
    },
    intent: {
      request: {
        learning_path_source: {
          learning_path_id: resource.learning_path_id,
          stage_id: 1,
        },
      },
    },
  })
  expect(startBodies).toEqual([{
    document_id: 'notes.md',
    description: 'BM25、RRF',
    count: 5,
    difficulty: 'medium',
    type: 'choice',
    user_id: 'default_user',
    learning_path_source: {
      learning_path_id: resource.learning_path_id,
      stage_id: 1,
    },
  }])

  await page.reload()
  await expect(page.getByRole('region', { name: '检测到另一项练习' })).toHaveCount(0)
  await expect(page.getByText('RRF 如何融合多路排序？')).toBeVisible()
  expect(startBodies).toHaveLength(1)
  expect(unexpectedRequests).toEqual([])
  expect(problems).toEqual([])
})

test('canonical quiz grade unlocks the next learning path stage', async ({ page }) => {
  const problems = trackBrowserProblems(page)
  const initialPath = sequentialLearningPathResource(0)
  const progressedPath = sequentialLearningPathResource(1)
  let gradeCommitted = false
  let gradeAttempts = 0
  let snapshotReads = 0
  let startBody = null
  const source = {
    learning_path_id: initialPath.learning_path_id,
    stage_id: 1,
  }
  const completion = {
    ...source,
    completed_through: 1,
    revision: 2,
  }
  const question = {
    index: 0,
    question: '基础概念的正确描述是哪一项？',
    options: ['A. 正确描述', 'B. 错误描述'],
    type: 'choice',
  }
  const grade = {
    index: 0,
    question: question.question,
    user_answer: question.options[0],
    correct_answer: 'B. 错误描述',
    is_correct: false,
    ai_feedback: '本阶段仍会在零分时记录为已完成。',
    knowledge_gap: '基础概念',
  }
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md'] } }
    }
    if (
      request.method() === 'GET'
      && path === `/learning-paths/${initialPath.learning_path_id}`
    ) {
      return { body: gradeCommitted ? progressedPath : initialPath }
    }
    if (request.method() === 'POST' && path === '/session/start') {
      startBody = request.postDataJSON()
      return {
        body: {
          session_id: 'quiz-path-progress',
          total: 1,
          questions: [question],
          revision: 1,
          expires_at: FUTURE_EXPIRES_AT,
          learning_path_source: source,
        },
      }
    }
    if (request.method() === 'POST' && path === '/session/quiz-path-progress/answer') {
      return {
        body: {
          evaluation_status: 'final',
          correct: false,
          correct_answer: 'B. 错误描述',
          explanation: '回答错误，但完成批改后仍推进阶段',
          is_last: true,
          next_index: null,
          revision: 2,
          expires_at: FUTURE_EXPIRES_AT,
        },
      }
    }
    if (request.method() === 'GET' && path === '/session/quiz-path-progress/result') {
      return {
        body: {
          session_id: 'quiz-path-progress',
          document_id: 'notes.md',
          total: 1,
          correct: 0,
          incorrect: 1,
          pending: 0,
          score: 0,
          details: [{
            index: 0,
            question: question.question,
            user_answer: question.options[0],
            correct_answer: 'B. 错误描述',
            correct: false,
            explanation: '回答错误，但完成批改后仍推进阶段',
            evaluation_status: 'final',
          }],
          revision: 3,
          expires_at: FUTURE_EXPIRES_AT,
        },
      }
    }
    if (request.method() === 'GET' && path === '/session/quiz-path-progress') {
      snapshotReads += 1
      return {
        body: {
          schema_version: 1,
          origin: 'standard',
          session_id: 'quiz-path-progress',
          document_id: 'notes.md',
          revision: 3,
          status: 'completed',
          total: 1,
          answered_count: 1,
          questions: [question],
          last_answer_index: 0,
          last_user_answer: question.options[0],
          last_answer_result: {
            evaluation_status: 'final',
            correct: false,
            correct_answer: 'B. 错误描述',
            explanation: '回答错误，但完成批改后仍推进阶段',
            is_last: true,
            next_index: null,
            revision: 2,
            expires_at: FUTURE_EXPIRES_AT,
          },
          result: {
            session_id: 'quiz-path-progress',
            document_id: 'notes.md',
            total: 1,
            correct: 0,
            incorrect: 1,
            pending: 0,
            score: 0,
            details: [],
            revision: 3,
            expires_at: FUTURE_EXPIRES_AT,
          },
          grading_report: null,
          learning_report: null,
          learning_path_source: source,
          learning_path_completion: null,
          expires_at: FUTURE_EXPIRES_AT,
          busy: false,
        },
      }
    }
    if (request.method() === 'POST' && path === '/session/quiz-path-progress/grade') {
      gradeAttempts += 1
      if (gradeAttempts === 1) {
        return {
          body: {
            session_id: 'quiz-other-session',
            total: 1,
            correct: 0,
            score: 0,
            grades: [grade],
            revision: 4,
            expires_at: FUTURE_EXPIRES_AT,
            learning_path_source: source,
            learning_path_completion: completion,
          },
        }
      }
      gradeCommitted = true
      return {
        body: {
          session_id: 'quiz-path-progress',
          total: 1,
          correct: 0,
          score: 0,
          grades: [grade],
          revision: 4,
          expires_at: FUTURE_EXPIRES_AT,
          learning_path_source: source,
          learning_path_completion: completion,
        },
      }
    }
    return null
  })

  await page.goto(`/learning-path?path_id=${initialPath.learning_path_id}`)
  await expect(page.getByText('已完成 0/2')).toBeVisible()
  await expect(page.getByText('完成上一阶段后解锁')).toBeVisible()
  await expect(page.locator('.tc-practice-btn')).toHaveCount(1)
  await page.getByRole('button', { name: '练习阶段 1：基础概念' }).click()
  await page.getByRole('button', { name: '开始答题' }).click()
  await expect(page.getByText(question.question)).toBeVisible()

  expect(startBody.learning_path_source).toEqual(source)
  await page.locator('.option-item').first().click()
  await page.getByRole('button', { name: '提交答案' }).click()
  await page.getByRole('button', { name: '查看结果' }).click()
  await expect(
    page.getByRole('button', { name: '返回学习路径，继续下一阶段' }),
  ).toHaveCount(0)

  await page.getByRole('button', { name: 'AI 批改讲解' }).click()
  await expect.poll(() => snapshotReads).toBe(1)
  expect(gradeAttempts).toBe(1)
  await expect(
    page.getByRole('button', { name: '返回学习路径，继续下一阶段' }),
  ).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'AI 批改讲解' })).toBeVisible()
  await page.getByRole('button', { name: 'AI 批改讲解' }).click()
  await expect(page.getByRole('heading', { name: 'AI 批改报告' })).toBeVisible()
  const returnButton = page.getByRole('button', {
    name: '返回学习路径，继续下一阶段',
  })
  await expect(returnButton).toBeVisible()
  await returnButton.click()

  await expect(page).toHaveURL(
    new RegExp(`/learning-path\\?path_id=${initialPath.learning_path_id}`),
  )
  await expect(page.getByText('已完成 1/2')).toBeVisible()
  await expect(page.getByText('已完成', { exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: '练习阶段 2：综合应用' })).toBeVisible()
  await expect(page.locator('.tc-practice-btn')).toHaveCount(1)
  expect(
    await page.evaluate(key => sessionStorage.getItem(key), QUIZ_RECOVERY_KEY),
  ).toBeNull()
  expect(unexpectedRequests).toEqual([])
  expect(problems).toEqual([])
  expect(gradeAttempts).toBe(2)
  expect(snapshotReads).toBe(1)
})

test('stale learning path stage start clears pending recovery and returns to the exact path', async ({ page }) => {
  const resource = sequentialLearningPathResource(1)
  const source = {
    learning_path_id: resource.learning_path_id,
    stage_id: 1,
  }
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md'] } }
    }
    if (request.method() === 'POST' && path === '/session/start') {
      return {
        status: 409,
        body: {
          code: 'learning_path_stage_unavailable',
          reason: 'stage_completed',
          detail: '该学习路径阶段已经完成，请返回路径继续下一阶段',
        },
      }
    }
    if (
      request.method() === 'GET'
      && path === `/learning-paths/${resource.learning_path_id}`
    ) {
      return { body: resource }
    }
    return null
  })
  const query = new URLSearchParams({
    document_id: 'notes.md',
    topic: '基础',
    launch_id: 'stale-stage-launch-1234',
    path_id: source.learning_path_id,
    stage_id: String(source.stage_id),
  })

  await page.goto(`/quiz?${query}`)
  await page.getByRole('button', { name: '开始答题' }).click()

  await expect(page.getByText('该阶段已不可继续')).toBeVisible()
  expect(await page.evaluate(key => sessionStorage.getItem(key), QUIZ_RECOVERY_KEY)).toBeNull()
  await page.getByRole('button', { name: '返回学习路径刷新进度' }).click()
  await expect(page).toHaveURL(
    new RegExp(`/learning-path\\?path_id=${resource.learning_path_id}`),
  )
  await expect(page.getByRole('heading', { name: '顺序学习路径' })).toBeVisible()
  expect(unexpectedRequests).toEqual([])
})

test('a missing bound path exits its unretryable start instead of looping on 404', async ({ page }) => {
  const source = {
    learning_path_id: 'lp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
    stage_id: 1,
  }
  let starts = 0
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md'] } }
    }
    if (request.method() === 'POST' && path === '/session/start') {
      starts += 1
      return {
        status: 404,
        body: {
          code: 'learning_path_not_found',
          reason: 'missing',
          detail: '学习路径不存在',
        },
      }
    }
    return null
  })
  const query = new URLSearchParams({
    document_id: 'notes.md',
    topic: '失效阶段',
    launch_id: 'missing-path-launch-1234',
    path_id: source.learning_path_id,
    stage_id: String(source.stage_id),
  })

  await page.goto(`/quiz?${query}`)
  await page.getByRole('button', { name: '开始答题' }).click()

  await expect(page.getByText('这条学习路径已不存在，请返回学习路径重新选择。')).toBeVisible()
  expect(starts).toBe(1)
  expect(await page.evaluate(key => sessionStorage.getItem(key), QUIZ_RECOVERY_KEY)).toBeNull()
  await page.getByRole('button', { name: '返回学习路径重新选择' }).click()
  await expect(page).toHaveURL(/\/learning-path$/)
  expect(starts).toBe(1)
  expect(unexpectedRequests).toEqual([])
})

test('a corrupt bound path clears quiz recovery and returns to learning path regeneration', async ({ page }) => {
  const source = {
    learning_path_id: 'lp_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
    stage_id: 1,
  }
  let starts = 0
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md'] } }
    }
    if (request.method() === 'POST' && path === '/session/start') {
      starts += 1
      return {
        status: 503,
        body: {
          code: 'learning_path_corrupt',
          detail: '学习路径数据损坏',
        },
      }
    }
    return null
  })
  const query = new URLSearchParams({
    document_id: 'notes.md',
    topic: '损坏路径阶段',
    launch_id: 'corrupt-path-launch-1234',
    path_id: source.learning_path_id,
    stage_id: String(source.stage_id),
  })

  await page.goto(`/quiz?${query}`)
  await page.getByRole('button', { name: '开始答题' }).click()

  await expect(page.getByText('这条学习路径数据损坏，请返回学习路径重新生成。')).toBeVisible()
  expect(starts).toBe(1)
  expect(await page.evaluate(key => sessionStorage.getItem(key), QUIZ_RECOVERY_KEY)).toBeNull()
  await page.getByRole('button', { name: '返回学习路径重新生成' }).click()
  await expect(page).toHaveURL(/\/learning-path$/)
  expect(starts).toBe(1)
  expect(unexpectedRequests).toEqual([])
})

test('a path binding mismatch is terminal and returns to the exact resource once', async ({ page }) => {
  const resource = learningPathResource()
  let starts = 0
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md'] } }
    }
    if (request.method() === 'POST' && path === '/session/start') {
      starts += 1
      return {
        status: 409,
        body: {
          code: 'learning_path_binding_mismatch',
          reason: 'binding_mismatch',
          detail: '练习与学习路径不匹配',
        },
      }
    }
    if (
      request.method() === 'GET'
      && path === `/learning-paths/${resource.learning_path_id}`
    ) {
      return { body: resource }
    }
    return null
  })
  const query = new URLSearchParams({
    document_id: 'notes.md',
    topic: 'BM25、RRF',
    launch_id: 'binding-mismatch-1234',
    path_id: resource.learning_path_id,
    stage_id: '1',
  })

  await page.goto(`/quiz?${query}`)
  await page.getByRole('button', { name: '开始答题' }).click()

  await expect(page.getByText('该阶段已不可继续')).toBeVisible()
  expect(starts).toBe(1)
  expect(await page.evaluate(key => sessionStorage.getItem(key), QUIZ_RECOVERY_KEY)).toBeNull()
  await page.getByRole('button', { name: '返回学习路径刷新进度' }).click()
  await expect(page.getByRole('heading', { name: '检索学习路径' })).toBeVisible()
  expect(starts).toBe(1)
  expect(unexpectedRequests).toEqual([])
})

test('quiz start rejects a mismatched path echo before persisting its session', async ({ page }) => {
  const resource = learningPathResource()
  const expectedSource = {
    learning_path_id: resource.learning_path_id,
    stage_id: 1,
  }
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md'] } }
    }
    if (request.method() === 'POST' && path === '/session/start') {
      return {
        body: {
          session_id: 'quiz-wrong-path-echo',
          total: 1,
          questions: [{ index: 0, question: '不应被采用的题目', options: ['A', 'B'], type: 'choice' }],
          revision: 1,
          expires_at: FUTURE_EXPIRES_AT,
          learning_path_source: {
            learning_path_id: 'lp_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
            stage_id: 1,
          },
        },
      }
    }
    return null
  })
  const query = new URLSearchParams({
    document_id: 'notes.md',
    topic: 'BM25、RRF',
    launch_id: 'wrong-path-echo-1234',
    path_id: expectedSource.learning_path_id,
    stage_id: String(expectedSource.stage_id),
  })

  await page.goto(`/quiz?${query}`)
  await page.getByRole('button', { name: '开始答题' }).click()

  await expect(page.getByRole('alert')).toContainText('学习路径归属不一致')
  expect(
    await page.evaluate(key => JSON.parse(sessionStorage.getItem(key)), QUIZ_RECOVERY_KEY),
  ).toMatchObject({ session: null, intent: { request: { learning_path_source: expectedSource } } })
  await expect(page.getByText('不应被采用的题目')).toHaveCount(0)
  expect(unexpectedRequests).toEqual([])
})

test('bound quiz snapshot restores a durable completion after the grade response is lost', async ({ page }) => {
  const progressedPath = sequentialLearningPathResource(1)
  const source = {
    learning_path_id: progressedPath.learning_path_id,
    stage_id: 1,
  }
  const completion = {
    ...source,
    completed_through: 1,
    revision: 2,
  }
  const recovery = {
    schema_version: 1,
    intent: {
      kind: 'standard',
      request: {
        document_id: 'notes.md',
        description: '基础',
        count: 1,
        difficulty: 'medium',
        type: 'choice',
        user_id: 'default_user',
        learning_path_source: source,
      },
    },
    launch_id: 'snapshot-path-launch-1234',
    launch_preset: {
      document_id: 'notes.md',
      topic: '基础',
      path_id: source.learning_path_id,
      stage_id: source.stage_id,
    },
    start_idempotency_key: 'snapshot-path-start-1234',
    session: {
      session_id: 'quiz-path-snapshot',
      revision: 3,
      expires_at: FUTURE_EXPIRES_AT,
    },
    acknowledged_answer_count: 1,
    pending_answer: null,
  }
  await page.addInitScript(({ key, value }) => {
    sessionStorage.setItem(key, JSON.stringify(value))
  }, { key: QUIZ_RECOVERY_KEY, value: recovery })

  const question = {
    index: 0,
    question: '已完成的路径题目',
    options: ['A', 'B'],
    type: 'choice',
  }
  const gradingReport = {
    session_id: 'quiz-path-snapshot',
    total: 1,
    correct: 0,
    score: 0,
    grades: [{
      index: 0,
      question: question.question,
      user_answer: 'A',
      correct_answer: 'B',
      is_correct: false,
      ai_feedback: '批改已持久化',
      knowledge_gap: '基础',
    }],
    revision: 4,
    expires_at: FUTURE_EXPIRES_AT,
    learning_path_source: source,
    learning_path_completion: completion,
  }
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md'] } }
    }
    if (request.method() === 'GET' && path === '/session/quiz-path-snapshot') {
      return {
        body: {
          schema_version: 1,
          origin: 'standard',
          session_id: 'quiz-path-snapshot',
          document_id: 'notes.md',
          revision: 4,
          status: 'completed',
          total: 1,
          answered_count: 1,
          questions: [question],
          last_answer_index: 0,
          last_user_answer: 'A',
          last_answer_result: {
            evaluation_status: 'final',
            correct: false,
            correct_answer: 'B',
            explanation: '回答错误',
            is_last: true,
            next_index: null,
            revision: 2,
            expires_at: FUTURE_EXPIRES_AT,
          },
          result: {
            session_id: 'quiz-path-snapshot',
            document_id: 'notes.md',
            total: 1,
            correct: 0,
            incorrect: 1,
            pending: 0,
            score: 0,
            details: [],
            revision: 3,
            expires_at: FUTURE_EXPIRES_AT,
          },
          grading_report: gradingReport,
          learning_report: null,
          learning_path_source: source,
          learning_path_completion: completion,
          expires_at: FUTURE_EXPIRES_AT,
          busy: false,
        },
      }
    }
    if (
      request.method() === 'GET'
      && path === `/learning-paths/${source.learning_path_id}`
    ) {
      return { body: progressedPath }
    }
    return null
  })
  const query = new URLSearchParams({
    document_id: 'notes.md',
    topic: '基础',
    launch_id: recovery.launch_id,
    path_id: source.learning_path_id,
    stage_id: String(source.stage_id),
  })

  await page.goto(`/quiz?${query}`)
  await expect(page.getByRole('heading', { name: 'AI 批改报告' })).toBeVisible()
  const returnButton = page.getByRole('button', {
    name: '返回学习路径，继续下一阶段',
  })
  await expect(returnButton).toBeVisible()
  await returnButton.click()
  await expect(page.getByText('已完成 1/2')).toBeVisible()
  expect(await page.evaluate(key => sessionStorage.getItem(key), QUIZ_RECOVERY_KEY)).toBeNull()
  expect(unexpectedRequests).toEqual([])
})

test('a fully completed learning path has no ready practice action', async ({ page }) => {
  const resource = sequentialLearningPathResource(2)
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md'] } }
    }
    if (
      request.method() === 'GET'
      && path === `/learning-paths/${resource.learning_path_id}`
    ) {
      return { body: resource }
    }
    return null
  })

  await page.goto(`/learning-path?path_id=${resource.learning_path_id}`)
  await expect(page.getByText('已完成 2/2')).toBeVisible()
  await expect(page.locator('.tc-practice-btn')).toHaveCount(0)
  expect(unexpectedRequests).toEqual([])
})

test('quiz rejects an empty partial learning path binding instead of downgrading it', async ({ page }) => {
  let starts = 0
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md'] } }
    }
    if (request.method() === 'POST' && path === '/session/start') {
      starts += 1
      return { status: 500, body: { detail: '不应发起练习' } }
    }
    return null
  })

  await page.goto(
    '/quiz?document_id=notes.md&topic=基础&launch_id=partial-path-launch-1234&path_id=',
  )

  await expect(page.getByRole('alert')).toContainText('练习链接参数无效')
  await expect(page.getByRole('button', { name: '开始答题' })).toBeDisabled()
  expect(starts).toBe(0)
  expect(unexpectedRequests).toEqual([])
})

test('a literal learning path topic named 全文 remains a valid bound quiz intent', async ({ page }) => {
  const source = {
    learning_path_id: 'lp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
    stage_id: 1,
  }
  let startBody = null
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md'] } }
    }
    if (request.method() === 'POST' && path === '/session/start') {
      startBody = await request.postDataJSON()
      return {
        body: {
          session_id: 'quiz-literal-full-text-topic',
          total: 1,
          questions: [{
            index: 0,
            question: '字面“全文”主题仍应生成题目',
            options: ['A', 'B'],
            type: 'choice',
          }],
          revision: 1,
          expires_at: FUTURE_EXPIRES_AT,
          learning_path_source: source,
        },
      }
    }
    return null
  })
  const query = new URLSearchParams({
    document_id: 'notes.md',
    topic: '全文',
    launch_id: 'literal-full-text-1234',
    path_id: source.learning_path_id,
    stage_id: String(source.stage_id),
  })

  await page.goto(`/quiz?${query}`)
  await page.getByRole('button', { name: '开始答题' }).click()
  await expect(page.getByText('字面“全文”主题仍应生成题目')).toBeVisible()
  expect(startBody).toMatchObject({
    description: '全文',
    learning_path_source: source,
  })
  expect(unexpectedRequests).toEqual([])
})

test('a late stale-stage response cannot leave its return target on a newer quiz URL', async ({ page }) => {
  const source = {
    learning_path_id: 'lp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
    stage_id: 1,
  }
  let releaseStart
  let markStartReceived
  const startGate = new Promise(resolve => { releaseStart = resolve })
  const startReceived = new Promise(resolve => { markStartReceived = resolve })
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md'] } }
    }
    if (request.method() === 'POST' && path === '/session/start') {
      markStartReceived()
      await startGate
      return {
        status: 409,
        body: {
          code: 'learning_path_stage_unavailable',
          reason: 'stage_completed',
          detail: '旧阶段已经完成',
        },
      }
    }
    return null
  })
  const oldQuery = new URLSearchParams({
    document_id: 'notes.md',
    topic: '旧阶段',
    launch_id: 'late-stale-stage-1234',
    path_id: source.learning_path_id,
    stage_id: String(source.stage_id),
  })

  await page.goto(`/quiz?${oldQuery}`)
  await page.getByRole('button', { name: '开始答题' }).click()
  await startReceived
  const nextUrl = '/quiz?document_id=notes.md&topic=新阶段&launch_id=new-stage-launch-1234'
  await page.evaluate(url => {
    history.pushState({}, '', url)
    dispatchEvent(new PopStateEvent('popstate'))
  }, nextUrl)
  await expect(page.getByRole('region', { name: '检测到另一项练习' })).toBeVisible()

  const staleResponsePromise = page.waitForResponse(response => (
    response.url().includes('/api/session/start')
  ))
  releaseStart()
  const staleResponse = await staleResponsePromise
  await staleResponse.finished()
  await page.evaluate(() => new Promise(resolve => {
    requestAnimationFrame(() => requestAnimationFrame(resolve))
  }))

  await expect(page.getByText('该阶段已不可继续')).toHaveCount(0)
  await expect(page.getByRole('textbox', { name: /出题主题/ })).toHaveValue('新阶段')
  await expect(page.getByRole('button', { name: '开始答题' })).toBeEnabled()
  expect(await page.evaluate(key => sessionStorage.getItem(key), QUIZ_RECOVERY_KEY)).toBeNull()
  expect(unexpectedRequests).toEqual([])
})

test('learning path reuses one create key after response loss and then reloads by id', async ({ page }) => {
  const resource = learningPathResource()
  const createAttempts = []
  let reads = 0
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md'] } }
    }
    if (request.method() === 'POST' && path === '/learning-paths') {
      createAttempts.push({
        key: request.headers()['idempotency-key'],
        body: request.postDataJSON(),
      })
      if (createAttempts.length === 1) return { abort: 'failed' }
      return { body: resource }
    }
    if (
      request.method() === 'GET'
      && path === `/learning-paths/${resource.learning_path_id}`
    ) {
      reads += 1
      return { body: resource }
    }
    return null
  })

  await page.goto('/learning-path')
  await page.getByRole('combobox', { name: '学习文档' }).selectOption('notes.md')
  await page.getByRole('button', { name: '生成学习路径' }).click()
  await expect(page.getByRole('alert')).toBeVisible()

  const pending = await page.evaluate(
    key => JSON.parse(sessionStorage.getItem(key)),
    LEARNING_PATH_RECOVERY_KEY,
  )
  expect(pending).toMatchObject({
    intent: { user_id: 'default_user', document_id: 'notes.md' },
    learning_path_id: null,
  })
  expect(pending.start_idempotency_key).toMatch(UUID_V4_PATTERN)

  await page.reload()
  await expect(page.getByRole('heading', { name: '检索学习路径' })).toBeVisible()
  await expect(page).toHaveURL(new RegExp(`path_id=${resource.learning_path_id}`))
  expect(createAttempts).toHaveLength(2)
  expect(createAttempts[0]).toEqual(createAttempts[1])
  expect(createAttempts[0].body).toEqual({
    document_id: 'notes.md',
    user_id: 'default_user',
  })

  const bound = await page.evaluate(
    key => JSON.parse(sessionStorage.getItem(key)),
    LEARNING_PATH_RECOVERY_KEY,
  )
  expect(bound.learning_path_id).toBe(resource.learning_path_id)

  await page.reload()
  await expect(page.getByRole('heading', { name: '检索学习路径' })).toBeVisible()
  expect(createAttempts).toHaveLength(2)
  expect(reads).toBe(1)
  expect(unexpectedRequests).toEqual([])
})

test('learning path discovers the latest server record without browser recovery state', async ({ page }) => {
  const resource = learningPathResource()
  let createCount = 0
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md'] } }
    }
    if (request.method() === 'GET' && path === '/learning-paths/current') {
      return { body: resource }
    }
    if (request.method() === 'POST' && path === '/learning-paths') {
      createCount += 1
    }
    return null
  })

  await page.goto('/learning-path')
  await expect(page.getByRole('heading', { name: '检索学习路径' })).toBeVisible()
  await expect(page).toHaveURL(new RegExp(`path_id=${resource.learning_path_id}`))
  expect(createCount).toBe(0)
  expect(
    await page.evaluate(key => sessionStorage.getItem(key), LEARNING_PATH_RECOVERY_KEY)
  ).toBeNull()
  expect(unexpectedRequests).toEqual([])
})

test('learning path ignores a late GET after browser history returns to the visible record', async ({ page }) => {
  const first = learningPathResource({
    id: 'lp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
    title: 'A 路径',
  })
  const late = learningPathResource({
    id: 'lp_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
    title: 'B 路径',
  })
  let releaseLate
  let markLateStarted
  const lateGate = new Promise(resolve => { releaseLate = resolve })
  const lateStarted = new Promise(resolve => { markLateStarted = resolve })
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md'] } }
    }
    if (
      request.method() === 'GET'
      && path === `/learning-paths/${first.learning_path_id}`
    ) {
      return { body: first }
    }
    if (
      request.method() === 'GET'
      && path === `/learning-paths/${late.learning_path_id}`
    ) {
      markLateStarted()
      await lateGate
      return { body: late }
    }
    return null
  })

  try {
    await page.goto(`/learning-path?path_id=${first.learning_path_id}`)
    await expect(page.getByRole('heading', { name: 'A 路径' })).toBeVisible()
    const lateAborted = waitForAbortedApiRequest(
      page,
      'GET',
      `/learning-paths/${late.learning_path_id}`,
    )
    await page.evaluate(pathId => {
      history.pushState({}, '', `/learning-path?path_id=${pathId}`)
      dispatchEvent(new PopStateEvent('popstate'))
    }, late.learning_path_id)
    await lateStarted
    await expect(page.getByText('正在恢复学习路径')).toBeVisible()

    await page.goBack()
    await expect(page).toHaveURL(new RegExp(`path_id=${first.learning_path_id}`))
    await expect(page.getByRole('heading', { name: 'A 路径' })).toBeVisible()
    await expect(page.getByText('正在恢复学习路径')).toHaveCount(0)
    expectBrowserAbort(await lateAborted)

    releaseLate()
    await expect(page.getByRole('heading', { name: 'A 路径' })).toBeVisible()
    await expect(page.getByRole('heading', { name: 'B 路径' })).toHaveCount(0)
    expect(unexpectedRequests).toEqual([])
  } finally {
    releaseLate()
  }
})

test('learning path remains readable after material deletion but cannot start practice', async ({ page }) => {
  const resource = learningPathResource()
  const recovery = {
    schema_version: 1,
    intent: { user_id: 'default_user', document_id: 'notes.md' },
    start_idempotency_key: 'learning-path-start-key-1',
    learning_path_id: resource.learning_path_id,
  }
  await page.addInitScript(({ key, value }) => {
    sessionStorage.setItem(key, JSON.stringify(value))
  }, { key: LEARNING_PATH_RECOVERY_KEY, value: recovery })

  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: [] } }
    }
    if (
      request.method() === 'GET'
      && path === `/learning-paths/${resource.learning_path_id}`
    ) {
      return { body: resource }
    }
    return null
  })

  await page.goto(`/learning-path?path_id=${resource.learning_path_id}`)
  await expect(page.getByRole('heading', { name: '检索学习路径' })).toBeVisible()
  await expect(page.getByText(/原材料已删除.*仅供查看/)).toBeVisible()
  await expect(page.getByRole('button', { name: '练习阶段 1：混合检索' }))
    .toHaveCount(0)
  await expect(page.getByText('材料已删除，仅供查看')).toBeVisible()
  expect(unexpectedRequests).toEqual([])
})

test('quiz requires an explicit choice when a new launch conflicts with recovery', async ({ page }) => {
  const problems = trackBrowserProblems(page)
  let oldSnapshotReads = 0
  const startBodies = []
  const recovery = {
    schema_version: 1,
    intent: {
      kind: 'standard',
      request: {
        document_id: 'old.md',
        description: '旧主题',
        count: 3,
        difficulty: 'medium',
        type: 'choice',
        user_id: 'default_user',
      },
    },
    launch_id: 'old-launch-1234',
    launch_preset: { document_id: 'old.md', topic: '旧主题' },
    start_idempotency_key: 'old-start-key-1234',
    session: {
      session_id: 'quiz-old',
      revision: 1,
      expires_at: FUTURE_EXPIRES_AT,
    },
    acknowledged_answer_count: 0,
    pending_answer: null,
  }
  await page.addInitScript(({ key, value }) => {
    sessionStorage.setItem(key, JSON.stringify(value))
  }, { key: QUIZ_RECOVERY_KEY, value: recovery })

  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['old.md', 'new.md'] } }
    }
    if (request.method() === 'GET' && path === '/session/quiz-old') {
      oldSnapshotReads += 1
      return {
        body: {
          schema_version: 1,
          origin: 'standard',
          session_id: 'quiz-old',
          document_id: 'old.md',
          revision: 1,
          status: 'active',
          total: 3,
          answered_count: 0,
          questions: [
            { index: 0, question: '旧练习题目 1', options: ['A', 'B'], type: 'choice' },
            { index: 1, question: '旧练习题目 2', options: ['A', 'B'], type: 'choice' },
            { index: 2, question: '旧练习题目 3', options: ['A', 'B'], type: 'choice' },
          ],
          last_answer_index: null,
          last_user_answer: null,
          last_answer_result: null,
          result: null,
          grading_report: null,
          learning_report: null,
          expires_at: FUTURE_EXPIRES_AT,
          busy: false,
        },
      }
    }
    if (request.method() === 'POST' && path === '/session/start') {
      startBodies.push(await request.postDataJSON())
      return {
        body: {
          session_id: 'quiz-new',
          total: 1,
          questions: [{
            index: 0,
            question: '新练习题目',
            options: ['A', 'B'],
            type: 'choice',
          }],
          revision: 1,
          expires_at: FUTURE_EXPIRES_AT,
        },
      }
    }
    return null
  })

  const sameIntentNewLaunchUrl = '/quiz?document_id=old.md&topic=旧主题&launch_id=new-launch-1234'
  await page.goto(sameIntentNewLaunchUrl)

  const conflict = page.getByRole('region', { name: '检测到另一项练习' })
  await expect(conflict).toContainText('old.md / 旧主题')
  await expect(page.getByRole('button', { name: '放弃并开始新练习' })).toBeEnabled()
  expect(oldSnapshotReads).toBe(0)
  expect(startBodies).toEqual([])

  await page.getByRole('button', { name: '继续当前练习' }).click()
  await expect(page).toHaveURL(/\/quiz$/)
  await expect(page.getByText('旧练习题目 1')).toBeVisible()
  expect(oldSnapshotReads).toBe(1)

  const oversizedTopicUrl = '/quiz?document_id=new.md&topic='
    + encodeURIComponent('x'.repeat(4001))
    + '&launch_id=oversized-launch-1234'
  await page.goto(oversizedTopicUrl)
  await expect(conflict).toBeVisible()
  await expect(conflict).toContainText('新练习链接无效')
  await expect(page.getByRole('button', { name: '放弃并开始新练习' })).toBeDisabled()
  expect(
    await page.evaluate(key => JSON.parse(sessionStorage.getItem(key)), QUIZ_RECOVERY_KEY)
  ).toMatchObject({ session: { session_id: 'quiz-old' } })

  const newIntentUrl = '/quiz?document_id=new.md&topic=新主题&launch_id=new-launch-5678'
  await page.goto(newIntentUrl)
  await expect(conflict).toBeVisible()
  await page.getByRole('button', { name: '放弃并开始新练习' }).click()
  await expect(page.getByRole('combobox', { name: '学习文档' })).toHaveValue('new.md')
  await expect(page.getByRole('textbox', { name: /出题主题/ })).toHaveValue('新主题')
  expect(oldSnapshotReads).toBe(1)
  expect(await page.evaluate(key => sessionStorage.getItem(key), QUIZ_RECOVERY_KEY)).toBeNull()

  await page.getByRole('button', { name: '开始答题' }).click()
  await expect(page.getByText('新练习题目')).toBeVisible()
  expect(startBodies).toEqual([{
    document_id: 'new.md',
    description: '新主题',
    count: 5,
    difficulty: 'medium',
    type: 'choice',
    user_id: 'default_user',
  }])
  expect(
    await page.evaluate(key => JSON.parse(sessionStorage.getItem(key)), QUIZ_RECOVERY_KEY)
  ).toMatchObject({
    launch_id: 'new-launch-5678',
    launch_preset: { document_id: 'new.md', topic: '新主题' },
  })
  expect(unexpectedRequests).toEqual([])
  expect(problems).toEqual([])
})

test('a late answer cannot restore a quiz after accepting a new launch', async ({ page }) => {
  const problems = trackBrowserProblems(page)
  let releaseAnswer
  let markAnswerStarted
  const answerGate = new Promise(resolve => { releaseAnswer = resolve })
  const answerStarted = new Promise(resolve => { markAnswerStarted = resolve })
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['old.md', 'new.md'] } }
    }
    if (request.method() === 'POST' && path === '/session/start') {
      const body = await request.postDataJSON()
      const isNew = body.document_id === 'new.md'
      return {
        body: {
          session_id: isNew ? 'quiz-new-after-late-answer' : 'quiz-old-in-flight',
          total: 1,
          questions: [{
            index: 0,
            question: isNew ? '新目标题目' : '旧目标题目',
            options: ['A', 'B'],
            type: 'choice',
          }],
          revision: 1,
          expires_at: FUTURE_EXPIRES_AT,
        },
      }
    }
    if (request.method() === 'POST' && path === '/session/quiz-old-in-flight/answer') {
      markAnswerStarted()
      await answerGate
      return {
        body: {
          evaluation_status: 'final',
          correct: true,
          correct_answer: 'A',
          explanation: '这条旧反馈不得复活',
          is_last: true,
          next_index: null,
          revision: 2,
          expires_at: FUTURE_EXPIRES_AT,
        },
      }
    }
    return null
  })

  try {
    await page.goto('/quiz')
    await page.getByRole('combobox', { name: '学习文档' }).selectOption('old.md')
    await page.getByRole('button', { name: '开始答题' }).click()
    await page.getByRole('button', { name: 'A' }).click()
    await page.getByRole('button', { name: '提交答案' }).click()
    await answerStarted

    const oldAnswerAborted = waitForAbortedApiRequest(
      page,
      'POST',
      '/session/quiz-old-in-flight/answer',
    )
    const nextLaunch = '/quiz?document_id=new.md&topic=新目标&launch_id=late-answer-launch-1234'
    await page.evaluate(url => {
      history.pushState({}, '', url)
      dispatchEvent(new PopStateEvent('popstate'))
    }, nextLaunch)
    await expect(page.getByRole('region', { name: '检测到另一项练习' })).toBeVisible()
    await page.getByRole('button', { name: '放弃并开始新练习' }).click()
    await expect(page.getByRole('combobox', { name: '学习文档' })).toHaveValue('new.md')
    expectBrowserAbort(await oldAnswerAborted)

    releaseAnswer()
    await expect(page.getByText('这条旧反馈不得复活')).toHaveCount(0)
    expect(await page.evaluate(key => sessionStorage.getItem(key), QUIZ_RECOVERY_KEY)).toBeNull()

    await page.getByRole('button', { name: '开始答题' }).click()
    await expect(page.getByText('新目标题目')).toBeVisible()
    expect(
      await page.evaluate(key => JSON.parse(sessionStorage.getItem(key)), QUIZ_RECOVERY_KEY)
    ).toMatchObject({
      launch_id: 'late-answer-launch-1234',
      session: { session_id: 'quiz-new-after-late-answer' },
    })
    expect(unexpectedRequests).toEqual([])
    expect(problems).toEqual([])
  } finally {
    releaseAnswer()
  }
})

test('quiz preset rejects a document that is no longer available', async ({ page }) => {
  const problems = trackBrowserProblems(page)
  let startRequests = 0
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md'] } }
    }
    if (request.method() === 'POST' && path === '/session/start') {
      startRequests += 1
      return { status: 500, body: { detail: '不应发起出题请求' } }
    }
    return null
  })

  await page.goto('/quiz?document_id=deleted.md&topic=RRF')

  await expect(page.getByRole('alert')).toContainText('学习文档不存在或已被删除')
  await expect(page.getByRole('combobox', { name: '学习文档' })).toHaveValue('')
  await expect(page.getByRole('textbox', { name: /出题主题/ })).toHaveValue('RRF')
  await expect(page.getByRole('button', { name: '开始答题' })).toBeDisabled()
  expect(startRequests).toBe(0)

  await page.getByRole('link', { name: '答题练习' }).click()
  await expect(page).toHaveURL(/\/quiz$/)
  await expect(page.getByRole('alert')).toHaveCount(0)
  await expect(page.getByRole('textbox', { name: /出题主题/ })).toHaveValue('')
  await expect(page.getByRole('combobox', { name: '学习文档' })).toHaveValue('')
  expect(unexpectedRequests).toEqual([])
  expect(problems).toEqual([])
})

test('adaptive loads documents on entry and refresh clears the recovered error', async ({ page }) => {
  let failDocuments = true
  let documentRequests = 0
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      documentRequests += 1
      return failDocuments
        ? { status: 503, body: { detail: '文档列表刷新失败' } }
        : { body: { documents: ['notes.md'] } }
    }
    return null
  })

  await page.goto('/adaptive')
  const refresh = page.getByRole('button', { name: '刷新文档' })

  await expect(page.getByRole('alert')).toContainText('文档列表刷新失败')
  await expect(page.getByLabel('学习目标')).toBeEditable()

  failDocuments = false
  await refresh.click()
  await expect(page.getByRole('alert')).toHaveCount(0)
  await expect(refresh).toBeEnabled()
  expect(documentRequests).toBeGreaterThanOrEqual(2)
  expect(unexpectedRequests).toEqual([])
})

test('adaptive start response loss retries the same intent with the same key', async ({ page }) => {
  const starts = []
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'POST' && path === '/agent/adaptive/start') {
      starts.push({
        body: await request.postDataJSON(),
        key: await request.headerValue('idempotency-key'),
      })
      if (starts.length === 1) return { abort: 'failed' }
      return {
        body: adaptiveSnapshot({ adaptive_session_id: 'adapt-start-replay' }),
      }
    }
    return null
  })

  await page.goto('/adaptive')
  await expect(page.getByLabel('用户 ID')).toHaveCount(0)
  await page.getByLabel('学习目标').fill('学习首字母')
  await page.getByLabel('文档 ID').fill('notes.md')
  await page.getByRole('button', { name: '开始自适应辅导' }).click()

  await expect(page.getByRole('alert')).toBeVisible()
  await expect(page.getByLabel('学习目标')).toBeDisabled()
  await page.reload()
  await expect(page.getByText('请选择首字母')).toBeVisible()

  expect(starts).toHaveLength(2)
  expect(starts[0].body).toEqual({
    user_id: 'default_user',
    document_id: 'notes.md',
    goal: '学习首字母',
  })
  expect(starts[1].body).toEqual(starts[0].body)
  expect(starts[0].key).toMatch(UUID_V4_PATTERN)
  expect(starts[1].key).toBe(starts[0].key)
  expect(unexpectedRequests).toEqual([])
})

test('adaptive submit response loss survives refresh with an identical key and body', async ({ page }) => {
  const active = adaptiveSnapshot({ adaptive_session_id: 'adapt-submit-replay' })
  const completed = adaptiveSnapshot({
    adaptive_session_id: 'adapt-submit-replay',
    done: true,
    questions: [],
    summary: '已安全完成本轮',
    terminate_reason: 'agent_finish',
    decision: {
      ...active.decision,
      action: 'finish',
      reason: '本轮目标已完成',
    },
    revision: 2,
  })
  const submits = []
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'POST' && path === '/agent/adaptive/start') {
      return { body: active }
    }
    if (request.method() === 'GET' && path === '/agent/adaptive/adapt-submit-replay') {
      return { body: active }
    }
    if (request.method() === 'POST' && path === '/agent/adaptive/submit') {
      submits.push({
        body: await request.postDataJSON(),
        key: await request.headerValue('idempotency-key'),
      })
      if (submits.length === 1) return { abort: 'failed' }
      return { body: completed }
    }
    return null
  })

  await page.goto('/adaptive')
  await page.getByLabel('学习目标').fill('学习首字母')
  await page.getByLabel('文档 ID').fill('notes.md')
  await page.getByRole('button', { name: '开始自适应辅导' }).click()
  await page.getByRole('radio', { name: 'Alpha' }).check()
  await page.getByRole('button', { name: /提交本轮/ }).click()

  await expect(page.getByRole('alert')).toBeVisible()
  await expect(page.getByRole('radio', { name: 'Alpha' })).toBeDisabled()
  const pendingBeforeReload = await page.evaluate(key => (
    JSON.parse(sessionStorage.getItem(key)).pending_submit
  ), ADAPTIVE_RECOVERY_KEY)

  await page.reload()
  await expect(page.getByText('已安全完成本轮')).toBeVisible()

  expect(submits).toHaveLength(2)
  expect(submits[0].body).toEqual({
    adaptive_session_id: 'adapt-submit-replay',
    answers: ['Alpha'],
    turn: 1,
    revision: 1,
  })
  expect(submits[1].body).toEqual(submits[0].body)
  expect(submits[0].key).toMatch(UUID_V4_PATTERN)
  expect(submits[1].key).toBe(submits[0].key)
  expect(pendingBeforeReload.body).toEqual(submits[0].body)
  expect(pendingBeforeReload.idempotency_key).toBe(submits[0].key)
  expect(unexpectedRequests).toEqual([])
})

test('adaptive start deadline retains the exact intent and key for UI retry', async ({ page }) => {
  await page.clock.install()
  const attempts = []
  let releaseFirst
  let markFirstStarted
  const firstGate = new Promise(resolve => { releaseFirst = resolve })
  const firstStarted = new Promise(resolve => { markFirstStarted = resolve })
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'POST' && path === '/agent/adaptive/start') {
      attempts.push({
        key: await request.headerValue('idempotency-key'),
        body: await request.postDataJSON(),
        rawBody: request.postData(),
      })
      if (attempts.length === 1) {
        markFirstStarted()
        await firstGate
        return {
          body: adaptiveSnapshot({
            adaptive_session_id: 'adapt-start-too-late',
            questions: [{
              index: 0,
              question: '不应采用的迟到自适应题目',
              options: ['旧答案'],
              type: 'choice',
            }],
          }),
        }
      }
      return {
        body: adaptiveSnapshot({
          adaptive_session_id: 'adapt-start-after-timeout',
          questions: [{
            index: 0,
            question: '超时重试后的自适应题目',
            options: ['新答案'],
            type: 'choice',
          }],
        }),
      }
    }
    return null
  })

  try {
    await page.goto('/adaptive')
    await page.getByLabel('学习目标').fill('验证自适应超时恢复')
    await page.getByLabel('文档 ID').fill('notes.md')
    const firstAborted = waitForAbortedApiRequest(page, 'POST', '/agent/adaptive/start')
    await page.getByRole('button', { name: '开始自适应辅导' }).click()
    await firstStarted
    await page.clock.fastForward(RECOVERY_DEADLINE_ADVANCE_MS)
    await expect(page.getByRole('alert')).toContainText(RECOVERY_TIMEOUT_MESSAGE)
    expectBrowserAbort(await firstAborted)
    const retry = page.getByRole('button', { name: '重试恢复' })
    await expect(retry).toBeEnabled()
    const pending = await page.evaluate(
      key => JSON.parse(sessionStorage.getItem(key)),
      ADAPTIVE_RECOVERY_KEY,
    )
    expect(pending).toMatchObject({
      intent: {
        user_id: 'default_user',
        document_id: 'notes.md',
        goal: '验证自适应超时恢复',
      },
      session: null,
      pending_submit: null,
    })
    expect(pending.start_idempotency_key).toBe(attempts[0].key)

    releaseFirst()
    await retry.click()
    await expect(page.getByText('超时重试后的自适应题目')).toBeVisible()
    await expect(page.getByText('不应采用的迟到自适应题目')).toHaveCount(0)
    expect(attempts).toHaveLength(2)
    expect(attempts[0]).toEqual(attempts[1])
    expect(attempts[0].key).toMatch(UUID_V4_PATTERN)
    expect(unexpectedRequests).toEqual([])
  } finally {
    releaseFirst()
  }
})

test('adaptive submit deadline preserves its frozen turn for exact replay', async ({ page }) => {
  await page.clock.install()
  const active = adaptiveSnapshot({ adaptive_session_id: 'adapt-submit-timeout' })
  const completed = adaptiveSnapshot({
    adaptive_session_id: 'adapt-submit-timeout',
    done: true,
    questions: [],
    summary: '冻结提交已用原请求安全恢复',
    terminate_reason: 'agent_finish',
    decision: {
      ...active.decision,
      action: 'finish',
      reason: '本轮目标已完成',
    },
    revision: 2,
  })
  const submits = []
  let snapshotReads = 0
  let releaseFirstRead
  let markFirstReadStarted
  let releaseFirst
  let markFirstStarted
  const firstReadGate = new Promise(resolve => { releaseFirstRead = resolve })
  const firstReadStarted = new Promise(resolve => { markFirstReadStarted = resolve })
  const firstGate = new Promise(resolve => { releaseFirst = resolve })
  const firstStarted = new Promise(resolve => { markFirstStarted = resolve })
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'POST' && path === '/agent/adaptive/start') {
      return { body: active }
    }
    if (request.method() === 'GET' && path === '/agent/adaptive/adapt-submit-timeout') {
      snapshotReads += 1
      if (snapshotReads === 1) {
        markFirstReadStarted()
        await firstReadGate
      }
      return { body: active }
    }
    if (request.method() === 'POST' && path === '/agent/adaptive/submit') {
      submits.push({
        key: await request.headerValue('idempotency-key'),
        body: await request.postDataJSON(),
        rawBody: request.postData(),
      })
      if (submits.length === 1) {
        markFirstStarted()
        await firstGate
      }
      return { body: completed }
    }
    return null
  })

  try {
    await page.goto('/adaptive')
    await page.getByLabel('学习目标').fill('验证冻结提交')
    await page.getByLabel('文档 ID').fill('notes.md')
    await page.getByRole('button', { name: '开始自适应辅导' }).click()
    await page.getByRole('radio', { name: 'Alpha' }).check()
    const firstAborted = waitForAbortedApiRequest(page, 'POST', '/agent/adaptive/submit')
    await page.getByRole('button', { name: /提交本轮/ }).click()
    await firstReadStarted
    await page.clock.fastForward(119_000)
    releaseFirstRead()
    await firstStarted
    await page.clock.fastForward(1_001)
    await expect(page.getByRole('alert')).toContainText(RECOVERY_TIMEOUT_MESSAGE)
    expectBrowserAbort(await firstAborted)
    const retry = page.getByRole('button', { name: '重试恢复' })
    await expect(retry).toBeEnabled()
    const pending = await page.evaluate(key => (
      JSON.parse(sessionStorage.getItem(key)).pending_submit
    ), ADAPTIVE_RECOVERY_KEY)
    expect(pending.body).toEqual({
      adaptive_session_id: 'adapt-submit-timeout',
      answers: ['Alpha'],
      turn: 1,
      revision: 1,
    })
    expect(pending.idempotency_key).toBe(submits[0].key)

    releaseFirst()
    await retry.click()
    await expect(page.getByText('冻结提交已用原请求安全恢复')).toBeVisible()
    expect(submits).toHaveLength(2)
    expect(submits[0]).toEqual(submits[1])
    expect(submits[0].key).toMatch(UUID_V4_PATTERN)
    expect(snapshotReads).toBe(2)
    expect(unexpectedRequests).toEqual([])
  } finally {
    releaseFirstRead()
    releaseFirst()
  }
})

test('adaptive snapshot deadline leaves durable recovery untouched until retry', async ({ page }) => {
  await page.clock.install()
  const active = adaptiveSnapshot({ adaptive_session_id: 'adapt-get-timeout' })
  const late = adaptiveSnapshot({
    adaptive_session_id: 'adapt-get-timeout',
    questions: [{
      index: 0,
      question: '不应采用的迟到 Adaptive GET',
      options: ['旧答案'],
      type: 'choice',
    }],
  })
  const canonical = adaptiveSnapshot({
    adaptive_session_id: 'adapt-get-timeout',
    questions: [{
      index: 0,
      question: 'Adaptive GET 重试成功',
      options: ['新答案'],
      type: 'choice',
    }],
  })
  const recovery = adaptiveRecovery(active)
  await page.addInitScript(({ key, value }) => {
    sessionStorage.setItem(key, JSON.stringify(value))
  }, { key: ADAPTIVE_RECOVERY_KEY, value: recovery })
  let reads = 0
  let releaseFirst
  let markFirstStarted
  const firstGate = new Promise(resolve => { releaseFirst = resolve })
  const firstStarted = new Promise(resolve => { markFirstStarted = resolve })
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/agent/adaptive/adapt-get-timeout') {
      reads += 1
      if (reads === 1) {
        markFirstStarted()
        await firstGate
        return { body: late }
      }
      return { body: canonical }
    }
    return null
  })

  try {
    const firstAborted = waitForAbortedApiRequest(
      page,
      'GET',
      '/agent/adaptive/adapt-get-timeout',
    )
    await page.goto('/adaptive')
    await firstStarted
    await page.clock.fastForward(RECOVERY_DEADLINE_ADVANCE_MS)
    await expect(page.getByRole('alert')).toContainText(RECOVERY_TIMEOUT_MESSAGE)
    expectBrowserAbort(await firstAborted)
    const retry = page.getByRole('button', { name: '重试恢复' })
    await expect(retry).toBeEnabled()
    expect(
      await page.evaluate(
        key => JSON.parse(sessionStorage.getItem(key)),
        ADAPTIVE_RECOVERY_KEY,
      ),
    ).toEqual(recovery)

    releaseFirst()
    await retry.click()
    await expect(page.getByText('Adaptive GET 重试成功')).toBeVisible()
    await expect(page.getByText('不应采用的迟到 Adaptive GET')).toHaveCount(0)
    expect(reads).toBe(2)
    expect(unexpectedRequests).toEqual([])
  } finally {
    releaseFirst()
  }
})

test('adaptive GET restores quiz, lesson, and readable completed learning path states', async ({ page }) => {
  const quiz = adaptiveSnapshot({ adaptive_session_id: 'adapt-get-quiz' })
  const lesson = adaptiveSnapshot({
    adaptive_session_id: 'adapt-get-teach',
    turn_type: 'teach',
    questions: [],
    lesson: '先比较两个算法的递归边界，再观察合并步骤。',
    decision: {
      ...quiz.decision,
      action: 'teach',
      topic: '递归边界',
    },
  })
  const done = adaptiveSnapshot({
    adaptive_session_id: 'adapt-get-done',
    done: true,
    questions: [],
    summary: '建议按新路径继续练习。',
    terminate_reason: 'switch_to_plan',
    decision: {
      ...quiz.decision,
      action: 'switch_to_plan',
      topic: '排序算法',
    },
    learning_path: {
      document_id: 'notes.md',
      title: '排序算法强化路径',
      total_stages: 2,
      stages: [{
        stage: 1,
        title: '比较排序基础',
        topics: ['快速排序'],
        description: '先掌握分区和递归边界。',
        estimated_minutes: 20,
      }, {
        stage: 2,
        title: '稳定性与复杂度',
        topics: ['归并排序'],
        description: '比较稳定性和空间复杂度。',
        estimated_minutes: 25,
      }],
    },
    learning_path_id: 'lp_cccccccccccccccccccccccccccccccc',
    revision: 4,
  })
  const snapshots = new Map([
    [quiz.adaptive_session_id, quiz],
    [lesson.adaptive_session_id, lesson],
    [done.adaptive_session_id, done],
  ])
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path.startsWith('/agent/adaptive/')) {
      return { body: snapshots.get(path.split('/').at(-1)) }
    }
    return null
  })

  await page.goto('/adaptive')

  for (const snapshot of [quiz, lesson, done]) {
    await page.evaluate(({ key, value }) => {
      sessionStorage.setItem(key, JSON.stringify(value))
    }, {
      key: ADAPTIVE_RECOVERY_KEY,
      value: adaptiveRecovery(snapshot),
    })
    await page.reload()

    if (snapshot === quiz) {
      await expect(page.getByText('请选择首字母')).toBeVisible()
    } else if (snapshot === lesson) {
      await expect(page.getByText('先比较两个算法的递归边界，再观察合并步骤。')).toBeVisible()
    } else {
      await expect(page.getByText('排序算法强化路径')).toBeVisible()
      await expect(page.getByText('先掌握分区和递归边界。')).toBeVisible()
      await expect(page.locator('.learning-path pre')).toHaveCount(0)
      await expect(page.getByRole('link', { name: '查看学习路径' })).toHaveAttribute(
        'href',
        '/learning-path?path_id=lp_cccccccccccccccccccccccccccccccc',
      )
      await expect(page.getByRole('link', { name: '从第一阶段开始练习' }))
        .toHaveCount(0)
    }
  }

  expect(unexpectedRequests).toEqual([])
})

test('adaptive recovery accepts a canonical completed path before replaying a pending submit', async ({ page }) => {
  const resource = learningPathResource({
    id: 'lp_dddddddddddddddddddddddddddddddd',
    title: '丢失响应后的持久路径',
  })
  const active = adaptiveSnapshot({ adaptive_session_id: 'adapt-path-committed' })
  const done = adaptiveSnapshot({
    adaptive_session_id: active.adaptive_session_id,
    done: true,
    questions: [],
    summary: '服务端已经提交持久学习路径。',
    terminate_reason: 'switch_to_plan',
    decision: {
      ...active.decision,
      action: 'switch_to_plan',
      topic: '混合检索',
    },
    learning_path: resource.path,
    learning_path_id: resource.learning_path_id,
    revision: 2,
  })
  const pending = {
    idempotency_key: 'adaptive-path-submit-key',
    body: {
      adaptive_session_id: active.adaptive_session_id,
      answers: ['Alpha'],
      turn: 1,
      revision: 1,
    },
  }
  let submitAttempts = 0
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (
      request.method() === 'GET'
      && path === `/agent/adaptive/${active.adaptive_session_id}`
    ) {
      return { body: done }
    }
    if (request.method() === 'POST' && path === '/agent/adaptive/submit') {
      submitAttempts += 1
      return { body: done }
    }
    return null
  })
  await page.addInitScript(({ key, value }) => {
    sessionStorage.setItem(key, JSON.stringify(value))
  }, {
    key: ADAPTIVE_RECOVERY_KEY,
    value: adaptiveRecovery(active, pending),
  })

  await page.goto('/adaptive')

  await expect(page.getByText('丢失响应后的持久路径')).toBeVisible()
  await expect(page.getByRole('link', { name: '查看学习路径' })).toHaveAttribute(
    'href',
    `/learning-path?path_id=${resource.learning_path_id}`,
  )
  const stored = await page.evaluate(
    key => JSON.parse(sessionStorage.getItem(key)),
    ADAPTIVE_RECOVERY_KEY,
  )
  expect(stored.pending_submit).toBeNull()
  expect(stored.snapshot.learning_path_id).toBe(resource.learning_path_id)
  expect(submitAttempts).toBe(0)
  expect(unexpectedRequests).toEqual([])
})

test('adaptive never trusts a cached path pointer before the canonical GET returns', async ({ page }) => {
  const cachedResource = learningPathResource({
    id: 'lp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
    title: '缓存中的旧路径',
  })
  const canonicalResource = learningPathResource({
    id: 'lp_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
    title: '服务端权威路径',
  })
  const completed = resource => adaptiveSnapshot({
    adaptive_session_id: 'adapt-canonical-path',
    done: true,
    questions: [],
    summary: '建议转入持久学习路径。',
    terminate_reason: 'switch_to_plan',
    decision: {
      ...adaptiveSnapshot().decision,
      action: 'switch_to_plan',
      topic: '检索',
    },
    learning_path: resource.path,
    learning_path_id: resource.learning_path_id,
    revision: 2,
  })
  const cached = completed(cachedResource)
  const canonical = completed(canonicalResource)
  let releaseGet
  const getBlocked = new Promise(resolve => { releaseGet = resolve })
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (
      request.method() === 'GET'
      && path === `/agent/adaptive/${cached.adaptive_session_id}`
    ) {
      await getBlocked
      return { body: canonical }
    }
    return null
  })
  await page.addInitScript(({ key, value }) => {
    sessionStorage.setItem(key, JSON.stringify(value))
  }, {
    key: ADAPTIVE_RECOVERY_KEY,
    value: adaptiveRecovery(cached),
  })

  await page.goto('/adaptive')
  await expect(page.getByText('正在同步学习进度...')).toBeVisible()
  await expect(page.getByText('缓存中的旧路径')).toHaveCount(0)
  await expect(page.getByRole('link', { name: '查看学习路径' })).toHaveCount(0)
  await expect(page.getByRole('link', { name: '前往学习路径重新生成' })).toHaveCount(0)

  releaseGet()
  await expect(page.getByText('服务端权威路径')).toBeVisible()
  await expect(page.getByText('缓存中的旧路径')).toHaveCount(0)
  await expect(page.getByRole('link', { name: '查看学习路径' })).toHaveAttribute(
    'href',
    `/learning-path?path_id=${canonicalResource.learning_path_id}`,
  )
  const stored = await page.evaluate(
    key => JSON.parse(sessionStorage.getItem(key)),
    ADAPTIVE_RECOVERY_KEY,
  )
  expect(stored.snapshot.learning_path_id).toBe(canonicalResource.learning_path_id)
  expect(
    await page.evaluate(key => sessionStorage.getItem(key), LEARNING_PATH_RECOVERY_KEY),
  ).toBeNull()
  expect(unexpectedRequests).toEqual([])
})

test('adaptive keeps a confirmed legacy path preview out of an unbound quiz', async ({ page }) => {
  const resource = learningPathResource({ title: '历史路径预览' })
  const legacy = adaptiveSnapshot({
    adaptive_session_id: 'adapt-legacy-path',
    done: true,
    questions: [],
    summary: '这条历史记录生成于持久路径上线之前。',
    terminate_reason: 'switch_to_plan',
    decision: {
      ...adaptiveSnapshot().decision,
      action: 'switch_to_plan',
      topic: '历史主题',
    },
    learning_path: resource.path,
    learning_path_id: null,
    revision: 2,
  })
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (
      request.method() === 'GET'
      && path === `/agent/adaptive/${legacy.adaptive_session_id}`
    ) {
      return { body: legacy }
    }
    return null
  })
  await page.addInitScript(({ key, value }) => {
    sessionStorage.setItem(key, JSON.stringify(value))
  }, {
    key: ADAPTIVE_RECOVERY_KEY,
    value: adaptiveRecovery(legacy),
  })

  await page.goto('/adaptive')

  await expect(page.getByText('历史路径预览')).toBeVisible()
  await expect(page.getByText('这是旧版路径预览，阶段进度尚未绑定。')).toBeVisible()
  await expect(page.getByRole('link', { name: '前往学习路径重新生成' }))
    .toHaveAttribute('href', '/learning-path')
  await expect(page.getByRole('link', { name: '从第一阶段开始练习' })).toHaveCount(0)
  expect(unexpectedRequests).toEqual([])
})

test('adaptive rejects invalid, non-terminal, and cross-document path pointers', async ({ page }) => {
  const resource = learningPathResource()
  const validDone = adaptiveSnapshot({
    adaptive_session_id: 'adapt-invalid-path',
    done: true,
    questions: [],
    summary: '无效路径不应被恢复。',
    terminate_reason: 'switch_to_plan',
    decision: {
      ...adaptiveSnapshot().decision,
      action: 'switch_to_plan',
      topic: '校验路径',
    },
    learning_path: resource.path,
    learning_path_id: resource.learning_path_id,
    revision: 2,
  })
  const invalidRecoveries = [
    adaptiveRecovery({ ...validDone, learning_path_id: 'lp_invalid' }),
    adaptiveRecovery({
      ...adaptiveSnapshot({ adaptive_session_id: 'adapt-active-with-path' }),
      learning_path_id: resource.learning_path_id,
    }),
    adaptiveRecovery({
      ...validDone,
      adaptive_session_id: 'adapt-wrong-document',
      learning_path: { ...resource.path, document_id: 'other.md' },
    }),
  ]
  let adaptiveReads = 0
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path.startsWith('/agent/adaptive/')) {
      adaptiveReads += 1
    }
    return null
  })
  await page.goto('/adaptive')

  for (const recovery of invalidRecoveries) {
    await page.evaluate(({ key, value }) => {
      sessionStorage.setItem(key, JSON.stringify(value))
    }, { key: ADAPTIVE_RECOVERY_KEY, value: recovery })
    await page.reload()
    await expect(page.getByLabel('学习目标')).toBeEditable()
    await expect(page.getByRole('link', { name: '查看学习路径' })).toHaveCount(0)
    expect(
      await page.evaluate(key => sessionStorage.getItem(key), ADAPTIVE_RECOVERY_KEY),
    ).toBeNull()
  }

  expect(adaptiveReads).toBe(0)
  expect(unexpectedRequests).toEqual([])
})

test('adaptive opens the canonical path before launching a stage-bound quiz', async ({ page }) => {
  const resource = learningPathResource({
    id: 'lp_eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee',
    title: 'Adaptive 持久路径',
    stageTitle: '权威阶段',
    topics: ['服务端绑定'],
  })
  const done = adaptiveSnapshot({
    adaptive_session_id: 'adapt-bound-path',
    done: true,
    questions: [],
    summary: '通过权威学习路径继续。',
    terminate_reason: 'switch_to_plan',
    decision: {
      ...adaptiveSnapshot().decision,
      action: 'switch_to_plan',
      topic: '服务端绑定',
    },
    learning_path: resource.path,
    learning_path_id: resource.learning_path_id,
    revision: 2,
  })
  let startBody = null
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md'] } }
    }
    if (
      request.method() === 'GET'
      && path === `/agent/adaptive/${done.adaptive_session_id}`
    ) {
      return { body: done }
    }
    if (
      request.method() === 'GET'
      && path === `/learning-paths/${resource.learning_path_id}`
    ) {
      return { body: resource }
    }
    if (request.method() === 'POST' && path === '/session/start') {
      startBody = await request.postDataJSON()
      return {
        body: {
          session_id: 'quiz-from-adaptive-path',
          total: 1,
          questions: [{
            index: 0,
            question: '持久路径绑定成功了吗？',
            options: ['是', '否'],
            type: 'choice',
          }],
          revision: 1,
          expires_at: FUTURE_EXPIRES_AT,
          learning_path_source: {
            learning_path_id: resource.learning_path_id,
            stage_id: 1,
          },
        },
      }
    }
    return null
  })
  await page.addInitScript(({ key, value }) => {
    sessionStorage.setItem(key, JSON.stringify(value))
  }, {
    key: ADAPTIVE_RECOVERY_KEY,
    value: adaptiveRecovery(done),
  })

  await page.goto('/adaptive')
  await page.getByRole('link', { name: '查看学习路径' }).click()
  await expect(page).toHaveURL(
    new RegExp(`/learning-path\\?path_id=${resource.learning_path_id}`),
  )
  await expect(page.getByRole('heading', { name: 'Adaptive 持久路径' })).toBeVisible()
  await page.getByRole('button', { name: '练习阶段 1：权威阶段' }).click()

  const quizUrl = new URL(page.url())
  expect(quizUrl.pathname).toBe('/quiz')
  expect(quizUrl.searchParams.get('path_id')).toBe(resource.learning_path_id)
  expect(quizUrl.searchParams.get('stage_id')).toBe('1')
  expect(quizUrl.searchParams.get('launch_id')).toMatch(UUID_V4_PATTERN)
  await page.getByRole('button', { name: '开始答题' }).click()
  await expect(page.getByText('持久路径绑定成功了吗？')).toBeVisible()
  expect(startBody.learning_path_source).toEqual({
    learning_path_id: resource.learning_path_id,
    stage_id: 1,
  })
  expect(unexpectedRequests).toEqual([])
})

test('adaptive clears a corrupt persisted snapshot before rendering it', async ({ page }) => {
  const corrupt = adaptiveSnapshot({
    adaptive_session_id: 'adapt-corrupt',
    decision: {
      ...adaptiveSnapshot().decision,
      difficulty_score: 'not-a-number',
    },
  })
  let snapshotRequests = 0
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path.startsWith('/agent/adaptive/')) {
      snapshotRequests += 1
      return { status: 500, body: { detail: '不应读取损坏会话' } }
    }
    return null
  })
  await page.addInitScript(({ key, value }) => {
    sessionStorage.setItem(key, JSON.stringify(value))
  }, {
    key: ADAPTIVE_RECOVERY_KEY,
    value: adaptiveRecovery(corrupt),
  })

  await page.goto('/adaptive')
  await expect(page.getByLabel('学习目标')).toBeEditable()
  await expect(page.getByRole('button', { name: '开始自适应辅导' })).toBeVisible()
  await expect.poll(() => page.evaluate(key => sessionStorage.getItem(key), ADAPTIVE_RECOVERY_KEY))
    .toBeNull()
  expect(snapshotRequests).toBe(0)
  expect(unexpectedRequests).toEqual([])
})

test('adaptive busy and stale recovery keeps frozen answers then hydrates canonical progress', async ({ page }) => {
  const active = adaptiveSnapshot({ adaptive_session_id: 'adapt-stale' })
  const advanced = adaptiveSnapshot({
    adaptive_session_id: 'adapt-stale',
    turn: 2,
    revision: 3,
    questions: [{
      index: 0,
      question: '服务端第二轮题目',
      options: ['继续', '结束'],
      type: 'choice',
    }],
    decision: {
      ...active.decision,
      topic: '第二轮',
    },
    trajectory: [active.trajectory[0], {
      ...active.trajectory[0],
      turn: 2,
      topic: '第二轮',
    }],
  })
  const pending = {
    idempotency_key: 'adaptive-submit-key-1234',
    body: {
      adaptive_session_id: 'adapt-stale',
      answers: ['Alpha'],
      turn: 1,
      revision: 1,
    },
  }
  let getCount = 0
  const submitted = []
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/agent/adaptive/adapt-stale') {
      getCount += 1
      if (getCount === 1) return { body: { ...active, busy: true } }
      if (getCount === 2) return { body: active }
      return { body: advanced }
    }
    if (request.method() === 'POST' && path === '/agent/adaptive/submit') {
      submitted.push({
        body: await request.postDataJSON(),
        key: await request.headerValue('idempotency-key'),
      })
      return {
        status: 409,
        body: {
          detail: '进度已变化',
          code: 'adaptive_session_stale',
          reason: 'stale',
        },
      }
    }
    return null
  })
  await page.addInitScript(({ key, value }) => {
    sessionStorage.setItem(key, JSON.stringify(value))
  }, {
    key: ADAPTIVE_RECOVERY_KEY,
    value: adaptiveRecovery(active, pending),
  })

  await page.goto('/adaptive')
  await expect(page.getByRole('radio', { name: 'Alpha' })).toBeDisabled()
  await expect(page.getByText('服务端第二轮题目')).toBeVisible()

  expect(submitted).toEqual([{
    body: pending.body,
    key: pending.idempotency_key,
  }])
  const recovered = await page.evaluate(key => (
    JSON.parse(sessionStorage.getItem(key))
  ), ADAPTIVE_RECOVERY_KEY)
  expect(recovered.pending_submit).toBeNull()
  expect(recovered.snapshot.turn).toBe(2)
  expect(getCount).toBeGreaterThanOrEqual(3)
  expect(unexpectedRequests).toEqual([])
})

test('adaptive same-turn stale recovery preserves the original pending request', async ({ page }) => {
  const active = adaptiveSnapshot({ adaptive_session_id: 'adapt-partial-stale' })
  const checkpointed = { ...active, revision: 2 }
  const completed = adaptiveSnapshot({
    adaptive_session_id: 'adapt-partial-stale',
    done: true,
    questions: [],
    summary: '已从服务端检查点继续完成',
    terminate_reason: 'agent_finish',
    decision: {
      ...active.decision,
      action: 'finish',
      reason: '已完成',
    },
    revision: 3,
  })
  const pending = {
    idempotency_key: 'adaptive-partial-stale-key',
    body: {
      adaptive_session_id: 'adapt-partial-stale',
      answers: ['Alpha'],
      turn: 1,
      revision: 1,
    },
  }
  let getCount = 0
  const submits = []
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/agent/adaptive/adapt-partial-stale') {
      getCount += 1
      return { body: getCount === 1 ? active : checkpointed }
    }
    if (request.method() === 'POST' && path === '/agent/adaptive/submit') {
      submits.push({
        body: await request.postDataJSON(),
        key: await request.headerValue('idempotency-key'),
      })
      if (submits.length === 1) {
        return {
          status: 409,
          body: {
            detail: '处理租约已变化',
            code: 'adaptive_session_stale',
            reason: 'stale',
          },
        }
      }
      return { body: completed }
    }
    return null
  })
  await page.addInitScript(({ key, value }) => {
    sessionStorage.setItem(key, JSON.stringify(value))
  }, {
    key: ADAPTIVE_RECOVERY_KEY,
    value: adaptiveRecovery(active, pending),
  })

  await page.goto('/adaptive')
  await expect(page.getByText('已从服务端检查点继续完成')).toBeVisible()

  expect(submits).toHaveLength(2)
  expect(submits[0]).toEqual({ body: pending.body, key: pending.idempotency_key })
  expect(submits[1]).toEqual(submits[0])
  expect(getCount).toBeGreaterThanOrEqual(2)
  expect(unexpectedRequests).toEqual([])
})

test('adaptive rejected start clears its receipt before a corrected request', async ({ page }) => {
  const corrected = adaptiveSnapshot({ adaptive_session_id: 'adapt-corrected-start' })
  const starts = []
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'POST' && path === '/agent/adaptive/start') {
      starts.push({
        key: await request.headerValue('idempotency-key'),
        body: await request.postDataJSON(),
      })
      if (starts.length === 1) {
        return { status: 400, body: { detail: '学习目标不符合要求' } }
      }
      return { body: corrected }
    }
    return null
  })

  await page.goto('/adaptive')
  await page.getByLabel('学习目标').fill('需要修正的目标')
  await page.getByLabel('文档 ID').fill('notes.md')
  await page.getByRole('button', { name: '开始自适应辅导' }).click()

  await expect(page.getByRole('alert')).toContainText('学习目标不符合要求')
  await expect(page.getByLabel('学习目标')).toBeEditable()
  await expect(page.getByLabel('学习目标')).toHaveValue('需要修正的目标')
  await expect.poll(() => (
    page.evaluate(key => sessionStorage.getItem(key), ADAPTIVE_RECOVERY_KEY)
  )).toBeNull()

  await page.getByLabel('学习目标').fill('修正后的目标')
  await page.getByRole('button', { name: '开始自适应辅导' }).click()
  await expect(page.getByText('请选择首字母')).toBeVisible()
  expect(starts).toHaveLength(2)
  expect(starts[1].key).not.toBe(starts[0].key)
  expect(starts[1].body.goal).toBe('修正后的目标')
  expect(unexpectedRequests).toEqual([])
})

test('adaptive rejected submit unlocks the canonical turn for a corrected request', async ({ page }) => {
  const active = adaptiveSnapshot({ adaptive_session_id: 'adapt-rejected' })
  const completed = adaptiveSnapshot({
    adaptive_session_id: 'adapt-rejected',
    done: true,
    questions: [],
    summary: '修正后的答案已提交',
    terminate_reason: 'agent_finish',
    decision: {
      ...active.decision,
      action: 'finish',
      reason: '已完成',
    },
    revision: 2,
  })
  const submits = []
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'POST' && path === '/agent/adaptive/start') {
      return { body: active }
    }
    if (request.method() === 'GET' && path === '/agent/adaptive/adapt-rejected') {
      return { body: active }
    }
    if (request.method() === 'POST' && path === '/agent/adaptive/submit') {
      submits.push({
        body: await request.postDataJSON(),
        key: await request.headerValue('idempotency-key'),
      })
      if (submits.length === 1) {
        return { status: 422, body: { detail: '答案格式无效' } }
      }
      return { body: completed }
    }
    return null
  })

  await page.goto('/adaptive')
  await page.getByLabel('学习目标').fill('学习首字母')
  await page.getByLabel('文档 ID').fill('notes.md')
  await page.getByRole('button', { name: '开始自适应辅导' }).click()
  await page.getByRole('radio', { name: 'Alpha' }).check()
  await page.getByRole('button', { name: /提交本轮/ }).click()

  await expect(page.getByRole('radio', { name: 'Alpha' })).toBeEnabled()
  await expect.poll(() => page.evaluate(key => (
    JSON.parse(sessionStorage.getItem(key)).pending_submit
  ), ADAPTIVE_RECOVERY_KEY)).toBeNull()
  expect(
    await page.evaluate(key => JSON.parse(sessionStorage.getItem(key)), ADAPTIVE_RECOVERY_KEY),
  ).toMatchObject({
    session: { adaptive_session_id: active.adaptive_session_id },
    snapshot: { adaptive_session_id: active.adaptive_session_id, revision: 1 },
    pending_submit: null,
  })

  await page.getByRole('radio', { name: 'Beta' }).check()
  await page.getByRole('button', { name: /提交本轮/ }).click()
  await expect(page.getByText('修正后的答案已提交')).toBeVisible()

  expect(submits).toHaveLength(2)
  expect(submits[0].body.answers).toEqual(['Alpha'])
  expect(submits[1].body.answers).toEqual(['Beta'])
  expect(submits[1].key).not.toBe(submits[0].key)
  expect(unexpectedRequests).toEqual([])
})

test('adaptive oversized start clears the entire recovery and unlocks setup', async ({ page }) => {
  const starts = []
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'POST' && path === '/agent/adaptive/start') {
      starts.push({
        key: await request.headerValue('idempotency-key'),
        body: await request.postDataJSON(),
      })
      return {
        status: 413,
        body: {
          detail: '自适应会话超过持久化上限，请重新开始',
          code: 'adaptive_session_too_large',
        },
      }
    }
    return null
  })

  await page.goto('/adaptive')
  await page.getByLabel('学习目标').fill('触发超大 start')
  await page.getByLabel('文档 ID').fill('notes.md')
  await page.getByRole('button', { name: '开始自适应辅导' }).click()

  await expect(page.getByRole('alert')).toContainText('自适应会话超过持久化上限')
  await expect(page.getByLabel('学习目标')).toBeEditable()
  await expect(page.getByLabel('文档 ID')).toBeEditable()
  await expect(page.getByRole('button', { name: '开始自适应辅导' })).toBeVisible()
  await expect.poll(() => (
    page.evaluate(key => sessionStorage.getItem(key), ADAPTIVE_RECOVERY_KEY)
  )).toBeNull()
  expect(starts).toHaveLength(1)
  expect(starts[0].key).toMatch(UUID_V4_PATTERN)
  expect(unexpectedRequests).toEqual([])
})

test('adaptive oversized submit discards the old session before a new start', async ({ page }) => {
  const oldSession = adaptiveSnapshot({ adaptive_session_id: 'adapt-oversized-submit' })
  const newSession = adaptiveSnapshot({
    adaptive_session_id: 'adapt-after-oversized-submit',
    questions: [{
      index: 0,
      question: '终态清理后的新会话题目',
      options: ['新答案'],
      type: 'choice',
    }],
  })
  const starts = []
  const submits = []
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'POST' && path === '/agent/adaptive/start') {
      starts.push({
        key: await request.headerValue('idempotency-key'),
        body: await request.postDataJSON(),
      })
      return { body: starts.length === 1 ? oldSession : newSession }
    }
    if (
      request.method() === 'GET'
      && path === `/agent/adaptive/${oldSession.adaptive_session_id}`
    ) {
      return { body: oldSession }
    }
    if (request.method() === 'POST' && path === '/agent/adaptive/submit') {
      submits.push({
        key: await request.headerValue('idempotency-key'),
        body: await request.postDataJSON(),
      })
      return {
        status: 413,
        body: {
          detail: '自适应会话超过持久化上限，请重新开始',
          code: 'adaptive_session_too_large',
        },
      }
    }
    return null
  })

  await page.goto('/adaptive')
  await page.getByLabel('学习目标').fill('旧会话目标')
  await page.getByLabel('文档 ID').fill('notes.md')
  await page.getByRole('button', { name: '开始自适应辅导' }).click()
  await page.getByRole('radio', { name: 'Alpha' }).check()
  await page.getByRole('button', { name: /提交本轮/ }).click()

  await expect(page.getByRole('alert')).toContainText('自适应会话超过持久化上限')
  await expect(page.getByText('请选择首字母')).toHaveCount(0)
  await expect(page.getByLabel('学习目标')).toBeEditable()
  await expect(page.getByLabel('文档 ID')).toBeEditable()
  await expect(page.getByRole('button', { name: '开始自适应辅导' })).toBeVisible()
  await expect.poll(() => (
    page.evaluate(key => sessionStorage.getItem(key), ADAPTIVE_RECOVERY_KEY)
  )).toBeNull()
  expect(submits).toHaveLength(1)
  expect(submits[0]).toMatchObject({
    body: {
      adaptive_session_id: oldSession.adaptive_session_id,
      answers: ['Alpha'],
      turn: 1,
      revision: 1,
    },
  })

  await page.getByLabel('学习目标').fill('终态后的新目标')
  await page.getByRole('button', { name: '开始自适应辅导' }).click()
  await expect(page.getByText('终态清理后的新会话题目')).toBeVisible()
  expect(starts).toHaveLength(2)
  expect(starts[1].key).not.toBe(starts[0].key)
  expect(submits).toHaveLength(1)
  expect(unexpectedRequests).toEqual([])
})

test('adaptive expired recovery clears only the expired session and returns to setup', async ({ page }) => {
  const expired = adaptiveSnapshot({ adaptive_session_id: 'adapt-expired' })
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path === '/agent/adaptive/adapt-expired') {
      return {
        status: 410,
        body: {
          detail: '自适应学习会话已过期',
          code: 'adaptive_session_expired',
          reason: 'expired',
        },
      }
    }
    return null
  })
  await page.addInitScript(({ key, value }) => {
    sessionStorage.setItem(key, JSON.stringify(value))
  }, {
    key: ADAPTIVE_RECOVERY_KEY,
    value: adaptiveRecovery(expired),
  })

  await page.goto('/adaptive')
  await expect(page.getByRole('alert')).toContainText('自适应学习会话已过期')
  await expect(page.getByLabel('学习目标')).toBeEditable()
  await expect(page.getByRole('button', { name: '开始自适应辅导' })).toBeVisible()
  await expect(page.getByText('请选择首字母')).toHaveCount(0)
  await expect.poll(() => page.evaluate(key => sessionStorage.getItem(key), ADAPTIVE_RECOVERY_KEY))
    .toBeNull()
  expect(unexpectedRequests).toEqual([])
})

test('adaptive ignores a late start response after reset and a newer start', async ({ page }) => {
  let releaseOld
  let markOldStarted
  const oldStarted = new Promise(resolve => { markOldStarted = resolve })
  const oldBlocked = new Promise(resolve => { releaseOld = resolve })
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'POST' && path === '/agent/adaptive/start') {
      const body = await request.postDataJSON()
      if (body.goal === '旧目标') {
        markOldStarted()
        await oldBlocked
        return {
          body: adaptiveSnapshot({
            adaptive_session_id: 'adapt-old',
            questions: [{
              index: 0,
              question: '不应出现的旧题目',
              options: ['旧答案'],
              type: 'choice',
            }],
          }),
        }
      }
      return {
        body: adaptiveSnapshot({
          adaptive_session_id: 'adapt-new',
          questions: [{
            index: 0,
            question: '新的有效题目',
            options: ['新答案'],
            type: 'choice',
          }],
        }),
      }
    }
    return null
  })

  try {
    await page.goto('/adaptive')
    await page.getByLabel('学习目标').fill('旧目标')
    await page.getByLabel('文档 ID').fill('notes.md')
    await page.getByRole('button', { name: '开始自适应辅导' }).click()
    await oldStarted

    const oldStartAborted = waitForAbortedApiRequest(
      page,
      'POST',
      '/agent/adaptive/start',
    )
    await page.getByRole('button', { name: '重新开始' }).click()
    expectBrowserAbort(await oldStartAborted)
    releaseOld()

    await page.getByLabel('学习目标').fill('新目标')
    await page.getByLabel('文档 ID').fill('notes.md')
    await page.getByRole('button', { name: '开始自适应辅导' }).click()
    await expect(page.getByText('新的有效题目')).toBeVisible()
    await expect(page.getByText('不应出现的旧题目')).toHaveCount(0)
    const stored = await page.evaluate(
      key => JSON.parse(sessionStorage.getItem(key)),
      ADAPTIVE_RECOVERY_KEY,
    )
    expect(stored.session.adaptive_session_id).toBe('adapt-new')
    expect(unexpectedRequests).toEqual([])
  } finally {
    releaseOld()
  }
})

test('quiz restart releases the learning path binding instead of carrying it into the next round', async ({ page }) => {
  // clearSession() 不动 config，而 navigateToLearningPath / startIncomingQuiz
  // 都在它后面补了 setConfig——只有 handleRestart 漏了。漏掉时下一轮请求仍带
  // learning_path_source，文档与主题控件也因绑定存在保持 disabled，页面上没有
  // 任何说明，用户改不回来。
  const pathId = `lp_${'a'.repeat(32)}`
  const starts = []
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md', 'other.md'] } }
    }
    if (request.method() === 'POST' && path === '/session/start') {
      const body = await request.postDataJSON()
      starts.push(body)
      return {
        body: {
          session_id: `quiz-${starts.length}`,
          total: 1,
          questions: [{ index: 0, question: '首字母是什么？', options: ['Alpha', 'Beta'] }],
          // 绑定必须原样回显，否则客户端按「路径回显不匹配」终止本次启动。
          learning_path_source: body.learning_path_source ?? null,
          learning_path_completion: null,
          expires_at: FUTURE_EXPIRES_AT,
        },
      }
    }
    if (request.method() === 'POST' && /^\/session\/quiz-\d+\/answer$/.test(path)) {
      return {
        body: {
          correct: true,
          correct_answer: 'Alpha',
          explanation: '回答正确',
          is_last: true,
          next_index: null,
        },
      }
    }
    if (request.method() === 'GET' && /^\/session\/quiz-\d+\/result$/.test(path)) {
      return {
        body: {
          session_id: path.split('/')[2],
          document_id: 'notes.md',
          total: 1,
          correct: 1,
          score: 1,
          details: [],
        },
      }
    }
    return null
  })

  await page.goto(
    '/quiz?document_id=notes.md&topic=阶段主题'
    + `&launch_id=restart-launch-1234&path_id=${pathId}&stage_id=1`,
  )
  await page.getByRole('button', { name: '开始答题' }).click()
  await page.getByRole('button', { name: /Alpha/ }).click()
  await page.getByRole('button', { name: '提交答案' }).click()
  await page.getByRole('button', { name: '查看结果' }).click()
  await page.getByRole('button', { name: '再来一轮' }).click()

  await expect(page.getByRole('combobox', { name: '学习文档' })).toBeEnabled()
  await page.getByRole('combobox', { name: '学习文档' }).selectOption('other.md')
  await page.getByRole('button', { name: '开始答题' }).click()
  await expect(page.getByRole('button', { name: /Alpha/ })).toBeVisible()

  expect(starts).toHaveLength(2)
  expect(starts[0]).toMatchObject({
    document_id: 'notes.md',
    learning_path_source: { learning_path_id: pathId, stage_id: 1 },
  })
  expect(starts[1].document_id).toBe('other.md')
  expect(starts[1].learning_path_source ?? null).toBeNull()
  expect(unexpectedRequests).toEqual([])
})

test('quiz restart clears errors and prior grading and learning reports', async ({ page }) => {
  let starts = 0
  const answerRequests = []
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md'] } }
    }
    if (request.method() === 'POST' && path === '/session/start') {
      starts += 1
      return {
        body: {
          session_id: `quiz-${starts}`,
          total: 1,
          questions: [{ index: 0, question: '首字母是什么？', options: ['Alpha', 'Beta'] }],
        },
      }
    }
    if (request.method() === 'POST' && /^\/session\/quiz-\d+\/answer$/.test(path)) {
      answerRequests.push({
        path,
        key: await request.headerValue('idempotency-key'),
        body: await request.postDataJSON(),
      })
      return answerRequests.length === 1
        ? { status: 422, body: { detail: '答案格式无效' } }
        : {
            body: {
              correct: true,
              correct_answer: 'Alpha',
              explanation: '回答正确',
              is_last: true,
              next_index: null,
            },
          }
    }
    if (request.method() === 'GET' && /^\/session\/quiz-\d+\/result$/.test(path)) {
      return {
        body: {
          session_id: path.includes('quiz-1') ? 'quiz-1' : 'quiz-2',
          document_id: 'notes.md',
          total: 1,
          correct: 1,
          score: 1,
          details: [],
        },
      }
    }
    if (request.method() === 'POST' && /^\/session\/quiz-\d+\/grade$/.test(path)) {
      const sessionId = path.includes('quiz-1') ? 'quiz-1' : 'quiz-2'
      return {
        body: {
          session_id: sessionId,
          total: 1,
          correct: 1,
          score: 1,
          grades: [{
            index: 0,
            question: '首字母是什么？',
            user_answer: 'Alpha',
            correct_answer: 'Alpha',
            is_correct: true,
            ai_feedback: null,
            knowledge_gap: null,
          }],
        },
      }
    }
    if (request.method() === 'POST' && path === '/session/quiz-2/report') {
      return {
        body: {
          session_id: 'quiz-2',
          document_id: 'notes.md',
          overall_score: 1,
          topic_mastery: [],
          strengths: ['首字母'],
          weaknesses: [],
          recommendations: ['继续练习'],
          summary: '第二轮学习报告摘要',
        },
      }
    }
    return null
  })

  await page.goto('/quiz')
  await page.getByRole('combobox', { name: '学习文档' }).selectOption('notes.md')

  await page.getByRole('button', { name: '开始答题' }).click()
  await page.getByRole('button', { name: /Alpha/ }).click()
  await page.getByRole('button', { name: '提交答案' }).click()
  await expect(page.getByRole('alert')).toContainText('答案格式无效')
  await expect(page.getByRole('button', { name: /Beta/ })).toBeEnabled()
  await expect.poll(() => page.evaluate(key => (
    JSON.parse(sessionStorage.getItem(key)).pending_answer
  ), QUIZ_RECOVERY_KEY)).toBeNull()
  await page.getByRole('button', { name: /Beta/ }).click()
  await page.getByRole('button', { name: '提交答案' }).click()
  await page.getByRole('button', { name: '查看结果' }).click()
  await page.getByRole('button', { name: 'AI 批改讲解' }).click()
  await expect(page.getByRole('heading', { name: 'AI 批改报告' })).toBeVisible()
  await page.getByRole('button', { name: '再来一轮' }).click()

  await expect(page.getByRole('alert')).toHaveCount(0)
  await expect(page.getByRole('heading', { name: 'AI 批改报告' })).toHaveCount(0)

  await page.getByRole('button', { name: '开始答题' }).click()
  await page.getByRole('button', { name: /Alpha/ }).click()
  await page.getByRole('button', { name: '提交答案' }).click()
  await page.getByRole('button', { name: '查看结果' }).click()
  await page.getByRole('button', { name: 'AI 批改讲解' }).click()
  await page.getByRole('button', { name: '学习评估报告' }).click()
  await expect(page.getByText('第二轮学习报告摘要')).toBeVisible()
  await page.getByRole('button', { name: '再来一轮' }).click()

  await expect(page.getByText('第二轮学习报告摘要')).toHaveCount(0)
  await expect(page.getByRole('alert')).toHaveCount(0)
  expect(starts).toBe(2)
  expect(answerRequests.slice(0, 2).map(request => request.body)).toEqual([
    { answer: 'Alpha', question_index: 0 },
    { answer: 'Beta', question_index: 0 },
  ])
  expect(answerRequests[1].key).not.toBe(answerRequests[0].key)
  expect(unexpectedRequests).toEqual([])
})

test('quiz deterministic start rejections discard their receipts and rotate keys', async ({ page }) => {
  const starts = []
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md'] } }
    }
    if (request.method() === 'POST' && path === '/session/start') {
      starts.push({
        key: await request.headerValue('idempotency-key'),
        body: await request.postDataJSON(),
      })
      if (starts.length === 1) {
        return { status: 422, body: { detail: '出题参数无效' } }
      }
      if (starts.length === 2) {
        return {
          status: 413,
          body: {
            detail: '答题会话超过持久化上限',
            code: 'quiz_session_too_large',
          },
        }
      }
      return {
        body: {
          session_id: 'quiz-after-start-rejections',
          total: 1,
          questions: [{
            index: 0,
            question: '确定拒绝后使用了新请求吗？',
            options: ['是', '否'],
            type: 'choice',
          }],
          revision: 1,
          expires_at: FUTURE_EXPIRES_AT,
        },
      }
    }
    return null
  })

  await page.goto('/quiz')
  await page.getByRole('combobox', { name: '学习文档' }).selectOption('notes.md')
  const topic = page.getByRole('textbox', { name: /出题主题/ })
  await topic.fill('第一次确定拒绝')
  await page.getByRole('button', { name: '开始答题' }).click()
  await expect(page.getByRole('alert')).toContainText('出题参数无效')
  await expect(topic).toBeEditable()
  await expect.poll(() => (
    page.evaluate(key => sessionStorage.getItem(key), QUIZ_RECOVERY_KEY)
  )).toBeNull()

  await topic.fill('第二次确定拒绝')
  await page.getByRole('button', { name: '开始答题' }).click()
  await expect(page.getByRole('alert')).toContainText('答题会话超过持久化上限')
  await expect(topic).toBeEditable()
  await expect.poll(() => (
    page.evaluate(key => sessionStorage.getItem(key), QUIZ_RECOVERY_KEY)
  )).toBeNull()

  await topic.fill('最终有效请求')
  await page.getByRole('button', { name: '开始答题' }).click()
  await expect(page.getByText('确定拒绝后使用了新请求吗？')).toBeVisible()
  expect(starts.map(start => start.body.description)).toEqual([
    '第一次确定拒绝',
    '第二次确定拒绝',
    '最终有效请求',
  ])
  expect(new Set(starts.map(start => start.key))).toHaveProperty('size', 3)
  for (const start of starts) expect(start.key).toMatch(UUID_V4_PATTERN)
  expect(unexpectedRequests).toEqual([])
})

test('quiz coded oversized answer clears pending before a corrected request', async ({ page }) => {
  const answers = []
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md'] } }
    }
    if (request.method() === 'POST' && path === '/session/start') {
      return {
        body: {
          session_id: 'quiz-oversized-answer',
          total: 1,
          questions: [{
            index: 0,
            question: '请选择可修正的答案',
            options: ['Alpha', 'Beta'],
            type: 'choice',
          }],
          revision: 1,
          expires_at: FUTURE_EXPIRES_AT,
        },
      }
    }
    if (request.method() === 'POST' && path === '/session/quiz-oversized-answer/answer') {
      answers.push({
        key: await request.headerValue('idempotency-key'),
        body: await request.postDataJSON(),
      })
      if (answers.length === 1) {
        return {
          status: 413,
          body: {
            detail: '答案导致会话超过持久化上限',
            code: 'quiz_session_too_large',
          },
        }
      }
      return {
        body: {
          evaluation_status: 'final',
          correct: true,
          correct_answer: 'Beta',
          explanation: '已使用修正后的新请求。',
          is_last: true,
          next_index: null,
          revision: 2,
          expires_at: FUTURE_EXPIRES_AT,
        },
      }
    }
    return null
  })

  await page.goto('/quiz')
  await page.getByRole('combobox', { name: '学习文档' }).selectOption('notes.md')
  await page.getByRole('button', { name: '开始答题' }).click()
  await page.getByRole('button', { name: 'Alpha' }).click()
  await page.getByRole('button', { name: '提交答案' }).click()

  await expect(page.getByRole('alert')).toContainText('答案导致会话超过持久化上限')
  await expect(page.getByRole('button', { name: 'Beta' })).toBeEnabled()
  await expect.poll(() => page.evaluate(key => (
    JSON.parse(sessionStorage.getItem(key)).pending_answer
  ), QUIZ_RECOVERY_KEY)).toBeNull()
  await page.getByRole('button', { name: 'Beta' }).click()
  await page.getByRole('button', { name: '提交答案' }).click()
  await expect(page.getByText('已使用修正后的新请求。')).toBeVisible()

  expect(answers.map(answer => answer.body)).toEqual([
    { answer: 'Alpha', question_index: 0 },
    { answer: 'Beta', question_index: 0 },
  ])
  expect(answers[1].key).not.toBe(answers[0].key)
  expect(unexpectedRequests).toEqual([])
})

test('quiz start deadline preserves the exact recovery request for explicit retry', async ({ page }) => {
  await page.clock.install()
  const attempts = []
  let releaseFirst
  let markFirstStarted
  const firstGate = new Promise(resolve => { releaseFirst = resolve })
  const firstStarted = new Promise(resolve => { markFirstStarted = resolve })
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md'] } }
    }
    if (request.method() === 'POST' && path === '/session/start') {
      attempts.push({
        key: await request.headerValue('idempotency-key'),
        body: await request.postDataJSON(),
        rawBody: request.postData(),
      })
      if (attempts.length === 1) {
        markFirstStarted()
        await firstGate
        return {
          body: {
            session_id: 'quiz-start-too-late',
            total: 1,
            questions: [{
              index: 0,
              question: '不应采用的迟到 Quiz 题目',
              options: ['旧答案'],
              type: 'choice',
            }],
            revision: 1,
            expires_at: FUTURE_EXPIRES_AT,
          },
        }
      }
      return {
        body: {
          session_id: 'quiz-start-after-timeout',
          total: 1,
          questions: [{
            index: 0,
            question: 'Quiz 超时重试后的题目',
            options: ['新答案'],
            type: 'choice',
          }],
          revision: 1,
          expires_at: FUTURE_EXPIRES_AT,
        },
      }
    }
    return null
  })

  try {
    await page.goto('/quiz')
    await page.getByRole('combobox', { name: '学习文档' }).selectOption('notes.md')
    const firstAborted = waitForAbortedApiRequest(page, 'POST', '/session/start')
    await page.getByRole('button', { name: '开始答题' }).click()
    await firstStarted
    await page.clock.fastForward(RECOVERY_DEADLINE_ADVANCE_MS)
    await expect(page.getByRole('alert')).toContainText(RECOVERY_TIMEOUT_MESSAGE)
    expectBrowserAbort(await firstAborted)
    const retry = page.getByRole('button', { name: '重试恢复' })
    await expect(retry).toBeEnabled()
    const pending = await page.evaluate(
      key => JSON.parse(sessionStorage.getItem(key)),
      QUIZ_RECOVERY_KEY,
    )
    expect(pending).toMatchObject({
      intent: { kind: 'standard', request: { document_id: 'notes.md' } },
      session: null,
      pending_answer: null,
    })
    expect(pending.start_idempotency_key).toBe(attempts[0].key)

    releaseFirst()
    await retry.click()
    await expect(page.getByText('Quiz 超时重试后的题目')).toBeVisible()
    await expect(page.getByText('不应采用的迟到 Quiz 题目')).toHaveCount(0)
    expect(attempts).toHaveLength(2)
    expect(attempts[0]).toEqual(attempts[1])
    expect(attempts[0].key).toMatch(UUID_V4_PATTERN)
    expect(unexpectedRequests).toEqual([])
  } finally {
    releaseFirst()
  }
})

test('quiz snapshot deadline leaves the session receipt unchanged until retry', async ({ page }) => {
  await page.clock.install()
  const sessionId = 'quiz-get-timeout'
  const recovery = quizRecovery({ sessionId })
  const late = quizSnapshot({ sessionId })
  late.questions[0].question = '不应采用的迟到 Quiz GET'
  const canonical = quizSnapshot({ sessionId })
  canonical.questions[0].question = 'Quiz GET 重试成功'
  await page.addInitScript(({ key, value }) => {
    sessionStorage.setItem(key, JSON.stringify(value))
  }, { key: QUIZ_RECOVERY_KEY, value: recovery })
  let reads = 0
  let releaseFirst
  let markFirstStarted
  const firstGate = new Promise(resolve => { releaseFirst = resolve })
  const firstStarted = new Promise(resolve => { markFirstStarted = resolve })
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md'] } }
    }
    if (request.method() === 'GET' && path === `/session/${sessionId}`) {
      reads += 1
      if (reads === 1) {
        markFirstStarted()
        await firstGate
        return { body: late }
      }
      return { body: canonical }
    }
    return null
  })

  try {
    const firstAborted = waitForAbortedApiRequest(page, 'GET', `/session/${sessionId}`)
    await page.goto('/quiz')
    await firstStarted
    await page.clock.fastForward(RECOVERY_DEADLINE_ADVANCE_MS)
    await expect(page.getByRole('alert')).toContainText(RECOVERY_TIMEOUT_MESSAGE)
    expectBrowserAbort(await firstAborted)
    const retry = page.getByRole('button', { name: '重试恢复' })
    await expect(retry).toBeEnabled()
    expect(
      await page.evaluate(
        key => JSON.parse(sessionStorage.getItem(key)),
        QUIZ_RECOVERY_KEY,
      ),
    ).toEqual(recovery)

    releaseFirst()
    await retry.click()
    await expect(page.getByText('Quiz GET 重试成功')).toBeVisible()
    await expect(page.getByText('不应采用的迟到 Quiz GET')).toHaveCount(0)
    expect(reads).toBe(2)
    expect(unexpectedRequests).toEqual([])
  } finally {
    releaseFirst()
  }
})

test('quiz answer deadline reconciles before replaying the frozen answer', async ({ page }) => {
  await page.clock.install()
  const sessionId = 'quiz-answer-timeout'
  const activeSnapshot = quizSnapshot({ sessionId })
  const recoveredFeedback = {
    evaluation_status: 'final',
    correct: true,
    correct_answer: 'Alpha',
    explanation: '已用原答案和请求键安全恢复',
    is_last: true,
    next_index: null,
    revision: 2,
    expires_at: FUTURE_EXPIRES_AT,
  }
  const completedSnapshot = quizSnapshot({
    sessionId,
    status: 'completed',
    lastAnswerResult: recoveredFeedback,
  })
  const answers = []
  let snapshotReads = 0
  let releaseAnswer
  let markAnswerStarted
  let releaseFailedRead
  let markFailedReadStarted
  const answerGate = new Promise(resolve => { releaseAnswer = resolve })
  const answerStarted = new Promise(resolve => { markAnswerStarted = resolve })
  const failedReadGate = new Promise(resolve => { releaseFailedRead = resolve })
  const failedReadStarted = new Promise(resolve => { markFailedReadStarted = resolve })
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md'] } }
    }
    if (request.method() === 'POST' && path === '/session/start') {
      return {
        body: {
          session_id: sessionId,
          total: 1,
          questions: activeSnapshot.questions,
          revision: 1,
          expires_at: FUTURE_EXPIRES_AT,
        },
      }
    }
    if (request.method() === 'POST' && path === `/session/${sessionId}/answer`) {
      answers.push({
        key: await request.headerValue('idempotency-key'),
        body: await request.postDataJSON(),
        rawBody: request.postData(),
      })
      if (answers.length === 1) {
        markAnswerStarted()
        await answerGate
      }
      return { body: recoveredFeedback }
    }
    if (request.method() === 'GET' && path === `/session/${sessionId}`) {
      snapshotReads += 1
      if (snapshotReads === 1) {
        markFailedReadStarted()
        await failedReadGate
        return { status: 503, body: { detail: '答案对账暂时不可用' } }
      }
      return { body: snapshotReads === 2 ? activeSnapshot : completedSnapshot }
    }
    return null
  })

  try {
    await page.goto('/quiz')
    await page.getByRole('combobox', { name: '学习文档' }).selectOption('notes.md')
    await page.getByRole('button', { name: '开始答题' }).click()
    await page.getByRole('button', { name: 'Alpha' }).click()
    const answerAborted = waitForAbortedApiRequest(
      page,
      'POST',
      `/session/${sessionId}/answer`,
    )
    await page.getByRole('button', { name: '提交答案' }).click()
    await answerStarted
    await page.clock.fastForward(RECOVERY_DEADLINE_ADVANCE_MS)
    await expect(page.getByRole('alert')).toContainText(RECOVERY_TIMEOUT_MESSAGE)
    expectBrowserAbort(await answerAborted)
    releaseAnswer()
    await failedReadStarted
    const pending = await page.evaluate(key => (
      JSON.parse(sessionStorage.getItem(key)).pending_answer
    ), QUIZ_RECOVERY_KEY)
    expect(pending).toMatchObject({ question_index: 0, answer: 'Alpha' })
    expect(pending.idempotency_key).toBe(answers[0].key)

    releaseFailedRead()
    await expect(page.getByRole('alert')).toContainText('答案对账暂时不可用')
    const retry = page.getByRole('button', { name: '重试恢复' })
    await expect(retry).toBeEnabled()
    await retry.click()
    await expect(page.getByText('已用原答案和请求键安全恢复')).toBeVisible()
    expect(answers).toHaveLength(2)
    expect(answers[0]).toEqual(answers[1])
    expect(answers[0].body).toEqual({ answer: 'Alpha', question_index: 0 })
    expect(answers[0].key).toMatch(UUID_V4_PATTERN)
    expect(snapshotReads).toBe(3)
    expect(
      await page.evaluate(
        key => JSON.parse(sessionStorage.getItem(key)).pending_answer,
        QUIZ_RECOVERY_KEY,
      ),
    ).toBeNull()
    expect(unexpectedRequests).toEqual([])
  } finally {
    releaseAnswer()
    releaseFailedRead()
  }
})

test('quiz result deadline exits advancing and hydrates the canonical snapshot', async ({ page }) => {
  await page.clock.install()
  const sessionId = 'quiz-result-timeout'
  const feedback = {
    evaluation_status: 'final',
    correct: true,
    correct_answer: 'Alpha',
    explanation: '答案已提交，等待结果。',
    is_last: true,
    next_index: null,
    revision: 2,
    expires_at: FUTURE_EXPIRES_AT,
  }
  const canonicalSnapshot = quizSnapshot({
    sessionId,
    status: 'completed',
    lastAnswerResult: feedback,
  })
  const lateResult = {
    ...canonicalSnapshot.result,
    correct: 0,
    incorrect: 1,
    score: 0,
  }
  let results = 0
  let snapshotReads = 0
  let releaseResult
  let markResultStarted
  let releaseReconcile
  let markReconcileStarted
  const resultGate = new Promise(resolve => { releaseResult = resolve })
  const resultStarted = new Promise(resolve => { markResultStarted = resolve })
  const reconcileGate = new Promise(resolve => { releaseReconcile = resolve })
  const reconcileStarted = new Promise(resolve => { markReconcileStarted = resolve })
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md'] } }
    }
    if (request.method() === 'POST' && path === '/session/start') {
      return {
        body: {
          session_id: sessionId,
          total: 1,
          questions: canonicalSnapshot.questions,
          revision: 1,
          expires_at: FUTURE_EXPIRES_AT,
        },
      }
    }
    if (request.method() === 'POST' && path === `/session/${sessionId}/answer`) {
      return { body: feedback }
    }
    if (request.method() === 'GET' && path === `/session/${sessionId}/result`) {
      results += 1
      markResultStarted()
      await resultGate
      return { body: lateResult }
    }
    if (request.method() === 'GET' && path === `/session/${sessionId}`) {
      snapshotReads += 1
      markReconcileStarted()
      await reconcileGate
      return { body: canonicalSnapshot }
    }
    return null
  })

  try {
    await page.goto('/quiz')
    await page.getByRole('combobox', { name: '学习文档' }).selectOption('notes.md')
    await page.getByRole('button', { name: '开始答题' }).click()
    await page.getByRole('button', { name: 'Alpha' }).click()
    await page.getByRole('button', { name: '提交答案' }).click()
    const resultAborted = waitForAbortedApiRequest(
      page,
      'GET',
      `/session/${sessionId}/result`,
    )
    await page.getByRole('button', { name: '查看结果' }).click()
    await resultStarted
    await page.clock.fastForward(RECOVERY_DEADLINE_ADVANCE_MS)
    await expect(page.getByRole('alert')).toContainText(RECOVERY_TIMEOUT_MESSAGE)
    expectBrowserAbort(await resultAborted)
    releaseResult()
    await reconcileStarted
    await expect(page.getByText('正在恢复答题进度...')).toBeVisible()
    expect(
      await page.evaluate(
        key => JSON.parse(sessionStorage.getItem(key)),
        QUIZ_RECOVERY_KEY,
      ),
    ).toMatchObject({
      session: { session_id: sessionId, revision: 2 },
      acknowledged_answer_count: 1,
      pending_answer: null,
    })

    releaseReconcile()
    await expect(page.locator('.quiz-results')).toBeVisible()
    await expect(page.locator('.score-number')).toHaveText('100')
    expect(results).toBe(1)
    expect(snapshotReads).toBe(1)
    expect(unexpectedRequests).toEqual([])
  } finally {
    releaseResult()
    releaseReconcile()
  }
})

test('quiz terminal result response clears recovery without snapshot retries', async ({ page }) => {
  const sessionId = 'quiz-result-expired'
  let resultReads = 0
  let snapshotReads = 0
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md'] } }
    }
    if (request.method() === 'POST' && path === '/session/start') {
      return {
        body: {
          session_id: sessionId,
          total: 1,
          questions: [{
            index: 0,
            question: '终态结果不应继续恢复吗？',
            options: ['是', '否'],
            type: 'choice',
          }],
          revision: 1,
          expires_at: FUTURE_EXPIRES_AT,
        },
      }
    }
    if (request.method() === 'POST' && path === `/session/${sessionId}/answer`) {
      return {
        body: {
          evaluation_status: 'final',
          correct: true,
          correct_answer: '是',
          explanation: '答案已记录。',
          is_last: true,
          next_index: null,
          revision: 2,
          expires_at: FUTURE_EXPIRES_AT,
        },
      }
    }
    if (request.method() === 'GET' && path === `/session/${sessionId}/result`) {
      resultReads += 1
      return {
        status: 410,
        body: {
          detail: '答题会话已过期',
          code: 'quiz_session_expired',
        },
      }
    }
    if (request.method() === 'GET' && path === `/session/${sessionId}`) {
      snapshotReads += 1
      return { status: 500, body: { detail: '终态后不应读取快照' } }
    }
    return null
  })

  await page.goto('/quiz')
  await page.getByRole('combobox', { name: '学习文档' }).selectOption('notes.md')
  await page.getByRole('button', { name: '开始答题' }).click()
  await page.getByRole('button', { name: '是' }).click()
  await page.getByRole('button', { name: '提交答案' }).click()
  await page.getByRole('button', { name: '查看结果' }).click()

  await expect(page.getByRole('alert')).toContainText('答题会话已过期')
  await expect(page.getByRole('button', { name: '开始答题' })).toBeVisible()
  await expect.poll(() => (
    page.evaluate(key => sessionStorage.getItem(key), QUIZ_RECOVERY_KEY)
  )).toBeNull()
  expect(resultReads).toBe(1)
  expect(snapshotReads).toBe(0)
  expect(unexpectedRequests).toEqual([])
})

test('quiz grading deadline hydrates the canonical report instead of reposting', async ({ page }) => {
  await page.clock.install()
  const sessionId = 'quiz-grade-timeout'
  const recovery = quizRecovery({ sessionId, acknowledgedAnswerCount: 1 })
  const canonicalGrade = {
    session_id: sessionId,
    total: 1,
    correct: 1,
    score: 1,
    grades: [{
      index: 0,
      question: '请选择首字母',
      user_answer: 'Alpha',
      correct_answer: 'Alpha',
      is_correct: true,
      ai_feedback: '权威快照中的批改',
      knowledge_gap: null,
    }],
    revision: 2,
    expires_at: FUTURE_EXPIRES_AT,
    learning_path_source: null,
    learning_path_completion: null,
  }
  const lateGrade = {
    ...canonicalGrade,
    grades: [{ ...canonicalGrade.grades[0], ai_feedback: '不应采用的迟到批改' }],
  }
  const baseSnapshot = quizSnapshot({ sessionId, status: 'completed' })
  const canonicalSnapshot = quizSnapshot({
    sessionId,
    status: 'completed',
    gradingReport: canonicalGrade,
  })
  await page.addInitScript(({ key, value }) => {
    sessionStorage.setItem(key, JSON.stringify(value))
  }, { key: QUIZ_RECOVERY_KEY, value: recovery })
  let reads = 0
  let grades = 0
  let releaseGrade
  let markGradeStarted
  let releaseReconcile
  let markReconcileStarted
  const gradeGate = new Promise(resolve => { releaseGrade = resolve })
  const gradeStarted = new Promise(resolve => { markGradeStarted = resolve })
  const reconcileGate = new Promise(resolve => { releaseReconcile = resolve })
  const reconcileStarted = new Promise(resolve => { markReconcileStarted = resolve })
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md'] } }
    }
    if (request.method() === 'GET' && path === `/session/${sessionId}`) {
      reads += 1
      if (reads === 1) return { body: baseSnapshot }
      markReconcileStarted()
      await reconcileGate
      return { body: canonicalSnapshot }
    }
    if (request.method() === 'POST' && path === `/session/${sessionId}/grade`) {
      grades += 1
      markGradeStarted()
      await gradeGate
      return { body: lateGrade }
    }
    return null
  })

  try {
    await page.goto('/quiz')
    const gradeAborted = waitForAbortedApiRequest(
      page,
      'POST',
      `/session/${sessionId}/grade`,
    )
    await page.getByRole('button', { name: 'AI 批改讲解' }).click()
    await gradeStarted
    await page.clock.fastForward(RECOVERY_DEADLINE_ADVANCE_MS)
    await expect(page.getByRole('alert')).toContainText(RECOVERY_TIMEOUT_MESSAGE)
    expectBrowserAbort(await gradeAborted)
    releaseGrade()
    await reconcileStarted

    releaseReconcile()
    await expect(page.getByRole('heading', { name: 'AI 批改报告' })).toBeVisible()
    await expect(page.getByText('权威快照中的批改')).toBeVisible()
    await expect(page.getByText('不应采用的迟到批改')).toHaveCount(0)
    expect(grades).toBe(1)
    expect(reads).toBe(2)
    expect(unexpectedRequests).toEqual([])
  } finally {
    releaseGrade()
    releaseReconcile()
  }
})

test('quiz report deadline hydrates the canonical report instead of reposting', async ({ page }) => {
  await page.clock.install()
  const sessionId = 'quiz-report-timeout'
  const recovery = quizRecovery({ sessionId, acknowledgedAnswerCount: 1 })
  const grade = {
    session_id: sessionId,
    total: 1,
    correct: 1,
    score: 1,
    grades: [],
    revision: 2,
    expires_at: FUTURE_EXPIRES_AT,
    learning_path_source: null,
    learning_path_completion: null,
  }
  const canonicalReport = {
    session_id: sessionId,
    document_id: 'notes.md',
    overall_score: 1,
    topic_mastery: [],
    strengths: ['权威快照中的学习报告'],
    weaknesses: [],
    recommendations: ['继续保持'],
    summary: '服务端快照已确认报告。',
    revision: 2,
    expires_at: FUTURE_EXPIRES_AT,
    learning_path_source: null,
    learning_path_completion: null,
  }
  const lateReport = {
    ...canonicalReport,
    strengths: ['不应采用的迟到学习报告'],
    summary: '迟到报告不应出现。',
  }
  const baseSnapshot = quizSnapshot({
    sessionId,
    status: 'completed',
    gradingReport: grade,
  })
  const canonicalSnapshot = quizSnapshot({
    sessionId,
    status: 'completed',
    gradingReport: grade,
    learningReport: canonicalReport,
  })
  await page.addInitScript(({ key, value }) => {
    sessionStorage.setItem(key, JSON.stringify(value))
  }, { key: QUIZ_RECOVERY_KEY, value: recovery })
  let reads = 0
  let reports = 0
  let releaseReport
  let markReportStarted
  let releaseReconcile
  let markReconcileStarted
  const reportGate = new Promise(resolve => { releaseReport = resolve })
  const reportStarted = new Promise(resolve => { markReportStarted = resolve })
  const reconcileGate = new Promise(resolve => { releaseReconcile = resolve })
  const reconcileStarted = new Promise(resolve => { markReconcileStarted = resolve })
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md'] } }
    }
    if (request.method() === 'GET' && path === `/session/${sessionId}`) {
      reads += 1
      if (reads === 1) return { body: baseSnapshot }
      markReconcileStarted()
      await reconcileGate
      return { body: canonicalSnapshot }
    }
    if (request.method() === 'POST' && path === `/session/${sessionId}/report`) {
      reports += 1
      markReportStarted()
      await reportGate
      return { body: lateReport }
    }
    return null
  })

  try {
    await page.goto('/quiz')
    const reportAborted = waitForAbortedApiRequest(
      page,
      'POST',
      `/session/${sessionId}/report`,
    )
    await page.getByRole('button', { name: '学习评估报告' }).click()
    await reportStarted
    await page.clock.fastForward(RECOVERY_DEADLINE_ADVANCE_MS)
    await expect(page.getByRole('alert')).toContainText(RECOVERY_TIMEOUT_MESSAGE)
    expectBrowserAbort(await reportAborted)
    releaseReport()
    await reconcileStarted

    releaseReconcile()
    await expect(page.getByText('服务端快照已确认报告。')).toBeVisible()
    await expect(page.getByText('权威快照中的学习报告')).toBeVisible()
    await expect(page.getByText('迟到报告不应出现。')).toHaveCount(0)
    expect(reports).toBe(1)
    expect(reads).toBe(2)
    expect(unexpectedRequests).toEqual([])
  } finally {
    releaseReport()
    releaseReconcile()
  }
})

test('quiz stale grade and coded oversized report reconcile by GET without reposting', async ({ page }) => {
  const sessionId = 'quiz-report-oversized'
  const recovery = quizRecovery({ sessionId, acknowledgedAnswerCount: 1 })
  const grade = {
    session_id: sessionId,
    total: 1,
    correct: 1,
    score: 1,
    grades: [],
    revision: 2,
    expires_at: FUTURE_EXPIRES_AT,
    learning_path_source: null,
    learning_path_completion: null,
  }
  const canonicalReport = {
    session_id: sessionId,
    document_id: 'notes.md',
    overall_score: 1,
    topic_mastery: [],
    strengths: ['GET 已确认唯一学习报告'],
    weaknesses: [],
    recommendations: ['继续保持'],
    summary: '服务端快照确认报告已经存在。',
    revision: 2,
    expires_at: FUTURE_EXPIRES_AT,
    learning_path_source: null,
    learning_path_completion: null,
  }
  const baseSnapshot = quizSnapshot({ sessionId, status: 'completed' })
  const gradedSnapshot = quizSnapshot({
    sessionId,
    status: 'completed',
    gradingReport: grade,
  })
  const canonicalSnapshot = quizSnapshot({
    sessionId,
    status: 'completed',
    gradingReport: grade,
    learningReport: canonicalReport,
  })
  await page.addInitScript(({ key, value }) => {
    sessionStorage.setItem(key, JSON.stringify(value))
  }, { key: QUIZ_RECOVERY_KEY, value: recovery })
  let reads = 0
  let grades = 0
  let reports = 0
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path === `/session/${sessionId}`) {
      reads += 1
      return {
        body: reads === 1
          ? baseSnapshot
          : reads === 2 ? gradedSnapshot : canonicalSnapshot,
      }
    }
    if (request.method() === 'POST' && path === `/session/${sessionId}/grade`) {
      grades += 1
      return {
        status: 409,
        body: {
          detail: '批改时服务端进度已变化',
          code: 'quiz_session_stale',
          reason: 'stale',
        },
      }
    }
    if (request.method() === 'POST' && path === `/session/${sessionId}/report`) {
      reports += 1
      return {
        status: 413,
        body: {
          detail: '报告响应超过持久化上限，结果可能已经提交',
          code: 'quiz_session_too_large',
        },
      }
    }
    return null
  })

  await page.goto('/quiz')
  await page.getByRole('button', { name: 'AI 批改讲解' }).click()
  await expect(page.getByRole('heading', { name: 'AI 批改报告' })).toBeVisible()
  expect(grades).toBe(1)
  expect(reads).toBe(2)

  await page.getByRole('button', { name: '学习评估报告' }).click()
  await expect(page.getByText('服务端快照确认报告已经存在。')).toBeVisible()
  await expect(page.getByText('GET 已确认唯一学习报告')).toBeVisible()
  expect(grades).toBe(1)
  expect(reports).toBe(1)
  expect(reads).toBe(3)
  expect(unexpectedRequests).toEqual([])
})

test('quiz reload reuses the pending start key after the first response is lost', async ({ page }) => {
  const startAttempts = []
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md'] } }
    }
    if (request.method() === 'POST' && path === '/session/start') {
      startAttempts.push({
        body: await request.postDataJSON(),
        key: await request.headerValue('idempotency-key'),
      })
      if (startAttempts.length === 1) {
        return { status: 503, body: { detail: '出题响应暂时不可用' } }
      }
      return {
        body: {
          session_id: 'quiz-start-recovered',
          total: 1,
          questions: [{ index: 0, question: '恢复后的题目', options: ['A', 'B'], type: 'choice' }],
          revision: 1,
          expires_at: FUTURE_EXPIRES_AT,
        },
      }
    }
    return null
  })

  await page.goto('/quiz')
  await page.getByRole('combobox', { name: '学习文档' }).selectOption('notes.md')
  await page.getByRole('button', { name: '开始答题' }).click()
  await expect(page.getByRole('alert')).toContainText('出题响应暂时不可用')

  const pending = await page.evaluate(key => JSON.parse(sessionStorage.getItem(key)), QUIZ_RECOVERY_KEY)
  expect(pending).toMatchObject({
    intent: { kind: 'standard', request: { document_id: 'notes.md' } },
    session: null,
  })
  expect(pending.start_idempotency_key).toMatch(UUID_V4_PATTERN)

  await page.reload()
  await expect(page.getByText('恢复后的题目')).toBeVisible()
  expect(startAttempts).toHaveLength(2)
  expect(startAttempts[1].body).toEqual(startAttempts[0].body)
  expect(startAttempts[1].key).toBe(startAttempts[0].key)
  expect(unexpectedRequests).toEqual([])
})

test('quiz reload reconciles a pending answer and replays the exact request key', async ({ page }) => {
  const answerAttempts = []
  let snapshotReads = 0
  let releaseFirstRead
  let markFirstReadStarted
  const firstReadGate = new Promise(resolve => { releaseFirstRead = resolve })
  const firstReadStarted = new Promise(resolve => { markFirstReadStarted = resolve })
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md'] } }
    }
    if (request.method() === 'POST' && path === '/session/start') {
      return {
        body: {
          session_id: 'quiz-retry',
          total: 1,
          questions: [{ index: 0, question: '请选择首字母', options: ['Alpha', 'Beta'] }],
          revision: 1,
          expires_at: FUTURE_EXPIRES_AT,
        },
      }
    }
    if (request.method() === 'GET' && path === '/session/quiz-retry') {
      snapshotReads += 1
      if (snapshotReads === 1) {
        markFirstReadStarted()
        await firstReadGate
      }
      const answered = snapshotReads > 2
      return {
        body: {
          schema_version: 1,
          origin: 'standard',
          session_id: 'quiz-retry',
          document_id: 'notes.md',
          revision: answered ? 2 : 1,
          status: answered ? 'completed' : 'active',
          total: 1,
          answered_count: answered ? 1 : 0,
          questions: [{ index: 0, question: '请选择首字母', options: ['Alpha', 'Beta'], type: 'choice' }],
          last_answer_index: answered ? 0 : null,
          last_user_answer: answered ? 'Alpha' : null,
          last_answer_result: answered ? {
            evaluation_status: 'final',
            correct: true,
            correct_answer: 'Alpha',
            explanation: '已安全恢复答案',
            is_last: true,
            next_index: null,
            revision: 2,
            expires_at: FUTURE_EXPIRES_AT,
          } : null,
          result: answered ? {
            session_id: 'quiz-retry',
            document_id: 'notes.md',
            total: 1,
            correct: 1,
            incorrect: 0,
            pending: 0,
            score: 1,
            details: [],
            revision: 2,
            expires_at: FUTURE_EXPIRES_AT,
          } : null,
          grading_report: null,
          learning_report: null,
          expires_at: FUTURE_EXPIRES_AT,
          busy: false,
        },
      }
    }
    if (request.method() === 'POST' && path === '/session/quiz-retry/answer') {
      answerAttempts.push({
        body: await request.postDataJSON(),
        key: await request.headerValue('idempotency-key'),
      })
      if (answerAttempts.length === 1) {
        return { status: 503, body: { detail: '答案提交暂时不可用' } }
      }
      return {
        body: {
          evaluation_status: 'final',
          correct: true,
          correct_answer: 'Alpha',
          explanation: '已安全恢复答案',
          is_last: true,
          next_index: null,
          revision: 2,
          expires_at: FUTURE_EXPIRES_AT,
        },
      }
    }
    return null
  })

  try {
    await page.goto('/quiz')
    await page.getByRole('combobox', { name: '学习文档' }).selectOption('notes.md')
    await page.getByRole('button', { name: '开始答题' }).click()

    await page.getByRole('button', { name: /Alpha/ }).click()
    await page.getByRole('button', { name: '提交答案' }).click()
    await firstReadStarted
    await expect(page.getByText('正在恢复答题进度...')).toBeVisible()
    expect(answerAttempts).toHaveLength(1)
    expect(await page.evaluate(key => JSON.parse(sessionStorage.getItem(key)), QUIZ_RECOVERY_KEY))
      .toMatchObject({
        session: { session_id: 'quiz-retry', revision: 1 },
        pending_answer: { question_index: 0, answer: 'Alpha' },
      })

    const firstReadAborted = waitForAbortedApiRequest(
      page,
      'GET',
      '/session/quiz-retry',
    )
    await page.reload()
    expectBrowserAbort(await firstReadAborted)
    releaseFirstRead()
    await expect(page.getByText('已安全恢复答案')).toBeVisible()

    expect(answerAttempts.map(attempt => attempt.body)).toEqual([
      { answer: 'Alpha', question_index: 0 },
      { answer: 'Alpha', question_index: 0 },
    ])
    expect(answerAttempts[0].key).toMatch(UUID_V4_PATTERN)
    expect(answerAttempts[1].key).toBe(answerAttempts[0].key)
    expect(snapshotReads).toBe(3)
    expect(
      await page.evaluate(key => JSON.parse(sessionStorage.getItem(key)), QUIZ_RECOVERY_KEY)
    ).toMatchObject({ pending_answer: null, acknowledged_answer_count: 0 })
    expect(unexpectedRequests).toEqual([])
  } finally {
    releaseFirstRead()
  }
})

test('quiz snapshot restores a completed learning report without replaying mutations', async ({ page }) => {
  const recovery = {
    schema_version: 1,
    intent: {
      kind: 'standard',
      request: {
        document_id: 'notes.md',
        description: '',
        count: 1,
        difficulty: 'medium',
        type: 'choice',
        user_id: 'default_user',
      },
    },
    start_idempotency_key: '11111111-1111-4111-8111-111111111111',
    session: {
      session_id: 'quiz-report-recovered',
      revision: 3,
      expires_at: FUTURE_EXPIRES_AT,
    },
    acknowledged_answer_count: 1,
    pending_answer: null,
  }
  await page.addInitScript(({ key, value }) => {
    sessionStorage.setItem(key, JSON.stringify(value))
  }, { key: QUIZ_RECOVERY_KEY, value: recovery })

  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md'] } }
    }
    if (request.method() === 'GET' && path === '/session/quiz-report-recovered') {
      return {
        body: {
          schema_version: 1,
          origin: 'standard',
          session_id: 'quiz-report-recovered',
          document_id: 'notes.md',
          revision: 4,
          status: 'completed',
          total: 1,
          answered_count: 1,
          questions: [{ index: 0, question: '已完成题目', options: ['A', 'B'], type: 'choice' }],
          last_answer_index: 0,
          last_user_answer: 'A',
          last_answer_result: {
            evaluation_status: 'final',
            correct: true,
            correct_answer: 'A',
            explanation: '回答正确',
            is_last: true,
            next_index: null,
            revision: 4,
            expires_at: FUTURE_EXPIRES_AT,
          },
          result: {
            session_id: 'quiz-report-recovered',
            document_id: 'notes.md',
            total: 1,
            correct: 1,
            incorrect: 0,
            pending: 0,
            score: 1,
            details: [],
            revision: 4,
            expires_at: FUTURE_EXPIRES_AT,
          },
          grading_report: {
            session_id: 'quiz-report-recovered',
            total: 1,
            correct: 1,
            score: 1,
            grades: [],
          },
          learning_report: {
            session_id: 'quiz-report-recovered',
            document_id: 'notes.md',
            overall_score: 1,
            summary: '刷新后恢复的学习报告',
            topic_mastery: [],
            strengths: ['基础概念'],
            weaknesses: [],
            recommendations: ['继续复习'],
          },
          expires_at: FUTURE_EXPIRES_AT,
          busy: false,
        },
      }
    }
    return null
  })

  await page.goto('/quiz')
  await expect(page.getByRole('heading', { name: '学习评估报告' })).toBeVisible()
  await expect(page.getByText('刷新后恢复的学习报告')).toBeVisible()
  expect(unexpectedRequests).toEqual([])
})

test('quiz highlights the correct option for supported labels, separators, and text', async ({ page }) => {
  const correctAnswers = ['C', 'C. 正确文本', '正确文本', ' C ） 正确   文本 ']
  let answerRequests = 0
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md'] } }
    }
    if (request.method() === 'POST' && path === '/session/start') {
      return {
        body: {
          session_id: 'quiz-answer-format',
          total: correctAnswers.length,
          questions: correctAnswers.map((_, index) => ({
            index,
            question: `答案匹配 ${index + 1}`,
            options: [
              'A. 错误文本',
              'B. 其他文本',
              index === correctAnswers.length - 1 ? 'C）正确 文本' : 'C. 正确文本',
            ],
          })),
        },
      }
    }
    if (request.method() === 'POST' && path === '/session/quiz-answer-format/answer') {
      const correctAnswer = correctAnswers[answerRequests]
      answerRequests += 1
      return {
        body: {
          correct: false,
          correct_answer: correctAnswer,
          explanation: `后端返回 ${correctAnswer}`,
          is_last: answerRequests === correctAnswers.length,
          next_index: answerRequests === correctAnswers.length ? null : answerRequests,
        },
      }
    }
    return null
  })

  await page.goto('/quiz')
  await page.getByRole('combobox', { name: '学习文档' }).selectOption('notes.md')
  await page.getByRole('button', { name: '开始答题' }).click()

  for (let index = 0; index < correctAnswers.length; index += 1) {
    const options = page.getByRole('group', { name: `答案匹配 ${index + 1}` })
    await options.getByRole('button', { name: /A\. 错误文本/ }).click()
    await page.getByRole('button', { name: '提交答案' }).click()

    const correctOption = options.getByRole('button', { name: /C(?:\.|）)\s*正确/ })
    await expect(correctOption).toHaveClass(/\bcorrect\b/)
    await expect(correctOption.locator('.option-check')).toHaveCount(1)
    await expect(options.getByRole('button', { name: /A\. 错误文本/ })).toHaveClass(/\bwrong\b/)

    if (index < correctAnswers.length - 1) {
      await page.getByRole('button', { name: '下一题' }).click()
    }
  }

  expect(answerRequests).toBe(4)
  expect(unexpectedRequests).toEqual([])
})

test('quiz returns to setup instead of retrying a terminal answer conflict', async ({ page }) => {
  let answerRequests = 0
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md'] } }
    }
    if (request.method() === 'POST' && path === '/session/start') {
      return {
        body: {
          session_id: 'quiz-terminal',
          total: 1,
          questions: [{ index: 0, question: '旧答题会话', options: ['Alpha', 'Beta'] }],
        },
      }
    }
    if (request.method() === 'POST' && path === '/session/quiz-terminal/answer') {
      answerRequests += 1
      return {
        status: 409,
        body: {
          detail: '答案状态不明确，请重新开始',
          code: 'idempotency_conflict',
          reason: 'ambiguous',
        },
      }
    }
    return null
  })

  await page.goto('/quiz')
  await page.getByRole('combobox', { name: '学习文档' }).selectOption('notes.md')
  await page.getByRole('button', { name: '开始答题' }).click()
  await page.getByRole('button', { name: /Alpha/ }).click()
  await page.getByRole('button', { name: '提交答案' }).click()

  await expect(page.getByRole('alert')).toContainText('答案状态不明确')
  await expect(page.getByRole('combobox', { name: '学习文档' })).toBeVisible()
  await expect(page.getByText('旧答题会话')).toHaveCount(0)
  await expect(page.getByRole('button', { name: '提交答案' })).toHaveCount(0)
  expect(answerRequests).toBe(1)
  expect(unexpectedRequests).toEqual([])
})

test('document upload ignores click and drop while the current upload is in progress', async ({ page }) => {
  let uploadRequests = 0
  let uploaded = false
  let releaseUpload
  let markUploadStarted
  const uploadGate = new Promise(resolve => { releaseUpload = resolve })
  const uploadStarted = new Promise(resolve => { markUploadStarted = resolve })
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: uploaded ? ['first.md'] : [] } }
    }
    if (request.method() === 'POST' && path === '/documents/upload') {
      uploadRequests += 1
      if (uploadRequests === 1) {
        markUploadStarted()
        await uploadGate
      }
      uploaded = true
      return { body: { document_id: 'first.md', chunks: 1 } }
    }
    return null
  })

  await page.goto('/documents')
  const uploadZone = page.getByRole('button', { name: '选择要上传的学习材料' })
  const fileInput = page.locator('input[type="file"]')
  await fileInput.evaluate(input => {
    window.__studyLoopFileInputClicks = 0
    input.addEventListener('click', () => { window.__studyLoopFileInputClicks += 1 })
  })

  await fileInput.setInputFiles({
    name: 'first.md',
    mimeType: 'text/markdown',
    buffer: Buffer.from('# first'),
  })
  await uploadStarted

  await expect(uploadZone).toHaveAttribute('aria-disabled', 'true')
  await expect(uploadZone).toHaveAttribute('aria-busy', 'true')
  await expect(uploadZone).toHaveAttribute('tabindex', '-1')

  await uploadZone.click({ force: true })
  expect(await page.evaluate(() => window.__studyLoopFileInputClicks)).toBe(0)

  await uploadZone.evaluate(zone => {
    const dataTransfer = new DataTransfer()
    dataTransfer.items.add(new File(['# second'], 'second.md', { type: 'text/markdown' }))
    zone.dispatchEvent(new DragEvent('drop', {
      bubbles: true,
      cancelable: true,
      dataTransfer,
    }))
  })
  await page.waitForTimeout(100)
  expect(uploadRequests).toBe(1)

  releaseUpload()
  await expect(uploadZone).toHaveAttribute('aria-busy', 'false', { timeout: 3_000 })
  expect(unexpectedRequests).toEqual([])
})

test('document load failure is recoverable and never shown as an empty library', async ({ page }) => {
  let failDocuments = true
  let documentRequests = 0
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      documentRequests += 1
      return failDocuments
        ? { status: 503, body: { detail: '文档服务暂时不可用' } }
        : { body: { documents: [] } }
    }
    return null
  })

  await page.goto('/documents')

  await expect(page.getByRole('alert')).toContainText('无法加载文档列表')
  await expect(page.getByText('还没有文档')).toHaveCount(0)

  failDocuments = false
  await page.getByRole('button', { name: '重新加载' }).click()
  await expect(page.getByText('还没有文档')).toBeVisible()
  await expect(page.getByRole('alert')).toHaveCount(0)
  expect(documentRequests).toBeGreaterThanOrEqual(2)
  expect(unexpectedRequests).toEqual([])
})

test('API errors show the safe request number for support correlation', async ({ page }) => {
  const requestId = `req_${'7'.repeat(32)}`
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return {
        status: 503,
        headers: { 'X-Request-ID': requestId },
        body: { detail: '文档服务暂时不可用' },
      }
    }
    return null
  })

  await page.goto('/documents')

  const alert = page.getByRole('alert')
  await expect(alert).toContainText('文档服务暂时不可用')
  await expect(alert).toContainText(`请求编号：${requestId}`)
  expect(unexpectedRequests).toEqual([])
})

test('dashboard keeps available data visible when one source fails', async ({ page }) => {
  let failDocuments = true
  let failSessions = false
  let invalidProfile = false
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return failDocuments
        ? { status: 503, body: { detail: '文档列表暂时不可用' } }
        : { body: { documents: ['notes.md'] } }
    }
    if (request.method() === 'GET' && path === '/user/default_user/sessions') {
      return failSessions
        ? { status: 503, body: { detail: '学习记录暂时不可用' } }
        : { body: [{ date: '2026-07-16', correct_rate: 0.75 }] }
    }
    if (request.method() === 'GET' && path === '/user/default_user/profile') {
      if (invalidProfile) return { body: [] }
      return {
        body: {
          topic_mastery: { 'notes.md': 0.8 },
          weak_points: ['检索'],
          total_sessions: 2,
        },
      }
    }
    if (request.method() === 'GET' && path === '/wrong-questions/notes.md') {
      return {
        body: {
          document_id: 'notes.md',
          total: 1,
          entries: [{
            entry_id: 'notes.md:0',
            document_id: 'notes.md',
            question: '不应残留的旧错题',
            options: ['错误', '正确'],
            question_type: 'choice',
            correct_answer: '正确',
            explanation: '旧错题解析',
            user_answer: '错误',
            knowledge_gap: '旧错题',
            session_id: 'old-session',
          }],
        },
      }
    }
    return null
  })

  await page.goto('/dashboard')

  await expect(page.getByRole('alert')).toContainText('部分数据暂不可用')
  await expect(page.getByRole('alert')).toContainText('文档列表')
  await expect(page.locator('.stat-card').filter({ hasText: '学习次数' })).toContainText('2')
  await expect(page.locator('.stat-card').filter({ hasText: '平均正确率' })).toContainText('75%')
  await expect(page.getByText('文档列表暂不可用，暂时无法查看错题')).toBeVisible()
  await expect(page.getByText('还没有学习数据')).toHaveCount(0)

  failDocuments = false
  failSessions = true
  await page.getByRole('button', { name: '重新加载' }).click()
  await expect(page.getByRole('alert')).toContainText('学习记录')
  await expect(page.locator('.stat-card').filter({ hasText: '平均正确率' })).toContainText('—')
  await expect(page.getByText('学习记录暂不可用')).toBeVisible()
  await page.getByLabel('错题文档').selectOption('notes.md')
  await expect(page.getByText('不应残留的旧错题')).toBeVisible()

  failDocuments = true
  failSessions = false
  await page.getByRole('button', { name: '重新加载' }).click()
  await expect(page.getByRole('alert')).toContainText('文档列表')
  await expect(page.locator('.stat-card').filter({ hasText: '平均正确率' })).toContainText('75%')
  await expect(page.getByLabel('错题文档')).toBeDisabled()
  await expect(page.getByText('不应残留的旧错题')).toHaveCount(0)
  await expect(page.getByText('文档列表暂不可用，暂时无法查看错题')).toBeVisible()
  await expect(page.getByRole('heading', { name: '材料掌握度' })).toBeVisible()

  failDocuments = false
  invalidProfile = true
  await page.getByRole('button', { name: '重新加载' }).click()
  await expect(page.getByRole('alert')).toContainText('学习画像（响应格式无效）')
  await expect(page.locator('.stat-card').filter({ hasText: '学习次数' })).toContainText('—')
  await expect(page.locator('.stat-card').filter({ hasText: '平均正确率' })).toContainText('75%')
  await expect(page.getByText('学习画像暂不可用').first()).toBeVisible()
  expect(unexpectedRequests).toEqual([])
})

test('dashboard keeps archived sessions in averages and trend', async ({ page }) => {
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: [] } }
    }
    if (request.method() === 'GET' && path === '/user/default_user/sessions') {
      return {
        body: [
          {
            type: 'archive',
            date: '2026-07-10',
            session_count: 4,
            avg_correct_rate: 0.25,
          },
          { date: '2026-07-16', correct_rate: 0.75 },
          { date: '2026-07-17', correct_rate: 1.0 },
        ],
      }
    }
    if (request.method() === 'GET' && path === '/user/default_user/profile') {
      return {
        body: {
          topic_mastery: { 'notes.md': 0.7 },
          weak_points: [],
          total_sessions: 6,
          average_correct_rate: 0.458,
        },
      }
    }
    return null
  })

  await page.goto('/dashboard')

  await expect(page.locator('.stat-card').filter({ hasText: '学习次数' })).toContainText('6')
  await expect(page.locator('.stat-card').filter({ hasText: '平均正确率' })).toContainText('46%')
  await expect(page.locator('.trend-dot')).toHaveCount(3)
  await expect(page.locator('.trend-svg title').first()).toContainText('4 次历史学习聚合：25%')
  expect(unexpectedRequests).toEqual([])
})

test('dashboard keeps a blocking retry state when every source fails', async ({ page }) => {
  let failAll = true
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (
      request.method() === 'GET'
      && [
        '/documents',
        '/user/default_user/sessions',
        '/user/default_user/profile',
      ].includes(path)
    ) {
      if (failAll) {
        return { status: 503, body: { detail: '学习报告数据暂时不可用' } }
      }
      if (path === '/documents') return { body: { documents: [] } }
      if (path.endsWith('/sessions')) return { body: [] }
      return { body: null }
    }
    return null
  })

  await page.goto('/dashboard')

  await expect(page.getByRole('alert')).toContainText('无法加载学习报告')
  await expect(page.getByText('还没有学习数据')).toHaveCount(0)

  failAll = false
  await page.getByRole('button', { name: '重新加载' }).click()
  await expect(page.getByText('还没有学习数据')).toBeVisible()
  await expect(page.getByRole('alert')).toHaveCount(0)
  expect(unexpectedRequests).toEqual([])
})

test('wrong-question load failure stays distinct from a valid empty result', async ({ page }) => {
  let failWrongQuestions = true
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md'] } }
    }
    if (request.method() === 'GET' && path === '/user/default_user/sessions') {
      return { body: [{ date: '2026-07-16', correct_rate: 0.5 }] }
    }
    if (request.method() === 'GET' && path === '/user/default_user/profile') {
      return { body: { topic_mastery: {}, weak_points: [], total_sessions: 1 } }
    }
    if (request.method() === 'GET' && path === '/wrong-questions/notes.md') {
      return failWrongQuestions
        ? { status: 503, body: { detail: '错题服务暂时不可用' } }
        : { body: { document_id: 'notes.md', total: 0, entries: [] } }
    }
    return null
  })

  await page.goto('/dashboard')
  await page.getByRole('combobox', { name: '错题文档' }).selectOption('notes.md')

  await expect(page.getByRole('alert')).toContainText('无法加载错题')
  await expect(page.getByText('该文档暂无错题')).toHaveCount(0)

  failWrongQuestions = false
  await page.getByRole('button', { name: '重新加载' }).click()
  await expect(page.getByText('该文档暂无错题')).toBeVisible()
  await expect(page.getByRole('alert')).toHaveCount(0)
  expect(unexpectedRequests).toEqual([])
})

test('late wrong-question responses cannot overwrite the selected document', async ({ page }) => {
  let releaseFirstRequest
  let markFirstRequestStarted
  const firstRequestGate = new Promise(resolve => { releaseFirstRequest = resolve })
  const firstRequestStarted = new Promise(resolve => { markFirstRequestStarted = resolve })
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['a.md', 'b.md'] } }
    }
    if (request.method() === 'GET' && path === '/user/default_user/sessions') {
      return { body: [{ date: '2026-07-16', correct_rate: 0.5 }] }
    }
    if (request.method() === 'GET' && path === '/user/default_user/profile') {
      return { body: { topic_mastery: {}, weak_points: [], total_sessions: 1 } }
    }
    if (request.method() === 'GET' && path === '/wrong-questions/a.md') {
      markFirstRequestStarted()
      await firstRequestGate
      return {
        body: {
          document_id: 'a.md',
          total: 1,
          entries: [{
            entry_id: 'a-1',
            question: 'A 文档错题',
            user_answer: 'A',
            correct_answer: 'B',
            explanation: '来自较慢的旧请求',
          }],
        },
      }
    }
    if (request.method() === 'GET' && path === '/wrong-questions/b.md') {
      return {
        body: {
          document_id: 'b.md',
          total: 1,
          entries: [{
            entry_id: 'b-1',
            question: 'B 文档错题',
            user_answer: 'A',
            correct_answer: 'B',
            explanation: '来自当前选择',
          }],
        },
      }
    }
    return null
  })

  await page.goto('/dashboard')
  const documentSelect = page.getByRole('combobox', { name: '错题文档' })
  await documentSelect.selectOption('a.md')
  await firstRequestStarted
  await documentSelect.selectOption('b.md')
  await expect(page.getByText('B 文档错题')).toBeVisible()

  const firstResponse = page.waitForResponse(
    response => new URL(response.url()).pathname.endsWith('/wrong-questions/a.md')
  )
  releaseFirstRequest()
  await firstResponse
  await page.waitForTimeout(100)

  await expect(documentSelect).toHaveValue('b.md')
  await expect(page.getByText('B 文档错题')).toBeVisible()
  await expect(page.getByText('A 文档错题')).toHaveCount(0)
  expect(unexpectedRequests).toEqual([])
})

test('Autonomous loads documents on entry and refresh recovers without breaking the form', async ({ page }) => {
  let failDocuments = true
  let documentRequests = 0
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      documentRequests += 1
      return failDocuments
        ? { status: 503, body: { detail: '文档服务暂时不可用' } }
        : { body: { documents: ['notes.md'] } }
    }
    return null
  })

  await page.goto('/autonomous')

  await expect(page.getByRole('alert')).toContainText(
    '文档列表加载失败：文档服务暂时不可用'
  )
  await expect(page.getByLabel('你的学习目标')).toBeEditable()

  const refresh = page.getByRole('button', { name: '刷新文档' })
  failDocuments = false
  await refresh.click()
  await expect(page.getByRole('alert')).toHaveCount(0)
  await expect(refresh).toBeEnabled()
  expect(documentRequests).toBeGreaterThanOrEqual(2)
  expect(unexpectedRequests).toEqual([])
})

test('Autonomous grounds a selected document and renders server evidence as plain text', async ({ page }) => {
  const problems = trackBrowserProblems(page)
  let startRequest
  const snippet = '<img src=x onerror=alert(1)> RAG 先检索相关片段，再生成回答。'
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'POST' && path === '/agent/autonomous') {
      startRequest = await request.postDataJSON()
      return {
        body: {
          awaiting_user_input: false,
          final_answer: 'RAG 会先检索材料，再基于检索内容生成回答。',
          finalize_reason: 'evidence_ready',
          rounds_used: 2,
          steps: [],
          tools_called: ['search_document'],
          truncated: false,
          citations: [{
            chunk_id: 'notes.md_chunk_3',
            document_id: 'notes.md',
            chunk_index: 3,
            rank: 1,
            snippet,
          }],
          invalid_citation_ids: [],
          invalid_citation_count: 0,
          abstained: false,
          grounding_status: 'citation_ids_valid',
          grounding_required: true,
          grounding_document_id: 'notes.md',
        },
      }
    }
    return null
  })

  await page.goto('/autonomous')
  const grounding = page.getByRole('checkbox', { name: /要求可核验文档引用/ })
  const documentId = page.getByLabel('文档 ID（可选）')
  await expect(grounding).toBeDisabled()
  await expect(grounding).not.toBeChecked()

  await documentId.fill('notes.md')
  await expect(grounding).toBeEnabled()
  await expect(grounding).toBeChecked()
  await documentId.fill('')
  await expect(grounding).toBeDisabled()
  await expect(grounding).not.toBeChecked()
  await documentId.fill('notes.md')
  await expect(grounding).toBeChecked()
  await grounding.uncheck()
  await expect(grounding).not.toBeChecked()
  await grounding.check()

  await page.getByLabel('你的学习目标').fill('解释这份材料里的 RAG')
  await page.getByRole('button', { name: '开始执行' }).click()

  expect(startRequest).toEqual({
    query: '解释这份材料里的 RAG',
    user_id: 'default_user',
    document_id: 'notes.md',
    grounding_required: true,
  })
  await expect(page.getByText('引用 ID 已核验')).toBeVisible()
  await expect(page.getByText('文档证据')).toBeVisible()
  await expect(page.locator('.citation-snippet')).toContainText(snippet)
  await expect(page.locator('.citation-snippet img')).toHaveCount(0)
  await expect(page.getByText('notes.md_chunk_3')).toBeVisible()
  expect(unexpectedRequests).toEqual([])
  expect(problems).toEqual([])
})

test('Autonomous shows a strict grounding refusal without fabricated evidence', async ({ page }) => {
  let startRequest
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'POST' && path === '/agent/autonomous') {
      startRequest = await request.postDataJSON()
      return {
        body: {
          awaiting_user_input: false,
          final_answer: '现有检索证据不足，无法提供满足完整引用约束的回答。',
          finalize_reason: 'grounding_required_with_invalid_citation',
          rounds_used: 2,
          steps: [{
            round_index: 0,
            tool_name: 'search_document',
            tool_args: null,
            observation_preview: null,
          }],
          tools_called: ['search_document'],
          truncated: false,
          citations: [],
          invalid_citation_ids: [],
          invalid_citation_count: 1,
          abstained: true,
          grounding_status: 'abstained',
          grounding_required: true,
          grounding_document_id: 'notes.md',
        },
      }
    }
    return null
  })

  await page.goto('/autonomous')
  await page.getByLabel('文档 ID（可选）').fill('notes.md')
  await page.getByLabel('你的学习目标').fill('给出有引用的结论')
  await page.getByRole('button', { name: '开始执行' }).click()

  expect(startRequest.grounding_required).toBe(true)
  await expect(page.getByText('证据不足，已安全拒答')).toBeVisible()
  await expect(page.locator('.result-final')).toHaveClass(/result-abstained/)
  await expect(page.getByText('文档证据')).toHaveCount(0)
  await expect(page.getByText('已丢弃 1 个不属于本轮检索结果的引用 ID。')).toBeVisible()
  expect(unexpectedRequests).toEqual([])
})

test('document delete control keeps a mobile-sized touch target', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['mobile-notes.md'] } }
    }
    return null
  })

  await page.goto('/documents')
  const deleteButton = page.getByRole('button', { name: '删除文档 mobile-notes.md' })
  const box = await deleteButton.boundingBox()

  expect(Math.round(box.width)).toBeGreaterThanOrEqual(44)
  expect(Math.round(box.height)).toBeGreaterThanOrEqual(44)
  expect(unexpectedRequests).toEqual([])
})

test('document deletion requires an explicit material-only confirmation', async ({ page }) => {
  let deleted = false
  let deleteRequests = 0
  let releaseDelete
  const deleteGate = new Promise(resolve => { releaseDelete = resolve })
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: deleted ? [] : ['notes.md'] } }
    }
    if (request.method() === 'DELETE' && path === '/documents/notes.md') {
      deleteRequests += 1
      deleted = true
      await deleteGate
      return {
        body: {
          status: 'material_deleted',
          document_id: 'notes.md',
          scope: 'material_only',
          learning_data_retained: true,
          document_id_reusable: false,
        },
      }
    }
    return null
  })

  await page.goto('/documents')
  const deleteButton = page.getByRole('button', { name: '删除文档 notes.md' })
  await deleteButton.click()

  const dialog = page.getByRole('alertdialog', { name: '确认删除“notes.md”？' })
  await expect(dialog).toBeVisible()
  await expect(page.locator('.documents-content')).toHaveAttribute('aria-hidden', 'true')
  await expect(page.locator('.documents-content')).toHaveJSProperty('inert', true)
  await expect(dialog).toContainText('这不是隐私数据彻底清除')
  await expect(dialog).toContainText('之后如需重新上传，请先重命名文件')
  expect(deleteRequests).toBe(0)

  await page.keyboard.press('Escape')
  await expect(dialog).toHaveCount(0)
  await expect(page.locator('.documents-content')).not.toHaveAttribute('aria-hidden')
  await expect(page.locator('.documents-content')).toHaveJSProperty('inert', false)
  await expect(deleteButton).toBeFocused()
  expect(deleteRequests).toBe(0)

  await deleteButton.click()
  await page.getByRole('button', { name: '确认仅删除材料' }).click()
  await expect(dialog).toHaveAttribute('aria-busy', 'true')
  await expect(dialog.getByRole('button', { name: '取消' })).toBeDisabled()
  await expect(dialog.getByRole('button', { name: '正在删除材料…' })).toBeDisabled()
  await page.keyboard.press('Tab')
  await expect(dialog).toBeFocused()
  releaseDelete()
  await expect(page.getByRole('heading', { name: '确认删除“notes.md”？' })).toHaveCount(0)
  await expect(page.getByText('notes.md', { exact: true })).toHaveCount(0)
  await expect(page.locator('.documents-status')).toContainText('学习记录仍保留')
  expect(deleteRequests).toBe(1)
  expect(unexpectedRequests).toEqual([])
})

test('document deletion reconciles a lost success response with the durable tombstone', async ({ page }) => {
  let deleted = false
  let deleteRequests = 0
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: deleted ? [] : ['notes.md'] } }
    }
    if (request.method() === 'DELETE' && path === '/documents/notes.md') {
      deleteRequests += 1
      deleted = true
      if (deleteRequests === 1) {
        return { status: 503, body: { detail: '响应在提交后丢失' } }
      }
      return {
        body: {
          status: 'material_deleted',
          document_id: 'notes.md',
          scope: 'material_only',
          learning_data_retained: true,
          document_id_reusable: false,
        },
      }
    }
    return null
  })

  await page.goto('/documents')
  await page.getByRole('button', { name: '删除文档 notes.md' }).click()
  await page.getByRole('button', { name: '确认仅删除材料' }).click()

  await expect(page.getByText('notes.md', { exact: true })).toHaveCount(0)
  await expect(page.locator('.documents-status')).toContainText('学习记录仍保留')
  expect(deleteRequests).toBe(2)
  expect(unexpectedRequests).toEqual([])
})

test('document deletion never leaves a hidden tombstone displayed as indexed', async ({ page }) => {
  let hidden = false
  let deleteRequests = 0
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: hidden ? [] : ['notes.md'] } }
    }
    if (request.method() === 'DELETE' && path === '/documents/notes.md') {
      hidden = true
      deleteRequests += 1
      return { status: 503, body: { detail: '删除收尾暂时失败' } }
    }
    return null
  })

  await page.goto('/documents')
  await page.getByRole('button', { name: '删除文档 notes.md' }).click()
  await page.getByRole('button', { name: '确认仅删除材料' }).click()

  const dialog = page.getByRole('alertdialog', { name: '确认删除“notes.md”？' })
  await expect(dialog.getByRole('alert')).toContainText('删除状态尚未确认')
  await expect(page.getByRole('heading', { name: 'notes.md', level: 3 })).toHaveCount(0)
  await expect(page.locator('.documents-status')).toContainText('删除收尾尚未确认')
  expect(deleteRequests).toBe(2)

  await dialog.getByRole('button', { name: '取消' }).click()
  await expect(dialog).toHaveCount(0)
  await expect(page.getByRole('heading', { name: 'notes.md', level: 3 })).toHaveCount(0)
  expect(unexpectedRequests).toEqual([])
})

test('document upload reconciles a lost response while a StrictMode list request finishes late', async ({ page }) => {
  let uploaded = false
  let getRequests = 0
  let releaseOldList
  const oldList = new Promise(resolve => { releaseOldList = resolve })
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      getRequests += 1
      if (getRequests === 1) {
        await oldList
        return { body: { documents: [] } }
      }
      return { body: { documents: uploaded ? ['notes.md'] : [] } }
    }
    if (request.method() === 'POST' && path === '/documents/upload') {
      uploaded = true
      return { status: 503, body: { detail: '响应在提交后丢失' } }
    }
    return null
  })

  await page.goto('/documents')
  const fileInput = page.locator('input[type="file"]')
  // StrictMode's first mount request is intentionally held open; the second
  // authoritative request makes the page safe to use before upload starts.
  await expect(fileInput).toBeEnabled()
  await fileInput.setInputFiles({
    name: 'notes.md',
    mimeType: 'text/markdown',
    buffer: Buffer.from('# notes'),
  })

  await expect(page.getByText('完成，已从文档列表同步确认')).toBeVisible()
  await expect(page.getByRole('heading', { name: 'notes.md' })).toBeVisible()
  releaseOldList()
  await expect.poll(() => getRequests).toBeGreaterThanOrEqual(2)
  await expect(page.getByRole('heading', { name: 'notes.md' })).toBeVisible()
  expect(unexpectedRequests).toEqual([])
})

test('document upload waits for the authoritative list and never treats a 409 as success', async ({ page }) => {
  let releaseList
  let uploadRequests = 0
  const initialList = new Promise(resolve => { releaseList = resolve })
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      await initialList
      return { body: { documents: ['notes.md'] } }
    }
    if (request.method() === 'POST' && path === '/documents/upload') {
      uploadRequests += 1
      return { status: 409, body: { detail: '文档已存在，请重命名后上传' } }
    }
    return null
  })

  await page.goto('/documents')
  const fileInput = page.locator('input[type="file"]')
  await expect(fileInput).toBeDisabled()
  await expect(page.getByRole('button', { name: '选择要上传的学习材料' })).toHaveAttribute(
    'aria-disabled',
    'true',
  )
  expect(uploadRequests).toBe(0)

  releaseList()
  await expect(page.getByRole('heading', { name: 'notes.md' })).toBeVisible()
  await expect(fileInput).toBeEnabled()
  await fileInput.setInputFiles({
    name: 'notes.md',
    mimeType: 'text/markdown',
    buffer: Buffer.from('# different notes'),
  })

  await expect(page.getByRole('alert')).toContainText('文档已存在，请重命名后上传')
  await expect(page.getByText('完成，已从文档列表同步确认')).toHaveCount(0)
  await expect(page.getByRole('heading', { name: 'notes.md' })).toBeVisible()
  expect(uploadRequests).toBe(1)
  expect(unexpectedRequests).toEqual([])
})

test('Autonomous HITL dialog isolates the page and resumes with the reply', async ({ page }) => {
  const problems = trackBrowserProblems(page)
  let startRequest
  let continueRequest
  let startKey
  let continueKey

  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'POST' && path === '/agent/autonomous') {
      startRequest = await request.postDataJSON()
      startKey = await request.headerValue('idempotency-key')
      return {
        body: {
          awaiting_user_input: true,
          conversation_id: 'playwright-hitl-check',
          user_question: '你希望重点学习哪一章？',
          rounds_used: 1,
          steps: [],
          tools_called: ['ask_user'],
          truncated: false,
        },
      }
    }

    if (request.method() === 'POST' && path === '/agent/autonomous/continue') {
      continueRequest = await request.postDataJSON()
      continueKey = await request.headerValue('idempotency-key')
      return {
        body: {
          awaiting_user_input: false,
          conversation_id: 'playwright-hitl-check',
          final_answer: '已根据你的回答生成学习建议。',
          finalize_reason: '测试完成',
          rounds_used: 2,
          steps: [],
          tools_called: ['ask_user', 'finalize'],
          truncated: false,
        },
      }
    }
    return null
  })

  await page.goto('/autonomous')
  await page.getByLabel('你的学习目标').fill('帮我制定学习计划')
  await page.getByRole('button', { name: '开始执行' }).click()

  const dialog = page.getByRole('dialog', { name: 'Agent 想问你' })
  const reply = page.getByRole('textbox', { name: '你的回答' })
  await expect(dialog).toHaveAttribute('aria-modal', 'true')
  await expect(reply).toBeFocused()
  await expect(page.locator('.autonomous-content')).toHaveAttribute('aria-hidden', 'true')
  await expect(page.locator('.autonomous-content')).toHaveJSProperty('inert', true)

  await page.keyboard.press('Shift+Tab')
  await expect(dialog.getByRole('button', { name: '取消整个执行' })).toBeFocused()
  await page.keyboard.press('Tab')
  await expect(reply).toBeFocused()

  await reply.fill('重点学习第三章')
  await dialog.getByRole('button', { name: /回答/ }).click()

  await expect(dialog).toBeHidden()
  await expect(page.getByText('已根据你的回答生成学习建议。')).toBeVisible()
  await expect(page.locator('.autonomous-content')).not.toHaveAttribute('aria-hidden')
  await expect(page.locator('.autonomous-content')).toHaveJSProperty('inert', false)
  expect(startRequest).toMatchObject({ query: '帮我制定学习计划' })
  expect(continueRequest).toEqual({
    conversation_id: 'playwright-hitl-check',
    user_reply: '重点学习第三章',
  })
  expect(startKey).toMatch(UUID_V4_PATTERN)
  expect(continueKey).toMatch(UUID_V4_PATTERN)
  expect(continueKey).not.toBe(startKey)
  expect(unexpectedRequests).toEqual([])
  expect(problems).toEqual([])
})

test('Autonomous restores a pending HITL question and draft after reload', async ({ page }) => {
  let continueRequest
  let continueKey
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'POST' && path === '/agent/autonomous') {
      return {
        body: {
          awaiting_user_input: true,
          conversation_id: 'playwright-hitl-reload',
          user_question: '刷新后仍需回答的问题？',
          rounds_used: 1,
          steps: [{ round_index: 0, tool_name: 'ask_user', tool_args: {} }],
          tools_called: ['ask_user'],
          truncated: false,
        },
      }
    }
    if (request.method() === 'POST' && path === '/agent/autonomous/continue') {
      continueRequest = await request.postDataJSON()
      continueKey = await request.headerValue('idempotency-key')
      return {
        body: {
          awaiting_user_input: false,
          conversation_id: 'playwright-hitl-reload',
          final_answer: '刷新恢复后提交成功。',
          finalize_reason: '恢复完成',
          rounds_used: 2,
          steps: [],
          tools_called: ['ask_user', 'finalize'],
          truncated: false,
        },
      }
    }
    return null
  })

  await page.goto('/autonomous')
  await page.getByLabel('你的学习目标').fill('保留这个学习目标')
  await expect(page.getByLabel('用户 ID')).toHaveCount(0)
  await page.getByLabel('文档 ID（可选）').fill('reload.md')
  await page.getByRole('button', { name: '开始执行' }).click()

  const dialog = page.getByRole('dialog', { name: 'Agent 想问你' })
  await expect(dialog).toContainText('本轮已启用引用约束')
  await dialog.getByRole('textbox', { name: '你的回答' }).fill('保留这个回答草稿')
  const stored = await page.evaluate(key => JSON.parse(sessionStorage.getItem(key)), AUTONOMOUS_SESSION_KEY)
  expect(stored).toMatchObject({
    schema_version: 2,
    kind: 'awaiting',
    conversation_id: 'playwright-hitl-reload',
    user_question: '刷新后仍需回答的问题？',
    draft: '保留这个回答草稿',
    request: {
      query: '保留这个学习目标',
      user_id: 'default_user',
      document_id: 'reload.md',
      grounding_required: true,
    },
  })
  expect(stored).not.toHaveProperty('steps')

  await page.reload()
  await expect(dialog).toBeVisible()
  await expect(dialog).toContainText('刷新后仍需回答的问题？')
  await expect(dialog).toContainText('本轮已启用引用约束')
  await expect(dialog.getByRole('textbox', { name: '你的回答' })).toHaveValue('保留这个回答草稿')
  await expect(page.getByLabel('你的学习目标')).toHaveValue('保留这个学习目标')

  await dialog.getByRole('button', { name: /回答/ }).click()
  await expect(page.getByText('刷新恢复后提交成功。')).toBeVisible()
  expect(continueRequest).toEqual({
    conversation_id: 'playwright-hitl-reload',
    user_reply: '保留这个回答草稿',
  })
  expect(continueKey).toMatch(UUID_V4_PATTERN)
  expect(await page.evaluate(key => sessionStorage.getItem(key), AUTONOMOUS_SESSION_KEY)).toBeNull()

  await page.reload()
  await expect(page.getByRole('dialog', { name: 'Agent 想问你' })).toHaveCount(0)
  await expect(page.getByLabel('你的学习目标')).toHaveValue('')
  expect(unexpectedRequests).toEqual([])
})

test('Autonomous keeps a failed continuation key when recovery crosses a reload', async ({ page }) => {
  const continueAttempts = []
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'POST' && path === '/agent/autonomous') {
      return {
        body: {
          awaiting_user_input: true,
          conversation_id: 'playwright-hitl-reload-retry',
          user_question: '请确认重试内容？',
          rounds_used: 1,
          steps: [],
          tools_called: ['ask_user'],
          truncated: false,
        },
      }
    }
    if (request.method() === 'POST' && path === '/agent/autonomous/continue') {
      continueAttempts.push({
        body: await request.postDataJSON(),
        key: await request.headerValue('idempotency-key'),
      })
      if (continueAttempts.length === 1) {
        return { status: 503, body: { detail: '续跑服务暂时不可用' } }
      }
      return {
        body: {
          awaiting_user_input: false,
          final_answer: '沿用原请求标识后完成。',
          finalize_reason: '安全重试完成',
          rounds_used: 2,
          steps: [],
          tools_called: ['ask_user', 'finalize'],
          truncated: false,
        },
      }
    }
    return null
  })

  await page.goto('/autonomous')
  await page.getByLabel('你的学习目标').fill('测试刷新后的安全重试')
  await page.getByRole('button', { name: '开始执行' }).click()
  const dialog = page.getByRole('dialog', { name: 'Agent 想问你' })
  const reply = dialog.getByRole('textbox', { name: '你的回答' })
  await reply.fill('沿用这份回答')
  await dialog.getByRole('button', { name: /回答/ }).click()
  await expect(dialog.getByRole('alert')).toContainText('续跑服务暂时不可用')

  await page.reload()
  await expect(dialog).toBeHidden()
  await expect(page.getByText('沿用原请求标识后完成。')).toBeVisible()

  expect(continueAttempts.map(attempt => attempt.body)).toEqual([
    { conversation_id: 'playwright-hitl-reload-retry', user_reply: '沿用这份回答' },
    { conversation_id: 'playwright-hitl-reload-retry', user_reply: '沿用这份回答' },
  ])
  expect(continueAttempts[0].key).toMatch(UUID_V4_PATTERN)
  expect(continueAttempts[1].key).toBe(continueAttempts[0].key)
  expect(unexpectedRequests).toEqual([])
})

test('Autonomous reset clears recovery, corrupt storage fails closed, and valid v1 awaiting state migrates', async ({ page }) => {
  const problems = trackBrowserProblems(page)
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'POST' && path === '/agent/autonomous') {
      return {
        body: {
          awaiting_user_input: true,
          conversation_id: 'playwright-hitl-reset',
          user_question: '是否取消？',
          rounds_used: 1,
          steps: [],
          tools_called: ['ask_user'],
          truncated: false,
        },
      }
    }
    if (
      request.method() === 'DELETE'
      && path === '/agent/autonomous/playwright-hitl-reset'
    ) {
      return { body: { status: 'canceled' } }
    }
    return null
  })

  await page.goto('/autonomous')
  await page.getByLabel('你的学习目标').fill('等待取消')
  await page.getByRole('button', { name: '开始执行' }).click()
  expect(await page.evaluate(key => sessionStorage.getItem(key), AUTONOMOUS_SESSION_KEY)).not.toBeNull()

  await page.getByRole('dialog', { name: 'Agent 想问你' })
    .getByRole('button', { name: '取消整个执行' }).click()
  await expect.poll(
    () => page.evaluate(key => sessionStorage.getItem(key), AUTONOMOUS_SESSION_KEY),
  ).toBeNull()

  await page.evaluate(key => sessionStorage.setItem(key, '{not-json'), AUTONOMOUS_SESSION_KEY)
  await page.reload()
  await expect(page.getByRole('dialog', { name: 'Agent 想问你' })).toHaveCount(0)
  await expect(page.getByLabel('你的学习目标')).toHaveValue('')
  expect(await page.evaluate(key => sessionStorage.getItem(key), AUTONOMOUS_SESSION_KEY)).toBeNull()

  await page.evaluate(key => sessionStorage.setItem(key, JSON.stringify({
    version: 1,
    conversation_id: 'legacy-custom-user',
    user_question: '不应恢复的旧用户会话？',
    draft: '旧草稿',
    request: {
      query: '旧学习目标',
      user_id: 'another-user',
      document_id: '',
      grounding_required: false,
    },
    continue_idempotency_key: null,
  })), LEGACY_AUTONOMOUS_SESSION_KEY)
  await page.reload()
  await expect(page.getByRole('dialog', { name: 'Agent 想问你' })).toHaveCount(0)
  await expect(page.getByLabel('你的学习目标')).toHaveValue('')
  expect(await page.evaluate(key => sessionStorage.getItem(key), AUTONOMOUS_SESSION_KEY)).toBeNull()

  await page.evaluate(key => sessionStorage.setItem(key, JSON.stringify({
    version: 1,
    conversation_id: 'legacy-valid-awaiting',
    user_question: '旧版暂停问题仍需回答？',
    draft: '迁移后的草稿',
    request: {
      query: '迁移旧版 Autonomous 会话',
      user_id: 'default_user',
      document_id: '',
      grounding_required: false,
    },
    continue_idempotency_key: null,
  })), LEGACY_AUTONOMOUS_SESSION_KEY)
  await page.reload()
  const migratedDialog = page.getByRole('dialog', { name: 'Agent 想问你' })
  await expect(migratedDialog).toContainText('旧版暂停问题仍需回答？')
  await expect(migratedDialog.getByRole('textbox', { name: '你的回答' }))
    .toHaveValue('迁移后的草稿')
  const migrated = await page.evaluate(
    key => JSON.parse(sessionStorage.getItem(key)),
    AUTONOMOUS_SESSION_KEY,
  )
  expect(migrated).toMatchObject({
    schema_version: 2,
    kind: 'awaiting',
    conversation_id: 'legacy-valid-awaiting',
    draft: '迁移后的草稿',
  })
  expect(await page.evaluate(
    key => sessionStorage.getItem(key),
    LEGACY_AUTONOMOUS_SESSION_KEY,
  )).toBeNull()
  expect(unexpectedRequests).toEqual([])
  expect(problems).toEqual([])
})

test('Autonomous preserves the initial goal and focus when starting fails', async ({ page }) => {
  const startRequests = []
  const startKeys = []
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'POST' && path === '/agent/autonomous') {
      startRequests.push(await request.postDataJSON())
      startKeys.push(await request.headerValue('idempotency-key'))
      if (startRequests.length === 1) {
        return { status: 503, body: { detail: '模型服务暂时不可用' } }
      }
      return {
        body: {
          awaiting_user_input: false,
          conversation_id: 'playwright-start-retry',
          final_answer: '已在重试后开始执行。',
          finalize_reason: '重试成功',
          rounds_used: 1,
          steps: [],
          tools_called: ['finalize'],
          truncated: false,
        },
      }
    }
    return null
  })

  await page.goto('/autonomous')
  const goal = page.getByLabel('你的学习目标')
  await goal.fill('帮我复习向量检索')
  await page.getByRole('button', { name: '开始执行' }).click()

  await expect(page.getByRole('alert')).toContainText('模型服务暂时不可用')
  await expect(goal).toHaveValue('帮我复习向量检索')
  await expect(goal).toBeDisabled()
  const pendingStart = await page.evaluate(
    key => JSON.parse(sessionStorage.getItem(key)),
    AUTONOMOUS_SESSION_KEY,
  )
  expect(pendingStart).toMatchObject({
    kind: 'pending_start',
    idempotency_key: startKeys[0],
  })
  const retry = page.getByRole('button', { name: '再次执行当前目标' })
  await expect(retry).toBeFocused()
  await retry.click()

  await expect(page.getByText('已在重试后开始执行。')).toBeVisible()
  expect(startRequests).toEqual([
    {
      query: '帮我复习向量检索',
      user_id: 'default_user',
      document_id: null,
      grounding_required: false,
    },
    {
      query: '帮我复习向量检索',
      user_id: 'default_user',
      document_id: null,
      grounding_required: false,
    },
  ])
  expect(startKeys[0]).toMatch(UUID_V4_PATTERN)
  expect(startKeys[1]).toBe(startKeys[0])
  expect(unexpectedRequests).toEqual([])
})

test('Autonomous rotates an explicitly rejected start key after edits or reset', async ({ page }) => {
  const attempts = []
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'POST' && path === '/agent/autonomous') {
      attempts.push({
        body: await request.postDataJSON(),
        key: await request.headerValue('idempotency-key'),
      })
      if (attempts.length % 2 === 1) {
        return { status: 422, body: { detail: '请求内容需修改' } }
      }
      return {
        body: {
          awaiting_user_input: false,
          final_answer: '新请求已执行。',
          finalize_reason: '测试完成',
          rounds_used: 1,
          steps: [],
          tools_called: ['finalize'],
          truncated: false,
        },
      }
    }
    return null
  })

  const scenarios = [
    {
      name: 'goal',
      mutate: async () => page.getByLabel('你的学习目标').fill('修改后的目标'),
    },
    {
      name: 'document',
      mutate: async () => page.getByLabel('文档 ID（可选）').fill('edited.md'),
    },
    {
      name: 'reset',
      mutate: async () => {
        await page.getByRole('button', { name: '清空并重新开始' }).click()
        await page.getByLabel('你的学习目标').fill('重置后的目标')
      },
    },
  ]

  for (const scenario of scenarios) {
    await test.step(scenario.name, async () => {
      await page.goto('/autonomous')
      await page.getByLabel('你的学习目标').fill('原始目标')
      const offset = attempts.length
      await page.getByRole('button', { name: '开始执行' }).click()
      await expect(page.getByRole('alert')).toContainText('请求内容需修改')

      await scenario.mutate()
      await page.getByRole('button', { name: /执行/ }).click()
      await expect(page.getByText('新请求已执行。')).toBeVisible()

      expect(attempts[offset].key).toMatch(UUID_V4_PATTERN)
      expect(attempts[offset + 1].key).toMatch(UUID_V4_PATTERN)
      expect(attempts[offset + 1].key).not.toBe(attempts[offset].key)
    })
  }
  expect(unexpectedRequests).toEqual([])
})

test('Autonomous preserves a HITL reply and retries after a continuation failure', async ({ page }) => {
  const continueRequests = []
  const continueKeys = []
  let releaseFirstContinue
  let markFirstContinueStarted
  const firstContinueGate = new Promise(resolve => { releaseFirstContinue = resolve })
  const firstContinueStarted = new Promise(resolve => { markFirstContinueStarted = resolve })
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'POST' && path === '/agent/autonomous') {
      return {
        body: {
          awaiting_user_input: true,
          conversation_id: 'playwright-hitl-retry',
          user_question: '你希望重点学习哪一章？',
          rounds_used: 1,
          steps: [],
          tools_called: ['ask_user'],
          truncated: false,
        },
      }
    }

    if (request.method() === 'POST' && path === '/agent/autonomous/continue') {
      continueRequests.push(await request.postDataJSON())
      continueKeys.push(await request.headerValue('idempotency-key'))
      if (continueRequests.length === 1) {
        markFirstContinueStarted()
        await firstContinueGate
        return { status: 503, body: { detail: '模型服务暂时不可用' } }
      }
      return {
        body: {
          awaiting_user_input: false,
          conversation_id: 'playwright-hitl-retry',
          final_answer: '已在重试后生成学习建议。',
          finalize_reason: '重试成功',
          rounds_used: 2,
          steps: [],
          tools_called: ['ask_user', 'finalize'],
          truncated: false,
        },
      }
    }
    return null
  })

  await page.goto('/autonomous')
  await page.getByLabel('你的学习目标').fill('帮我制定学习计划')
  await page.getByRole('button', { name: '开始执行' }).click()

  const dialog = page.getByRole('dialog', { name: 'Agent 想问你' })
  const reply = dialog.getByRole('textbox', { name: '你的回答' })
  await reply.fill('重点学习第三章')
  await dialog.getByRole('button', { name: /回答/ }).click()

  await firstContinueStarted
  await expect(dialog).toHaveAttribute('aria-busy', 'true')
  await expect(reply).toHaveJSProperty('readOnly', true)
  await expect(reply).toBeFocused()
  releaseFirstContinue()

  await expect(dialog).toBeVisible()
  await expect(dialog).toHaveAttribute('aria-busy', 'false')
  await expect(dialog.getByRole('alert')).toContainText('模型服务暂时不可用')
  await expect(dialog.getByRole('alert')).toContainText('你的回答已保留')
  await expect(reply).toHaveValue('重点学习第三章')
  await expect(reply).toHaveJSProperty('readOnly', true)
  await expect(dialog.getByRole('button', { name: '取消整个执行' })).toBeDisabled()
  await expect(reply).toBeFocused()
  await expect(page.locator('.autonomous-content')).toHaveJSProperty('inert', true)

  await dialog.getByRole('button', { name: '重试回答' }).click()
  await expect(dialog).toBeHidden()
  await expect(page.getByText('已在重试后生成学习建议。')).toBeVisible()
  expect(continueRequests).toEqual([
    { conversation_id: 'playwright-hitl-retry', user_reply: '重点学习第三章' },
    { conversation_id: 'playwright-hitl-retry', user_reply: '重点学习第三章' },
  ])
  expect(continueKeys[0]).toMatch(UUID_V4_PATTERN)
  expect(continueKeys[1]).toBe(continueKeys[0])
  expect(unexpectedRequests).toEqual([])
})

test('Autonomous keeps a 422 HITL draft and rotates its key after editing', async ({ page }) => {
  const continueAttempts = []
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'POST' && path === '/agent/autonomous') {
      return {
        body: {
          awaiting_user_input: true,
          conversation_id: 'playwright-hitl-edit',
          user_question: '你希望重点学习哪一章？',
          rounds_used: 1,
          steps: [],
          tools_called: ['ask_user'],
          truncated: false,
        },
      }
    }
    if (request.method() === 'POST' && path === '/agent/autonomous/continue') {
      continueAttempts.push({
        body: await request.postDataJSON(),
        key: await request.headerValue('idempotency-key'),
      })
      if (continueAttempts.length === 1) {
        return { status: 422, body: { detail: '用户回答安全检查未通过，请修改后重试' } }
      }
      return {
        body: {
          awaiting_user_input: false,
          final_answer: '修改后的回答已提交。',
          finalize_reason: '测试完成',
          rounds_used: 2,
          steps: [],
          tools_called: ['ask_user', 'finalize'],
          truncated: false,
        },
      }
    }
    return null
  })

  await page.goto('/autonomous')
  await page.getByLabel('你的学习目标').fill('制定计划')
  await page.getByRole('button', { name: '开始执行' }).click()
  const dialog = page.getByRole('dialog', { name: 'Agent 想问你' })
  const reply = dialog.getByRole('textbox', { name: '你的回答' })
  await reply.fill('第三章')
  await dialog.getByRole('button', { name: /回答/ }).click()
  await expect(dialog).toBeVisible()
  await expect(dialog.getByRole('alert')).toContainText('用户回答安全检查未通过')
  await expect(dialog.getByRole('alert')).toContainText('你的回答已保留')
  await expect(reply).toHaveValue('第三章')

  await reply.fill('第四章')
  await dialog.getByRole('button', { name: '重试回答' }).click()
  await expect(page.getByText('修改后的回答已提交。')).toBeVisible()

  expect(continueAttempts[0].body.user_reply).toBe('第三章')
  expect(continueAttempts[1].body.user_reply).toBe('第四章')
  expect(continueAttempts[0].key).toMatch(UUID_V4_PATTERN)
  expect(continueAttempts[1].key).toMatch(UUID_V4_PATTERN)
  expect(continueAttempts[1].key).not.toBe(continueAttempts[0].key)
  expect(unexpectedRequests).toEqual([])
})

test('Autonomous exits HITL retry mode when the continuation was consumed', async ({ page }) => {
  let continueRequests = 0
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'POST' && path === '/agent/autonomous') {
      return {
        body: {
          awaiting_user_input: true,
          conversation_id: 'playwright-hitl-consumed',
          user_question: '是否更新学习画像？',
          rounds_used: 1,
          steps: [],
          tools_called: ['ask_user'],
          truncated: false,
        },
      }
    }
    if (request.method() === 'POST' && path === '/agent/autonomous/continue') {
      continueRequests += 1
      return {
        status: 410,
        body: { detail: '续跑已执行部分操作，无法安全重试；请重新开始' },
      }
    }
    return null
  })

  await page.goto('/autonomous')
  await page.getByLabel('你的学习目标').fill('更新我的学习建议')
  await page.getByRole('button', { name: '开始执行' }).click()
  expect(await page.evaluate(key => sessionStorage.getItem(key), AUTONOMOUS_SESSION_KEY)).not.toBeNull()

  const dialog = page.getByRole('dialog', { name: 'Agent 想问你' })
  await dialog.getByRole('textbox', { name: '你的回答' }).fill('继续')
  await dialog.getByRole('button', { name: /回答/ }).click()

  await expect(dialog).toBeHidden()
  await expect(page.getByRole('alert')).toContainText(
    '续跑已执行部分操作，无法安全重试；请重新开始'
  )
  await expect(page.getByRole('button', { name: '重试回答' })).toHaveCount(0)
  await expect(page.getByLabel('你的学习目标')).toHaveValue('更新我的学习建议')
  await expect(page.getByRole('button', { name: '再次执行当前目标' })).toHaveCount(0)
  await expect(page.getByRole('button', { name: '清空并重新开始' })).toBeFocused()
  expect(await page.evaluate(key => sessionStorage.getItem(key), AUTONOMOUS_SESSION_KEY)).toBeNull()
  await page.reload()
  await expect(page.getByRole('dialog', { name: 'Agent 想问你' })).toHaveCount(0)
  expect(continueRequests).toBe(1)
  expect(unexpectedRequests).toEqual([])
})

test('Autonomous blocks blind retry after an ambiguous receipt', async ({ page }) => {
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'POST' && path === '/agent/autonomous') {
      return {
        status: 409,
        body: {
          detail: '此前请求可能已执行写操作，请刷新学习状态后重新开始',
          code: 'idempotency_conflict',
          reason: 'ambiguous',
        },
      }
    }
    return null
  })

  await page.goto('/autonomous')
  await page.getByLabel('你的学习目标').fill('更新我的学习画像')
  await page.getByRole('button', { name: '开始执行' }).click()

  await expect(page.getByRole('alert')).toContainText('此前请求可能已执行写操作')
  await expect(page.getByRole('button', { name: '再次执行当前目标' })).toHaveCount(0)
  await expect(page.getByRole('button', { name: '清空并重新开始' })).toBeFocused()
  expect(unexpectedRequests).toEqual([])
})

test('Autonomous generates a request key without crypto.randomUUID', async ({ page }) => {
  const problems = trackBrowserProblems(page)
  let requestKey
  await page.addInitScript(() => {
    Object.defineProperty(globalThis.crypto, 'randomUUID', {
      configurable: true,
      value: undefined,
    })
  })
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'POST' && path === '/agent/autonomous') {
      requestKey = await request.headerValue('idempotency-key')
      return {
        body: {
          awaiting_user_input: false,
          final_answer: '兼容模式提交成功。',
          finalize_reason: '测试完成',
          rounds_used: 1,
          steps: [],
          tools_called: ['finalize'],
          truncated: false,
        },
      }
    }
    return null
  })

  await page.goto('/autonomous')
  await page.getByLabel('你的学习目标').fill('测试非安全上下文')
  await page.getByRole('button', { name: '开始执行' }).click()

  await expect(page.getByText('兼容模式提交成功。')).toBeVisible()
  expect(requestKey).toMatch(UUID_V4_PATTERN)
  expect(unexpectedRequests).toEqual([])
  expect(problems).toEqual([])
})

test('Autonomous replays a lost start response after reload with the exact body and key', async ({ page }) => {
  const attempts = []
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'POST' && path === '/agent/autonomous') {
      attempts.push({
        body: await request.postDataJSON(),
        key: await request.headerValue('idempotency-key'),
      })
      if (attempts.length === 1) return { abort: 'failed' }
      return {
        body: {
          awaiting_user_input: false,
          final_answer: '刷新后已从原请求回执恢复。',
          finalize_reason: 'receipt_replayed',
          rounds_used: 1,
          steps: [],
          tools_called: ['finalize'],
          truncated: false,
        },
      }
    }
    return null
  })

  await page.goto('/autonomous')
  await page.getByLabel('文档 ID（可选）').fill('start-reload.md')
  await page.getByLabel('你的学习目标').fill('刷新也只能执行一次')
  await page.getByRole('button', { name: '开始执行' }).click()

  await expect(page.getByRole('alert')).toContainText('Failed to fetch')
  const pending = await page.evaluate(
    key => JSON.parse(sessionStorage.getItem(key)),
    AUTONOMOUS_SESSION_KEY,
  )
  expect(pending).toMatchObject({
    schema_version: 2,
    kind: 'pending_start',
    request: {
      query: '刷新也只能执行一次',
      user_id: 'default_user',
      document_id: 'start-reload.md',
      grounding_required: true,
    },
  })
  expect(pending.idempotency_key).toMatch(UUID_V4_PATTERN)
  await expect(page.getByLabel('你的学习目标')).toBeDisabled()

  await page.reload()
  await expect(page.getByText('刷新后已从原请求回执恢复。')).toBeVisible()
  expect(attempts).toHaveLength(2)
  expect(attempts[1]).toEqual(attempts[0])
  expect(await page.evaluate(
    key => sessionStorage.getItem(key),
    AUTONOMOUS_SESSION_KEY,
  )).toBeNull()
  expect(unexpectedRequests).toEqual([])
})

test('Autonomous keeps an in-progress continuation locked and reloads into the next pause with the same key', async ({ page }) => {
  const continueAttempts = []
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'POST' && path === '/agent/autonomous') {
      return {
        body: {
          awaiting_user_input: true,
          conversation_id: 'continue-in-progress-old',
          user_question: '第一轮确认内容？',
          rounds_used: 1,
          steps: [],
          tools_called: ['ask_user'],
          truncated: false,
        },
      }
    }
    if (request.method() === 'POST' && path === '/agent/autonomous/continue') {
      continueAttempts.push({
        body: await request.postDataJSON(),
        key: await request.headerValue('idempotency-key'),
      })
      if (continueAttempts.length === 1) {
        return {
          status: 409,
          body: {
            detail: '相同请求正在处理中，请勿并发重复提交',
            code: 'idempotency_conflict',
            reason: 'in_progress',
          },
        }
      }
      return {
        body: {
          awaiting_user_input: true,
          conversation_id: 'continue-in-progress-next',
          user_question: '第二轮仍需确认什么？',
          rounds_used: 2,
          steps: [],
          tools_called: ['ask_user'],
          truncated: false,
        },
      }
    }
    return null
  })

  await page.goto('/autonomous')
  await page.getByLabel('你的学习目标').fill('测试续跑中的恢复锁')
  await page.getByRole('button', { name: '开始执行' }).click()
  const dialog = page.getByRole('dialog', { name: 'Agent 想问你' })
  const reply = dialog.getByRole('textbox', { name: '你的回答' })
  await reply.fill('沿用这个不可变回答')
  await dialog.getByRole('button', { name: /回答/ }).click()

  await expect(dialog.getByRole('alert')).toContainText('相同请求正在处理中')
  await expect(reply).toHaveJSProperty('readOnly', true)
  await expect(dialog.getByRole('button', { name: '取消整个执行' })).toBeDisabled()
  const pending = await page.evaluate(
    key => JSON.parse(sessionStorage.getItem(key)),
    AUTONOMOUS_SESSION_KEY,
  )
  expect(pending).toMatchObject({
    kind: 'pending_continue',
    body: {
      conversation_id: 'continue-in-progress-old',
      user_reply: '沿用这个不可变回答',
    },
  })

  await page.reload()
  await expect(dialog).toContainText('第二轮仍需确认什么？')
  await expect(reply).toHaveValue('')
  expect(continueAttempts).toHaveLength(2)
  expect(continueAttempts[1]).toEqual(continueAttempts[0])
  const nextPause = await page.evaluate(
    key => JSON.parse(sessionStorage.getItem(key)),
    AUTONOMOUS_SESSION_KEY,
  )
  expect(nextPause).toMatchObject({
    kind: 'awaiting',
    conversation_id: 'continue-in-progress-next',
    user_question: '第二轮仍需确认什么？',
    draft: '',
  })
  expect(nextPause).not.toHaveProperty('idempotency_key')
  expect(unexpectedRequests).toEqual([])
})

test('Autonomous ignores a late response after navigation and lets the mounted page reconcile the durable start', async ({ page }) => {
  let releaseFirst
  let markFirstStarted
  const firstGate = new Promise(resolve => { releaseFirst = resolve })
  const firstStarted = new Promise(resolve => { markFirstStarted = resolve })
  const attempts = []
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'POST' && path === '/agent/autonomous') {
      attempts.push({
        body: await request.postDataJSON(),
        key: await request.headerValue('idempotency-key'),
      })
      if (attempts.length === 1) {
        markFirstStarted()
        await firstGate
        return {
          body: {
            awaiting_user_input: true,
            conversation_id: 'late-unmounted-pause',
            user_question: '这个迟到问题不应覆盖恢复指针？',
            rounds_used: 1,
            steps: [],
            tools_called: ['ask_user'],
            truncated: false,
          },
        }
      }
      return {
        body: {
          awaiting_user_input: false,
          final_answer: '已由当前页面完成权威对账。',
          finalize_reason: 'current_mount_replay',
          rounds_used: 1,
          steps: [],
          tools_called: ['finalize'],
          truncated: false,
        },
      }
    }
    return null
  })

  await page.goto('/autonomous')
  await page.getByLabel('你的学习目标').fill('验证卸载后的晚响应隔离')
  await page.getByRole('button', { name: '开始执行' }).click()
  await firstStarted
  await page.getByRole('link', { name: '文档管理' }).click()
  await expect(page).toHaveURL(/\/documents$/)
  await expect(page.getByRole('heading', { name: '文档管理' })).toBeVisible()
  releaseFirst()
  await page.waitForTimeout(100)

  const stillPending = await page.evaluate(
    key => JSON.parse(sessionStorage.getItem(key)),
    AUTONOMOUS_SESSION_KEY,
  )
  expect(stillPending.kind).toBe('pending_start')
  expect(stillPending.idempotency_key).toBe(attempts[0].key)

  await page.getByRole('link', { name: '自主 Agent' }).click()
  await expect(page.getByText('已由当前页面完成权威对账。')).toBeVisible()
  expect(attempts).toHaveLength(2)
  expect(attempts[1]).toEqual(attempts[0])
  await expect(page.getByText('这个迟到问题不应覆盖恢复指针？')).toHaveCount(0)
  expect(unexpectedRequests).toEqual([])
})

test('Autonomous retains a paused recovery when cancel fails and clears it only after confirmed missing', async ({ page }) => {
  let cancelAttempts = 0
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'POST' && path === '/agent/autonomous') {
      return {
        body: {
          awaiting_user_input: true,
          conversation_id: 'cancel-retry-session',
          user_question: '是否真的放弃这次执行？',
          rounds_used: 1,
          steps: [],
          tools_called: ['ask_user'],
          truncated: false,
        },
      }
    }
    if (
      request.method() === 'DELETE'
      && path === '/agent/autonomous/cancel-retry-session'
    ) {
      cancelAttempts += 1
      return cancelAttempts === 1
        ? { status: 503, body: { detail: '取消服务暂时不可用' } }
        : { body: { status: 'missing' } }
    }
    return null
  })

  await page.goto('/autonomous')
  await page.getByLabel('你的学习目标').fill('测试可靠取消')
  await page.getByRole('button', { name: '开始执行' }).click()
  const dialog = page.getByRole('dialog', { name: 'Agent 想问你' })
  const cancel = dialog.getByRole('button', { name: '取消整个执行' })
  await cancel.click()

  await expect(dialog.getByRole('alert')).toContainText('取消服务暂时不可用')
  await expect(dialog).toBeVisible()
  const retained = await page.evaluate(
    key => JSON.parse(sessionStorage.getItem(key)),
    AUTONOMOUS_SESSION_KEY,
  )
  expect(retained).toMatchObject({
    kind: 'awaiting',
    conversation_id: 'cancel-retry-session',
  })

  await cancel.click()
  await expect(dialog).toBeHidden()
  await expect(page.getByLabel('你的学习目标')).toHaveValue('')
  expect(cancelAttempts).toBe(2)
  expect(await page.evaluate(
    key => sessionStorage.getItem(key),
    AUTONOMOUS_SESSION_KEY,
  )).toBeNull()
  expect(unexpectedRequests).toEqual([])
})

test('short answers stay pending until one canonical AI grading completes', async ({ page }) => {
  const problems = trackBrowserProblems(page)
  let releaseGrade
  let markGradeStarted
  const gradeGate = new Promise(resolve => { releaseGrade = resolve })
  const gradeStarted = new Promise(resolve => { markGradeStarted = resolve })
  let gradeRequests = 0
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md'] } }
    }
    if (request.method() === 'POST' && path === '/session/start') {
      return {
        body: {
          session_id: 'short-canonical',
          total: 1,
          questions: [{
            index: 0,
            question: '请解释 RAG 的基本流程。',
            options: null,
            type: 'short_answer',
          }],
        },
      }
    }
    if (request.method() === 'POST' && path === '/session/short-canonical/answer') {
      return {
        body: {
          evaluation_status: 'pending_ai',
          correct: null,
          correct_answer: null,
          explanation: null,
          is_last: true,
          next_index: null,
        },
      }
    }
    if (request.method() === 'GET' && path === '/session/short-canonical/result') {
      return {
        body: {
          session_id: 'short-canonical',
          document_id: 'notes.md',
          total: 1,
          correct: 0,
          incorrect: 0,
          pending: 1,
          score: null,
          details: [],
        },
      }
    }
    if (request.method() === 'POST' && path === '/session/short-canonical/grade') {
      gradeRequests += 1
      markGradeStarted()
      await gradeGate
      return {
        body: {
          session_id: 'short-canonical',
          total: 1,
          correct: 1,
          score: 1,
          grades: [{
            index: 0,
            question: '请解释 RAG 的基本流程。',
            user_answer: '先检索证据，再依据证据生成答案。',
            correct_answer: '检索增强生成。',
            is_correct: true,
            ai_feedback: '语义正确，说明了检索和生成两个阶段。',
            knowledge_gap: null,
          }],
        },
      }
    }
    return null
  })

  await page.goto('/quiz')
  await page.getByRole('combobox', { name: '学习文档' }).selectOption('notes.md')
  await page.getByRole('button', { name: '简答题' }).click()
  await page.getByRole('button', { name: '开始答题' }).click()
  await page.getByRole('textbox', { name: '请解释 RAG 的基本流程。' }).fill('先检索证据，再依据证据生成答案。')
  await page.getByRole('button', { name: '提交答案' }).click()

  await expect(page.getByText('答案已记录')).toBeVisible()
  await expect(page.getByText('✗ 错误')).toHaveCount(0)
  await expect(page.getByText('正确答案：')).toHaveCount(0)

  await page.getByRole('button', { name: '开始 AI 批改' }).click()
  await gradeStarted
  await expect(page.getByText('待评')).toBeVisible()
  await expect(page.getByText('AI 待批改')).toBeVisible()

  releaseGrade()
  await expect(page.getByRole('heading', { name: 'AI 批改报告' })).toBeVisible()
  await expect(page.getByText('100 分')).toBeVisible()
  await expect(page.getByText('语义正确，说明了检索和生成两个阶段。')).toBeVisible()
  expect(gradeRequests).toBe(1)
  expect(unexpectedRequests).toEqual([])
  expect(problems).toEqual([])
})

test('a late grading response cannot revive a restarted quiz', async ({ page }) => {
  const problems = trackBrowserProblems(page)
  let starts = 0
  let releaseGrade
  let markGradeStarted
  const gradeGate = new Promise(resolve => { releaseGrade = resolve })
  const gradeStarted = new Promise(resolve => { markGradeStarted = resolve })
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md'] } }
    }
    if (request.method() === 'POST' && path === '/session/start') {
      starts += 1
      return {
        body: {
          session_id: `late-grade-${starts}`,
          total: 1,
          questions: [{
            index: 0,
            question: starts === 1 ? '旧会话题目' : '新会话题目',
            options: ['A. 正确', 'B. 错误'],
            type: 'choice',
          }],
        },
      }
    }
    if (request.method() === 'POST' && path === '/session/late-grade-1/answer') {
      return {
        body: {
          evaluation_status: 'final',
          correct: true,
          correct_answer: 'A. 正确',
          explanation: '回答正确。',
          is_last: true,
          next_index: null,
        },
      }
    }
    if (request.method() === 'GET' && path === '/session/late-grade-1/result') {
      return {
        body: {
          session_id: 'late-grade-1',
          document_id: 'notes.md',
          total: 1,
          correct: 1,
          incorrect: 0,
          pending: 0,
          score: 1,
          details: [],
        },
      }
    }
    if (request.method() === 'POST' && path === '/session/late-grade-1/grade') {
      markGradeStarted()
      await gradeGate
      return {
        body: {
          session_id: 'late-grade-1',
          total: 1,
          correct: 1,
          score: 1,
          grades: [{
            index: 0,
            question: '旧会话题目',
            user_answer: 'A. 正确',
            correct_answer: 'A. 正确',
            is_correct: true,
            ai_feedback: null,
            knowledge_gap: null,
          }],
        },
      }
    }
    return null
  })

  try {
    await page.goto('/quiz')
    await page.getByRole('combobox', { name: '学习文档' }).selectOption('notes.md')
    await page.getByRole('button', { name: '开始答题' }).click()
    await page.getByRole('button', { name: 'A. 正确' }).click()
    await page.getByRole('button', { name: '提交答案' }).click()
    await page.getByRole('button', { name: '查看结果' }).click()
    await page.getByRole('button', { name: 'AI 批改讲解' }).click()
    await gradeStarted

    const oldGradeAborted = waitForAbortedApiRequest(
      page,
      'POST',
      '/session/late-grade-1/grade',
    )
    await page.getByRole('button', { name: '再来一轮' }).click()
    expectBrowserAbort(await oldGradeAborted)
    releaseGrade()

    await page.getByRole('button', { name: '开始答题' }).click()
    await expect(page.getByText('新会话题目')).toBeVisible()
    await expect(page.getByRole('heading', { name: 'AI 批改报告' })).toHaveCount(0)
    expect(starts).toBe(2)
    expect(unexpectedRequests).toEqual([])
    expect(problems).toEqual([])
  } finally {
    releaseGrade()
  }
})

test('a late learning-report response cannot replace a new quiz', async ({ page }) => {
  const problems = trackBrowserProblems(page)
  let starts = 0
  let releaseReport
  let markReportStarted
  const reportGate = new Promise(resolve => { releaseReport = resolve })
  const reportStarted = new Promise(resolve => { markReportStarted = resolve })
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md'] } }
    }
    if (request.method() === 'POST' && path === '/session/start') {
      starts += 1
      return {
        body: {
          session_id: `late-report-${starts}`,
          total: 1,
          questions: [{
            index: 0,
            question: starts === 1 ? '报告旧题目' : '报告后的新题目',
            options: ['A. 正确', 'B. 错误'],
            type: 'choice',
          }],
        },
      }
    }
    if (request.method() === 'POST' && path === '/session/late-report-1/answer') {
      return {
        body: {
          evaluation_status: 'final',
          correct: true,
          correct_answer: 'A. 正确',
          explanation: '回答正确。',
          is_last: true,
          next_index: null,
        },
      }
    }
    if (request.method() === 'GET' && path === '/session/late-report-1/result') {
      return {
        body: {
          session_id: 'late-report-1',
          document_id: 'notes.md',
          total: 1,
          correct: 1,
          incorrect: 0,
          pending: 0,
          score: 1,
          details: [],
        },
      }
    }
    if (request.method() === 'POST' && path === '/session/late-report-1/grade') {
      return {
        body: {
          session_id: 'late-report-1',
          total: 1,
          correct: 1,
          score: 1,
          grades: [{
            index: 0,
            question: '报告旧题目',
            user_answer: 'A. 正确',
            correct_answer: 'A. 正确',
            is_correct: true,
            ai_feedback: null,
            knowledge_gap: null,
          }],
        },
      }
    }
    if (request.method() === 'POST' && path === '/session/late-report-1/report') {
      markReportStarted()
      await reportGate
      return {
        body: {
          session_id: 'late-report-1',
          document_id: 'notes.md',
          overall_score: 1,
          topic_mastery: [],
          strengths: ['旧会话'],
          weaknesses: [],
          recommendations: ['旧建议'],
          summary: '不应复活的旧报告。',
        },
      }
    }
    return null
  })

  try {
    await page.goto('/quiz')
    await page.getByRole('combobox', { name: '学习文档' }).selectOption('notes.md')
    await page.getByRole('button', { name: '开始答题' }).click()
    await page.getByRole('button', { name: 'A. 正确' }).click()
    await page.getByRole('button', { name: '提交答案' }).click()
    await page.getByRole('button', { name: '查看结果' }).click()
    await page.getByRole('button', { name: 'AI 批改讲解' }).click()
    await page.getByRole('button', { name: '学习评估报告' }).click()
    await reportStarted

    const oldReportAborted = waitForAbortedApiRequest(
      page,
      'POST',
      '/session/late-report-1/report',
    )
    await page.getByRole('button', { name: '再来一轮' }).click()
    expectBrowserAbort(await oldReportAborted)
    releaseReport()

    await page.getByRole('button', { name: '开始答题' }).click()
    await expect(page.getByText('报告后的新题目')).toBeVisible()
    await expect(page.getByText('不应复活的旧报告。')).toHaveCount(0)
    expect(starts).toBe(2)
    expect(unexpectedRequests).toEqual([])
    expect(problems).toEqual([])
  } finally {
    releaseReport()
  }
})

test('Dashboard requires an explicit choice before replacing Quiz recovery', async ({ page }) => {
  const problems = trackBrowserProblems(page)
  const oldRecovery = {
    schema_version: 1,
    intent: {
      kind: 'standard',
      request: {
        document_id: 'old.md',
        description: '旧练习',
        count: 1,
        difficulty: 'medium',
        type: 'choice',
        user_id: 'default_user',
      },
    },
    launch_id: null,
    launch_preset: null,
    start_idempotency_key: 'old-start-key-1234',
    session: null,
    acknowledged_answer_count: 0,
    pending_answer: null,
  }
  await page.addInitScript(({ key, value }) => {
    sessionStorage.setItem(key, JSON.stringify(value))
  }, { key: QUIZ_RECOVERY_KEY, value: oldRecovery })

  let oldStarts = 0
  let practiceStarts = 0
  const wrongEntry = {
    entry_id: 'new.md:0',
    document_id: 'new.md',
    question: '只属于新材料的错题',
    options: ['错误', '正确'],
    question_type: 'choice',
    correct_answer: '正确',
    explanation: '新材料解析',
    user_answer: '错误',
    knowledge_gap: '新材料知识点',
    session_id: 'new-source-session',
  }
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['old.md', 'new.md'] } }
    }
    if (request.method() === 'GET' && path === '/user/default_user/sessions') {
      return { body: [] }
    }
    if (request.method() === 'GET' && path === '/user/default_user/profile') {
      return {
        body: {
          topic_mastery: { 'old.md': 0.5, 'new.md': 0.2 },
          weak_points: [],
          total_sessions: 1,
        },
      }
    }
    if (request.method() === 'GET' && path === '/wrong-questions/new.md') {
      return { body: { document_id: 'new.md', total: 1, entries: [wrongEntry] } }
    }
    if (request.method() === 'POST' && path === '/session/start') {
      oldStarts += 1
      return {
        body: {
          session_id: 'old-session',
          total: 1,
          questions: [{
            index: 0,
            question: '应继续显示的旧练习',
            options: ['A', 'B'],
            type: 'choice',
          }],
          revision: 1,
          expires_at: FUTURE_EXPIRES_AT,
        },
      }
    }
    if (request.method() === 'POST' && path === '/wrong-questions/new.md/practice') {
      practiceStarts += 1
      return {
        body: {
          session_id: 'new-practice',
          total: 1,
          questions: [{
            index: 0,
            question: '只属于新材料的错题',
            options: ['错误', '正确'],
            type: 'choice',
          }],
          revision: 1,
          expires_at: FUTURE_EXPIRES_AT,
        },
      }
    }
    return null
  })

  await page.goto('/dashboard')
  await page.getByLabel('错题文档').selectOption('new.md')
  await page.getByRole('button', { name: /开始重练/ }).click()

  await expect(page).toHaveURL(/\/dashboard$/)
  await expect(page.getByRole('heading', { name: '检测到尚未结束的练习' })).toBeVisible()
  expect(practiceStarts).toBe(0)
  expect(
    await page.evaluate(key => JSON.parse(sessionStorage.getItem(key)), QUIZ_RECOVERY_KEY)
  ).toMatchObject({ start_idempotency_key: 'old-start-key-1234' })

  await page.getByRole('button', { name: '继续当前练习' }).click()
  await expect(page.getByText('应继续显示的旧练习')).toBeVisible()
  expect(oldStarts).toBe(1)
  expect(practiceStarts).toBe(0)

  await page.goto('/dashboard')
  await page.getByLabel('错题文档').selectOption('new.md')
  await page.getByRole('button', { name: /开始重练/ }).click()
  await page.getByRole('button', { name: '放弃并开始错题重练' }).click()

  await expect(page).toHaveURL(/\/quiz$/)
  // 这句题干在 Dashboard 的错题预览里也有（同一份 wrongEntry），只按文字找会在
  // 跳转瞬间命中尚未卸载的 Dashboard 节点，于是下面的计数断言跑在 practice
  // 请求之前。作答用的 radiogroup 只有 practice 响应渲染出题目后才存在。
  await expect(
    page.getByRole('group', { name: '只属于新材料的错题' })
  ).toBeVisible()
  expect(practiceStarts).toBe(1)
  expect(
    await page.evaluate(key => JSON.parse(sessionStorage.getItem(key)), QUIZ_RECOVERY_KEY)
  ).toMatchObject({
    intent: {
      kind: 'wrong_question',
      request: { document_id: 'new.md', user_id: 'default_user' },
    },
    session: { session_id: 'new-practice' },
  })
  expect(unexpectedRequests).toEqual([])
  expect(problems).toEqual([])
})

test('Dashboard starts a persisted wrong-question practice in Quiz', async ({ page }) => {
  const problems = trackBrowserProblems(page)
  let practiceKey
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['retrieval.md'] } }
    }
    if (request.method() === 'GET' && path === '/user/default_user/sessions') {
      return { body: [] }
    }
    if (request.method() === 'GET' && path === '/user/default_user/profile') {
      return {
        body: {
          topic_mastery: { 'retrieval.md': 0.4 },
          weak_points: ['混合检索'],
          total_sessions: 1,
        },
      }
    }
    if (request.method() === 'GET' && path === '/wrong-questions/retrieval.md') {
      return {
        body: {
          document_id: 'retrieval.md',
          total: 2,
          entries: [
            {
              entry_id: 'source-session:0',
              document_id: 'retrieval.md',
              question: 'RRF 的作用是什么？',
              options: ['只做向量检索', '融合多路检索排名'],
              question_type: 'choice',
              correct_answer: '融合多路检索排名',
              explanation: 'RRF 会融合多个有序结果集。',
              user_answer: '只做向量检索',
              knowledge_gap: '混合检索',
              session_id: 'source-session',
            },
            {
              entry_id: 'source-session:1',
              document_id: 'retrieval.md',
              question: '为什么 RRF 不直接比较原始分数？',
              options: null,
              question_type: 'short_answer',
              correct_answer: '不同检索器的分数尺度不可直接比较',
              explanation: '名次比不同检索器的原始分数更容易统一。',
              user_answer: '因为计算更快',
              knowledge_gap: '分数校准',
              session_id: 'source-session',
            },
          ],
        },
      }
    }
    if (
      request.method() === 'POST'
      && path === '/wrong-questions/retrieval.md/practice'
    ) {
      practiceKey = await request.headerValue('idempotency-key')
      return {
        body: {
          session_id: 'practice-session',
          total: 2,
          questions: [
            {
              index: 0,
              question: 'RRF 的作用是什么？',
              options: ['只做向量检索', '融合多路检索排名'],
              type: 'choice',
            },
            {
              index: 1,
              question: '为什么 RRF 不直接比较原始分数？',
              options: null,
              type: 'short_answer',
            },
          ],
          revision: 1,
          expires_at: FUTURE_EXPIRES_AT,
        },
      }
    }
    if (
      request.method() === 'POST'
      && path === '/session/practice-session/answer'
    ) {
      return {
        body: {
          correct: true,
          correct_answer: '融合多路检索排名',
          explanation: 'RRF 会融合多个有序结果集。',
          is_last: false,
          next_index: 1,
          revision: 2,
          expires_at: FUTURE_EXPIRES_AT,
        },
      }
    }
    return null
  })

  await page.goto('/dashboard')
  await page.getByLabel('错题文档').selectOption('retrieval.md')
  await page.getByRole('button', { name: '开始重练（2）' }).click()

  await expect(page).toHaveURL(/\/quiz$/)
  // 同上：题干在 Dashboard 的错题预览里也有，按文字找会命中尚未卸载的
  // Dashboard 节点，practiceKey 那时还没被 practice 请求赋值。
  await expect(
    page.getByRole('group', { name: 'RRF 的作用是什么？' })
  ).toBeVisible()
  expect(practiceKey).toMatch(UUID_V4_PATTERN)
  expect(
    await page.evaluate(key => JSON.parse(sessionStorage.getItem(key)), QUIZ_RECOVERY_KEY)
  ).toMatchObject({
    intent: {
      kind: 'wrong_question',
      request: { document_id: 'retrieval.md', user_id: 'default_user' },
    },
    session: { session_id: 'practice-session', revision: 1 },
  })
  await page.getByRole('button', { name: '融合多路检索排名' }).click()
  await page.getByRole('button', { name: '提交答案' }).click()
  await page.getByRole('button', { name: '下一题' }).click()
  await expect(
    page.getByRole('textbox', { name: '为什么 RRF 不直接比较原始分数？' })
  ).toBeVisible()
  expect(unexpectedRequests).toEqual([])
  expect(problems).toEqual([])
})

test('Dashboard persists wrong-practice intent before navigation and reload reuses its key', async ({ page }) => {
  const problems = trackBrowserProblems(page)
  const practiceKeys = []
  const entry = {
    entry_id: 'a.md:0',
    document_id: 'a.md',
    question: 'a.md 的错题',
    options: ['错误', '正确'],
    question_type: 'choice',
    correct_answer: '正确',
    explanation: '解析',
    user_answer: '错误',
    knowledge_gap: '测试',
    session_id: 'a.md-session',
  }
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['a.md'] } }
    }
    if (request.method() === 'GET' && path === '/user/default_user/sessions') {
      return { body: [] }
    }
    if (request.method() === 'GET' && path === '/user/default_user/profile') {
      return {
        body: {
          topic_mastery: { 'a.md': 0.3 },
          weak_points: [],
          total_sessions: 1,
        },
      }
    }
    if (request.method() === 'GET' && path === '/wrong-questions/a.md') {
      return { body: { document_id: 'a.md', total: 1, entries: [entry] } }
    }
    if (request.method() === 'POST' && path === '/wrong-questions/a.md/practice') {
      practiceKeys.push(await request.headerValue('idempotency-key'))
      if (practiceKeys.length === 1) {
        return { status: 503, body: { detail: '重练响应暂时不可用' } }
      }
      return {
        body: {
          session_id: 'recovered-a-practice',
          total: 1,
          questions: [{
            index: 0,
            question: 'a.md 的错题',
            options: ['错误', '正确'],
            type: 'choice',
          }],
          revision: 1,
          expires_at: FUTURE_EXPIRES_AT,
        },
      }
    }
    return null
  })

  await page.goto('/dashboard')
  await page.getByLabel('错题文档').selectOption('a.md')
  await page.getByRole('button', { name: '开始重练（1）' }).click()
  await expect(page).toHaveURL(/\/quiz$/)
  await expect(page.getByRole('alert')).toContainText('重练响应暂时不可用')
  const pending = await page.evaluate(key => JSON.parse(sessionStorage.getItem(key)), QUIZ_RECOVERY_KEY)
  expect(pending).toMatchObject({
    intent: {
      kind: 'wrong_question',
      request: { document_id: 'a.md', user_id: 'default_user' },
    },
    session: null,
  })

  await page.reload()
  await expect(page.getByText('a.md 的错题')).toBeVisible()
  expect(practiceKeys).toHaveLength(2)
  expect(practiceKeys[0]).toMatch(UUID_V4_PATTERN)
  expect(practiceKeys[1]).toBe(practiceKeys[0])
  expect(unexpectedRequests).toEqual([])
  expect(
    problems.filter(problem => !problem.includes('503 (Service Unavailable)')),
  ).toEqual([])
})

test('Dashboard reuses a pending practice key when the same intent is started again', async ({ page }) => {
  // 上一条覆盖的是刷新 /quiz（走 Quiz.recoverQuiz 里的 start_idempotency_key）。
  // 这条走的是另一条路：不刷新，导航回 Dashboard 再次发起同一个意图。
  // Dashboard.handleRepractice 曾无条件生成新键，服务端按新键又建一个持久
  // 会话，旧的被孤儿化却仍占容量，答完后学习历史还会多出一条记录。
  const problems = trackBrowserProblems(page)
  const practiceKeys = []
  const entry = {
    entry_id: 'a.md:0',
    document_id: 'a.md',
    question: 'a.md 的错题',
    options: ['错误', '正确'],
    question_type: 'choice',
    correct_answer: '正确',
    explanation: '解析',
    user_answer: '错误',
    knowledge_gap: '测试',
    session_id: 'a.md-session',
  }
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['a.md'] } }
    }
    if (request.method() === 'GET' && path === '/user/default_user/sessions') {
      return { body: [] }
    }
    if (request.method() === 'GET' && path === '/user/default_user/profile') {
      return {
        body: { topic_mastery: { 'a.md': 0.3 }, weak_points: [], total_sessions: 1 },
      }
    }
    if (request.method() === 'GET' && path === '/wrong-questions/a.md') {
      return { body: { document_id: 'a.md', total: 1, entries: [entry] } }
    }
    if (request.method() === 'POST' && path === '/wrong-questions/a.md/practice') {
      practiceKeys.push(await request.headerValue('idempotency-key'))
      if (practiceKeys.length === 1) {
        return { status: 503, body: { detail: '重练响应暂时不可用' } }
      }
      return {
        body: {
          session_id: 'reused-a-practice',
          total: 1,
          questions: [{
            index: 0,
            question: 'a.md 的错题',
            options: ['错误', '正确'],
            type: 'choice',
          }],
          revision: 1,
          expires_at: FUTURE_EXPIRES_AT,
        },
      }
    }
    return null
  })

  await page.goto('/dashboard')
  await page.getByLabel('错题文档').selectOption('a.md')
  await page.getByRole('button', { name: '开始重练（1）' }).click()
  await expect(page).toHaveURL(/\/quiz$/)
  await expect(page.getByRole('alert')).toContainText('重练响应暂时不可用')

  await page.getByRole('link', { name: '学习报告' }).click()
  await expect(page).toHaveURL(/\/dashboard$/)
  await page.getByLabel('错题文档').selectOption('a.md')
  await page.getByRole('button', { name: '开始重练（1）' }).click()

  // 同一个尚未建立会话的意图不是冲突，不该要求用户在两个同名记录之间选择。
  await expect(page.getByRole('button', { name: '放弃并开始错题重练' })).toHaveCount(0)
  await expect(page).toHaveURL(/\/quiz$/)
  await expect(page.getByRole('group', { name: 'a.md 的错题' })).toBeVisible()

  expect(practiceKeys).toHaveLength(2)
  expect(practiceKeys[0]).toMatch(UUID_V4_PATTERN)
  expect(practiceKeys[1]).toBe(practiceKeys[0])
  expect(unexpectedRequests).toEqual([])
  expect(
    problems.filter(problem => !problem.includes('503 (Service Unavailable)')),
  ).toEqual([])
})

test('a response from an unmounted Quiz cannot overwrite a newer Dashboard recovery', async ({ page }) => {
  const problems = trackBrowserProblems(page)
  let releaseAnswer
  let markAnswerStarted
  let releasePractice
  let markPracticeStarted
  const answerGate = new Promise(resolve => { releaseAnswer = resolve })
  const answerStarted = new Promise(resolve => { markAnswerStarted = resolve })
  const practiceGate = new Promise(resolve => { releasePractice = resolve })
  const practiceStarted = new Promise(resolve => { markPracticeStarted = resolve })
  const wrongEntry = {
    entry_id: 'b.md:0',
    document_id: 'b.md',
    question: '只属于 B 文档的错题',
    options: ['错误', '正确'],
    question_type: 'choice',
    correct_answer: '正确',
    explanation: 'B 文档解析',
    user_answer: '错误',
    knowledge_gap: 'B 文档知识点',
    session_id: 'b-source-session',
  }
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['a.md', 'b.md'] } }
    }
    if (request.method() === 'POST' && path === '/session/start') {
      return {
        body: {
          session_id: 'old-a-session',
          total: 1,
          questions: [{
            index: 0,
            question: 'A 文档中的旧题目',
            options: ['A. 正确', 'B. 错误'],
            type: 'choice',
          }],
          revision: 1,
          expires_at: FUTURE_EXPIRES_AT,
        },
      }
    }
    if (request.method() === 'POST' && path === '/session/old-a-session/answer') {
      markAnswerStarted()
      await answerGate
      return {
        body: {
          evaluation_status: 'final',
          correct: true,
          correct_answer: 'A. 正确',
          explanation: '旧请求已经完成。',
          is_last: true,
          next_index: null,
          revision: 2,
          expires_at: FUTURE_EXPIRES_AT,
        },
      }
    }
    if (request.method() === 'GET' && path === '/user/default_user/sessions') {
      return { body: [] }
    }
    if (request.method() === 'GET' && path === '/user/default_user/profile') {
      return {
        body: {
          topic_mastery: { 'b.md': 0.2 },
          weak_points: [],
          total_sessions: 1,
        },
      }
    }
    if (request.method() === 'GET' && path === '/wrong-questions/b.md') {
      return { body: { document_id: 'b.md', total: 1, entries: [wrongEntry] } }
    }
    if (request.method() === 'POST' && path === '/wrong-questions/b.md/practice') {
      markPracticeStarted()
      await practiceGate
      return {
        body: {
          session_id: 'new-b-practice',
          total: 1,
          questions: [{
            index: 0,
            question: '只属于 B 文档的错题',
            options: ['错误', '正确'],
            type: 'choice',
          }],
          revision: 1,
          expires_at: FUTURE_EXPIRES_AT,
        },
      }
    }
    return null
  })

  try {
    await page.goto('/quiz')
    await page.getByRole('combobox', { name: '学习文档' }).selectOption('a.md')
    await page.getByRole('button', { name: '开始答题' }).click()
    await page.getByRole('button', { name: 'A. 正确' }).click()
    await page.getByRole('button', { name: '提交答案' }).click()
    await answerStarted

    const oldAnswerAborted = waitForAbortedApiRequest(
      page,
      'POST',
      '/session/old-a-session/answer',
    )
    await page.getByRole('link', { name: '学习报告' }).click()
    expectBrowserAbort(await oldAnswerAborted)
    releaseAnswer()

    await page.getByLabel('错题文档').selectOption('b.md')
    await page.getByRole('button', { name: '开始重练（1）' }).click()
    await expect(page.getByRole('heading', { name: '检测到尚未结束的练习' })).toBeVisible()
    expect(
      await page.evaluate(key => JSON.parse(sessionStorage.getItem(key)), QUIZ_RECOVERY_KEY),
    ).toMatchObject({
      intent: {
        kind: 'standard',
        request: { document_id: 'a.md', user_id: 'default_user' },
      },
      session: { session_id: 'old-a-session' },
    })
    await page.getByRole('button', { name: '放弃并开始错题重练' }).click()
    await expect(page).toHaveURL(/\/quiz$/)
    await practiceStarted
    expect(
      await page.evaluate(key => JSON.parse(sessionStorage.getItem(key)), QUIZ_RECOVERY_KEY),
    ).toMatchObject({
      intent: {
        kind: 'wrong_question',
        request: { document_id: 'b.md', user_id: 'default_user' },
      },
      session: null,
    })

    releasePractice()
    await expect(page.getByText('只属于 B 文档的错题')).toBeVisible()
    expect(unexpectedRequests).toEqual([])
    expect(problems).toEqual([])
  } finally {
    releaseAnswer()
    releasePractice()
  }
})
