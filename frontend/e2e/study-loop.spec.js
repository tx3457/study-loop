import { expect, test } from '@playwright/test'

const API_PREFIX = '/api'
const UUID_V4_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i
const AUTONOMOUS_SESSION_KEY = 'study-loop.autonomous.awaiting.v1'
const QUIZ_RECOVERY_KEY = 'study-loop.quiz.recovery.v1'
const ADAPTIVE_RECOVERY_KEY = 'study-loop.adaptive.recovery.v1'
const FUTURE_EXPIRES_AT = 4_102_444_800

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
        body: JSON.stringify(response.body),
      })
    }
  )

  return unexpectedRequests
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

test('learning path selection clears results and errors from the previous document', async ({ page }) => {
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['a.md', 'b.md'] } }
    }
    if (request.method() === 'POST' && path === '/learning-path/a.md') {
      return {
        body: {
          document_id: 'a.md',
          title: 'A 文档专属路径',
          total_stages: 1,
          stages: [{
            stage: 1,
            title: '理解 A',
            topics: ['A'],
            description: '只属于 A 文档的内容',
            estimated_minutes: 10,
          }],
        },
      }
    }
    if (request.method() === 'POST' && path === '/learning-path/b.md') {
      return { status: 503, body: { detail: 'B 文档路径生成失败' } }
    }
    return null
  })

  await page.goto('/learning-path')
  const documentSelect = page.getByRole('combobox', { name: '学习文档' })

  await documentSelect.selectOption('a.md')
  await page.getByRole('button', { name: '生成学习路径' }).click()
  await expect(page.getByRole('heading', { name: 'A 文档专属路径' })).toBeVisible()

  await documentSelect.selectOption('b.md')
  await expect(page.getByRole('heading', { name: 'A 文档专属路径' })).toHaveCount(0)

  await page.getByRole('button', { name: '生成学习路径' }).click()
  await expect(page.getByRole('alert')).toContainText('B 文档路径生成失败')

  await documentSelect.selectOption('a.md')
  await expect(page.getByRole('alert')).toHaveCount(0)
  expect(unexpectedRequests).toEqual([])
})

test('learning path stage opens a refresh-safe quiz preset without auto-starting', async ({ page }) => {
  const problems = trackBrowserProblems(page)
  const startBodies = []
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md'] } }
    }
    if (request.method() === 'POST' && path === '/learning-path/notes.md') {
      return {
        body: {
          document_id: 'notes.md',
          title: '检索学习路径',
          total_stages: 1,
          stages: [{
            stage: 1,
            title: '混合检索',
            topics: ['BM25', 'RRF'],
            description: '理解稀疏检索与融合排序',
            estimated_minutes: 20,
          }],
        },
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
  await expect(page.getByRole('combobox', { name: '学习文档' })).toHaveValue('notes.md')
  await expect(page.getByRole('textbox', { name: /出题主题/ })).toHaveValue('BM25、RRF')
  expect(startBodies).toEqual([])

  await page.reload()

  quizUrl = new URL(page.url())
  expect(quizUrl.searchParams.get('document_id')).toBe('notes.md')
  expect(quizUrl.searchParams.get('topic')).toBe('BM25、RRF')
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
    launch_preset: { document_id: 'notes.md', topic: 'BM25、RRF' },
  })
  expect(startBodies).toEqual([{
    document_id: 'notes.md',
    description: 'BM25、RRF',
    count: 5,
    difficulty: 'medium',
    type: 'choice',
    user_id: 'default_user',
  }])

  await page.reload()
  await expect(page.getByRole('region', { name: '检测到另一项练习' })).toHaveCount(0)
  await expect(page.getByText('RRF 如何融合多路排序？')).toBeVisible()
  expect(startBodies).toHaveLength(1)
  expect(unexpectedRequests).toEqual([])
  expect(problems).toEqual([])
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

  await page.goto('/quiz')
  await page.getByRole('combobox', { name: '学习文档' }).selectOption('old.md')
  await page.getByRole('button', { name: '开始答题' }).click()
  await page.getByRole('button', { name: 'A' }).click()
  await page.getByRole('button', { name: '提交答案' }).click()
  await answerStarted

  const nextLaunch = '/quiz?document_id=new.md&topic=新目标&launch_id=late-answer-launch-1234'
  await page.evaluate(url => {
    history.pushState({}, '', url)
    dispatchEvent(new PopStateEvent('popstate'))
  }, nextLaunch)
  await expect(page.getByRole('region', { name: '检测到另一项练习' })).toBeVisible()
  await page.getByRole('button', { name: '放弃并开始新练习' }).click()
  await expect(page.getByRole('combobox', { name: '学习文档' })).toHaveValue('new.md')

  const oldResponse = page.waitForResponse(response => (
    response.url().includes('/api/session/quiz-old-in-flight/answer')
  ))
  releaseAnswer()
  await oldResponse
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
      await page.getByRole('link', { name: '从第一阶段开始练习' }).click()
      const quizUrl = new URL(page.url())
      expect(quizUrl.pathname).toBe('/quiz')
      expect(quizUrl.searchParams.get('document_id')).toBe('notes.md')
      expect(quizUrl.searchParams.get('topic')).toBe('快速排序')
      expect(quizUrl.searchParams.get('launch_id')).toMatch(UUID_V4_PATTERN)
    }
  }

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

  await page.getByRole('radio', { name: 'Beta' }).check()
  await page.getByRole('button', { name: /提交本轮/ }).click()
  await expect(page.getByText('修正后的答案已提交')).toBeVisible()

  expect(submits).toHaveLength(2)
  expect(submits[0].body.answers).toEqual(['Alpha'])
  expect(submits[1].body.answers).toEqual(['Beta'])
  expect(submits[1].key).not.toBe(submits[0].key)
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

  await page.goto('/adaptive')
  await page.getByLabel('学习目标').fill('旧目标')
  await page.getByLabel('文档 ID').fill('notes.md')
  await page.getByRole('button', { name: '开始自适应辅导' }).click()
  await oldStarted

  await page.getByRole('button', { name: '重新开始' }).click()
  await page.getByLabel('学习目标').fill('新目标')
  await page.getByLabel('文档 ID').fill('notes.md')
  await page.getByRole('button', { name: '开始自适应辅导' }).click()
  await expect(page.getByText('新的有效题目')).toBeVisible()

  const oldResponseDelivered = page.waitForResponse(response => {
    const request = response.request()
    return request.method() === 'POST'
      && new URL(request.url()).pathname.endsWith('/agent/adaptive/start')
      && request.postDataJSON()?.goal === '旧目标'
  })
  releaseOld()
  await oldResponseDelivered
  await expect(page.getByText('新的有效题目')).toBeVisible()
  await expect(page.getByText('不应出现的旧题目')).toHaveCount(0)
  const stored = await page.evaluate(key => JSON.parse(sessionStorage.getItem(key)), ADAPTIVE_RECOVERY_KEY)
  expect(stored.session.adaptive_session_id).toBe('adapt-new')
  expect(unexpectedRequests).toEqual([])
})

test('quiz restart clears transient errors and prior grading and learning reports', async ({ page }) => {
  let starts = 0
  let answerRequests = 0
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
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
      answerRequests += 1
      return answerRequests === 1
        ? { status: 503, body: { detail: '答案暂时无法提交' } }
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
  await expect(page.getByRole('alert')).toContainText('答案暂时无法提交')
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
      const answered = snapshotReads > 1
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

  await page.goto('/quiz')
  await page.getByRole('combobox', { name: '学习文档' }).selectOption('notes.md')
  await page.getByRole('button', { name: '开始答题' }).click()

  await page.getByRole('button', { name: /Alpha/ }).click()
  await page.getByRole('button', { name: '提交答案' }).click()
  await expect(page.getByRole('alert')).toContainText('答案提交暂时不可用')
  expect(await page.evaluate(key => JSON.parse(sessionStorage.getItem(key)), QUIZ_RECOVERY_KEY))
    .toMatchObject({
      session: { session_id: 'quiz-retry', revision: 1 },
      pending_answer: { question_index: 0, answer: 'Alpha' },
    })

  await page.reload()
  await expect(page.getByText('已安全恢复答案')).toBeVisible()

  expect(answerAttempts.map(attempt => attempt.body)).toEqual([
    { answer: 'Alpha', question_index: 0 },
    { answer: 'Alpha', question_index: 0 },
  ])
  expect(answerAttempts[0].key).toMatch(UUID_V4_PATTERN)
  expect(answerAttempts[1].key).toBe(answerAttempts[0].key)
  expect(snapshotReads).toBe(2)
  expect(
    await page.evaluate(key => JSON.parse(sessionStorage.getItem(key)), QUIZ_RECOVERY_KEY)
  ).toMatchObject({ pending_answer: null, acknowledged_answer_count: 0 })
  expect(unexpectedRequests).toEqual([])
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
  let releaseUpload
  let markUploadStarted
  const uploadGate = new Promise(resolve => { releaseUpload = resolve })
  const uploadStarted = new Promise(resolve => { markUploadStarted = resolve })
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: [] } }
    }
    if (request.method() === 'POST' && path === '/documents/upload') {
      uploadRequests += 1
      if (uploadRequests === 1) {
        markUploadStarted()
        await uploadGate
      }
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

test('dashboard keeps available data visible when one source fails', async ({ page }) => {
  let failSessions = true
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: ['notes.md'] } }
    }
    if (request.method() === 'GET' && path === '/user/default_user/sessions') {
      return failSessions
        ? { status: 503, body: { detail: '学习记录暂时不可用' } }
        : { body: [{ date: '2026-07-16', correct_rate: 0.75 }] }
    }
    if (request.method() === 'GET' && path === '/user/default_user/profile') {
      return {
        body: {
          topic_mastery: { 'notes.md': 0.8 },
          weak_points: ['检索'],
          total_sessions: 2,
        },
      }
    }
    return null
  })

  await page.goto('/dashboard')

  await expect(page.getByRole('alert')).toContainText('部分数据暂不可用')
  await expect(page.getByRole('alert')).toContainText('学习记录')
  await expect(page.locator('.stat-card').filter({ hasText: '学习次数' })).toContainText('2')
  await expect(page.getByText('还没有学习数据')).toHaveCount(0)

  failSessions = false
  await page.getByRole('button', { name: '重新加载' }).click()
  await expect(page.getByRole('alert')).toHaveCount(0)
  await expect(page.locator('.stat-card').filter({ hasText: '平均正确率' })).toContainText('75%')
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

  expect(box.width).toBeGreaterThanOrEqual(44)
  expect(box.height).toBeGreaterThanOrEqual(44)
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
  await page.getByLabel('用户 ID').fill('reload-user')
  await page.getByLabel('文档 ID（可选）').fill('reload.md')
  await page.getByRole('button', { name: '开始执行' }).click()

  const dialog = page.getByRole('dialog', { name: 'Agent 想问你' })
  await expect(dialog).toContainText('本轮已启用引用约束')
  await dialog.getByRole('textbox', { name: '你的回答' }).fill('保留这个回答草稿')
  const stored = await page.evaluate(key => JSON.parse(sessionStorage.getItem(key)), AUTONOMOUS_SESSION_KEY)
  expect(stored).toMatchObject({
    version: 1,
    conversation_id: 'playwright-hitl-reload',
    user_question: '刷新后仍需回答的问题？',
    draft: '保留这个回答草稿',
    request: {
      query: '保留这个学习目标',
      user_id: 'reload-user',
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
  await expect(dialog).toBeVisible()
  await expect(reply).toHaveValue('沿用这份回答')
  await dialog.getByRole('button', { name: /回答/ }).click()
  await expect(page.getByText('沿用原请求标识后完成。')).toBeVisible()

  expect(continueAttempts.map(attempt => attempt.body)).toEqual([
    { conversation_id: 'playwright-hitl-reload-retry', user_reply: '沿用这份回答' },
    { conversation_id: 'playwright-hitl-reload-retry', user_reply: '沿用这份回答' },
  ])
  expect(continueAttempts[0].key).toMatch(UUID_V4_PATTERN)
  expect(continueAttempts[1].key).toBe(continueAttempts[0].key)
  expect(unexpectedRequests).toEqual([])
})

test('Autonomous reset clears recovery state and corrupt storage fails closed', async ({ page }) => {
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
    return null
  })

  await page.goto('/autonomous')
  await page.getByLabel('你的学习目标').fill('等待取消')
  await page.getByRole('button', { name: '开始执行' }).click()
  expect(await page.evaluate(key => sessionStorage.getItem(key), AUTONOMOUS_SESSION_KEY)).not.toBeNull()

  await page.getByRole('dialog', { name: 'Agent 想问你' })
    .getByRole('button', { name: '取消整个执行' }).click()
  expect(await page.evaluate(key => sessionStorage.getItem(key), AUTONOMOUS_SESSION_KEY)).toBeNull()

  await page.evaluate(key => sessionStorage.setItem(key, '{not-json'), AUTONOMOUS_SESSION_KEY)
  await page.reload()
  await expect(page.getByRole('dialog', { name: 'Agent 想问你' })).toHaveCount(0)
  await expect(page.getByLabel('你的学习目标')).toHaveValue('')
  expect(await page.evaluate(key => sessionStorage.getItem(key), AUTONOMOUS_SESSION_KEY)).toBeNull()
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

test('Autonomous rotates a failed start key after edits or reset', async ({ page }) => {
  const attempts = []
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'POST' && path === '/agent/autonomous') {
      attempts.push({
        body: await request.postDataJSON(),
        key: await request.headerValue('idempotency-key'),
      })
      if (attempts.length % 2 === 1) {
        return { status: 503, body: { detail: '模型服务暂时不可用' } }
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
      name: 'user',
      mutate: async () => page.getByLabel('用户 ID').fill('edited-user'),
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
      await expect(page.getByRole('alert')).toContainText('模型服务暂时不可用')

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

  await page.goto('/quiz')
  await page.getByRole('combobox', { name: '学习文档' }).selectOption('notes.md')
  await page.getByRole('button', { name: '开始答题' }).click()
  await page.getByRole('button', { name: 'A. 正确' }).click()
  await page.getByRole('button', { name: '提交答案' }).click()
  await page.getByRole('button', { name: '查看结果' }).click()
  await page.getByRole('button', { name: 'AI 批改讲解' }).click()
  await gradeStarted

  await page.getByRole('button', { name: '再来一轮' }).click()
  await page.getByRole('button', { name: '开始答题' }).click()
  await expect(page.getByText('新会话题目')).toBeVisible()

  releaseGrade()
  await expect(page.getByText('新会话题目')).toBeVisible()
  await expect(page.getByRole('heading', { name: 'AI 批改报告' })).toHaveCount(0)
  expect(starts).toBe(2)
  expect(unexpectedRequests).toEqual([])
  expect(problems).toEqual([])
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

  await page.goto('/quiz')
  await page.getByRole('combobox', { name: '学习文档' }).selectOption('notes.md')
  await page.getByRole('button', { name: '开始答题' }).click()
  await page.getByRole('button', { name: 'A. 正确' }).click()
  await page.getByRole('button', { name: '提交答案' }).click()
  await page.getByRole('button', { name: '查看结果' }).click()
  await page.getByRole('button', { name: 'AI 批改讲解' }).click()
  await page.getByRole('button', { name: '学习评估报告' }).click()
  await reportStarted

  await page.getByRole('button', { name: '再来一轮' }).click()
  await page.getByRole('button', { name: '开始答题' }).click()
  await expect(page.getByText('报告后的新题目')).toBeVisible()

  releaseReport()
  await expect(page.getByText('报告后的新题目')).toBeVisible()
  await expect(page.getByText('不应复活的旧报告。')).toHaveCount(0)
  expect(starts).toBe(2)
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
  await expect(page.getByText('RRF 的作用是什么？')).toBeVisible()
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

  await page.goto('/quiz')
  await page.getByRole('combobox', { name: '学习文档' }).selectOption('a.md')
  await page.getByRole('button', { name: '开始答题' }).click()
  await page.getByRole('button', { name: 'A. 正确' }).click()
  await page.getByRole('button', { name: '提交答案' }).click()
  await answerStarted

  await page.getByRole('link', { name: '学习报告' }).click()
  await page.getByLabel('错题文档').selectOption('b.md')
  await page.getByRole('button', { name: '开始重练（1）' }).click()
  await expect(page).toHaveURL(/\/quiz$/)
  await practiceStarted

  const oldResponsePromise = page.waitForResponse(response => (
    response.request().method() === 'POST'
    && new URL(response.url()).pathname === `${API_PREFIX}/session/old-a-session/answer`
  ))
  releaseAnswer()
  const oldResponse = await oldResponsePromise
  await oldResponse.finished()
  await page.evaluate(() => new Promise(resolve => {
    requestAnimationFrame(() => requestAnimationFrame(resolve))
  }))
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
})
