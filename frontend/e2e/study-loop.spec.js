import { expect, test } from '@playwright/test'

const API_PREFIX = '/api'
const UUID_V4_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i
const AUTONOMOUS_SESSION_KEY = 'study-loop.autonomous.awaiting.v1'

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
      const response = await handler({ path, request })

      if (!response) {
        unexpectedRequests.push(`${request.method()} ${path}`)
        await route.fulfill({
          status: 500,
          contentType: 'application/json',
          body: JSON.stringify({ detail: `Unexpected E2E API request: ${request.method()} ${path}` }),
        })
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

test('adaptive document refresh clears its own recovered load error', async ({ page }) => {
  let documentRequests = 0
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      documentRequests += 1
      return documentRequests === 1
        ? { status: 503, body: { detail: '文档列表刷新失败' } }
        : { body: { documents: ['notes.md'] } }
    }
    return null
  })

  await page.goto('/adaptive')
  const refresh = page.getByRole('button', { name: '刷新文档' })

  await refresh.click()
  await expect(page.getByRole('alert')).toContainText('文档列表刷新失败')
  await expect(page.getByLabel('学习目标')).toBeEditable()

  await refresh.click()
  await expect(page.getByRole('alert')).toHaveCount(0)
  await expect(refresh).toBeEnabled()
  expect(documentRequests).toBe(2)
  expect(unexpectedRequests).toEqual([])
})

test('adaptive retries a turn safely and rotates the key only when answers change', async ({ page }) => {
  const submitAttempts = []
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'POST' && path === '/agent/adaptive/start') {
      return {
        body: {
          adaptive_session_id: 'adapt-retry',
          turn: 1,
          done: false,
          turn_type: 'quiz',
          questions: [{ index: 0, question: '请选择首字母', options: ['Alpha', 'Beta'] }],
          trajectory: [],
        },
      }
    }
    if (request.method() === 'POST' && path === '/agent/adaptive/submit') {
      submitAttempts.push({
        body: await request.postDataJSON(),
        key: await request.headerValue('idempotency-key'),
      })
      if (submitAttempts.length === 1) {
        return { status: 503, body: { detail: '自适应批改暂时不可用' } }
      }
      if (submitAttempts.length === 2) {
        return {
          status: 409,
          body: {
            detail: '本轮仍在处理中',
            code: 'idempotency_conflict',
            reason: 'in_progress',
          },
        }
      }
      return {
        body: {
          adaptive_session_id: 'adapt-retry',
          turn: 1,
          done: true,
          summary: '已安全完成本轮',
          trajectory: [],
        },
      }
    }
    return null
  })

  await page.goto('/adaptive')
  await page.getByLabel('学习目标').fill('学习首字母')
  await page.getByLabel('文档 ID').fill('notes.md')
  await page.getByRole('button', { name: '开始自适应辅导' }).click()

  await page.getByRole('radio', { name: 'Alpha' }).check()
  await page.getByRole('button', { name: /提交本轮/ }).click()
  await expect(page.getByRole('alert')).toContainText('自适应批改暂时不可用')

  await page.getByRole('button', { name: /提交本轮/ }).click()
  await expect(page.getByRole('alert')).toContainText('本轮仍在处理中')

  await page.getByRole('radio', { name: 'Beta' }).check()
  await page.getByRole('button', { name: /提交本轮/ }).click()
  await expect(page.getByText('已安全完成本轮')).toBeVisible()

  expect(submitAttempts.map(attempt => attempt.body)).toEqual([
    { adaptive_session_id: 'adapt-retry', answers: ['Alpha'], turn: 1 },
    { adaptive_session_id: 'adapt-retry', answers: ['Alpha'], turn: 1 },
    { adaptive_session_id: 'adapt-retry', answers: ['Beta'], turn: 1 },
  ])
  expect(submitAttempts[0].key).toMatch(UUID_V4_PATTERN)
  expect(submitAttempts[1].key).toBe(submitAttempts[0].key)
  expect(submitAttempts[2].key).toMatch(UUID_V4_PATTERN)
  expect(submitAttempts[2].key).not.toBe(submitAttempts[0].key)
  expect(unexpectedRequests).toEqual([])
})

test('adaptive discards a session after a terminal submit conflict', async ({ page }) => {
  let submitRequests = 0
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'POST' && path === '/agent/adaptive/start') {
      return {
        body: {
          adaptive_session_id: 'adapt-terminal',
          turn: 2,
          done: false,
          turn_type: 'quiz',
          questions: [{ index: 0, question: '旧会话题目', options: ['继续', '停止'] }],
          trajectory: [],
        },
      }
    }
    if (request.method() === 'POST' && path === '/agent/adaptive/submit') {
      submitRequests += 1
      return {
        status: 409,
        body: {
          detail: '提交轮次已过期，请重新开始',
        },
      }
    }
    return null
  })

  await page.goto('/adaptive')
  await page.getByLabel('学习目标').fill('测试终端冲突')
  await page.getByLabel('文档 ID').fill('notes.md')
  await page.getByRole('button', { name: '开始自适应辅导' }).click()
  await page.getByRole('radio', { name: '继续' }).check()
  await page.getByRole('button', { name: /提交本轮/ }).click()

  await expect(page.getByRole('alert')).toContainText('提交轮次已过期')
  await expect(page.getByText('旧会话题目')).toHaveCount(0)
  await expect(page.getByLabel('学习目标')).toBeEditable()
  await expect(page.getByRole('button', { name: /提交本轮/ })).toHaveCount(0)
  expect(submitRequests).toBe(1)
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

test('quiz retries an answer safely and rotates the key only when the answer changes', async ({ page }) => {
  const answerAttempts = []
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
      if (answerAttempts.length === 2) {
        return {
          status: 409,
          body: {
            detail: '答案请求仍在处理中',
            code: 'idempotency_conflict',
            reason: 'in_progress',
          },
        }
      }
      return {
        body: {
          correct: false,
          correct_answer: 'Alpha',
          explanation: '已记录修改后的答案',
          is_last: true,
          next_index: null,
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

  await page.getByRole('button', { name: '提交答案' }).click()
  await expect(page.getByRole('alert')).toContainText('答案请求仍在处理中')

  await page.getByRole('button', { name: /Beta/ }).click()
  await page.getByRole('button', { name: '提交答案' }).click()
  await expect(page.getByText('已记录修改后的答案')).toBeVisible()

  expect(answerAttempts.map(attempt => attempt.body)).toEqual([
    { answer: 'Alpha', question_index: 0 },
    { answer: 'Alpha', question_index: 0 },
    { answer: 'Beta', question_index: 0 },
  ])
  expect(answerAttempts[0].key).toMatch(UUID_V4_PATTERN)
  expect(answerAttempts[1].key).toBe(answerAttempts[0].key)
  expect(answerAttempts[2].key).toMatch(UUID_V4_PATTERN)
  expect(answerAttempts[2].key).not.toBe(answerAttempts[0].key)
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

test('dashboard load failure is recoverable and not presented as missing learning data', async ({ page }) => {
  let failSessions = true
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { body: { documents: [] } }
    }
    if (request.method() === 'GET' && path === '/user/default_user/sessions') {
      return failSessions
        ? { status: 503, body: { detail: '学习记录暂时不可用' } }
        : { body: [] }
    }
    if (request.method() === 'GET' && path === '/user/default_user/profile') {
      return { body: null }
    }
    return null
  })

  await page.goto('/dashboard')

  await expect(page.getByRole('alert')).toContainText('无法加载学习报告')
  await expect(page.getByText('还没有学习数据')).toHaveCount(0)

  failSessions = false
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
    response => response.url().endsWith('/wrong-questions/a.md')
  )
  releaseFirstRequest()
  await firstResponse
  await page.waitForTimeout(100)

  await expect(documentSelect).toHaveValue('b.md')
  await expect(page.getByText('B 文档错题')).toBeVisible()
  await expect(page.getByText('A 文档错题')).toHaveCount(0)
  expect(unexpectedRequests).toEqual([])
})

test('Autonomous document refresh reports failures without breaking the form', async ({ page }) => {
  const unexpectedRequests = await mockApi(page, ({ path, request }) => {
    if (request.method() === 'GET' && path === '/documents') {
      return { status: 503, body: { detail: '文档服务暂时不可用' } }
    }
    return null
  })

  await page.goto('/autonomous')
  await page.getByRole('button', { name: '刷新文档' }).click()

  await expect(page.getByRole('alert')).toContainText(
    '文档列表加载失败：文档服务暂时不可用'
  )
  await expect(page.getByLabel('你的学习目标')).toBeEditable()
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
    },
  })
  expect(stored).not.toHaveProperty('steps')

  await page.reload()
  await expect(dialog).toBeVisible()
  await expect(dialog).toContainText('刷新后仍需回答的问题？')
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
    { query: '帮我复习向量检索', user_id: 'default_user', document_id: null },
    { query: '帮我复习向量检索', user_id: 'default_user', document_id: null },
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

test('Autonomous rotates a failed continuation key after editing the reply', async ({ page }) => {
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
        return { status: 503, body: { detail: '模型服务暂时不可用' } }
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
  await expect(dialog.getByRole('alert')).toContainText('模型服务暂时不可用')

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
