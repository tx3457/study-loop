import { expect, test } from '@playwright/test'

const API_PREFIX = '/api'

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

  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'POST' && path === '/agent/autonomous') {
      startRequest = await request.postDataJSON()
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
  expect(unexpectedRequests).toEqual([])
  expect(problems).toEqual([])
})

test('Autonomous preserves the initial goal and focus when starting fails', async ({ page }) => {
  const startRequests = []
  const unexpectedRequests = await mockApi(page, async ({ path, request }) => {
    if (request.method() === 'POST' && path === '/agent/autonomous') {
      startRequests.push(await request.postDataJSON())
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
  expect(unexpectedRequests).toEqual([])
})

test('Autonomous preserves a HITL reply and retries after a continuation failure', async ({ page }) => {
  const continueRequests = []
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

  const dialog = page.getByRole('dialog', { name: 'Agent 想问你' })
  await dialog.getByRole('textbox', { name: '你的回答' }).fill('继续')
  await dialog.getByRole('button', { name: /回答/ }).click()

  await expect(dialog).toBeHidden()
  await expect(page.getByRole('alert')).toContainText(
    '续跑已执行部分操作，无法安全重试；请重新开始'
  )
  await expect(page.getByRole('button', { name: '重试回答' })).toHaveCount(0)
  await expect(page.getByLabel('你的学习目标')).toHaveValue('更新我的学习建议')
  await expect(page.getByRole('button', { name: '再次执行当前目标' })).toBeFocused()
  expect(continueRequests).toBe(1)
  expect(unexpectedRequests).toEqual([])
})
