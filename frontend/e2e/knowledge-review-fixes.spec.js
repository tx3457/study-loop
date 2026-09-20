import { expect, test } from '@playwright/test'

const API_PREFIX = '/api'

function jobResult(id, status, errorCode = null) {
  return {
    id,
    knowledge_base_id: 'kb-review',
    status,
    stage: status === 'succeeded' ? 'complete' : 'index',
    revision: 7,
    error_code: errorCode,
  }
}

function knowledgeBase(overrides = {}) {
  return {
    id: 'kb-review',
    name: '评审知识库',
    description: '',
    status: 'ready',
    revision: 7,
    epoch: 2,
    document_count: 1,
    created_at: '2026-09-20T00:00:00Z',
    ...overrides,
  }
}

async function routeAutonomousWebImport(page, terminalStatus) {
  let jobCalls = 0
  await page.route(url => url.pathname.startsWith(`${API_PREFIX}/`), async route => {
    const request = route.request()
    const path = new URL(request.url()).pathname.slice(API_PREFIX.length)
    let status = 200
    let body

    if (request.method() === 'GET' && path === '/documents') {
      body = { documents: [] }
    } else if (request.method() === 'GET' && path === '/knowledge-bases/capabilities') {
      body = { enabled: true, available: true, web_search_available: true }
    } else if (request.method() === 'GET' && path === '/knowledge-bases') {
      body = { knowledge_bases: [knowledgeBase()] }
    } else if (request.method() === 'GET' && path === '/knowledge-bases/kb-review') {
      body = knowledgeBase({ revision: jobCalls >= 3 && terminalStatus === 'succeeded' ? 8 : 7 })
    } else if (request.method() === 'POST' && path === '/agent/autonomous') {
      body = {
        awaiting_user_input: false,
        final_answer: '联网资料已经找到。',
        rounds_used: 1,
        steps: [],
        tools_called: ['web_search'],
        truncated: false,
        knowledge_base_id: 'kb-review',
        web_enabled: true,
        web_only: false,
        source_citations: [{
          kind: 'web_snapshot',
          evidence_id: 'web-evidence',
          snapshot_id: 'snapshot-review',
          title: '网页资料',
          url: 'https://example.com/review',
          fetched_at: '2026-09-20T00:00:00Z',
          content_hash: 'review-hash',
          snippet: '用于回归测试的网页资料。',
        }],
        citations: [],
        invalid_citation_count: 0,
        grounding_status: 'citation_ids_valid',
        grounding_required: true,
      }
    } else if (request.method() === 'POST' && path === '/knowledge-bases/kb-review/web-import') {
      status = 202
      body = { job_id: 'job-web-review', knowledge_base_id: 'kb-review' }
    } else if (request.method() === 'GET' && path === '/knowledge-jobs/job-web-review') {
      jobCalls += 1
      const currentStatus = jobCalls === 1 ? 'queued' : jobCalls === 2 ? 'running' : terminalStatus
      body = jobResult('job-web-review', currentStatus, currentStatus === 'failed' ? 'web_import_failed' : null)
    } else {
      status = 500
      body = { detail: `Unexpected request: ${request.method()} ${path}` }
    }
    await route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) })
  })
  return () => jobCalls
}

for (const terminalStatus of ['succeeded', 'failed']) {
  test(`web import waits through queued and running before ${terminalStatus}`, async ({ page }) => {
    const getJobCalls = await routeAutonomousWebImport(page, terminalStatus)
    await page.goto('/autonomous?knowledge_base_id=kb-review')
    await page.getByLabel('联网补充').check()
    await page.getByLabel('你的学习目标').fill('查找网页资料')
    await page.getByRole('button', { name: '开始执行' }).click()
    await page.getByRole('button', { name: '收录到当前知识库' }).click()

    if (terminalStatus === 'succeeded') {
      await expect(page.getByText('网页快照已收录到当前知识库。')).toBeVisible()
    } else {
      await expect(page.getByText('收录失败：web_import_failed')).toBeVisible()
      await expect(page.getByText('网页快照已收录到当前知识库。')).toHaveCount(0)
    }
    expect(getJobCalls()).toBe(3)
  })
}

async function routeKnowledgeBaseDelete(page, terminalStatus) {
  let jobCalls = 0
  await page.route(url => url.pathname.startsWith(`${API_PREFIX}/`), async route => {
    const request = route.request()
    const url = new URL(request.url())
    const path = url.pathname.slice(API_PREFIX.length)
    let status = 200
    let body

    if (request.method() === 'GET' && path === '/knowledge-bases/capabilities') {
      body = { enabled: true, available: true, web_search_available: false }
    } else if (request.method() === 'GET' && path === '/knowledge-bases/kb-review') {
      body = knowledgeBase()
    } else if (request.method() === 'GET' && path === '/knowledge-bases/kb-review/documents') {
      body = { documents: [], total: 0, limit: 50, offset: 0 }
    } else if (request.method() === 'GET' && path === '/documents') {
      body = { documents: [] }
    } else if (request.method() === 'DELETE' && path === '/knowledge-bases/kb-review') {
      status = 202
      body = { job_id: 'job-delete-review', knowledge_base_id: 'kb-review' }
    } else if (request.method() === 'GET' && path === '/knowledge-jobs/job-delete-review') {
      jobCalls += 1
      const currentStatus = jobCalls === 1 ? 'queued' : jobCalls === 2 ? 'running' : terminalStatus
      body = jobResult('job-delete-review', currentStatus, currentStatus === 'failed' ? 'delete_failed' : null)
    } else if (request.method() === 'GET' && path === '/knowledge-bases') {
      body = { knowledge_bases: [] }
    } else {
      status = 500
      body = { detail: `Unexpected request: ${request.method()} ${path}${url.search}` }
    }
    await route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) })
  })
  return () => jobCalls
}

for (const terminalStatus of ['succeeded', 'failed']) {
  test(`knowledge-base deletion waits through queued and running before ${terminalStatus}`, async ({ page }) => {
    const getJobCalls = await routeKnowledgeBaseDelete(page, terminalStatus)
    await page.goto('/knowledge-bases/kb-review')
    await page.getByRole('button', { name: '删除知识库' }).click()
    await page.getByRole('dialog', { name: '删除知识库' }).getByRole('button', { name: '确认删除' }).click()

    if (terminalStatus === 'succeeded') {
      await expect(page).toHaveURL(/\/knowledge-bases$/)
    } else {
      await expect(page).toHaveURL(/\/knowledge-bases\/kb-review$/)
      await expect(page.getByRole('alert')).toContainText('delete_failed')
      await expect(page.getByRole('dialog', { name: '删除知识库' })).toBeVisible()
    }
    expect(getJobCalls()).toBe(3)
  })
}

test('documents after the first 100 remain browsable, replaceable and deletable on mobile', async ({ page }) => {
  const allDocuments = Array.from({ length: 101 }, (_, index) => ({
    id: `doc-${String(index + 1).padStart(3, '0')}`,
    knowledge_base_id: 'kb-review',
    name: `资料-${String(index + 1).padStart(3, '0')}.md`,
    kind: 'file',
    version_id: 'v1',
    status: 'ready',
    content_hash: `hash-${index + 1}`,
    created_at: `2026-09-${String(Math.min(index + 1, 30)).padStart(2, '0')}T00:00:00Z`,
  }))
  let revision = 7
  let replacementAccepted = false
  let deletionAccepted = false

  await page.setViewportSize({ width: 390, height: 844 })
  await page.route(url => url.pathname.startsWith(`${API_PREFIX}/`), async route => {
    const request = route.request()
    const url = new URL(request.url())
    const path = url.pathname.slice(API_PREFIX.length)
    let status = 200
    let body

    if (request.method() === 'GET' && path === '/knowledge-bases/capabilities') {
      body = { enabled: true, available: true, web_search_available: false }
    } else if (request.method() === 'GET' && path === '/knowledge-bases/kb-review') {
      body = knowledgeBase({ revision, document_count: allDocuments.length })
    } else if (request.method() === 'GET' && path === '/knowledge-bases/kb-review/documents') {
      const limit = Number(url.searchParams.get('limit'))
      const offset = Number(url.searchParams.get('offset'))
      body = { documents: allDocuments.slice(offset, offset + limit), total: allDocuments.length, limit, offset }
    } else if (request.method() === 'GET' && path === '/documents') {
      body = { documents: [] }
    } else if (request.method() === 'PUT' && path === '/knowledge-bases/kb-review/documents/doc-101') {
      replacementAccepted = true
      status = 202
      body = { job_id: 'job-replace-101', knowledge_base_id: 'kb-review' }
    } else if (request.method() === 'GET' && path === '/knowledge-jobs/job-replace-101') {
      if (replacementAccepted) {
        allDocuments[100] = { ...allDocuments[100], name: '资料-101-新版.md', version_id: 'v2' }
        replacementAccepted = false
        revision += 1
      }
      body = jobResult('job-replace-101', 'succeeded')
    } else if (request.method() === 'DELETE' && path === '/knowledge-bases/kb-review/documents/doc-101') {
      deletionAccepted = true
      status = 202
      body = { job_id: 'job-delete-101', knowledge_base_id: 'kb-review' }
    } else if (request.method() === 'GET' && path === '/knowledge-jobs/job-delete-101') {
      if (deletionAccepted) {
        allDocuments.pop()
        deletionAccepted = false
        revision += 1
      }
      body = jobResult('job-delete-101', 'succeeded')
    } else {
      status = 500
      body = { detail: `Unexpected request: ${request.method()} ${path}${url.search}` }
    }
    await route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) })
  })

  await page.goto('/knowledge-bases/kb-review')
  await expect(page.getByText('共 101 份资料')).toBeVisible()
  await page.getByRole('button', { name: '下一页' }).click()
  await page.getByRole('button', { name: '下一页' }).click()
  await expect(page.getByText('资料-101.md', { exact: true })).toBeVisible()

  await page.getByLabel('替换资料 资料-101.md').setInputFiles({
    name: '资料-101-新版.md',
    mimeType: 'text/markdown',
    buffer: Buffer.from('# 101 新版'),
  })
  await expect(page.getByText('资料-101-新版.md', { exact: true })).toBeVisible()

  await page.getByRole('button', { name: '删除资料' }).click()
  await page.getByRole('dialog', { name: '删除知识库资料' }).getByRole('button', { name: '确认删除' }).click()
  await expect(page.getByText('共 100 份资料')).toBeVisible()
  await expect(page.getByText('资料-051.md', { exact: true })).toBeVisible()
  await expect(page.getByText('第 2 / 2 页')).toBeVisible()
})

test('a delayed document page from the previous knowledge base cannot replace the current one', async ({ page }) => {
  let releaseOldDocuments
  const oldDocumentsReleased = new Promise(resolve => { releaseOldDocuments = resolve })

  await page.route(url => url.pathname.startsWith(`${API_PREFIX}/`), async route => {
    const request = route.request()
    const path = new URL(request.url()).pathname.slice(API_PREFIX.length)
    let status = 200
    let body
    if (path === '/knowledge-bases/capabilities') {
      body = { enabled: true, available: true, web_search_available: false }
    } else if (request.method() === 'GET' && path === '/knowledge-bases/kb-old') {
      body = knowledgeBase({ id: 'kb-old', name: '旧知识库' })
    } else if (request.method() === 'GET' && path === '/knowledge-bases/kb-new') {
      body = knowledgeBase({ id: 'kb-new', name: '新知识库' })
    } else if (request.method() === 'GET' && path === '/knowledge-bases/kb-old/documents') {
      await oldDocumentsReleased
      body = {
        documents: [{ id: 'doc-old', name: '旧资料.md', kind: 'file', status: 'ready' }],
        total: 1,
        limit: 50,
        offset: 0,
      }
    } else if (request.method() === 'GET' && path === '/knowledge-bases/kb-new/documents') {
      body = {
        documents: [{ id: 'doc-new', name: '新资料.md', kind: 'file', status: 'ready' }],
        total: 1,
        limit: 50,
        offset: 0,
      }
    } else if (request.method() === 'GET' && path === '/documents') {
      body = { documents: [] }
    } else {
      status = 500
      body = { detail: `Unexpected request: ${request.method()} ${path}` }
    }
    await route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) })
  })

  const oldDocumentsRequest = page.waitForRequest(request => (
    request.method() === 'GET'
    && new URL(request.url()).pathname === `${API_PREFIX}/knowledge-bases/kb-old/documents`
  ))
  await page.goto('/knowledge-bases/kb-old')
  await oldDocumentsRequest
  await page.evaluate(() => {
    history.pushState({}, '', '/knowledge-bases/kb-new')
    dispatchEvent(new PopStateEvent('popstate'))
  })
  await expect(page.getByText('新资料.md', { exact: true })).toBeVisible()
  releaseOldDocuments()
  await expect(page.getByText('旧资料.md', { exact: true })).toHaveCount(0)
  await expect(page.getByRole('heading', { name: '新知识库' })).toBeVisible()
})
