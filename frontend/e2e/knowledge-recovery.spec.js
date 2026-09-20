import { expect, test } from '@playwright/test'

const API_PREFIX = '/api'
const RECOVERY_KEY = 'study-loop.autonomous.recovery.v3'
const UUID_V4_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i

test('a paused knowledge-base Autonomous run survives refresh and continues the same server session', async ({ page }) => {
  const starts = []
  const continuations = []
  const unexpected = []

  await page.setViewportSize({ width: 390, height: 844 })
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
      body = { knowledge_bases: [{ id: 'kb-paused', name: '医学知识库', description: '', status: 'ready', revision: 12, epoch: 3, document_count: 2, created_at: '2026-09-20T00:00:00Z' }] }
    } else if (request.method() === 'GET' && path === '/knowledge-bases/kb-paused') {
      body = { id: 'kb-paused', name: '医学知识库', description: '', status: 'ready', revision: 12, epoch: 3, document_count: 2, created_at: '2026-09-20T00:00:00Z' }
    } else if (request.method() === 'POST' && path === '/agent/autonomous') {
      starts.push({
        body: await request.postDataJSON(),
        key: await request.headerValue('idempotency-key'),
      })
      body = {
        awaiting_user_input: true,
        conversation_id: 'kb-server-schema4-session',
        user_question: '需要偏重诊断还是治疗？',
        rounds_used: 1,
        steps: [],
        tools_called: ['query_knowledge_base', 'ask_user'],
        truncated: false,
        knowledge_base_id: 'kb-paused',
        web_enabled: false,
        grounding_required: true,
        grounding_status: 'pending',
        source_citations: [],
        citations: [],
        invalid_citation_count: 0,
      }
    } else if (request.method() === 'POST' && path === '/agent/autonomous/continue') {
      continuations.push({
        body: await request.postDataJSON(),
        key: await request.headerValue('idempotency-key'),
      })
      body = {
        awaiting_user_input: false,
        final_answer: '已按诊断重点继续分析。',
        rounds_used: 2,
        steps: [],
        tools_called: ['query_knowledge_base'],
        truncated: false,
        knowledge_base_id: 'kb-paused',
        web_enabled: false,
        grounding_required: true,
        grounding_status: 'citation_ids_valid',
        source_citations: [{
          kind: 'kb_chunk',
          evidence_id: 'evidence-after-resume',
          knowledge_base_id: 'kb-paused',
          document_id: 'doc-med',
          source_version_id: 'version-med',
          chunk_id: 'chunk-med',
          title: '诊断笔记.md',
          snippet: '诊断需要结合病史与检查。',
          source_status: 'current',
        }],
        citations: [],
        invalid_citation_count: 0,
      }
    } else {
      unexpected.push(`${request.method()} ${path}`)
      status = 500
      body = { detail: `Unexpected request: ${request.method()} ${path}` }
    }
    await route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) })
  })

  await page.goto('/autonomous?knowledge_base_id=kb-paused')
  await expect(page.getByLabel('知识库（可选）')).toHaveValue('kb-paused')
  await page.getByLabel('你的学习目标').fill('整理鉴别诊断')
  await page.getByRole('button', { name: '开始执行' }).click()
  await expect(page.getByRole('dialog', { name: 'Agent 想问你' })).toContainText('需要偏重诊断还是治疗？')

  const persisted = await page.evaluate(key => JSON.parse(sessionStorage.getItem(key)), RECOVERY_KEY)
  expect(persisted).toMatchObject({
    schema_version: 3,
    kind: 'awaiting',
    conversation_id: 'kb-server-schema4-session',
    request: {
      query: '整理鉴别诊断',
      knowledge_base_id: 'kb-paused',
      grounding_required: true,
      web_enabled: false,
    },
  })

  await page.getByLabel('你的回答').fill('偏重诊断')
  await page.reload()
  const restoredDialog = page.getByRole('dialog', { name: 'Agent 想问你' })
  await expect(restoredDialog).toContainText('需要偏重诊断还是治疗？')
  await expect(page.getByLabel('你的回答')).toHaveValue('偏重诊断')
  expect(starts).toHaveLength(1)

  await restoredDialog.getByRole('button', { name: '回答 ⏎ (Ctrl+Enter)' }).click()
  await expect(page.getByText('已按诊断重点继续分析。')).toBeVisible()
  await expect(page.getByText('诊断笔记.md')).toBeVisible()
  expect(continuations).toHaveLength(1)
  expect(continuations[0].body).toEqual({
    conversation_id: 'kb-server-schema4-session',
    user_reply: '偏重诊断',
  })
  expect(continuations[0].key).toMatch(UUID_V4_PATTERN)
  await expect.poll(() => page.evaluate(key => sessionStorage.getItem(key), RECOVERY_KEY)).toBeNull()
  expect(unexpected).toEqual([])
})
