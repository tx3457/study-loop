import { expect, test } from '@playwright/test'

const API_PREFIX = '/api'
const UUID_V4_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i

async function expectStyledAction(locator) {
  const styles = await locator.evaluate(element => {
    const computed = getComputedStyle(element)
    return {
      height: element.getBoundingClientRect().height,
      background: computed.backgroundColor,
      radius: computed.borderRadius,
    }
  })
  expect(styles.height).toBeGreaterThanOrEqual(44)
  expect(styles.background).not.toBe('rgba(0, 0, 0, 0)')
  expect(styles.radius).not.toBe('0px')
}

async function mockKnowledgeApi(page) {
  const requests = []
  const bases = [
    { id: 'kb-med', name: '医学复习', description: '课程资料', status: 'ready', revision: 2, epoch: 1, document_count: 0, created_at: '2026-09-20T00:00:00Z' },
  ]
  const documents = []

  await page.route(url => url.pathname.startsWith(`${API_PREFIX}/`), async route => {
    const request = route.request()
    const url = new URL(request.url())
    const path = url.pathname.slice(API_PREFIX.length)
    requests.push(`${request.method()} ${path}`)
    let status = 200
    let body

    if (request.method() === 'GET' && path === '/knowledge-bases/capabilities') {
      body = { enabled: true, available: true, web_search_available: true }
    } else if (request.method() === 'GET' && path === '/knowledge-bases') {
      body = { knowledge_bases: bases }
    } else if (request.method() === 'POST' && path === '/knowledge-bases') {
      const input = await request.postDataJSON()
      const created = { id: 'kb-ml', name: input.name, description: input.description, status: 'ready', revision: 1, epoch: 1, document_count: 0, created_at: '2026-09-20T00:00:00Z' }
      bases.push(created)
      body = created
    } else if (request.method() === 'GET' && path === '/knowledge-bases/kb-ml') {
      body = bases.find(item => item.id === 'kb-ml')
    } else if (request.method() === 'GET' && path === '/knowledge-bases/kb-ml/documents') {
      body = { documents }
    } else if (request.method() === 'GET' && path === '/documents') {
      body = { documents: ['legacy.md'] }
    } else if (request.method() === 'POST' && path === '/knowledge-bases/kb-ml/documents/upload') {
      expect(request.postDataBuffer().toString()).toContain('expected_revision')
      status = 202
      body = { job_id: 'job-upload', knowledge_base_id: 'kb-ml' }
    } else if (request.method() === 'GET' && path === '/knowledge-jobs/job-upload') {
      documents.splice(0, documents.length, { id: 'doc-1', knowledge_base_id: 'kb-ml', name: 'rag.md', kind: 'file', version_id: 'v1', status: 'ready', content_hash: 'sha256', created_at: '2026-09-20T00:00:00Z' })
      const kb = bases.find(item => item.id === 'kb-ml')
      kb.revision = 2
      kb.document_count = 1
      body = { id: 'job-upload', knowledge_base_id: 'kb-ml', status: 'succeeded', stage: 'complete', revision: 2, error_code: null }
    } else if (request.method() === 'GET' && path === '/knowledge-bases/kb-ml/graph') {
      body = {
        revision: bases.find(item => item.id === 'kb-ml').revision,
        epoch: 1,
        truncated: false,
        nodes: [
          { id: 'node-rag', label: 'RAG', aliases: ['检索增强生成'], source_version_ids: ['v1'] },
          { id: 'node-retrieval', label: '检索', aliases: [] },
        ],
        edges: [{ id: 'edge-1', source: 'node-rag', target: 'node-retrieval', label: '使用', source_version_ids: ['v1'] }],
      }
    } else if (request.method() === 'GET' && path === '/knowledge-bases/kb-ml/sources/v1') {
      body = {
        source_version_id: 'v1',
        title: 'rag.md',
        source_status: 'deleted',
        text: 'RAG 来源正文，用于核对图谱事实。',
        locator: { page: 3, section: '检索增强生成' },
        blocks: [{ text: '检索先找到相关资料。', locator: { page: 3, paragraph: 2 } }],
      }
    } else if (request.method() === 'PUT' && path === '/knowledge-bases/kb-ml/documents/doc-1') {
      const multipart = request.postDataBuffer().toString()
      expect(multipart).toContain('expected_revision')
      expect(multipart).toContain('replacement.md')
      status = 202
      body = { job_id: 'job-replace', knowledge_base_id: 'kb-ml' }
    } else if (request.method() === 'GET' && path === '/knowledge-jobs/job-replace') {
      documents[0] = { ...documents[0], name: 'replacement.md', version_id: 'v2' }
      const kb = bases.find(item => item.id === 'kb-ml')
      kb.revision = 3
      body = { id: 'job-replace', knowledge_base_id: 'kb-ml', status: 'succeeded', stage: 'complete', revision: 3, error_code: null }
    } else if (request.method() === 'POST' && path === '/knowledge-bases/kb-ml/corrections') {
      const input = await request.postDataJSON()
      expect(input).toMatchObject({ kind: 'rename_entity', entity_id: 'node-rag', label: '检索增强生成', expected_revision: 3 })
      status = 202
      body = { job_id: 'job-correction', knowledge_base_id: 'kb-ml' }
    } else if (request.method() === 'GET' && path === '/knowledge-jobs/job-correction') {
      const kb = bases.find(item => item.id === 'kb-ml')
      kb.revision = 4
      body = { id: 'job-correction', knowledge_base_id: 'kb-ml', status: 'succeeded', stage: 'complete', revision: 4, error_code: null }
    } else if (request.method() === 'POST' && path === '/agent/autonomous') {
      const input = await request.postDataJSON()
      expect(input).toMatchObject({ knowledge_base_id: 'kb-ml', document_id: null, grounding_required: true, web_enabled: true })
      body = {
        awaiting_user_input: false,
        final_answer: 'RAG 将检索与生成结合。',
        rounds_used: 1,
        steps: [],
        tools_called: ['query_knowledge_base', 'web_search'],
        truncated: false,
        knowledge_base_id: 'kb-ml',
        web_enabled: true,
        web_only: false,
        source_citations: [
          { kind: 'kb_chunk', evidence_id: 'e1', knowledge_base_id: 'kb-ml', document_id: 'doc-1', source_version_id: 'v1', chunk_id: 'chunk-1', title: 'rag.md', snippet: '<b>可信文本</b>', source_status: 'historical' },
          { kind: 'web_snapshot', evidence_id: 'e2', snapshot_id: 'snapshot-1', title: 'Web source', url: 'https://example.com/rag', fetched_at: '2026-09-20T00:00:00Z', content_hash: 'webhash', snippet: '网页摘要' },
        ],
        citations: [],
        invalid_citation_count: 0,
        grounding_status: 'citation_ids_valid',
        grounding_required: true,
      }
    } else if (request.method() === 'POST' && path === '/knowledge-bases/kb-ml/web-import') {
      expect(await request.postDataJSON()).toMatchObject({ snapshot_id: 'snapshot-1', expected_revision: 4 })
      status = 202
      body = { job_id: 'job-web', knowledge_base_id: 'kb-ml' }
    } else if (request.method() === 'GET' && path === '/knowledge-jobs/job-web') {
      body = { id: 'job-web', knowledge_base_id: 'kb-ml', status: 'succeeded', stage: 'complete', revision: 4, error_code: null }
    } else if (request.method() === 'DELETE' && path === '/knowledge-bases/kb-ml') {
      bases.splice(bases.findIndex(item => item.id === 'kb-ml'), 1)
      status = 202
      body = { job_id: 'job-delete', knowledge_base_id: 'kb-ml' }
    } else if (request.method() === 'GET' && path === '/knowledge-jobs/job-delete') {
      body = { id: 'job-delete', knowledge_base_id: 'kb-ml', status: 'succeeded', stage: 'complete', revision: 5, error_code: null }
    } else {
      status = 500
      body = { detail: `Unexpected request: ${request.method()} ${path}` }
    }

    await route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) })
  })
  return requests
}

test('knowledge base flow covers create, upload, graph correction, grounded ask, web import and delete', async ({ page }) => {
  const requests = await mockKnowledgeApi(page)
  await page.goto('/knowledge-bases')
  await expect(page.getByRole('heading', { name: '知识库', exact: true })).toBeVisible()

  await page.getByLabel('知识库名称').fill('机器学习')
  await page.getByLabel('知识库说明').fill('论文与课程笔记')
  await page.getByRole('button', { name: '创建知识库' }).click()
  await page.getByRole('link', { name: /机器学习/ }).click()

  await expectStyledAction(page.getByRole('button', { name: '重建索引' }))
  await expectStyledAction(page.getByText('上传资料', { exact: true }))
  await expectStyledAction(page.getByRole('button', { name: '保存名称' }))

  await page.getByLabel('上传资料').setInputFiles({ name: 'rag.md', mimeType: 'text/markdown', buffer: Buffer.from('# RAG') })
  await expect(page.getByText('rag.md', { exact: true })).toBeVisible()
  await expect(page.getByText('文档 · 已入库')).toBeVisible()
  await expectStyledAction(page.getByText('替换资料', { exact: true }))
  await page.getByLabel('替换资料 rag.md').setInputFiles({ name: 'replacement.md', mimeType: 'text/markdown', buffer: Buffer.from('# Replacement') })
  await expect(page.getByText('replacement.md', { exact: true })).toBeVisible()

  await page.getByRole('tab', { name: '知识图谱' }).click()
  await expect(page.getByRole('img', { name: '知识图谱可视化' })).toBeVisible()
  await expect(page.getByRole('list', { name: '图谱实体列表' })).toContainText('RAG')
  await expect(page.getByRole('list', { name: '图谱实体列表' })).toContainText('别名：检索增强生成')
  await expect(page.getByRole('list', { name: '图谱实体列表' })).not.toContainText('node-rag')
  await expect(page.getByRole('list', { name: '图谱关系列表' })).toContainText('RAG · 使用 · 检索')
  await expect(page.getByRole('list', { name: '图谱关系列表' })).not.toContainText('edge-1')
  await page.locator('.graph-layer g[aria-label="选择实体 RAG"]').click()
  await expect(page.getByLabel('聚焦实体')).toHaveValue('node-rag')
  const graphLayer = page.locator('.graph-layer')
  await expect(graphLayer).toHaveAttribute('transform', /scale\(1\)/)
  await page.getByRole('button', { name: '放大图谱' }).click()
  await expect(graphLayer).toHaveAttribute('transform', /scale\(1\.2\)/)
  const graphBox = await page.getByRole('img', { name: '知识图谱可视化' }).boundingBox()
  await page.mouse.move(graphBox.x + 120, graphBox.y + 120)
  await page.mouse.down()
  await page.mouse.move(graphBox.x + 160, graphBox.y + 145)
  await page.mouse.up()
  await expect(graphLayer).toHaveAttribute('transform', /translate\((?!0 0)/)
  await page.setViewportSize({ width: 390, height: 844 })
  await page.getByRole('button', { name: '查看来源' }).click()
  const drawer = page.getByRole('complementary', { name: '来源详情' })
  await expect(drawer).toContainText('RAG 来源正文，用于核对图谱事实。')
  await expect(drawer).toContainText('检索先找到相关资料。')
  await expect(drawer).toContainText('第 3 页')
  await expect(drawer).toContainText('来源已删除')
  await drawer.getByRole('button', { name: '关闭来源详情' }).click({ timeout: 3000 })
  await expect(drawer).toHaveCount(0)
  await page.setViewportSize({ width: 1280, height: 720 })
  await page.locator('.graph-layer circle[aria-label="选择关系 RAG 与 检索"]').click()
  await expect(page.getByLabel('纠错类型')).toHaveValue('delete_relation')
  await expect(page.getByLabel('纠错关系')).toHaveValue('edge-1')
  await page.getByLabel('纠错类型').selectOption('rename_entity')
  await page.getByLabel('纠错实体').selectOption({ label: 'RAG（别名：检索增强生成）' })
  await page.getByLabel('新名称').fill('检索增强生成')
  await page.getByRole('button', { name: '提交纠错' }).click()

  await page.getByRole('tab', { name: '问知识库' }).click()
  await page.getByRole('link', { name: '打开知识库问答' }).click()
  await page.getByLabel('联网补充').check()
  await page.getByLabel('你的学习目标').fill('解释 RAG')
  await page.getByRole('button', { name: '开始执行' }).click()
  await expect(page.getByText('知识库来源')).toBeVisible()
  await expect(page.getByText('历史来源')).toBeVisible()
  await expect(page.getByText('<b>可信文本</b>')).toBeVisible()
  await expect(page.locator('.source-snippet b')).toHaveCount(0)
  await expect(page.getByRole('link', { name: 'https://example.com/rag' })).toHaveAttribute('rel', /noopener/)
  await page.getByRole('button', { name: '收录到当前知识库' }).click()

  await page.goto('/knowledge-bases/kb-ml')
  await page.getByRole('button', { name: '删除知识库' }).click()
  await expect(page.getByRole('dialog', { name: '删除知识库' })).toBeVisible()
  await page.getByRole('button', { name: '确认删除' }).click()
  await expect(page).toHaveURL(/\/knowledge-bases$/)
  expect(requests).toContain('GET /knowledge-jobs/job-upload')
})

test('knowledge UI is mobile usable and capabilities failure does not break legacy navigation', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.route(url => url.pathname.startsWith(`${API_PREFIX}/`), async route => {
    const path = new URL(route.request().url()).pathname.slice(API_PREFIX.length)
    if (path === '/knowledge-bases/capabilities') {
      await route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ detail: 'disabled' }) })
      return
    }
    if (path === '/documents') {
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ documents: [] }) })
      return
    }
    await route.fulfill({ status: 500, contentType: 'application/json', body: '{}' })
  })
  await page.goto('/documents')
  await page.getByRole('button', { name: '打开导航' }).click()
  await expect(page.getByRole('link', { name: '知识库' })).toHaveCount(0)
  await expect(page.getByRole('link', { name: '文档管理' })).toBeVisible()
})

test('revision conflicts refresh state, preserve edits, and include the request id', async ({ page }) => {
  let detailCalls = 0
  await page.route(url => url.pathname.startsWith(`${API_PREFIX}/`), async route => {
    const request = route.request()
    const path = new URL(request.url()).pathname.slice(API_PREFIX.length)
    let status = 200
    let headers
    let body
    if (path === '/knowledge-bases/capabilities') {
      body = { enabled: true, available: true, web_search_available: false }
    } else if (request.method() === 'GET' && path === '/knowledge-bases/kb-race') {
      detailCalls += 1
      body = { id: 'kb-race', name: detailCalls > 1 ? '服务端新名称' : '原名称', description: '', status: 'ready', revision: detailCalls > 1 ? 8 : 7, epoch: 1, document_count: 0, created_at: '2026-09-20T00:00:00Z' }
    } else if (request.method() === 'GET' && path === '/knowledge-bases/kb-race/documents') {
      body = { documents: [] }
    } else if (request.method() === 'GET' && path === '/documents') {
      body = { documents: [] }
    } else if (request.method() === 'PATCH' && path === '/knowledge-bases/kb-race') {
      status = 409
      headers = { 'X-Request-ID': 'req_0123456789abcdef0123456789abcdef' }
      body = { detail: '版本冲突' }
    } else {
      status = 500
      body = { detail: `Unexpected request: ${request.method()} ${path}` }
    }
    await route.fulfill({ status, headers, contentType: 'application/json', body: JSON.stringify(body) })
  })

  await page.goto('/knowledge-bases/kb-race')
  const name = page.getByLabel('知识库名称')
  await name.fill('我尚未保存的名称')
  await page.getByRole('button', { name: '保存名称' }).click()
  await expect(page.getByRole('alert')).toContainText('已刷新最新版本')
  await expect(name).toHaveValue('我尚未保存的名称')
  await expect(page.getByText('版本 8')).toBeVisible()
})

test('an existing dirty knowledge base can rebuild from saved materials and corrections', async ({ page }) => {
  let rebuilt = false
  await page.route(url => url.pathname.startsWith(`${API_PREFIX}/`), async route => {
    const request = route.request()
    const path = new URL(request.url()).pathname.slice(API_PREFIX.length)
    let status = 200
    let body
    if (path === '/knowledge-bases/capabilities') {
      body = { enabled: true, available: true, web_search_available: false }
    } else if (request.method() === 'GET' && path === '/knowledge-bases/kb-dirty') {
      body = { id: 'kb-dirty', name: '待恢复知识库', description: '', status: rebuilt ? 'ready' : 'dirty', revision: rebuilt ? 10 : 9, epoch: 2, document_count: 2, created_at: '2026-09-20T00:00:00Z' }
    } else if (request.method() === 'GET' && path === '/knowledge-bases/kb-dirty/documents') {
      body = { documents: [] }
    } else if (request.method() === 'GET' && path === '/documents') {
      body = { documents: [] }
    } else if (request.method() === 'POST' && path === '/knowledge-bases/kb-dirty/rebuild') {
      expect(await request.postDataJSON()).toEqual({ expected_revision: 9 })
      expect(await request.headerValue('idempotency-key')).toMatch(UUID_V4_PATTERN)
      status = 202
      body = { job_id: 'job-rebuild', knowledge_base_id: 'kb-dirty' }
    } else if (request.method() === 'GET' && path === '/knowledge-jobs/job-rebuild') {
      rebuilt = true
      body = { id: 'job-rebuild', knowledge_base_id: 'kb-dirty', status: 'succeeded', stage: 'complete', revision: 10, error_code: null }
    } else {
      status = 500
      body = { detail: `Unexpected request: ${request.method()} ${path}` }
    }
    await route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) })
  })

  await page.goto('/knowledge-bases/kb-dirty')
  await expect(page.getByText('需要恢复 · 版本 9')).toBeVisible()
  await expect(page.getByText('使用当前资料和已保存的图谱纠错重新建立索引。')).toBeVisible()
  await page.getByRole('button', { name: '重建索引' }).click()
  await expect(page.getByText('索引已重建')).toBeVisible()
  await expect(page.getByText('可用 · 版本 10')).toBeVisible()
})

test('graph correction form submits merge, entity deletion, and relation deletion with the current revision', async ({ page }) => {
  let revision = 4
  let jobSequence = 0
  const corrections = []
  const nodes = [
    { id: 'entity-rag', label: 'RAG', aliases: ['检索增强生成'], source_version_ids: ['v1'] },
    { id: 'entity-search', label: '检索', aliases: [], source_version_ids: ['v1'] },
    { id: 'entity-generation', label: '生成', aliases: [], source_version_ids: ['v1'] },
    { id: 'entity-obsolete', label: '旧概念', aliases: [], source_version_ids: ['v1'] },
  ]
  const edges = [
    { id: 'relation-uses', source: 'entity-rag', target: 'entity-search', label: '使用', source_version_ids: ['v1'] },
  ]

  await page.setViewportSize({ width: 390, height: 844 })
  await page.route(url => url.pathname.startsWith(`${API_PREFIX}/`), async route => {
    const request = route.request()
    const path = new URL(request.url()).pathname.slice(API_PREFIX.length)
    let status = 200
    let body
    if (path === '/knowledge-bases/capabilities') {
      body = { enabled: true, available: true, web_search_available: false }
    } else if (request.method() === 'GET' && path === '/knowledge-bases/kb-corrections') {
      body = { id: 'kb-corrections', name: '图谱纠错', description: '', status: 'ready', revision, epoch: 1, document_count: 1, created_at: '2026-09-20T00:00:00Z' }
    } else if (request.method() === 'GET' && path === '/knowledge-bases/kb-corrections/documents') {
      body = { documents: [] }
    } else if (request.method() === 'GET' && path === '/documents') {
      body = { documents: [] }
    } else if (request.method() === 'GET' && path === '/knowledge-bases/kb-corrections/graph') {
      body = { revision, epoch: 1, truncated: false, nodes, edges }
    } else if (request.method() === 'POST' && path === '/knowledge-bases/kb-corrections/corrections') {
      const correction = await request.postDataJSON()
      if (correction.kind === 'merge_entities') {
        expect(correction.entity_ids).toHaveLength(new Set(correction.entity_ids).size)
        expect(correction.entity_ids).not.toContain(correction.target_id)
      }
      corrections.push(correction)
      expect(await request.headerValue('idempotency-key')).toMatch(UUID_V4_PATTERN)
      jobSequence += 1
      status = 202
      body = { job_id: `correction-${jobSequence}`, knowledge_base_id: 'kb-corrections' }
    } else if (request.method() === 'GET' && /^\/knowledge-jobs\/correction-\d+$/.test(path)) {
      const correction = corrections.at(-1)
      if (correction.kind === 'merge_entities') {
        for (const entityId of correction.entity_ids) {
          const index = nodes.findIndex(node => node.id === entityId)
          if (index >= 0) nodes.splice(index, 1)
        }
      } else if (correction.kind === 'delete_entity') {
        const index = nodes.findIndex(node => node.id === correction.entity_id)
        if (index >= 0) nodes.splice(index, 1)
      }
      revision += 1
      body = { id: path.split('/').at(-1), knowledge_base_id: 'kb-corrections', status: 'succeeded', stage: 'complete', revision, error_code: null }
    } else {
      status = 500
      body = { detail: `Unexpected request: ${request.method()} ${path}` }
    }
    await route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) })
  })

  await page.goto('/knowledge-bases/kb-corrections')
  await page.getByRole('tab', { name: '知识图谱' }).click()

  await page.getByLabel('纠错类型').selectOption('merge_entities')
  await page.getByLabel('合并后的目标实体').selectOption('entity-rag')
  await expect(page.getByLabel('合并来源实体').locator('option[value="entity-rag"]')).toHaveCount(0)
  await page.getByLabel('合并来源实体').selectOption('entity-generation')
  await page.getByRole('button', { name: '提交纠错' }).click()
  await expect(page.getByText('图谱纠错已应用')).toBeVisible()
  await expect(page.getByRole('list', { name: '图谱实体列表' }).getByText('生成', { exact: true })).toHaveCount(0)

  await page.getByLabel('纠错类型').selectOption('delete_entity')
  await page.getByLabel('纠错实体').selectOption('entity-obsolete')
  await page.getByRole('button', { name: '提交纠错' }).click()
  await expect(page.getByText('可用 · 版本 6')).toBeVisible()

  await page.getByLabel('纠错类型').selectOption('delete_relation')
  await page.getByLabel('纠错关系').selectOption('relation-uses')
  await page.getByRole('button', { name: '提交纠错' }).click()
  await expect(page.getByText('可用 · 版本 7')).toBeVisible()

  expect(corrections).toEqual([
    { kind: 'merge_entities', entity_ids: ['entity-generation'], target_id: 'entity-rag', expected_revision: 4 },
    { kind: 'delete_entity', entity_id: 'entity-obsolete', expected_revision: 5 },
    { kind: 'delete_relation', edge_id: 'relation-uses', expected_revision: 6 },
  ])
})

test('document deletion confirms the destructive action and removes the document after its job succeeds', async ({ page }) => {
  let deleted = false
  let revision = 2
  const deleteBodies = []
  await page.route(url => url.pathname.startsWith(`${API_PREFIX}/`), async route => {
    const request = route.request()
    const path = new URL(request.url()).pathname.slice(API_PREFIX.length)
    let status = 200
    let body
    if (path === '/knowledge-bases/capabilities') {
      body = { enabled: true, available: true, web_search_available: false }
    } else if (request.method() === 'GET' && path === '/knowledge-bases/kb-doc-delete') {
      body = { id: 'kb-doc-delete', name: '资料删除', description: '', status: 'ready', revision, epoch: 1, document_count: deleted ? 0 : 1, created_at: '2026-09-20T00:00:00Z' }
    } else if (request.method() === 'GET' && path === '/knowledge-bases/kb-doc-delete/documents') {
      body = { documents: deleted ? [] : [{ id: 'doc-delete', knowledge_base_id: 'kb-doc-delete', name: 'obsolete.md', kind: 'file', version_id: 'v1', status: 'ready', content_hash: 'sha256', created_at: '2026-09-20T00:00:00Z' }] }
    } else if (request.method() === 'GET' && path === '/documents') {
      body = { documents: [] }
    } else if (request.method() === 'DELETE' && path === '/knowledge-bases/kb-doc-delete/documents/doc-delete') {
      deleteBodies.push(await request.postDataJSON())
      expect(await request.headerValue('idempotency-key')).toMatch(UUID_V4_PATTERN)
      status = 202
      body = { job_id: 'job-doc-delete', knowledge_base_id: 'kb-doc-delete' }
    } else if (request.method() === 'GET' && path === '/knowledge-jobs/job-doc-delete') {
      deleted = true
      revision = 3
      body = { id: 'job-doc-delete', knowledge_base_id: 'kb-doc-delete', status: 'succeeded', stage: 'complete', revision, error_code: null }
    } else {
      status = 500
      body = { detail: `Unexpected request: ${request.method()} ${path}` }
    }
    await route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) })
  })

  await page.goto('/knowledge-bases/kb-doc-delete')
  await page.getByRole('button', { name: '删除资料' }).click()
  const dialog = page.getByRole('dialog', { name: '删除知识库资料' })
  await expect(dialog).toContainText('资料将从这个知识库的新检索和图谱中移除')
  await dialog.getByRole('button', { name: '确认删除' }).click()
  await expect(page.locator('.kb-document-list').getByText('obsolete.md', { exact: true })).toHaveCount(0)
  await expect(page.getByText('尚无资料。上传文件，或显式复制一份已有文档。')).toBeVisible()
  expect(deleteBodies).toEqual([{ expected_revision: 2 }])
})

test('a failed indexing job can be retried with the failed job id and current revision', async ({ page }) => {
  let retryAccepted = false
  const retryBodies = []
  await page.route(url => url.pathname.startsWith(`${API_PREFIX}/`), async route => {
    const request = route.request()
    const path = new URL(request.url()).pathname.slice(API_PREFIX.length)
    let status = 200
    let body
    if (path === '/knowledge-bases/capabilities') {
      body = { enabled: true, available: true, web_search_available: false }
    } else if (request.method() === 'GET' && path === '/knowledge-bases/kb-retry') {
      body = { id: 'kb-retry', name: '失败重试', description: '', status: 'ready', revision: retryAccepted ? 6 : 5, epoch: 1, document_count: retryAccepted ? 1 : 0, created_at: '2026-09-20T00:00:00Z' }
    } else if (request.method() === 'GET' && path === '/knowledge-bases/kb-retry/documents') {
      body = { documents: retryAccepted ? [{ id: 'doc-recovered', knowledge_base_id: 'kb-retry', name: 'retry.md', kind: 'file', version_id: 'v1', status: 'ready', content_hash: 'sha256', created_at: '2026-09-20T00:00:00Z' }] : [] }
    } else if (request.method() === 'GET' && path === '/documents') {
      body = { documents: [] }
    } else if (request.method() === 'POST' && path === '/knowledge-bases/kb-retry/documents/upload') {
      status = 202
      body = { job_id: 'job-failed', knowledge_base_id: 'kb-retry' }
    } else if (request.method() === 'GET' && path === '/knowledge-jobs/job-failed') {
      body = { id: 'job-failed', knowledge_base_id: 'kb-retry', status: 'failed', stage: 'extract', revision: 5, error_code: 'extract_failed' }
    } else if (request.method() === 'POST' && path === '/knowledge-jobs/job-failed/retry') {
      retryBodies.push(await request.postDataJSON())
      expect(await request.headerValue('idempotency-key')).toMatch(UUID_V4_PATTERN)
      status = 202
      body = { job_id: 'job-recovered', knowledge_base_id: 'kb-retry' }
    } else if (request.method() === 'GET' && path === '/knowledge-jobs/job-recovered') {
      retryAccepted = true
      body = { id: 'job-recovered', knowledge_base_id: 'kb-retry', status: 'succeeded', stage: 'complete', revision: 6, error_code: null }
    } else {
      status = 500
      body = { detail: `Unexpected request: ${request.method()} ${path}` }
    }
    await route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) })
  })

  await page.goto('/knowledge-bases/kb-retry')
  await page.getByLabel('上传资料').setInputFiles({ name: 'retry.md', mimeType: 'text/markdown', buffer: Buffer.from('# Retry') })
  const failed = page.locator('.kb-job-failed')
  await expect(failed).toContainText('extract_failed')
  await failed.getByRole('button', { name: '重试任务' }).click()
  await expect(page.getByText('retry.md', { exact: true })).toBeVisible()
  await expect(page.getByText('后台任务已恢复并完成')).toBeVisible()
  expect(retryBodies).toEqual([{ expected_revision: 5 }])
})
