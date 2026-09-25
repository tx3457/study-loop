import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import {
  createIdempotencyKey,
  createKnowledgeBase,
  createKnowledgeCorrection,
  deleteKnowledgeBase,
  deleteKnowledgeDocument,
  getDocuments,
  getKnowledgeBase,
  getKnowledgeBases,
  getKnowledgeCapabilities,
  getKnowledgeDocuments,
  getKnowledgeGraph,
  getKnowledgeJob,
  getKnowledgeSource,
  importLegacyDocument,
  retryKnowledgeJob,
  replaceKnowledgeDocument,
  rebuildKnowledgeBase,
  uploadKnowledgeDocument,
  updateKnowledgeBase,
} from '../api/client'
import './KnowledgeBases.css'

const POLL_MS = 600
const DOCUMENT_PAGE_SIZE = 50

function statusLabel(status) {
  return ({ ready: '可用', updating: '更新中', dirty: '需要恢复', deleting: '删除中' })[status] || status
}

function documentKindLabel(kind) {
  return ({ file: '文档', web: '网页', legacy_copy: '旧文档副本' })[kind] || '资料'
}

function documentStatusLabel(status) {
  return ({
    active: '已入库',
    ready: '已入库',
    indexing: '处理中',
    updating: '处理中',
    failed: '处理失败',
    dirty: '需要恢复',
    deleting: '删除中',
  })[status] || '状态未知'
}

function requestError(error, fallback) {
  return error?.message || fallback
}

const SOURCE_STATUS = {
  current: '当前来源',
  deleted: '来源已删除',
  historical: '历史来源',
  unavailable: '来源不可用',
}

function boundedSourceText(value, limit = 12_000) {
  return typeof value === 'string' ? value.slice(0, limit) : ''
}

function locatorText(locator) {
  if (typeof locator === 'string') return locator
  if (!locator || typeof locator !== 'object') return ''
  const parts = []
  if (Number.isInteger(locator.page)) parts.push(`第 ${locator.page} 页`)
  if (locator.section) parts.push(`章节：${locator.section}`)
  if (Number.isInteger(locator.paragraph)) parts.push(`第 ${locator.paragraph} 段`)
  if (locator.label) parts.push(String(locator.label))
  return parts.join(' · ')
}

function entityDisplayName(node) {
  const aliases = Array.isArray(node?.aliases) ? node.aliases.filter(Boolean) : []
  return aliases.length ? `${node.label}（别名：${aliases.join('、')}）` : node?.label || '未命名实体'
}

function ConfirmDialog({ title, children, busy, onCancel, onConfirm }) {
  return (
    <div className="kb-dialog-backdrop" role="presentation">
      <section className="kb-dialog" role="dialog" aria-modal="true" aria-label={title}>
        <h2>{title}</h2>
        <p>{children}</p>
        <div className="kb-actions">
          <button type="button" className="btn-secondary" onClick={onCancel} disabled={busy}>取消</button>
          <button type="button" className="btn-danger" onClick={onConfirm} disabled={busy}>
            {busy ? '处理中…' : '确认删除'}
          </button>
        </div>
      </section>
    </div>
  )
}

function GraphView({ graph, onSource, onSelectNode, onSelectEdge }) {
  const [view, setView] = useState({ x: 0, y: 0, scale: 1 })
  const drag = useRef(null)
  const nodes = Array.isArray(graph?.nodes) ? graph.nodes.slice(0, 200) : []
  const nodeById = new Map(nodes.map(node => [node.id, node]))
  const edges = (Array.isArray(graph?.edges) ? graph.edges : []).filter(edge => {
    const source = edge.source || edge.source_id
    const target = edge.target || edge.target_id
    return nodeById.has(source) && nodeById.has(target)
  }).slice(0, 400)
  const positioned = nodes.map((node, index) => {
    const angle = (Math.PI * 2 * index) / Math.max(nodes.length, 1)
    const radius = nodes.length < 3 ? 100 : 150
    return { ...node, x: 220 + Math.cos(angle) * radius, y: 190 + Math.sin(angle) * radius }
  })
  const positions = new Map(positioned.map(node => [node.id, node]))

  if (!nodes.length) return <p className="kb-empty">当前筛选下没有图谱实体。</p>

  return (
    <div className="graph-layout">
      <div className="graph-viewport">
        <div className="graph-controls" aria-label="图谱视图控制">
          <button type="button" aria-label="放大图谱" onClick={() => setView(current => ({ ...current, scale: Math.min(2.4, Number((current.scale + .2).toFixed(1))) }))}>＋</button>
          <button type="button" aria-label="缩小图谱" onClick={() => setView(current => ({ ...current, scale: Math.max(.5, Number((current.scale - .2).toFixed(1))) }))}>−</button>
          <button type="button" aria-label="重置图谱视图" onClick={() => setView({ x: 0, y: 0, scale: 1 })}>重置</button>
        </div>
        <svg
          className="graph-canvas"
          viewBox="0 0 440 380"
          role="img"
          aria-label="知识图谱可视化"
          tabIndex="0"
          onPointerDown={event => {
            if (event.target.closest?.('[role="button"]')) return
            drag.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, startX: view.x, startY: view.y }
            event.currentTarget.setPointerCapture(event.pointerId)
          }}
          onPointerMove={event => {
            if (drag.current?.pointerId !== event.pointerId) return
            const factor = 440 / Math.max(event.currentTarget.clientWidth, 1)
            setView(current => ({ ...current, x: drag.current.startX + (event.clientX - drag.current.x) * factor, y: drag.current.startY + (event.clientY - drag.current.y) * factor }))
          }}
          onPointerUp={event => {
            if (drag.current?.pointerId === event.pointerId) drag.current = null
          }}
          onKeyDown={event => {
            const movement = 16
            const delta = { ArrowLeft: [movement, 0], ArrowRight: [-movement, 0], ArrowUp: [0, movement], ArrowDown: [0, -movement] }[event.key]
            if (!delta) return
            event.preventDefault()
            setView(current => ({ ...current, x: current.x + delta[0], y: current.y + delta[1] }))
          }}
        >
        <g className="graph-layer" transform={`translate(${view.x} ${view.y}) scale(${view.scale})`}>
          {edges.map(edge => {
          const source = positions.get(edge.source || edge.source_id)
          const target = positions.get(edge.target || edge.target_id)
          return <g key={edge.id}><line className="graph-edge-line" x1={source.x} y1={source.y} x2={target.x} y2={target.y} /><circle className="graph-edge-handle" cx={(source.x + target.x) / 2} cy={(source.y + target.y) / 2} r="8" role="button" tabIndex="0" aria-label={`选择关系 ${source.label} 与 ${target.label}`} onClick={event => { event.stopPropagation(); onSelectEdge(edge) }} onKeyDown={event => { if (['Enter', ' '].includes(event.key)) { event.preventDefault(); onSelectEdge(edge) } }} /></g>
          })}
          {positioned.map(node => (
          <g key={node.id} transform={`translate(${node.x} ${node.y})`} role="button" tabIndex="0" aria-label={`选择实体 ${node.label}`} onClick={event => { event.stopPropagation(); onSelectNode(node) }} onKeyDown={event => { if (['Enter', ' '].includes(event.key)) { event.preventDefault(); onSelectNode(node) } }}>
            <circle r="24" />
            <text textAnchor="middle" y="38">{node.label}</text>
          </g>
          ))}
        </g>
        </svg>
      </div>
      <div>
        <h3>实体与关系</h3>
        <ul className="graph-accessible-list" aria-label="图谱实体列表">
          {nodes.map(node => {
            const sourceId = node.source_version_id || node.source_version_ids?.[0]
            return (
              <li key={node.id}>
                <strong>{node.label}</strong>
                {Array.isArray(node.aliases) && node.aliases.length > 0 && <span>别名：{node.aliases.join('、')}</span>}
                <button type="button" className="btn-link" aria-label={`选择实体 ${node.label}`} onClick={() => onSelectNode(node)}>选择实体</button>
                {sourceId && <button type="button" className="btn-link" onClick={() => onSource(sourceId)}>查看来源</button>}
              </li>
            )
          })}
        </ul>
        <ul className="graph-edge-list" aria-label="图谱关系列表">
          {edges.map(edge => {
            const source = nodeById.get(edge.source || edge.source_id)
            const target = nodeById.get(edge.target || edge.target_id)
            return <li key={edge.id}><button type="button" onClick={() => onSelectEdge(edge)}>{source?.label || '未命名实体'} · {edge.label || edge.relation || '相关'} · {target?.label || '未命名实体'}</button></li>
          })}
        </ul>
      </div>
    </div>
  )
}

export default function KnowledgeBases() {
  const { knowledgeBaseId } = useParams()
  const navigate = useNavigate()
  const mounted = useRef(true)
  const pollController = useRef(null)
  const detailRequestId = useRef(0)
  const [capabilities, setCapabilities] = useState(null)
  const [bases, setBases] = useState([])
  const [base, setBase] = useState(null)
  const [documents, setDocuments] = useState([])
  const [documentPage, setDocumentPage] = useState({ total: 0, limit: DOCUMENT_PAGE_SIZE, offset: 0 })
  const documentOffset = useRef(0)
  const [legacyDocuments, setLegacyDocuments] = useState([])
  const [graph, setGraph] = useState(null)
  const [tab, setTab] = useState('documents')
  const [busy, setBusy] = useState(false)
  const [job, setJob] = useState(null)
  const [error, setError] = useState(null)
  const [notice, setNotice] = useState('')
  const [createForm, setCreateForm] = useState({ name: '', description: '' })
  const [rename, setRename] = useState('')
  const [legacyId, setLegacyId] = useState('')
  const [graphFilter, setGraphFilter] = useState({ search: '', focus: '' })
  const [correction, setCorrection] = useState({ kind: 'rename_entity', entity_id: '', label: '', entity_ids: [], target_id: '', edge_id: '' })
  const [source, setSource] = useState(null)
  const [confirm, setConfirm] = useState(null)

  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      pollController.current?.abort()
    }
  }, [])

  const loadList = useCallback(async () => {
    const result = await getKnowledgeBases()
    if (mounted.current) setBases(Array.isArray(result?.knowledge_bases) ? result.knowledge_bases : [])
  }, [])

  const loadDetail = useCallback(async ({ preserveForms = false, offset = documentOffset.current } = {}) => {
    if (!knowledgeBaseId) return
    const requestId = ++detailRequestId.current
    const [baseResult, initialDocsResult] = await Promise.all([
      getKnowledgeBase(knowledgeBaseId),
      getKnowledgeDocuments(knowledgeBaseId, { limit: DOCUMENT_PAGE_SIZE, offset }),
    ])
    if (!mounted.current || requestId !== detailRequestId.current) return
    let docsResult = initialDocsResult
    let pageDocuments = Array.isArray(docsResult?.documents) ? docsResult.documents : []
    const hasPagination = (
      Number.isInteger(docsResult?.total)
      && docsResult.total >= 0
      && Number.isInteger(docsResult?.limit)
      && docsResult.limit > 0
      && Number.isInteger(docsResult?.offset)
      && docsResult.offset >= 0
    )
    if (!hasPagination && (offset !== 0 || pageDocuments.length >= DOCUMENT_PAGE_SIZE)) {
      throw new Error('资料分页响应缺少 total、limit 或 offset')
    }
    let total = hasPagination ? docsResult.total : pageDocuments.length
    let limit = hasPagination ? docsResult.limit : DOCUMENT_PAGE_SIZE
    let resolvedOffset = hasPagination ? docsResult.offset : 0
    if (total > 0 && resolvedOffset >= total) {
      const validOffset = Math.floor((total - 1) / limit) * limit
      docsResult = await getKnowledgeDocuments(knowledgeBaseId, { limit, offset: validOffset })
      if (!mounted.current || requestId !== detailRequestId.current) return
      if (
        !Number.isInteger(docsResult?.total)
        || docsResult.total < 0
        || !Number.isInteger(docsResult?.limit)
        || docsResult.limit <= 0
        || !Number.isInteger(docsResult?.offset)
        || docsResult.offset < 0
      ) {
        throw new Error('资料分页响应缺少 total、limit 或 offset')
      }
      pageDocuments = Array.isArray(docsResult.documents) ? docsResult.documents : []
      total = docsResult.total
      limit = docsResult.limit
      resolvedOffset = docsResult.offset
    }
    if (total === 0) resolvedOffset = 0
    setBase(baseResult)
    if (!preserveForms) setRename(baseResult?.name || '')
    documentOffset.current = resolvedOffset
    setDocuments(pageDocuments)
    setDocumentPage({ total, limit, offset: resolvedOffset })
  }, [knowledgeBaseId])

  const refreshAfterConflict = useCallback(async (err) => {
    if (err?.status !== 409) return false
    await loadDetail({ preserveForms: true })
    if (mounted.current) setError(
      `知识库已在其他位置更新，已刷新最新版本；你的输入仍保留，请确认后重试。${err?.requestId ? `（请求编号：${err.requestId}）` : ''}`,
    )
    return true
  }, [loadDetail])

  useEffect(() => {
    detailRequestId.current += 1
    documentOffset.current = 0
    setDocumentPage({ total: 0, limit: DOCUMENT_PAGE_SIZE, offset: 0 })
  }, [knowledgeBaseId])

  useEffect(() => {
    const controller = new AbortController()
    setError(null)
    getKnowledgeCapabilities({ signal: controller.signal })
      .then(result => {
        if (!mounted.current) return
        setCapabilities(result)
        if (result?.enabled && result?.available) {
          const loading = knowledgeBaseId ? loadDetail() : loadList()
          void loading.catch(err => {
            if (mounted.current) setError(requestError(err, '知识库加载失败'))
          })
        }
      })
      .catch(err => {
        if (err.name !== 'AbortError' && mounted.current) setCapabilities({ enabled: false, available: false })
      })
    return () => controller.abort()
  }, [knowledgeBaseId, loadDetail, loadList])

  useEffect(() => {
    if (!knowledgeBaseId) return
    getDocuments().then(result => {
      if (mounted.current) setLegacyDocuments(Array.isArray(result?.documents) ? result.documents : [])
    }).catch(() => {})
  }, [knowledgeBaseId])

  const pollJob = useCallback(async (jobId, message) => {
    pollController.current?.abort()
    const controller = new AbortController()
    pollController.current = controller
    while (!controller.signal.aborted && mounted.current) {
      const result = await getKnowledgeJob(jobId, { signal: controller.signal })
      if (!mounted.current || controller.signal.aborted) return null
      setJob(result)
      if (result?.status === 'succeeded') {
        setNotice(message)
        await loadDetail()
        if (tab === 'graph') {
          const updated = await getKnowledgeGraph(knowledgeBaseId, graphFilter)
          if (mounted.current) setGraph(updated)
        }
        return result
      }
      if (result?.status === 'failed') throw new Error(result.error_code || '后台任务失败')
      await new Promise(resolve => setTimeout(resolve, POLL_MS))
    }
    return null
  }, [graphFilter, knowledgeBaseId, loadDetail, tab])

  async function runMutation(action, successMessage) {
    setBusy(true)
    setError(null)
    setNotice('')
    try {
      const result = await action()
      if (result?.job_id) await pollJob(result.job_id, successMessage)
      else {
        setNotice(successMessage)
        await (knowledgeBaseId ? loadDetail() : loadList())
      }
      return true
    } catch (err) {
      if (!(await refreshAfterConflict(err))) setError(requestError(err, '操作失败'))
      return false
    } finally {
      if (mounted.current) setBusy(false)
    }
  }

  async function handleCreate(event) {
    event.preventDefault()
    const ok = await runMutation(
      () => createKnowledgeBase({ name: createForm.name.trim(), description: createForm.description.trim() }, createIdempotencyKey()),
      '知识库已创建',
    )
    if (ok) setCreateForm({ name: '', description: '' })
  }

  async function handleUpload(event) {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file || !base) return
    await runMutation(
      () => uploadKnowledgeDocument(base.id, file, base.revision, createIdempotencyKey()),
      `${file.name} 已完成索引`,
    )
  }

  async function handleReplacement(document, event) {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file || !base) return
    await runMutation(
      () => replaceKnowledgeDocument(base.id, document.id, file, base.revision, createIdempotencyKey()),
      `${document.name} 已替换为新版本`,
    )
  }

  async function handleDeleteBase() {
    setBusy(true)
    setError(null)
    try {
      const accepted = await deleteKnowledgeBase(base.id, base.revision, createIdempotencyKey())
      const acceptedJobId = accepted?.job_id
      while (mounted.current && acceptedJobId) {
        const current = await getKnowledgeJob(acceptedJobId)
        if (!mounted.current) return
        if (current?.status === 'succeeded') break
        if (current?.status === 'failed') throw new Error(current.error_code || '知识库删除失败')
        await new Promise(resolve => setTimeout(resolve, POLL_MS))
      }
      if (!mounted.current) return
      setConfirm(null)
      navigate('/knowledge-bases')
    } catch (err) {
      if (!(await refreshAfterConflict(err))) setError(requestError(err, '知识库删除失败'))
    } finally {
      if (mounted.current) setBusy(false)
    }
  }

  async function handleRetryJob() {
    if (!job?.id || !base) return
    await runMutation(
      () => retryKnowledgeJob(job.id, base.revision, createIdempotencyKey()),
      '后台任务已恢复并完成',
    )
  }

  async function loadGraph(event) {
    event?.preventDefault()
    setBusy(true)
    setError(null)
    try {
      const result = await getKnowledgeGraph(knowledgeBaseId, graphFilter)
      if (mounted.current) setGraph(result)
    } catch (err) {
      if (mounted.current) setError(requestError(err, '图谱加载失败'))
    } finally {
      if (mounted.current) setBusy(false)
    }
  }

  async function loadDocumentPage(offset) {
    setBusy(true)
    setError(null)
    try {
      await loadDetail({ preserveForms: true, offset })
    } catch (err) {
      if (mounted.current) setError(requestError(err, '资料列表加载失败'))
    } finally {
      if (mounted.current) setBusy(false)
    }
  }

  useEffect(() => {
    if (tab === 'graph' && knowledgeBaseId && !graph) loadGraph()
    // load once when the tab becomes visible; explicit searches call loadGraph.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab, knowledgeBaseId])

  const correctionBody = useMemo(() => {
    const common = { kind: correction.kind, expected_revision: base?.revision }
    if (correction.kind === 'rename_entity') return { ...common, entity_id: correction.entity_id.trim(), label: correction.label.trim() }
    if (correction.kind === 'merge_entities') return { ...common, entity_ids: correction.entity_ids, target_id: correction.target_id.trim() }
    if (correction.kind === 'delete_entity') return { ...common, entity_id: correction.entity_id.trim() }
    return { ...common, edge_id: correction.edge_id.trim() }
  }, [base?.revision, correction])

  const graphNodes = Array.isArray(graph?.nodes) ? graph.nodes.slice(0, 200) : []
  const graphNodeById = new Map(graphNodes.map(node => [node.id, node]))
  const graphEdges = (Array.isArray(graph?.edges) ? graph.edges : []).filter(edge => (
    graphNodeById.has(edge.source || edge.source_id)
    && graphNodeById.has(edge.target || edge.target_id)
  )).slice(0, 400)

  function selectGraphNode(node) {
    setGraphFilter(value => ({ ...value, focus: node.id }))
    setCorrection(value => {
      if (value.kind === 'merge_entities') {
        if (node.id === value.target_id) return value
        const entityIds = value.entity_ids.includes(node.id)
          ? value.entity_ids
          : [...value.entity_ids, node.id]
        return { ...value, entity_ids: entityIds }
      }
      return { ...value, entity_id: node.id }
    })
  }

  function selectGraphEdge(edge) {
    setCorrection(value => ({ ...value, kind: 'delete_relation', edge_id: edge.id }))
  }

  async function openSource(sourceVersionId) {
    setError(null)
    try {
      setSource(await getKnowledgeSource(knowledgeBaseId, sourceVersionId))
    } catch (err) {
      setError(requestError(err, '来源加载失败'))
    }
  }

  if (capabilities === null) return <div className="knowledge-page"><p className="kb-loading">正在检查知识库功能…</p></div>
  if (!capabilities.enabled || !capabilities.available) {
    return (
      <div className="knowledge-page">
        <header className="page-header"><h1 className="page-title">知识库</h1></header>
        <section className="kb-unavailable" role="status">
          <h2>知识库功能当前不可用</h2>
          <p>已有文档、学习路径和练习不受影响。服务启用后，此入口会自动出现。</p>
          <Link className="btn-primary kb-inline-link" to="/documents">返回文档管理</Link>
        </section>
      </div>
    )
  }

  if (!knowledgeBaseId) {
    return (
      <div className="knowledge-page">
        <header className="page-header"><h1 className="page-title">知识库</h1><p className="page-desc">按主题隔离资料，在同一知识库中检索、浏览图谱并进行可核验问答。</p></header>
        {error && <div className="error-banner" role="alert">{error}</div>}
        {notice && <div className="kb-notice" role="status">{notice}</div>}
        <form className="kb-create-card" onSubmit={handleCreate}>
          <h2>创建知识库</h2>
          <label>知识库名称<input aria-label="知识库名称" required maxLength={120} value={createForm.name} onChange={event => setCreateForm(form => ({ ...form, name: event.target.value }))} /></label>
          <label>知识库说明<textarea aria-label="知识库说明" rows={2} maxLength={1000} value={createForm.description} onChange={event => setCreateForm(form => ({ ...form, description: event.target.value }))} /></label>
          <button className="btn-primary" disabled={busy || !createForm.name.trim()}>创建知识库</button>
        </form>
        <section className="kb-list-section"><h2>我的知识库</h2>
          {bases.length ? <div className="kb-grid">{bases.map(item => (
            <Link className="kb-card" key={item.id} to={`/knowledge-bases/${encodeURIComponent(item.id)}`}>
              <span className="kb-card-title">{item.name}</span><span>{item.description || '暂无说明'}</span>
              <span className={`kb-status status-${item.status}`}>{statusLabel(item.status)} · {item.document_count} 份资料</span>
            </Link>
          ))}</div> : <p className="kb-empty">还没有知识库。先创建一个主题空间。</p>}
        </section>
      </div>
    )
  }

  if (!base) return <div className="knowledge-page"><p className="kb-loading">正在加载知识库…</p>{error && <div className="error-banner" role="alert">{error}</div>}</div>

  return (
    <div className="knowledge-page">
      <header className="page-header kb-detail-header">
        <div><Link className="kb-back" to="/knowledge-bases">← 全部知识库</Link><h1 className="page-title">{base.name}</h1><p className="page-desc">{base.description || '暂无说明'}</p></div>
        <span className={`kb-status status-${base.status}`}>{statusLabel(base.status)} · 版本 {base.revision}</span>
      </header>
      {error && <div className="error-banner" role="alert">{error}</div>}
      {notice && <div className="kb-notice" role="status">{notice}</div>}
      {job && ['queued', 'running'].includes(job.status) && <div className="kb-job" role="status">正在处理：{job.stage || job.status}</div>}
      {job?.status === 'failed' && <div className="kb-job-failed" role="alert"><span>后台任务失败：{job.error_code || '未知错误'}</span><button type="button" className="btn-secondary" disabled={busy} onClick={handleRetryJob}>重试任务</button></div>}

      <section className={`kb-rebuild-card ${base.status === 'dirty' ? 'kb-rebuild-needed' : ''}`}>
        <div><strong>{base.status === 'dirty' ? '索引需要恢复' : '重建知识库索引'}</strong><p>使用当前资料和已保存的图谱纠错重新建立索引。</p></div>
        <button
          type="button"
          className={base.status === 'dirty' ? 'btn-primary' : 'btn-secondary'}
          disabled={busy || ['updating', 'deleting'].includes(base.status)}
          onClick={() => runMutation(
            () => rebuildKnowledgeBase(base.id, base.revision, createIdempotencyKey()),
            '索引已重建',
          )}
        >
          重建索引
        </button>
      </section>

      <div className="kb-tabs" role="tablist" aria-label="知识库详情">
        {[['documents', '资料'], ['graph', '知识图谱'], ['ask', '问知识库']].map(([value, label]) => <button key={value} type="button" role="tab" aria-selected={tab === value} onClick={() => setTab(value)}>{label}</button>)}
      </div>

      {tab === 'documents' && <section className="kb-panel">
        <div className="kb-toolbar">
          <label className="btn-primary kb-file-button">上传资料<input aria-label="上传资料" type="file" onChange={handleUpload} disabled={busy || base.status !== 'ready'} /></label>
          <div className="kb-import"><label>复制已有文档<select aria-label="已有文档" value={legacyId} onChange={event => setLegacyId(event.target.value)}><option value="">请选择</option>{legacyDocuments.map(id => <option key={id}>{id}</option>)}</select></label><button type="button" className="btn-secondary" disabled={busy || !legacyId} onClick={() => runMutation(() => importLegacyDocument(base.id, legacyId, base.revision, createIdempotencyKey()), `${legacyId} 已复制到知识库`)}>复制</button></div>
        </div>
        <ul className="kb-document-list">{documents.map(document => <li key={document.id}><div><strong>{document.name}</strong><span>{documentKindLabel(document.kind)} · {documentStatusLabel(document.status)}</span></div><div className="kb-document-actions"><label className="btn-secondary kb-file-button">替换资料<input aria-label={`替换资料 ${document.name}`} type="file" onChange={event => handleReplacement(document, event)} disabled={busy || base.status !== 'ready'} /></label><button type="button" className="btn-danger-text" onClick={() => setConfirm({ type: 'document', value: document })}>删除资料</button></div></li>)}</ul>
        {documentPage.total > 0 && <nav className="kb-pagination" aria-label="资料分页"><span>共 {documentPage.total} 份资料</span><span>第 {Math.floor(documentPage.offset / documentPage.limit) + 1} / {Math.ceil(documentPage.total / documentPage.limit)} 页</span><div><button type="button" className="btn-secondary" disabled={busy || documentPage.offset === 0} onClick={() => loadDocumentPage(Math.max(0, documentPage.offset - documentPage.limit))}>上一页</button><button type="button" className="btn-secondary" disabled={busy || documentPage.offset + documentPage.limit >= documentPage.total} onClick={() => loadDocumentPage(documentPage.offset + documentPage.limit)}>下一页</button></div></nav>}
        {documentPage.total === 0 && <p className="kb-empty">尚无资料。上传文件，或显式复制一份已有文档。</p>}
        <form className="kb-rename" onSubmit={event => { event.preventDefault(); runMutation(() => updateKnowledgeBase(base.id, { name: rename.trim(), expected_revision: base.revision }, createIdempotencyKey()), '知识库已重命名') }}><label>知识库名称<input value={rename} onChange={event => setRename(event.target.value)} /></label><button className="btn-secondary" disabled={busy || !rename.trim()}>保存名称</button></form>
        <button type="button" className="btn-danger-text kb-delete-base" onClick={() => setConfirm({ type: 'base', value: base })}>删除知识库</button>
      </section>}

      {tab === 'graph' && <section className="kb-panel">
        <form className="graph-filter" onSubmit={loadGraph}><label>搜索实体<input value={graphFilter.search} onChange={event => setGraphFilter(value => ({ ...value, search: event.target.value }))} /></label><label>聚焦实体<select aria-label="聚焦实体" value={graphFilter.focus} onChange={event => setGraphFilter(value => ({ ...value, focus: event.target.value }))}><option value="">显示全部已加载实体</option>{graphNodes.map(node => <option key={node.id} value={node.id}>{entityDisplayName(node)}</option>)}</select></label><button className="btn-secondary" disabled={busy}>查看图谱</button></form>
        {graph?.truncated && <p className="kb-warning" role="status">图谱较大，当前仅显示前 200 个实体和 400 条关系。</p>}
        <GraphView graph={graph} onSource={openSource} onSelectNode={selectGraphNode} onSelectEdge={selectGraphEdge} />
        <form className="correction-form" onSubmit={event => { event.preventDefault(); runMutation(() => createKnowledgeCorrection(base.id, correctionBody, createIdempotencyKey()), '图谱纠错已应用') }}>
          <h2>纠正图谱</h2>
          <label>纠错类型<select aria-label="纠错类型" value={correction.kind} onChange={event => setCorrection(value => ({ ...value, kind: event.target.value }))}><option value="rename_entity">重命名实体</option><option value="merge_entities">合并实体</option><option value="delete_entity">删除实体</option><option value="delete_relation">删除关系</option></select></label>
          {['rename_entity', 'delete_entity'].includes(correction.kind) && <label>实体<select aria-label="纠错实体" required value={correction.entity_id} onChange={event => setCorrection(value => ({ ...value, entity_id: event.target.value }))}><option value="">请选择实体</option>{graphNodes.map(node => <option key={node.id} value={node.id}>{entityDisplayName(node)}</option>)}</select></label>}
          {correction.kind === 'rename_entity' && <label>新名称<input aria-label="新名称" required value={correction.label} onChange={event => setCorrection(value => ({ ...value, label: event.target.value }))} /></label>}
          {correction.kind === 'merge_entities' && <><label>合并来源实体<select aria-label="合并来源实体" multiple required value={correction.entity_ids} onChange={event => setCorrection(value => ({ ...value, entity_ids: [...event.target.selectedOptions].map(option => option.value) }))}>{graphNodes.filter(node => node.id !== correction.target_id).map(node => <option key={node.id} value={node.id}>{entityDisplayName(node)}</option>)}</select></label><label>合并后的目标实体<select aria-label="合并后的目标实体" required value={correction.target_id} onChange={event => setCorrection(value => ({ ...value, target_id: event.target.value, entity_ids: value.entity_ids.filter(id => id !== event.target.value) }))}><option value="">请选择目标实体</option>{graphNodes.map(node => <option key={node.id} value={node.id}>{entityDisplayName(node)}</option>)}</select></label></>}
          {correction.kind === 'delete_relation' && <label>关系<select aria-label="纠错关系" required value={correction.edge_id} onChange={event => setCorrection(value => ({ ...value, edge_id: event.target.value }))}><option value="">请选择关系</option>{graphEdges.map(edge => { const sourceNode = graphNodeById.get(edge.source || edge.source_id); const targetNode = graphNodeById.get(edge.target || edge.target_id); return <option key={edge.id} value={edge.id}>{sourceNode?.label || '未命名实体'} · {edge.label || edge.relation || '相关'} · {targetNode?.label || '未命名实体'}</option> })}</select></label>}
          <button className="btn-primary" disabled={busy}>提交纠错</button>
        </form>
      </section>}

      {tab === 'ask' && <section className="kb-panel kb-ask-panel"><h2>基于当前知识库提问</h2><p>问答会固定当前知识库范围，并强制使用可核验来源。联网补充默认关闭，可在问答页显式开启。</p><Link className="btn-primary kb-inline-link" to={`/autonomous?knowledge_base_id=${encodeURIComponent(base.id)}`}>打开知识库问答</Link></section>}

      {source && <aside className="source-drawer" aria-label="来源详情"><button type="button" aria-label="关闭来源详情" onClick={() => setSource(null)}>×</button><h2>{source.title || source.name || '来源详情'}</h2>{source.source_status && <span className={`source-status source-status-${source.source_status}`}>{SOURCE_STATUS[source.source_status] || source.source_status}</span>}<p className="source-body-text">{boundedSourceText(source.text) || (source.source_status === 'deleted' ? '来源已删除，原文不再保留。' : '暂无可显示的来源正文。')}</p>{locatorText(source.locator) && <p className="source-locator">{locatorText(source.locator)}</p>}{Array.isArray(source.blocks) && source.blocks.slice(0, 20).map((block, index) => { const text = boundedSourceText(typeof block === 'string' ? block : block?.text, 2000); if (!text) return null; return <section className="source-block" key={block?.id || index}><p>{text}</p>{locatorText(block?.locator) && <small>{locatorText(block.locator)}</small>}</section> })}</aside>}

      {confirm?.type === 'base' && <ConfirmDialog title="删除知识库" busy={busy} onCancel={() => setConfirm(null)} onConfirm={handleDeleteBase}>这会永久删除这个知识库的全部资料：原始文件、解析文本、图谱索引和模型缓存。文档库里的材料和学习记录不受影响；已有对话里引用过的片段仍保留在对话中。</ConfirmDialog>}
      {confirm?.type === 'document' && <ConfirmDialog title="删除知识库资料" busy={busy} onCancel={() => setConfirm(null)} onConfirm={async () => { const ok = await runMutation(() => deleteKnowledgeDocument(base.id, confirm.value.id, base.revision, createIdempotencyKey()), `${confirm.value.name} 已删除`); if (ok) setConfirm(null) }}>资料的原始文件、解析文本（含历史版本）和索引会被永久删除。已有回答里的引用会显示为“来源已删除”，对话中已引用的片段仍会保留。</ConfirmDialog>}
    </div>
  )
}
