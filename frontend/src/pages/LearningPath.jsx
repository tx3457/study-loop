import { useState, useEffect, useCallback } from 'react'
import DocumentPrerequisite from '../components/DocumentPrerequisite'
import { getDocuments, generateLearningPath } from '../api/client'
import './LearningPath.css'

/* ═══════════════════════════════════════════════════════════════════
   Learning Path Page — AI 学习路径生成 + 展示
   ═══════════════════════════════════════════════════════════════════ */

export default function LearningPath() {
  const [documents, setDocuments] = useState([])
  const [selectedDoc, setSelectedDoc] = useState('')
  const [path, setPath] = useState(null)
  const [loading, setLoading] = useState(false)
  const [docsLoading, setDocsLoading] = useState(true)
  const [docsError, setDocsError] = useState(null)
  const [error, setError] = useState(null)

  const loadDocuments = useCallback(async () => {
    setDocsLoading(true)
    setDocsError(null)
    try {
      const data = await getDocuments()
      const nextDocuments = data.documents || []
      setDocuments(nextDocuments)
      setSelectedDoc(current => nextDocuments.includes(current) ? current : '')
    } catch (err) {
      setDocsError(err.message)
    } finally {
      setDocsLoading(false)
    }
  }, [])

  useEffect(() => { loadDocuments() }, [loadDocuments])

  const handleDocumentChange = (event) => {
    setSelectedDoc(event.target.value)
    setPath(null)
    setError(null)
  }

  const handleGenerate = async () => {
    if (!selectedDoc) return
    setLoading(true)
    setError(null)
    setPath(null)

    try {
      const result = await generateLearningPath(selectedDoc)
      setPath(result)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const totalMinutes = path?.stages.reduce((sum, s) => sum + s.estimated_minutes, 0) || 0

  return (
    <div className="lp-page">
      {/* ── 页面标题 ──────────────────────────────────────────────────── */}
      <header className="page-header">
        <h1 className="page-title">学习路径</h1>
        <p className="page-desc">选择文档，AI 将分析内容并生成分阶段学习计划。</p>
      </header>

      {/* ── 文档选择 + 生成按钮 ────────────────────────────────────────── */}
      {docsError ? (
        <div className="load-error-state" role="alert">
          <p className="state-title">无法加载文档列表</p>
          <p className="state-desc">{docsError}</p>
          <button type="button" className="state-action" onClick={loadDocuments}>
            重新加载文档
          </button>
        </div>
      ) : !docsLoading && documents.length === 0 ? (
        <DocumentPrerequisite description="生成学习路径前，需要先上传一份已经完成解析的学习材料。" />
      ) : (
        <div className="lp-controls">
          <select
            className="lp-select"
            aria-label="学习文档"
            value={selectedDoc}
            onChange={handleDocumentChange}
            disabled={docsLoading || loading}
          >
            <option value="">
              {docsLoading ? '加载文档列表...' : '-- 选择文档 --'}
            </option>
            {documents.map(doc => (
              <option key={doc} value={doc}>{doc}</option>
            ))}
          </select>

          <button
            type="button"
            className="lp-generate-btn"
            onClick={handleGenerate}
            disabled={!selectedDoc || loading}
          >
            {loading ? (
              <>
                <span className="btn-spinner" />
                AI 分析中...
              </>
            ) : (
              <>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
                </svg>
                生成学习路径
              </>
            )}
          </button>
        </div>
      )}

      {/* ── 错误 ──────────────────────────────────────────────────────── */}
      {error && (
        <div className="error-banner" role="alert">
          <span>&#9888;</span>
          <span>{error}</span>
          <button className="error-close" aria-label="关闭错误提示" onClick={() => setError(null)}>&times;</button>
        </div>
      )}

      {/* ── 加载骨架 ──────────────────────────────────────────────────── */}
      {loading && (
        <div className="lp-skeleton" aria-busy="true">
          <div className="lp-loading-copy" role="status" aria-live="polite">
            <strong>正在整理学习路径</strong>
            <span>正在分析材料、检索重点并组织阶段，通常需要约 1 分钟。</span>
          </div>
          {[1, 2, 3].map(i => (
            <div key={i} className="skeleton-stage" style={{ animationDelay: `${i * 0.15}s` }}>
              <div className="skeleton-circle" />
              <div className="skeleton-lines">
                <div className="skeleton-line w60" />
                <div className="skeleton-line w80" />
                <div className="skeleton-line w40" />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── 学习路径展示 ──────────────────────────────────────────────── */}
      {path && !loading && (
        <div className="lp-result">
          {/* 路径标题 */}
          <div className="lp-header-card">
            <h2 className="lp-path-title">{path.title}</h2>
            <div className="lp-meta-row">
              <span className="lp-meta-item">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
                {path.total_stages} 个阶段
              </span>
              <span className="lp-meta-item">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                预计 {totalMinutes} 分钟
              </span>
            </div>
          </div>

          {/* 时间线 */}
          <div className="lp-timeline">
            {path.stages.map((stage, idx) => (
              <div
                key={stage.stage}
                className="timeline-item"
                style={{ animationDelay: `${idx * 0.1}s` }}
              >
                {/* 左侧节点 + 连线 */}
                <div className="timeline-track">
                  <div className="timeline-node">
                    <span className="node-number">{stage.stage}</span>
                  </div>
                  {idx < path.stages.length - 1 && <div className="timeline-line" />}
                </div>

                {/* 右侧卡片 */}
                <div className="timeline-card">
                  <div className="tc-header">
                    <h3 className="tc-title">{stage.title}</h3>
                    <span className="tc-time">
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                      {stage.estimated_minutes} min
                    </span>
                  </div>
                  <p className="tc-desc">{stage.description}</p>
                  <div className="tc-topics">
                    {stage.topics.map((topic, i) => (
                      <span key={i} className="topic-tag">{topic}</span>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
