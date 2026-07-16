import { useState, useEffect, useCallback, useRef } from 'react'
import { uploadDocument, getDocuments, deleteDocument } from '../api/client'
import './Documents.css'

/* ═══════════════════════════════════════════════════════════════════
   Documents Page — 文档上传 + 文档列表管理
   ═══════════════════════════════════════════════════════════════════ */

export default function Documents() {
  const [documents, setDocuments] = useState([])
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(null) // {filename, status}
  const [dragOver, setDragOver] = useState(false)
  const [deleting, setDeleting] = useState(null) // document_id being deleted
  const [error, setError] = useState(null)
  const [listError, setListError] = useState(null)
  const fileInputRef = useRef(null)

  /* ── 加载文档列表 ──────────────────────────────────────────────── */
  const fetchDocuments = useCallback(async () => {
    setLoading(true)
    setListError(null)
    try {
      const data = await getDocuments()
      setDocuments(data.documents || [])
    } catch (err) {
      setListError(err.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchDocuments() }, [fetchDocuments])

  /* ── 上传处理 ──────────────────────────────────────────────────── */
  const handleUpload = async (file) => {
    if (!file) return

    const validTypes = [
      '.pdf', '.docx', '.txt', '.md',
      '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif',
    ]
    const ext = '.' + file.name.split('.').pop().toLowerCase()
    if (!validTypes.includes(ext)) {
      setError(`不支持的文件格式 "${ext}"，请上传 PDF、DOCX、文本或常见图片文件`)
      return
    }

    setUploading(true)
    setUploadProgress({ filename: file.name, status: 'uploading' })
    setError(null)

    try {
      const result = await uploadDocument(file)
      setUploadProgress({
        filename: file.name,
        status: 'done',
        chunks: result.chunks,
      })
      // 1.5s 后清除进度提示并刷新列表
      setTimeout(() => {
        setUploadProgress(null)
        setUploading(false)
        fetchDocuments()
      }, 1500)
    } catch (err) {
      setUploadProgress({ filename: file.name, status: 'error' })
      setError(err.message)
      setUploading(false)
    }
  }

  /* ── 拖拽事件 ──────────────────────────────────────────────────── */
  const handleDragOver = (e) => {
    e.preventDefault()
    setDragOver(true)
  }
  const handleDragLeave = (e) => {
    e.preventDefault()
    setDragOver(false)
  }
  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    handleUpload(file)
  }

  /* ── 删除处理 ──────────────────────────────────────────────────── */
  const handleDelete = async (docId) => {
    if (deleting) return
    setDeleting(docId)
    setError(null)

    try {
      await deleteDocument(docId)
      // 等动画播完再移除
      setTimeout(() => {
        setDocuments(prev => prev.filter(d => d !== docId))
        setDeleting(null)
      }, 350)
    } catch (err) {
      setError(err.message)
      setDeleting(null)
    }
  }

  /* ── 文件名 → 图标映射 ────────────────────────────────────────── */
  const getFileIcon = (name) => {
    if (name.endsWith('.md')) return '📝'
    if (/\.(png|jpe?g|gif|bmp|tiff?)$/i.test(name)) return '🖼️'
    if (name.endsWith('.pdf')) return '📕'
    if (name.endsWith('.docx')) return '📘'
    return '📄'
  }

  return (
    <div className="documents-page">
      {/* ── 页面标题 ────────────────────────────────────────────────── */}
      <header className="page-header">
        <div>
          <h1 className="page-title">文档管理</h1>
          <p className="page-desc">上传学习材料，AI 会自动分析并向量化存储，为后续出题和学习路径生成做准备。</p>
        </div>
        <div className="header-stats">
          <div className="stat-card">
            <span className="stat-value">{documents.length}</span>
            <span className="stat-label">篇文档</span>
          </div>
        </div>
      </header>

      {/* ── 上传区域 ────────────────────────────────────────────────── */}
      <section
        className={`upload-zone ${dragOver ? 'drag-over' : ''} ${uploading ? 'uploading' : ''}`}
        role="button"
        tabIndex={uploading ? -1 : 0}
        aria-label="选择要上传的学习材料"
        aria-disabled={uploading}
        aria-busy={uploading}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !uploading && fileInputRef.current?.click()}
        onKeyDown={(event) => {
          if (!uploading && (event.key === 'Enter' || event.key === ' ')) {
            event.preventDefault()
            fileInputRef.current?.click()
          }
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.docx,.txt,.md,.png,.jpg,.jpeg,.gif,.bmp,.tiff,.tif"
          onChange={(event) => {
            handleUpload(event.target.files?.[0])
            event.target.value = ''
          }}
          style={{ display: 'none' }}
        />

        {uploadProgress ? (
          <div className="upload-progress" role="status" aria-live="polite">
            <div className={`progress-icon ${uploadProgress.status}`}>
              {uploadProgress.status === 'uploading' && (
                <svg className="spinner" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
                </svg>
              )}
              {uploadProgress.status === 'done' && <span className="check-mark">&#10003;</span>}
              {uploadProgress.status === 'error' && <span className="error-mark">!</span>}
            </div>
            <p className="progress-filename">{uploadProgress.filename}</p>
            <p className="progress-status">
              {uploadProgress.status === 'uploading' && '正在上传并向量化处理...'}
              {uploadProgress.status === 'done' && `完成，共 ${uploadProgress.chunks} 个文本块`}
              {uploadProgress.status === 'error' && '上传失败'}
            </p>
          </div>
        ) : (
          <div className="upload-idle">
            <div className="upload-icon-wrapper">
              <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="17 8 12 3 7 8"/>
                <line x1="12" y1="3" x2="12" y2="15"/>
              </svg>
            </div>
            <p className="upload-title">拖拽文件到此处，或点击选择</p>
            <p className="upload-hint">支持 PDF / DOCX / TXT / Markdown 和常见图片</p>
          </div>
        )}

        {/* 拖拽时的装饰边框 */}
        <div className="upload-border" />
      </section>

      {/* ── 错误提示 ────────────────────────────────────────────────── */}
      {error && (
        <div className="error-banner" role="alert">
          <span className="error-icon">&#9888;</span>
          <span>{error}</span>
          <button className="error-close" aria-label="关闭错误提示" onClick={() => setError(null)}>&times;</button>
        </div>
      )}

      {/* ── 文档列表 ────────────────────────────────────────────────── */}
      <section className="documents-section">
        <h2 className="section-title">已上传文档</h2>

        {loading ? (
          <div className="doc-list-skeleton">
            {[1, 2, 3].map(i => (
              <div key={i} className="skeleton-card" style={{ animationDelay: `${i * 0.1}s` }} />
            ))}
          </div>
        ) : listError ? (
          <div className="load-error-state" role="alert">
            <p className="state-title">无法加载文档列表</p>
            <p className="state-desc">{listError}</p>
            <button type="button" className="state-action" onClick={fetchDocuments}>
              重新加载
            </button>
          </div>
        ) : documents.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round">
                <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/>
                <polyline points="14 2 14 8 20 8"/>
                <line x1="16" y1="13" x2="8" y2="13"/>
                <line x1="16" y1="17" x2="8" y2="17"/>
                <polyline points="10 9 9 9 8 9"/>
              </svg>
            </div>
            <p className="empty-title">还没有文档</p>
            <p className="empty-desc">上传你的第一份学习材料开始吧</p>
          </div>
        ) : (
          <div className="doc-grid">
            {documents.map((docId, idx) => (
              <div
                key={docId}
                className={`doc-card ${deleting === docId ? 'deleting' : ''}`}
                style={{ animationDelay: `${idx * 0.06}s` }}
              >
                {/* 书脊装饰 */}
                <div className="card-spine" />

                <div className="card-body">
                  <div className="card-icon">{getFileIcon(docId)}</div>
                  <div className="card-info">
                    <h3 className="card-name" title={docId}>{docId}</h3>
                    <p className="card-meta">已向量化 · 可用于出题</p>
                  </div>
                  <button
                    className="card-delete"
                    aria-label={`删除文档 ${docId}`}
                    onClick={(e) => {
                      e.stopPropagation()
                      handleDelete(docId)
                    }}
                    title="删除文档"
                    disabled={deleting === docId}
                  >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="3 6 5 6 21 6"/>
                      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                    </svg>
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  )
}
