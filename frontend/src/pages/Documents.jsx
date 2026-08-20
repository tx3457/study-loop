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
  const [deleteCandidate, setDeleteCandidate] = useState(null)
  const [deleteError, setDeleteError] = useState(null)
  const [statusMessage, setStatusMessage] = useState('')
  const [error, setError] = useState(null)
  const [listError, setListError] = useState(null)
  const fileInputRef = useRef(null)
  const uploadingRef = useRef(false)
  const mutationRef = useRef(null)
  const documentsRef = useRef([])
  const listRequestId = useRef(0)
  const mountedRef = useRef(true)
  const uploadTimerRef = useRef(null)
  const deleteTriggerRef = useRef(null)
  const deleteDialogRef = useRef(null)
  const deleteCancelRef = useRef(null)
  const documentsSectionRef = useRef(null)

  /* ── 加载文档列表 ──────────────────────────────────────────────── */
  const fetchDocuments = useCallback(async ({ background = false, apply = true } = {}) => {
    const requestId = ++listRequestId.current
    if (!background) setLoading(true)
    if (apply) setListError(null)
    try {
      const data = await getDocuments()
      const nextDocuments = Array.isArray(data?.documents) ? data.documents : null
      if (!nextDocuments) throw new Error('文档列表响应格式无效')
      if (!mountedRef.current || requestId !== listRequestId.current) return null
      if (apply) {
        documentsRef.current = nextDocuments
        setDocuments(nextDocuments)
      }
      return nextDocuments
    } catch (err) {
      if (mountedRef.current && requestId === listRequestId.current && apply) {
        setListError(err.message)
      }
      return null
    } finally {
      if (
        mountedRef.current
        && requestId === listRequestId.current
        && (!background || apply)
      ) {
        setLoading(false)
      }
    }
  }, [])

  useEffect(() => {
    mountedRef.current = true
    fetchDocuments()
    return () => {
      mountedRef.current = false
      listRequestId.current += 1
      clearTimeout(uploadTimerRef.current)
    }
  }, [fetchDocuments])

  /* ── 上传处理 ──────────────────────────────────────────────────── */
  const handleUpload = async (file) => {
    if (
      !file
      || loading
      || listError
      || uploadingRef.current
      || mutationRef.current
      || deleteCandidate
    ) return

    const validTypes = [
      '.pdf', '.docx', '.txt', '.md',
      '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif',
    ]
    const ext = '.' + file.name.split('.').pop().toLowerCase()
    if (!validTypes.includes(ext)) {
      setError(`不支持的文件格式 "${ext}"，请上传 PDF、DOCX、文本或常见图片文件`)
      return
    }

    const existedBefore = documentsRef.current.includes(file.name)
    const mutationId = `upload:${file.name}`
    mutationRef.current = mutationId
    uploadingRef.current = true
    listRequestId.current += 1
    clearTimeout(uploadTimerRef.current)
    setUploading(true)
    setUploadProgress({ filename: file.name, status: 'uploading' })
    setError(null)

    try {
      const result = await uploadDocument(file)
      if (!mountedRef.current || mutationRef.current !== mutationId) return
      const authoritative = await fetchDocuments({ background: true })
      if (!mountedRef.current || mutationRef.current !== mutationId) return
      if (authoritative && !authoritative.includes(file.name)) {
        setUploadProgress({ filename: file.name, status: 'error' })
        setError('服务端已响应上传，但权威文档列表尚未包含该材料；请重新加载后再决定是否重试')
        return
      }
      if (!authoritative) {
        const nextDocuments = documentsRef.current.includes(file.name)
          ? documentsRef.current
          : [...documentsRef.current, file.name]
        documentsRef.current = nextDocuments
        setDocuments(nextDocuments)
        setListError(null)
      }
      setUploadProgress({
        filename: file.name,
        status: 'done',
        chunks: result.chunks,
      })
      setStatusMessage(
        authoritative
          ? `材料 ${file.name} 已上传并建立索引`
          : `材料 ${file.name} 已上传；文档列表暂时无法重新核对，已按服务端确认结果更新`,
      )
    } catch (err) {
      if (!mountedRef.current || mutationRef.current !== mutationId) return
      const authoritative = await fetchDocuments({ background: true })
      if (!mountedRef.current || mutationRef.current !== mutationId) return
      const requestCouldHaveCommitted = !err.status || err.status >= 500
      if (
        requestCouldHaveCommitted
        && !existedBefore
        && authoritative?.includes(file.name)
      ) {
        setUploadProgress({ filename: file.name, status: 'done', synced: true })
        setStatusMessage(`材料 ${file.name} 的上传结果已从文档列表同步`)
      } else {
        setUploadProgress({ filename: file.name, status: 'error' })
        setError(
          err.status && err.status < 500
            ? err.message
            : authoritative === null
            ? '上传状态暂时无法确认，请重新加载文档列表后再决定是否重试'
            : err.message,
        )
      }
    } finally {
      if (mutationRef.current === mutationId) mutationRef.current = null
      uploadingRef.current = false
      if (mountedRef.current) {
        setUploading(false)
        clearTimeout(uploadTimerRef.current)
        uploadTimerRef.current = setTimeout(() => {
          if (mountedRef.current) setUploadProgress(null)
        }, 1500)
      }
    }
  }

  /* ── 拖拽事件 ──────────────────────────────────────────────────── */
  const handleDragOver = (e) => {
    e.preventDefault()
    if (loading || listError || uploadingRef.current || mutationRef.current || deleteCandidate) return
    setDragOver(true)
  }
  const handleDragLeave = (e) => {
    e.preventDefault()
    setDragOver(false)
  }
  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    if (loading || listError || uploadingRef.current || mutationRef.current || deleteCandidate) return
    const file = e.dataTransfer.files[0]
    handleUpload(file)
  }

  /* ── 删除处理 ──────────────────────────────────────────────────── */
  const requestDelete = (docId, trigger) => {
    if (mutationRef.current || deleteCandidate) return
    deleteTriggerRef.current = trigger
    setDeleteError(null)
    setDeleteCandidate(docId)
  }

  const closeDeleteDialog = useCallback((restoreTrigger = true, force = false) => {
    const deleteInFlight = mutationRef.current?.startsWith('delete:')
    if ((deleting || deleteInFlight) && !force) return
    setDeleteCandidate(null)
    setDeleteError(null)
    const requestedTarget = restoreTrigger
      ? deleteTriggerRef.current
      : documentsSectionRef.current
    const focusTarget = requestedTarget?.isConnected
      ? requestedTarget
      : documentsSectionRef.current
    deleteTriggerRef.current = null
    window.requestAnimationFrame(() => focusTarget?.focus())
  }, [deleting])

  const confirmDelete = async () => {
    const docId = deleteCandidate
    if (!docId || mutationRef.current) return
    const mutationId = `delete:${docId}`
    mutationRef.current = mutationId
    listRequestId.current += 1
    setDeleting(docId)
    setDeleteError(null)
    setError(null)

    let response = null
    let requestError = null
    const isValidDeleteResponse = value => (
      value?.status === 'material_deleted'
      && value.document_id === docId
      && value.scope === 'material_only'
      && value.learning_data_retained === true
      && value.document_id_reusable === false
    )
    try {
      response = await deleteDocument(docId)
    } catch (err) {
      requestError = err
    }

    if (response && !isValidDeleteResponse(response)) {
      requestError = new Error('删除响应格式无效，尚未确认材料状态')
      response = null
    }

    if (!response) {
      const observed = await fetchDocuments({ background: true, apply: false })
      if (observed && !observed.includes(docId)) {
        try {
          // The first response may have been lost, or the backend may have
          // stopped at its durable deleting tombstone. The same DELETE safely
          // acknowledges success or finishes that transition.
          response = await deleteDocument(docId)
        } catch (retryError) {
          requestError = retryError
        }
      }
    }

    if (response && !isValidDeleteResponse(response)) {
      requestError = new Error('删除响应格式无效，尚未确认材料状态')
      response = null
    }

    if (!mountedRef.current || mutationRef.current !== mutationId) return
    if (response) {
      const authoritative = await fetchDocuments({ background: true })
      if (!mountedRef.current || mutationRef.current !== mutationId) return
      if (authoritative?.includes(docId)) {
        setDeleteError('服务端已响应删除，但权威文档列表仍包含该材料；请再次确认以安全重试。')
        setDeleting(null)
        mutationRef.current = null
        return
      }
      if (!authoritative) {
        const nextDocuments = documentsRef.current.filter(item => item !== docId)
        documentsRef.current = nextDocuments
        setDocuments(nextDocuments)
        setListError(null)
      }
      setStatusMessage(
        authoritative
          ? `材料 ${docId} 已从检索索引删除；学习记录仍保留`
          : `材料 ${docId} 已删除；文档列表暂时无法重新核对，已按服务端确认结果更新`,
      )
      setDeleting(null)
      mutationRef.current = null
      closeDeleteDialog(false, true)
      return
    }

    const observed = await fetchDocuments({ background: true, apply: false })
    if (!mountedRef.current || mutationRef.current !== mutationId) return
    const materialHidden = observed !== null && !observed.includes(docId)
    if (observed !== null) {
      documentsRef.current = observed
      setDocuments(observed)
      setListError(null)
      setLoading(false)
    }
    if (materialHidden) {
      setStatusMessage(
        `材料 ${docId} 已从列表和新检索中隐藏，但删除收尾尚未确认；刷新列表时会继续安全重试`,
      )
    }
    setDeleteError(
      observed === null
        ? '删除状态尚未确认，且文档列表暂不可用；请再次确认以安全重试。'
        : materialHidden
          ? '删除状态尚未确认。材料已停止显示，请再次确认以安全重试。'
        : requestError?.message || '删除失败，请稍后重试',
    )
    setDeleting(null)
    mutationRef.current = null
  }

  useEffect(() => {
    if (!deleteCandidate) return undefined
    const focusFrame = window.requestAnimationFrame(() => {
      const target = deleteCancelRef.current?.disabled
        ? deleteDialogRef.current
        : deleteCancelRef.current
      target?.focus()
    })
    const handleKeyDown = (event) => {
      if (event.key === 'Escape' && !deleting) {
        event.preventDefault()
        closeDeleteDialog()
        return
      }
      if (event.key !== 'Tab') return
      const focusable = [...(deleteDialogRef.current?.querySelectorAll(
        'button:not(:disabled), [href], input:not(:disabled), [tabindex]:not([tabindex="-1"])',
      ) || [])]
      if (focusable.length === 0) {
        event.preventDefault()
        deleteDialogRef.current?.focus()
        return
      }
      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first.focus()
      }
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => {
      window.cancelAnimationFrame(focusFrame)
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [closeDeleteDialog, deleteCandidate, deleting])

  /* ── 文件名 → 图标映射 ────────────────────────────────────────── */
  const getFileIcon = (name) => {
    if (name.endsWith('.md')) return '📝'
    if (/\.(png|jpe?g|gif|bmp|tiff?)$/i.test(name)) return '🖼️'
    if (name.endsWith('.pdf')) return '📕'
    if (name.endsWith('.docx')) return '📘'
    return '📄'
  }

  const mutationBlocked = uploading || deleting !== null || deleteCandidate !== null
  const uploadBlocked = mutationBlocked || loading || Boolean(listError)

  return (
    <div className="documents-page">
      <div
        className="documents-content"
        aria-hidden={deleteCandidate ? 'true' : undefined}
        inert={deleteCandidate ? true : undefined}
      >
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
        tabIndex={uploadBlocked ? -1 : 0}
        aria-label="选择要上传的学习材料"
        aria-disabled={uploadBlocked}
        aria-busy={uploading}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !uploadBlocked && fileInputRef.current?.click()}
        onKeyDown={(event) => {
          if (!uploadBlocked && (event.key === 'Enter' || event.key === ' ')) {
            event.preventDefault()
            fileInputRef.current?.click()
          }
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.docx,.txt,.md,.png,.jpg,.jpeg,.gif,.bmp,.tiff,.tif"
          disabled={uploadBlocked}
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
              {uploadProgress.status === 'done' && (
                uploadProgress.synced
                  ? '完成，已从文档列表同步确认'
                  : `完成，共 ${uploadProgress.chunks} 个文本块`
              )}
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
      <p className="documents-status" role="status" aria-live="polite">
        {statusMessage}
      </p>

      <section ref={documentsSectionRef} className="documents-section" tabIndex="-1">
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
                      requestDelete(docId, e.currentTarget)
                    }}
                    title="删除文档"
                    disabled={mutationBlocked}
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

      {deleteCandidate && (
        <div className="document-delete-overlay">
          <div
            ref={deleteDialogRef}
            className="document-delete-dialog"
            role="alertdialog"
            aria-modal="true"
            aria-labelledby="document-delete-title"
            aria-describedby="document-delete-description"
            aria-busy={Boolean(deleting)}
            tabIndex="-1"
          >
            <p className="document-delete-eyebrow">仅删除材料</p>
            <h2 id="document-delete-title">确认删除“{deleteCandidate}”？</h2>
            <div id="document-delete-description" className="document-delete-copy">
              <p>StudyLoop 会删除该材料的检索索引，之后不能再发起新的原文检索；你电脑上的原文件不受影响。</p>
              <p>学习历史、学习画像、错题以及已保存会话中的题目、引用片段等工件仍会保留，也可能继续显示或被原会话使用。这不是隐私数据彻底清除。</p>
              <p>为避免旧学习记录与新内容混淆，该文件名会被保留，之后如需重新上传，请先重命名文件。</p>
            </div>
            {deleteError && <p className="document-delete-error" role="alert">{deleteError}</p>}
            <div className="document-delete-actions">
              <button
                ref={deleteCancelRef}
                type="button"
                className="document-delete-cancel"
                onClick={() => closeDeleteDialog()}
                disabled={Boolean(deleting)}
              >
                取消
              </button>
              <button
                type="button"
                className="document-delete-confirm"
                onClick={confirmDelete}
                disabled={Boolean(deleting)}
              >
                {deleting ? '正在删除材料…' : '确认仅删除材料'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
