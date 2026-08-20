import { useCallback, useEffect, useRef, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import DocumentPrerequisite from '../components/DocumentPrerequisite'
import {
  createIdempotencyKey,
  createLearningPathResource,
  getCurrentLearningPathResource,
  getDocuments,
  getLearningPathResource,
} from '../api/client'
import {
  bindLearningPathRecovery,
  clearLearningPathRecovery,
  createLearningPathRecovery,
  isLearningPathId,
  normalizeLearningPathResource,
  readLearningPathRecovery,
  writeLearningPathRecovery,
} from '../state/learningPathRecovery'
import {
  abortRecoveryRequests,
  runRecoveryRequest,
  startRecoveryRequest,
} from '../state/recoveryRequest'
import './LearningPath.css'

function recoveryDescriptor(queryPathId) {
  const recovery = readLearningPathRecovery()
  if (queryPathId) {
    return {
      kind: 'get',
      identity: `get:${queryPathId}`,
      pathId: queryPathId,
      recovery: recovery?.learning_path_id === queryPathId ? recovery : null,
    }
  }
  if (!recovery) {
    return {
      kind: 'current',
      identity: 'current:default_user',
      recovery: null,
    }
  }
  if (recovery.learning_path_id) {
    return {
      kind: 'get',
      identity: `get:${recovery.learning_path_id}`,
      pathId: recovery.learning_path_id,
      recovery,
    }
  }
  return {
    kind: 'create',
    identity: `create:${recovery.start_idempotency_key}`,
    recovery,
  }
}

function normalizeExpectedResource(
  value,
  expectedDocumentId = null,
  expectedLearningPathId = null,
) {
  const resource = normalizeLearningPathResource(value)
  if (!resource || (
    expectedDocumentId
    && resource.path.document_id !== expectedDocumentId
  ) || (
    expectedLearningPathId
    && resource.learning_path_id !== expectedLearningPathId
  )) {
    throw new Error('服务端返回的学习路径格式无效，请稍后重试')
  }
  return resource
}

function isTerminalCreateError(error) {
  return [400, 404, 410, 413, 422].includes(error?.status)
    || (
      error?.status === 409
      && error?.reason === 'payload_mismatch'
    )
}

export default function LearningPath() {
  const navigate = useNavigate()
  const location = useLocation()
  const queryPathId = new URLSearchParams(location.search).get('path_id')
  const queryPathIdValid = queryPathId == null || isLearningPathId(queryPathId)

  const [documents, setDocuments] = useState([])
  const [selectedDoc, setSelectedDoc] = useState('')
  const [resource, setResource] = useState(null)
  const [generating, setGenerating] = useState(false)
  const [recovering, setRecovering] = useState(false)
  const [docsStatus, setDocsStatus] = useState('loading')
  const [docsError, setDocsError] = useState(null)
  const [error, setError] = useState(null)
  const [canRetryRecovery, setCanRetryRecovery] = useState(false)
  const [recoveryNonce, setRecoveryNonce] = useState(0)

  const mountedRef = useRef(false)
  const operationEpochRef = useRef(0)
  const documentsEpochRef = useRef(0)
  const documentsFlightRef = useRef(null)
  const recoveryFlightRef = useRef(null)
  const activeRecoveryControllers = useRef(new Set())
  const actionInFlightRef = useRef(false)
  const resourceRef = useRef(null)

  const publishResource = useCallback((nextResource) => {
    resourceRef.current = nextResource
    setResource(nextResource)
    setSelectedDoc(nextResource.path.document_id)
  }, [])

  useEffect(() => {
    resourceRef.current = resource
  }, [resource])

  useEffect(() => {
    const recoveryControllers = activeRecoveryControllers.current
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      queueMicrotask(() => {
        if (!mountedRef.current) {
          operationEpochRef.current += 1
          documentsEpochRef.current += 1
          abortRecoveryRequests(recoveryControllers)
          recoveryFlightRef.current = null
          actionInFlightRef.current = false
        }
      })
    }
  }, [])

  const loadDocuments = useCallback(async () => {
    const epoch = ++documentsEpochRef.current
    setDocsStatus('loading')
    setDocsError(null)

    let flight = documentsFlightRef.current
    if (!flight) {
      const promise = getDocuments()
      flight = { promise }
      documentsFlightRef.current = flight
      promise.finally(() => {
        if (documentsFlightRef.current === flight) {
          documentsFlightRef.current = null
        }
      }).catch(() => {})
    }

    try {
      const data = await flight.promise
      if (!mountedRef.current || epoch !== documentsEpochRef.current) return
      if (!data || !Array.isArray(data.documents)) {
        throw new Error('文档列表响应格式无效')
      }
      const nextDocuments = data.documents.filter(doc => typeof doc === 'string')
      setDocuments(nextDocuments)
      setSelectedDoc(current => (
        current || resourceRef.current?.path.document_id || ''
      ))
      setDocsStatus('ready')
    } catch (err) {
      if (!mountedRef.current || epoch !== documentsEpochRef.current) return
      setDocuments([])
      setDocsStatus('error')
      setDocsError(err.message)
    }
  }, [])

  useEffect(() => {
    loadDocuments()
  }, [loadDocuments])

  useEffect(() => {
    if (!queryPathIdValid) {
      operationEpochRef.current += 1
      abortRecoveryRequests(activeRecoveryControllers.current)
      recoveryFlightRef.current = null
      actionInFlightRef.current = false
      setGenerating(false)
      setError('学习路径链接无效')
      setCanRetryRecovery(false)
      setRecovering(false)
      return
    }
    if (actionInFlightRef.current) {
      if (!queryPathId) {
        setRecovering(false)
        return
      }
      operationEpochRef.current += 1
      abortRecoveryRequests(activeRecoveryControllers.current)
      recoveryFlightRef.current = null
      actionInFlightRef.current = false
      setGenerating(false)
    }
    if (queryPathId && resourceRef.current?.learning_path_id === queryPathId) {
      operationEpochRef.current += 1
      abortRecoveryRequests(activeRecoveryControllers.current)
      recoveryFlightRef.current = null
      setRecovering(false)
      return
    }

    const descriptor = recoveryDescriptor(queryPathId)
    let active = true
    const epoch = ++operationEpochRef.current
    if (descriptor.kind === 'create') {
      setSelectedDoc(descriptor.recovery.intent.document_id)
    }
    setRecovering(true)
    setError(null)
    setCanRetryRecovery(false)

    let flight = recoveryFlightRef.current
    if (!flight || flight.identity !== descriptor.identity) {
      flight?.abort()
      const request = startRecoveryRequest(signal => (
        descriptor.kind === 'get'
          ? getLearningPathResource(descriptor.pathId, { signal })
          : descriptor.kind === 'current'
            ? getCurrentLearningPathResource(null, { signal })
            : createLearningPathResource({
                ...descriptor.recovery.intent,
                idempotency_key: descriptor.recovery.start_idempotency_key,
                signal,
              })
      ), activeRecoveryControllers.current)
      flight = {
        identity: descriptor.identity,
        ...request,
      }
      recoveryFlightRef.current = flight
      flight.promise.finally(() => {
        if (recoveryFlightRef.current === flight) {
          recoveryFlightRef.current = null
        }
      }).catch(() => {})
    }

    flight.promise.then((value) => {
      if (!active || !mountedRef.current || epoch !== operationEpochRef.current) return
      if (descriptor.kind === 'current' && value == null) {
        setCanRetryRecovery(false)
        setError(null)
        return
      }
      const expectedDocumentId = descriptor.recovery?.intent.document_id || null
      const expectedLearningPathId = descriptor.kind === 'get'
        ? descriptor.pathId
        : null
      const nextResource = normalizeExpectedResource(
        value,
        expectedDocumentId,
        expectedLearningPathId,
      )
      const current = readLearningPathRecovery()
      if (
        current
        && descriptor.recovery
        && current.start_idempotency_key
          === descriptor.recovery.start_idempotency_key
      ) {
        const bound = bindLearningPathRecovery(
          current,
          nextResource.learning_path_id,
        )
        if (!bound || !writeLearningPathRecovery(bound)) {
          throw new Error('学习路径已恢复，但浏览器无法保存恢复指针')
        }
      }
      publishResource(nextResource)
      setCanRetryRecovery(false)
      setError(null)
      if (queryPathId !== nextResource.learning_path_id) {
        const search = new URLSearchParams({
          path_id: nextResource.learning_path_id,
        })
        navigate(`${location.pathname}?${search}`, { replace: true })
      }
    }).catch((err) => {
      if (!active || !mountedRef.current || epoch !== operationEpochRef.current) return
      const terminal = descriptor.kind === 'create'
        ? isTerminalCreateError(err)
        : err.status === 404 || err.status === 410
          || (err.status === 409 && err.reason === 'payload_mismatch')
      if (terminal && descriptor.recovery) {
        clearLearningPathRecovery(
          descriptor.recovery.start_idempotency_key,
        )
      }
      if (terminal && queryPathId) {
        navigate(location.pathname, { replace: true })
      }
      if (terminal) {
        resourceRef.current = null
        setResource(null)
      }
      setCanRetryRecovery(!terminal)
      setError(terminal
        ? '这条学习路径无法继续恢复，请重新生成'
        : err.message)
    }).finally(() => {
      if (active && mountedRef.current && epoch === operationEpochRef.current) {
        setRecovering(false)
      }
    })
    return () => {
      active = false
    }
  }, [
    location.pathname,
    navigate,
    publishResource,
    queryPathId,
    queryPathIdValid,
    recoveryNonce,
  ])

  const handleDocumentChange = (event) => {
    setSelectedDoc(event.target.value)
    setError(null)
  }

  const handleGenerate = async () => {
    const pendingCreation = readLearningPathRecovery()
    const lockedRecovery = (
      pendingCreation
      && !pendingCreation.learning_path_id
    ) ? pendingCreation : null
    const requestedDocumentId = lockedRecovery?.intent.document_id || selectedDoc
    const lockedIntentMatches = Boolean(
      lockedRecovery
      && lockedRecovery.intent.document_id === requestedDocumentId,
    )
    if (
      !requestedDocumentId
      || (!lockedIntentMatches && (
        docsStatus !== 'ready'
        || !documents.includes(requestedDocumentId)
      ))
      || actionInFlightRef.current
    ) return

    let recovery = lockedRecovery || pendingCreation
    if (
      !recovery
      || recovery.learning_path_id
      || recovery.intent.document_id !== requestedDocumentId
    ) {
      try {
        recovery = createLearningPathRecovery(
          requestedDocumentId,
          createIdempotencyKey(),
        )
      } catch (err) {
        setError(err.message)
        return
      }
    }
    if (!recovery || !writeLearningPathRecovery(recovery)) {
      setError('浏览器无法保存恢复信息，尚未开始生成')
      return
    }

    const epoch = ++operationEpochRef.current
    abortRecoveryRequests(activeRecoveryControllers.current)
    recoveryFlightRef.current = null
    actionInFlightRef.current = true
    setGenerating(true)
    setError(null)
    setCanRetryRecovery(false)
    if (queryPathId) {
      navigate(location.pathname, { replace: true })
    }

    try {
      const value = await runRecoveryRequest(
        signal => createLearningPathResource({
          ...recovery.intent,
          idempotency_key: recovery.start_idempotency_key,
          signal,
        }),
        activeRecoveryControllers.current,
      )
      if (!mountedRef.current || epoch !== operationEpochRef.current) return
      const nextResource = normalizeExpectedResource(
        value,
        recovery.intent.document_id,
      )
      const current = readLearningPathRecovery()
      if (current?.start_idempotency_key !== recovery.start_idempotency_key) {
        return
      }
      const bound = bindLearningPathRecovery(
        current,
        nextResource.learning_path_id,
      )
      if (!bound || !writeLearningPathRecovery(bound)) {
        throw new Error('学习路径已生成，但浏览器无法保存恢复指针')
      }
      actionInFlightRef.current = false
      publishResource(nextResource)
      const search = new URLSearchParams({
        path_id: nextResource.learning_path_id,
      })
      navigate(`${location.pathname}?${search}`, { replace: true })
    } catch (err) {
      if (!mountedRef.current || epoch !== operationEpochRef.current) return
      const terminal = isTerminalCreateError(err)
      const mismatch = err.status === 409 && err.reason === 'payload_mismatch'
      if (terminal) {
        clearLearningPathRecovery(recovery.start_idempotency_key)
      }
      setCanRetryRecovery(!terminal)
      setError(mismatch
        ? '恢复请求与服务端记录不一致，请重新点击生成'
        : err.message)
    } finally {
      if (epoch === operationEpochRef.current) {
        actionInFlightRef.current = false
        if (mountedRef.current) setGenerating(false)
      }
    }
  }

  const path = resource?.path || null
  const completedThrough = resource?.progress.completed_through || 0
  const readyStageId = path && completedThrough < path.total_stages
    ? completedThrough + 1
    : null
  const totalMinutes = path?.stages.reduce(
    (sum, stage) => sum + stage.estimated_minutes,
    0,
  ) || 0
  const materialStatus = !path || docsStatus === 'loading'
    ? 'unknown'
    : docsStatus === 'error'
      ? 'unknown'
      : documents.includes(path.document_id) ? 'available' : 'deleted'
  const busy = generating || recovering
  const pendingCreation = readLearningPathRecovery()
  const creationIntentLocked = Boolean(
    pendingCreation && !pendingCreation.learning_path_id,
  )
  const selectedCreationDoc = creationIntentLocked
    ? pendingCreation.intent.document_id
    : selectedDoc

  const handlePracticeStage = (stage) => {
    if (
      materialStatus !== 'available'
      || stage.stage !== readyStageId
      || !resource
    ) return
    const topic = (stage.topics.join('、') || stage.title).trim()
    const params = new URLSearchParams({
      document_id: path.document_id,
      topic,
      path_id: resource.learning_path_id,
      stage_id: String(stage.stage),
    })
    try {
      params.set('launch_id', createIdempotencyKey())
      navigate(`/quiz?${params.toString()}`)
    } catch (err) {
      setError(err.message)
    }
  }

  return (
    <div className="lp-page">
      <header className="page-header">
        <h1 className="page-title">学习路径</h1>
        <p className="page-desc">选择文档，AI 将分析内容并生成分阶段学习计划。</p>
      </header>

      {docsStatus === 'error' && !creationIntentLocked ? (
        <div className="load-error-state" role="alert">
          <p className="state-title">无法加载文档列表</p>
          <p className="state-desc">{docsError}</p>
          <button type="button" className="state-action" onClick={loadDocuments}>
            重新加载文档
          </button>
        </div>
      ) : docsStatus === 'ready'
        && documents.length === 0
        && !path
        && !creationIntentLocked ? (
        <DocumentPrerequisite description="生成学习路径前，需要先上传一份已经完成解析的学习材料。" />
      ) : (
        <div className="lp-controls">
          <select
            className="lp-select"
            aria-label="学习文档"
            value={selectedCreationDoc}
            onChange={handleDocumentChange}
            disabled={docsStatus === 'loading' || busy || creationIntentLocked}
          >
            <option value="">
              {docsStatus === 'loading' ? '加载文档列表...' : '-- 选择文档 --'}
            </option>
            {path
              && !documents.includes(path.document_id)
              && (!creationIntentLocked
                || path.document_id !== pendingCreation.intent.document_id) && (
              <option value={path.document_id} disabled>
                {path.document_id}（材料已删除）
              </option>
            )}
            {creationIntentLocked
              && !documents.includes(pendingCreation.intent.document_id) && (
              <option value={pendingCreation.intent.document_id}>
                {pendingCreation.intent.document_id}（待确认）
              </option>
            )}
            {documents.map(doc => (
              <option key={doc} value={doc}>{doc}</option>
            ))}
          </select>

          <button
            type="button"
            className="lp-generate-btn"
            onClick={handleGenerate}
            disabled={
              !selectedCreationDoc
              || busy
              || (!creationIntentLocked && (
                docsStatus !== 'ready'
                || !documents.includes(selectedDoc)
              ))
            }
          >
            {generating ? (
              <><span className="btn-spinner" />AI 分析中...</>
            ) : creationIntentLocked ? '重试恢复' : '生成学习路径'}
          </button>
        </div>
      )}

      {error && (
        <div className="error-banner" role="alert">
          <span>&#9888;</span>
          <span>{error}</span>
          {canRetryRecovery && !creationIntentLocked && (
            <button
              type="button"
              className="error-retry"
              onClick={() => setRecoveryNonce(value => value + 1)}
            >
              重试恢复
            </button>
          )}
          <button
            type="button"
            className="error-close"
            aria-label="关闭错误提示"
            onClick={() => setError(null)}
          >
            &times;
          </button>
        </div>
      )}

      {busy && (
        <div className="lp-skeleton" aria-busy="true">
          <div className="lp-loading-copy" role="status" aria-live="polite">
            <strong>{recovering ? '正在恢复学习路径' : '正在整理学习路径'}</strong>
            <span>
              {recovering
                ? '正在读取服务端保存的路径。'
                : '正在分析材料、检索重点并组织阶段，通常需要约 1 分钟。'}
            </span>
          </div>
          {[1, 2, 3].map(index => (
            <div key={index} className="skeleton-stage">
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

      {path && !busy && (
        <div className="lp-result">
          {materialStatus !== 'available' && (
            <div className="lp-retained-note" role="status">
              {materialStatus === 'deleted'
                ? '原材料已删除；这条学习路径作为学习记录保留，仅供查看。'
                : '暂时无法确认原材料状态；恢复文档列表前不能开始阶段练习。'}
            </div>
          )}
          <div className="lp-header-card">
            <h2 className="lp-path-title">{path.title}</h2>
            <div className="lp-meta-row">
              <span className="lp-meta-item">材料：{path.document_id}</span>
              <span className="lp-meta-item">{path.total_stages} 个阶段</span>
              <span className="lp-meta-item">预计 {totalMinutes} 分钟</span>
              <span className="lp-meta-item">
                已完成 {completedThrough}/{path.total_stages}
              </span>
            </div>
          </div>

          <div className="lp-timeline">
            {path.stages.map((stage, index) => {
              const stageStatus = stage.stage <= completedThrough
                ? 'completed'
                : stage.stage === readyStageId ? 'ready' : 'locked'
              return (
              <div
                key={stage.stage}
                className={`timeline-item is-${stageStatus}`}
              >
                <div className="timeline-track">
                  <div className="timeline-node">
                    <span className="node-number">
                      {stageStatus === 'completed' ? '✓' : stage.stage}
                    </span>
                  </div>
                  {index < path.stages.length - 1 && <div className="timeline-line" />}
                </div>
                <div
                  className="timeline-card"
                  aria-current={stageStatus === 'ready' ? 'step' : undefined}
                >
                  <div className="tc-header">
                    <h3 className="tc-title">{stage.title}</h3>
                    <span className="tc-time">{stage.estimated_minutes} min</span>
                  </div>
                  <p className="tc-desc">{stage.description}</p>
                  <div className="tc-topics">
                    {stage.topics.map(topic => (
                      <span key={topic} className="topic-tag">{topic}</span>
                    ))}
                  </div>
                  <div className="tc-actions">
                    {stageStatus === 'completed' ? (
                      <span className="tc-stage-status completed">已完成</span>
                    ) : stageStatus === 'locked' ? (
                      <span className="tc-stage-status locked">
                        完成上一阶段后解锁
                      </span>
                    ) : materialStatus === 'available' ? (
                      <button
                        type="button"
                        className="tc-practice-btn"
                        aria-label={`练习阶段 ${stage.stage}：${stage.title}`}
                        onClick={() => handlePracticeStage(stage)}
                      >
                        练习本阶段 →
                      </button>
                    ) : (
                      <span className="tc-stage-status unavailable">
                        {materialStatus === 'deleted' ? '材料已删除，仅供查看' : '暂不可练习'}
                      </span>
                    )}
                  </div>
                </div>
              </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
