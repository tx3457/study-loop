import { useState, useEffect, useCallback, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  createIdempotencyKey,
  getUserProfile,
  getUserSessions,
  getDocuments,
  getWrongQuestions,
} from '../api/client'
import {
  createWrongQuestionQuizRecovery,
  readQuizRecovery,
  sameQuizIntent,
  writeQuizRecovery,
} from '../state/quizRecovery'
import './Dashboard.css'

/* ═══════════════════════════════════════════════════════════════════
   Dashboard — 学习评估仪表盘

   - SVG 雷达图：各学习材料掌握度
   - SVG 折线图：学习趋势（正确率随时间变化）
   - 薄弱知识点列表
   - 错题本（选择文档查看）
   ═══════════════════════════════════════════════════════════════════ */

export default function Dashboard() {
  const navigate = useNavigate()
  const [profile, setProfile] = useState(null)
  const [sessions, setSessions] = useState([])
  const [documents, setDocuments] = useState([])
  const [wrongDoc, setWrongDoc] = useState('')
  const [wrongQuestions, setWrongQuestions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [noProfile, setNoProfile] = useState(false)
  const [wrongLoading, setWrongLoading] = useState(false)
  const [wrongError, setWrongError] = useState(null)
  const [practiceError, setPracticeError] = useState(null)
  const [practiceConflict, setPracticeConflict] = useState(null)
  const [error, setError] = useState(null)
  const [partialError, setPartialError] = useState(null)
  const [sourceStatus, setSourceStatus] = useState({
    documents: 'loading',
    sessions: 'loading',
    profile: 'loading',
  })
  const dashboardRequestId = useRef(0)
  const wrongRequestId = useRef(0)

  /* ── 加载数据 ──────────────────────────────────────────────────── */
  const loadDashboard = useCallback(async () => {
    const requestId = ++dashboardRequestId.current
    setLoading(true)
    setError(null)
    setPartialError(null)
    setSourceStatus({ documents: 'loading', sessions: 'loading', profile: 'loading' })
    setDocuments([])
    setSessions([])
    setProfile(null)
    setNoProfile(false)
    wrongRequestId.current += 1
    setWrongDoc('')
    setWrongQuestions(null)
    setWrongLoading(false)
    setWrongError(null)
    setPracticeError(null)
    setPracticeConflict(null)
    const sources = [
      {
        key: 'documents',
        label: '文档列表',
        load: getDocuments,
        apply: data => {
          if (!data || !Array.isArray(data.documents)) {
            throw new Error('响应格式无效')
          }
          setDocuments(data.documents)
        },
      },
      {
        key: 'sessions',
        label: '学习记录',
        load: getUserSessions,
        apply: data => {
          if (!Array.isArray(data)) throw new Error('响应格式无效')
          setSessions(data)
        },
      },
      {
        key: 'profile',
        label: '学习画像',
        load: getUserProfile,
        apply: data => {
          if (data !== null && (typeof data !== 'object' || Array.isArray(data))) {
            throw new Error('响应格式无效')
          }
          setProfile(data)
          setNoProfile(!data)
        },
      },
    ]
    try {
      const results = await Promise.allSettled(sources.map(source => source.load()))
      if (requestId !== dashboardRequestId.current) return

      const failures = []
      const nextStatus = { documents: 'failed', sessions: 'failed', profile: 'failed' }
      results.forEach((result, index) => {
        if (result.status === 'fulfilled') {
          try {
            sources[index].apply(result.value)
            nextStatus[sources[index].key] = 'available'
            return
          } catch (validationError) {
            failures.push(`${sources[index].label}（${validationError.message}）`)
            return
          }
        }
        const message = result.reason?.message
        failures.push(message
          ? `${sources[index].label}（${message}）`
          : sources[index].label)
      })
      setSourceStatus(nextStatus)

      if (failures.length === sources.length) {
        setError(`全部数据源加载失败：${failures.join('、')}`)
      } else if (failures.length > 0) {
        setPartialError(`部分数据加载失败：${failures.join('、')}`)
      }
    } finally {
      if (requestId === dashboardRequestId.current) setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadDashboard()
    return () => {
      dashboardRequestId.current += 1
    }
  }, [loadDashboard])

  /* ── 加载错题本 ────────────────────────────────────────────────── */
  const loadWrongQuestions = useCallback(async (documentId) => {
    const requestId = ++wrongRequestId.current
    setWrongLoading(true)
    setWrongError(null)
    setPracticeError(null)
    try {
      const data = await getWrongQuestions(documentId)
      if (requestId !== wrongRequestId.current) return
      setWrongQuestions(data)
    } catch (err) {
      if (requestId !== wrongRequestId.current) return
      setWrongQuestions(null)
      setWrongError(err.message)
    } finally {
      if (requestId === wrongRequestId.current) setWrongLoading(false)
    }
  }, [])

  useEffect(() => {
    if (!wrongDoc) {
      wrongRequestId.current += 1
      setWrongQuestions(null)
      setWrongError(null)
      setPracticeError(null)
      setWrongLoading(false)
      return
    }
    loadWrongQuestions(wrongDoc)
  }, [loadWrongQuestions, wrongDoc])

  const handleRepractice = () => {
    if (!wrongDoc) return
    setPracticeError(null)
    try {
      const candidate = createWrongQuestionQuizRecovery(
        wrongDoc,
        createIdempotencyKey(),
      )
      if (!candidate) {
        throw new Error('无法创建错题重练恢复记录，请刷新页面后重试')
      }
      const currentRecovery = readQuizRecovery()
      // 与 Quiz.handleStart / Adaptive.handleStart 一致：同一个还没建立会话的
      // 意图必须沿用原来的幂等键。换新键服务端会按新键再建一个持久会话，旧的
      // 被孤儿化但仍占容量，答完后学习历史还会多出一条记录。
      const reusable = Boolean(
        currentRecovery
        && !currentRecovery.session
        && sameQuizIntent(currentRecovery.intent, candidate.intent)
      )
      const nextRecovery = reusable ? currentRecovery : candidate
      if (currentRecovery && !reusable) {
        setPracticeConflict({ currentRecovery, nextRecovery })
        return
      }
      if (!writeQuizRecovery(nextRecovery)) {
        throw new Error('浏览器无法保存重练进度，请检查存储权限后重试')
      }
      navigate('/quiz')
    } catch (err) {
      setPracticeError(err.message)
    }
  }

  const handleWrongDocumentChange = (event) => {
    const nextDocument = event.target.value
    wrongRequestId.current += 1
    setPracticeError(null)
    setPracticeConflict(null)
    setWrongQuestions(null)
    setWrongError(null)
    setWrongLoading(Boolean(nextDocument))
    setWrongDoc(nextDocument)
  }

  const continueCurrentPractice = () => {
    setPracticeConflict(null)
    setPracticeError(null)
    navigate('/quiz')
  }

  const replaceCurrentPractice = () => {
    const nextRecovery = practiceConflict?.nextRecovery
    if (!nextRecovery) return
    setPracticeError(null)
    if (!writeQuizRecovery(nextRecovery)) {
      setPracticeError('浏览器无法保存重练进度；当前练习仍已保留，请检查存储权限后重试')
      return
    }
    setPracticeConflict(null)
    navigate('/quiz')
  }

  const recoveryLabel = (recovery) => {
    if (!recovery) return ''
    const request = recovery.intent.request
    return `${request.document_id} / ${
      recovery.intent.kind === 'standard' ? request.description : '错题重练'
    }`
  }

  /* ── 衍生数据 ──────────────────────────────────────────────────── */
  const documentsAvailable = sourceStatus.documents === 'available'
  const sessionsAvailable = sourceStatus.sessions === 'available'
  const profileAvailable = sourceStatus.profile === 'available'
  const mastery = profileAvailable ? profile?.topic_mastery || {} : {}
  const masteryEntries = Object.entries(mastery)
  const weakPoints = profileAvailable ? profile?.weak_points || [] : []
  // 归档代表多次历史学习：平均分按 session_count 加权，趋势中保留一个聚合点，
  // 避免归档后 Dashboard 只展示最近几次记录。
  const trendSessions = sessions
    .map(session => session.type === 'archive'
      ? { ...session, correct_rate: session.avg_correct_rate, isArchive: true }
      : session)
    .filter(session => Number.isFinite(Number(session.correct_rate)))
  const scoreSummary = trendSessions.reduce((summary, session) => {
    const weight = session.isArchive ? Math.max(Number(session.session_count) || 0, 0) : 1
    return {
      weightedScore: summary.weightedScore + Number(session.correct_rate) * weight,
      count: summary.count + weight,
    }
  }, { weightedScore: 0, count: 0 })
  const profileAverage = profileAvailable ? profile?.average_correct_rate : null
  const avgScore = typeof profileAverage === 'number' && Number.isFinite(profileAverage)
    ? Math.round(profileAverage * 100)
    : sessionsAvailable && scoreSummary.count
      ? Math.round((scoreSummary.weightedScore / scoreSummary.count) * 100)
      : null

  if (loading) {
    return (
      <div className="dash-page">
        <header className="page-header">
          <h1 className="page-title">学习报告</h1>
        </header>
        <div className="dash-loading">
          <div className="loading-spinner" />
          <p>加载数据中...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="dash-page">
      <header className="page-header">
        <h1 className="page-title">学习报告</h1>
        <p className="page-desc">基于学习记忆的分析：材料掌握度、薄弱知识点、学习趋势与错题回顾。</p>
      </header>

      {partialError && !error && (
        <div className="load-error-state dash-partial-error" role="alert">
          <p className="state-title">部分数据暂不可用</p>
          <p className="state-desc">{partialError}</p>
          <button type="button" className="state-action" onClick={loadDashboard}>
            重新加载
          </button>
        </div>
      )}

      {error ? (
        <div className="load-error-state" role="alert">
          <p className="state-title">无法加载学习报告</p>
          <p className="state-desc">{error}</p>
          <button type="button" className="state-action" onClick={loadDashboard}>
            重新加载
          </button>
        </div>
      ) : !partialError && noProfile && trendSessions.length === 0 ? (
        <div className="dash-empty">
          <div className="empty-icon">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round">
              <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
            </svg>
          </div>
          <p className="empty-title">还没有学习数据</p>
          <p className="empty-desc">完成一次答题练习并 AI 批改后，这里会显示你的学习报告</p>
        </div>
      ) : (
        <>
          {/* ── 概览卡片 ──────────────────────────────────────────── */}
          <div className="stats-row">
            <div className="stat-card">
              <span className="stat-value">
                {profileAvailable ? profile?.total_sessions || 0 : '—'}
              </span>
              <span className="stat-label">学习次数</span>
            </div>
            <div className="stat-card">
              <span className="stat-value">
                {avgScore == null ? '—' : <>{avgScore}<small>%</small></>}
              </span>
              <span className="stat-label">平均正确率</span>
            </div>
            <div className="stat-card">
              <span className="stat-value">{profileAvailable ? masteryEntries.length : '—'}</span>
              <span className="stat-label">学习文档</span>
            </div>
            <div className="stat-card">
              <span className="stat-value">{profileAvailable ? weakPoints.length : '—'}</span>
              <span className="stat-label">薄弱知识点</span>
            </div>
          </div>

          {/* ── 雷达图 + 薄弱知识点 ──────────────────────────────── */}
          <div className="dash-row">
            <div className="dash-card radar-card">
              <h2 className="card-title">材料掌握度</h2>
              {!profileAvailable ? (
                <p className="card-empty">学习画像暂不可用</p>
              ) : masteryEntries.length > 0 ? (
                <RadarChart data={masteryEntries} />
              ) : (
                <p className="card-empty">暂无掌握度数据</p>
              )}
            </div>

            <div className="dash-card weak-card">
              <h2 className="card-title">薄弱知识点</h2>
              {!profileAvailable ? (
                <p className="card-empty">学习画像暂不可用</p>
              ) : weakPoints.length > 0 ? (
                <div className="weak-list">
                  {weakPoints.slice(0, 12).map((wp, i) => (
                    <span key={i} className="weak-tag" style={{ animationDelay: `${i * 0.04}s` }}>
                      {wp}
                    </span>
                  ))}
                </div>
              ) : (
                <p className="card-empty">暂无薄弱知识点</p>
              )}
            </div>
          </div>

          {/* ── 学习趋势折线图 ────────────────────────────────────── */}
          <div className="dash-card trend-card">
            <h2 className="card-title">学习趋势</h2>
            {!sessionsAvailable ? (
              <p className="card-empty">学习记录暂不可用</p>
            ) : trendSessions.length > 1 ? (
              <TrendChart sessions={trendSessions} />
            ) : (
              <p className="card-empty">至少需要 2 次答题记录才能显示趋势</p>
            )}
          </div>

          {/* ── 错题本 ────────────────────────────────────────────── */}
          <div className="dash-card wrong-card">
            <div className="wrong-header">
              <h2 className="card-title">错题本</h2>
              <div className="wrong-actions">
                {wrongQuestions?.total > 0
                  && !wrongLoading
                  && !wrongError
                  && !practiceConflict && (
                  <button
                    type="button"
                    className="state-action wrong-practice"
                    onClick={handleRepractice}
                  >
                    {`开始重练（${wrongQuestions.total}）`}
                  </button>
                )}
                <select
                  className="wrong-select"
                  aria-label="错题文档"
                  value={wrongDoc}
                  onChange={handleWrongDocumentChange}
                  disabled={!documentsAvailable}
                >
                  <option value="">-- 选择文档 --</option>
                  {documents.map(d => <option key={d} value={d}>{d}</option>)}
                </select>
              </div>
            </div>

            {wrongLoading && <div className="loading-spinner small" />}

            {!documentsAvailable && (
              <p className="card-empty">文档列表暂不可用，暂时无法查看错题</p>
            )}

            {practiceConflict && !wrongLoading && !wrongError && (
              <section
                className="practice-intent-conflict"
                role="region"
                aria-live="polite"
                aria-labelledby="practice-intent-conflict-title"
                aria-describedby="practice-intent-conflict-desc"
              >
                <h3 id="practice-intent-conflict-title">检测到尚未结束的练习</h3>
                <p id="practice-intent-conflict-desc">
                  开始错题重练会替换浏览器中保存的当前进度。请选择继续当前练习，或明确放弃后开始新的错题重练。
                </p>
                <dl>
                  <div>
                    <dt>当前进度</dt>
                    <dd>{recoveryLabel(practiceConflict.currentRecovery)}</dd>
                  </div>
                  <div>
                    <dt>新的重练</dt>
                    <dd>{recoveryLabel(practiceConflict.nextRecovery)}</dd>
                  </div>
                </dl>
                <div className="practice-intent-actions">
                  <button type="button" className="state-action" onClick={continueCurrentPractice}>
                    继续当前练习
                  </button>
                  <button type="button" className="state-action danger" onClick={replaceCurrentPractice}>
                    放弃并开始错题重练
                  </button>
                </div>
              </section>
            )}

            {wrongError && !wrongLoading && (
              <div className="load-error-state" role="alert">
                <p className="state-title">无法加载错题</p>
                <p className="state-desc">{wrongError}</p>
                <button
                  type="button"
                  className="state-action"
                  onClick={() => loadWrongQuestions(wrongDoc)}
                >
                  重新加载
                </button>
              </div>
            )}

            {practiceError && !wrongLoading && !wrongError && (
              <div className="load-error-state" role="alert">
                <p className="state-title">无法开始错题重练</p>
                <p className="state-desc">{practiceError}</p>
                <button type="button" className="state-action" onClick={handleRepractice}>
                  重试
                </button>
              </div>
            )}

            {wrongQuestions && !wrongLoading && !wrongError && !practiceError && (
              wrongQuestions.total === 0 ? (
                <p className="card-empty">该文档暂无错题</p>
              ) : (
                <div className="wrong-list">
                  {wrongQuestions.entries.map((entry, idx) => (
                    <div key={entry.entry_id} className="wrong-item" style={{ animationDelay: `${idx * 0.05}s` }}>
                      <div className="wi-header">
                        <span className="wi-index">#{idx + 1}</span>
                        {entry.knowledge_gap && <span className="wi-gap">{entry.knowledge_gap}</span>}
                      </div>
                      <p className="wi-question">{entry.question}</p>
                      <div className="wi-answers">
                        <div className="wi-answer wrong">
                          <span className="wi-label">你的答案</span>
                          <span>{entry.user_answer}</span>
                        </div>
                        <div className="wi-answer correct">
                          <span className="wi-label">正确答案</span>
                          <span>{entry.correct_answer}</span>
                        </div>
                      </div>
                      <p className="wi-explain">{entry.explanation}</p>
                    </div>
                  ))}
                </div>
              )
            )}

            {documentsAvailable && !wrongDoc && !wrongLoading && (
              <p className="card-empty">选择文档查看错题</p>
            )}
          </div>
        </>
      )}
    </div>
  )
}


/* ═══════════════════════════════════════════════════════════════════
   RadarChart — SVG 雷达图组件
   ═══════════════════════════════════════════════════════════════════ */

function RadarChart({ data }) {
  const size = 260
  const cx = size / 2
  const cy = size / 2
  const levels = 4                      // 同心圆层数
  const maxR = 100                      // 最大半径
  const n = data.length

  // 保证至少 3 个轴（不足时填充）
  const padded = n >= 3 ? data : [...data, ...Array(3 - n).fill(['', 0])]
  const count = padded.length
  const angleStep = (2 * Math.PI) / count

  // 数据点 → 坐标
  const getPoint = (value, i) => {
    const angle = angleStep * i - Math.PI / 2
    const r = value * maxR
    return [cx + r * Math.cos(angle), cy + r * Math.sin(angle)]
  }

  // 同心网格
  const gridLevels = Array.from({ length: levels }, (_, l) => {
    const r = maxR * ((l + 1) / levels)
    const points = Array.from({ length: count }, (_, i) => {
      const angle = angleStep * i - Math.PI / 2
      return `${cx + r * Math.cos(angle)},${cy + r * Math.sin(angle)}`
    }).join(' ')
    return points
  })

  // 数据多边形
  const dataPoints = padded.map(([, val], i) => getPoint(Math.min(val, 1), i))
  const dataPolygon = dataPoints.map(p => p.join(',')).join(' ')

  return (
    <div className="radar-wrapper">
      <svg viewBox={`0 0 ${size} ${size}`} className="radar-svg">
        {/* 同心网格 */}
        {gridLevels.map((pts, i) => (
          <polygon key={i} points={pts} className="radar-grid" />
        ))}

        {/* 轴线 */}
        {padded.map(([, ], i) => {
          const [x, y] = getPoint(1, i)
          return <line key={i} x1={cx} y1={cy} x2={x} y2={y} className="radar-axis" />
        })}

        {/* 数据填充 */}
        <polygon points={dataPolygon} className="radar-fill" />
        <polygon points={dataPolygon} className="radar-stroke" />

        {/* 数据点 */}
        {dataPoints.map(([x, y], i) => (
          <circle key={i} cx={x} cy={y} r="4" className="radar-dot" />
        ))}

        {/* 标签 */}
        {padded.map(([label, val], i) => {
          const angle = angleStep * i - Math.PI / 2
          const lr = maxR + 22
          const lx = cx + lr * Math.cos(angle)
          const ly = cy + lr * Math.sin(angle)
          if (!label) return null
          return (
            <text key={i} x={lx} y={ly} className="radar-label" textAnchor="middle" dominantBaseline="central">
              {label.length > 8 ? label.slice(0, 7) + '…' : label}
              <tspan x={lx} dy="14" className="radar-value">{Math.round(val * 100)}%</tspan>
            </text>
          )
        })}
      </svg>
    </div>
  )
}


/* ═══════════════════════════════════════════════════════════════════
   TrendChart — SVG 折线图组件
   ═══════════════════════════════════════════════════════════════════ */

function TrendChart({ sessions }) {
  // 按日期正序（最旧在左）
  const sorted = [...sessions].sort((a, b) => (a.date || '').localeCompare(b.date || ''))
  const w = 600, h = 200
  const padL = 48, padR = 20, padT = 20, padB = 36
  const chartW = w - padL - padR
  const chartH = h - padT - padB
  const n = sorted.length

  const getX = (i) => padL + (chartW / (n - 1)) * i
  const getY = (rate) => padT + chartH * (1 - rate)

  const points = sorted.map((s, i) => [getX(i), getY(s.correct_rate || 0)])
  const polyline = points.map(p => p.join(',')).join(' ')

  // 渐变区域
  const areaPath = `M${points[0][0]},${padT + chartH} ${points.map(p => `L${p[0]},${p[1]}`).join(' ')} L${points[n - 1][0]},${padT + chartH} Z`

  return (
    <div className="trend-wrapper">
      <svg viewBox={`0 0 ${w} ${h}`} className="trend-svg">
        <defs>
          <linearGradient id="trendGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--accent-blue)" stopOpacity="0.2" />
            <stop offset="100%" stopColor="var(--accent-blue)" stopOpacity="0" />
          </linearGradient>
        </defs>

        {/* Y 轴刻度线 */}
        {[0, 0.25, 0.5, 0.75, 1].map(v => (
          <g key={v}>
            <line x1={padL} y1={getY(v)} x2={w - padR} y2={getY(v)} className="trend-gridline" />
            <text x={padL - 8} y={getY(v)} className="trend-y-label" textAnchor="end" dominantBaseline="central">
              {Math.round(v * 100)}%
            </text>
          </g>
        ))}

        {/* 渐变区域 */}
        <path d={areaPath} fill="url(#trendGrad)" />

        {/* 折线 */}
        <polyline points={polyline} className="trend-line" />

        {/* 数据点 + X轴标签 */}
        {points.map(([x, y], i) => (
          <g key={i}>
            <title>
              {sorted[i].isArchive
                ? `${sorted[i].session_count} 次历史学习聚合：${Math.round(sorted[i].correct_rate * 100)}%`
                : `${sorted[i].date || '本次学习'}：${Math.round(sorted[i].correct_rate * 100)}%`}
            </title>
            <circle cx={x} cy={y} r="4" className="trend-dot" />
            <text x={x} y={h - 8} className="trend-x-label" textAnchor="middle">
              {(sorted[i].date || '').slice(5)}
            </text>
          </g>
        ))}
      </svg>
    </div>
  )
}
