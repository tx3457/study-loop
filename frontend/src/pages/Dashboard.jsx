import { useState, useEffect, useCallback, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  getUserProfile,
  getUserSessions,
  getDocuments,
  getWrongQuestions,
  startWrongQuestionPractice,
} from '../api/client'
import './Dashboard.css'

/* ═══════════════════════════════════════════════════════════════════
   Dashboard — 学习评估仪表盘

   - SVG 雷达图：各文档知识点掌握度
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
  const [practiceLoading, setPracticeLoading] = useState(false)
  const [error, setError] = useState(null)
  const [partialError, setPartialError] = useState(null)
  const dashboardRequestId = useRef(0)
  const wrongRequestId = useRef(0)
  const practiceRequestId = useRef(0)

  /* ── 加载数据 ──────────────────────────────────────────────────── */
  const loadDashboard = useCallback(async () => {
    const requestId = ++dashboardRequestId.current
    setLoading(true)
    setError(null)
    setPartialError(null)
    const sources = [
      {
        label: '文档列表',
        load: getDocuments,
        apply: data => setDocuments(data.documents || []),
      },
      {
        label: '学习记录',
        load: getUserSessions,
        apply: data => setSessions(Array.isArray(data) ? data : []),
      },
      {
        label: '学习画像',
        load: getUserProfile,
        apply: data => {
          setProfile(data)
          setNoProfile(!data)
        },
      },
    ]
    try {
      const results = await Promise.allSettled(sources.map(source => source.load()))
      if (requestId !== dashboardRequestId.current) return

      const failures = []
      results.forEach((result, index) => {
        if (result.status === 'fulfilled') {
          sources[index].apply(result.value)
          return
        }
        const message = result.reason?.message
        failures.push(message
          ? `${sources[index].label}（${message}）`
          : sources[index].label)
      })

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
      practiceRequestId.current += 1
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

  const handleRepractice = async () => {
    if (!wrongDoc || practiceLoading) return
    const documentId = wrongDoc
    const requestId = ++practiceRequestId.current
    setPracticeLoading(true)
    setPracticeError(null)
    try {
      const practice = await startWrongQuestionPractice(documentId)
      if (requestId !== practiceRequestId.current) return
      navigate('/quiz', {
        state: {
          wrongQuestionPractice: {
            ...practice,
            document_id: documentId,
          },
        },
      })
    } catch (err) {
      if (requestId !== practiceRequestId.current) return
      setPracticeError(err.message)
    } finally {
      if (requestId === practiceRequestId.current) {
        setPracticeLoading(false)
      }
    }
  }

  const handleWrongDocumentChange = (event) => {
    practiceRequestId.current += 1
    setPracticeLoading(false)
    setPracticeError(null)
    setWrongDoc(event.target.value)
  }

  /* ── 衍生数据 ──────────────────────────────────────────────────── */
  const mastery = profile?.topic_mastery || {}
  const masteryEntries = Object.entries(mastery)
  const weakPoints = profile?.weak_points || []
  const realSessions = sessions.filter(s => !s.type) // 排除 archive 条目
  const avgScore = realSessions.length
    ? Math.round((realSessions.reduce((s, e) => s + (e.correct_rate || 0), 0) / realSessions.length) * 100)
    : 0

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
        <p className="page-desc">基于三层记忆系统的学习分析：知识点掌握度、学习趋势、错题回顾。</p>
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
      ) : !partialError && noProfile && realSessions.length === 0 ? (
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
              <span className="stat-value">{profile?.total_sessions || 0}</span>
              <span className="stat-label">学习次数</span>
            </div>
            <div className="stat-card">
              <span className="stat-value">{avgScore}<small>%</small></span>
              <span className="stat-label">平均正确率</span>
            </div>
            <div className="stat-card">
              <span className="stat-value">{masteryEntries.length}</span>
              <span className="stat-label">学习文档</span>
            </div>
            <div className="stat-card">
              <span className="stat-value">{weakPoints.length}</span>
              <span className="stat-label">薄弱知识点</span>
            </div>
          </div>

          {/* ── 雷达图 + 薄弱知识点 ──────────────────────────────── */}
          <div className="dash-row">
            <div className="dash-card radar-card">
              <h2 className="card-title">知识点掌握度</h2>
              {masteryEntries.length > 0 ? (
                <RadarChart data={masteryEntries} />
              ) : (
                <p className="card-empty">暂无掌握度数据</p>
              )}
            </div>

            <div className="dash-card weak-card">
              <h2 className="card-title">薄弱知识点</h2>
              {weakPoints.length > 0 ? (
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
            {realSessions.length > 1 ? (
              <TrendChart sessions={realSessions} />
            ) : (
              <p className="card-empty">至少需要 2 次答题记录才能显示趋势</p>
            )}
          </div>

          {/* ── 错题本 ────────────────────────────────────────────── */}
          <div className="dash-card wrong-card">
            <div className="wrong-header">
              <h2 className="card-title">错题本</h2>
              <div className="wrong-actions">
                {wrongQuestions?.total > 0 && !wrongLoading && !wrongError && (
                  <button
                    type="button"
                    className="state-action wrong-practice"
                    onClick={handleRepractice}
                    disabled={practiceLoading}
                  >
                    {practiceLoading ? '正在准备...' : `开始重练（${wrongQuestions.total}）`}
                  </button>
                )}
                <select
                  className="wrong-select"
                  aria-label="错题文档"
                  value={wrongDoc}
                  onChange={handleWrongDocumentChange}
                >
                  <option value="">-- 选择文档 --</option>
                  {documents.map(d => <option key={d} value={d}>{d}</option>)}
                </select>
              </div>
            </div>

            {wrongLoading && <div className="loading-spinner small" />}

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

            {!wrongDoc && !wrongLoading && (
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
