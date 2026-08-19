import { useState, useEffect, useRef, useCallback } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import DocumentPrerequisite from '../components/DocumentPrerequisite'
import {
  createIdempotencyKey,
  generateReport,
  getDocuments,
  getSessionResult,
  gradeSession,
  isTerminalExecutionError,
  startSession,
  submitAnswer,
} from '../api/client'
import './Quiz.css'

/* ═══════════════════════════════════════════════════════════════════
   Quiz Page — 答题界面

   状态机：setup → loading → answering ⇄ feedback → results → grading → report
   ═══════════════════════════════════════════════════════════════════ */

const DIFFICULTY_OPTIONS = [
  { value: 'easy',   label: '基础', color: '#059669' },
  { value: 'medium', label: '进阶', color: '#D97706' },
  { value: 'hard',   label: '挑战', color: '#DC2626' },
]

const TYPE_OPTIONS = [
  { value: 'choice',       label: '选择题' },
  { value: 'true_false',   label: '判断题' },
  { value: 'short_answer', label: '简答题' },
]

const LABELED_OPTION_PATTERN = /^([A-Za-z])(?:\s*[.．、:：)）]\s*|\s+)(.+)$/u

function normalizeText(value) {
  return typeof value === 'string'
    ? value.trim().replace(/\s+/gu, ' ').toLocaleLowerCase()
    : ''
}

function splitOption(value) {
  const raw = typeof value === 'string' ? value.trim() : ''
  const match = raw.match(LABELED_OPTION_PATTERN)
  return {
    value: normalizeText(raw),
    label: match?.[1].toUpperCase() || null,
    text: normalizeText(match?.[2] || raw),
  }
}

function isCorrectOption(option, correctAnswer) {
  const candidate = splitOption(option)
  const expected = splitOption(correctAnswer)
  if (!expected.value) return false
  if (candidate.value === expected.value) return true
  if (candidate.label && /^[A-Za-z]$/.test(expected.value)) {
    return candidate.label === expected.value.toUpperCase()
  }
  if (candidate.label && candidate.text === expected.value) return true
  return Boolean(
    candidate.label
    && expected.label
    && candidate.label === expected.label
    && candidate.text === expected.text
  )
}

export default function Quiz() {
  const location = useLocation()
  const navigate = useNavigate()
  const practiceHydrated = useRef(false)

  /* ── 状态 ──────────────────────────────────────────────────────── */
  const [phase, setPhase] = useState('setup') // setup | loading | answering | feedback | results | grading | report
  const [documents, setDocuments] = useState([])
  const [docsLoading, setDocsLoading] = useState(true)
  const [docsError, setDocsError] = useState(null)
  const [error, setError] = useState(null)

  // AI 批改 + 学习报告
  const [gradingReport, setGradingReport] = useState(null)
  const [learningReport, setLearningReport] = useState(null)
  const [grading, setGrading] = useState(false)
  const [reporting, setReporting] = useState(false)

  // 配置
  const [config, setConfig] = useState({
    document_id: '',
    description: '',
    count: 5,
    difficulty: 'medium',
    type: 'choice',
    user_id: 'default_user',
  })

  // 答题状态
  const [sessionId, setSessionId] = useState(null)
  const [questions, setQuestions] = useState([])
  const [currentIdx, setCurrentIdx] = useState(0)
  const [selectedAnswer, setSelectedAnswer] = useState('')
  const [feedback, setFeedback] = useState(null)
  const [submitting, setSubmitting] = useState(false)
  const answerIdempotencyKey = useRef(null)

  // 结果
  const [result, setResult] = useState(null)

  // 计时器
  const [elapsed, setElapsed] = useState(0)
  const timerRef = useRef(null)

  /* ── 加载文档列表 ──────────────────────────────────────────────── */
  const loadDocuments = useCallback(async () => {
    setDocsLoading(true)
    setDocsError(null)
    try {
      const data = await getDocuments()
      const nextDocuments = data.documents || []
      setDocuments(nextDocuments)
      setConfig(current => nextDocuments.includes(current.document_id)
        ? current
        : { ...current, document_id: '' })
    } catch (err) {
      setDocsError(err.message)
    } finally {
      setDocsLoading(false)
    }
  }, [])

  useEffect(() => { loadDocuments() }, [loadDocuments])

  /* ── 计时器 ────────────────────────────────────────────────────── */
  const startTimer = useCallback(() => {
    setElapsed(0)
    timerRef.current = setInterval(() => setElapsed(t => t + 1), 1000)
  }, [])

  const stopTimer = useCallback(() => {
    clearInterval(timerRef.current)
    timerRef.current = null
  }, [])

  useEffect(() => () => clearInterval(timerRef.current), [])

  useEffect(() => {
    const practice = location.state?.wrongQuestionPractice
    if (practiceHydrated.current || !practice?.session_id || !practice?.questions?.length) {
      return
    }

    practiceHydrated.current = true
    answerIdempotencyKey.current = null
    setSessionId(practice.session_id)
    setQuestions(practice.questions)
    setConfig(current => ({
      ...current,
      document_id: practice.document_id || current.document_id,
      description: '错题重练',
      count: practice.total || practice.questions.length,
    }))
    setCurrentIdx(0)
    setSelectedAnswer('')
    setFeedback(null)
    setResult(null)
    setError(null)
    setPhase('answering')
    startTimer()
    navigate(location.pathname, { replace: true, state: null })
  }, [location.pathname, location.state, navigate, startTimer])

  /* ── 开始答题 ──────────────────────────────────────────────────── */
  const handleStart = async () => {
    if (!config.document_id) return
    answerIdempotencyKey.current = null
    setPhase('loading')
    setError(null)

    try {
      const data = await startSession(config)
      setSessionId(data.session_id)
      setQuestions(data.questions)
      setCurrentIdx(0)
      setSelectedAnswer('')
      setFeedback(null)
      setResult(null)
      setPhase('answering')
      startTimer()
    } catch (err) {
      setError(err.message)
      setPhase('setup')
    }
  }

  /* ── 提交答案 ──────────────────────────────────────────────────── */
  const handleSubmit = async () => {
    if (!selectedAnswer || submitting) return
    const answer = selectedAnswer
    const questionIndex = currentIdx
    setSubmitting(true)
    setError(null)

    try {
      const idempotencyKey = answerIdempotencyKey.current || createIdempotencyKey()
      answerIdempotencyKey.current = idempotencyKey
      const fb = await submitAnswer(sessionId, {
        answer,
        question_index: questionIndex,
        idempotency_key: idempotencyKey,
      })
      answerIdempotencyKey.current = null
      setFeedback(fb)
      setPhase('feedback')
    } catch (err) {
      const staleState = err.status === 409 && !err.code
      if (err.status === 404 || staleState || isTerminalExecutionError(err)) {
        clearSession()
      }
      setError(err.message)
    } finally {
      setSubmitting(false)
    }
  }

  /* ── 下一题 / 查看结果 ──────────────────────────────────────────── */
  const handleNext = async () => {
    if (feedback?.is_last) {
      stopTimer()
      try {
        const res = await getSessionResult(sessionId)
        setResult(res)
        setPhase('results')
      } catch (err) {
        setError(err.message)
      }
    } else {
      answerIdempotencyKey.current = null
      setCurrentIdx(idx => idx + 1)
      setSelectedAnswer('')
      setFeedback(null)
      setPhase('answering')
    }
  }

  function clearSession() {
    stopTimer()
    answerIdempotencyKey.current = null
    setPhase('setup')
    setSessionId(null)
    setQuestions([])
    setCurrentIdx(0)
    setSelectedAnswer('')
    setFeedback(null)
    setSubmitting(false)
    setResult(null)
    setGradingReport(null)
    setLearningReport(null)
    setGrading(false)
    setReporting(false)
    setElapsed(0)
  }

  /* ── 重新开始 ──────────────────────────────────────────────────── */
  const handleRestart = () => {
    clearSession()
    setError(null)
  }

  const handleAnswerChange = (answer) => {
    if (answer !== selectedAnswer) answerIdempotencyKey.current = null
    setSelectedAnswer(answer)
  }

  /* ── AI 批改 ─────────────────────────────────────────────────────── */
  const handleGrade = async () => {
    if (grading) return
    setGrading(true)
    setError(null)
    try {
      const report = await gradeSession(sessionId)
      setGradingReport(report)
      setPhase('grading')
    } catch (err) {
      setError(err.message)
    } finally {
      setGrading(false)
    }
  }

  /* ── 学习报告 ───────────────────────────────────────────────────── */
  const handleReport = async () => {
    if (reporting) return
    setReporting(true)
    setError(null)
    try {
      const report = await generateReport(sessionId)
      setLearningReport(report)
      setPhase('report')
    } catch (err) {
      setError(err.message)
    } finally {
      setReporting(false)
    }
  }

  /* ── 工具函数 ──────────────────────────────────────────────────── */
  const formatTime = (s) => {
    const m = Math.floor(s / 60)
    const sec = s % 60
    return `${m.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`
  }

  const currentQ = questions[currentIdx]
  const currentQuestionType = currentQ?.type || config.type

  return (
    <div className="quiz-page">
      <header className="page-header">
        <h1 className="page-title">答题练习</h1>
        <p className="page-desc">选择文档和参数，AI 根据内容出题，答完即时反馈。</p>
      </header>

      {error && (
        <div className="error-banner" role="alert">
          <span>&#9888;</span>
          <span>{error}</span>
          <button className="error-close" aria-label="关闭错误提示" onClick={() => setError(null)}>&times;</button>
        </div>
      )}

      {/* ══════════════ Setup Phase ══════════════ */}
      {phase === 'setup' && docsError && (
        <div className="load-error-state" role="alert">
          <p className="state-title">无法加载文档列表</p>
          <p className="state-desc">{docsError}</p>
          <button type="button" className="state-action" onClick={loadDocuments}>
            重新加载文档
          </button>
        </div>
      )}

      {phase === 'setup' && !docsError && !docsLoading && documents.length === 0 && (
        <DocumentPrerequisite description="开始答题前，需要先上传一份学习材料供系统检索和出题。" />
      )}

      {phase === 'setup' && !docsError && (docsLoading || documents.length > 0) && (
        <div className="quiz-setup">
          {/* 文档选择 */}
          <div className="setup-field">
            <label className="field-label" htmlFor="quiz-document">学习文档</label>
            <select
              id="quiz-document"
              className="field-select"
              value={config.document_id}
              onChange={e => setConfig(c => ({ ...c, document_id: e.target.value }))}
              disabled={docsLoading}
            >
              <option value="">{docsLoading ? '加载中...' : '-- 选择文档 --'}</option>
              {documents.map(d => <option key={d} value={d}>{d}</option>)}
            </select>
          </div>

          {/* 出题主题 */}
          <div className="setup-field">
            <label className="field-label" htmlFor="quiz-topic">出题主题 <span className="field-hint">（可选，留空则覆盖全文）</span></label>
            <input
              id="quiz-topic"
              className="field-input"
              type="text"
              placeholder="例如：第三章 向量检索"
              value={config.description}
              onChange={e => setConfig(c => ({ ...c, description: e.target.value }))}
            />
          </div>

          {/* 题数 */}
          <div className="setup-field">
            <span id="quiz-count-label" className="field-label">题目数量</span>
            <div className="count-selector" role="group" aria-labelledby="quiz-count-label">
              {[3, 5, 8, 10].map(n => (
                <button
                  type="button"
                  key={n}
                  className={`count-btn ${config.count === n ? 'active' : ''}`}
                  aria-pressed={config.count === n}
                  onClick={() => setConfig(c => ({ ...c, count: n }))}
                >
                  {n} 题
                </button>
              ))}
            </div>
          </div>

          {/* 难度 */}
          <div className="setup-field">
            <span id="quiz-difficulty-label" className="field-label">难度</span>
            <div className="difficulty-selector" role="group" aria-labelledby="quiz-difficulty-label">
              {DIFFICULTY_OPTIONS.map(d => (
                <button
                  type="button"
                  key={d.value}
                  className={`diff-btn ${config.difficulty === d.value ? 'active' : ''}`}
                  aria-pressed={config.difficulty === d.value}
                  style={{ '--diff-color': d.color }}
                  onClick={() => setConfig(c => ({ ...c, difficulty: d.value }))}
                >
                  {d.label}
                </button>
              ))}
            </div>
          </div>

          {/* 题型 */}
          <div className="setup-field">
            <span id="quiz-type-label" className="field-label">题型</span>
            <div className="type-selector" role="group" aria-labelledby="quiz-type-label">
              {TYPE_OPTIONS.map(t => (
                <button
                  type="button"
                  key={t.value}
                  className={`type-btn ${config.type === t.value ? 'active' : ''}`}
                  aria-pressed={config.type === t.value}
                  onClick={() => setConfig(c => ({ ...c, type: t.value }))}
                >
                  {t.label}
                </button>
              ))}
            </div>
          </div>

          <button
            type="button"
            className="start-btn"
            onClick={handleStart}
            disabled={docsLoading || !config.document_id}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="5 3 19 12 5 21 5 3"/>
            </svg>
            开始答题
          </button>
        </div>
      )}

      {/* ══════════════ Loading Phase ══════════════ */}
      {phase === 'loading' && (
        <div className="quiz-loading" role="status" aria-live="polite">
          <div className="loading-spinner" />
          <p className="loading-text">AI 正在出题...</p>
          <p className="loading-hint">正在检索文档并生成题目，请稍候</p>
        </div>
      )}

      {/* ══════════════ Answering / Feedback Phase ══════════════ */}
      {(phase === 'answering' || phase === 'feedback') && currentQ && (
        <div className="quiz-active">
          {/* 进度条 + 计时 */}
          <div className="quiz-toolbar">
            <div
              className="progress-bar-wrapper"
              role="progressbar"
              aria-label="答题进度"
              aria-valuemin="0"
              aria-valuemax={questions.length}
              aria-valuenow={currentIdx + (phase === 'feedback' ? 1 : 0)}
            >
              <div
                className="progress-bar-fill"
                style={{ width: `${((currentIdx + (phase === 'feedback' ? 1 : 0)) / questions.length) * 100}%` }}
              />
            </div>
            <div className="toolbar-info">
              <span className="toolbar-progress">
                {currentIdx + 1} / {questions.length}
              </span>
              <span className="toolbar-timer">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                {formatTime(elapsed)}
              </span>
            </div>
          </div>

          {/* 题目卡片 */}
          <div className="question-card" key={currentIdx}>
            <p id={`quiz-question-${currentIdx}`} className="question-text">{currentQ.question}</p>

            {/* 选项列表 */}
            {currentQ.options && currentQ.options.length > 0 && (
              <div className="options-list" role="group" aria-labelledby={`quiz-question-${currentIdx}`}>
                {currentQ.options.map((opt, i) => {
                  const letter = String.fromCharCode(65 + i)
                  const isSelected = selectedAnswer === opt
                  const isCorrect = isCorrectOption(opt, feedback?.correct_answer)
                  let optClass = 'option-item'
                  if (phase === 'feedback') {
                    if (isCorrect) optClass += ' correct'
                    else if (isSelected && !feedback?.correct) optClass += ' wrong'
                    else optClass += ' disabled'
                  } else if (isSelected) {
                    optClass += ' selected'
                  }

                  return (
                    <button
                      type="button"
                      key={i}
                      className={optClass}
                      aria-pressed={isSelected}
                      onClick={() => phase === 'answering' && handleAnswerChange(opt)}
                      disabled={phase === 'feedback' || submitting}
                    >
                      <span className="option-letter">{letter}</span>
                      <span className="option-text">{opt}</span>
                      {phase === 'feedback' && isCorrect && (
                        <span className="option-check">&#10003;</span>
                      )}
                      {phase === 'feedback' && isSelected && !feedback?.correct && !isCorrect && (
                        <span className="option-cross">&#10007;</span>
                      )}
                    </button>
                  )
                })}
              </div>
            )}

            {/* 简答题输入 */}
            {currentQuestionType === 'short_answer' && (
              <textarea
                className="short-answer-input"
                aria-labelledby={`quiz-question-${currentIdx}`}
                placeholder="请输入你的答案..."
                value={selectedAnswer}
                onChange={e => handleAnswerChange(e.target.value)}
                disabled={phase === 'feedback' || submitting}
                rows={3}
              />
            )}

            {/* 反馈区域 */}
            {phase === 'feedback' && feedback && (
              <div className={`feedback-box ${feedback.correct ? 'correct' : 'wrong'}`}>
                <div className="feedback-header">
                  <span className="feedback-icon">
                    {feedback.correct ? '✓ 正确' : '✗ 错误'}
                  </span>
                </div>
                <p className="feedback-answer">
                  <strong>正确答案：</strong>{feedback.correct_answer}
                </p>
                <p className="feedback-explain">{feedback.explanation}</p>
              </div>
            )}

            {/* 操作按钮 */}
            <div className="question-actions">
              {phase === 'answering' ? (
                <button
                  className="submit-btn"
                  onClick={handleSubmit}
                  disabled={!selectedAnswer || submitting}
                >
                  {submitting ? '提交中...' : '提交答案'}
                </button>
              ) : (
                <button className="next-btn" onClick={handleNext}>
                  {feedback?.is_last ? '查看结果' : '下一题'}
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="9 18 15 12 9 6"/></svg>
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ══════════════ Results Phase ══════════════ */}
      {phase === 'results' && result && (
        <div className="quiz-results">
          {/* 得分卡片 */}
          <div className="result-score-card">
            <div className="score-ring">
              <svg viewBox="0 0 120 120" className="score-svg">
                <circle cx="60" cy="60" r="52" className="ring-bg" />
                <circle
                  cx="60" cy="60" r="52"
                  className="ring-fill"
                  style={{
                    strokeDasharray: `${2 * Math.PI * 52}`,
                    strokeDashoffset: `${2 * Math.PI * 52 * (1 - result.score)}`,
                  }}
                />
              </svg>
              <div className="score-text">
                <span className="score-number">{Math.round(result.score * 100)}</span>
                <span className="score-unit">分</span>
              </div>
            </div>
            <div className="score-details">
              <div className="score-detail-item">
                <span className="sd-value">{result.correct}</span>
                <span className="sd-label">正确</span>
              </div>
              <div className="score-detail-divider" />
              <div className="score-detail-item">
                <span className="sd-value">{result.total - result.correct}</span>
                <span className="sd-label">错误</span>
              </div>
              <div className="score-detail-divider" />
              <div className="score-detail-item">
                <span className="sd-value">{formatTime(elapsed)}</span>
                <span className="sd-label">用时</span>
              </div>
            </div>
          </div>

          <div className="result-actions">
            <button className="grade-btn" onClick={handleGrade} disabled={grading}>
              {grading ? (
                <><span className="btn-spinner" />批改中...</>
              ) : (
                <>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>
                  </svg>
                  AI 批改讲解
                </>
              )}
            </button>
            <button className="restart-btn" onClick={handleRestart}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>
              </svg>
              再来一轮
            </button>
          </div>
        </div>
      )}

      {/* ══════════════ Grading Phase ══════════════ */}
      {phase === 'grading' && gradingReport && (
        <div className="quiz-grading">
          <div className="grading-header">
            <h2 className="grading-title">AI 批改报告</h2>
            <div className="grading-summary">
              <span className="gs-score">{Math.round(gradingReport.score * 100)} 分</span>
              <span className="gs-detail">{gradingReport.correct}/{gradingReport.total} 正确</span>
            </div>
          </div>

          <div className="grade-list">
            {gradingReport.grades.map((g, idx) => (
              <div
                key={idx}
                className={`grade-item ${g.is_correct ? 'correct' : 'wrong'}`}
                style={{ animationDelay: `${idx * 0.06}s` }}
              >
                <div className="gi-header">
                  <span className={`gi-badge ${g.is_correct ? 'correct' : 'wrong'}`}>
                    {g.is_correct ? '✓ 正确' : '✗ 错误'}
                  </span>
                  <span className="gi-index">第 {g.index + 1} 题</span>
                </div>

                <p className="gi-question">{g.question}</p>

                <div className="gi-answers">
                  <div className={`gi-answer ${g.is_correct ? 'correct' : 'wrong'}`}>
                    <span className="gi-label">你的答案</span>
                    <span>{g.user_answer}</span>
                  </div>
                  {!g.is_correct && (
                    <div className="gi-answer correct">
                      <span className="gi-label">正确答案</span>
                      <span>{g.correct_answer}</span>
                    </div>
                  )}
                </div>

                {g.ai_feedback && (
                  <div className="gi-feedback">
                    <strong>AI 讲解：</strong>{g.ai_feedback}
                  </div>
                )}

                {g.knowledge_gap && (
                  <span className="gi-gap">
                    知识盲点：{g.knowledge_gap}
                  </span>
                )}
              </div>
            ))}
          </div>

          <div className="result-actions">
            <button className="report-btn" onClick={handleReport} disabled={reporting}>
              {reporting ? (
                <><span className="btn-spinner" />生成中...</>
              ) : (
                <>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
                  </svg>
                  学习评估报告
                </>
              )}
            </button>
            <button className="restart-btn" onClick={handleRestart}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>
              </svg>
              再来一轮
            </button>
          </div>
        </div>
      )}

      {/* ══════════════ Report Phase ══════════════ */}
      {phase === 'report' && learningReport && (
        <div className="quiz-report">
          <div className="report-header-card">
            <h2 className="report-title">学习评估报告</h2>
            <p className="report-summary">{learningReport.summary}</p>
            <div className="report-score">
              总分：<strong>{Math.round(learningReport.overall_score * 100)}</strong> / 100
            </div>
          </div>

          {/* 知识点掌握度 */}
          {learningReport.topic_mastery.length > 0 && (
            <div className="report-section">
              <h3 className="rs-title">知识点掌握度</h3>
              <div className="mastery-bars">
                {learningReport.topic_mastery.map((tm, i) => (
                  <div key={i} className="mastery-item" style={{ animationDelay: `${i * 0.08}s` }}>
                    <div className="mi-header">
                      <span className="mi-topic">{tm.topic}</span>
                      <span className="mi-pct">{Math.round(tm.mastery_pct)}%</span>
                    </div>
                    <div className="mi-bar-bg">
                      <div
                        className={`mi-bar-fill ${tm.mastery_pct >= 70 ? 'good' : 'weak'}`}
                        style={{ width: `${Math.min(tm.mastery_pct, 100)}%` }}
                      />
                    </div>
                    <span className="mi-detail">{tm.correct_count}/{tm.question_count} 正确</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* 优势 / 薄弱 */}
          <div className="report-row">
            {learningReport.strengths.length > 0 && (
              <div className="report-section half">
                <h3 className="rs-title strengths">&#10003; 掌握良好</h3>
                <div className="tag-list">
                  {learningReport.strengths.map((s, i) => (
                    <span key={i} className="tag good">{s}</span>
                  ))}
                </div>
              </div>
            )}
            {learningReport.weaknesses.length > 0 && (
              <div className="report-section half">
                <h3 className="rs-title weaknesses">&#9888; 需要加强</h3>
                <div className="tag-list">
                  {learningReport.weaknesses.map((w, i) => (
                    <span key={i} className="tag weak">{w}</span>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* 建议 */}
          {learningReport.recommendations.length > 0 && (
            <div className="report-section">
              <h3 className="rs-title">&#128161; 学习建议</h3>
              <ol className="rec-list">
                {learningReport.recommendations.map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ol>
            </div>
          )}

          <button className="restart-btn center" onClick={handleRestart}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>
            </svg>
            再来一轮
          </button>
        </div>
      )}
    </div>
  )
}
