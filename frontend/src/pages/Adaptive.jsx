/**
 * Adaptive Learning 页面（Direction A：Agent 驱动的自适应学习闭环）
 *
 * 演示"会教书"的闭环 agent：
 *   start → agent 决策开场 + 出题 → 用户作答 → submit → 批改 + agent 决策下一步
 *   → 不达标则按决策(advance/remediate/continue)出下一轮 → 直到掌握度达标/练够/转规划
 *
 * 看点：每轮展示 agent 的 decision(action + reason)，以及掌握度/得分轨迹曲线。
 *
 * 状态机：idle → starting → answering → (submitting → answering)* → done | error
 */
import { useState, useRef } from 'react'
import { startAdaptive, submitAdaptive, getDocuments } from '../api/client'
import './Adaptive.css'

const ACTION_META = {
  advance:        { text: '升难度',     cls: 'act-advance' },
  teach:          { text: '讲解',       cls: 'act-teach' },
  remediate:      { text: '补薄弱点',   cls: 'act-remediate' },
  continue:       { text: '继续巩固',   cls: 'act-continue' },
  switch_to_plan: { text: '转学习路径', cls: 'act-plan' },
  finish:         { text: '结束',       cls: 'act-finish' },
}

function ActionBadge({ action }) {
  const m = ACTION_META[action] || { text: action, cls: 'act-continue' }
  return <span className={`act-badge ${m.cls}`}>{m.text}</span>
}

export default function Adaptive() {
  const [phase, setPhase] = useState('idle')   // idle | starting | answering | submitting | done | error
  const [req, setReq] = useState({ user_id: 'default_user', document_id: '', goal: '' })
  const [documents, setDocuments] = useState([])
  const [resp, setResp] = useState(null)        // 最近一轮的 turn 响应
  const [answers, setAnswers] = useState([])
  const [error, setError] = useState(null)
  const sid = useRef(null)

  const busy = phase === 'starting' || phase === 'submitting'

  async function loadDocs() {
    try {
      const data = await getDocuments()
      setDocuments(data.documents || [])
    } catch (err) {
      setError(err.message)
    }
  }

  /** 统一处理一轮响应:done / 讲解轮(reading)/ 出题轮(answering) */
  function applyTurn(r) {
    setResp(r)
    if (r.done) { setPhase('done'); return }
    if (r.turn_type === 'teach') {
      setPhase('reading')
    } else {
      setAnswers(new Array(r.questions?.length || 0).fill(''))
      setPhase('answering')
    }
  }

  async function handleStart(e) {
    e.preventDefault()
    if (!req.document_id.trim() || !req.goal.trim()) return
    setPhase('starting'); setError(null); setResp(null)
    try {
      const r = await startAdaptive({
        user_id: req.user_id || 'default_user',
        document_id: req.document_id.trim(),
        goal: req.goal.trim(),
      })
      sid.current = r.adaptive_session_id
      applyTurn(r)
    } catch (err) {
      setError(err.message); setPhase('error')
    }
  }

  async function handleSubmit() {
    if (answers.some(a => !a.trim())) return
    setPhase('submitting')
    try {
      const r = await submitAdaptive({ adaptive_session_id: sid.current, answers })
      applyTurn(r)
    } catch (err) {
      setError(err.message); setPhase('error')
    }
  }

  /** 讲解轮:读完点"继续"，提交空答案推进到出题验证 */
  async function handleContinueLesson() {
    setPhase('submitting')
    try {
      const r = await submitAdaptive({ adaptive_session_id: sid.current, answers: [] })
      applyTurn(r)
    } catch (err) {
      setError(err.message); setPhase('error')
    }
  }

  function setAnswer(i, val) {
    setAnswers(prev => { const next = [...prev]; next[i] = val; return next })
  }

  function handleReset() {
    setPhase('idle'); setResp(null); setAnswers([]); setError(null)
    sid.current = null
  }

  const r = resp
  const d = r?.decision

  return (
    <div className="adaptive-page">
      <header className="page-header">
        <h1 className="page-title">自适应辅导</h1>
        <p className="page-desc">
          系统会根据每轮答题表现调整难度、补充薄弱点或生成新的学习路径。
        </p>
      </header>

      {/* ── 开场输入区 ──────────────────────────────────────────────── */}
      <form className="adp-form" onSubmit={handleStart}>
        <div className="form-row">
          <label htmlFor="adaptive-goal">学习目标</label>
          <textarea
            id="adaptive-goal"
            rows={2}
            value={req.goal}
            onChange={e => setReq(s => ({ ...s, goal: e.target.value }))}
            placeholder="例如：快速排序与归并排序"
            disabled={phase !== 'idle' && phase !== 'error'}
          />
        </div>
        <div className="form-row form-row-inline">
          <div>
            <label htmlFor="adaptive-user">用户 ID</label>
            <input
              id="adaptive-user"
              value={req.user_id}
              onChange={e => setReq(s => ({ ...s, user_id: e.target.value }))}
              disabled={phase !== 'idle' && phase !== 'error'}
            />
          </div>
          <div className="doc-input">
            <div className="field-label-row">
              <label htmlFor="adaptive-document">文档 ID</label>
              <button type="button" className="btn-link" onClick={loadDocs}>刷新文档</button>
            </div>
            <input
              id="adaptive-document"
              list="adp-doc-list"
              value={req.document_id}
              onChange={e => setReq(s => ({ ...s, document_id: e.target.value }))}
              placeholder="选择已建库的文档"
              disabled={phase !== 'idle' && phase !== 'error'}
            />
            <datalist id="adp-doc-list">
              {documents.map(doc => <option key={doc} value={doc} />)}
            </datalist>
          </div>
        </div>
        <div className="form-actions">
          {(phase === 'idle' || phase === 'error') && (
            <button type="submit" className="btn-primary"
              disabled={!req.document_id.trim() || !req.goal.trim()}>
              开始自适应辅导
            </button>
          )}
          {(phase === 'done' || phase === 'answering' || phase === 'error') && (
            <button type="button" className="btn-ghost" onClick={handleReset}>重新开始</button>
          )}
          {phase === 'starting' && <span className="hint">Agent 决策开场中...</span>}
        </div>
      </form>

      {error && <div className="error-banner" role="alert">⚠️ {error}</div>}

      {/* ── 上一轮成绩 ──────────────────────────────────────────────── */}
      {r && r.last_report_score != null && (
        <div className="report-banner">
          上一轮得分 <b>{(r.last_report_score * 100).toFixed(0)}%</b>
          {r.mastery != null && <> ｜ 当前掌握度 <b>{(r.mastery * 100).toFixed(0)}%</b></>}
          {r.last_report_gaps?.length > 0 && (
            <span className="gaps">盲点：{r.last_report_gaps.join('、')}</span>
          )}
        </div>
      )}

      {/* ── 逐题反馈(grader 现成内容,别浪费)──────────────────────── */}
      {r?.last_report_feedback?.length > 0 && (
        <section className="feedback-section">
          <h3>🧾 上一轮逐题反馈</h3>
          {r.last_report_feedback.map((f, i) => (
            <div key={i} className={`fb-item ${f.is_correct ? 'fb-ok' : 'fb-bad'}`}>
              <div className="fb-head">
                <span className="fb-mark">{f.is_correct ? '✓' : '✗'}</span>
                <span className="fb-q">{f.index + 1}. {f.question}</span>
              </div>
              {!f.is_correct && (
                <div className="fb-body">
                  <div className="fb-line">你的答案：<b>{f.your_answer}</b> ｜ 正确：<b>{f.correct_answer}</b></div>
                  {f.ai_feedback && <div className="fb-ai">{f.ai_feedback}</div>}
                  {f.knowledge_gap && <div className="fb-gap">盲点：{f.knowledge_gap}</div>}
                </div>
              )}
            </div>
          ))}
        </section>
      )}

      {/* ── Agent 决策卡片(核心看点)──────────────────────────────── */}
      {d && (
        <div className="decision-card">
          <div className="decision-head">
            <span className="decision-title">🧠 Agent 决策</span>
            <ActionBadge action={d.action} />
            <span className="decision-meta">主题「{d.topic}」 ｜ 难度 {(d.difficulty_score ?? 0).toFixed(2)}</span>
          </div>
          {d.reason && <div className="decision-reason">{d.reason}</div>}
          {d.target_weak_points?.length > 0 && (
            <div className="decision-targets">针对薄弱点：{d.target_weak_points.join('、')}</div>
          )}
        </div>
      )}

      {/* ── 讲解轮(teach):纯讲解 + 继续 ───────────────────────────── */}
      {r?.turn_type === 'teach' && phase !== 'done' && r?.lesson && (
        <section className="lesson-section">
          <h3>📖 讲解：{d?.topic}</h3>
          <div className="lesson-body">{r.lesson}</div>
          <button className="btn-primary" onClick={handleContinueLesson} disabled={busy}>
            {phase === 'submitting' ? '出题中...' : '我懂了，出题验证 ⏎'}
          </button>
        </section>
      )}

      {/* ── 答题区 ──────────────────────────────────────────────────── */}
      {phase !== 'done' && r?.questions?.length > 0 && (
        <section className="quiz-section">
          <h3>📝 第 {r.turn} 轮 · 共 {r.questions.length} 题</h3>
          {r.questions.map((q, i) => (
            <div className="quiz-item" key={i}>
              <div id={`adaptive-question-${r.turn}-${i}`} className="quiz-q">{i + 1}. {q.question}</div>
              {q.options && q.options.length > 0 ? (
                <div
                  className="quiz-options"
                  role="radiogroup"
                  aria-labelledby={`adaptive-question-${r.turn}-${i}`}
                >
                  {q.options.map((opt, oi) => (
                    <label key={oi} className={`opt ${answers[i] === opt ? 'opt-sel' : ''}`}>
                      <input
                        type="radio"
                        name={`q-${r.turn}-${i}`}
                        checked={answers[i] === opt}
                        onChange={() => setAnswer(i, opt)}
                        disabled={busy}
                      />
                      {opt}
                    </label>
                  ))}
                </div>
              ) : (
                <input
                  id={`adaptive-answer-${r.turn}-${i}`}
                  className="quiz-text"
                  value={answers[i] || ''}
                  onChange={e => setAnswer(i, e.target.value)}
                  placeholder="输入你的答案"
                  disabled={busy}
                  aria-labelledby={`adaptive-question-${r.turn}-${i}`}
                />
              )}
            </div>
          ))}
          <button
            className="btn-primary"
            onClick={handleSubmit}
            disabled={busy || answers.length === 0 || answers.some(a => !a.trim())}
          >
            {phase === 'submitting' ? '批改 + 决策中...' : '提交本轮 ⏎'}
          </button>
        </section>
      )}

      {/* ── 结束总结 ────────────────────────────────────────────────── */}
      {phase === 'done' && r && (
        <section className="done-section">
          <h3>🏁 辅导结束</h3>
          <div className="done-summary">{r.summary}</div>
          {r.learning_path && (
            <div className="learning-path">
              <b>📚 转入的学习路径：</b>
              <pre>{JSON.stringify(r.learning_path, null, 2)}</pre>
            </div>
          )}
        </section>
      )}

      {/* ── 掌握度 / 得分轨迹 ──────────────────────────────────────── */}
      {r?.trajectory?.length > 0 && (
        <section className="traj-section">
          <h3>📈 学习轨迹</h3>
          <table className="traj-table">
            <thead>
              <tr><th>轮</th><th>决策</th><th>主题</th><th>难度</th><th>得分</th><th>掌握度</th></tr>
            </thead>
            <tbody>
              {r.trajectory.map((t, i) => (
                <tr key={i}>
                  <td>T{t.turn}</td>
                  <td><ActionBadge action={t.action} /></td>
                  <td>{t.topic}</td>
                  <td>{(t.difficulty_score ?? 0).toFixed(2)}</td>
                  <td>{t.score != null ? `${(t.score * 100).toFixed(0)}%` : '—'}</td>
                  <td>{t.mastery_after != null ? `${(t.mastery_after * 100).toFixed(0)}%` : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </div>
  )
}
