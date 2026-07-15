/**
 * Autonomous Agent 页面
 *
 * 演示真 ReAct + HITL：
 *   - 用户输入学习目标
 *   - Agent 自主决定调工具 / finalize / 问用户
 *   - LLM 调 ask_user → 弹框让用户答 → 提交后调 /continue 续跑
 *
 * 状态机：
 *   idle → running → (awaiting → running)* → done | error
 */
import { useState, useRef } from 'react'
import {
  runAutonomous,
  continueAutonomous,
  getDocuments,
} from '../api/client'
import './Autonomous.css'

const initialState = {
  phase: 'idle',     // idle | running | awaiting | done | error
  request: { query: '', user_id: 'default_user', document_id: '' },
  response: null,
  error: null,
}

export default function Autonomous() {
  const [state, setState] = useState(initialState)
  const [askReply, setAskReply] = useState('')
  const [documents, setDocuments] = useState([])
  const lastConversationId = useRef(null)

  /** 处理 agent 响应：分发到对应状态 */
  function _handleResponse(resp) {
    if (resp.awaiting_user_input) {
      lastConversationId.current = resp.conversation_id
      setState(s => ({ ...s, phase: 'awaiting', response: resp }))
      setAskReply('')
    } else {
      setState(s => ({ ...s, phase: 'done', response: resp }))
    }
  }

  /** 首次提交 */
  async function handleStart(e) {
    e.preventDefault()
    if (!state.request.query.trim()) return
    setState(s => ({ ...s, phase: 'running', response: null, error: null }))
    try {
      const resp = await runAutonomous({
        query: state.request.query,
        user_id: state.request.user_id || 'default_user',
        document_id: state.request.document_id || null,
      })
      _handleResponse(resp)
    } catch (err) {
      setState(s => ({ ...s, phase: 'error', error: err.message }))
    }
  }

  /** 提交 ask_user 回答续跑 */
  async function handleContinue() {
    if (!askReply.trim() || !lastConversationId.current) return
    setState(s => ({ ...s, phase: 'running' }))
    try {
      const resp = await continueAutonomous({
        conversation_id: lastConversationId.current,
        user_reply: askReply.trim(),
      })
      _handleResponse(resp)
    } catch (err) {
      setState(s => ({ ...s, phase: 'error', error: err.message }))
    }
  }

  function handleReset() {
    setState(initialState)
    setAskReply('')
    lastConversationId.current = null
  }

  async function loadDocs() {
    try {
      const data = await getDocuments()
      setDocuments(data.documents || [])
    } catch (err) {
      setState(s => ({ ...s, error: err.message }))
    }
  }

  const r = state.response

  return (
    <div className="autonomous-page">
      <header className="page-header">
        <h2>🤖 Autonomous Agent</h2>
        <p className="page-sub">
          ReAct 范式：Agent 自主决定调工具 / 询问你 / 给最终答案
        </p>
      </header>

      {/* ── 输入区 ──────────────────────────────────────────────────────── */}
      <form className="auto-form" onSubmit={handleStart}>
        <div className="form-row">
          <label>你的学习目标</label>
          <textarea
            rows={3}
            value={state.request.query}
            onChange={e => setState(s => ({ ...s, request: { ...s.request, query: e.target.value } }))}
            placeholder="例如：帮我规划学习 RAG 的路径，然后出 3 道选择题"
            disabled={state.phase === 'running' || state.phase === 'awaiting'}
          />
        </div>

        <div className="form-row form-row-inline">
          <div>
            <label>用户 ID</label>
            <input
              value={state.request.user_id}
              onChange={e => setState(s => ({ ...s, request: { ...s.request, user_id: e.target.value } }))}
              disabled={state.phase === 'running' || state.phase === 'awaiting'}
            />
          </div>
          <div className="doc-input">
            <label>
              文档 ID（可选 — Agent 缺时会主动问你）
              <button type="button" className="btn-link" onClick={loadDocs}>↻ 刷新列表</button>
            </label>
            <input
              list="doc-list"
              value={state.request.document_id}
              onChange={e => setState(s => ({ ...s, request: { ...s.request, document_id: e.target.value } }))}
              placeholder="留空让 Agent 主动询问"
              disabled={state.phase === 'running' || state.phase === 'awaiting'}
            />
            <datalist id="doc-list">
              {documents.map(d => <option key={d} value={d} />)}
            </datalist>
          </div>
        </div>

        <div className="form-actions">
          <button
            type="submit"
            className="btn-primary"
            disabled={state.phase === 'running' || state.phase === 'awaiting' || !state.request.query.trim()}
          >
            {state.phase === 'running' ? '执行中...' : '🚀 开始执行'}
          </button>
          {(state.phase === 'done' || state.phase === 'error') && (
            <button type="button" className="btn-ghost" onClick={handleReset}>重新开始</button>
          )}
        </div>
      </form>

      {/* ── 执行进度 ────────────────────────────────────────────────────── */}
      {r && (
        <div className="auto-result">
          {/* Plan */}
          {r.plan && r.plan.length > 0 && (
            <section className="result-section">
              <h3>📋 Plan ({r.plan.length} 步)</h3>
              <ol className="plan-list">
                {r.plan.map((step, i) => <li key={i}>{step}</li>)}
              </ol>
            </section>
          )}

          {/* Steps */}
          {r.steps && r.steps.length > 0 && (
            <section className="result-section">
              <h3>⚙️ 执行步骤 ({r.steps.length})</h3>
              <ul className="step-list">
                {r.steps.map((s, i) => (
                  <li
                    key={i}
                    className={`step-item ${s.blocked_reason ? 'step-blocked' : ''} ${s.tool_name === 'finalize' ? 'step-finalize' : ''} ${s.tool_name === 'ask_user' ? 'step-ask' : ''}`}
                  >
                    <div className="step-head">
                      <span className="step-round">R{s.round_index + 1}</span>
                      <span className="step-tool">{s.tool_name || '(无)'}</span>
                      {s.blocked_reason && <span className="step-tag tag-blocked">⛔ {s.blocked_reason}</span>}
                      {s.tool_name === 'finalize' && <span className="step-tag tag-finalize">✅ 完成</span>}
                      {s.tool_name === 'ask_user' && <span className="step-tag tag-ask">🤔 询问</span>}
                    </div>
                    {s.tool_args && Object.keys(s.tool_args).length > 0 && (
                      <pre className="step-args">{JSON.stringify(s.tool_args, null, 2)}</pre>
                    )}
                    {s.observation_preview && (
                      <div className="step-obs">{s.observation_preview}</div>
                    )}
                  </li>
                ))}
              </ul>
            </section>
          )}

          {/* Final answer */}
          {r.final_answer && (
            <section className="result-section result-final">
              <h3>📝 最终回复</h3>
              <div className="final-answer">{r.final_answer}</div>
              {r.finalize_reason && (
                <div className="final-meta">
                  结束原因：<code>{r.finalize_reason}</code>
                </div>
              )}
            </section>
          )}

          {/* Meta */}
          <section className="result-section">
            <h3>📊 元信息</h3>
            <div className="meta-grid">
              <div><span className="meta-key">轮次</span><span>{r.rounds_used}</span></div>
              <div><span className="meta-key">truncated</span><span>{r.truncated ? '是' : '否'}</span></div>
              <div><span className="meta-key">工具调用</span><span>{(r.tools_called || []).join(', ') || '无'}</span></div>
            </div>
          </section>
        </div>
      )}

      {/* ── 错误显示 ────────────────────────────────────────────────────── */}
      {state.phase === 'error' && (
        <div className="error-banner">⚠️ {state.error}</div>
      )}

      {/* ── HITL Modal：ask_user 弹框 ───────────────────────────────────── */}
      {state.phase === 'awaiting' && r?.user_question && (
        <div className="modal-overlay" role="dialog">
          <div className="modal-card">
            <div className="modal-header">🤔 Agent 想问你</div>
            <div className="modal-question">{r.user_question}</div>
            <textarea
              className="modal-input"
              rows={3}
              value={askReply}
              onChange={e => setAskReply(e.target.value)}
              placeholder="输入你的回答..."
              autoFocus
              onKeyDown={e => {
                if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) handleContinue()
              }}
            />
            <div className="modal-actions">
              <button className="btn-ghost" onClick={handleReset}>取消整个执行</button>
              <button
                className="btn-primary"
                onClick={handleContinue}
                disabled={!askReply.trim()}
              >
                回答 ⏎ (Ctrl+Enter)
              </button>
            </div>
            <div className="modal-meta">
              conversation_id: <code>{lastConversationId.current?.slice(0, 16)}...</code>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
