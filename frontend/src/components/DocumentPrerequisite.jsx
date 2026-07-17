import { Link } from 'react-router-dom'

export default function DocumentPrerequisite({ description }) {
  return (
    <section className="prerequisite-state">
      <div className="prerequisite-icon" aria-hidden="true">
        <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
          <polyline points="14 2 14 8 20 8" />
          <path d="M12 18v-6m-3 3 3-3 3 3" />
        </svg>
      </div>
      <h2 className="prerequisite-title">先上传学习材料</h2>
      <p className="prerequisite-desc">{description}</p>
      <Link className="state-action" to="/documents">前往文档管理</Link>
    </section>
  )
}
