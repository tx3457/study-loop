import { NavLink, Outlet } from 'react-router-dom'
import './Layout.css'

/* ── 侧边栏导航项 ─────────────────────────────────────────────────── */
const NAV_ITEMS = [
  { to: '/documents',     icon: '📄', label: '文档管理' },
  { to: '/learning-path', icon: '🗺️', label: '学习路径' },
  { to: '/quiz',          icon: '✏️',  label: '答题练习' },
  { to: '/autonomous',    icon: '🤖', label: 'Autonomous Agent' },
  { to: '/adaptive',      icon: '🎓', label: '自适应辅导' },
  { to: '/dashboard',     icon: '📊', label: '学习报告' },
]

export default function Layout() {
  return (
    <div className="layout">
      {/* ── 侧边栏 ──────────────────────────────────────────────────── */}
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="brand-icon">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/>
              <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>
            </svg>
          </div>
          <div>
            <h1 className="brand-title">StudyLoop</h1>
            <p className="brand-subtitle">AI Adaptive Learning</p>
          </div>
        </div>

        <nav className="sidebar-nav">
          {NAV_ITEMS.map(item => (
            <NavLink
              key={item.to}
              to={item.disabled ? '#' : item.to}
              className={({ isActive }) =>
                `nav-item ${isActive && !item.disabled ? 'active' : ''} ${item.disabled ? 'disabled' : ''}`
              }
              onClick={e => item.disabled && e.preventDefault()}
            >
              <span className="nav-icon">{item.icon}</span>
              <span className="nav-label">{item.label}</span>
              {item.disabled && <span className="nav-badge">Soon</span>}
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="footer-divider" />
          <p className="footer-text">Document-grounded learning</p>
        </div>
      </aside>

      {/* ── 主内容区 ────────────────────────────────────────────────── */}
      <main className="main-content">
        <Outlet />
      </main>
    </div>
  )
}
