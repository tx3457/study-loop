import { useCallback, useEffect, useRef, useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import './Layout.css'

const NAV_ITEMS = [
  { to: '/documents', icon: 'document', label: '文档管理' },
  { to: '/learning-path', icon: 'path', label: '学习路径' },
  { to: '/quiz', icon: 'quiz', label: '答题练习' },
  { to: '/autonomous', icon: 'agent', label: '自主 Agent' },
  { to: '/adaptive', icon: 'adaptive', label: '自适应辅导' },
  { to: '/dashboard', icon: 'report', label: '学习报告' },
]

const ICONS = {
  document: <><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M8 13h8M8 17h8"/></>,
  path: <><circle cx="6" cy="18" r="2"/><circle cx="18" cy="6" r="2"/><path d="M8 18h3a3 3 0 0 0 3-3V9a3 3 0 0 1 3-3"/></>,
  quiz: <><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L8 18l-4 1 1-4z"/></>,
  agent: <><rect x="4" y="7" width="16" height="13" rx="3"/><path d="M9 11h.01M15 11h.01M8 16h8M12 7V3M9 3h6"/></>,
  adaptive: <><path d="m3 8 9-5 9 5-9 5z"/><path d="M7 10.5V15c0 1.7 2.2 3 5 3s5-1.3 5-3v-4.5M21 8v6"/></>,
  report: <><path d="M4 19V9M10 19V5M16 19v-7M22 19H2"/></>,
}

function LineIcon({ name, size = 20 }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {ICONS[name]}
    </svg>
  )
}

function BrandMark({ size = 28 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/>
      <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>
    </svg>
  )
}

export default function Layout() {
  const [mobileOpen, setMobileOpen] = useState(false)
  const [isMobile, setIsMobile] = useState(() =>
    typeof window !== 'undefined' && window.matchMedia('(max-width: 768px)').matches
  )
  const menuButtonRef = useRef(null)
  const sidebarRef = useRef(null)
  const closeButtonRef = useRef(null)

  const closeMobileNavigation = useCallback(() => {
    setMobileOpen(false)
    window.requestAnimationFrame(() => menuButtonRef.current?.focus())
  }, [])

  useEffect(() => {
    const query = window.matchMedia('(max-width: 768px)')
    const syncViewport = () => {
      setIsMobile(query.matches)
      if (!query.matches) setMobileOpen(false)
    }
    query.addEventListener('change', syncViewport)
    return () => query.removeEventListener('change', syncViewport)
  }, [])

  useEffect(() => {
    if (!isMobile || !mobileOpen) return undefined

    const previousOverflow = document.body.style.overflow
    const focusFrame = window.requestAnimationFrame(() => closeButtonRef.current?.focus())
    const handleKeyDown = (event) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        closeMobileNavigation()
        return
      }

      if (event.key === 'Tab') {
        const sidebar = sidebarRef.current
        if (!sidebar) return
        const focusable = [...sidebar.querySelectorAll(
          'a[href], button:not([disabled]), [tabindex]:not([tabindex="-1"])'
        )]
        const first = focusable[0]
        const last = focusable.at(-1)

        if (!first || !last) return
        if (event.shiftKey && (document.activeElement === first || !sidebar.contains(document.activeElement))) {
          event.preventDefault()
          last.focus()
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault()
          first.focus()
        }
      }
    }

    document.body.style.overflow = 'hidden'
    document.addEventListener('keydown', handleKeyDown)
    return () => {
      window.cancelAnimationFrame(focusFrame)
      document.body.style.overflow = previousOverflow
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [closeMobileNavigation, isMobile, mobileOpen])

  const navigationHidden = isMobile && !mobileOpen

  return (
    <div className="layout">
      <a className="skip-link" href="#main-content">跳到主要内容</a>

      <header
        className="mobile-header"
        aria-hidden={isMobile && mobileOpen ? true : undefined}
        inert={isMobile && mobileOpen}
      >
        <NavLink className="mobile-brand" to="/documents" aria-label="StudyLoop 首页" onClick={() => setMobileOpen(false)}>
          <span className="mobile-brand-mark"><BrandMark size={22} /></span>
          <span>StudyLoop</span>
        </NavLink>
        <button
          ref={menuButtonRef}
          type="button"
          className="menu-toggle"
          aria-label="打开导航"
          aria-expanded={mobileOpen}
          aria-controls="primary-navigation"
          onClick={() => setMobileOpen(true)}
        >
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
            <path d="M4 7h16M4 12h16M4 17h16" />
          </svg>
        </button>
      </header>

      <button
        type="button"
        className={`sidebar-backdrop ${mobileOpen ? 'visible' : ''}`}
        aria-label="关闭导航"
        aria-hidden="true"
        tabIndex={-1}
        onClick={closeMobileNavigation}
      />

      <aside
        ref={sidebarRef}
        id="primary-navigation"
        className={`sidebar ${mobileOpen ? 'open' : ''}`}
        role={isMobile ? 'dialog' : undefined}
        aria-modal={isMobile && mobileOpen ? true : undefined}
        aria-label="主要导航"
        aria-hidden={navigationHidden ? true : undefined}
        inert={navigationHidden}
      >
        <div className="sidebar-brand">
          <div className="brand-icon"><BrandMark /></div>
          <div className="brand-copy">
            <h1 className="brand-title">StudyLoop</h1>
            <p className="brand-subtitle">AI Adaptive Learning</p>
          </div>
          <button
            ref={closeButtonRef}
            type="button"
            className="mobile-close"
            aria-label="关闭导航"
            tabIndex={navigationHidden ? -1 : undefined}
            onClick={closeMobileNavigation}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
              <path d="m6 6 12 12M18 6 6 18" />
            </svg>
          </button>
        </div>

        <nav className="sidebar-nav" aria-label="学习功能">
          {NAV_ITEMS.map(item => (
            <NavLink
              key={item.to}
              to={item.to}
              tabIndex={navigationHidden ? -1 : undefined}
              onClick={isMobile ? closeMobileNavigation : undefined}
              className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
            >
              <span className="nav-icon"><LineIcon name={item.icon} /></span>
              <span className="nav-label">{item.label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="footer-divider" />
          <p className="footer-text">Document-grounded learning</p>
        </div>
      </aside>

      <main
        id="main-content"
        className="main-content"
        aria-hidden={isMobile && mobileOpen ? true : undefined}
        inert={isMobile && mobileOpen}
      >
        <Outlet />
      </main>
    </div>
  )
}
