# Design

## Source of truth

- Status: Active
- Last refreshed: 2026-07-16
- Primary product surfaces: 文档管理、学习路径、答题练习、Autonomous Agent、自适应辅导、学习报告
- Evidence reviewed: `frontend/src/components/Layout.jsx`、`frontend/src/components/Layout.css`、`frontend/src/pages/*`、`frontend/src/index.css`、桌面与 390px Playwright 页面截图

## Brand

- Personality: 安静、可信、专注学习，带轻量书房感
- Trust signals: 清晰的处理状态、来源明确的文档、可解释的学习结果
- Avoid: 过多 emoji、炫技动画、密集仪表盘、含糊的 AI 能力宣传

## Product goals

- Goals: 让用户顺利完成“上传材料 → 制定路径或练习 → 查看反馈”的学习闭环
- Non-goals: 通用聊天产品、复杂 LMS 后台、生产级多租户管理
- Success signals: 首次用户能找到下一步；错误可恢复；桌面和手机均可完成核心操作

## Personas and jobs

- Primary personas: 使用个人材料复习的学习者；审阅项目能力的技术评估者
- User jobs: 管理材料、规划学习、生成练习、查看掌握情况、运行辅助 Agent
- Key contexts of use: 桌面深度学习；手机查看状态和完成轻量操作

## Information architecture

- Primary navigation: 文档、学习路径、答题、自主 Agent、自适应辅导、报告
- Core routes/screens: `/documents`、`/learning-path`、`/quiz`、`/autonomous`、`/adaptive`、`/dashboard`
- Content hierarchy: 页面目标 → 当前操作 → 状态/错误 → 结果 → 次要技术信息

## Design principles

- Next action first: 每个页面首先说明当前可以完成什么
- Consistent shell: 标题、说明、表单、按钮和反馈使用一致层级
- Progressive disclosure: 技术轨迹和高级选项不压过主要学习任务
- Tradeoffs: 保留现有书房视觉与纯 CSS 实现，优先修复结构和可用性

## Visual language

- Color: 暖白背景、深海军蓝导航、蓝色主操作、金色品牌点缀
- Typography: 衬线字体用于品牌和标题，无衬线字体用于正文和控件
- Spacing/layout rhythm: 4/8px 节奏；内容宽度按任务保持 800–960px
- Shape/radius/elevation: 6–16px 圆角；低对比边框和轻阴影
- Motion: 仅用于状态变化，尊重 `prefers-reduced-motion`
- Imagery/iconography: 使用一致的线性 SVG 图标；emoji 只用于内容语义而非主导航

## Components

- Existing components to reuse: `Layout`、页面 header、卡片、表单控件、错误提示
- New/changed components: 响应式导航抽屉、移动端顶栏、统一导航图标
- Variants and states: 默认、hover、focus-visible、active、disabled、loading、error
- Token/component ownership: 全局 token 与共享状态在 `index.css`；布局在 `Layout.css`；页面细节留在页面 CSS

## Accessibility

- Target standard: 以 WCAG 2.1 AA 为方向；当前验证覆盖键盘、焦点与基础语义，不替代完整合规审计
- Keyboard/focus behavior: 导航、上传区和全部按钮可用键盘操作；使用明确 `:focus-visible`
- Contrast/readability: 正文与控件文字保持可读对比度，不只依赖颜色表达状态
- Screen-reader semantics: 使用 landmark、可读标签、`aria-live` 和装饰图标隐藏
- Reduced motion and sensory considerations: 关闭非必要动画与位移

## Responsive behavior

- Supported breakpoints/devices: 390px 手机至桌面；主断点 768px
- Layout adaptations: 桌面固定侧栏；手机使用顶栏和可关闭的导航抽屉
- Touch/hover differences: 触控目标至少约 44px；核心信息不依赖 hover

## Interaction states

- Loading: 保留布局并显示进度或 skeleton
- Empty: 解释为什么为空，并给出直接下一步
- Error: 靠近相关操作显示，可关闭或重试
- Success: 明确完成内容并刷新依赖数据
- Disabled: 说明前置条件，不使用视觉上类似可点击的状态
- Offline/slow network, if applicable: 长模型请求保持 busy 状态，失败后保留用户输入

## Content voice

- Tone: 简洁、直接、鼓励但不过度拟人化
- Terminology: 面向用户使用中文；必要时保留 Agent、Embedding 等行业术语
- Microcopy rules: 描述动作与结果，不描述内部实现过程；错误信息给出下一步

## Implementation constraints

- Framework/styling system: React 19、React Router、纯 CSS
- Design-token constraints: 复用 `index.css` 变量，不引入第二套 token
- Performance constraints: 不增加图标库或 UI 框架；避免阻塞首屏的远程资源
- Compatibility constraints: Chromium 与现代桌面/移动浏览器
- Test/screenshot expectations: lint/build 通过；`npm run test:e2e` 覆盖桌面、390px 移动端和核心交互

## Open questions

- [ ] 是否将多用户登录作为产品目标 / owner: product / impact: 导航和数据隔离
- [ ] 是否需要公开部署域名与品牌素材 / owner: product / impact: 品牌与部署配置
- [x] Playwright E2E 已固化进 CI，使用 Mock API 避免 Provider 成本和外部波动
