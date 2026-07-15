"""
Multi-Agent 共享状态定义（Phase 4 Day 15）

OrchestratorState  → 所有 Agent 共享的父状态
QuizAgentState     → QuizAgent 内部状态（extra 字段不传回 Orchestrator）
"""
from typing import TypedDict


class OrchestratorState(TypedDict, total=False):
    """主编排器共享状态（所有 Agent 均可读写）"""

    # ── 全局输入 ──────────────────────────────────────────
    action: str           # "quiz" | "grade" | "plan"
    user_id: str
    document_id: str
    description: str      # 出题主题 / 学习目标
    count: int
    difficulty: str       # easy / medium / hard（无用户画像时的 fallback）
    type: str             # choice / short_answer

    # ── AdaptAgent → QuizAgent ────────────────────────────
    difficulty_score: float   # 0.0-1.0 连续值（最近发展区 = mastery + 0.15）
    weak_points: list[str]    # 注入出题 prompt 的薄弱知识点

    # ── QuizAgent 输出 ─────────────────────────────────────
    quiz: dict | None

    # ── GraderAgent 输入 ───────────────────────────────────
    session_id: str | None    # 待批改的会话 ID（存在 services/session.py 的 sessions 里）

    # ── GraderAgent 输出 ───────────────────────────────────
    grading_report: dict | None

    # ── PlannerAgent 输出 ──────────────────────────────────
    learning_path: dict | None

    # ── CriticAgent 输出（T2 美团 JD 改造）─────────────────
    critique_history: list[dict]   # 每轮 CritiqueReport.model_dump() 累积
    revision_count: int            # Tutor 重出次数（上限 2）

    # ── Reflection 回灌（借鉴 aider/coders/base_coder.py:933-944 的 reflected_message）
    # critic 触发重出时,把 critique 格式化为反思文本塞给下一轮 quiz_agent.generate,
    # 让 LLM 知道"上一轮被拒的具体原因",而不只是简单重出。
    reflected_message: str

    # ── Sufficiency Check（P1-1 升级）──────────────────────
    # 上游检测到证据不充分时被设为 True，传给 critic 让 LLM judge 更严格
    insufficient_evidence: bool


class QuizAgentState(OrchestratorState, total=False):
    """QuizAgent 内部状态

    继承 OrchestratorState 的所有字段，额外增加检索/生成/审核的中间状态。
    这些 extra 字段只在 QuizAgent subgraph 内部流转，不会传回 Orchestrator。
    """
    chunks: list[str]      # hybrid_query_document 返回的文本块
    retrieve_count: int    # 重试计数（最多 3 次）
    generate_count: int    # 生成计数（审核不通过时重出，最多 2 次）
    review_passed: bool    # 审核结果

    # ── Sufficiency Check（P1-1 升级）───────────────────────
    sufficiency_passed: bool         # 当前 chunks 是否够生成高质题
    sufficiency_reason: str          # passed / passed_low_coverage / too_few_chunks / low_diversity
    rewrite_count: int               # query 改写次数（上限 MAX_REWRITES=1）


class TutorState(OrchestratorState, total=False):
    """Supervisor-based Multi-Agent 共享状态（Phase 1 骨架，灰度并存于旧 OrchestratorState 流之外）

    继承 OrchestratorState 的所有字段（action/user_id/document_id/description/count/
    difficulty/difficulty_score/weak_points/quiz/session_id/grading_report/learning_path/
    critique_history/revision_count/reflected_message/insufficient_evidence），
    额外增加 supervisor 编排所需的会话级字段。这些字段只在 tutor_graph 内部流转。
    """

    # ── 会话级（Phase 2 接 checkpointer 用 thread_id 续跑）──────
    thread_id: str               # 会话线程 ID（Phase 2 checkpointer key）
    goal: str                    # 学习目标（语义同 adaptive_loop 的 goal）
    mode: str                    # 编排模式：guided（引导式辅导）/ assist（开放问答）/ oneshot（单次）

    # ── Supervisor 编排状态 ────────────────────────────────────
    next_agent: str              # supervisor 决策的下一个 worker 节点名
    last_action: str             # 上一步执行的教学动作（advance/remediate/...）
    turn: int                    # 当前教学轮次（语义同 adaptive_loop 的 turn）
    history: list[dict]          # 轨迹记录（每轮一条 dict，便于压成 LLM 观察文本）
    last_report: dict | None     # 最近一次批改报告（GradingReport.model_dump()）
    allow_teach: bool            # 是否允许 teach（上一步刚讲过则 False，避免连续只讲不练）
    lesson: str | None           # tutor 节点产出的讲解文本
    answers: list[str]           # 学生本轮作答（供 grader）
    supervisor_reason: str       # supervisor 本次决策的可解释理由
    done: bool                   # 会话是否结束
    terminate_reason: str        # 终止原因（agent_finish/mastery_reached/max_turns/max_handoffs）
    handoff_count: int           # supervisor 累计 handoff 次数（上限 MAX_HANDOFFS，防失控循环）

    # ── supervisor 进度标记（避免空 history 时 cold-start 死循环）──────────────
    diagnosed: bool              # diagnostic 是否已跑过（supervisor 派 diagnostic 时置位，防反复诊断）

    # ── critic 质量门（Phase 2 纳入 supervisor 调度）────────────────────────────
    critic_passed: bool          # 当前 quiz 是否已通过 critic 审核（True 后 supervisor 才放行等待作答）
    quiz_served: bool            # 当前 quiz 是否已下发给学生并等过作答（防 grader 后又被当未审核 quiz 卡住）
    quiz_served_graded: bool     # 本轮下发题目是否已批改完成（grader_worker 置位，防 answers 残留触发重复批改）

    # ── assist 模式专用 ────────────────────────────────────────
    messages: list               # 对话消息列表（assistant 工具循环 / 多轮对话，持久化供 interrupt 重放）
    tools_called: list[str]      # 本会话已调用过的工具名（审计 / 防重复）
    awaiting_user_input: bool    # 是否在等用户输入（HITL：assist 模式下挂起等回复）
    final_answer: str            # assistant finalize 产出的面向用户最终回复
    assistant_done: bool         # assistant 是否已 finalize/截断收尾（供 supervisor 收尾）

    # ── 跨会话个人记忆（diagnostic_worker 产出，supervisor/assistant 读）────────────
    returning_context: dict      # 跨会话"欢迎回来"上下文（is_returning/last_score/mastery/趋势/top_weak/welcome_msg）
    preferences: dict            # 用户偏好快照（保留字段，便于注入/前端展示）
    memory_block: str            # <memory-context> 栅栏包裹的画像卡，注入 supervisor/assistant prompt
