"""
TeachingSupervisor：Supervisor-based Multi-Agent 的决策大脑

和现有 orchestrator.py 的本质区别：
  规则 workflow：input_guard → _route(if/else 按 action 硬分流) → worker...
                 → critic → _should_revise(阈值) → 重出 or 收尾
  LLM supervisor：input_guard → teaching_supervisor(LLM 动态编排) ⇄ worker...
                  supervisor 每轮看"完整观察"（mode/turn/已完成 worker/批改/critique/轨迹）
                  推理出"下一步派哪个 worker、做什么动作"，worker 干完回流 supervisor 再决策，
                  直到 finish / 达到 MAX_HANDOFFS。

通过 env MAS_SUPERVISOR_ENABLED 控制是否启用（默认 false），与现有
orchestrator/worker/routers 链路灰度并存。

健壮性（fail-soft，沿用 adaptive_loop 的精神）：
  supervisor_mode=="rule"（ablation）或 llm_parse 任何异常 → _rule_fallback_next 规则兜底，
  确定性复刻 orchestrator._route / _should_revise 阈值 / adaptive_loop._rule_fallback 的难度逻辑，
  保证一次 LLM 抽风不会打断整个辅导会话。
"""
import logging
import os

from langgraph.types import Command

from models.supervisor import SupervisorDecision
from services.adaptive_loop import MASTERY_TARGET
from services.llm import llm_parse

logger = logging.getLogger(__name__)

# ── 编排参数 ─────────────────────────────────────────────────────────────────
MAX_HANDOFFS = 8                 # supervisor 最多 8 次 handoff，防 LLM 编排失控无限循环
MAX_REVISIONS = 2                # critic↔reviser 精修上限（复刻 orchestrator._should_revise 的 revision_count<2）

# 合法枚举（_normalize_decision 用）
# critic（质量门）/ reviser（精修）/ await_answers（等待作答 interrupt）均纳入
# supervisor 可调度范围。await_answers 是内部编排目标（不出题、不批改，
# 仅声明"quiz 已过审 → 去等学生作答"），不会暴露给 LLM 让它乱填。
_AGENTS = {"diagnostic", "planner", "quiz", "grader", "tutor", "assistant",
           "critic", "reviser", "await_answers", "finish"}
_ACTIONS = {"advance", "remediate", "continue", "switch_to_plan", "finish"}
_DIFF_WORDS = {"easy", "medium", "hard"}
_TYPES = {"choice", "true_false", "short_answer"}

# next_agent → tutor_graph 里的节点名映射（节点名与 next_agent 多数同名，留映射便于演进）
_AGENT_TO_NODE = {
    "diagnostic": "diagnostic",
    "planner": "planner",
    "quiz": "quiz",
    "grader": "grader",
    "tutor": "tutor",
    "assistant": "assistant",
    "critic": "critic",
    "reviser": "reviser",
    "await_answers": "wait_for_answers",   # quiz 过审 → 暂停等学生作答（HITL interrupt）
}
_FINISH_NODE = "output_guard"     # finish / done / 超限 → 走出口 guard 收尾（tutor_graph 里是 tutor_output_guard 节点）


# ── env 开关（沿用项目惯例 .lower() in ("1","true","yes")）──────────────────────
def supervisor_enabled() -> bool:
    """MAS_SUPERVISOR_ENABLED=true → 启用 supervisor-based MAS（默认 false，灰度）。"""
    return os.getenv("MAS_SUPERVISOR_ENABLED", "false").lower() in ("1", "true", "yes")


def supervisor_mode() -> str:
    """SUPERVISOR_MODE：'llm'（默认，LLM 动态编排）/ 'rule'（ablation，纯规则兜底）。"""
    mode = os.getenv("SUPERVISOR_MODE", "llm").lower()
    return mode if mode in ("llm", "rule") else "llm"


# ── Supervisor 系统 prompt（参考 adaptive_loop._DECIDE_SYSTEM 的教学语义）────────
_SUPERVISOR_SYSTEM = (
    "你是一个教学主管（TeachingSupervisor），负责动态编排一组专职 worker 来辅导学生。"
    "你每一步要根据『当前观察』（模式 / 轮次 / 已完成的 worker / 最近批改结果 / 题目质量审核 / 历史轨迹）"
    "推理出『下一步派哪个 worker、做什么教学动作』。你不是写死的 if/else 路由，要像真人教研主管一样判断。\n\n"
    "可调度的 worker：\n"
    "- diagnostic：诊断学情，读用户画像 / 掌握度 / 薄弱点（会话开场或需要重新评估时派它）\n"
    "- quiz：出题（hybrid 检索 + 生成 + 质量审核）\n"
    "- critic：题目质量审核（出题后必须先过这道质量门，三维评分：难度/相关性/覆盖度）\n"
    "- reviser：题目精修（critic 不通过且未到精修上限时，只改有问题的题，保留好题）\n"
    "- grader：批改学生本轮作答，产出得分与逐题盲点\n"
    "- tutor：纯讲解（学生没懂、反复错同一点时，先讲清概念再出题验证）\n"
    "- planner：生成系统学习路径（知识缺口系统性、零散补救无效时）\n"
    "- assistant：开放式问答 / 工具调用（学生主动提问、闲聊式 assist 时）\n"
    "- finish：会话结束 / 收尾\n\n"
    "决策原则（与自适应教学语义一致）：\n"
    "- 会话刚开始、还没诊断 → 先 diagnostic 摸清学情\n"
    "- 已诊断、该出题练 → quiz（action=continue/advance/remediate 控制难度走向）\n"
    "- 刚出完题、还没审核 → critic 审核题目质量（这一步不能跳过）\n"
    "- critic 低分（overall<0.7 或含 high severity）且精修次数 <2 → reviser 精修后再回 critic 重审；\n"
    "  critic 通过 → 让系统下发题目并等学生作答（不要再继续出题）\n"
    "- 学生交了作答、还没批改 → grader\n"
    "- 学生在某知识点反复错、光换难度没用 → tutor 先讲（action=remediate 语义的『讲』）\n"
    "- 连续答得好、掌握度高 → quiz + action=advance 升难度；已达标/练够 → finish\n"
    "- 知识缺口系统性、零散补救无效 → planner（action=switch_to_plan）\n\n"
    "必须给出 reason（你的编排理由，一两句话）。topic 用中文关键词；"
    "target_weak_points 从学生已暴露的盲点里选；difficulty_score 用 0-1 连续值。"
)


def _summarize_report(report: dict | None) -> str:
    """把批改报告压成一行摘要（得分 + 盲点），喂给 supervisor 观察文本。"""
    if not report:
        return "（暂无批改结果）"
    score = report.get("score")
    correct = report.get("correct")
    total = report.get("total")
    gaps: list[str] = []
    for g in report.get("grades", []) or []:
        if isinstance(g, dict) and not g.get("is_correct") and g.get("knowledge_gap"):
            gaps.append(str(g["knowledge_gap"]))
    score_str = f"{score:.2f}" if isinstance(score, (int, float)) else "?"
    gaps_str = ("，盲点：" + "、".join(gaps[:5])) if gaps else ""
    return f"得分 {correct}/{total} = {score_str}{gaps_str}"


def _summarize_critique(critique_history: list[dict]) -> str:
    """把最近一次 critique 压成一行摘要（整体分 + 是否含 high severity 建议）。"""
    if not critique_history:
        return "（暂无题目质量审核）"
    latest = critique_history[-1] or {}
    overall = latest.get("overall_score", 1.0)
    has_high = any(
        isinstance(s, dict) and s.get("severity") == "high"
        for s in latest.get("suggestions", []) or []
    )
    return f"题目质量整体分 {overall:.2f}（阈值 0.7）{'，含 high severity 建议' if has_high else ''}"


def _summarize_history(history: list[dict]) -> str:
    """把轨迹压成多行（第N步: agent/action/topic）。"""
    if not history:
        return "（暂无历史）"
    lines = []
    for i, h in enumerate(history, 1):
        agent = h.get("next_agent") or h.get("agent") or "?"
        action = h.get("action") or h.get("last_action") or "?"
        topic = h.get("topic") or ""
        lines.append(f"  第{i}步: {agent} | 动作 {action} | 主题「{topic}」")
    return "\n".join(lines)


def _build_observation(state: dict) -> str:
    """把当前会话状态压成 LLM 观察文本（mode/turn/已完成 worker/批改/critique/轨迹）。"""
    mode = state.get("mode", "guided")
    turn = state.get("turn", 0)
    goal = state.get("goal") or state.get("description") or ""
    weak_points = state.get("weak_points", []) or []
    history = state.get("history", []) or []
    completed = sorted({h.get("next_agent") or h.get("agent") for h in history if isinstance(h, dict)} - {None})

    parts = [
        f"学习目标: {goal}",
        f"编排模式: {mode}",
        f"当前轮次: {turn}（handoff 已用 {state.get('handoff_count', 0)}/{MAX_HANDOFFS}）",
        f"已完成的 worker: {completed if completed else '（无）'}",
    ]
    if weak_points:
        parts.append(f"已知薄弱点: {', '.join(weak_points[:8])}")
    rc = state.get("returning_context") or {}
    if rc.get("is_returning") and rc.get("welcome_msg"):
        parts.append("跨会话记忆: " + rc["welcome_msg"])
    parts.append("最近批改结果: " + _summarize_report(state.get("last_report")))
    parts.append("题目质量审核: " + _summarize_critique(state.get("critique_history", []) or []))
    parts.append("历史轨迹:\n" + _summarize_history(history))
    if not state.get("allow_teach", True):
        parts.append("注意：上一步已经讲解过了，这一步请用出题验证（不要再派 tutor 讲）。")
    return "\n\n".join(parts)


def _critic_verdict(state: dict) -> tuple[bool, float, bool]:
    """读最新一条 critique，返回 (有 critique, overall_score, 是否含 high severity)。

    复刻 orchestrator._should_revise 的判定语义（overall<0.7 或含 high severity → 需精修）。
    """
    history_critique = state.get("critique_history", []) or []
    if not history_critique:
        return False, 1.0, False
    latest = history_critique[-1] or {}
    overall = latest.get("overall_score", 1.0)
    has_high = any(
        isinstance(s, dict) and s.get("severity") == "high"
        for s in latest.get("suggestions", []) or []
    )
    return True, overall, has_high


def _rule_fallback_next(state: dict) -> SupervisorDecision:
    """规则兜底（fail-soft）：确定性复刻 quiz→critic→(reviser↔critic)→await_answers→grader 全链路。

    复刻来源：
      1) orchestrator._route：按 action 字段硬分流（plan→planner / grade→grader）
      2) 质量门：quiz 出题后 → critic 审核；critic 低分(overall<0.7/high severity)且
         revision_count<MAX_REVISIONS → reviser 精修后回 critic 重审；通过 → await_answers 等作答
      3) orchestrator._should_revise：critic↔reviser 精修上限 revision_count<2
      4) adaptive_loop._rule_fallback：开场 medium 巩固；有结果时 score<0.5 降难度补薄弱点，否则升难度

    确定性判定优先级（从最贴近"当前刚发生什么"往回推）：
      A. 学生已交作答但还没批改 → grader
      B. 有 quiz 还没过 critic（critic_passed 未置位）→ critic / reviser
      C. quiz 已过 critic 还没下发 → await_answers（去 wait_for_answers 暂停等作答）
      D. 显式 action=plan → planner
      E. 无历史 → diagnostic 冷启动
      F. 进入新一轮出题（按 last_report 难度走向）→ quiz
    """
    weak_points = state.get("weak_points", []) or []
    goal = state.get("goal") or state.get("description") or ""
    history = state.get("history", []) or []
    last_report = state.get("last_report")
    quiz = state.get("quiz")

    # (A) 学生已交作答但还没批改 → grader（answers 由 wait_for_answers interrupt 恢复后写入）
    action = state.get("action")
    if action == "grade" or (state.get("answers") and not state.get("quiz_served_graded")):
        return SupervisorDecision(
            next_agent="grader", action="continue", topic=goal,
            reason="规则兜底：有待批改作答 → grader",
        )

    # (B)(C) 有 quiz 时走质量门：critic → (reviser↔critic) → await_answers
    #   critic_passed 由 critic 节点在"通过"时置位；quiz_served 由 wait_for_answers 置位。
    #   只有"未下发的新 quiz"才进质量门，已下发并批改过的旧 quiz 不再卡这里。
    if quiz and not state.get("quiz_served"):
        has_critique, overall, has_high = _critic_verdict(state)
        if not has_critique:
            # 刚出完题，还没审核 → critic
            return SupervisorDecision(
                next_agent="critic", action="continue", topic=goal,
                reason="规则兜底：出题后先过质量门 → critic",
            )
        needs_revise = (overall < 0.7) or has_high
        if needs_revise and state.get("revision_count", 0) < MAX_REVISIONS and not state.get("critic_passed"):
            # critic 低分且未到精修上限 → reviser 精修后回 critic 重审
            return SupervisorDecision(
                next_agent="reviser", action="continue", topic=goal,
                target_weak_points=weak_points[:3],
                reason=f"规则兜底：critic 低分(overall={overall:.2f})精修 → reviser",
            )
        # critic 通过（或已到精修上限放行）→ 下发题目并等作答
        return SupervisorDecision(
            next_agent="await_answers", action="continue", topic=goal,
            reason=f"规则兜底：critic 通过(overall={overall:.2f}) → 下发题目等学生作答",
        )

    # (D) 复刻 orchestrator._route：显式 action=plan → planner
    if action == "plan":
        return SupervisorDecision(
            next_agent="planner", action="switch_to_plan", topic=goal,
            reason="规则兜底：action=plan → planner",
        )

    # (E) 开场（还没诊断过）→ 先诊断学情（对应 orchestrator quiz 流先走 adapt_reader）
    #   用 diagnosed 标记防反复诊断；history 非空也视为已诊断。
    if not state.get("diagnosed") and not history:
        return SupervisorDecision(
            next_agent="diagnostic", action="continue", topic=goal,
            difficulty="medium", difficulty_score=0.5,
            reason="规则兜底：冷启动先诊断学情 → diagnostic",
        )

    # (F) 进入新一轮出题：复刻 adaptive_loop._rule_fallback 的难度逻辑映射到 quiz
    if last_report is None:
        # 跨会话个性化：diagnostic_worker 识别为回访用户 → 续上次进度
        #   按上次掌握度定开场难度（ZPD = mastery+0.15），有薄弱点则优先复习。
        rc = state.get("returning_context") or {}
        if rc.get("is_returning"):
            mastery = rc.get("mastery")
            # 到期复习项（SRS）优先于一般薄弱点：先安排"该复习的"，让记忆闭环
            due = rc.get("due_reviews") or []
            rc_weak = due or rc.get("top_weak_points") or weak_points
            if isinstance(mastery, (int, float)):
                dscore = round(min(mastery + 0.15, 1.0), 2)
                dword = "hard" if dscore >= 0.66 else ("easy" if dscore < 0.4 else "medium")
                mastery_str = f"{mastery:.0%}"
            else:
                dscore, dword = 0.5, "medium"
                mastery_str = "未知"
            reason = f"欢迎回来：{'优先安排到期复习' if due else '续上次进度'}（掌握度 {mastery_str}）"
            if rc_weak:
                reason += "，复习：" + "、".join(rc_weak[:2])
            return SupervisorDecision(
                next_agent="quiz",
                action="remediate" if rc_weak else "continue",
                topic=goal, difficulty=dword, difficulty_score=dscore,
                target_weak_points=rc_weak[:3], count=3,
                reason=reason,
            )
        return SupervisorDecision(
            next_agent="quiz", action="continue", topic=goal,
            difficulty="medium", difficulty_score=0.5, count=3,
            reason="规则兜底：诊断后中等难度开场出题 → quiz",
        )
    score = last_report.get("score", 0.0)
    if score < 0.5:
        return SupervisorDecision(
            next_agent="quiz", action="remediate", topic=goal,
            difficulty="easy", difficulty_score=0.3,
            target_weak_points=weak_points[:3], count=3,
            reason="规则兜底：得分偏低，降难度补薄弱点重练 → quiz",
        )
    return SupervisorDecision(
        next_agent="quiz", action="advance", topic=goal,
        difficulty="hard", difficulty_score=0.75, count=3,
        reason="规则兜底：得分良好，升难度进阶 → quiz",
    )


def _oneshot_next(state: dict) -> SupervisorDecision:
    """oneshot 模式单跳路由：按 state["action"] 单跳到对应 worker 后 finish。

    与 guided 的 _rule_fallback_next 不同：oneshot 不进 supervisor⇄worker 循环、不 interrupt、
    不批改作答；action 已经明确，纯规则即可（不调 LLM）。worker 干完回流 supervisor 后，
    用"产物是否已存在"判定 worker 已回来 → finish。

    三个 action：
      plan  → goto planner；已有 learning_path → finish
      grade → goto grader；已有 grading_report/last_report → finish
      quiz  → goto quiz → 可选过一次 critic → finish（不等作答、不 reviser 循环）

    避免无限循环：每个 worker 只派一次，用产物存在与否做单调推进。
    """
    action = (state.get("action") or "quiz").lower()
    goal = state.get("goal") or state.get("description") or ""

    # ── plan：planner 单跳 ──
    if action == "plan":
        if state.get("learning_path"):
            return SupervisorDecision(
                next_agent="finish", action="finish", topic=goal, done=True,
                reason="oneshot：learning_path 已生成 → finish",
            )
        return SupervisorDecision(
            next_agent="planner", action="switch_to_plan", topic=goal,
            reason="oneshot：单跳 planner 生成学习路径",
        )

    # ── grade：grader 单跳 ──
    if action == "grade":
        if state.get("grading_report") or state.get("last_report"):
            return SupervisorDecision(
                next_agent="finish", action="finish", topic=goal, done=True,
                reason="oneshot：grading_report 已产出 → finish",
            )
        return SupervisorDecision(
            next_agent="grader", action="continue", topic=goal,
            reason="oneshot：单跳 grader 批改作答",
        )

    # ── quiz（默认）：quiz → 可选 critic 一次 → finish ──
    #   ONESHOT_QUIZ_CRITIC=true（默认）时出题后过一次 critic 质量门再 finish；
    #   false 则 quiz 出完即 finish。无论是否过 critic，都不进 reviser↔critic 循环。
    if not state.get("quiz"):
        return SupervisorDecision(
            next_agent="quiz", action="continue", topic=goal,
            difficulty=state.get("difficulty", "medium"),
            difficulty_score=state.get("difficulty_score", 0.5),
            question_type=state.get("type", "choice"),
            count=state.get("count", 3),
            reason="oneshot：单跳 quiz 出题",
        )
    # 已有 quiz：过一次 critic（若开启且还没审过）后 finish
    if _oneshot_quiz_critic_enabled() and not state.get("critique_history"):
        return SupervisorDecision(
            next_agent="critic", action="continue", topic=goal,
            reason="oneshot：出题后过一次 critic 质量门",
        )
    return SupervisorDecision(
        next_agent="finish", action="finish", topic=goal, done=True,
        reason="oneshot：quiz 已生成 → finish",
    )


def _oneshot_quiz_critic_enabled() -> bool:
    """ONESHOT_QUIZ_CRITIC=true（默认）→ oneshot quiz 出题后过一次 critic 质量门再 finish。"""
    return os.getenv("ONESHOT_QUIZ_CRITIC", "true").lower() in ("1", "true", "yes")


def _assist_next(state: dict) -> SupervisorDecision:
    """assist 模式路由：assistant 未收尾 → goto assistant；已 finalize → finish。

    assistant worker 内部跑 ReAct 多轮并用 interrupt 处理 ask_user；finalize/截断时
    置 assistant_done=True 回流 supervisor，此处据此单调推进到 finish。
    """
    goal = state.get("goal") or state.get("description") or ""
    if state.get("assistant_done"):
        return SupervisorDecision(
            next_agent="finish", action="finish", topic=goal, done=True,
            reason="assist：assistant 已 finalize → finish",
        )
    return SupervisorDecision(
        next_agent="assistant", action="continue", topic=goal,
        reason="assist：路由到 assistant 自由问答",
    )


def _normalize_decision(d: SupervisorDecision) -> SupervisorDecision:
    """归一化非法枚举 + clamp 数值，防 LLM 乱填把下游图搞崩。"""
    if d.next_agent not in _AGENTS:
        d.next_agent = "diagnostic"
    if d.action not in _ACTIONS:
        d.action = "continue"
    if d.difficulty not in _DIFF_WORDS:
        d.difficulty = "medium"
    if d.question_type not in _TYPES:
        d.question_type = "choice"
    try:
        d.difficulty_score = max(0.0, min(1.0, float(d.difficulty_score)))
    except (TypeError, ValueError):
        d.difficulty_score = 0.5
    try:
        d.count = max(1, min(10, int(d.count)))
    except (TypeError, ValueError):
        d.count = 3
    # next_agent=finish 与 done 语义对齐
    if d.next_agent == "finish" or d.action == "finish":
        d.done = True
    return d


def _terminate_update(state: dict, reason: str, decision: SupervisorDecision | None = None) -> dict:
    """收尾时写入 TutorState 的终止字段。"""
    update = {
        "done": True,
        "terminate_reason": reason,
        "next_agent": "finish",
    }
    if decision is not None:
        update["supervisor_reason"] = decision.reason
        update["last_action"] = decision.action
    return update


def _oneshot_command(state: dict, decision: SupervisorDecision, handoffs: int) -> Command:
    """把 oneshot 决策转成 Command：finish → output_guard；否则单跳到 worker（不重置质量门）。

    与 guided 路径不同：oneshot 不清 critique_history/quiz_served（否则 quiz→critic→quiz 死循环），
    只透传教学参数；产物存在判定由 _oneshot_next 负责单调推进到 finish。
    """
    base_update = {
        "next_agent": decision.next_agent,
        "supervisor_reason": decision.reason,
        "last_action": decision.action,
        "turn": state.get("turn", 0) + 1,
        "handoff_count": handoffs + 1,
    }
    if decision.done or decision.next_agent == "finish":
        update = {**base_update, **_terminate_update(state, "agent_finish", decision)}
        return Command(goto=_FINISH_NODE, update=update)

    goto_node = _AGENT_TO_NODE.get(decision.next_agent, "quiz")
    base_update.update({
        "description": decision.topic or state.get("description", ""),
        "difficulty": decision.difficulty,
        "difficulty_score": decision.difficulty_score,
        "type": decision.question_type,
        "count": decision.count,
    })
    if decision.target_weak_points:
        base_update["weak_points"] = decision.target_weak_points
    return Command(goto=goto_node, update=base_update)


async def teaching_supervisor(state: dict) -> Command:
    """Supervisor 节点：观察 → 决策（LLM 或规则）→ Command 动态 goto worker / 收尾。

    流程：
      ① done 或 handoff_count >= MAX_HANDOFFS → goto output_guard 收尾
      ② supervisor_mode=="rule" 或 llm_parse 异常 → _rule_fallback_next 规则兜底
      ③ 否则 llm_parse(SupervisorDecision) → _normalize_decision
      ④ next_agent=="finish"/done → goto output_guard；否则 goto 对应 worker 节点
      update 写 next_agent/supervisor_reason/last_action/turn+1/handoff_count+1。
    """
    # ① 终止前置判定（agent 主动结束 / handoff 上限）
    if state.get("done"):
        logger.info("[supervisor] state.done=True → finish")
        return Command(goto=_FINISH_NODE, update=_terminate_update(state, "agent_finish"))
    handoffs = state.get("handoff_count", 0)
    if handoffs >= MAX_HANDOFFS:
        logger.warning(f"[supervisor] handoff_count {handoffs} >= MAX_HANDOFFS={MAX_HANDOFFS} → finish")
        return Command(goto=_FINISH_NODE, update=_terminate_update(state, "max_handoffs"))

    # ②a oneshot 模式：按 action 单跳到对应 worker → finish，纯规则不调 LLM。
    #     不进 guided 的 supervisor⇄worker 循环、不 interrupt、不 reviser 循环。
    #     置于 mastery 兜底之前：oneshot grade/quiz 始终以 agent_finish 单跳收尾（确定性语义）。
    decision: SupervisorDecision
    if state.get("mode") == "oneshot":
        decision = _oneshot_next(state)
        logger.info(f"[supervisor] oneshot → {decision.next_agent} | {decision.reason[:50]}")
        decision = _normalize_decision(decision)
        return _oneshot_command(state, decision, handoffs)

    # ②b assist 模式：路由到 assistant ReAct worker；finalize 后 finish。
    #     assistant 内部用 interrupt 处理 ask_user（HITL）；assistant_done 置位 → 收尾。
    if state.get("mode") == "assist":
        decision = _assist_next(state)
        logger.info(f"[supervisor] assist → {decision.next_agent} | {decision.reason[:50]}")
        decision = _normalize_decision(decision)
        return _oneshot_command(state, decision, handoffs)

    # 掌握度达标兜底终止（与 adaptive_loop.should_terminate 的 mastery_reached 对齐）
    mastery = state.get("difficulty_score")  # diagnostic 写回的画像难度（≈ mastery+0.15），此处仅作弱信号
    last_report = state.get("last_report") or {}
    report_score = last_report.get("score")
    if isinstance(report_score, (int, float)) and report_score >= MASTERY_TARGET and state.get("turn", 0) > 0:
        logger.info(f"[supervisor] last report score {report_score:.2f} >= MASTERY_TARGET → finish")
        return Command(goto=_FINISH_NODE, update=_terminate_update(state, "mastery_reached"))

    # ②③ 决策：rule 模式 / llm 模式（失败回退 rule）
    if supervisor_mode() == "rule":
        decision = _rule_fallback_next(state)
        logger.info(f"[supervisor] rule mode → {decision.next_agent} | {decision.action} | {decision.reason[:50]}")
    else:
        try:
            client = state.get("_client")  # 测试可经 state 注入 mock client（默认 None 用模块级）
            # 注入跨会话记忆画像卡（栅栏包裹防注入）：让 LLM 决策时"记得"用户画像
            sup_messages = [{"role": "system", "content": _SUPERVISOR_SYSTEM}]
            memory_block = state.get("memory_block")
            if memory_block:
                sup_messages.append({"role": "system", "content": memory_block})
            sup_messages.append({"role": "user", "content": _build_observation(state)})
            resp = await llm_parse(
                sup_messages,
                response_format=SupervisorDecision,
                client=client,
                max_tokens=512,
            )
            parsed = resp.choices[0].message.parsed
            if parsed is None:
                raise ValueError("parsed supervisor decision is None")
            decision = _normalize_decision(parsed)
            logger.info(
                f"[supervisor] llm → {decision.next_agent} | {decision.action} | "
                f"diff={decision.difficulty_score:.2f} | {decision.reason[:50]}"
            )
        except Exception as e:
            logger.warning(f"[supervisor] llm_parse failed, rule fallback: {e}")
            decision = _rule_fallback_next(state)

    decision = _normalize_decision(decision)

    # ④ 路由：finish / done → 收尾；否则派对应 worker
    base_update = {
        "next_agent": decision.next_agent,
        "supervisor_reason": decision.reason,
        "last_action": decision.action,
        "turn": state.get("turn", 0) + 1,
        "handoff_count": handoffs + 1,
    }

    if decision.done or decision.next_agent == "finish":
        update = {**base_update, **_terminate_update(state, "agent_finish", decision)}
        return Command(goto=_FINISH_NODE, update=update)

    goto_node = _AGENT_TO_NODE.get(decision.next_agent, "diagnostic")

    # 派 diagnostic 时置 diagnosed，防 history 为空时反复诊断（adapt_reader 不写 history）
    if decision.next_agent == "diagnostic":
        base_update["diagnosed"] = True

    # 新一轮出题前重置质量门状态：清掉上一轮的 critique / 精修计数 / 过审与下发标记，
    # 让新 quiz 重新走 critic→(reviser↔critic)→await_answers 全门。
    if decision.next_agent == "quiz":
        base_update.update({
            "critique_history": [],
            "revision_count": 0,
            "reflected_message": "",
            "critic_passed": False,
            "quiz_served": False,
        })

    # 透传教学参数给下游 worker（topic/难度/题型/题数/薄弱点）
    base_update.update({
        "description": decision.topic or state.get("description", ""),
        "difficulty": decision.difficulty,
        "difficulty_score": decision.difficulty_score,
        "type": decision.question_type,
        "count": decision.count,
    })
    if decision.target_weak_points:
        base_update["weak_points"] = decision.target_weak_points
    return Command(goto=goto_node, update=base_update)
