"""
Tutor Graph：Supervisor-based Multi-Agent guided 教学闭环 + HITL interrupt

结构（与 orchestrator.py 的静态 if/else 流水线对照）：
  START → tutor_input_guard → teaching_supervisor
            ⇄ {diagnostic, quiz, critic, reviser, grader, planner, tutor, assistant}
            → wait_for_answers（HITL interrupt：暂停等学生作答）→ grader
            │（finish / done / 超 MAX_HANDOFFS）
            ↓
        tutor_output_guard → END

guided 模式闭环：
  diagnostic → quiz → critic →（reviser↔critic 精修）→ wait_for_answers(暂停)
            → [Command(resume=answers)] → grader → supervisor 决策下一轮 / finish

  - teaching_supervisor 用 langgraph Command 动态 goto（不是写死的 conditional_edges），
    worker 干完 add_edge 回流 supervisor，由 supervisor 再决策，形成 supervisor⇄worker 循环。
  - wait_for_answers 用 langgraph.types.interrupt 暂停（依赖 checkpointer 持久化中断点），
    前端拿到题目 → 学生作答 → Command(resume=answers) 续跑，wait_for_answers 后 add_edge 到 grader。

灰度并存：
  - 默认 MAS_SUPERVISOR_ENABLED=false；旧 orchestrator/adaptive/autonomous 链路一律不改。
  - 入口/出口用轻量 tutor guard（不复用 orchestrator 的 input_guard/output_guard，
    后者按 quiz/grade/plan action 语义设计，会对 guided 流误报）。
"""
import logging
from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt

from agents.assistant_agent import assistant_agent
from agents.critic_agent import critic_agent
from agents.diagnostic_worker import diagnostic_worker
from agents.grader_worker import grader_worker
from agents.planner_agent import planner_agent
from agents.quiz_agent import quiz_agent
from agents.reviser_agent import reviser_agent
from agents.state import TutorState
from agents.supervisor import teaching_supervisor
from services.injection import check_injection, check_output_leak
from services.tutor_sessions import ensure_tutor_session, tutor_quiz_view
from services.tools import dispatch_tool

logger = logging.getLogger(__name__)


# ── 轻量 tutor guard（不复用 orchestrator 的 input/output_guard）────────────────
class TutorGuardError(ValueError):
    """guided 流输入/输出校验失败。main.py 的 ValueError handler 会转 400。"""
    pass


async def tutor_input_guard(state: TutorState) -> dict:
    """guided 入口轻量校验：document_id 非空 + 对 goal 做注入检测。

    不复用 orchestrator.input_guard：后者按 quiz/grade/plan action 校验 count/session_id，
    会对没有传统 action 语义的 guided 流误报。这里只做 guided 真正需要的两件事。

    assist 模式（自由问答）允许无 document_id（如『什么是 RAG』这类通识问题），故放宽校验。
    """
    document_id = (state.get("document_id") or "").strip()
    if not document_id and state.get("mode") != "assist":
        raise TutorGuardError("guided 辅导必须提供 document_id")

    goal = (state.get("goal") or state.get("description") or "").strip()
    if goal:
        is_injection, reason = await check_injection(goal)
        if is_injection:
            logger.warning(f"[tutor_input_guard] injection detected in goal: {reason}")
            raise TutorGuardError(f"输入安全检查未通过：{reason}")
    return {}


async def tutor_output_guard(state: TutorState) -> dict:
    """guided 出口轻量校验：仅在有 quiz 时做泄露检测（不卡题数/格式，避免误报降级流）。"""
    quiz = state.get("quiz")
    if quiz:
        for i, q in enumerate(quiz.get("questions", []) or []):
            for field in ("question", "answer", "explanation"):
                is_leak, reason = check_output_leak(q.get(field, "") or "")
                if is_leak:
                    logger.warning(f"[tutor_output_guard] output leak in Q{i+1}.{field}: {reason}")
                    raise TutorGuardError(f"输出安全检查未通过：第 {i+1} 题 {field} {reason}")
    return {}


# ── Supervisor 节点：包一层声明 Command 动态 goto 的可达目标集合 ────────────────
async def _supervisor_node(
    state: TutorState,
) -> Command[Literal[
    "diagnostic", "quiz", "critic", "reviser", "grader", "planner", "tutor",
    "assistant", "wait_for_answers", "output_guard",
]]:
    """薄壳：转交 agents.supervisor.teaching_supervisor，为图声明动态 goto 的可达节点集合。"""
    return await teaching_supervisor(state)


# ── critic 节点：OrchestratorState/TutorState → CriticState，回写 critique + 过审标记 ─
async def _critic_node(state: TutorState) -> dict:
    """质量门：调 critic_agent subgraph 评估题目质量，累积 critique 并置 critic_passed。

    与 orchestrator._critic_adapter 同构：critic 自主拉一次最新 chunks 作证据，revision_count +1。
    根据 overall_score/severity 判定是否通过（overall>=0.7 且无 high severity），
    置位 critic_passed，供 supervisor 规则路径放行到 wait_for_answers。
    """
    # 拉一次最新 chunks 给 critic 作证据（独立于 quiz_agent 内部检索）
    chunks: list[str] = []
    description = state.get("description", "") or state.get("goal", "")
    document_id = state.get("document_id", "")
    if description and document_id:
        try:
            result_json = await dispatch_tool(
                "search_document",
                {"document_id": document_id, "query": description},
            )
            chunks = json.loads(result_json).get("chunks", [])
        except Exception as e:
            logger.warning(f"[tutor_graph._critic_node] pre-fetch chunks failed: {e}")

    critic_input = {
        "quiz": state.get("quiz", {}),
        "chunks": chunks,
        "user_id": state.get("user_id", ""),
        "document_id": document_id,
        "difficulty_score": state.get("difficulty_score", 0.5),
        "weak_points": state.get("weak_points", []),
        "insufficient_evidence": state.get("insufficient_evidence", False),
    }
    try:
        result = await critic_agent.ainvoke(critic_input)
        critique = result.get("critique", {})
    except Exception as e:
        logger.exception(f"[tutor_graph._critic_node] critic_agent invoke failed: {e}")
        critique = {}

    history = list(state.get("critique_history", []))
    if critique:
        history.append(critique)

    # 通过判定（复刻 _should_revise 的反面：overall>=0.7 且无 high severity → 通过）
    overall = critique.get("overall_score", 1.0) if critique else 1.0
    has_high = any(
        isinstance(s, dict) and s.get("severity") == "high"
        for s in (critique.get("suggestions", []) if critique else []) or []
    )
    passed = (overall >= 0.7) and not has_high
    logger.info(f"[tutor_graph._critic_node] overall={overall:.2f} has_high={has_high} → passed={passed}")

    return {
        "critique_history": history,
        "revision_count": state.get("revision_count", 0) + 1,
        "critic_passed": passed,
    }


# ── wait_for_answers 节点：HITL interrupt（暂停等学生作答）────────────────────────
def _ensure_session(state: TutorState) -> str:
    """确保有一个 QuizSession 承接本轮题目，返回 session_id（供 grader 读取批改）。

    quiz 是 QuizResponse.model_dump()，questions 是 dict 列表，需还原为 Question 对象建 session。
    """
    return ensure_tutor_session(state).session_id


async def wait_for_answers(state: TutorState) -> dict:
    """HITL interrupt 节点：下发题目、暂停等学生作答；恢复后把 answers 写入 session 供批改。

    第一次执行：建 QuizSession → interrupt({"quiz","turn",...}) 暂停（依赖 checkpointer 持久化）。
    Command(resume=answers) 续跑：interrupt 返回 answers → 写入 session.user_answers + 标记 completed。
    后 add_edge("wait_for_answers", "grader")：grader 读 session_id 批改。
    """
    session_id = _ensure_session(state)

    # interrupt 暂停：把题目/轮次暴露给前端；恢复时返回 Command(resume=...) 的 answers
    answers = interrupt({
        "quiz": tutor_quiz_view(ensure_tutor_session(state).questions),
        "turn": state.get("turn", 0),
        "session_id": session_id,
        "supervisor_reason": state.get("supervisor_reason", ""),
    })

    # ── 恢复后续跑：把学生作答写进 session，标记完成，供 grader 批改 ──
    # Validate the complete payload before mutating the session. This keeps an
    # invalid submission retryable instead of poisoning the session as completed.
    qs = ensure_tutor_session(state, completed_answers=answers)
    answers = list(qs.user_answers)

    return {
        "answers": answers,
        "session_id": session_id,
        "quiz_served": True,
        "quiz_served_graded": False,
    }


# ── tutor stub：阻止 supervisor 连续只讲不练 ───────────────────────────────
async def _tutor_stub(state: TutorState) -> dict:
    """tutor 占位：标记不再连续讲（allow_teach=False），避免 supervisor 连续只讲不练。"""
    logger.info("[tutor_graph._tutor_stub] stub, mark allow_teach=False")
    return {"allow_teach": False}


# ── 编译图 ───────────────────────────────────────────────────────────────────
_builder = StateGraph(TutorState)

# 节点名沿用 input_guard/output_guard，并与 supervisor._FINISH_NODE 对齐，
# 但绑定的是 guided 专用的轻量 guard 函数（不复用 orchestrator 的 quiz/grade/plan 校验）。
_builder.add_node("input_guard",         tutor_input_guard)   # guided 轻量输入校验
_builder.add_node("teaching_supervisor", _supervisor_node)    # LLM 动态编排大脑
_builder.add_node("diagnostic",          diagnostic_worker)   # 诊断+跨会话记忆 → difficulty_score/weak_points/returning_context
_builder.add_node("quiz",                quiz_agent)          # hybrid 检索 + 生成 + 格式审核
_builder.add_node("critic",              _critic_node)        # 题目质量门（三维评分）
_builder.add_node("reviser",             reviser_agent)       # 题目精修（critic↔reviser 循环）
_builder.add_node("grader",              grader_worker)       # 批改 + 画像写回 + 轨迹回填
_builder.add_node("planner",             planner_agent)       # 学习路径生成
_builder.add_node("tutor",               _tutor_stub)         # 纯讲解（stub）
_builder.add_node("assistant",           assistant_agent)     # 开放问答 ReAct worker（ask_user→interrupt）
_builder.add_node("wait_for_answers",    wait_for_answers)    # HITL interrupt：等学生作答
_builder.add_node("output_guard",        tutor_output_guard)  # guided 轻量输出校验

# 入口：START → input_guard → supervisor
_builder.add_edge(START, "input_guard")
_builder.add_edge("input_guard", "teaching_supervisor")

# worker 回流 supervisor（supervisor 用 Command 动态 goto 出去，不需静态 conditional_edges）
for _worker in ("diagnostic", "quiz", "critic", "reviser", "planner", "tutor", "assistant"):
    _builder.add_edge(_worker, "teaching_supervisor")

# wait_for_answers（恢复后）→ grader；grader → supervisor 再决策下一轮 / finish
_builder.add_edge("wait_for_answers", "grader")
_builder.add_edge("grader", "teaching_supervisor")

# 出口：output_guard → END
_builder.add_edge("output_guard", END)

# 默认编译（无 checkpointer）：仅供 import / 路由自检 / 不含 interrupt 的单测路由用。
# interrupt 必须配 checkpointer 才能持久化中断点，故 routers 用 compile_tutor_graph 工厂。
tutor_graph = _builder.compile()


# ── Durable Checkpointer 工厂（interrupt 依赖 checkpointer 持久化中断点）──────
def compile_tutor_graph(checkpointer):
    """用同一个 StateGraph builder 编译一份带 checkpointer 的 tutor_graph。

    interrupt 暂停时把中断点持久化到 checkpointer；Command(resume=...) 用同 thread_id 续跑。
    生命周期由调用方管理（async with open_sqlite_checkpointer(...)）。

    Args:
        checkpointer: LangGraph BaseCheckpointSaver 实例（如 AsyncSqliteSaver）

    Returns:
        CompiledGraph，ainvoke 时必须传 config={"configurable": {"thread_id": ...}}
    """
    return _builder.compile(checkpointer=checkpointer)
