"""
QuizAgent：Hybrid 检索 + Sufficiency Check + Context Engineering 出题 + 质量审核

Phase 8 P1-1 升级：插入 sufficiency_check 节点作为 generate 前置质量门。

节点流程：
  retrieve → sufficiency_check ─┬─ sufficient → generate → review
                                ├─ insufficient (rewrite_count<1) → rewrite_query → retrieve
                                └─ insufficient (rewrite_count>=1) → degrade → generate → review

为什么是双门设计：
  - sufficiency_check：前置门，heuristic-based（快、零 LLM 开销），筛掉 chunks 明显不够的情况
  - critic_agent：后置门（在 orchestrator 层）LLM judge，深度评估生成题质量
  两道门互补，形成 defense-in-depth

Tiered fallback：
  L1 改写 query 重试检索（LLM 改写）
  L2 降级出题：减题数、降难度、insufficient_evidence=True 让 critic 更严格
  L3 不抛错让用户重新上传——保留可用性，体验优先
"""
import logging

from langgraph.graph import END, START, StateGraph

from agents.state import QuizAgentState
from services.rag import generate_question_from_chunks
from services.retry import RetryExhausted, with_retry
from services.sufficiency import MAX_REWRITES, check_sufficiency, rewrite_query
from services.tracing import traceable
from services.vectorstore import retrieve_with_rewrite

logger = logging.getLogger(__name__)


# ── retrieve（不变）────────────────────────────────────────────────────────
@traceable(name="quiz_agent.retrieve", run_type="retriever")
async def retrieve(state: QuizAgentState) -> dict:
    """检索：HyDE / Multi-query 改写（按 env 开关）→ Hybrid（BM25 + 向量 + RRF）→ Reranker。"""
    result = await retrieve_with_rewrite(state["document_id"], state["description"])
    return {
        "chunks": result["documents"][0],
        "retrieve_count": state.get("retrieve_count", 0) + 1,
    }


# ── sufficiency_check（P1-1 新增）─────────────────────────────────────────
@traceable(name="quiz_agent.sufficiency_check", run_type="chain")
async def sufficiency_check(state: QuizAgentState) -> dict:
    """检索充分性判定，纯 heuristic，无 LLM 调用。"""
    chunks = state.get("chunks", []) or []
    weak_points = state.get("weak_points", []) or []
    passed, reason = check_sufficiency(chunks, weak_points)
    logger.info(
        f"[quiz_agent] sufficiency: passed={passed}, reason={reason}, "
        f"chunks={len(chunks)}, weak_points={len(weak_points)}"
    )
    return {"sufficiency_passed": passed, "sufficiency_reason": reason}


# ── rewrite_query（P1-1 新增）─────────────────────────────────────────────
@traceable(name="quiz_agent.rewrite_query", run_type="llm")
async def rewrite_query_node(state: QuizAgentState) -> dict:
    """LLM 改写检索 query，下游会回到 retrieve 节点重试。"""
    new_desc = await rewrite_query(
        state.get("description", ""),
        state.get("weak_points", []),
    )
    return {
        "description": new_desc,
        "rewrite_count": state.get("rewrite_count", 0) + 1,
    }


# ── degrade（P1-1 新增）───────────────────────────────────────────────────
@traceable(name="quiz_agent.degrade", run_type="chain")
async def degrade(state: QuizAgentState) -> dict:
    """证据不足时降级生成参数：减题数、降难度，并标记给 critic。"""
    count = state.get("count", 5)
    diff_score = state.get("difficulty_score", 0.5)
    new_count = max(count // 2, 2)
    new_diff = max(diff_score - 0.2, 0.2)
    logger.warning(
        f"[quiz_agent] degrade: count {count}→{new_count}, "
        f"difficulty_score {diff_score:.2f}→{new_diff:.2f}"
    )
    return {
        "count": new_count,
        "difficulty_score": new_diff,
        "insufficient_evidence": True,
    }


# ── generate（注入 reflected_message 反思回灌）─────────────────────────────
@traceable(name="quiz_agent.generate", run_type="llm")
async def generate(state: QuizAgentState) -> dict:
    """调用 LLM 出题，注入 CE 参数（difficulty_score + weak_points）+ 反思回灌。

    Reflection 回灌(借鉴 aider/coders/base_coder.py:933-944):
      上轮被 critic 拒绝时,state["reflected_message"] 携带格式化的拒绝原因,
      本轮 generate 把它当作 user 指令的一部分塞给 LLM,让 LLM 知道
      "上次错在哪",而不只是机械重出。
    """
    count = state.get("count", 5)
    reflected = state.get("reflected_message", "") or ""
    if reflected:
        logger.info(
            f"[quiz_agent] reflection injected (round {state.get('revision_count', 0)}): "
            f"{len(reflected)} chars"
        )

    try:
        quiz = await with_retry(lambda: generate_question_from_chunks(
            state["chunks"],
            count,
            state.get("difficulty", "medium"),
            state.get("type", "choice"),
            difficulty_score=state.get("difficulty_score"),
            weak_points=state.get("weak_points"),
            reflected_message=reflected,
        ))
    except RetryExhausted:
        logger.warning("[quiz_agent] generate retries exhausted, falling back to simplified generation")
        quiz = await generate_question_from_chunks(
            state["chunks"][:2],
            min(count, 3),
            "easy",
            state.get("type", "choice"),
            reflected_message=reflected,
        )
    return {
        "quiz": quiz.model_dump(),
        "generate_count": state.get("generate_count", 0) + 1,
    }


# ── review（2026-06-03 降级为格式校验，质量判断交给主图 critic）───────────────
def _validate_quiz_format(questions: list, qtype: str) -> tuple[bool, str]:
    """纯规则校验题目「结构是否完整可用」（能渲染 / 能批改）。无 LLM 调用。"""
    if not questions:
        return False, "no questions generated"
    for i, q in enumerate(questions):
        if not isinstance(q, dict):
            return False, f"Q{i} not a dict"
        if not (q.get("question") or "").strip():
            return False, f"Q{i} empty question"
        if not str(q.get("answer") or "").strip():
            return False, f"Q{i} empty answer"
        if qtype == "choice":
            opts = [o for o in (q.get("options") or []) if str(o).strip()]
            if len(opts) < 2:
                return False, f"Q{i} choice needs >=2 options"
    return True, "format ok"


@traceable(name="quiz_agent.review", run_type="chain")
async def review(state: QuizAgentState) -> dict:
    """题目「格式/结构」校验——纯规则，零 LLM 调用。

    重构（2026-06-03）：原 review 是 LLM 自审（判断是否基于原文/难度/答案正确），
    与主图 critic_agent 的三维评分（难度/相关性/覆盖度）职责重叠，每题多花一次 LLM。
    现降级为纯结构校验：只保证题目结构完整、能渲染能批改；深度质量统一交给
    主图 critic 评估。形成「规则廉价门（sufficiency + 本节点）→ LLM 昂贵门（critic）」
    的分层防御，与 sufficiency_check 的设计哲学一致。
    """
    quiz = state.get("quiz", {}) or {}
    questions = quiz.get("questions", []) or []
    qtype = state.get("type", "choice")
    passed, reason = _validate_quiz_format(questions, qtype)
    if not passed:
        logger.warning(f"[quiz_agent] format check failed: {reason}")
    return {"review_passed": passed}


# ── 条件路由 ───────────────────────────────────────────────────────────────
def _route_after_sufficiency(state: QuizAgentState) -> str:
    """sufficiency 通过 → generate；未通过且改写未到上限 → rewrite_query；否则 → degrade"""
    if state.get("sufficiency_passed"):
        return "generate"
    if state.get("rewrite_count", 0) >= MAX_REWRITES:
        return "degrade"
    return "rewrite_query"


def _should_regenerate(state: QuizAgentState) -> str:
    """review 通过或已重出 1 次 → END；否则重新 generate"""
    if state.get("review_passed", False) or state.get("generate_count", 0) >= 2:
        return END
    return "generate"


# ── 编译 Subgraph ──────────────────────────────────────────────────────────
_builder = StateGraph(QuizAgentState)
_builder.add_node("retrieve", retrieve)
_builder.add_node("sufficiency_check", sufficiency_check)
_builder.add_node("rewrite_query", rewrite_query_node)
_builder.add_node("degrade", degrade)
_builder.add_node("generate", generate)
_builder.add_node("review", review)

_builder.add_edge(START, "retrieve")
_builder.add_edge("retrieve", "sufficiency_check")
_builder.add_conditional_edges("sufficiency_check", _route_after_sufficiency, {
    "generate": "generate",
    "rewrite_query": "rewrite_query",
    "degrade": "degrade",
})
_builder.add_edge("rewrite_query", "retrieve")     # 改写后回到 retrieve 重检
_builder.add_edge("degrade", "generate")           # 降级直接 generate
_builder.add_edge("generate", "review")
_builder.add_conditional_edges("review", _should_regenerate)

quiz_agent = _builder.compile()
