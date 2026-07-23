"""
Critic Agent：独立 subgraph，Tutor↔Critic 反思循环的核心

── 与现有 quiz_agent.py:77-98 review 节点的差异 ─────────────────────────────
| 维度        | 当前 review                    | critic_agent                   |
|-------------|--------------------------------|---------------------------------|
| 输出        | ReviewResult(passed: bool)     | CritiqueReport(3D + suggestions)|
| 与 tutor 关系| 节点内部循环                    | 独立 subgraph 与 quiz_agent 串接|
| 能力        | 仅 LLM 单次判定                | 可调 dispatch_tool 二次检索证据  |
| 输出消费方  | _should_regenerate 单一分支    | OrchestratorState.critique_history（可累积）|

── Subgraph 流程 ────────────────────────────────────────────────────────────
  analyze → (optional) reinforce_evidence → produce_report → END
            ↑                                ↓
            └── 当 LLM 判定证据不足时触发 ──┘

  analyze            : LLM 看 quiz + chunks + 用户画像 → 初判 3 维度评分
  reinforce_evidence : 若初判置信度低 → dispatch_tool('search_document') 拉新 chunks
  produce_report     : 综合证据生成最终 CritiqueReport（含 suggestions 列表）

"""
import json
import logging
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from models.critique import CritiqueReport, CritiqueSuggestion, DimensionScore
from services.tools import dispatch_tool
from services.llm import _client, llm_chat, model as _model
from services.tracing import traceable

logger = logging.getLogger(__name__)

# Critic 判定置信度阈值：低于此值则触发二次检索
_REINFORCE_THRESHOLD = 0.5


class CriticState(TypedDict, total=False):
    """Critic subgraph 内部状态（不污染 OrchestratorState）"""
    # 输入
    quiz: dict                 # 待评估的题目（QuizResponse.model_dump()）
    chunks: list[str]          # 出题时使用的 chunks
    user_id: str
    document_id: str
    difficulty_score: float    # 用户画像难度（用于判定 difficulty 维度）
    weak_points: list[str]     # 用户薄弱点（用于判定 coverage 维度）
    insufficient_evidence: bool  # 上游 sufficiency_check 标记，触发更严格审核

    # 中间
    initial_report: dict | None       # analyze 产出的初判
    reinforce_chunks: list[str]       # 二次检索拿到的 chunks
    triggered_search: bool

    # 输出（写回 OrchestratorState）
    critique: dict                    # CritiqueReport.model_dump()


_ANALYZE_SYSTEM = (
    "你是题目质量审核 Agent。基于以下证据评估一组测验题：\n"
    "1. 难度匹配度（difficulty）：题目难度是否与用户画像 difficulty_score 匹配\n"
    "2. 相关性（relevance）：题目是否基于提供的 chunks（不能凭空捏造）\n"
    "3. 覆盖度（coverage）：题目是否覆盖用户的 weak_points\n\n"
    "每维度打 0.0-1.0 分（必须给具体理由，引用原文）。若证据不足以判定，"
    "请把对应维度的 score 设为 ≤0.5 触发二次检索。\n\n"
    "严格输出 JSON：\n"
    "{\n"
    '  "difficulty": {"score": float, "reasoning": str},\n'
    '  "relevance": {"score": float, "reasoning": str},\n'
    '  "coverage": {"score": float, "reasoning": str}\n'
    "}"
)


_FINALIZE_SYSTEM = (
    "你是 Critic Agent 的最终汇总环节。给定 3 维度评分（可能含二次检索的"
    "补充证据），产出 CritiqueReport：\n"
    "1. overall_score: 三维度等权平均\n"
    "2. suggestions: 针对低分维度给出具体改进建议（severity high/medium/low）\n\n"
    "严格输出 JSON：\n"
    "{\n"
    '  "overall_score": float,\n'
    '  "suggestions": [\n'
    '    {"target": "difficulty|relevance|coverage|general",\n'
    '     "severity": "low|medium|high",\n'
    '     "action": str}\n'
    "  ]\n"
    "}"
)


@traceable(name="critic_agent.analyze", run_type="llm")
async def _analyze(state: CriticState) -> dict:
    """初判：LLM 看 quiz + chunks + 画像，输出 3 维度评分"""
    quiz_text = json.dumps(state.get("quiz", {}), ensure_ascii=False)[:1500]
    chunks_text = "\n".join(state.get("chunks", [])[:3])[:1500]

    user_msg = (
        f"【用户画像】difficulty_score={state.get('difficulty_score', 0.5):.2f}, "
        f"weak_points={state.get('weak_points', [])}\n\n"
        f"【参考 chunks】\n{chunks_text}\n\n"
        f"【待审核题目】\n{quiz_text}"
    )
    # 上游 sufficiency_check 标记证据不足时，提示 critic 对 relevance 更严格
    if state.get("insufficient_evidence"):
        user_msg += (
            "\n\n⚠️ 上游 sufficiency_check 报告本次检索证据不充分（chunks 数量少、"
            "多样性低或未覆盖 weak_points）。请对 relevance 维度更严格打分，"
            "若题目难以从 chunks 直接验证，score 应低于 0.6。"
        )

    try:
        resp = await llm_chat(
            [
                {"role": "system", "content": _ANALYZE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            client=_client,
            model=_model,
            response_format={"type": "json_object"},
        )
        report_text = resp.choices[0].message.content or "{}"
        report_dict = json.loads(report_text)
    except Exception as e:
        logger.warning(f"[critic] analyze failed: {e}, fallback to neutral scores")
        report_dict = {
            "difficulty": {"score": 0.5, "reasoning": "analyze fallback"},
            "relevance": {"score": 0.5, "reasoning": "analyze fallback"},
            "coverage": {"score": 0.5, "reasoning": "analyze fallback"},
        }

    return {"initial_report": report_dict}


@traceable(name="critic_agent.reinforce_evidence", run_type="tool")
async def _reinforce_evidence(state: CriticState) -> dict:
    """二次检索：当 relevance 或 coverage 置信度低时，调 dispatch_tool 拉新 chunks"""
    weak_points = state.get("weak_points", [])
    document_id = state.get("document_id", "")

    # 用 weak_points 拼一个新查询
    if weak_points:
        query = " ".join(weak_points[:3])
    else:
        # 用题目第一题的 question 反查
        questions = state.get("quiz", {}).get("questions", [])
        query = questions[0].get("question", "") if questions else ""

    if not query or not document_id:
        return {"reinforce_chunks": [], "triggered_search": False}

    try:
        result_json = await dispatch_tool(
            "search_document",
            {"document_id": document_id, "query": query},
        )
        result = json.loads(result_json)
        new_chunks = result.get("chunks", [])
        logger.info(f"[critic] reinforce_evidence: got {len(new_chunks)} new chunks for query={query[:50]}")
    except Exception as e:
        logger.warning(f"[critic] reinforce_evidence failed: {e}")
        new_chunks = []

    return {"reinforce_chunks": new_chunks, "triggered_search": True}


@traceable(name="critic_agent.produce_report", run_type="llm")
async def _produce_report(state: CriticState) -> dict:
    """最终汇总：综合初判 + 二次证据 → CritiqueReport"""
    initial = state.get("initial_report", {})
    reinforce_chunks = state.get("reinforce_chunks", [])
    triggered = state.get("triggered_search", False)

    finalize_input = {
        "initial_scores": initial,
        "reinforce_chunks_preview": "\n".join(reinforce_chunks[:2])[:800] if reinforce_chunks else "（未触发二次检索）",
    }

    try:
        resp = await llm_chat(
            [
                {"role": "system", "content": _FINALIZE_SYSTEM},
                {"role": "user", "content": json.dumps(finalize_input, ensure_ascii=False)},
            ],
            client=_client,
            model=_model,
            response_format={"type": "json_object"},
        )
        finalize_dict = json.loads(resp.choices[0].message.content or "{}")
    except Exception as e:
        logger.warning(f"[critic] produce_report failed: {e}, neutral fallback")
        finalize_dict = {"overall_score": 0.5, "suggestions": []}

    # 拼装 CritiqueReport（使用 Pydantic 校验）
    try:
        report = CritiqueReport(
            difficulty=DimensionScore(**initial.get("difficulty", {"score": 0.5, "reasoning": "missing"})),
            relevance=DimensionScore(**initial.get("relevance", {"score": 0.5, "reasoning": "missing"})),
            coverage=DimensionScore(**initial.get("coverage", {"score": 0.5, "reasoning": "missing"})),
            overall_score=float(finalize_dict.get("overall_score", 0.5)),
            suggestions=[
                CritiqueSuggestion(**s) for s in finalize_dict.get("suggestions", [])
                if isinstance(s, dict) and "target" in s and "severity" in s and "action" in s
            ],
            triggered_search=triggered,
            evidence_chunks_count=len(reinforce_chunks),
        )
    except Exception as e:
        logger.warning(f"[critic] CritiqueReport assembly failed: {e}, returning neutral")
        report = CritiqueReport(
            difficulty=DimensionScore(score=0.5, reasoning="assembly fallback"),
            relevance=DimensionScore(score=0.5, reasoning="assembly fallback"),
            coverage=DimensionScore(score=0.5, reasoning="assembly fallback"),
            overall_score=0.5,
            suggestions=[],
            triggered_search=triggered,
            evidence_chunks_count=len(reinforce_chunks),
        )

    return {"critique": report.model_dump()}


def _should_reinforce(state: CriticState) -> str:
    """条件路由：初判任一维度 < threshold 则走 reinforce_evidence，否则直接 produce_report"""
    initial = state.get("initial_report", {})
    scores = [
        initial.get("relevance", {}).get("score", 1.0),
        initial.get("coverage", {}).get("score", 1.0),
    ]
    if any(s < _REINFORCE_THRESHOLD for s in scores):
        return "reinforce_evidence"
    return "produce_report"


# ── 编译 Subgraph ───────────────────────────────────────────────────────
_builder = StateGraph(CriticState)
_builder.add_node("analyze", _analyze)
_builder.add_node("reinforce_evidence", _reinforce_evidence)
_builder.add_node("produce_report", _produce_report)

_builder.add_edge(START, "analyze")
_builder.add_conditional_edges("analyze", _should_reinforce, {
    "reinforce_evidence": "reinforce_evidence",
    "produce_report": "produce_report",
})
_builder.add_edge("reinforce_evidence", "produce_report")
_builder.add_edge("produce_report", END)

critic_agent = _builder.compile()
