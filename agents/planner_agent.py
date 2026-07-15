"""
PlannerAgent(Phase 9 P0-7:升级为完整 LangGraph 子图 + reviewer↔reviser 循环)

升级前(Phase 8 P4):
  - 薄壳子图:START → plan → END,plan 节点直接调 services/learning_path.generate_learning_path
  - 行为正确但架构不一致:5 阶段是 services 里的串行 await,不是 LangGraph 节点

升级后(本次):
  - 5 阶段全部拆为 LangGraph 节点(extract_brief / explore / compress / synthesize / critique)
  - critique 不通过时进入 path_reviser 精修节点(借鉴 Task 6 reviewer↔reviser 范式)
  - path_reviser → critique 形成循环子图(max 2 轮,与 quiz 流的 revision_count 上限对齐)
  - services/learning_path.py 的纯函数保持不动(routers/learning_path.py 等外部调用方向后兼容)

为什么 path_reviser 而非 synthesize 重跑?
  借鉴 Task 6 reviser_agent 思想:
    - synthesize 重跑 = 整段重写 LearningPath,会改坏 critique 已认可的阶段
    - path_reviser 精修 = 只改 critique.issues 涉及的阶段,保留其它阶段
  对齐 gpt-researcher editor.py:138-142(reviewer↔reviser)+ Task 6 quiz 流 reviser

env 开关:
  PATH_REVISER_ENABLED=true(默认)→ critique 不通过走 path_reviser
  false → 回退老路径(critique 不通过走 synthesize 整段重写,与 Phase 8 P4 行为一致)
"""
import json
import logging
import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from openai import AsyncOpenAI

from agents.state import OrchestratorState
from models.learning_path import (
    CompressedReport,
    ExplorationReport,
    LearningPath,
    PathBrief,
    PathCritique,
)
from services.learning_path import (
    compress,
    critique,
    explore,
    extract_brief,
    synthesize,
)
from services.tracing import traceable

load_dotenv(Path(__file__).parent.parent / ".env")
logger = logging.getLogger(__name__)

# 路径规划用 json_schema 结构化输出 → 走 structured 供应商
from services.llm import structured_client as _client, structured_model as _model

# 与 quiz 流 revision_count 上限对齐
_MAX_PATH_REVISIONS = 2


def _path_reviser_enabled() -> bool:
    """PATH_REVISER_ENABLED=true → critique 不通过走 path_reviser(精修)
    false → 回退到 synthesize 整段重写(Phase 8 P4 老行为)"""
    return os.getenv("PATH_REVISER_ENABLED", "true").lower() in ("1", "true", "yes")


# ══════════════════════════════════════════════════════════════════════════
# Planner 内部 state(不污染 OrchestratorState,由 adapter 做 I/O 转换)
# ══════════════════════════════════════════════════════════════════════════
class PlannerState(TypedDict, total=False):
    # ── 输入 ──────────────────────────────────────────────────────
    document_id: str
    user_intent: str
    enable_critique: bool        # 评测 ablation 用,默认 True

    # ── 阶段中间态(model_dump 后的 dict)──────────────────────────
    brief: dict                   # PathBrief
    exploration_report: dict      # ExplorationReport
    compressed_report: dict       # CompressedReport
    critique: dict                # PathCritique

    # ── 输出 ──────────────────────────────────────────────────────
    learning_path: dict           # LearningPath

    # ── 反思循环计数 ───────────────────────────────────────────────
    revision_count: int


# ══════════════════════════════════════════════════════════════════════════
# 节点函数:每个都是薄壳,实际逻辑在 services/learning_path.py(向后兼容)
# ══════════════════════════════════════════════════════════════════════════
@traceable(name="planner.extract_brief", run_type="chain")
async def _node_extract_brief(state: PlannerState) -> dict:
    brief = await extract_brief(
        state["document_id"],
        state.get("user_intent", ""),
    )
    return {"brief": brief.model_dump()}


@traceable(name="planner.explore", run_type="retriever")
async def _node_explore(state: PlannerState) -> dict:
    brief = PathBrief(**state["brief"])
    report = await explore(state["document_id"], brief)
    return {"exploration_report": report.model_dump()}


@traceable(name="planner.compress", run_type="chain")
async def _node_compress(state: PlannerState) -> dict:
    brief = PathBrief(**state["brief"])
    report = ExplorationReport(**state["exploration_report"])
    compressed = await compress(report, brief)
    return {"compressed_report": compressed.model_dump()}


@traceable(name="planner.synthesize", run_type="llm")
async def _node_synthesize(state: PlannerState) -> dict:
    brief = PathBrief(**state["brief"])
    compressed = CompressedReport(**state["compressed_report"])

    # path_reviser 关闭时,critique 不通过会回到这里整段重写(老行为)
    revision_hint = ""
    if not _path_reviser_enabled():
        crit = state.get("critique") or {}
        revision_hint = crit.get("revision_hints", "") if crit.get("needs_revision") else ""

    path = await synthesize(
        state["document_id"], brief, compressed, revise_hint=revision_hint,
    )
    return {"learning_path": path.model_dump()}


@traceable(name="planner.critique", run_type="llm")
async def _node_critique(state: PlannerState) -> dict:
    if not state.get("enable_critique", True):
        return {
            "critique": PathCritique(
                overall_score=1.0, issues=[], needs_revision=False, revision_hints="",
            ).model_dump(),
            "revision_count": state.get("revision_count", 0) + 1,
        }

    path = LearningPath(**state["learning_path"])
    compressed = CompressedReport(**state["compressed_report"])
    crit = await critique(path, compressed)
    return {
        "critique": crit.model_dump(),
        "revision_count": state.get("revision_count", 0) + 1,
    }


_PATH_REVISER_SYSTEM = (
    "你是学习路径精修专家。给定一份 LearningPath 和审稿人的 issues + revision_hints,"
    "你的任务是**最小化修改**:只改 issues 涉及的阶段(stage),完全保留没问题的阶段。\n\n"
    "改的时候:\n"
    "- 优先解决 revision_hints 里列出的具体问题\n"
    "- 阶段数量、document_id、title 通常保持不变\n"
    "- 每个 stage 的 estimated_minutes 必须在 10-30 之间\n"
    "- 输出完整 LearningPath(同结构,只是部分 stage 已被改过)"
)


@traceable(name="planner.path_reviser", run_type="llm")
async def _node_path_reviser(state: PlannerState) -> dict:
    """精修节点:只改 critique 指出的阶段,而非整段重写(借鉴 Task 6 reviser_agent)"""
    path_dict = state.get("learning_path") or {}
    critique_dict = state.get("critique") or {}

    if not path_dict or not critique_dict.get("needs_revision"):
        return {}

    path_text = json.dumps(path_dict, ensure_ascii=False)[:3000]
    crit_text = json.dumps(critique_dict, ensure_ascii=False)[:1500]
    user_msg = (
        f"【当前 LearningPath(JSON)】\n{path_text}\n\n"
        f"【审稿人 critique(JSON)】\n{crit_text}\n\n"
        "请输出精修后的完整 LearningPath。"
    )

    try:
        resp = await _client.beta.chat.completions.parse(
            model=_model,
            messages=[
                {"role": "system", "content": _PATH_REVISER_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            response_format=LearningPath,
        )
        revised = resp.choices[0].message.parsed
        # 保护:document_id 不允许被改
        revised.document_id = path_dict.get("document_id", revised.document_id)
        logger.info(
            f"[planner.path_reviser] revised {revised.total_stages} stages "
            f"(round {state.get('revision_count', 0)})"
        )
        return {"learning_path": revised.model_dump()}
    except Exception as e:
        logger.warning(f"[planner.path_reviser] LLM call failed: {e}, return path unchanged")
        return {}  # 不更新 learning_path,critique 下一轮会基于原 path 决定是否继续


# ══════════════════════════════════════════════════════════════════════════
# 条件路由:critique → END / path_reviser / synthesize(老路径)
# ══════════════════════════════════════════════════════════════════════════
def _route_after_critique(state: PlannerState) -> str:
    """critique 后路由:通过→END;需要修订且未到上限→reviser(或 synthesize 老路径)"""
    crit = state.get("critique") or {}
    if not crit.get("needs_revision"):
        return "end"
    if state.get("revision_count", 0) >= _MAX_PATH_REVISIONS:
        logger.info(
            f"[planner] max revisions reached ({_MAX_PATH_REVISIONS}), exiting loop"
        )
        return "end"
    return "path_reviser" if _path_reviser_enabled() else "synthesize"


# ══════════════════════════════════════════════════════════════════════════
# 编译子图
# ══════════════════════════════════════════════════════════════════════════
_builder = StateGraph(PlannerState)
_builder.add_node("extract_brief", _node_extract_brief)
_builder.add_node("explore",       _node_explore)
_builder.add_node("compress",      _node_compress)
_builder.add_node("synthesize",    _node_synthesize)
_builder.add_node("critique",      _node_critique)
_builder.add_node("path_reviser",  _node_path_reviser)

# 主流水线
_builder.add_edge(START,           "extract_brief")
_builder.add_edge("extract_brief", "explore")
_builder.add_edge("explore",       "compress")
_builder.add_edge("compress",      "synthesize")
_builder.add_edge("synthesize",    "critique")

# reviewer↔reviser 循环(critique → path_reviser → critique)
_builder.add_conditional_edges("critique", _route_after_critique, {
    "end":          END,
    "path_reviser": "path_reviser",
    "synthesize":   "synthesize",   # PATH_REVISER_ENABLED=false 时回退老路径
})
_builder.add_edge("path_reviser", "critique")  # 关键边:形成循环

planner_subgraph = _builder.compile()


# ══════════════════════════════════════════════════════════════════════════
# Adapter:把 OrchestratorState 转 PlannerState 并 ainvoke 子图
# (类似 _critic_adapter,保持 OrchestratorState 干净)
# ══════════════════════════════════════════════════════════════════════════
async def planner_agent(state: OrchestratorState) -> dict:
    """Orchestrator 调用入口:转 state → 跑子图 → 取 learning_path 写回。"""
    planner_input: PlannerState = {
        "document_id": state["document_id"],
        "user_intent": state.get("description") or "",
        "enable_critique": True,
        "revision_count": 0,
    }
    try:
        result = await planner_subgraph.ainvoke(planner_input)
        return {"learning_path": result.get("learning_path")}
    except Exception as e:
        logger.exception(f"[planner_agent] subgraph invoke failed: {e}")
        return {"learning_path": None}
