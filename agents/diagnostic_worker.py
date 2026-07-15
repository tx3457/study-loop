"""
Diagnostic Worker：supervisor 专用诊断节点（跨会话记忆增强）

灰度并存：不改 adapt_agent.adapt_reader（旧 orchestrator/adaptive 链路继续用它）。
本 worker 仅在 supervisor MAS 的 tutor_graph 里替换 diagnostic 节点：
  ① 复用 adapt_reader：算 difficulty_score（ZPD = mastery+0.15）+ weak_points + decision_log
     —— 直接 ainvoke 它，不重写那套 mastery/preferences fallback 逻辑。
  ② 额外（本次新增）：build_returning_context（跨会话"欢迎回来"）+ build_profile_card（画像卡）
     + build_memory_context_block（<memory-context> 栅栏），写回 TutorState 供 supervisor
     冷启动后第一轮个性化决策（续上次 / 复习薄弱点），让助手真正"记得你"。

为什么单独写一条 decision_log（面试讲点）：
  把"是否识别为回访用户、欢迎语素材"持久化成可审计事件，复用 decision_log 的审计精神。
"""
import logging

from agents.adapt_agent import adapt_reader
from agents.state import TutorState
from services.memory import append_decision
from services.memory_context import (
    build_memory_context_block,
    build_profile_card,
    build_returning_context,
)

logger = logging.getLogger(__name__)


async def diagnostic_worker(state: TutorState) -> dict:
    """诊断 + 跨会话记忆召回。返回 difficulty_score/weak_points/returning_context/memory_block。"""
    user_id = state.get("user_id", "")
    document_id = state.get("document_id", "")
    update: dict = {}

    # ① 复用 adapt_reader 的 ZPD 难度 + weak_points + decision_log（不重写）
    try:
        base = await adapt_reader.ainvoke(state)
        update["difficulty_score"] = base.get("difficulty_score", state.get("difficulty_score", 0.5))
        update["weak_points"] = base.get("weak_points", []) or []
    except Exception as e:
        logger.warning(f"[diagnostic_worker] adapt_reader 失败，沿用现有难度: {e}")

    # ② 跨会话记忆：returning_context（欢迎回来）+ profile_card（栅栏注入素材）
    try:
        rc = await build_returning_context(user_id, document_id)
    except Exception as e:
        logger.warning(f"[diagnostic_worker] build_returning_context 失败: {e}")
        rc = {"is_returning": False}
    try:
        card = await build_profile_card(user_id)
    except Exception:
        card = ""
    update["returning_context"] = rc
    update["memory_block"] = build_memory_context_block(card)

    # decision_log：记录是否识别为回访用户（可审计）
    try:
        await append_decision(user_id, {
            "agent": "diagnostic_worker",
            "document_id": document_id,
            "is_returning": rc.get("is_returning", False),
            "rationale": rc.get("welcome_msg") or "cold start（新用户，无历史记忆）",
        })
    except Exception as e:
        logger.warning(f"[diagnostic_worker] append_decision 失败（忽略）: {e}")

    return update
