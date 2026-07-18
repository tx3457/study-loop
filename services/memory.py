"""
Memory Banks（Phase 8 P3 升级，借鉴 NovelClaw 的多 bank 设计）

原来 2 个 namespace（profile + sessions 各一坨）→ 6 个语义 bank：

  Semantic Layer（长期画像）:
    - preferences    用户偏好（题型、难度倾向、学习风格）
    - mastery        per-document EMA 掌握度
    - weak_points    薄弱知识点（带时间戳，最近 20 条）

  Episodic Layer（短期事件）:
    - session_briefs 单次答题摘要
    - error_log      每个错题的详细记录（不聚合）
    - decision_log   Adapt Agent / Critic 的决策记录（why this difficulty/topic）

API 分两层：
  低层（通用）: read_bank_state / write_bank_state / append_bank_event / list_bank_events
  高层（语义）: update_mastery / append_weak_points / append_session_brief / ...

向后兼容：
  - get_user_profile / get_user_sessions 继续工作（聚合多个 bank）
  - write_episodic_memory / update_semantic_memory 内部分发到对应 bank
  - 首次读旧 profile 命名空间时自动迁移到新 bank（一次性）

为什么细分（面试讲点）：
  1. 调用方按需取：adapt_reader 只读 preferences+mastery+weak_points，不必拉一坨 profile
  2. 写入责任清晰：error 单独成 bank，方便后续做错题本 / 间隔重复 / 知识点聚类
  3. decision_log 形成可审计的 reasoning trace，复用 P1-2 audit 的精神到记忆层
"""

import logging
import os
import time
import uuid
from collections import Counter
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv

from models.grader import GradingReport

load_dotenv(Path(__file__).parent.parent / ".env")

logger = logging.getLogger(__name__)

# 归档阈值：session_briefs 超过此值时把最老的一半聚合归档
ARCHIVE_THRESHOLD = 10
# weak_points 最多保留多少条（最新优先）
WEAK_POINTS_MAX = 20
# ASCII "STUDYMEM" encoded as a positive signed bigint. PostgresStore.setup()
# creates indexes concurrently, so the cross-worker lock must be session-level
# and held by a separate autocommit connection for the full migration call.
_POSTGRES_STORE_SETUP_LOCK_ID = 0x53545544594D454D


def _setup_postgres_store(store, database_url: str) -> None:
    import psycopg

    # Closing this dedicated connection releases the session-level advisory
    # lock on both success and failure. Non-blocking attempts are required:
    # a blocking advisory-lock query keeps a transaction active while waiting,
    # which makes CREATE INDEX CONCURRENTLY wait for that transaction.
    with psycopg.connect(database_url, autocommit=True) as connection:
        while True:
            acquired = connection.execute(
                "SELECT pg_try_advisory_lock(%s)",
                (_POSTGRES_STORE_SETUP_LOCK_ID,),
            ).fetchone()
            if acquired and acquired[0]:
                break
            time.sleep(0.05)
        store.setup()


def _enter_postgres_store(store_context, database_url: str):
    store = store_context.__enter__()
    try:
        _setup_postgres_store(store, database_url)
    except BaseException as exc:
        store_context.__exit__(type(exc), exc, exc.__traceback__)
        raise
    return store


# ── Store 后端选择（有 DATABASE_URL 走 PG，否则用内存）──
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    from langgraph.store.postgres import PostgresStore

    _store_ctx = PostgresStore.from_conn_string(DATABASE_URL)
    store = _enter_postgres_store(_store_ctx, DATABASE_URL)
else:
    from langgraph.store.memory import InMemoryStore

    store = InMemoryStore()


# ── 6 个 bank 的命名空间 ──────────────────────────────────────────────────
SEMANTIC_BANKS = ("preferences", "mastery", "weak_points")
EPISODIC_BANKS = ("session_briefs", "error_log", "decision_log")


def _bank_ns(user_id: str, bank: str) -> tuple:
    return ("users", user_id, bank)


def _now_iso() -> str:
    return datetime.now().isoformat()


# ═══════════════════════════════════════════════════════════════════════════
# 低层通用 API
# ═══════════════════════════════════════════════════════════════════════════
async def read_bank_state(user_id: str, bank: str) -> dict | None:
    """读 semantic bank 的完整状态（统一 key='current'）。"""
    item = store.get(_bank_ns(user_id, bank), "current")
    return item.value if item else None


async def write_bank_state(user_id: str, bank: str, payload: dict) -> None:
    """覆盖写 semantic bank。"""
    store.put(_bank_ns(user_id, bank), "current", payload)


async def append_bank_event(user_id: str, bank: str, key: str, payload: dict) -> None:
    """append 单条事件到 episodic bank。"""
    payload.setdefault("timestamp", _now_iso())
    store.put(_bank_ns(user_id, bank), key, payload)


async def list_bank_events(user_id: str, bank: str, limit: int = 50) -> list[dict]:
    """读 episodic bank 全部事件，按 timestamp 倒序，截到 limit。"""
    results = store.search(_bank_ns(user_id, bank))
    items = [r.value for r in results]
    items.sort(key=lambda x: x.get("timestamp", x.get("date", "")), reverse=True)
    return items[:limit]


# ═══════════════════════════════════════════════════════════════════════════
# 高层语义 API（semantic banks）
# ═══════════════════════════════════════════════════════════════════════════
async def get_preferences(user_id: str) -> dict:
    return await read_bank_state(user_id, "preferences") or {}


async def update_preferences(user_id: str, patch: dict) -> None:
    """partial update：只覆盖 patch 里的字段，其他保留。"""
    current = await get_preferences(user_id)
    current.update(patch)
    current["last_updated"] = _now_iso()
    await write_bank_state(user_id, "preferences", current)


async def get_mastery(user_id: str, document_id: str | None = None):
    """document_id=None 返回整个 dict，否则返回单个 doc 的 mastery（None 表示无数据）。"""
    state = await read_bank_state(user_id, "mastery") or {}
    return state if document_id is None else state.get(document_id)


async def update_mastery(user_id: str, document_id: str, current_score: float) -> float:
    """EMA: new = old * 0.6 + current * 0.4。返回新 mastery 值。"""
    state = await read_bank_state(user_id, "mastery") or {}
    old = state.get(document_id, current_score)   # 首次以本次分数为基准
    new_val = round(old * 0.6 + current_score * 0.4, 3)
    state[document_id] = new_val
    state["last_updated"] = _now_iso()
    await write_bank_state(user_id, "mastery", state)
    return new_val


async def get_weak_points(user_id: str, document_id: str | None = None) -> list[str]:
    """返回薄弱点字符串列表（最新在前）。

    document_id=None → 返回全部文档的薄弱点（向后兼容，旧调用方语义不变）。
    document_id 给定 → 只返回「该文档」+「无文档标记的 legacy 通用」薄弱点。

    为什么按文档隔离（2026-06-03 跨学科改进）：
      weak_points 原本全局一锅。用户上传医学/数学/影视等不同领域文档时，
      出某文档题会拿到别领域的薄弱点，污染 sufficiency 的 coverage 判定，
      还互相挤占 WEAK_POINTS_MAX 上限。改为 per-document，与 mastery 对齐。
    """
    state = await read_bank_state(user_id, "weak_points") or {}
    points = state.get("points", [])
    if document_id is not None:
        # 缺失 document_id 字段的旧条目 e.get() 返回 None，视为「通用」一并返回
        points = [e for e in points if e.get("document_id") in (document_id, None)]
    return [e["point"] for e in points if e.get("point")]


async def append_weak_points(user_id: str, new_points: list[str],
                             document_id: str | None = None) -> None:
    """去重追加薄弱点，新条目放最前。

    2026-06-03 跨学科改进——按 document_id 分桶：
      - 去重在 (document_id, point) 维度：不同文档的同名薄弱点各留一份
      - WEAK_POINTS_MAX 上限按「每文档」算，避免医学薄弱点挤占数学的额度
    """
    state = await read_bank_state(user_id, "weak_points") or {"points": []}
    existing = state.get("points", [])
    ts = _now_iso()

    # 拆成「本文档」与「其他文档」两堆：只对本文档去重 + 限额，其他文档原样保留
    same_doc = [e for e in existing if e.get("document_id") == document_id]
    other_doc = [e for e in existing if e.get("document_id") != document_id]

    same_doc_points = {e.get("point") for e in same_doc}
    new_entries = [{"point": p.strip(), "ts": ts, "document_id": document_id}
                   for p in new_points
                   if p and p.strip() and p.strip() not in same_doc_points]

    same_doc_merged = (new_entries + same_doc)[:WEAK_POINTS_MAX]
    state["points"] = same_doc_merged + other_doc
    state["last_updated"] = ts
    await write_bank_state(user_id, "weak_points", state)


async def get_prioritized_weak_points(user_id: str, document_id: str | None = None,
                                      limit: int = 10) -> list[str]:
    """按 recency（ts 倒序）加权排序薄弱点，近期错的优先召回（importance/decay 思想）。

    与 get_weak_points 的区别（不动后者，保持旧调用方语义）：
      - 显式按时间戳倒序，越近期的盲点越靠前（记忆"时间衰减"——老盲点权重低）
      - 截到 limit，供召回/画像卡用（避免一次塞太多薄弱点稀释信号）
    借鉴 hermes-agent holographic 的 trust/recency 排序思路，但用零成本的时间近度规则。
    """
    state = await read_bank_state(user_id, "weak_points") or {}
    points = state.get("points", [])
    if document_id is not None:
        points = [e for e in points if e.get("document_id") in (document_id, None)]
    points = sorted(points, key=lambda e: e.get("ts", ""), reverse=True)
    return [e["point"] for e in points if e.get("point")][:limit]


# ═══════════════════════════════════════════════════════════════════════════
# 高层语义 API（episodic banks）
# ═══════════════════════════════════════════════════════════════════════════
async def append_session_brief(user_id: str, brief: dict) -> None:
    key = brief.get("session_id") or f"sess_{uuid.uuid4().hex[:12]}"
    await append_bank_event(user_id, "session_briefs", key, brief)


async def append_error(user_id: str, error: dict) -> None:
    key = error.get("error_id") or f"err_{uuid.uuid4().hex[:12]}"
    await append_bank_event(user_id, "error_log", key, error)


async def append_decision(user_id: str, decision: dict) -> None:
    """Adapt/Critic 决策记录。formatted: {agent: str, decision: str, rationale: str, ...}"""
    key = decision.get("decision_id") or f"dec_{uuid.uuid4().hex[:12]}"
    await append_bank_event(user_id, "decision_log", key, decision)


async def list_errors(user_id: str, document_id: str | None = None, limit: int = 50) -> list[dict]:
    """错题本：可按文档过滤。"""
    items = await list_bank_events(user_id, "error_log", limit=limit * 2)
    if document_id is not None:
        items = [e for e in items if e.get("document_id") == document_id]
    return items[:limit]


# ═══════════════════════════════════════════════════════════════════════════
# 旧 API（向后兼容）—— 内部委托到新 bank
# ═══════════════════════════════════════════════════════════════════════════
async def _maybe_migrate_legacy(user_id: str) -> None:
    """旧版 profile / sessions 命名空间数据迁移到新 bank。仅在新 bank 为空时执行。"""
    # 已有任何新 bank 数据则跳过
    for bank in SEMANTIC_BANKS + EPISODIC_BANKS:
        if store.get(_bank_ns(user_id, bank), "current") or store.search(_bank_ns(user_id, bank)):
            return

    # 1. 旧 profile → mastery + weak_points + preferences
    legacy_profile = store.get(("users", user_id, "profile"), "profile")
    if legacy_profile and legacy_profile.value:
        v = legacy_profile.value
        if v.get("topic_mastery"):
            await write_bank_state(user_id, "mastery", {
                **v["topic_mastery"],
                "last_updated": v.get("last_updated", _now_iso()),
            })
        if v.get("weak_points"):
            ts = _now_iso()
            await write_bank_state(user_id, "weak_points", {
                "points": [{"point": p, "ts": ts} for p in v["weak_points"]],
                "last_updated": ts,
            })

    # 2. 旧 sessions → session_briefs（按 key 透传）
    legacy_sessions = store.search(("users", user_id, "sessions"))
    for r in legacy_sessions:
        store.put(_bank_ns(user_id, "session_briefs"), r.key, r.value)


async def get_user_profile(user_id: str) -> dict | None:
    """旧 API：聚合 preferences + mastery + weak_points + session count，重建旧字段名。"""
    await _maybe_migrate_legacy(user_id)

    prefs = await get_preferences(user_id)
    mastery_state = (await read_bank_state(user_id, "mastery") or {}).copy()
    mastery_state.pop("last_updated", None)
    wp = await get_weak_points(user_id)
    session_count = len(store.search(_bank_ns(user_id, "session_briefs")))

    if not (prefs or mastery_state or wp or session_count):
        return None

    return {
        "user_id": user_id,
        "topic_mastery": mastery_state,
        "weak_points": wp,
        "total_sessions": session_count,
        "preferences": prefs,
        "last_updated": prefs.get("last_updated") or _now_iso(),
    }


async def get_user_sessions(user_id: str) -> list[dict]:
    """旧 API：从 session_briefs bank 读，按日期倒序。"""
    await _maybe_migrate_legacy(user_id)
    items = await list_bank_events(user_id, "session_briefs", limit=200)
    items.sort(key=lambda x: x.get("date", x.get("timestamp", "")), reverse=True)
    return items


async def write_episodic_memory(user_id: str, report: GradingReport, document_id: str) -> None:
    """旧 API：拆到 session_briefs（聚合摘要）+ error_log（每错题单条）。"""
    knowledge_gaps = [g.knowledge_gap for g in report.grades
                      if not g.is_correct and g.knowledge_gap]
    await append_session_brief(user_id, {
        "session_id": report.session_id,
        "document_id": document_id,
        "date": date.today().isoformat(),
        "correct_rate": report.score,
        "total_questions": report.total,
        "knowledge_gaps": knowledge_gaps,
    })

    # 每个错题单独入 error_log（便于错题本 / 间隔重复）
    for g in report.grades:
        if not g.is_correct:
            await append_error(user_id, {
                "session_id": report.session_id,
                "document_id": document_id,
                "question": g.question,
                "user_answer": g.user_answer,
                "correct_answer": g.correct_answer,
                "knowledge_gap": g.knowledge_gap,
            })

    await maybe_archive_session_briefs(user_id)


async def update_semantic_memory(user_id: str, report: GradingReport, document_id: str) -> None:
    """旧 API：更新 mastery（EMA）+ 追加 weak_points。"""
    await update_mastery(user_id, document_id, report.score)
    new_gaps = [g.knowledge_gap for g in report.grades
                if not g.is_correct and g.knowledge_gap]
    if new_gaps:
        await append_weak_points(user_id, new_gaps, document_id)


async def consolidate_session_extras(user_id: str, report: GradingReport, document_id: str,
                                     history: list[dict] | None = None,
                                     question_type: str | None = None) -> dict:
    """会话结束的增量 consolidation（adapt_writer 之外）——零 LLM，借鉴 claw-code 启发式压缩。

    与 adapt_writer 分工（避免重复写）：
      adapt_writer 已写 mastery(EMA) / weak_points / session_briefs / error_log / decision_log；
      这里只补它没做的两件事：
        ① 填 preferences "只读不写" 缺口：从 report + history 规则推断偏好并 update_preferences
        ② 刷新可读"学习者画像卡"存入 preferences.profile_card（随快照持久、供前端/注入展示）
      最后落盘本机快照（仅 InMemoryStore；Postgres no-op）。

    全程 fail-soft：任一步异常都不抛出（grader_worker 已在外层 try，这里再兜一层）。
    返回写入的 preferences patch（便于 grader_worker 回填 state / 测试断言）。

    延迟导入 preference_learning / memory_context / memory_persist：避免与本模块的模块级循环依赖。
    """
    from services.memory_context import build_profile_card
    from services.memory_persist import save_snapshot
    from services.preference_learning import infer_preferences

    patch: dict = {}
    # ① 偏好推断 → 填 preferences 只读不写缺口
    try:
        current = await get_preferences(user_id)
        patch = infer_preferences(report, history or [], current, question_type=question_type)
        if patch:
            await update_preferences(user_id, patch)
    except Exception as e:
        logger.warning(f"[consolidate] infer/update preferences 失败: {e}")

    # ② 刷新画像卡（在偏好更新之后构建，含最新 preferred_*）
    try:
        card = await build_profile_card(user_id)
        if card:
            await update_preferences(user_id, {"profile_card": card})
    except Exception as e:
        logger.warning(f"[consolidate] build/store profile_card 失败: {e}")

    # ③ 本机快照落盘（原子写；Postgres 后端 no-op）
    try:
        save_snapshot()
    except Exception as e:
        logger.warning(f"[consolidate] save_snapshot 失败: {e}")

    return patch


# ═══════════════════════════════════════════════════════════════════════════
# 归档（session_briefs 满了就把最旧的一半聚合）
# ═══════════════════════════════════════════════════════════════════════════
def _aggregate_episodes(episodes: list[dict]) -> dict:
    dates = sorted(e.get("date", "") for e in episodes if e.get("date"))
    avg_rate = round(sum(e.get("correct_rate", 0) for e in episodes) / len(episodes), 3)
    all_gaps = [gap for e in episodes for gap in e.get("knowledge_gaps", [])]
    top_gaps = [gap for gap, _ in Counter(all_gaps).most_common(10)]
    return {
        "type": "archive",
        "period": f"{dates[0]} ~ {dates[-1]}" if dates else "unknown",
        "session_count": len(episodes),
        "avg_correct_rate": avg_rate,
        "top_knowledge_gaps": top_gaps,
        "timestamp": _now_iso(),
    }


async def maybe_archive_session_briefs(user_id: str) -> None:
    results = store.search(_bank_ns(user_id, "session_briefs"))
    if len(results) <= ARCHIVE_THRESHOLD:
        return

    sorted_results = sorted(results, key=lambda r: r.value.get("date", ""))
    cutoff = len(sorted_results) // 2
    to_archive = sorted_results[:cutoff]

    archive_entry = _aggregate_episodes([r.value for r in to_archive])
    archive_key = f"archive_{to_archive[0].value.get('date', 'old')}"
    store.put(_bank_ns(user_id, "session_briefs"), archive_key, archive_entry)

    for r in to_archive:
        store.delete(_bank_ns(user_id, "session_briefs"), r.key)
