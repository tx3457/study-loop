"""
Memory Banks

数据按 6 个语义 bank 存储：

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

分 bank 的原因：
  1. 调用方按需取：adapt_reader 只读 preferences+mastery+weak_points，不必拉一坨 profile
  2. 写入责任清晰：error 单独成 bank，方便后续做错题本 / 间隔重复 / 知识点聚类
  3. decision_log 形成可审计的 reasoning trace
"""

import asyncio
import copy
import hashlib
import logging
import os
import threading
import time
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from datetime import date, datetime
from pathlib import Path
from typing import TypeVar

from dotenv import load_dotenv

from models.grader import GradingReport
from models.quiz import Question

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
_POSTGRES_STORE_SETUP_LOCK_TIMEOUT_ENV = "MEMORY_STORE_SETUP_LOCK_TIMEOUT_SECONDS"
_POSTGRES_STORE_SETUP_LOCK_TIMEOUT_DEFAULT_SECONDS = 300.0
_POSTGRES_STORE_SETUP_LOCK_POLL_SECONDS = 0.05
_POSTGRES_SESSION_ARCHIVE_LOCK_TIMEOUT_ENV = (
    "MEMORY_SESSION_ARCHIVE_LOCK_TIMEOUT_SECONDS"
)
_POSTGRES_SESSION_ARCHIVE_LOCK_TIMEOUT_DEFAULT_SECONDS = 10.0
_POSTGRES_SESSION_ARCHIVE_LOCK_POLL_SECONDS = 0.02
_MEMORY_COMMIT_CANCEL_DRAIN_TIMEOUT_SECONDS = 15.0
_WRONG_QUESTION_SOURCE_PREFIX = "wrong-question:"
_STORE_LOCK = threading.RLock()
_BACKGROUND_MEMORY_COMMITS: set[asyncio.Task] = set()
_T = TypeVar("_T")


class PostgresStoreSetupLockTimeoutError(TimeoutError):
    """Another worker did not finish learner-memory schema setup in time."""


class PostgresSessionArchiveLockTimeoutError(TimeoutError):
    """Another worker did not release a user's learner-memory lock in time."""


def _postgres_store_setup_lock_timeout_seconds() -> float:
    source = os.getenv(
        _POSTGRES_STORE_SETUP_LOCK_TIMEOUT_ENV,
        str(_POSTGRES_STORE_SETUP_LOCK_TIMEOUT_DEFAULT_SECONDS),
    )
    try:
        value = float(source)
    except (TypeError, ValueError):
        raise ValueError(
            f"{_POSTGRES_STORE_SETUP_LOCK_TIMEOUT_ENV} must be a positive number"
        ) from None
    if not 0 < value < float("inf"):
        raise ValueError(
            f"{_POSTGRES_STORE_SETUP_LOCK_TIMEOUT_ENV} must be a positive number"
        )
    return value


def _postgres_session_archive_lock_timeout_seconds() -> float:
    source = os.getenv(
        _POSTGRES_SESSION_ARCHIVE_LOCK_TIMEOUT_ENV,
        str(_POSTGRES_SESSION_ARCHIVE_LOCK_TIMEOUT_DEFAULT_SECONDS),
    )
    try:
        value = float(source)
    except (TypeError, ValueError):
        raise ValueError(
            f"{_POSTGRES_SESSION_ARCHIVE_LOCK_TIMEOUT_ENV} must be a positive number"
        ) from None
    if not 0 < value < float("inf"):
        raise ValueError(
            f"{_POSTGRES_SESSION_ARCHIVE_LOCK_TIMEOUT_ENV} must be a positive number"
        )
    return value


def _setup_postgres_store(store, database_url: str) -> None:
    import psycopg

    timeout_seconds = _postgres_store_setup_lock_timeout_seconds()
    # Closing this dedicated connection releases the session-level advisory
    # lock on both success and failure. Non-blocking attempts are required:
    # a blocking advisory-lock query keeps a transaction active while waiting,
    # which makes CREATE INDEX CONCURRENTLY wait for that transaction.
    with psycopg.connect(database_url, autocommit=True) as connection:
        deadline = time.monotonic() + timeout_seconds
        while True:
            acquired = connection.execute(
                "SELECT pg_try_advisory_lock(%s)",
                (_POSTGRES_STORE_SETUP_LOCK_ID,),
            ).fetchone()
            if acquired and acquired[0]:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                unit = "second" if timeout_seconds == 1 else "seconds"
                raise PostgresStoreSetupLockTimeoutError(
                    f"Timed out after {timeout_seconds:g} {unit} waiting for "
                    "PostgreSQL learner-memory schema lock; another worker may "
                    "still be initializing it. Increase "
                    f"{_POSTGRES_STORE_SETUP_LOCK_TIMEOUT_ENV} or initialize "
                    "the store before starting workers."
                )
            time.sleep(min(_POSTGRES_STORE_SETUP_LOCK_POLL_SECONDS, remaining))
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


def _store_access_lock():
    """本地 Store/快照共享锁；PostgreSQL 并发由数据库和 advisory lock 管理。"""
    return _STORE_LOCK if not DATABASE_URL else nullcontext()


# ── 6 个 bank 的命名空间 ──────────────────────────────────────────────────
SEMANTIC_BANKS = ("preferences", "mastery", "weak_points")
EPISODIC_BANKS = ("session_briefs", "error_log", "decision_log")


def _bank_ns(user_id: str, bank: str) -> tuple:
    return ("users", user_id, bank)


def _now_iso() -> str:
    return datetime.now().isoformat()


def _session_archive_lock_id(user_id: str) -> int:
    """为每个用户生成稳定的负数 advisory-lock id，与 schema 锁隔离。"""
    digest = hashlib.sha256(
        f"study-loop:session-archive:{user_id}".encode("utf-8")
    ).digest()
    value = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
    return -(value or 1)


def _search_all_items(
    namespace: tuple,
    *,
    event_filter: dict | None = None,
    batch_size: int = 100,
) -> list:
    """分页读取完整 namespace，避免 Store 默认只返回 10 条。"""
    with _store_access_lock():
        items = []
        offset = 0
        while True:
            batch = store.search(
                namespace,
                filter=event_filter,
                limit=batch_size,
                offset=offset,
            )
            items.extend(batch)
            if len(batch) < batch_size:
                return items
            offset += len(batch)


# ═══════════════════════════════════════════════════════════════════════════
# 低层通用 API
# ═══════════════════════════════════════════════════════════════════════════
def _read_bank_state_sync(user_id: str, bank: str) -> dict | None:
    with _store_access_lock():
        item = store.get(_bank_ns(user_id, bank), "current")
    return copy.deepcopy(item.value) if item else None


def _write_bank_state_sync(user_id: str, bank: str, payload: dict) -> None:
    with _store_access_lock():
        store.put(_bank_ns(user_id, bank), "current", copy.deepcopy(payload))


async def read_bank_state(user_id: str, bank: str) -> dict | None:
    """读 semantic bank 的完整状态（统一 key='current'）。"""
    return _read_bank_state_sync(user_id, bank)


async def write_bank_state(user_id: str, bank: str, payload: dict) -> None:
    """覆盖写 semantic bank。"""
    _write_bank_state_sync(user_id, bank, payload)


async def mutate_bank_state(
    user_id: str,
    bank: str,
    mutator: Callable[[dict | None], tuple[dict, _T]],
) -> _T:
    """Atomically read/modify/write one user's bank.

    PostgreSQL workers share the existing per-user advisory lock; the local
    fallback uses the snapshot/store RLock. The mutator must be synchronous and
    side-effect free apart from changing the returned state.
    """

    result: list[_T] = []

    def operation() -> None:
        with _store_access_lock():
            current = _read_bank_state_sync(user_id, bank)
            before = copy.deepcopy(current)
            updated, value = mutator(copy.deepcopy(current))
            if updated != before:
                _write_bank_state_sync(user_id, bank, updated)
            result.append(value)

    if DATABASE_URL:
        await asyncio.to_thread(
            _run_with_postgres_session_archive_lock,
            user_id,
            DATABASE_URL,
            operation,
        )
    else:
        operation()
    return result[0]


async def append_bank_event(user_id: str, bank: str, key: str, payload: dict) -> None:
    """append 单条事件到 episodic bank。"""
    stored_payload = copy.deepcopy(payload)
    stored_payload.setdefault("timestamp", _now_iso())
    with _store_access_lock():
        store.put(_bank_ns(user_id, bank), key, stored_payload)


async def list_bank_events(user_id: str, bank: str, limit: int = 50) -> list[dict]:
    """读 episodic bank 全部事件，按 timestamp 倒序，截到 limit。"""
    if limit <= 0:
        return []
    results = _search_all_items(_bank_ns(user_id, bank))
    items = [copy.deepcopy(r.value) for r in results]
    items.sort(key=lambda x: x.get("timestamp", x.get("date", "")), reverse=True)
    return items[:limit]


# ═══════════════════════════════════════════════════════════════════════════
# 高层语义 API（semantic banks）
# ═══════════════════════════════════════════════════════════════════════════
async def get_preferences(user_id: str) -> dict:
    return await read_bank_state(user_id, "preferences") or {}


async def update_preferences(user_id: str, patch: dict) -> None:
    """partial update：只覆盖 patch 里的字段，其他保留。"""
    def apply_patch(current: dict | None) -> tuple[dict, None]:
        updated = current or {}
        updated.update(copy.deepcopy(patch))
        updated["last_updated"] = _now_iso()
        return updated, None

    await mutate_bank_state(user_id, "preferences", apply_patch)


async def get_mastery(user_id: str, document_id: str | None = None):
    """document_id=None 返回整个 dict，否则返回单个 doc 的 mastery（None 表示无数据）。"""
    state = await read_bank_state(user_id, "mastery") or {}
    if document_id is not None:
        return state.get(document_id)
    state.pop("_applied_sessions", None)
    return state


def _update_mastery_sync(
    user_id: str,
    document_id: str,
    current_score: float,
    *,
    session_id: str | None = None,
) -> float:
    """单行状态内同时保存 EMA 和已应用 session，令部分失败后的重试幂等。"""
    state = _read_bank_state_sync(user_id, "mastery") or {}
    applied_by_document = copy.deepcopy(state.get("_applied_sessions") or {})
    applied_sessions = list(applied_by_document.get(document_id) or [])
    if session_id and session_id in applied_sessions:
        existing = state.get(document_id)
        return float(existing) if isinstance(existing, (int, float)) else current_score

    old = state.get(document_id, current_score)   # 首次以本次分数为基准
    new_val = round(old * 0.6 + current_score * 0.4, 3)
    state[document_id] = new_val
    if session_id:
        applied_sessions.append(session_id)
        applied_by_document[document_id] = applied_sessions
        state["_applied_sessions"] = applied_by_document
    state["last_updated"] = _now_iso()
    _write_bank_state_sync(user_id, "mastery", state)
    return new_val


async def update_mastery(user_id: str, document_id: str, current_score: float) -> float:
    """EMA: new = old * 0.6 + current * 0.4。返回新 mastery 值。"""
    if DATABASE_URL:
        result: list[float] = []

        def operation() -> None:
            result.append(
                _update_mastery_sync(user_id, document_id, current_score)
            )

        await asyncio.to_thread(
            _run_with_postgres_session_archive_lock,
            user_id,
            DATABASE_URL,
            operation,
        )
        return result[0]
    return _update_mastery_sync(user_id, document_id, current_score)


async def get_weak_points(user_id: str, document_id: str | None = None) -> list[str]:
    """返回薄弱点字符串列表（最新在前）。

    document_id=None → 返回全部文档的薄弱点（兼容未指定文档范围的调用）。
    document_id 给定 → 只返回「该文档」+「无文档标记的 legacy 通用」薄弱点。

    为什么按文档隔离：
      用户上传医学/数学/影视等不同领域文档时，全局 weak_points 会把别领域
      的薄弱点带入当前文档，污染 sufficiency 的 coverage 判定，并互相挤占
      WEAK_POINTS_MAX 上限。per-document 隔离与 mastery 的粒度一致。
    """
    state = await read_bank_state(user_id, "weak_points") or {}
    points = state.get("points", [])
    if document_id is not None:
        # 缺失 document_id 字段的旧条目 e.get() 返回 None，视为「通用」一并返回
        points = [e for e in points if e.get("document_id") in (document_id, None)]
    return [e["point"] for e in points if e.get("point")]


def _append_weak_points_sync(
    user_id: str,
    new_points: list[str],
    document_id: str | None = None,
) -> None:
    """去重追加薄弱点，新条目放最前。

    按 document_id 分桶：
      - 去重在 (document_id, point) 维度：不同文档的同名薄弱点各留一份
      - WEAK_POINTS_MAX 上限按「每文档」算，避免医学薄弱点挤占数学的额度
    """
    state = _read_bank_state_sync(user_id, "weak_points") or {"points": []}
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
    _write_bank_state_sync(user_id, "weak_points", state)


async def append_weak_points(user_id: str, new_points: list[str],
                             document_id: str | None = None) -> None:
    if DATABASE_URL:
        await asyncio.to_thread(
            _run_with_postgres_session_archive_lock,
            user_id,
            DATABASE_URL,
            lambda: _append_weak_points_sync(
                user_id,
                new_points,
                document_id,
            ),
        )
        return
    _append_weak_points_sync(user_id, new_points, document_id)


async def get_prioritized_weak_points(user_id: str, document_id: str | None = None,
                                      limit: int = 10) -> list[str]:
    """按 recency（ts 倒序）加权排序薄弱点，近期错的优先召回（importance/decay 思想）。

    与 get_weak_points 的区别（后者继续保持原有调用语义）：
      - 显式按时间戳倒序，越近期的盲点越靠前（记忆"时间衰减"——老盲点权重低）
      - 截到 limit，供召回/画像卡用（避免一次塞太多薄弱点稀释信号）
    使用零成本的时间近度规则。
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
def _append_session_brief_sync(user_id: str, brief: dict) -> None:
    key = brief.get("session_id") or f"sess_{uuid.uuid4().hex[:12]}"
    namespace = _bank_ns(user_id, "session_briefs")
    with _store_access_lock():
        # session_id 是不可变事件键：请求重试不能覆盖归档线程已经读取的值，
        # 否则 archive 写入旧统计后会删除刚覆盖的新值。
        if brief.get("session_id") and store.get(namespace, key) is not None:
            return
        if brief.get("session_id"):
            # 已归档会话的迟到重试不应重新变成 raw brief 并重复计数。
            archives = _search_all_items(
                namespace,
                event_filter={"type": "archive"},
            )
            if any(key in (item.value.get("session_ids") or []) for item in archives):
                return
        stored_brief = copy.deepcopy(brief)
        stored_brief.setdefault("timestamp", _now_iso())
        store.put(namespace, key, stored_brief)


async def append_session_brief(user_id: str, brief: dict) -> None:
    if DATABASE_URL:
        # 与归档共用同一个跨进程锁，使“查重→写入”和“扫描→归档→删除”
        # 不会在多个 worker 间交错。阻塞的 psycopg 调用在线程中执行，不冻结
        # FastAPI 事件循环；上层 commit_learning_memory 会在请求取消时等待提交收尾。
        await asyncio.to_thread(
            _run_with_postgres_session_archive_lock,
            user_id,
            DATABASE_URL,
            lambda: _append_session_brief_sync(user_id, brief),
        )
        return
    _append_session_brief_sync(user_id, brief)


async def append_error(user_id: str, error: dict) -> None:
    key = error.get("error_id") or f"err_{uuid.uuid4().hex[:12]}"
    payload = {**error, "error_id": key}
    await append_bank_event(user_id, "error_log", key, payload)


async def resolve_error(
    user_id: str,
    error_id: str,
    resolved_session_id: str,
) -> bool:
    """把重练答对的错题标为已解决，同时保留原始事件供审计。"""
    namespace = _bank_ns(user_id, "error_log")
    with _store_access_lock():
        item = store.get(namespace, error_id)
        if item is None:
            return False
        payload = {
            **copy.deepcopy(item.value),
            "resolved_at": _now_iso(),
            "resolved_session_id": resolved_session_id,
        }
        store.put(namespace, error_id, payload)
    return True


async def append_decision(user_id: str, decision: dict) -> None:
    """Adapt/Critic 决策记录。formatted: {agent: str, decision: str, rationale: str, ...}"""
    key = decision.get("decision_id") or f"dec_{uuid.uuid4().hex[:12]}"
    await append_bank_event(user_id, "decision_log", key, decision)


async def list_errors(user_id: str, document_id: str | None = None, limit: int = 50) -> list[dict]:
    """未解决错题：在存储层按文档过滤，分页读取后再按时间排序。"""
    if limit <= 0:
        return []

    event_filter = {"document_id": document_id} if document_id is not None else None
    results = _search_all_items(
        _bank_ns(user_id, "error_log"),
        event_filter=event_filter,
        batch_size=max(100, limit),
    )
    items = [
        {
            **copy.deepcopy(result.value),
            # 升级前 payload 没有 error_id；注入真实 store key 才能在
            # 重练答对时回写同一条记录，而不是生成无法解析的展示 ID。
            "error_id": result.value.get("error_id") or result.key,
        }
        for result in results
        if not result.value.get("resolved_at")
    ]

    items.sort(
        key=lambda item: item.get("timestamp", item.get("date", "")),
        reverse=True,
    )
    return items[:limit]


# ═══════════════════════════════════════════════════════════════════════════
# 兼容 API：内部委托到语义 bank
# ═══════════════════════════════════════════════════════════════════════════
async def _maybe_migrate_legacy(user_id: str) -> None:
    """将 legacy profile / sessions 命名空间迁移到语义 bank，仅在目标为空时执行。"""
    # 已有任何新 bank 数据则跳过
    for bank in SEMANTIC_BANKS + EPISODIC_BANKS:
        with _store_access_lock():
            has_state = store.get(_bank_ns(user_id, bank), "current")
            has_events = store.search(_bank_ns(user_id, bank))
        if has_state or has_events:
            return

    # 1. legacy profile → mastery + weak_points + preferences
    with _store_access_lock():
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

    # 2. legacy sessions → session_briefs（按 key 透传）
    legacy_sessions = _search_all_items(("users", user_id, "sessions"))
    for r in legacy_sessions:
        with _store_access_lock():
            store.put(_bank_ns(user_id, "session_briefs"), r.key, r.value)


def _count_sessions(results: list) -> int:
    """按真实会话数统计；archive 物理上是一条，但代表多次会话。"""
    session_ids: set[str] = set()
    legacy_archive_count = 0
    for result in results:
        value = result.value
        if value.get("type") == "archive":
            archived_ids = value.get("session_ids")
            if archived_ids:
                session_ids.update(str(item) for item in archived_ids)
            else:
                count = value.get("session_count", 0)
                if isinstance(count, int) and not isinstance(count, bool):
                    legacy_archive_count += max(count, 0)
            continue
        session_ids.add(str(value.get("session_id") or result.key))
    return len(session_ids) + legacy_archive_count


def _average_session_score(results: list) -> float | None:
    """按归档所代表的会话数加权，且忽略 crash-window 的 raw 重复项。"""
    seen_ids: set[str] = set()
    weighted_score = 0.0
    scored_count = 0

    archives = sorted(
        (result for result in results if result.value.get("type") == "archive"),
        key=lambda result: result.key,
    )
    for result in archives:
        value = result.value
        rate = value.get("avg_correct_rate")
        if not isinstance(rate, (int, float)) or isinstance(rate, bool):
            continue
        archived_ids = [str(item) for item in (value.get("session_ids") or [])]
        if archived_ids:
            new_ids = [item for item in archived_ids if item not in seen_ids]
            seen_ids.update(archived_ids)
            weight = len(new_ids)
        else:
            count = value.get("session_count", 0)
            weight = count if isinstance(count, int) and not isinstance(count, bool) else 0
        weighted_score += float(rate) * max(weight, 0)
        scored_count += max(weight, 0)

    for result in results:
        value = result.value
        if value.get("type") == "archive":
            continue
        session_id = str(value.get("session_id") or result.key)
        if session_id in seen_ids:
            continue
        rate = value.get("correct_rate")
        if isinstance(rate, (int, float)) and not isinstance(rate, bool):
            weighted_score += float(rate)
            scored_count += 1
        seen_ids.add(session_id)

    return round(weighted_score / scored_count, 3) if scored_count else None


async def get_user_session_count(user_id: str) -> int:
    """返回包含归档内容的精确会话数，不受历史展示上限影响。"""
    await _maybe_migrate_legacy(user_id)
    return _count_sessions(
        _search_all_items(_bank_ns(user_id, "session_briefs"))
    )


async def get_user_profile(user_id: str) -> dict | None:
    """兼容 API：聚合 preferences + mastery + weak_points + session count。"""
    await _maybe_migrate_legacy(user_id)

    prefs = await get_preferences(user_id)
    mastery_state = (await read_bank_state(user_id, "mastery") or {}).copy()
    mastery_state.pop("last_updated", None)
    mastery_state.pop("_applied_sessions", None)
    wp = await get_weak_points(user_id)
    session_results = _search_all_items(_bank_ns(user_id, "session_briefs"))
    session_count = _count_sessions(session_results)
    average_correct_rate = _average_session_score(session_results)

    if not (prefs or mastery_state or wp or session_count):
        return None

    return {
        "user_id": user_id,
        "topic_mastery": mastery_state,
        "weak_points": wp,
        "total_sessions": session_count,
        "average_correct_rate": average_correct_rate,
        "preferences": prefs,
        "last_updated": prefs.get("last_updated") or _now_iso(),
    }


async def get_user_sessions(user_id: str) -> list[dict]:
    """兼容 API：从 session_briefs bank 读取，按日期倒序。"""
    await _maybe_migrate_legacy(user_id)
    results = _search_all_items(_bank_ns(user_id, "session_briefs"))
    archived_ids = {
        str(session_id)
        for result in results
        if result.value.get("type") == "archive"
        for session_id in (result.value.get("session_ids") or [])
    }
    items = [
        copy.deepcopy(result.value)
        for result in results
        if result.value.get("type") == "archive"
        or str(result.value.get("session_id") or result.key) not in archived_ids
    ]
    items.sort(key=lambda x: x.get("date") or x.get("timestamp", ""), reverse=True)
    return items[:200]


async def write_episodic_memory(
    user_id: str,
    report: GradingReport,
    document_id: str,
    questions: list[Question] | None = None,
) -> None:
    """兼容 API：写入 session_briefs（聚合摘要）+ error_log（每错题单条）。"""
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
        question = (
            questions[g.index]
            if questions is not None and 0 <= g.index < len(questions)
            else None
        )
        source_error_id = None
        if question and question.source.startswith(_WRONG_QUESTION_SOURCE_PREFIX):
            source_error_id = question.source.removeprefix(
                _WRONG_QUESTION_SOURCE_PREFIX
            )

        if g.is_correct:
            if source_error_id:
                resolved = await resolve_error(
                    user_id,
                    source_error_id,
                    report.session_id,
                )
                if not resolved:
                    logger.warning(
                        "重练已答对，但源错题不存在: user=%s error_id=%s",
                        user_id,
                        source_error_id,
                    )
            continue

        await append_error(user_id, {
            # 同一份批改报告重试时覆盖原记录，不重复追加错题。
            # 重练仍答错时覆盖原错题，使未解决列表不会分裂出副本。
            "error_id": source_error_id or f"{report.session_id}:{g.index}",
            "session_id": report.session_id,
            "question_index": g.index,
            "document_id": document_id,
            "question": g.question,
            "options": question.options if question else None,
            "question_type": question.type if question else "short_answer",
            "user_answer": g.user_answer,
            "correct_answer": g.correct_answer,
            "explanation": (
                question.explanation
                if question
                else (g.ai_feedback or g.knowledge_gap or "")
            ),
            "knowledge_gap": g.knowledge_gap,
            "source_error_id": source_error_id,
        })

def _update_semantic_memory_sync(
    user_id: str,
    report: GradingReport,
    document_id: str,
) -> None:
    """在一个用户级临界区中提交幂等 mastery 与可重试 weak-points。"""
    with _store_access_lock():
        _update_mastery_sync(
            user_id,
            document_id,
            report.score,
            session_id=report.session_id,
        )
        new_gaps = [
            grade.knowledge_gap
            for grade in report.grades
            if not grade.is_correct and grade.knowledge_gap
        ]
        if new_gaps:
            _append_weak_points_sync(user_id, new_gaps, document_id)


async def update_semantic_memory(user_id: str, report: GradingReport, document_id: str) -> None:
    """更新 mastery（按 session 幂等）并追加 weak_points。"""
    if DATABASE_URL:
        await asyncio.to_thread(
            _run_with_postgres_session_archive_lock,
            user_id,
            DATABASE_URL,
            lambda: _update_semantic_memory_sync(user_id, report, document_id),
        )
        return
    _update_semantic_memory_sync(user_id, report, document_id)


async def persist_memory_snapshot() -> bool:
    """把当前 InMemoryStore 原子落盘；PostgreSQL 后端自动 no-op。"""
    from services.memory_persist import persist_snapshot

    try:
        return await persist_snapshot()
    except Exception as exc:
        logger.warning("[memory] persist snapshot 失败（保留进程内状态）: %s", exc)
        return False


async def _complete_memory_commit(operation: Awaitable[None]) -> None:
    """请求取消时仍等待已开始的记忆提交结束，再把取消信号交还调用方。"""
    task = asyncio.create_task(operation)
    try:
        await asyncio.shield(task)
        return
    except asyncio.CancelledError as cancellation:
        # shield 防止外层取消传播给提交任务。连续取消也只延后响应取消，不能把
        # learner memory 轻易留在“session 已写、mastery 未写”的半提交状态。
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _MEMORY_COMMIT_CANCEL_DRAIN_TIMEOUT_SECONDS
        while not task.done():
            remaining = deadline - loop.time()
            if remaining <= 0:
                _track_background_memory_commit(task)
                logger.warning(
                    "cancelled request stopped waiting for learner-memory commit "
                    "after %.0fs; commit continues in background",
                    _MEMORY_COMMIT_CANCEL_DRAIN_TIMEOUT_SECONDS,
                )
                raise cancellation
            try:
                await asyncio.wait_for(
                    asyncio.shield(task),
                    timeout=remaining,
                )
            except asyncio.CancelledError:
                continue
            except asyncio.TimeoutError:
                continue
        if task.cancelled():
            raise cancellation
        task.result()  # 若提交自身失败，优先暴露真实写入错误。
        raise cancellation


def _track_background_memory_commit(task: asyncio.Task) -> None:
    """保留超时后的提交任务，并消费异常，避免任务被回收或静默告警。"""
    _BACKGROUND_MEMORY_COMMITS.add(task)

    def on_done(completed: asyncio.Task) -> None:
        _BACKGROUND_MEMORY_COMMITS.discard(completed)
        try:
            completed.result()
        except asyncio.CancelledError:
            logger.warning("background learner-memory commit was cancelled")
        except Exception:
            logger.exception("background learner-memory commit failed")

    task.add_done_callback(on_done)


async def commit_learning_memory(
    user_id: str,
    report: GradingReport,
    document_id: str,
    *,
    questions: list[Question] | None = None,
    after_write: Callable[[], Awaitable[None]] | None = None,
    on_core_written: Callable[[], None] | None = None,
) -> None:
    """完整提交 episodic + semantic memory，再归档并刷新本地快照。"""

    async def operation() -> None:
        await write_episodic_memory(
            user_id,
            report,
            document_id,
            questions=questions,
        )
        await update_semantic_memory(user_id, report, document_id)
        if on_core_written is not None:
            on_core_written()
        if after_write is not None:
            try:
                await after_write()
            except Exception as exc:
                logger.warning(
                    "learner-memory optional audit write failed user=%s: %s",
                    user_id,
                    exc,
                )
        await maybe_archive_session_briefs(user_id)
        await persist_memory_snapshot()

    await _complete_memory_commit(operation())


async def consolidate_session_extras(user_id: str, report: GradingReport, document_id: str,
                                     history: list[dict] | None = None,
                                     question_type: str | None = None,
                                     session_id: str | None = None) -> dict:
    """会话结束的增量 consolidation（adapt_writer 之外），全程零 LLM。

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
    from services.preference_learning import infer_preferences

    patch: dict = {}
    applied = True
    # ① 偏好推断 → 填 preferences 只读不写缺口。Tutor 重放按 session 幂等。
    try:
        if session_id:
            def apply_once(current: dict | None) -> tuple[dict, tuple[dict, bool]]:
                current = current or {}
                applied_sessions = list(current.get("_consolidated_sessions") or [])
                if session_id in applied_sessions:
                    return current, ({}, False)
                inferred = infer_preferences(
                    report,
                    history or [],
                    current,
                    question_type=question_type,
                )
                updated = {**current, **inferred}
                applied_sessions.append(session_id)
                updated["_consolidated_sessions"] = applied_sessions
                updated["last_updated"] = _now_iso()
                return updated, (inferred, True)

            patch, applied = await mutate_bank_state(
                user_id,
                "preferences",
                apply_once,
            )
        else:
            current = await get_preferences(user_id)
            patch = infer_preferences(
                report,
                history or [],
                current,
                question_type=question_type,
            )
            if patch:
                await update_preferences(user_id, patch)
    except Exception as e:
        logger.warning(f"[consolidate] infer/update preferences 失败: {e}")

    if not applied:
        return patch

    # ② 刷新画像卡（在偏好更新之后构建，含最新 preferred_*）
    try:
        card = await build_profile_card(user_id)
        if card:
            await update_preferences(user_id, {"profile_card": card})
    except Exception as e:
        logger.warning(f"[consolidate] build/store profile_card 失败: {e}")

    # ③ 本机快照落盘（原子写；Postgres 后端 no-op）
    await persist_memory_snapshot()

    return patch


# ═══════════════════════════════════════════════════════════════════════════
# 归档（session_briefs 满了就把最旧的一半聚合）
# ═══════════════════════════════════════════════════════════════════════════
def _aggregate_episodes(episodes: list[tuple[str, dict]]) -> dict:
    values = [value for _, value in episodes]
    session_ids = [str(value.get("session_id") or key) for key, value in episodes]
    dates = sorted(value.get("date", "") for value in values if value.get("date"))
    avg_rate = round(
        sum(value.get("correct_rate", 0) for value in values) / len(values),
        3,
    )
    all_gaps = [
        gap
        for value in values
        for gap in value.get("knowledge_gaps", [])
    ]
    top_gaps = [gap for gap, _ in Counter(all_gaps).most_common(10)]
    period_start = dates[0] if dates else None
    period_end = dates[-1] if dates else None
    return {
        "type": "archive",
        "period": (
            f"{period_start} ~ {period_end}"
            if period_start and period_end
            else "unknown"
        ),
        "period_start": period_start,
        "period_end": period_end,
        # date 供现有历史排序使用，避免 archive 因缺 date 永远排在最前面。
        "date": period_end or "",
        "session_ids": session_ids,
        "session_count": len(session_ids),
        "avg_correct_rate": avg_rate,
        "top_knowledge_gaps": top_gaps,
        "total_questions": sum(value.get("total_questions", 0) for value in values),
        "timestamp": _now_iso(),
    }


def _archive_session_briefs_sync(user_id: str) -> None:
    """在进程内锁中执行归档，确保本地快照看到完整的前/后状态。"""
    with _store_access_lock():
        _archive_session_briefs_unlocked(user_id)


def _archive_session_briefs_unlocked(user_id: str) -> None:
    """执行一次归档；PostgreSQL 调用方还必须先持有该用户的 advisory lock。"""
    namespace = _bank_ns(user_id, "session_briefs")
    results = _search_all_items(namespace)
    archived_ids = {
        str(session_id)
        for result in results
        if result.value.get("type") == "archive"
        for session_id in (result.value.get("session_ids") or [])
    }
    raw_results = []
    for result in results:
        if result.value.get("type") == "archive":
            continue
        session_id = str(result.value.get("session_id") or result.key)
        if session_id in archived_ids:
            # archive 先写后删；若上次进程在删除途中退出，这里完成清理。
            store.delete(namespace, result.key)
            continue
        raw_results.append(result)
    if len(raw_results) <= ARCHIVE_THRESHOLD:
        return

    sorted_results = sorted(
        raw_results,
        key=lambda result: (
            result.value.get("date") or "",
            result.value.get("timestamp") or "",
            result.key,
        ),
    )
    cutoff = len(sorted_results) // 2
    to_archive = sorted_results[:cutoff]

    episodes = [(result.key, result.value) for result in to_archive]
    archive_entry = _aggregate_episodes(episodes)
    digest = hashlib.sha256(
        "\x1f".join(archive_entry["session_ids"]).encode("utf-8")
    ).hexdigest()[:20]
    archive_key = f"archive_{digest}"
    store.put(namespace, archive_key, archive_entry)

    for result in to_archive:
        store.delete(namespace, result.key)


def _run_with_postgres_session_archive_lock(
    user_id: str,
    database_url: str,
    operation: Callable[[], None],
) -> None:
    """在用户级 PostgreSQL session lock 内执行 session 写入或归档。"""
    import psycopg

    lock_id = _session_archive_lock_id(user_id)
    timeout_seconds = _postgres_session_archive_lock_timeout_seconds()
    connect_timeout = max(1, int(timeout_seconds + 0.999))
    statement_timeout_ms = max(1, int(timeout_seconds * 1000))
    # 使用独立直连而不是 Store 的池连接；with 退出会关闭 session，并在成功、
    # 异常两条路径上释放 session-level advisory lock。
    with psycopg.connect(
        database_url,
        autocommit=True,
        connect_timeout=connect_timeout,
        options=f"-c statement_timeout={statement_timeout_ms}",
    ) as connection:
        deadline = time.monotonic() + timeout_seconds
        while True:
            acquired = connection.execute(
                "SELECT pg_try_advisory_lock(%s)",
                (lock_id,),
            ).fetchone()
            if acquired and acquired[0]:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PostgresSessionArchiveLockTimeoutError(
                    f"Timed out after {timeout_seconds:g} seconds waiting for "
                    "learner-memory session/archive lock"
                )
            time.sleep(
                min(_POSTGRES_SESSION_ARCHIVE_LOCK_POLL_SECONDS, remaining)
            )
        operation()


def _archive_session_briefs_with_postgres_lock(
    user_id: str,
    database_url: str,
) -> bool:
    _run_with_postgres_session_archive_lock(
        user_id,
        database_url,
        lambda: _archive_session_briefs_sync(user_id),
    )
    return True


async def maybe_archive_session_briefs(user_id: str) -> None:
    if not DATABASE_URL:
        # 此路径内无 await，同一事件循环上的多个请求不会交错执行归档。
        _archive_session_briefs_sync(user_id)
        return

    try:
        await asyncio.to_thread(
            _archive_session_briefs_with_postgres_lock,
            user_id,
            DATABASE_URL,
        )
    except Exception as exc:
        # 归档是有界压缩，不应让一次锁连接故障破坏已经写入的学习结果；
        # raw brief 会保留，并由后续请求或下次归档继续处理。
        logger.warning("session brief 归档跳过 user=%s: %s", user_id, exc)
