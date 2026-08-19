"""
本机记忆快照持久化（仅 InMemoryStore 兜底）。

问题：services/memory.py 无 DATABASE_URL 时用 InMemoryStore（进程内 dict），重启即丢，
跨会话记忆默认只在同一 server 进程内成立。
生产用 PostgresStore 本身持久，无需快照。

方案：
  - save_snapshot()：遍历 store.list_namespaces() + search(ns) 导出全部 6-bank 到 JSON，
    原子写（写临时文件 + os.replace 原子 rename，避免写一半崩溃腐坏旧快照）。
  - load_snapshot()：启动时回灌 store.put(ns, key, value)。
  - 仅 InMemoryStore 时启用；有 DATABASE_URL（PostgresStore）时 no-op（本身持久）。

路径 env MEMORY_SNAPSHOT_PATH（默认 ./.memory_snapshot.json，已加入 .gitignore）。
"""
import asyncio
import copy
import json
import logging
import os
import tempfile
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_PATH = str(Path(__file__).parent.parent / ".memory_snapshot.json")
_SNAPSHOT_VERSION = 1
_SNAPSHOT_LOCK = threading.Lock()
_PAGE_SIZE = 100


def snapshot_path() -> str:
    return os.getenv("MEMORY_SNAPSHOT_PATH") or _DEFAULT_PATH


def _is_inmemory() -> bool:
    """仅 InMemoryStore 时做快照；有 DATABASE_URL（PostgresStore）时本身持久 → no-op。"""
    return not os.getenv("DATABASE_URL")


def _all_namespaces(store) -> list[tuple]:
    namespaces = []
    offset = 0
    while True:
        batch = store.list_namespaces(limit=_PAGE_SIZE, offset=offset)
        namespaces.extend(batch)
        if len(batch) < _PAGE_SIZE:
            return namespaces
        offset += len(batch)


def _all_items(store, namespace: tuple) -> list:
    items = []
    offset = 0
    while True:
        batch = store.search(
            namespace,
            limit=_PAGE_SIZE,
            offset=offset,
        )
        items.extend(batch)
        if len(batch) < _PAGE_SIZE:
            return items
        offset += len(batch)


def save_snapshot(path: str | None = None) -> bool:
    """导出 InMemoryStore 全量数据到 JSON（原子写）。返回是否实际写入。"""
    if not _is_inmemory():
        return False
    with _SNAPSHOT_LOCK:
        return _save_snapshot_unlocked(path)


def _save_snapshot_unlocked(path: str | None = None) -> bool:
    # 延迟导入避免与 memory.py 的模块级循环依赖。所有 InMemoryStore 访问共用
    # 同一把锁，先在锁内序列化出一致视图，再释放锁执行磁盘 I/O。
    from services.memory import _STORE_LOCK, store

    path = path or snapshot_path()
    try:
        with _STORE_LOCK:
            items: list[dict] = []
            for ns in _all_namespaces(store):
                for it in _all_items(store, ns):
                    items.append(
                        {
                            "ns": list(it.namespace),
                            "key": it.key,
                            "value": copy.deepcopy(it.value),
                        }
                    )
        # 值在写入 Store 时也会 deepcopy；拿到独立副本后即可释放锁，避免
        # JSON 编码大快照时阻塞事件循环里的下一次记忆写入。
        serialized = json.dumps(
            {"version": _SNAPSHOT_VERSION, "items": items},
            ensure_ascii=False,
        )
    except Exception as e:
        logger.warning(f"[memory_persist] dump store failed: {e}")
        return False

    try:
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        # 原子写：同目录临时文件 + os.replace（rename 原子，崩溃时旧快照不被腐坏）
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".memsnap_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(serialized)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        logger.info(f"[memory_persist] saved {len(items)} items → {path}")
        return True
    except Exception as e:
        logger.warning(f"[memory_persist] save_snapshot failed: {e}")
        return False


async def persist_snapshot(path: str | None = None) -> bool:
    """在线程中串行刷新快照，避免阻塞 FastAPI 事件循环。"""
    return await asyncio.to_thread(save_snapshot, path)


def load_snapshot(path: str | None = None) -> int:
    """启动时从 JSON 回灌 InMemoryStore。返回回灌条数（0 = 无快照 / 非 InMemory / 失败）。"""
    if not _is_inmemory():
        return 0
    from services.memory import _STORE_LOCK, store

    path = path or snapshot_path()
    if not os.path.exists(path):
        return 0
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        logger.warning(f"[memory_persist] load_snapshot read failed: {e}")
        return 0

    items = payload.get("items", []) if isinstance(payload, dict) else []
    n = 0
    with _STORE_LOCK:
        for rec in items:
            try:
                store.put(tuple(rec["ns"]), rec["key"], rec["value"])
                n += 1
            except Exception as e:
                logger.warning(f"[memory_persist] put failed for {rec.get('ns')}: {e}")
    logger.info(f"[memory_persist] loaded {n} items ← {path}")
    return n
