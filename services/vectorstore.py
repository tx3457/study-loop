import asyncio
import concurrent.futures
import hashlib
import logging
import math
import os
import queue
import re
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from chromadb.errors import ChromaError, NotFoundError
from services.provider_config import (
    build_managed_async_openai,
    load_provider_configs,
    run_with_provider_deadline,
)
from overrides import override
from services.retry import with_retry
from services.tracing import traceable
from services.reranker import RerankerUnavailable, rerank_docs, reranker_enabled
from services.bm25 import build_bm25_index, rank_bm25
from services.query_rewriter import (
    hyde_enabled,
    hyde_rewrite,
    multi_query_rewrite,
    multiquery_enabled,
    rrf_merge_ranked_lists,
)
from services.tokenization import BM25_TOKENIZER_ID
from services.usage import usage_ledger

logger = logging.getLogger(__name__)
load_dotenv(Path(__file__).parent.parent / ".env")

_embedding_config = load_provider_configs()["embedding"]
embedding_model = _embedding_config.model
# embedding 供应商可与 chat 分离（DeepSeek 无 embedding 接口）：
# EMBEDDING_* 未配置则跟随 LLM_*
# connect 超时放宽：SiliconFlow 高峰期 TLS 建连超过 SDK 默认 5s（见 services/llm.py）
client = build_managed_async_openai(_embedding_config)

# 绝对路径,避免不同启动目录(systemd/docker/测试)各自指向不同的 ./chroma_db
_CHROMA_DIR = os.getenv("CHROMA_DIR") or str(Path(__file__).parent.parent / "chroma_db")
chromadb_client = chromadb.PersistentClient(_CHROMA_DIR)


def _positive_finite_float_environment(name: str, default: float) -> float:
    """Read a non-secret duration without letting a malformed deploy config crash import."""
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = float(raw_value)
    except ValueError:
        value = 0.0
    if not math.isfinite(value) or value <= 0:
        logger.warning("[vectorstore] invalid %s; using safe default", name)
        return default
    return value


def _positive_integer_environment(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = int(raw_value)
    except ValueError:
        value = 0
    if value <= 0:
        logger.warning("[vectorstore] invalid %s; using safe default", name)
        return default
    return value


# The application deliberately serializes its process-local PersistentClient.
# The semaphore bounds both the one active worker and its waiting backlog; it
# is independent from event loops so tests and short-lived CLI loops cannot
# leak a loop-bound gate.
CHROMA_IO_MAX_PENDING = _positive_integer_environment("CHROMA_IO_MAX_PENDING", 16)
CHROMA_IO_QUEUE_WAIT_SECONDS = _positive_finite_float_environment(
    "CHROMA_IO_QUEUE_WAIT_SECONDS", 10.0
)
CHROMA_IO_CANCEL_DRAIN_SECONDS = _positive_finite_float_environment(
    "CHROMA_IO_CANCEL_DRAIN_SECONDS", 2.0
)
CHROMA_IO_OPERATION_TIMEOUT_SECONDS = _positive_finite_float_environment(
    "CHROMA_IO_OPERATION_TIMEOUT_SECONDS", 30.0
)
CHROMA_IO_SHUTDOWN_DRAIN_SECONDS = _positive_finite_float_environment(
    "CHROMA_IO_SHUTDOWN_DRAIN_SECONDS", 5.0
)
_CHROMA_IO_QUEUE_POLL_SECONDS = 0.01
_chroma_io_slots = threading.BoundedSemaphore(CHROMA_IO_MAX_PENDING)
_chroma_io_active = threading.Lock()
_chroma_io_jobs: queue.Queue = queue.Queue(maxsize=1)
_chroma_io_worker_lock = threading.Lock()
_chroma_io_worker: threading.Thread | None = None
_chroma_background_jobs: set[concurrent.futures.Future] = set()
_chroma_background_jobs_lock = threading.Lock()
_chroma_io_lifecycle_lock = threading.Lock()
_chroma_io_accepting = True
_chroma_io_generation = 0


class ChromaIOWaitTimeoutError(ChromaError):
    """The bounded embedded-Chroma queue did not accept work in time."""

    @classmethod
    @override
    def name(cls) -> str:
        return "ChromaIOWaitTimeout"


class ChromaIOOperationTimeoutError(ChromaError):
    """A Chroma call exceeded its public waiting budget but keeps draining."""

    @classmethod
    @override
    def name(cls) -> str:
        return "ChromaIOOperationTimeout"


class ChromaIOShuttingDownError(ChromaError):
    """The process is draining the embedded Chroma worker."""

    @classmethod
    @override
    def name(cls) -> str:
        return "ChromaIOShuttingDown"


def _chroma_io_is_accepting() -> bool:
    with _chroma_io_lifecycle_lock:
        return _chroma_io_accepting


def _capture_chroma_io_generation() -> int:
    """Bind a request to one lifecycle generation before it starts waiting."""
    with _chroma_io_lifecycle_lock:
        if not _chroma_io_accepting:
            raise ChromaIOShuttingDownError("embedded Chroma worker is shutting down")
        return _chroma_io_generation


def start_vectorstore_io() -> None:
    """Open a clean lifecycle; never reopen while a timed-out worker survives."""
    global _chroma_io_accepting, _chroma_io_generation, _chroma_io_worker
    with _chroma_io_lifecycle_lock, _chroma_io_worker_lock:
        if _chroma_io_accepting:
            return
        if _chroma_io_worker is not None and _chroma_io_worker.is_alive():
            raise ChromaIOShuttingDownError("embedded Chroma worker is still draining")
        _chroma_io_worker = None
        _chroma_io_generation += 1
        _chroma_io_accepting = True


async def _acquire_chroma_io_slot(deadline: float) -> None:
    """Wait for bounded queue capacity without tying it to one event loop."""
    loop = asyncio.get_running_loop()
    while not _chroma_io_slots.acquire(blocking=False):
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise ChromaIOWaitTimeoutError("embedded Chroma I/O queue is full")
        await asyncio.sleep(min(_CHROMA_IO_QUEUE_POLL_SECONDS, remaining))


def _consume_chroma_worker_result(
    worker: concurrent.futures.Future,
    operation_name: str,
    detached: threading.Event,
) -> None:
    """Consume late failures so cancellation hand-off cannot create warnings."""
    try:
        worker.result()
    except asyncio.CancelledError:
        if detached.is_set():
            logger.warning(
                "[vectorstore] detached Chroma worker cancelled: operation=%s",
                operation_name,
            )
    except BaseException as exc:
        if detached.is_set():
            logger.error(
                "[vectorstore] detached Chroma worker failed: "
                "operation=%s error_type=%s",
                operation_name,
                type(exc).__name__,
            )


def _track_chroma_worker(
    worker: concurrent.futures.Future,
    operation_name: str,
    detached: threading.Event,
) -> None:
    """Release capacity and retain a hand-off until its synchronous I/O exits."""
    with _chroma_background_jobs_lock:
        _chroma_background_jobs.add(worker)

    def on_done(completed: concurrent.futures.Future) -> None:
        with _chroma_background_jobs_lock:
            _chroma_background_jobs.discard(completed)
        _chroma_io_slots.release()
        _consume_chroma_worker_result(completed, operation_name, detached)

    worker.add_done_callback(on_done)


async def _drain_cancelled_chroma_io(worker: concurrent.futures.Future) -> bool:
    """Drain repeated cancellation only until one absolute finite deadline."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + CHROMA_IO_CANCEL_DRAIN_SECONDS
    while not worker.done():
        remaining = deadline - loop.time()
        if remaining <= 0:
            return False
        try:
            await asyncio.sleep(min(_CHROMA_IO_QUEUE_POLL_SECONDS, remaining))
        except asyncio.CancelledError:
            continue
    return True


def _execute_chroma_job(job: tuple) -> None:
    """Execute one job in a short-lived frame so its client refs are released."""
    func, args, kwargs, worker = job
    if not worker.set_running_or_notify_cancel():
        _chroma_io_active.release()
        return
    try:
        result = func(*args, **kwargs)
    except BaseException as exc:
        _chroma_io_active.release()
        worker.set_exception(exc)
    else:
        _chroma_io_active.release()
        worker.set_result(result)


def _chroma_worker_main() -> None:
    """Run exactly one daemon worker, never tied to a request event loop."""
    while True:
        job = _chroma_io_jobs.get()
        if job is None:
            return
        _execute_chroma_job(job)
        del job


def _ensure_chroma_worker() -> None:
    global _chroma_io_worker
    with _chroma_io_worker_lock:
        if _chroma_io_worker is not None and _chroma_io_worker.is_alive():
            return
        _chroma_io_worker = threading.Thread(
            target=_chroma_worker_main,
            name="studyloop-chroma",
            daemon=True,
        )
        _chroma_io_worker.start()


def _submit_chroma_job(
    worker: concurrent.futures.Future,
    func,
    args: tuple,
    kwargs: dict,
    *,
    generation: int,
    operation_name: str,
    detached: threading.Event,
) -> None:
    """Atomically reject a waiter from an earlier stopped lifecycle."""
    with _chroma_io_lifecycle_lock:
        if not _chroma_io_accepting or generation != _chroma_io_generation:
            raise ChromaIOShuttingDownError("embedded Chroma worker is shutting down")
        _ensure_chroma_worker()
        _track_chroma_worker(worker, operation_name, detached)
        _chroma_io_jobs.put_nowait((func, args, kwargs, worker))


async def shutdown_vectorstore_io() -> None:
    """Reject new work and give owned Chroma transactions a finite shutdown drain."""
    global _chroma_io_accepting, _chroma_io_worker
    with _chroma_io_lifecycle_lock:
        _chroma_io_accepting = False
        worker = _chroma_io_worker

    loop = asyncio.get_running_loop()
    deadline = loop.time() + CHROMA_IO_SHUTDOWN_DRAIN_SECONDS
    while True:
        with _chroma_background_jobs_lock:
            pending = bool(_chroma_background_jobs)
        if not pending and not _chroma_io_active.locked():
            break
        remaining = deadline - loop.time()
        if remaining <= 0:
            logger.error(
                "[vectorstore] Chroma shutdown drain timed out: error_type=TimeoutError"
            )
            return
        await asyncio.sleep(min(_CHROMA_IO_QUEUE_POLL_SECONDS, remaining))

    # A request may have passed the final accepting check immediately before
    # shutdown flipped it. Re-read after the drain rather than trusting the
    # initial snapshot, then stop whichever worker actually owns the channel.
    with _chroma_io_worker_lock:
        worker = _chroma_io_worker
    if worker is not None and worker.is_alive():
        _chroma_io_jobs.put_nowait(None)
        while worker.is_alive():
            remaining = deadline - loop.time()
            if remaining <= 0:
                logger.error(
                    "[vectorstore] Chroma worker shutdown timed out: error_type=TimeoutError"
                )
                return
            await asyncio.sleep(min(_CHROMA_IO_QUEUE_POLL_SECONDS, remaining))

    with _chroma_io_lifecycle_lock:
        if _chroma_io_worker is worker:
            _chroma_io_worker = None


def _run_chroma_io_sync(func, /, *args, operation_name: str, **kwargs):
    """Use the daemon channel from blocking probes as well as async APIs."""
    generation = _capture_chroma_io_generation()
    deadline = time.monotonic() + CHROMA_IO_QUEUE_WAIT_SECONDS
    slot_acquired = False
    active_acquired = False
    submitted = False
    detached = threading.Event()
    try:
        while not _chroma_io_slots.acquire(blocking=False):
            if time.monotonic() >= deadline:
                raise ChromaIOWaitTimeoutError("embedded Chroma I/O queue is full")
            time.sleep(_CHROMA_IO_QUEUE_POLL_SECONDS)
        slot_acquired = True
        while not _chroma_io_active.acquire(blocking=False):
            if time.monotonic() >= deadline:
                raise ChromaIOWaitTimeoutError("embedded Chroma worker is busy")
            time.sleep(_CHROMA_IO_QUEUE_POLL_SECONDS)
        active_acquired = True
        worker: concurrent.futures.Future = concurrent.futures.Future()
        _submit_chroma_job(
            worker,
            func,
            args,
            kwargs,
            generation=generation,
            operation_name=operation_name,
            detached=detached,
        )
        submitted = True
        try:
            return worker.result(timeout=CHROMA_IO_OPERATION_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            detached.set()
            raise ChromaIOOperationTimeoutError(
                "embedded Chroma operation timed out"
            ) from None
    finally:
        if not submitted:
            if active_acquired:
                _chroma_io_active.release()
            if slot_acquired:
                _chroma_io_slots.release()


async def _acquire_chroma_io_active(deadline: float) -> None:
    loop = asyncio.get_running_loop()
    if not _chroma_io_is_accepting():
        raise ChromaIOShuttingDownError("embedded Chroma worker is shutting down")
    while not _chroma_io_active.acquire(blocking=False):
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise ChromaIOWaitTimeoutError("embedded Chroma worker is busy")
        await asyncio.sleep(min(_CHROMA_IO_QUEUE_POLL_SECONDS, remaining))


async def _wait_for_chroma_worker(
    worker: concurrent.futures.Future,
    deadline: float,
) -> object:
    loop = asyncio.get_running_loop()
    while not worker.done():
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise ChromaIOOperationTimeoutError("embedded Chroma operation timed out")
        await asyncio.sleep(min(_CHROMA_IO_QUEUE_POLL_SECONDS, remaining))
    return worker.result()


async def _run_chroma_io(
    func,
    /,
    *args,
    operation_name: str,
    cancellation_event: threading.Event | None = None,
    **kwargs,
):
    """Serialize Chroma calls, with bounded pressure and cancellation hand-off.

    Once submitted, a mutating operation must continue: staging publication and
    tombstone cleanup otherwise could be left in an ambiguous state.  A caller
    waits only a bounded additional interval after cancellation; the worker
    remains strongly referenced and releases the slot only when it really exits.
    """
    loop = asyncio.get_running_loop()
    generation = _capture_chroma_io_generation()
    queue_deadline = loop.time() + CHROMA_IO_QUEUE_WAIT_SECONDS
    submitted = False
    slot_acquired = False
    active_acquired = False
    detached = threading.Event()
    try:
        await _acquire_chroma_io_slot(queue_deadline)
        slot_acquired = True
        await _acquire_chroma_io_active(queue_deadline)
        active_acquired = True
        worker: concurrent.futures.Future = concurrent.futures.Future()
        _submit_chroma_job(
            worker,
            func,
            args,
            kwargs,
            generation=generation,
            operation_name=operation_name,
            detached=detached,
        )
        submitted = True
        operation_deadline = loop.time() + CHROMA_IO_OPERATION_TIMEOUT_SECONDS
        return await _wait_for_chroma_worker(worker, operation_deadline)
    except asyncio.CancelledError as cancellation:
        if not submitted:
            raise cancellation
        if worker.cancel():
            raise cancellation
        if cancellation_event is not None:
            cancellation_event.set()
        completed = await _drain_cancelled_chroma_io(worker)
        if not completed:
            detached.set()
            logger.warning(
                "[vectorstore] cancelled request stopped waiting for Chroma I/O; "
                "operation=%s",
                operation_name,
            )
        raise cancellation
    except ChromaIOOperationTimeoutError:
        if worker.cancel():
            raise
        if cancellation_event is not None:
            cancellation_event.set()
        detached.set()
        raise
    finally:
        if not submitted:
            # No worker owns the capacity/active lock when a queue waiter
            # times out or is cancelled, so it must not be left for a callback.
            if active_acquired:
                _chroma_io_active.release()
            if slot_acquired:
                _chroma_io_slots.release()


def probe_vectorstore_readiness() -> None:
    """Read Chroma's persistent collection catalog without mutating it.

    Chroma's ``heartbeat()`` only returns the current time, so it cannot prove
    that the embedded metadata database remains readable.
    """
    _run_chroma_io_sync(
        chromadb_client.count_collections,
        operation_name="probe_vectorstore_readiness",
    )

# 单次 embedding 请求最多 chunk 数:大文档分批,避免撞厂商单请求 input 上限
EMBED_BATCH_SIZE = 64
_STAGING_PREFIX = "studyloop-staging-"
_STAGING_TTL_SECONDS = max(60, int(os.getenv("STAGING_COLLECTION_TTL_SECONDS", "3600")))
_active_staging_names: set[str] = set()
_VALID_COLLECTION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,510}[A-Za-z0-9]$")


class DocumentAlreadyExistsError(Exception):
    """A published collection already owns this document id."""


class DocumentOwnerMismatchError(NotFoundError):
    """A storage collection exists, but not for the requested public id."""


# 建库早于属主概念的 collection 没有 owner_id metadata，归属这个命名空间。
# 单用户部署里所有请求也解析到它，所以历史数据不会因为引入属主而失联。
DEFAULT_DOCUMENT_OWNER = "default_user"


def _storage_document_id(document_id: str, owner_id: str) -> str:
    """Map an (owner, public filename) pair to a Chroma-safe collection name.

    属主是存储身份的一部分，而不是取到 collection 之后再比对的一个字段：
    换个 owner_id，连 collection 的名字都算不出来，所以「忘记校验属主」这种
    错误在这里不成立。

    默认属主沿用原来的命名（ASCII 原样、其余走摘要），既有库不需要迁移；
    其他属主一律走 owner 参与的摘要，不同属主的同名文件天然不同名。
    """
    if owner_id == DEFAULT_DOCUMENT_OWNER:
        if _VALID_COLLECTION_NAME.fullmatch(document_id):
            return document_id
        digest = hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:56]
        return f"doc-{digest}"
    # \x00 作分隔符：文件名里不可能出现，("a", "bc") 和 ("ab", "c") 不会撞。
    identity = f"{owner_id}\x00{document_id}".encode("utf-8")
    return f"u-{hashlib.sha256(identity).hexdigest()[:56]}"


def _collection_owner(metadata: dict) -> str:
    """Owner of a stored collection; pre-scoping collections are default-owned."""
    owner = metadata.get("owner_id")
    return owner if isinstance(owner, str) and owner else DEFAULT_DOCUMENT_OWNER


def _require_public_document_owner(collection, document_id: str, owner_id: str):
    """Reject internal collection-name aliases and other owners' documents."""
    metadata = collection.metadata if isinstance(collection.metadata, dict) else {}
    public_id = (
        metadata.get("source_document_id")
        or metadata.get("source_filename")
        or collection.name
    )
    # 属主不符按「不存在」报，不按「禁止访问」报：后者会告诉调用方这个
    # 文档确实存在、只是不属于他，等于一个存在性探测接口。
    if public_id != document_id or _collection_owner(metadata) != owner_id:
        raise DocumentOwnerMismatchError(f"Document {document_id!r} not found")
    return collection


async def _get_public_document_collection(
    document_id: str,
    owner_id: str,
    *,
    include_tombstone: bool = False,
):
    collection = await _run_chroma_io(
        chromadb_client.get_collection,
        name=_storage_document_id(document_id, owner_id),
        operation_name="get_collection",
    )
    collection = _require_public_document_owner(collection, document_id, owner_id)
    metadata = collection.metadata if isinstance(collection.metadata, dict) else {}
    if (
        not include_tombstone
        and metadata.get("ingest_status") in {"deleting", "deleted"}
    ):
        raise NotFoundError(f"Document {document_id!r} not found")
    return collection


async def ensure_document_available(document_id: str, *, owner_id: str) -> None:
    """Resolve a public document id and fail if its collection is unavailable."""
    await _get_public_document_collection(document_id, owner_id)


async def _embed(texts: list[str]):
    """embedding 统一入口：接 with_retry（厂商偶发连接抖动/超时 → 指数退避重试）。

    所有 embedding 调用通过同一重试与超时策略执行。
    """

    async def embed_attempt():
        response = await with_retry(
            lambda: client.embeddings.create(model=embedding_model, input=texts),
            max_retries=3,
            base_delay=1.0,
            timeout=30,
        )
        usage_ledger.record("embedding", response)
        return response

    return await run_with_provider_deadline(embed_attempt)


def _staging_is_stale(metadata: dict, now: float | None = None) -> bool:
    created_at = metadata.get("created_at")
    if not isinstance(created_at, (int, float)):
        return True
    return (now or time.time()) - created_at >= _STAGING_TTL_SECONDS


def _ensure_document_slot_available_sync(document_id: str, owner_id: str) -> None:
    """Reject real duplicates while migrating legacy empty ghost collections."""
    storage_id = _storage_document_id(document_id, owner_id)
    try:
        existing = chromadb_client.get_collection(name=storage_id)
    except NotFoundError:
        return

    metadata = existing.metadata if isinstance(existing.metadata, dict) else {}
    status = metadata.get("ingest_status")
    if status in {"deleting", "deleted"}:
        raise DocumentAlreadyExistsError(
            f"文档 '{document_id}' 已关联保留的学习历史，不能同名重传，请先重命名文件"
        )
    count = existing.count()
    removable = (status == "indexing" and _staging_is_stale(metadata)) or (
        status != "indexed" and count == 0
    )
    if removable:
        try:
            chromadb_client.delete_collection(name=storage_id)
        except NotFoundError:
            pass
        _invalidate_bm25_cache(storage_id)
        logger.info(
            "[vectorstore] removed incomplete collection before upload: %s", document_id
        )
        return

    raise DocumentAlreadyExistsError(f"文档 '{document_id}' 已存在，请先删除后再上传")


async def _ensure_document_slot_available(document_id: str, owner_id: str) -> None:
    await _run_chroma_io(
        _ensure_document_slot_available_sync,
        document_id,
        owner_id,
        operation_name="ensure_document_slot_available",
    )


def _deal_document_sync(
    document_id: str,
    filename: str,
    chunks: list[str],
    embeddings: list,
    cancellation_event: threading.Event,
    owner_id: str,
) -> int:
    """Perform one complete staging→published transition on the Chroma worker."""
    storage_id = _storage_document_id(document_id, owner_id)
    # Re-check inside the same serialized worker immediately before mutation.
    # The async preflight only avoids wasting embedding calls on obvious dupes.
    _ensure_document_slot_available_sync(document_id, owner_id)
    staging_name = f"{_STAGING_PREFIX}{uuid.uuid4().hex}"
    collection = None
    try:
        # This synchronous transaction is never abandoned after submission.
        # Its finally cleanup therefore still runs if the HTTP caller leaves.
        collection = chromadb_client.create_collection(
            name=staging_name,
            metadata={
                "ingest_status": "indexing",
                "source_filename": filename,
                "source_document_id": document_id,
                "owner_id": owner_id,
                "created_at": int(time.time()),
            },
        )
        _active_staging_names.add(staging_name)
        if cancellation_event.is_set():
            raise asyncio.CancelledError
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=[f"{storage_id}_chunk_{i}" for i in range(len(chunks))],
            metadatas=[
                {"source": chunks[i][:50], "doc_id": document_id, "chunk_index": i}
                for i in range(len(chunks))
            ],
        )
        if cancellation_event.is_set():
            raise asyncio.CancelledError

        try:
            collection.modify(
                name=storage_id,
                metadata={
                    "ingest_status": "indexed",
                    "source_filename": filename,
                    "source_document_id": document_id,
                    "owner_id": owner_id,
                },
            )
            # The canonical name may now resolve to new text even if a later
            # client cancellation prevents the outer coroutine from returning.
            _invalidate_bm25_cache(storage_id)
            if cancellation_event.is_set():
                raise asyncio.CancelledError
        except Exception as publish_error:
            # Chroma 的 rename 冲突没有稳定的专用异常类型；以正式 collection
            # 是否已出现判定并发同名上传，统一返回 409 而不是误报存储故障。
            try:
                chromadb_client.get_collection(name=storage_id)
                published_exists = True
            except NotFoundError:
                published_exists = False
            if published_exists:
                _invalidate_bm25_cache(storage_id)
                raise DocumentAlreadyExistsError(
                    f"文档 '{document_id}' 已存在，请先删除后再上传"
                ) from publish_error
            raise
    except BaseException:
        if collection is not None:
            try:
                _cleanup_staging_collection_sync(staging_name, document_id)
            except Exception as cleanup_error:
                logger.error(
                    "[vectorstore] staging cleanup failure: error_type=%s",
                    type(cleanup_error).__name__,
                )
        raise
    finally:
        _active_staging_names.discard(staging_name)

    return len(chunks)


def _cleanup_staging_collection_sync(staging_name: str, document_id: str) -> None:
    """Delete only this transaction's unguessable, immutable staging name."""
    # ``staging_name`` is generated before create and never reused; unlike
    # collection.name it cannot turn into the public collection after rename.
    # A catalog re-read would make rollback depend on a second failed I/O and
    # provides no stronger ownership proof than this capability-like UUID.
    del document_id
    try:
        chromadb_client.delete_collection(name=staging_name)
    except NotFoundError:
        # A successful rename/ACK-loss path no longer owns this staging name.
        return


async def deal_document(
    document_id: str, filename: str, chunks: list[str], *, owner_id: str
):
    # 同名重传不能继续用 add：Chroma 会忽略重复 id，导致接口成功但正文仍未更新。
    await _ensure_document_slot_available(document_id, owner_id)

    # 先完成所有外部 embedding 调用，provider 失败时不产生任何 Chroma 写入。
    # 分批 embed:几百 chunk 一次性 embed 会撞 API input 上限(多数厂商 ~2048 条/8k token)
    embeddings: list = []
    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        resp = await _embed(chunks[start : start + EMBED_BATCH_SIZE])
        embeddings.extend(item.embedding for item in resp.data)

    # Staging 的创建、写入、发布和失败清理必须由同一个不可中断 worker 收束。
    cancellation_event = threading.Event()
    return await _run_chroma_io(
        _deal_document_sync,
        document_id,
        filename,
        chunks,
        embeddings,
        cancellation_event,
        owner_id,
        operation_name="deal_document",
        cancellation_event=cancellation_event,
    )


# ── BM25 索引缓存(按 document_id),避免每次查询全量 collection.get + 重建索引 ──────
# 每条缓存持有该文档的全部 chunk 正文和 BM25 索引，且只在文档更新/删除时失效。
# 没有上限时，进程内存随「查询过多少个不同文档」单调增长，永远不回落。
#
# 限额按缓存正文的总字符数，不按条目数：一个大文档抵得上几百个小文档，
# 按条目数封不住内存。单条上界由上传大小限制决定。
_BM25_CACHE_MAX_CHARS = max(100_000, int(os.getenv("BM25_CACHE_MAX_CHARS", "5000000")))
_bm25_cache: "OrderedDict[str, dict]" = OrderedDict()
_bm25_cached_chars = 0
_bm25_cache_lock = threading.Lock()
_bm25_cache_epoch = 0
_bm25_document_versions: dict[str, int] = {}


# 下面三个 helper 都不自带加锁，调用方必须已持有 _bm25_cache_lock。
# 参数是 _storage_document_id 的结果，不是公开 document_id——传错不会报错，
# 只会让缓存失效不掉，继续吐已经删掉的正文，所以参数名写成 cache_key。
def _bm25_cache_take(cache_key: str) -> dict | None:
    """读一条并标记为最近使用（LRU）。"""
    entry = _bm25_cache.get(cache_key)
    if entry is not None:
        _bm25_cache.move_to_end(cache_key)
    return entry


def _bm25_cache_drop(cache_key: str) -> None:
    """删一条并扣减字符计数。"""
    global _bm25_cached_chars
    entry = _bm25_cache.pop(cache_key, None)
    if entry is not None:
        # 不用 .get(..., 0) 兜底：计数一旦漂移，限额会静默失效且不报任何错。
        _bm25_cached_chars -= entry["cached_chars"]


def _bm25_cache_store(cache_key: str, entry: dict) -> None:
    """写入一条，必要时按 LRU 淘汰到预算以内。"""
    global _bm25_cached_chars
    _bm25_cache_drop(cache_key)            # 同键覆盖也要先扣旧值
    _bm25_cache[cache_key] = entry
    _bm25_cached_chars += entry["cached_chars"]
    # 留住至少一条：单个文档超过整个预算时，反复重建索引比多占一份内存更糟。
    # 内存上界因此是「预算 + 一个文档」，仍然有界。
    while len(_bm25_cache) > 1 and _bm25_cached_chars > _BM25_CACHE_MAX_CHARS:
        evicted_id, evicted = _bm25_cache.popitem(last=False)
        _bm25_cached_chars -= evicted["cached_chars"]
        logger.debug("[bm25_cache] evicted %s to stay within budget", evicted_id)


def _invalidate_bm25_cache(cache_key: str) -> None:
    """Fence in-flight builders, then remove every cached copy for a document.

    键是 _storage_document_id 的结果而不是公开 document_id：属主参与存储身份后，
    两个属主可以各有一份同名文档，按公开名做键会让他们共用同一条缓存，
    等于把一方的正文喂给另一方。
    """
    with _bm25_cache_lock:
        _bm25_document_versions[cache_key] = (
            _bm25_document_versions.get(cache_key, 0) + 1
        )
        _bm25_cache_drop(cache_key)


def _get_or_build_bm25_index_sync(
    collection,
    document_id: str,
    owner_id: str,
    cache_epoch: int,
    document_version: int,
    cancellation_event: threading.Event,
) -> dict:
    """Single-flight corpus read/build/publish on the bounded storage worker."""
    cache_key = _storage_document_id(document_id, owner_id)
    with _bm25_cache_lock:
        cached = _bm25_cache_take(cache_key)
        if cached is not None:
            return cached
    all_results = collection.get(include=["documents"])
    all_docs = all_results["documents"]
    all_ids = all_results["ids"]
    bm25 = build_bm25_index(all_docs)
    if cancellation_event.is_set():
        raise asyncio.CancelledError
    entry = {
        "bm25": bm25,
        "all_docs": all_docs,
        "all_ids": all_ids,
        "tokenizer_id": BM25_TOKENIZER_ID,
        "cached_chars": sum(len(doc) for doc in all_docs),
    }
    current = chromadb_client.get_collection(name=cache_key)
    _require_public_document_owner(current, document_id, owner_id)
    metadata = current.metadata if isinstance(current.metadata, dict) else {}
    if metadata.get("ingest_status") in {"deleting", "deleted"}:
        raise NotFoundError(f"Document {document_id!r} not found")
    with _bm25_cache_lock:
        if (
            cancellation_event.is_set()
            or cache_epoch != _bm25_cache_epoch
            or document_version != _bm25_document_versions.get(cache_key, 0)
        ):
            raise NotFoundError(f"Document {document_id!r} not found")
        _bm25_cache_store(cache_key, entry)
    return entry


async def _get_bm25_index(collection, document_id: str, owner_id: str) -> dict:
    """取或构建某文档的 BM25 索引,带进程内缓存。返回 {bm25, all_docs, all_ids}。

    缓存键含属主（见 _invalidate_bm25_cache），在 deal_document / delete_document
    时失效。分词由 services.tokenization 统一提供，确保索引和查询使用同一规则。
    """
    cache_key = _storage_document_id(document_id, owner_id)
    with _bm25_cache_lock:
        cached = _bm25_cache_take(cache_key)
        cache_epoch = _bm25_cache_epoch
        document_version = _bm25_document_versions.get(cache_key, 0)
        if cached is not None:
            return cached
    cancellation_event = threading.Event()
    return await _run_chroma_io(
        _get_or_build_bm25_index_sync,
        collection,
        document_id,
        owner_id,
        cache_epoch,
        document_version,
        cancellation_event,
        operation_name="get_or_build_bm25_index",
        cancellation_event=cancellation_event,
    )


def clear_bm25_cache() -> None:
    """清空 BM25 缓存（测试用 / 手动失效）。"""
    global _bm25_cache_epoch, _bm25_cached_chars
    with _bm25_cache_lock:
        _bm25_cache_epoch += 1
        _bm25_cache.clear()
        _bm25_cached_chars = 0


async def query_document(document_id: str, query: str, *, owner_id: str):
    collection = await _get_public_document_collection(document_id, owner_id)
    query_vec = await _embed([query])
    return await _run_chroma_io(
        collection.query,
        query_embeddings=[query_vec.data[0].embedding],
        n_results=5,
        operation_name="query_document",
    )


@traceable(
    name="hybrid_retrieval",
    run_type="retriever",
    metadata={"strategy": "bm25+vector+rrf+rerank", "k": 60},
)
async def hybrid_query_document(
    document_id: str,
    query: str,
    n_results: int = 5,
    enable_rerank: bool | None = None,
    *,
    owner_id: str,
) -> dict:
    """
    Hybrid 检索：向量 + BM25 → RRF 融合 → Cross-Encoder 精排。

    两阶段检索：
      召回阶段:vec + BM25,RRF 取 top n_results * 3(给精排足够候选)
      精排阶段:CrossEncoder 真正读 (query, doc) 对打分,取 top n_results

    enable_rerank:
      None  → 用 RERANKER_ENABLED 环境变量(默认 false)
      True  → 强制开启(评测对照组用)
      False → 强制关闭(评测对照组用)
    """
    use_rerank = reranker_enabled() if enable_rerank is None else enable_rerank
    # 精排阶段需要更多候选,召回阶段取 3x;不精排时保持原有 2x 行为以最小化变化
    recall_multiplier = 3 if use_rerank else 2
    recall_n = n_results * recall_multiplier

    # 1. 向量检索
    collection = await _get_public_document_collection(document_id, owner_id)
    query_vec = await _embed([query])
    vec_results = await _run_chroma_io(
        collection.query,
        query_embeddings=[query_vec.data[0].embedding],
        n_results=recall_n,
        operation_name="hybrid_vector_query",
    )
    vec_docs = vec_results["documents"][0]  # list[str]
    vec_ids = vec_results["ids"][0]  # list[str]

    # 2. BM25 检索:索引按(属主, document_id)缓存(文档增删时失效),避免每次全量重建
    idx = await _get_bm25_index(collection, document_id, owner_id)
    all_docs, all_ids, bm25 = idx["all_docs"], idx["all_ids"], idx["bm25"]
    if bm25 is None:
        bm25_ids, bm25_docs = [], []
    else:
        bm25_ranked = await _run_chroma_io(
            rank_bm25,
            bm25,
            query,
            recall_n,
            operation_name="rank_bm25",
        )
        bm25_ids = [all_ids[i] for i, _ in bm25_ranked]
        bm25_docs = [all_docs[i] for i, _ in bm25_ranked]

    # 3. RRF 融合(召回融合,k=60)
    K = 60
    rrf_scores: dict[str, float] = {}
    id_to_doc: dict[str, str] = {}

    for rank, doc_id in enumerate(vec_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (K + rank + 1)
        id_to_doc[doc_id] = vec_docs[rank]

    for rank, doc_id in enumerate(bm25_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (K + rank + 1)
        id_to_doc[doc_id] = bm25_docs[rank]

    rrf_top_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:recall_n]
    rrf_top_docs = [id_to_doc[doc_id] for doc_id in rrf_top_ids]

    # 4. Cross-Encoder 精排(可降级)
    if use_rerank and len(rrf_top_docs) > 1:
        try:
            reranked = await rerank_docs(
                query, rrf_top_docs, rrf_top_ids, top_k=n_results
            )
            top_docs = [r[0] for r in reranked]
            top_ids = [r[2] for r in reranked]
            return {"documents": [top_docs], "ids": [top_ids]}
        except RerankerUnavailable as e:
            logger.warning(
                "[hybrid] reranker unavailable; fallback to RRF-only: error_type=%s",
                type(e).__name__,
            )
            # 降级:用 RRF top n_results

    # 5. 不精排 / 精排失败 → 用 RRF top n_results
    top_ids = rrf_top_ids[:n_results]
    top_docs = [id_to_doc[doc_id] for doc_id in top_ids]
    return {"documents": [top_docs], "ids": [top_ids]}


@traceable(
    name="retrieve_with_rewrite",
    run_type="retriever",
    metadata={"strategy": "hyde+multiquery+hybrid+rrf"},
)
async def retrieve_with_rewrite(
    document_id: str, query: str, n_results: int = 5, *, owner_id: str
) -> dict:
    """生产检索入口：按 env 决定是否做 HyDE / Multi-query 改写，再 hybrid 检索 + RRF 合并。

    组合矩阵：
      Multi-query 关 + HyDE 关 → 直接 hybrid(query)
      Multi-query 开 + HyDE 关 → N 个变体各 hybrid → RRF 合并
      Multi-query 关 + HyDE 开 → hybrid(HyDE 改写后的假设答案)
      Multi-query 开 + HyDE 开 → N 变体 → 各 HyDE 改写 → 各 hybrid → RRF 合并

    env 开关见 services/query_rewriter.py：
      QUERY_REWRITE_ENABLED（总开关）/ HYDE_ENABLED / MULTIQUERY_ENABLED / MULTIQUERY_N
      全关时直接调用 hybrid_query_document。

    健壮性处理：
      ① query 去重——HyDE/multi-query 改写失败会 fallback 回原 query，可能产生重复，去重避免重复检索与 RRF 偏置
      ② 全失败兜底——所有改写后的子查询都检索失败时，退回单 query hybrid，绝不返回空 chunks 饿死出题
    """
    queries: list[str] = [query]
    if multiquery_enabled():
        queries = await multi_query_rewrite(query)
    if hyde_enabled():
        queries = list(await asyncio.gather(*[hyde_rewrite(q) for q in queries]))

    # ① 去重（保序）
    seen: set[str] = set()
    deduped: list[str] = []
    for q in queries:
        if q and q not in seen:
            seen.add(q)
            deduped.append(q)
    queries = deduped or [query]

    # 单 query：直接走 hybrid，省去无意义的 RRF 合并
    if len(queries) == 1:
        return await hybrid_query_document(
            document_id, queries[0], n_results=n_results, owner_id=owner_id
        )

    # 多 query：各自 hybrid 检索 → RRF 合并
    results = await asyncio.gather(
        *[
            hybrid_query_document(
                document_id, q, n_results=n_results, owner_id=owner_id
            )
            for q in queries
        ],
        return_exceptions=True,
    )
    ranked_lists: list[list[tuple[str, str]]] = []
    for r in results:
        if isinstance(r, Exception):
            logger.warning(
                "[retrieve_with_rewrite] one sub-query failed; skipped: error_type=%s",
                type(r).__name__,
            )
            continue
        docs = r.get("documents", [[]])[0] or []
        ids = r.get("ids", [[]])[0] or []
        ranked_lists.append(
            [(ids[i], docs[i]) for i in range(min(len(docs), len(ids)))]
        )

    # ② 全失败兜底：退回单 query hybrid
    if not ranked_lists:
        logger.warning(
            "[retrieve_with_rewrite] all rewritten sub-queries failed, fallback to original query"
        )
        return await hybrid_query_document(
            document_id, query, n_results=n_results, owner_id=owner_id
        )

    merged = rrf_merge_ranked_lists(ranked_lists, top_k=n_results)
    return {
        "documents": [[m[1] for m in merged]],
        "ids": [[m[0] for m in merged]],
    }


async def bm25_only_query_document(
    document_id: str, query: str, n_results: int = 5, *, owner_id: str
) -> dict:
    """纯 BM25 检索（用于评测对照组，与 hybrid 和 naive 三组对比）"""
    collection = await _get_public_document_collection(document_id, owner_id)
    idx = await _get_bm25_index(collection, document_id, owner_id)
    all_docs, all_ids, bm25 = idx["all_docs"], idx["all_ids"], idx["bm25"]
    if bm25 is None:
        return {"documents": [[]], "ids": [[]]}
    ranked = await _run_chroma_io(
        rank_bm25,
        bm25,
        query,
        n_results,
        operation_name="rank_bm25",
    )
    top_ids = [all_ids[i] for i, _ in ranked]
    top_docs = [all_docs[i] for i, _ in ranked]
    return {"documents": [top_docs], "ids": [top_ids]}


def _get_all_document_sync(owner_id: str):
    collections = chromadb_client.list_collections()
    visible = []
    now = time.time()
    for collection in collections:
        metadata = collection.metadata if isinstance(collection.metadata, dict) else {}
        status = metadata.get("ingest_status")
        if status == "deleting":
            public_id = (
                metadata.get("source_document_id")
                or metadata.get("source_filename")
                or collection.name
            )
            try:
                # 墓碑清理是全局维护，按该 collection 自己的属主解析存储 id，
                # 不能用调用方的属主——否则别人的残留永远清不掉。
                _delete_document_sync(public_id, _collection_owner(metadata))
            except Exception as cleanup_error:
                logger.warning(
                    "[vectorstore] deleting tombstone cleanup deferred for %s: %s",
                    public_id,
                    type(cleanup_error).__name__,
                )
            continue
        if status == "deleted":
            continue
        if status == "indexing":
            if collection.name not in _active_staging_names and _staging_is_stale(
                metadata, now
            ):
                try:
                    chromadb_client.delete_collection(name=collection.name)
                except NotFoundError:
                    pass
                except Exception as cleanup_error:
                    logger.error(
                        "[vectorstore] stale staging cleanup failure: error_type=%s",
                        type(cleanup_error).__name__,
                    )
            continue

        # 没有 metadata 且 count=0 的 embedding ghost：隐藏并迁移清理。
        if status != "indexed" and collection.count() == 0:
            try:
                chromadb_client.delete_collection(name=collection.name)
            except NotFoundError:
                pass
            except Exception as cleanup_error:
                logger.error(
                    "[vectorstore] legacy ghost cleanup failure: error_type=%s",
                    type(cleanup_error).__name__,
                )
            continue
        # 上面的清理对所有属主一视同仁（否则别人的陈旧 staging 永远不回收），
        # 但能被看见的只有自己的文档。
        if _collection_owner(metadata) != owner_id:
            continue
        visible.append(collection)
    return visible


async def get_all_document(*, owner_id: str):
    return await _run_chroma_io(
        _get_all_document_sync,
        owner_id,
        operation_name="get_all_document",
    )


def _deleted_tombstone_metadata(document_id: str, owner_id: str) -> dict:
    return {
        "ingest_status": "deleted",
        "source_document_id": document_id,
        "source_filename": document_id,
        # 墓碑也要认属主：否则同名重传时属主校验会把自己的墓碑判成别人的。
        "owner_id": owner_id,
        "deleted_at": int(time.time()),
    }


def _delete_document_sync(document_id: str, owner_id: str) -> str:
    storage_id = _storage_document_id(document_id, owner_id)
    _invalidate_bm25_cache(storage_id)
    try:
        collection = chromadb_client.get_collection(name=storage_id)
    except NotFoundError:
        # Old releases physically removed collections. Reserve the public id
        # even when only retained learning history remains, so a later upload
        # cannot silently inherit that history. A concurrent create/upload is
        # resolved by re-reading and deleting the one canonical collection.
        try:
            chromadb_client.create_collection(
                name=storage_id,
                metadata=_deleted_tombstone_metadata(document_id, owner_id),
            )
            return "material_deleted"
        except Exception as create_error:
            try:
                collection = chromadb_client.get_collection(name=storage_id)
            except NotFoundError:
                raise create_error

    collection = _require_public_document_owner(collection, document_id, owner_id)
    metadata = collection.metadata if isinstance(collection.metadata, dict) else {}
    if metadata.get("ingest_status") == "deleted" and collection.count() == 0:
        return "material_deleted"

    # Keep a durable tombstone because retained learner history is keyed by the
    # public document id. Reusing the same filename for different content would
    # silently attach old mastery, errors and durable sessions to the new file.
    deleting_metadata = {
        **metadata,
        "ingest_status": "deleting",
        "source_document_id": document_id,
        "source_filename": metadata.get("source_filename") or document_id,
    }
    collection.modify(metadata=deleting_metadata)
    # Close the invalidate→metadata transition window: a builder that started
    # after the first fence could still validate the formerly indexed
    # collection and publish old text before this modify completed.
    _invalidate_bm25_cache(storage_id)

    ids = list((collection.get(include=[]).get("ids") or []))
    for start in range(0, len(ids), 1000):
        collection.delete(ids=ids[start : start + 1000])
    if collection.count() != 0:
        raise RuntimeError("document tombstone still contains indexed chunks")

    collection.modify(
        metadata={
            **deleting_metadata,
            "ingest_status": "deleted",
            "deleted_at": int(time.time()),
        }
    )
    return "material_deleted"


async def delete_document(document_id: str, *, owner_id: str) -> str:
    # Chroma is synchronous. The complete deleting→empty→deleted transition
    # stays on the serialized worker even if the HTTP request is cancelled.
    return await _run_chroma_io(
        _delete_document_sync,
        document_id,
        owner_id,
        operation_name="delete_document",
    )
