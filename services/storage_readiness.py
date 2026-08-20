"""Bounded, non-generative probes for the application's required storage."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import logging
import math
import sqlite3
import tempfile
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from weakref import WeakKeyDictionary

from services.request_context import current_request_id


logger = logging.getLogger(__name__)


def _probe_chroma_catalog() -> None:
    # Import lazily so importing the health router does not create another
    # storage client or turn provider configuration into a readiness concern.
    from services.vectorstore import probe_vectorstore_readiness

    probe_vectorstore_readiness()


def _probe_learner_memory_store() -> None:
    # This checks the long-lived PostgresStore connection actually used by the
    # learner profile.  A fresh SELECT alone can look healthy after PostgreSQL
    # restarts while that existing connection remains unusable.
    from services.memory import probe_learner_memory_readiness

    probe_learner_memory_readiness()


def _selected_local_state_paths() -> tuple[tuple[Path, ...], Path]:
    """Resolve the SQLite/snapshot paths selected by live service singletons."""
    from routers.autonomous import autonomous_sessions
    from services.adaptive_sessions import adaptive_sessions
    from services.idempotency import request_idempotency
    from services.learning_path_store import learning_path_store
    from services.memory_persist import snapshot_path
    from services.quiz_sessions import quiz_sessions

    stores = (
        request_idempotency,
        learning_path_store,
        quiz_sessions,
        adaptive_sessions,
        autonomous_sessions,
    )
    unique_paths = {
        Path(store._sqlite_path).resolve(strict=False) for store in stores
    }
    return tuple(sorted(unique_paths, key=str)), Path(snapshot_path()).resolve(
        strict=False
    )


def _probe_local_state_paths(
    sqlite_paths: tuple[Path, ...],
    snapshot: Path,
) -> None:
    """Verify local durable state can be read and transactionally written."""
    resolved_snapshot = snapshot.resolve(strict=False)
    resolved_sqlite_paths = tuple(
        path.resolve(strict=False) for path in sqlite_paths
    )
    if resolved_snapshot in resolved_sqlite_paths:
        raise RuntimeError("learner-memory snapshot conflicts with SQLite state")

    for path in resolved_sqlite_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=0.5)
        try:
            connection.execute("PRAGMA busy_timeout = 500")
            connection.execute(
                "SELECT name FROM sqlite_master LIMIT 1"
            ).fetchone()
            connection.execute("BEGIN IMMEDIATE")
            # A RESERVED lock alone can succeed for a read-only database.  A
            # transactional schema write exercises the main database and its
            # journal/WAL path, while the rollback leaves no probe artifact.
            probe_table = f"studyloop_readiness_{uuid.uuid4().hex}"
            connection.execute(
                f'CREATE TABLE "{probe_table}" (probe INTEGER NOT NULL)'
            )
            connection.rollback()
        finally:
            connection.close()

    resolved_snapshot.parent.mkdir(parents=True, exist_ok=True)
    if resolved_snapshot.exists() and not resolved_snapshot.is_file():
        raise RuntimeError("learner-memory snapshot path is not a file")
    if resolved_snapshot.exists():
        # save_snapshot() atomically replaces this exact target.  A writable
        # parent alone is insufficient on platforms where a read-only target
        # cannot be replaced, so open the existing file read/write without
        # modifying its contents.
        with resolved_snapshot.open("r+b") as existing_snapshot:
            existing_snapshot.read(1)
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        dir=resolved_snapshot.parent,
        prefix=".studyloop-ready-",
        delete=False,
    )
    probe_path = Path(handle.name)
    try:
        with handle:
            handle.write(b"ready")
    finally:
        probe_path.unlink(missing_ok=True)


def _probe_local_state() -> None:
    sqlite_paths, snapshot = _selected_local_state_paths()
    _probe_local_state_paths(sqlite_paths, snapshot)


def _configured_database_url() -> str | None:
    # Read the backend selected at process startup rather than a mutable copy
    # of os.environ.  All durable stores use this same deployment setting.
    from services.memory import DATABASE_URL

    return DATABASE_URL or None


async def _probe_postgres_connection(
    database_url: str,
    connect_timeout_seconds: int,
    statement_timeout_ms: int,
) -> None:
    """Verify a new PostgreSQL connection and one bounded query."""
    import psycopg

    connection = await psycopg.AsyncConnection.connect(
        database_url,
        autocommit=True,
        connect_timeout=connect_timeout_seconds,
        options=f"-c statement_timeout={statement_timeout_ms}ms",
    )
    try:
        cursor = await connection.execute("SELECT 1")
        row = await cursor.fetchone()
        if row != (1,):
            raise RuntimeError("unexpected PostgreSQL readiness result")
    finally:
        await connection.close()


@dataclass(frozen=True, slots=True)
class _ProbeOutcome:
    ready: bool
    error_type: str | None = None


class _RetainedThreadProbe:
    """Keep one cross-loop worker alive across timeouts and cancellations."""

    def __init__(self, probe: Callable[[], None], *, thread_name: str) -> None:
        self._probe = probe
        self._thread_name = thread_name
        self._future: Future[_ProbeOutcome] | None = None
        self._lock = threading.Lock()

    def _capture(self) -> _ProbeOutcome:
        try:
            self._probe()
        except BaseException as exc:
            return _ProbeOutcome(False, type(exc).__name__)
        return _ProbeOutcome(True)

    def _start_worker(self) -> Future[_ProbeOutcome]:
        future: Future[_ProbeOutcome] = Future()

        def run() -> None:
            future.set_result(self._capture())

        threading.Thread(
            target=run,
            name=self._thread_name,
            daemon=True,
        ).start()
        return future

    def _clear_completed(self, future: Future[_ProbeOutcome]) -> None:
        # A waiter may already have timed out.  Clear immediately on worker
        # completion so a later request cannot consume a stale success.
        with self._lock:
            if self._future is future:
                self._future = None

    async def run(self, timeout_seconds: float) -> _ProbeOutcome:
        created = False
        with self._lock:
            future = self._future
            if future is None:
                future = self._start_worker()
                self._future = future
                created = True
        if created:
            future.add_done_callback(self._clear_completed)
        # asyncio.wrap_future() registers a callback for every waiter.  A
        # permanently blocked storage call would therefore retain one event
        # loop/Future per health request.  Polling keeps only the single
        # completion callback installed above and remains cross-loop safe.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        while not future.done():
            remaining = deadline - loop.time()
            if remaining <= 0:
                # The daemon worker cannot be force-cancelled.  Later checks
                # reuse it rather than spawning an unbounded thread pool.
                return _ProbeOutcome(False, "TimeoutError")
            await asyncio.sleep(min(0.01, remaining))
        return future.result()


class _RetainedAsyncProbe:
    """Keep one keyed async probe alive until its own driver budget ends."""

    def __init__(
        self,
        probe: Callable[[str, int, int], Awaitable[None]],
    ) -> None:
        self._probe = probe
        self._task: asyncio.Task[_ProbeOutcome] | None = None
        self._key: str | None = None

    async def _capture(
        self,
        database_url: str,
        connect_timeout_seconds: int,
        statement_timeout_ms: int,
        timeout_seconds: float,
    ) -> _ProbeOutcome:
        try:
            async with asyncio.timeout(timeout_seconds):
                await self._probe(
                    database_url,
                    connect_timeout_seconds,
                    statement_timeout_ms,
                )
        except TimeoutError:
            return _ProbeOutcome(False, "TimeoutError")
        except Exception as exc:
            return _ProbeOutcome(False, type(exc).__name__)
        return _ProbeOutcome(True)

    def _clear_completed(self, task: asyncio.Task[_ProbeOutcome]) -> None:
        # Done callbacks run on this probe's owning event loop.
        if self._task is task:
            self._task = None
            self._key = None

    async def run(
        self,
        *,
        key: str,
        database_url: str,
        connect_timeout_seconds: int,
        statement_timeout_ms: int,
        timeout_seconds: float,
    ) -> _ProbeOutcome:
        task = self._task
        if task is not None and task.cancelled():
            task = None
            self._task = None
            self._key = None
        if task is not None and self._key != key:
            if not task.done():
                return _ProbeOutcome(False, "ProbeConfigurationChanged")
            self._task = None
            self._key = None
            task = None
        if task is None:
            task = asyncio.create_task(
                self._capture(
                    database_url,
                    connect_timeout_seconds,
                    statement_timeout_ms,
                    timeout_seconds,
                )
            )
            self._task = task
            self._key = key
            task.add_done_callback(self._clear_completed)
        try:
            # The task's own timeout asks psycopg to cancel the operation, but
            # driver cleanup can itself wait on a half-open connection.  Give
            # each HTTP waiter an independent hard budget while retaining the
            # single task until that cleanup really finishes.
            return await asyncio.wait_for(
                asyncio.shield(task), timeout=timeout_seconds
            )
        except TimeoutError:
            return _ProbeOutcome(False, "TimeoutError")


@dataclass(slots=True)
class _LoopState:
    lock: asyncio.Lock
    postgres: _RetainedAsyncProbe


class StorageReadinessChecker:
    """Probe Chroma and the configured PostgreSQL data plane with one budget."""

    def __init__(
        self,
        *,
        chroma_probe: Callable[[], None] = _probe_chroma_catalog,
        learner_memory_probe: Callable[[], None] = _probe_learner_memory_store,
        local_state_probe: Callable[[], None] = _probe_local_state,
        database_url_loader: Callable[[], str | None] = _configured_database_url,
        postgres_probe: Callable[
            [str, int, int], Awaitable[None]
        ] = _probe_postgres_connection,
        chroma_timeout_seconds: float = 2.0,
        postgres_timeout_seconds: float = 4.0,
        postgres_connect_timeout_seconds: int = 2,
        postgres_statement_timeout_ms: int = 1_500,
        cache_ttl_seconds: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        numeric_values = {
            "chroma_timeout_seconds": chroma_timeout_seconds,
            "postgres_timeout_seconds": postgres_timeout_seconds,
            "postgres_connect_timeout_seconds": postgres_connect_timeout_seconds,
            "postgres_statement_timeout_ms": postgres_statement_timeout_ms,
        }
        for name, value in numeric_values.items():
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(cache_ttl_seconds) or cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds must be finite and non-negative")

        self._database_url_loader = database_url_loader
        self._chroma_timeout_seconds = chroma_timeout_seconds
        self._postgres_timeout_seconds = postgres_timeout_seconds
        self._postgres_connect_timeout_seconds = postgres_connect_timeout_seconds
        self._postgres_statement_timeout_ms = postgres_statement_timeout_ms
        self._cache_ttl_seconds = cache_ttl_seconds
        self._clock = clock
        self._postgres_probe = postgres_probe
        self._loop_states: WeakKeyDictionary = WeakKeyDictionary()
        self._loop_states_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._chroma = _RetainedThreadProbe(
            chroma_probe, thread_name="studyloop-ready-chroma"
        )
        self._learner_memory = _RetainedThreadProbe(
            learner_memory_probe, thread_name="studyloop-ready-memory"
        )
        self._local_state = _RetainedThreadProbe(
            local_state_probe, thread_name="studyloop-ready-local"
        )
        self._cached_payload: dict[str, Any] | None = None
        self._cached_database_key: str | None = None
        self._cache_expires_at = 0.0

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cached_payload = None
            self._cached_database_key = None
            self._cache_expires_at = 0.0

    def _loop_state(self) -> _LoopState:
        loop = asyncio.get_running_loop()
        with self._loop_states_lock:
            # Values contain loop-bound locks, so proactively remove closed
            # loops instead of relying on WeakKeyDictionary alone.
            for known_loop in list(self._loop_states):
                if known_loop is not loop and known_loop.is_closed():
                    del self._loop_states[known_loop]
            state = self._loop_states.get(loop)
            if state is None:
                state = _LoopState(
                    lock=asyncio.Lock(),
                    postgres=_RetainedAsyncProbe(self._postgres_probe),
                )
                self._loop_states[loop] = state
            return state

    def _read_cache(self, database_key: str) -> dict[str, Any] | None:
        with self._cache_lock:
            if (
                self._cached_payload is not None
                and self._cached_database_key == database_key
                and self._clock() < self._cache_expires_at
            ):
                return copy.deepcopy(self._cached_payload)
        return None

    def _write_cache(
        self, database_key: str, payload: dict[str, Any]
    ) -> None:
        with self._cache_lock:
            self._cached_payload = copy.deepcopy(payload)
            self._cached_database_key = database_key
            self._cache_expires_at = self._clock() + self._cache_ttl_seconds

    def _database_url(self) -> str | None:
        value = self._database_url_loader()
        if value is None or not value.strip():
            return None
        return value

    @staticmethod
    def _database_key(database_url: str | None) -> str:
        if database_url is None:
            return "not_configured"
        return hashlib.sha256(database_url.encode("utf-8")).hexdigest()

    def _log_unavailable(self, component: str, outcome: _ProbeOutcome) -> None:
        logger.warning(
            "storage readiness probe unavailable request_id=%s component=%s "
            "error_type=%s",
            current_request_id(),
            component,
            outcome.error_type or "UnknownError",
        )

    async def check(self) -> dict[str, Any]:
        database_url = self._database_url()
        database_key = self._database_key(database_url)
        cached = self._read_cache(database_key)
        if cached is not None:
            return cached

        loop_state = self._loop_state()
        async with loop_state.lock:
            cached = self._read_cache(database_key)
            if cached is not None:
                return cached

            chroma_future = self._chroma.run(self._chroma_timeout_seconds)
            if database_url is None:
                chroma_outcome, local_state_outcome = await asyncio.gather(
                    chroma_future,
                    self._local_state.run(self._postgres_timeout_seconds),
                )
                postgres_outcome = None
                memory_outcome = None
            else:
                (
                    chroma_outcome,
                    postgres_outcome,
                    memory_outcome,
                ) = await asyncio.gather(
                    chroma_future,
                    loop_state.postgres.run(
                        key=database_key,
                        database_url=database_url,
                        connect_timeout_seconds=(
                            self._postgres_connect_timeout_seconds
                        ),
                        statement_timeout_ms=self._postgres_statement_timeout_ms,
                        timeout_seconds=self._postgres_timeout_seconds,
                    ),
                    self._learner_memory.run(self._postgres_timeout_seconds),
                )
                local_state_outcome = None

            if chroma_outcome.ready:
                chroma_check = {
                    "status": "ready",
                    "required": True,
                    "code": "chroma_ready",
                }
            else:
                self._log_unavailable("chroma", chroma_outcome)
                chroma_check = {
                    "status": "unavailable",
                    "required": True,
                    "code": "chroma_unavailable",
                }

            if database_url is None:
                assert local_state_outcome is not None
                if local_state_outcome.ready:
                    local_state_check = {
                        "status": "ready",
                        "required": True,
                        "code": "local_state_ready",
                    }
                else:
                    self._log_unavailable("local_state", local_state_outcome)
                    local_state_check = {
                        "status": "unavailable",
                        "required": True,
                        "code": "local_state_unavailable",
                    }
                postgres_check = {
                    "status": "not_configured",
                    "required": False,
                    "code": "postgres_not_configured",
                }
                state_ready = local_state_outcome.ready
            else:
                assert postgres_outcome is not None
                assert memory_outcome is not None
                if not postgres_outcome.ready:
                    self._log_unavailable("postgres", postgres_outcome)
                if not memory_outcome.ready:
                    self._log_unavailable("learner_memory", memory_outcome)
                state_ready = (
                    postgres_outcome.ready and memory_outcome.ready
                )
                postgres_check = {
                    "status": "ready" if state_ready else "unavailable",
                    "required": True,
                    "code": (
                        "postgres_ready"
                        if state_ready
                        else "postgres_unavailable"
                    ),
                }
                local_state_check = {
                    "status": "not_configured",
                    "required": False,
                    "code": "local_state_not_configured",
                }

            ready = chroma_outcome.ready and state_ready
            payload = {
                "name": "StudyLoop",
                "status": "ready" if ready else "unready",
                "code": "storage_ready" if ready else "storage_unavailable",
                "checks": {
                    "chroma": chroma_check,
                    "postgres": postgres_check,
                    "local_state": local_state_check,
                },
            }
            self._write_cache(database_key, payload)
            return payload


storage_readiness_checker = StorageReadinessChecker()
