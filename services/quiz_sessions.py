"""Durable Web Quiz sessions with cross-worker fencing.

The browser only receives a safe projection of this state.  Full questions,
answers, grading caches, and memory-effect markers stay in a versioned JSON
payload backed by SQLite locally or PostgreSQL when ``DATABASE_URL`` is set.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from pydantic import ValidationError

from models.session import QuizSessionAggregate


load_dotenv(Path(__file__).parent.parent / ".env")
logger = logging.getLogger(__name__)

_TABLE = "studyloop_quiz_sessions"
_POSTGRES_SCHEMA_LOCK_ID = 0x5354554459515549  # ASCII "STUDYQUI"


class QuizSessionAlreadyExistsError(RuntimeError):
    pass


class QuizSessionCapacityError(RuntimeError):
    pass


class QuizSessionPayloadTooLargeError(ValueError):
    pass


class QuizSessionCorruptError(RuntimeError):
    pass


class QuizSessionStartConflictError(RuntimeError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class QuizSessionApiError(RuntimeError):
    """Stable HTTP-facing failure without exposing storage internals."""

    def __init__(self, status_code: int, code: str, detail: str, reason: str | None = None):
        self.status_code = status_code
        self.code = code
        self.detail = detail
        self.reason = reason
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class StoredQuizSession:
    aggregate: QuizSessionAggregate
    revision: int
    created_at: float
    updated_at: float
    expires_at: float
    busy: bool
    expired: bool


@dataclass(frozen=True, slots=True)
class QuizSessionCreateResult:
    record: StoredQuizSession
    created: bool


@dataclass(frozen=True, slots=True)
class QuizSessionClaim:
    claimed: bool
    reason: str | None = None
    token: str | None = None
    record: StoredQuizSession | None = None


class QuizSessionStore:
    """Versioned mutable session store used only by the stable Web Quiz API."""

    def __init__(
        self,
        *,
        database_url: str | None = None,
        sqlite_path: str | None = None,
        ttl_seconds: float = 24 * 60 * 60,
        operation_lease_seconds: float = 10 * 60,
        max_count: int = 500,
        max_payload_bytes: int = 2 * 1024 * 1024,
        postgres_connect_timeout_seconds: int = 5,
        postgres_lock_timeout_ms: int = 5_000,
        postgres_statement_timeout_ms: int = 15_000,
        schema_init_wait_timeout_seconds: float = 30,
        cancel_drain_timeout_seconds: float = 20,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if database_url and sqlite_path:
            raise ValueError("database_url and sqlite_path are mutually exclusive")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if operation_lease_seconds <= 0:
            raise ValueError("operation_lease_seconds must be positive")
        if max_count <= 0:
            raise ValueError("max_count must be positive")
        if max_payload_bytes <= 0:
            raise ValueError("max_payload_bytes must be positive")
        if (
            not math.isfinite(postgres_connect_timeout_seconds)
            or postgres_connect_timeout_seconds <= 0
        ):
            raise ValueError("postgres_connect_timeout_seconds must be positive")
        if not math.isfinite(postgres_lock_timeout_ms) or postgres_lock_timeout_ms <= 0:
            raise ValueError("postgres_lock_timeout_ms must be positive")
        if not math.isfinite(postgres_statement_timeout_ms) or postgres_statement_timeout_ms <= 0:
            raise ValueError("postgres_statement_timeout_ms must be positive")
        if (
            not math.isfinite(schema_init_wait_timeout_seconds)
            or schema_init_wait_timeout_seconds <= 0
        ):
            raise ValueError("schema_init_wait_timeout_seconds must be positive")
        if not math.isfinite(cancel_drain_timeout_seconds) or cancel_drain_timeout_seconds <= 0:
            raise ValueError("cancel_drain_timeout_seconds must be positive")

        self._database_url = database_url
        self._sqlite_path = (
            sqlite_path
            or os.getenv("QUIZ_SESSION_DB_PATH")
            or os.getenv("IDEMPOTENCY_DB_PATH", "./.idempotency.sqlite3")
        )
        self._ttl_seconds = ttl_seconds
        self._operation_lease_seconds = operation_lease_seconds
        self._max_count = max_count
        self._max_payload_bytes = max_payload_bytes
        self._postgres_connect_timeout_seconds = postgres_connect_timeout_seconds
        self._postgres_lock_timeout_ms = postgres_lock_timeout_ms
        self._postgres_statement_timeout_ms = postgres_statement_timeout_ms
        self._schema_init_wait_timeout_seconds = schema_init_wait_timeout_seconds
        self._cancel_drain_timeout_seconds = cancel_drain_timeout_seconds
        self._clock = clock
        self._schema_ready = False
        self._schema_lock = threading.Lock()
        self._background_workers: set[asyncio.Task] = set()

    @classmethod
    def from_environment(cls) -> "QuizSessionStore":
        database_url = os.getenv("DATABASE_URL") or None
        return cls(
            database_url=database_url,
            ttl_seconds=float(os.getenv("QUIZ_SESSION_TTL_SECONDS", "86400")),
            operation_lease_seconds=float(os.getenv("QUIZ_SESSION_OPERATION_LEASE_SECONDS", "600")),
            max_count=int(os.getenv("QUIZ_SESSION_MAX_COUNT", "500")),
            max_payload_bytes=int(
                os.getenv("QUIZ_SESSION_MAX_PAYLOAD_BYTES", str(2 * 1024 * 1024))
            ),
            postgres_connect_timeout_seconds=int(
                os.getenv("QUIZ_SESSION_PG_CONNECT_TIMEOUT_SECONDS", "5")
            ),
            postgres_lock_timeout_ms=int(os.getenv("QUIZ_SESSION_PG_LOCK_TIMEOUT_MS", "5000")),
            postgres_statement_timeout_ms=int(
                os.getenv("QUIZ_SESSION_PG_STATEMENT_TIMEOUT_MS", "15000")
            ),
            schema_init_wait_timeout_seconds=float(
                os.getenv("QUIZ_SESSION_SCHEMA_INIT_WAIT_TIMEOUT_SECONDS", "30")
            ),
            cancel_drain_timeout_seconds=float(
                os.getenv("QUIZ_SESSION_CANCEL_DRAIN_TIMEOUT_SECONDS", "20")
            ),
        )

    async def create(
        self,
        aggregate: QuizSessionAggregate,
        *,
        start_key: str | None = None,
        start_request: dict[str, Any] | None = None,
    ) -> QuizSessionCreateResult:
        payload_json = self._serialize_payload(aggregate)
        immutable_hash = self._immutable_hash(aggregate)
        key_hash, request_hash = self._start_hashes(start_key, start_request)
        return await self._run_thread(
            self._create_sync,
            aggregate.session.session_id,
            aggregate.session.status,
            payload_json,
            immutable_hash,
            key_hash,
            request_hash,
            self._clock(),
        )

    async def find_start(
        self,
        start_key: str,
        start_request: dict[str, Any],
    ) -> StoredQuizSession | None:
        key_hash, request_hash = self._start_hashes(start_key, start_request)
        return await self._run_thread(
            self._find_start_sync,
            key_hash,
            request_hash,
            self._clock(),
        )

    async def inspect(self, session_id: str) -> StoredQuizSession | None:
        if not session_id:
            return None
        return await self._run_thread(self._inspect_sync, session_id, self._clock())

    async def claim(self, session_id: str, operation: str) -> QuizSessionClaim:
        if not session_id:
            return QuizSessionClaim(claimed=False, reason="missing")
        if not operation or len(operation) > 64:
            raise ValueError("invalid quiz session operation")
        token = secrets.token_urlsafe(32)
        worker = asyncio.create_task(
            asyncio.to_thread(
                self._claim_sync,
                session_id,
                operation,
                token,
                self._clock(),
            )
        )
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            deadline = asyncio.get_running_loop().time() + self._cancel_drain_timeout_seconds
            completed, decision = await self._drain_cancelled_worker(
                worker,
                deadline=deadline,
                track_on_timeout=False,
            )

            # A database thread may have committed ownership just before the
            # request was cancelled. The caller never received the fencing
            # token, so the store must release that otherwise-orphaned claim.
            if completed:
                if decision is not None and decision.claimed:
                    cleanup = asyncio.create_task(self.release(session_id, token))
                    await self._drain_cancelled_worker(
                        cleanup,
                        deadline=deadline,
                    )
            else:
                cleanup = asyncio.create_task(self._release_late_claim(worker, session_id, token))
                self._track_background_worker(cleanup)
            raise

    async def checkpoint(
        self,
        session_id: str,
        claim_token: str,
        aggregate: QuizSessionAggregate,
    ) -> StoredQuizSession | None:
        payload_json = self._serialize_payload(aggregate)
        immutable_hash = self._immutable_hash(aggregate)
        return await self._run_thread(
            self._checkpoint_sync,
            session_id,
            claim_token,
            aggregate.session.status,
            payload_json,
            immutable_hash,
            self._clock(),
        )

    async def complete(
        self,
        session_id: str,
        claim_token: str,
        aggregate: QuizSessionAggregate,
    ) -> StoredQuizSession | None:
        payload_json = self._serialize_payload(aggregate)
        immutable_hash = self._immutable_hash(aggregate)
        return await self._run_thread(
            self._complete_sync,
            session_id,
            claim_token,
            aggregate.session.status,
            payload_json,
            immutable_hash,
            self._clock(),
        )

    async def release(self, session_id: str, claim_token: str) -> bool:
        return await self._run_thread(
            self._release_sync,
            session_id,
            claim_token,
            self._clock(),
        )

    async def _run_thread(self, function, *args):
        worker = asyncio.create_task(asyncio.to_thread(function, *args))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            await self._drain_cancelled_worker(worker)
            raise

    async def _release_late_claim(
        self,
        worker: asyncio.Task,
        session_id: str,
        token: str,
    ) -> None:
        try:
            decision = await asyncio.shield(worker)
            if decision.claimed:
                await self.release(session_id, token)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            logger.error(
                "late Quiz claim cleanup failed: error_type=%s",
                type(exc).__name__,
            )

    async def _drain_cancelled_worker(
        self,
        worker: asyncio.Task,
        *,
        deadline: float | None = None,
        track_on_timeout: bool = True,
    ) -> tuple[bool, Any]:
        """Drain a shielded database worker without extending its deadline."""
        loop = asyncio.get_running_loop()
        if deadline is None:
            deadline = loop.time() + self._cancel_drain_timeout_seconds

        while not worker.done():
            remaining = deadline - loop.time()
            if remaining <= 0:
                if track_on_timeout:
                    self._track_background_worker(worker)
                return False, None
            try:
                await asyncio.wait_for(asyncio.shield(worker), timeout=remaining)
            except asyncio.CancelledError:
                continue
            except TimeoutError:
                continue
            except BaseException:
                # The worker failure is consumed below so cancellation remains
                # the public outcome.
                break

        try:
            return True, worker.result()
        except BaseException as exc:
            logger.error(
                "cancelled Quiz store worker failed: error_type=%s",
                type(exc).__name__,
            )
            return True, None

    def _track_background_worker(self, worker: asyncio.Task) -> None:
        self._background_workers.add(worker)

        def on_done(completed: asyncio.Task) -> None:
            self._background_workers.discard(completed)
            try:
                completed.result()
            except BaseException as exc:
                logger.error(
                    "background Quiz store worker failed: error_type=%s",
                    type(exc).__name__,
                )

        worker.add_done_callback(on_done)

    def _serialize_payload(self, aggregate: QuizSessionAggregate) -> str:
        # Revalidate a deep JSON copy so assignment to nested legacy models can
        # never bypass durable state-machine invariants.
        validated = QuizSessionAggregate.model_validate(aggregate.model_dump(mode="json"))
        payload_json = json.dumps(
            validated.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        payload_bytes = len(payload_json.encode("utf-8"))
        if payload_bytes > self._max_payload_bytes:
            raise QuizSessionPayloadTooLargeError(
                f"quiz session payload is {payload_bytes} bytes; limit is {self._max_payload_bytes}"
            )
        return payload_json

    @staticmethod
    def _start_hashes(
        start_key: str | None,
        start_request: dict[str, Any] | None,
    ) -> tuple[str | None, str | None]:
        if start_key is None:
            return None, None
        if start_request is None:
            raise ValueError("start_request is required with start_key")
        key_hash = hashlib.sha256(start_key.encode("utf-8")).hexdigest()
        canonical = json.dumps(
            start_request,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        request_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return key_hash, request_hash

    @staticmethod
    def _immutable_hash(aggregate: QuizSessionAggregate) -> str:
        session = aggregate.session
        immutable = {
            "session_id": session.session_id,
            "document_id": session.document_id,
            "user_id": session.user_id,
            "questions": [question.model_dump(mode="json") for question in session.questions],
        }
        # Preserve the exact legacy hash for every unbound Quiz.  A stage
        # binding is immutable only for newly-created Learning Path quizzes.
        if aggregate.learning_path_source is not None:
            immutable["learning_path_source"] = aggregate.learning_path_source.model_dump(
                mode="json"
            )
        canonical = json.dumps(
            immutable,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _connect(self):
        if self._database_url:
            import psycopg
            from psycopg.conninfo import conninfo_to_dict

            try:
                connection_parameters = conninfo_to_dict(self._database_url)
            except Exception:
                raise ValueError("PostgreSQL Quiz session DATABASE_URL is invalid") from None
            explicit_options = connection_parameters.get("options")
            environment_options = os.getenv("PGOPTIONS", "").strip()
            service_configured = bool(
                connection_parameters.get("service") or os.getenv("PGSERVICE")
            )
            if service_configured and explicit_options is None and not environment_options:
                raise ValueError(
                    "PostgreSQL service DSNs must expose connection options "
                    "through DATABASE_URL or PGOPTIONS so Quiz session "
                    "safety limits can be merged without silently discarding "
                    "service-file options"
                )
            existing_options = (
                str(explicit_options) if explicit_options is not None else environment_options
            ).strip()
            bounded_options = (
                f"-c lock_timeout={self._postgres_lock_timeout_ms}ms "
                f"-c statement_timeout={self._postgres_statement_timeout_ms}ms"
            )
            return psycopg.connect(
                self._database_url,
                connect_timeout=self._postgres_connect_timeout_seconds,
                options=f"{existing_options} {bounded_options}".strip(),
            )
        path = Path(self._sqlite_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=10)
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    @contextmanager
    def _transaction(self, *, write: bool = False):
        connection = self._connect()
        try:
            if write and not self._postgres:
                connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @property
    def _postgres(self) -> bool:
        return self._database_url is not None

    @property
    def _placeholder(self) -> str:
        return "%s" if self._postgres else "?"

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        acquired = self._schema_lock.acquire(timeout=self._schema_init_wait_timeout_seconds)
        if not acquired:
            raise TimeoutError("quiz session schema initialization timed out")
        try:
            if self._schema_ready:
                return
            with self._transaction(write=True) as connection:
                if self._postgres:
                    connection.execute(
                        "SELECT pg_advisory_xact_lock(%s)",
                        (_POSTGRES_SCHEMA_LOCK_ID,),
                    )
                connection.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {_TABLE} (
                        session_id TEXT PRIMARY KEY,
                        schema_version INTEGER NOT NULL CHECK (schema_version = 1),
                        payload_json TEXT NOT NULL,
                        status TEXT NOT NULL CHECK (status IN ('active', 'completed')),
                        immutable_hash TEXT NOT NULL,
                        revision BIGINT NOT NULL CHECK (revision >= 1),
                        start_key_hash TEXT UNIQUE,
                        start_request_hash TEXT,
                        created_at DOUBLE PRECISION NOT NULL,
                        updated_at DOUBLE PRECISION NOT NULL,
                        expires_at DOUBLE PRECISION NOT NULL,
                        claim_token TEXT,
                        claim_operation TEXT,
                        claim_expires_at DOUBLE PRECISION,
                        CHECK (
                            (start_key_hash IS NULL AND start_request_hash IS NULL)
                            OR
                            (start_key_hash IS NOT NULL AND start_request_hash IS NOT NULL)
                        ),
                        CHECK (
                            (claim_token IS NULL AND claim_operation IS NULL
                             AND claim_expires_at IS NULL)
                            OR
                            (claim_token IS NOT NULL AND claim_operation IS NOT NULL
                             AND claim_expires_at IS NOT NULL)
                        )
                    )
                    """
                )
                connection.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {_TABLE}_status_expiry_idx
                    ON {_TABLE} (status, expires_at)
                    """
                )
                connection.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {_TABLE}_updated_idx
                    ON {_TABLE} (updated_at)
                    """
                )
            self._schema_ready = True
        finally:
            self._schema_lock.release()

    def _create_sync(
        self,
        session_id: str,
        status: str,
        payload_json: str,
        immutable_hash: str,
        start_key_hash: str | None,
        start_request_hash: str | None,
        now: float,
    ) -> QuizSessionCreateResult:
        self._ensure_schema()
        p = self._placeholder
        with self._transaction(write=True) as connection:
            self._lock_capacity(connection)
            if start_key_hash is not None:
                row = self._select_by_start_key(connection, start_key_hash)
                if row is not None:
                    self._validate_start_request(row, start_request_hash)
                    return QuizSessionCreateResult(
                        record=self._row_to_record(row, now), created=False
                    )

            existing = connection.execute(
                f"SELECT session_id FROM {_TABLE} WHERE session_id = {p}",
                (session_id,),
            ).fetchone()
            if existing is not None:
                raise QuizSessionAlreadyExistsError(session_id)

            self._make_capacity(connection, now)
            expires_at = now + self._ttl_seconds
            connection.execute(
                f"""
                INSERT INTO {_TABLE} (
                    session_id, schema_version, payload_json, status, revision,
                    immutable_hash, start_key_hash, start_request_hash, created_at,
                    updated_at, expires_at
                ) VALUES ({p}, 1, {p}, {p}, 1, {p}, {p}, {p}, {p}, {p}, {p})
                """,
                (
                    session_id,
                    payload_json,
                    status,
                    immutable_hash,
                    start_key_hash,
                    start_request_hash,
                    now,
                    now,
                    expires_at,
                ),
            )
            row = self._select_by_id(connection, session_id)
            if row is None:
                raise RuntimeError("quiz session disappeared after create")
            return QuizSessionCreateResult(record=self._row_to_record(row, now), created=True)

    def _find_start_sync(
        self,
        start_key_hash: str,
        start_request_hash: str,
        now: float,
    ) -> StoredQuizSession | None:
        self._ensure_schema()
        with self._transaction() as connection:
            row = self._select_by_start_key(connection, start_key_hash)
        if row is None:
            return None
        self._validate_start_request(row, start_request_hash)
        return self._row_to_record(row, now)

    def _inspect_sync(self, session_id: str, now: float) -> StoredQuizSession | None:
        self._ensure_schema()
        with self._transaction() as connection:
            row = self._select_by_id(connection, session_id)
        return None if row is None else self._row_to_record(row, now)

    def _claim_sync(
        self,
        session_id: str,
        operation: str,
        token: str,
        now: float,
    ) -> QuizSessionClaim:
        self._ensure_schema()
        p = self._placeholder
        lease_expires = now + self._operation_lease_seconds
        with self._transaction(write=True) as connection:
            cursor = connection.execute(
                f"""
                UPDATE {_TABLE}
                SET claim_token = {p}, claim_operation = {p},
                    claim_expires_at = {p}, updated_at = {p}
                WHERE session_id = {p} AND expires_at > {p}
                  AND (claim_token IS NULL OR claim_expires_at <= {p})
                """,
                (
                    token,
                    operation,
                    lease_expires,
                    now,
                    session_id,
                    now,
                    now,
                ),
            )
            if cursor.rowcount == 1:
                row = self._select_by_id(connection, session_id)
                if row is None:
                    raise RuntimeError("quiz session disappeared after claim")
                return QuizSessionClaim(
                    claimed=True,
                    token=token,
                    record=self._row_to_record(row, now),
                )

            row = self._select_by_id(connection, session_id)
            if row is None:
                return QuizSessionClaim(claimed=False, reason="missing")
            if row[8] <= now:
                return QuizSessionClaim(claimed=False, reason="expired")
            return QuizSessionClaim(claimed=False, reason="in_progress")

    def _checkpoint_sync(
        self,
        session_id: str,
        claim_token: str,
        status: str,
        payload_json: str,
        immutable_hash: str,
        now: float,
    ) -> StoredQuizSession | None:
        self._ensure_schema()
        p = self._placeholder
        lease_expires = now + self._operation_lease_seconds
        with self._transaction(write=True) as connection:
            cursor = connection.execute(
                f"""
                UPDATE {_TABLE}
                SET payload_json = {p}, status = {p}, revision = revision + 1,
                    updated_at = {p}, claim_expires_at = {p}
                WHERE session_id = {p} AND claim_token = {p}
                  AND claim_expires_at > {p} AND expires_at > {p}
                  AND immutable_hash = {p}
                """,
                (
                    payload_json,
                    status,
                    now,
                    lease_expires,
                    session_id,
                    claim_token,
                    now,
                    now,
                    immutable_hash,
                ),
            )
            if cursor.rowcount != 1:
                return None
            row = self._select_by_id(connection, session_id)
            return None if row is None else self._row_to_record(row, now)

    def _complete_sync(
        self,
        session_id: str,
        claim_token: str,
        status: str,
        payload_json: str,
        immutable_hash: str,
        now: float,
    ) -> StoredQuizSession | None:
        self._ensure_schema()
        p = self._placeholder
        expires_at = now + self._ttl_seconds
        with self._transaction(write=True) as connection:
            cursor = connection.execute(
                f"""
                UPDATE {_TABLE}
                SET payload_json = {p}, status = {p}, revision = revision + 1,
                    updated_at = {p}, expires_at = {p}, claim_token = NULL,
                    claim_operation = NULL, claim_expires_at = NULL
                WHERE session_id = {p} AND claim_token = {p}
                  AND claim_expires_at > {p} AND expires_at > {p}
                  AND immutable_hash = {p}
                """,
                (
                    payload_json,
                    status,
                    now,
                    expires_at,
                    session_id,
                    claim_token,
                    now,
                    now,
                    immutable_hash,
                ),
            )
            if cursor.rowcount != 1:
                return None
            row = self._select_by_id(connection, session_id)
            return None if row is None else self._row_to_record(row, now)

    def _release_sync(self, session_id: str, claim_token: str, now: float) -> bool:
        self._ensure_schema()
        p = self._placeholder
        with self._transaction(write=True) as connection:
            cursor = connection.execute(
                f"""
                UPDATE {_TABLE}
                SET claim_token = NULL, claim_operation = NULL,
                    claim_expires_at = NULL, updated_at = {p}
                WHERE session_id = {p} AND claim_token = {p}
                """,
                (now, session_id, claim_token),
            )
            return cursor.rowcount == 1

    def _lock_capacity(self, connection) -> None:
        if self._postgres:
            connection.execute(f"LOCK TABLE {_TABLE} IN SHARE ROW EXCLUSIVE MODE")

    def _make_capacity(self, connection, now: float) -> None:
        p = self._placeholder
        connection.execute(
            f"""
            DELETE FROM {_TABLE}
            WHERE expires_at <= {p}
              AND (claim_token IS NULL OR claim_expires_at <= {p})
            """,
            (now, now),
        )
        count = connection.execute(f"SELECT COUNT(*) FROM {_TABLE}").fetchone()[0]
        required = count - self._max_count + 1
        if required <= 0:
            return
        rows = connection.execute(
            f"""
            SELECT session_id FROM {_TABLE}
            WHERE status = 'completed'
              AND (claim_token IS NULL OR claim_expires_at <= {p})
            ORDER BY updated_at ASC, session_id ASC
            """,
            (now,),
        ).fetchall()
        if len(rows) < required:
            raise QuizSessionCapacityError("all quiz session capacity is active")
        for (session_id,) in rows[:required]:
            connection.execute(
                f"DELETE FROM {_TABLE} WHERE session_id = {p}",
                (session_id,),
            )

    def _select_by_id(self, connection, session_id: str):
        p = self._placeholder
        return connection.execute(
            f"""
            SELECT session_id, schema_version, payload_json, status, revision,
                   immutable_hash, created_at, updated_at, expires_at,
                   claim_token, claim_expires_at, start_request_hash
            FROM {_TABLE} WHERE session_id = {p}
            """,
            (session_id,),
        ).fetchone()

    def _select_by_start_key(self, connection, start_key_hash: str):
        p = self._placeholder
        return connection.execute(
            f"""
            SELECT session_id, schema_version, payload_json, status, revision,
                   immutable_hash, created_at, updated_at, expires_at,
                   claim_token, claim_expires_at, start_request_hash
            FROM {_TABLE} WHERE start_key_hash = {p}
            """,
            (start_key_hash,),
        ).fetchone()

    @staticmethod
    def _validate_start_request(row, request_hash: str | None) -> None:
        if row[11] != request_hash:
            raise QuizSessionStartConflictError("payload_mismatch")

    @staticmethod
    def _decode_payload(payload_json: str) -> QuizSessionAggregate:
        try:
            raw = json.loads(payload_json)
            if not isinstance(raw, dict):
                raise TypeError("quiz session payload is not a JSON object")
            return QuizSessionAggregate.model_validate(raw)
        except (TypeError, json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise QuizSessionCorruptError("invalid durable quiz session payload") from exc

    def _row_to_record(self, row, now: float) -> StoredQuizSession:
        if row[1] != 1:
            raise QuizSessionCorruptError("unsupported quiz session schema version")
        aggregate = self._decode_payload(row[2])
        if aggregate.session.session_id != row[0] or aggregate.session.status != row[3]:
            raise QuizSessionCorruptError("quiz session row does not match payload")
        immutable_hash = row[5]
        if self._immutable_hash(aggregate) != immutable_hash:
            raise QuizSessionCorruptError("quiz session immutable fields changed")
        claim_token = row[9]
        claim_expires_at = row[10]
        busy = bool(
            claim_token is not None and claim_expires_at is not None and claim_expires_at > now
        )
        return StoredQuizSession(
            aggregate=aggregate,
            revision=int(row[4]),
            created_at=float(row[6]),
            updated_at=float(row[7]),
            expires_at=float(row[8]),
            busy=busy,
            expired=float(row[8]) <= now,
        )


quiz_sessions = QuizSessionStore.from_environment()
