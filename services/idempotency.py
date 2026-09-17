"""Durable request receipts for safe retries of tool-using HTTP requests.

The receipt is intentionally separate from the in-memory audit trail. A clean
request attempt owns a renewable, fenced lease, so an abandoned attempt can be
reclaimed without letting its stale worker commit. A request that reached a
non-replayable tool leaves the reclaimable state before the handler starts and
is failed closed after an interrupted run. This is an at-most-once guard, not a
distributed exactly-once transaction with the LangGraph memory store.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv


load_dotenv(Path(__file__).parent.parent / ".env")
logger = logging.getLogger(__name__)

_TABLE = "studyloop_idempotency_receipts"
_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")
# ASCII "STUDYIDE" encoded as a positive signed bigint. This key is distinct
# from the Autonomous session schema lock and stable across worker processes.
_POSTGRES_SCHEMA_LOCK_ID = 0x5354554459494445


class IdempotencyConflictError(RuntimeError):
    """The supplied key cannot safely start a new execution."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class ReceiptLease:
    """Opaque ownership proof for one clean, replayable request attempt."""

    key: str
    owner_token: str
    recovery_token: str
    expires_at: float


@dataclass(frozen=True, slots=True)
class BeginDecision:
    replayed: bool
    response: dict[str, Any] | None = None
    lease: ReceiptLease | None = None


class InvalidIdempotencyKeyError(ValueError):
    """The caller sent a malformed Idempotency-Key header.

    This is one of the few genuinely client-caused ValueErrors on the request
    path, so it carries its own type and its own 400 handler instead of relying
    on a catch-all ValueError handler that would also swallow internal
    invariant failures.
    """


def normalize_idempotency_key(value: object) -> str | None:
    """Return a validated key, treating FastAPI's direct-call default as absent."""
    if not isinstance(value, str):
        return None
    key = value.strip()
    if not _KEY_PATTERN.fullmatch(key):
        raise InvalidIdempotencyKeyError(
            "Idempotency-Key 必须为 8-128 位字母、数字或 . _ : -"
        )
    return key


def request_fingerprint(operation: str, payload: dict[str, Any]) -> str:
    """同一个请求必须永远算出同一个指纹，否则幂等收据认不出重放。

    allow_nan=False 是这个保证的一部分：NaN 序列化成的 "NaN" 不是合法 JSON，
    而且 NaN != NaN，一旦放进 payload，同一个请求每次都会算出不同的指纹。
    """
    canonical = json.dumps(
        {"operation": operation, "payload": payload},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class IdempotencyStore:
    """Atomic receipt store backed by SQLite locally or PostgreSQL in production."""

    def __init__(
        self,
        *,
        database_url: str | None = None,
        sqlite_path: str | None = None,
        lease_seconds: float = 10 * 60,
        postgres_connect_timeout_seconds: int = 5,
        postgres_lock_timeout_ms: int = 5_000,
        postgres_statement_timeout_ms: int = 15_000,
        postgres_tcp_user_timeout_ms: int = 30_000,
        schema_init_wait_timeout_seconds: float = 30,
        cancel_drain_timeout_seconds: float = 20,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if database_url and sqlite_path:
            raise ValueError("database_url and sqlite_path are mutually exclusive")
        if not math.isfinite(lease_seconds) or lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if (
            not math.isfinite(postgres_connect_timeout_seconds)
            or postgres_connect_timeout_seconds <= 0
        ):
            raise ValueError("postgres_connect_timeout_seconds must be positive")
        if not math.isfinite(postgres_lock_timeout_ms) or postgres_lock_timeout_ms <= 0:
            raise ValueError("postgres_lock_timeout_ms must be positive")
        if not math.isfinite(postgres_statement_timeout_ms) or postgres_statement_timeout_ms <= 0:
            raise ValueError("postgres_statement_timeout_ms must be positive")
        if not math.isfinite(postgres_tcp_user_timeout_ms) or postgres_tcp_user_timeout_ms <= 0:
            raise ValueError("postgres_tcp_user_timeout_ms must be positive")
        if (
            not math.isfinite(schema_init_wait_timeout_seconds)
            or schema_init_wait_timeout_seconds <= 0
        ):
            raise ValueError("schema_init_wait_timeout_seconds must be positive")
        if not math.isfinite(cancel_drain_timeout_seconds) or cancel_drain_timeout_seconds <= 0:
            raise ValueError("cancel_drain_timeout_seconds must be positive")
        self._database_url = database_url
        self._sqlite_path = sqlite_path or os.getenv(
            "IDEMPOTENCY_DB_PATH", "./.idempotency.sqlite3"
        )
        self._lease_seconds = lease_seconds
        self._postgres_connect_timeout_seconds = postgres_connect_timeout_seconds
        self._postgres_lock_timeout_ms = postgres_lock_timeout_ms
        self._postgres_statement_timeout_ms = postgres_statement_timeout_ms
        self._postgres_tcp_user_timeout_ms = postgres_tcp_user_timeout_ms
        self._schema_init_wait_timeout_seconds = schema_init_wait_timeout_seconds
        self._cancel_drain_timeout_seconds = cancel_drain_timeout_seconds
        self._clock = clock
        self._schema_ready = False
        self._schema_lock = threading.Lock()
        self._background_workers: set[asyncio.Task] = set()

    @classmethod
    def from_environment(cls) -> "IdempotencyStore":
        database_url = os.getenv("DATABASE_URL") or None
        return cls(
            database_url=database_url,
            lease_seconds=float(os.getenv("IDEMPOTENCY_RECEIPT_LEASE_SECONDS", "600")),
            postgres_connect_timeout_seconds=int(
                os.getenv("IDEMPOTENCY_PG_CONNECT_TIMEOUT_SECONDS", "5")
            ),
            postgres_lock_timeout_ms=int(os.getenv("IDEMPOTENCY_PG_LOCK_TIMEOUT_MS", "5000")),
            postgres_statement_timeout_ms=int(
                os.getenv("IDEMPOTENCY_PG_STATEMENT_TIMEOUT_MS", "15000")
            ),
            postgres_tcp_user_timeout_ms=int(
                os.getenv("IDEMPOTENCY_PG_TCP_USER_TIMEOUT_MS", "30000")
            ),
            schema_init_wait_timeout_seconds=float(
                os.getenv("IDEMPOTENCY_SCHEMA_INIT_WAIT_TIMEOUT_SECONDS", "30")
            ),
            cancel_drain_timeout_seconds=float(
                os.getenv("IDEMPOTENCY_CANCEL_DRAIN_TIMEOUT_SECONDS", "20")
            ),
        )

    async def begin(
        self,
        key: str,
        operation: str,
        payload: dict[str, Any],
    ) -> BeginDecision:
        if not isinstance(key, str) or not key or len(key) > 128:
            raise ValueError("invalid idempotency key")
        owner_token = secrets.token_urlsafe(32)
        recovery_token = secrets.token_urlsafe(32)
        worker = asyncio.create_task(
            asyncio.to_thread(
                self._begin_sync,
                key,
                operation,
                request_fingerprint(operation, payload),
                owner_token,
                recovery_token,
            )
        )
        try:
            # A cancelled await must not cancel the Future proxy while the DB
            # thread can still commit a claim behind it.
            return await asyncio.shield(worker)
        except asyncio.CancelledError as cancelled:
            deadline = asyncio.get_running_loop().time() + self._cancel_drain_timeout_seconds
            completed, _ = await self._drain_cancelled_worker(
                worker,
                deadline=deadline,
                track_on_timeout=False,
            )
            if completed:
                cleanup = asyncio.create_task(self._run_thread(self._abort_sync, key, owner_token))
                await self._drain_cancelled_worker(
                    cleanup,
                    deadline=deadline,
                )
            else:
                cleanup = asyncio.create_task(
                    self._abort_late_begin(
                        worker,
                        key=key,
                        owner_token=owner_token,
                    )
                )
                self._track_background_worker(cleanup)
            raise cancelled

    async def renew(self, lease: ReceiptLease) -> ReceiptLease | None:
        """Extend a live clean claim, or return ``None`` after ownership loss."""
        self._validate_lease(lease)
        return await self._run_thread(
            self._renew_sync,
            lease.key,
            lease.owner_token,
        )

    async def mark_effect_started(
        self,
        lease: ReceiptLease,
        tool_name: str,
    ) -> None:
        self._validate_lease(lease)
        await self._run_thread(
            self._mark_effect_started_sync,
            lease.key,
            lease.owner_token,
            tool_name,
        )

    async def complete(
        self,
        lease: ReceiptLease,
        response: dict[str, Any],
    ) -> None:
        self._validate_lease(lease)
        response_json = json.dumps(
            response,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        await self._run_thread(
            self._complete_sync,
            lease.key,
            lease.owner_token,
            response_json,
        )

    async def reconcile_completed(
        self,
        key: str,
        operation: str,
        payload: dict[str, Any],
        response: dict[str, Any],
        *,
        allow_effect_started: bool = False,
    ) -> None:
        """Repair a receipt from a separately persisted canonical outcome.

        This token-free transition is intentionally narrower than ``complete``:
        callers must already have validated a durable operation outcome, while
        this store rechecks the immutable operation/payload binding under the
        receipt row lock. It is used to close the session-outcome -> receipt
        crash window without allowing an abandoned worker to rerun a handler.
        """
        if not isinstance(key, str) or not key or len(key) > 128:
            raise ValueError("invalid idempotency key")
        response_json = json.dumps(
            response,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        await self._run_thread(
            self._reconcile_completed_sync,
            key,
            operation,
            request_fingerprint(operation, payload),
            response_json,
            allow_effect_started,
        )

    async def recovery_token(
        self,
        key: str,
        operation: str,
        payload: dict[str, Any],
    ) -> str | None:
        """Return the server-generated recovery capability for an exact binding."""
        if not isinstance(key, str) or not key or len(key) > 128:
            raise ValueError("invalid idempotency key")
        return await self._run_thread(
            self._recovery_token_sync,
            key,
            operation,
            request_fingerprint(operation, payload),
        )

    async def abort(self, lease: ReceiptLease) -> bool:
        """Release a clean claim or persist ambiguity after an effect started."""
        self._validate_lease(lease)
        return await self._run_thread(
            self._abort_sync,
            lease.key,
            lease.owner_token,
        )

    async def has_effect_started(self, key: str) -> bool:
        return await self._run_thread(self._has_effect_started_sync, key)

    async def _run_thread(self, function, *args):
        """Let a bounded DB thread settle before propagating cancellation."""
        worker = asyncio.create_task(asyncio.to_thread(function, *args))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError as cancelled:
            await self._drain_cancelled_worker(worker)
            raise cancelled

    async def _abort_late_begin(
        self,
        worker: asyncio.Task,
        *,
        key: str,
        owner_token: str,
    ) -> None:
        """Abort only the captured owner after a late begin worker settles."""
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            logger.error(
                "late idempotency begin worker failed: error_type=%s",
                type(exc).__name__,
            )
        try:
            await self._run_thread(self._abort_sync, key, owner_token)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            logger.error(
                "late idempotency begin cleanup failed: error_type=%s",
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
                break

        try:
            return True, worker.result()
        except BaseException as exc:
            logger.error(
                "cancelled idempotency store worker failed: error_type=%s",
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
                    "background idempotency store worker failed: error_type=%s",
                    type(exc).__name__,
                )

        worker.add_done_callback(on_done)

    @staticmethod
    def _validate_lease(lease: ReceiptLease) -> None:
        if not isinstance(lease, ReceiptLease):
            raise ValueError("receipt lease is required")
        if not lease.key or len(lease.key) > 128:
            raise ValueError("invalid receipt lease key")
        if not lease.owner_token or len(lease.owner_token) > 256:
            raise ValueError("invalid receipt lease owner token")
        if not lease.recovery_token or len(lease.recovery_token) > 256:
            raise ValueError("invalid receipt recovery token")
        if not math.isfinite(lease.expires_at):
            raise ValueError("invalid receipt lease expiry")

    def _now(self) -> float:
        try:
            value = float(self._clock())
        except (TypeError, ValueError) as exc:
            raise ValueError("idempotency clock returned an invalid value") from exc
        if not math.isfinite(value):
            raise ValueError("idempotency clock returned an invalid value")
        return value

    def _lease_expiry(self, now: float) -> float:
        expires_at = now + self._lease_seconds
        if not math.isfinite(expires_at):
            raise ValueError("idempotency lease expiry is invalid")
        return expires_at

    def _connect(self):
        if self._database_url:
            import psycopg
            from psycopg.conninfo import conninfo_to_dict

            try:
                connection_parameters = conninfo_to_dict(self._database_url)
            except Exception:
                raise ValueError("PostgreSQL idempotency DATABASE_URL is invalid") from None
            explicit_options = connection_parameters.get("options")
            environment_options = os.getenv("PGOPTIONS", "").strip()
            service_configured = bool(
                connection_parameters.get("service") or os.getenv("PGSERVICE")
            )
            if service_configured and explicit_options is None and not environment_options:
                raise ValueError(
                    "PostgreSQL service DSNs must expose connection options "
                    "through DATABASE_URL or PGOPTIONS so idempotency safety "
                    "limits can be merged without silently discarding "
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
                tcp_user_timeout=self._postgres_tcp_user_timeout_ms,
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
            raise TimeoutError("idempotency schema initialization timed out")
        try:
            if self._schema_ready:
                return
            statement = f"""
                CREATE TABLE IF NOT EXISTS {_TABLE} (
                    idempotency_key TEXT PRIMARY KEY,
                    operation TEXT NOT NULL,
                    request_fingerprint TEXT NOT NULL,
                    state TEXT NOT NULL,
                    response_json TEXT,
                    effect_tool TEXT,
                    owner_token TEXT,
                    recovery_token TEXT,
                    lease_expires_at DOUBLE PRECISION,
                    created_at DOUBLE PRECISION NOT NULL,
                    updated_at DOUBLE PRECISION NOT NULL
                )
            """
            with self._transaction(write=True) as connection:
                if self._postgres:
                    connection.execute(
                        "SELECT pg_advisory_xact_lock(%s)",
                        (_POSTGRES_SCHEMA_LOCK_ID,),
                    )
                connection.execute(statement)
                if self._postgres:
                    connection.execute(
                        f"ALTER TABLE {_TABLE} ADD COLUMN IF NOT EXISTS owner_token TEXT"
                    )
                    connection.execute(
                        f"ALTER TABLE {_TABLE} ADD COLUMN IF NOT EXISTS recovery_token TEXT"
                    )
                    connection.execute(
                        f"ALTER TABLE {_TABLE} "
                        "ADD COLUMN IF NOT EXISTS lease_expires_at DOUBLE PRECISION"
                    )
                else:
                    columns = {
                        row[1]
                        for row in connection.execute(f"PRAGMA table_info({_TABLE})").fetchall()
                    }
                    if "owner_token" not in columns:
                        connection.execute(f"ALTER TABLE {_TABLE} ADD COLUMN owner_token TEXT")
                    if "recovery_token" not in columns:
                        connection.execute(f"ALTER TABLE {_TABLE} ADD COLUMN recovery_token TEXT")
                    if "lease_expires_at" not in columns:
                        connection.execute(
                            f"ALTER TABLE {_TABLE} ADD COLUMN lease_expires_at DOUBLE PRECISION"
                        )
            self._schema_ready = True
        finally:
            self._schema_lock.release()

    def _begin_sync(
        self,
        key: str,
        operation: str,
        request_fingerprint: str,
        owner_token: str,
        recovery_token: str,
    ) -> BeginDecision:
        self._ensure_schema()
        p = self._placeholder
        with self._transaction(write=True) as connection:
            row = self._select_receipt(connection, key, for_update=True)
            now = self._now()
            if row is None:
                expires_at = self._lease_expiry(now)
                if self._postgres:
                    cursor = connection.execute(
                        f"""
                        INSERT INTO {_TABLE}
                            (idempotency_key, operation, request_fingerprint,
                             state, owner_token, lease_expires_at,
                             recovery_token, created_at, updated_at)
                        VALUES ({p}, {p}, {p}, 'pending_v2', {p}, {p}, {p}, {p}, {p})
                        ON CONFLICT (idempotency_key) DO NOTHING
                        """,
                        (
                            key,
                            operation,
                            request_fingerprint,
                            owner_token,
                            expires_at,
                            recovery_token,
                            now,
                            now,
                        ),
                    )
                else:
                    cursor = connection.execute(
                        f"""
                        INSERT OR IGNORE INTO {_TABLE}
                            (idempotency_key, operation, request_fingerprint,
                             state, owner_token, lease_expires_at,
                             recovery_token, created_at, updated_at)
                        VALUES ({p}, {p}, {p}, 'pending_v2', {p}, {p}, {p}, {p}, {p})
                        """,
                        (
                            key,
                            operation,
                            request_fingerprint,
                            owner_token,
                            expires_at,
                            recovery_token,
                            now,
                            now,
                        ),
                    )
                if cursor.rowcount == 1:
                    return BeginDecision(
                        replayed=False,
                        lease=ReceiptLease(key, owner_token, recovery_token, expires_at),
                    )
                row = self._select_receipt(connection, key, for_update=True)
                now = self._now()

            if row is None:
                raise RuntimeError("idempotency receipt disappeared after claim")

            (
                existing_operation,
                existing_fingerprint,
                state,
                response_json,
                existing_token,
                lease_expires_at,
                existing_recovery_token,
            ) = row
            if existing_operation != operation or existing_fingerprint != request_fingerprint:
                raise IdempotencyConflictError("payload_mismatch")
            if state == "completed" and response_json:
                response = json.loads(response_json)
                if not isinstance(response, dict):
                    raise RuntimeError("completed idempotency response is not an object")
                return BeginDecision(replayed=True, response=response)
            if state == "pending":
                # A pre-v2 process may still own this row during a rolling
                # deployment. It must never be reclaimed automatically.
                raise IdempotencyConflictError("in_progress")
            if state == "pending_v2":
                if not existing_token or lease_expires_at is None or not existing_recovery_token:
                    raise IdempotencyConflictError("ambiguous")
                if float(lease_expires_at) > now:
                    raise IdempotencyConflictError("in_progress")
                expires_at = self._lease_expiry(now)
                cursor = connection.execute(
                    f"""
                    UPDATE {_TABLE}
                    SET owner_token = {p}, lease_expires_at = {p},
                        updated_at = {p}
                    WHERE idempotency_key = {p} AND state = 'pending_v2'
                      AND owner_token = {p} AND lease_expires_at <= {p}
                    """,
                    (
                        owner_token,
                        expires_at,
                        now,
                        key,
                        existing_token,
                        now,
                    ),
                )
                if cursor.rowcount != 1:
                    raise IdempotencyConflictError("in_progress")
                return BeginDecision(
                    replayed=False,
                    lease=ReceiptLease(
                        key,
                        owner_token,
                        existing_recovery_token,
                        expires_at,
                    ),
                )
            raise IdempotencyConflictError("ambiguous")

    def _select_receipt(self, connection, key: str, *, for_update: bool):
        p = self._placeholder
        suffix = " FOR UPDATE" if self._postgres and for_update else ""
        return connection.execute(
            f"""
            SELECT operation, request_fingerprint, state, response_json,
                   owner_token, lease_expires_at, recovery_token
            FROM {_TABLE} WHERE idempotency_key = {p}{suffix}
            """,
            (key,),
        ).fetchone()

    def _renew_sync(self, key: str, owner_token: str) -> ReceiptLease | None:
        self._ensure_schema()
        p = self._placeholder
        with self._transaction(write=True) as connection:
            row = self._select_receipt(connection, key, for_update=True)
            now = self._now()
            if row is None:
                return None
            state = row[2]
            existing_token = row[4]
            existing_expiry = row[5]
            recovery_token = row[6]
            if state == "effect_started_v2" and existing_token == owner_token and recovery_token:
                connection.execute(
                    f"""
                    UPDATE {_TABLE} SET updated_at = {p}
                    WHERE idempotency_key = {p}
                      AND state = 'effect_started_v2' AND owner_token = {p}
                    """,
                    (now, key, owner_token),
                )
                return ReceiptLease(
                    key,
                    owner_token,
                    recovery_token,
                    self._lease_expiry(now),
                )
            if (
                state != "pending_v2"
                or existing_token != owner_token
                or existing_expiry is None
                or not recovery_token
                or float(existing_expiry) <= now
            ):
                return None
            expires_at = max(
                float(existing_expiry),
                self._lease_expiry(now),
            )
            cursor = connection.execute(
                f"""
                UPDATE {_TABLE}
                SET lease_expires_at = {p}, updated_at = {p}
                WHERE idempotency_key = {p} AND state = 'pending_v2'
                  AND owner_token = {p} AND lease_expires_at > {p}
                """,
                (expires_at, now, key, owner_token, now),
            )
            if cursor.rowcount != 1:
                return None
            return ReceiptLease(key, owner_token, recovery_token, expires_at)

    def _mark_effect_started_sync(
        self,
        key: str,
        owner_token: str,
        tool_name: str,
    ) -> None:
        self._ensure_schema()
        p = self._placeholder
        with self._transaction(write=True) as connection:
            self._select_receipt(connection, key, for_update=True)
            now = self._now()
            cursor = connection.execute(
                f"""
                UPDATE {_TABLE}
                SET state = 'effect_started_v2', effect_tool = {p},
                    lease_expires_at = NULL, updated_at = {p}
                WHERE idempotency_key = {p} AND owner_token = {p}
                  AND (
                    (state = 'pending_v2' AND lease_expires_at > {p})
                    OR state = 'effect_started_v2'
                  )
                """,
                (tool_name, now, key, owner_token, now),
            )
            if cursor.rowcount != 1:
                raise IdempotencyConflictError("receipt_not_pending")

    def _complete_sync(
        self,
        key: str,
        owner_token: str,
        response_json: str,
    ) -> None:
        self._ensure_schema()
        p = self._placeholder
        with self._transaction(write=True) as connection:
            self._select_receipt(connection, key, for_update=True)
            now = self._now()
            cursor = connection.execute(
                f"""
                UPDATE {_TABLE}
                SET state = 'completed', response_json = {p},
                    owner_token = NULL, lease_expires_at = NULL,
                    updated_at = {p}
                WHERE idempotency_key = {p} AND owner_token = {p}
                  AND (
                    (state = 'pending_v2' AND lease_expires_at > {p})
                    OR state = 'effect_started_v2'
                  )
                """,
                (response_json, now, key, owner_token, now),
            )
            if cursor.rowcount != 1:
                raise IdempotencyConflictError("receipt_not_completable")

    def _reconcile_completed_sync(
        self,
        key: str,
        operation: str,
        request_fingerprint: str,
        response_json: str,
        allow_effect_started: bool,
    ) -> None:
        self._ensure_schema()
        p = self._placeholder
        with self._transaction(write=True) as connection:
            row = self._select_receipt(connection, key, for_update=True)
            if row is None:
                raise IdempotencyConflictError("receipt_missing")
            (
                existing_operation,
                existing_fingerprint,
                state,
                existing_response_json,
                _existing_token,
                _lease_expires_at,
                _recovery_token,
            ) = row
            if existing_operation != operation or existing_fingerprint != request_fingerprint:
                raise IdempotencyConflictError("payload_mismatch")
            if state == "completed":
                try:
                    existing_response = json.loads(existing_response_json)
                    expected_response = json.loads(response_json)
                except (TypeError, json.JSONDecodeError) as exc:
                    raise RuntimeError("completed idempotency response is invalid JSON") from exc
                if existing_response != expected_response:
                    raise RuntimeError("canonical outcome disagrees with completed receipt")
                return
            repairable_states = {"pending_v2"}
            if allow_effect_started:
                repairable_states.update({"effect_started_v2", "ambiguous"})
            if state not in repairable_states:
                # Legacy unleased rows cannot prove that their former worker
                # participated in the canonical session transition.
                raise IdempotencyConflictError("ambiguous")
            now = self._now()
            allowed_sql = (
                "('pending_v2', 'effect_started_v2', 'ambiguous')"
                if allow_effect_started
                else "('pending_v2')"
            )
            cursor = connection.execute(
                f"""
                UPDATE {_TABLE}
                SET state = 'completed', response_json = {p},
                    owner_token = NULL, lease_expires_at = NULL,
                    updated_at = {p}
                WHERE idempotency_key = {p} AND operation = {p}
                  AND request_fingerprint = {p}
                  AND state IN {allowed_sql}
                """,
                (
                    response_json,
                    now,
                    key,
                    operation,
                    request_fingerprint,
                ),
            )
            if cursor.rowcount != 1:
                raise IdempotencyConflictError("receipt_not_completable")

    def _recovery_token_sync(
        self,
        key: str,
        operation: str,
        request_fingerprint: str,
    ) -> str | None:
        self._ensure_schema()
        with self._transaction() as connection:
            row = self._select_receipt(connection, key, for_update=False)
        if row is None:
            return None
        if row[0] != operation or row[1] != request_fingerprint:
            raise IdempotencyConflictError("payload_mismatch")
        recovery_token = row[6]
        return recovery_token if isinstance(recovery_token, str) else None

    def _abort_sync(self, key: str, owner_token: str) -> bool:
        self._ensure_schema()
        p = self._placeholder
        with self._transaction(write=True) as connection:
            row = connection.execute(
                f"""
                SELECT state, owner_token FROM {_TABLE}
                WHERE idempotency_key = {p}
                {"FOR UPDATE" if self._postgres else ""}
                """,
                (key,),
            ).fetchone()
            if row is None:
                return False
            state, existing_token = row
            if existing_token != owner_token:
                return state in {
                    "effect_started",
                    "effect_started_v2",
                    "ambiguous",
                }
            if state == "pending_v2":
                cursor = connection.execute(
                    f"""
                    DELETE FROM {_TABLE}
                    WHERE idempotency_key = {p} AND state = 'pending_v2'
                      AND owner_token = {p}
                    """,
                    (key, owner_token),
                )
                if cursor.rowcount != 1:
                    return False
                return False
            if state == "effect_started_v2":
                now = self._now()
                cursor = connection.execute(
                    f"""
                    UPDATE {_TABLE}
                    SET state = 'ambiguous', owner_token = NULL,
                        lease_expires_at = NULL, updated_at = {p}
                    WHERE idempotency_key = {p}
                      AND state = 'effect_started_v2' AND owner_token = {p}
                    """,
                    (now, key, owner_token),
                )
                return cursor.rowcount == 1
            return state in {"effect_started", "ambiguous"}

    def _has_effect_started_sync(self, key: str) -> bool:
        self._ensure_schema()
        p = self._placeholder
        with self._transaction() as connection:
            row = connection.execute(
                f"SELECT state FROM {_TABLE} WHERE idempotency_key = {p}",
                (key,),
            ).fetchone()
        return bool(row and row[0] in {"effect_started", "effect_started_v2", "ambiguous"})


async def abort_idempotency_claim(
    store: IdempotencyStore,
    lease: ReceiptLease,
) -> bool:
    """Bound exact-lease cleanup after the owning request was cancelled."""
    cleanup = asyncio.create_task(store.abort(lease))
    deadline = asyncio.get_running_loop().time() + store._cancel_drain_timeout_seconds
    completed, result = await store._drain_cancelled_worker(
        cleanup,
        deadline=deadline,
    )
    if completed and isinstance(result, bool):
        return result
    # The exact-token abort remains strongly referenced in the background.
    # Until it settles, callers must conservatively assume a durable effect.
    return True


request_idempotency = IdempotencyStore.from_environment()
