"""Durable Adaptive-session aggregates with cross-worker fencing.

Full private Adaptive state is revalidated before every durable write. SQLite
is used locally and PostgreSQL is selected when DATABASE_URL is configured.
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

from models.adaptive_session import AdaptiveSessionAggregate


load_dotenv(Path(__file__).parent.parent / ".env")
logger = logging.getLogger(__name__)

_TABLE = "studyloop_adaptive_sessions"
_SCHEMA_VERSION = 1
_POSTGRES_SCHEMA_LOCK_ID = 0x5354554441444150  # ASCII "STUDYADAP"
_AGGREGATE_TO_ROW_STATUS = {"active": "active", "completed": "done"}
_ROW_TO_AGGREGATE_STATUS = {"active": "active", "done": "completed"}


class AdaptiveSessionAlreadyExistsError(RuntimeError):
    """A different create request already owns the supplied session id."""


class AdaptiveSessionCapacityError(RuntimeError):
    """Capacity is full of live sessions that cannot be evicted safely."""


class AdaptiveSessionPayloadTooLargeError(ValueError):
    """The versioned JSON envelope exceeds the configured payload limit."""


class AdaptiveSessionCorruptError(RuntimeError):
    """Durable state failed schema or row/envelope consistency validation."""


class AdaptiveSessionStartConflictError(RuntimeError):
    """A start idempotency key was reused with a different request."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class StoredAdaptiveSession:
    aggregate: AdaptiveSessionAggregate
    revision: int
    created_at: float
    updated_at: float
    expires_at: float
    busy: bool
    expired: bool


@dataclass(frozen=True, slots=True)
class AdaptiveSessionCreateResult:
    record: StoredAdaptiveSession
    created: bool


@dataclass(frozen=True, slots=True)
class AdaptiveSessionClaim:
    claimed: bool
    reason: str | None = None
    token: str | None = None
    record: StoredAdaptiveSession | None = None


class AdaptiveSessionStore:
    """Versioned JSON session store for the Adaptive API.

    Mutations after creation require a short-lived, opaque claim token. Every
    write is fenced by that token, its unexpired lease, the session TTL, and the
    immutable hash supplied at creation.
    """

    def __init__(
        self,
        *,
        database_url: str | None = None,
        sqlite_path: str | None = None,
        ttl_seconds: float = 60 * 60,
        operation_lease_seconds: float = 10 * 60,
        max_count: int = 200,
        max_payload_bytes: int = 2 * 1024 * 1024,
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
        if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if not math.isfinite(operation_lease_seconds) or operation_lease_seconds <= 0:
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
        self._sqlite_path = (
            sqlite_path
            or os.getenv("ADAPTIVE_SESSION_DB_PATH")
            or os.getenv("IDEMPOTENCY_DB_PATH", "./.idempotency.sqlite3")
        )
        self._ttl_seconds = ttl_seconds
        self._operation_lease_seconds = operation_lease_seconds
        self._max_count = max_count
        self._max_payload_bytes = max_payload_bytes
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
    def from_environment(cls) -> "AdaptiveSessionStore":
        return cls(
            database_url=os.getenv("DATABASE_URL") or None,
            ttl_seconds=float(os.getenv("ADAPTIVE_SESSION_TTL_SECONDS", "3600")),
            operation_lease_seconds=float(
                os.getenv("ADAPTIVE_SESSION_OPERATION_LEASE_SECONDS", "600")
            ),
            max_count=int(os.getenv("ADAPTIVE_SESSION_MAX_COUNT", "200")),
            max_payload_bytes=int(
                os.getenv(
                    "ADAPTIVE_SESSION_MAX_PAYLOAD_BYTES",
                    str(2 * 1024 * 1024),
                )
            ),
            postgres_connect_timeout_seconds=int(
                os.getenv("ADAPTIVE_SESSION_PG_CONNECT_TIMEOUT_SECONDS", "5")
            ),
            postgres_lock_timeout_ms=int(os.getenv("ADAPTIVE_SESSION_PG_LOCK_TIMEOUT_MS", "5000")),
            postgres_statement_timeout_ms=int(
                os.getenv("ADAPTIVE_SESSION_PG_STATEMENT_TIMEOUT_MS", "15000")
            ),
            postgres_tcp_user_timeout_ms=int(
                os.getenv("ADAPTIVE_SESSION_PG_TCP_USER_TIMEOUT_MS", "30000")
            ),
            schema_init_wait_timeout_seconds=float(
                os.getenv("ADAPTIVE_SESSION_SCHEMA_INIT_WAIT_TIMEOUT_SECONDS", "30")
            ),
            cancel_drain_timeout_seconds=float(
                os.getenv("ADAPTIVE_SESSION_CANCEL_DRAIN_TIMEOUT_SECONDS", "20")
            ),
        )

    async def create(
        self,
        aggregate: AdaptiveSessionAggregate,
        *,
        start_key: str | None = None,
        start_request: dict[str, Any] | None = None,
    ) -> AdaptiveSessionCreateResult:
        validated, payload_json, immutable_hash, row_status = self._prepare_aggregate(aggregate)
        key_hash, request_hash = self._start_hashes(start_key, start_request)
        return await self._run_thread(
            self._create_sync,
            validated.adaptive_session_id,
            row_status,
            payload_json,
            immutable_hash,
            key_hash,
            request_hash,
        )

    async def find_start(
        self,
        start_key: str,
        start_request: dict[str, Any],
    ) -> StoredAdaptiveSession | None:
        key_hash, request_hash = self._start_hashes(start_key, start_request)
        if key_hash is None or request_hash is None:
            raise ValueError("start_key is required")
        return await self._run_thread(
            self._find_start_sync,
            key_hash,
            request_hash,
        )

    async def inspect(self, session_id: str) -> StoredAdaptiveSession | None:
        if not session_id:
            return None
        return await self._run_thread(self._inspect_sync, session_id)

    async def claim(self, session_id: str, operation: str) -> AdaptiveSessionClaim:
        if not session_id:
            return AdaptiveSessionClaim(claimed=False, reason="missing")
        if not isinstance(operation, str) or not operation or len(operation) > 64:
            raise ValueError("invalid adaptive session operation")

        token = secrets.token_urlsafe(32)
        worker = asyncio.create_task(
            asyncio.to_thread(
                self._claim_sync,
                session_id,
                operation,
                token,
            )
        )
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError as cancelled:
            deadline = asyncio.get_running_loop().time() + self._cancel_drain_timeout_seconds
            completed, _ = await self._drain_cancelled_worker(
                worker,
                deadline=deadline,
                track_on_timeout=False,
            )
            if completed:
                cleanup = asyncio.create_task(
                    self._run_thread(self._release_sync, session_id, token)
                )
                await self._drain_cancelled_worker(
                    cleanup,
                    deadline=deadline,
                )
            else:
                cleanup = asyncio.create_task(self._release_late_claim(worker, session_id, token))
                self._track_background_worker(cleanup)
            raise cancelled

    async def checkpoint(
        self,
        session_id: str,
        claim_token: str,
        aggregate: AdaptiveSessionAggregate,
        expected_revision: int,
    ) -> StoredAdaptiveSession | None:
        self._validate_claim_token(claim_token)
        self._validate_expected_revision(expected_revision)
        validated, payload_json, immutable_hash, row_status = self._prepare_aggregate(aggregate)
        if validated.adaptive_session_id != session_id:
            raise ValueError("adaptive session id does not match aggregate")
        return await self._run_thread(
            self._checkpoint_sync,
            session_id,
            claim_token,
            expected_revision,
            row_status,
            payload_json,
            immutable_hash,
        )

    async def complete(
        self,
        session_id: str,
        claim_token: str,
        aggregate: AdaptiveSessionAggregate,
        expected_revision: int,
    ) -> StoredAdaptiveSession | None:
        self._validate_claim_token(claim_token)
        self._validate_expected_revision(expected_revision)
        validated, payload_json, immutable_hash, row_status = self._prepare_aggregate(aggregate)
        if validated.adaptive_session_id != session_id:
            raise ValueError("adaptive session id does not match aggregate")
        return await self._run_thread(
            self._complete_sync,
            session_id,
            claim_token,
            expected_revision,
            row_status,
            payload_json,
            immutable_hash,
        )

    async def release(self, session_id: str, claim_token: str) -> bool:
        if not session_id:
            return False
        self._validate_claim_token(claim_token)
        return await self._run_thread(
            self._release_sync,
            session_id,
            claim_token,
        )

    async def _run_thread(self, function, *args):
        worker = asyncio.create_task(asyncio.to_thread(function, *args))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError as cancelled:
            await self._drain_cancelled_worker(worker)
            raise cancelled

    async def _release_late_claim(
        self,
        worker: asyncio.Task,
        session_id: str,
        token: str,
    ) -> None:
        """Release only the captured token after a late claim worker settles."""
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            logger.error(
                "late Adaptive claim worker failed: error_type=%s",
                type(exc).__name__,
            )
        try:
            await self._run_thread(self._release_sync, session_id, token)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            logger.error(
                "late Adaptive claim cleanup failed: error_type=%s",
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
                "cancelled Adaptive store worker failed: error_type=%s",
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
                    "background Adaptive store worker failed: error_type=%s",
                    type(exc).__name__,
                )

        worker.add_done_callback(on_done)

    @staticmethod
    def _validate_claim_token(claim_token: str) -> None:
        if not isinstance(claim_token, str) or not claim_token or len(claim_token) > 256:
            raise ValueError("invalid adaptive session claim token")

    @staticmethod
    def _validate_expected_revision(expected_revision: int) -> None:
        if (
            not isinstance(expected_revision, int)
            or isinstance(expected_revision, bool)
            or expected_revision < 1
        ):
            raise ValueError("invalid adaptive session expected revision")

    def _prepare_aggregate(
        self,
        aggregate: AdaptiveSessionAggregate,
    ) -> tuple[AdaptiveSessionAggregate, str, str, str]:
        if not isinstance(aggregate, AdaptiveSessionAggregate):
            raise ValueError("adaptive session aggregate is required")
        try:
            # Revalidate a detached JSON copy. Nested models can otherwise be
            # mutated after construction without assignment validation.
            validated = AdaptiveSessionAggregate.validate_for_persistence(
                aggregate.model_dump(mode="json")
            )
            payload_json = json.dumps(
                validated.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValidationError, ValueError) as exc:
            raise ValueError("invalid adaptive session aggregate") from exc
        payload_bytes = len(payload_json.encode("utf-8"))
        if payload_bytes > self._max_payload_bytes:
            raise AdaptiveSessionPayloadTooLargeError(
                f"adaptive session payload is {payload_bytes} bytes; "
                f"limit is {self._max_payload_bytes}"
            )
        immutable_hash = self._immutable_hash(validated)
        try:
            row_status = _AGGREGATE_TO_ROW_STATUS[validated.status]
        except KeyError as exc:
            raise ValueError("invalid adaptive session status") from exc
        return validated, payload_json, immutable_hash, row_status

    @staticmethod
    def _immutable_hash(aggregate: AdaptiveSessionAggregate) -> str:
        canonical = json.dumps(
            {
                "adaptive_session_id": aggregate.adaptive_session_id,
                "user_id": aggregate.user_id,
                "document_id": aggregate.document_id,
                "goal": aggregate.goal,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _start_hashes(
        start_key: str | None,
        start_request: dict[str, Any] | None,
    ) -> tuple[str | None, str | None]:
        if start_key is None:
            if start_request is not None:
                raise ValueError("start_key is required with start_request")
            return None, None
        if not isinstance(start_key, str) or not start_key or len(start_key) > 1024:
            raise ValueError("invalid adaptive start key")
        if not isinstance(start_request, dict):
            raise ValueError("start_request is required with start_key")
        key_hash = hashlib.sha256(start_key.encode("utf-8")).hexdigest()
        try:
            canonical = json.dumps(
                start_request,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("adaptive start request must be valid JSON") from exc
        request_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return key_hash, request_hash

    def _connect(self):
        if self._database_url:
            import psycopg
            from psycopg.conninfo import conninfo_to_dict

            try:
                connection_parameters = conninfo_to_dict(self._database_url)
            except Exception:
                raise ValueError("PostgreSQL Adaptive session DATABASE_URL is invalid") from None
            explicit_options = connection_parameters.get("options")
            environment_options = os.getenv("PGOPTIONS", "").strip()
            service_configured = bool(
                connection_parameters.get("service") or os.getenv("PGSERVICE")
            )
            if service_configured and explicit_options is None and not environment_options:
                raise ValueError(
                    "PostgreSQL service DSNs must expose connection options "
                    "through DATABASE_URL or PGOPTIONS so Adaptive session "
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

            connection = psycopg.connect(
                self._database_url,
                connect_timeout=self._postgres_connect_timeout_seconds,
                tcp_user_timeout=self._postgres_tcp_user_timeout_ms,
                options=f"{existing_options} {bounded_options}".strip(),
            )
            return connection
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

    def _now(self) -> float:
        try:
            value = float(self._clock())
        except (TypeError, ValueError) as exc:
            raise ValueError("adaptive session clock returned an invalid value") from exc
        if not math.isfinite(value):
            raise ValueError("adaptive session clock returned an invalid value")
        return value

    def _claimed_expires_at(self, now: float) -> float:
        # A valid claim must never outlive the session it is allowed to commit.
        return now + max(self._ttl_seconds, self._operation_lease_seconds)

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        acquired = self._schema_lock.acquire(timeout=self._schema_init_wait_timeout_seconds)
        if not acquired:
            raise TimeoutError("Adaptive session schema initialization timed out")
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
                        schema_version INTEGER NOT NULL
                            CHECK (schema_version = 1),
                        payload_json TEXT NOT NULL,
                        status TEXT NOT NULL
                            CHECK (status IN ('active', 'done')),
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
                            (start_key_hash IS NOT NULL
                             AND start_request_hash IS NOT NULL)
                        ),
                        CHECK (
                            (claim_token IS NULL AND claim_operation IS NULL
                             AND claim_expires_at IS NULL)
                            OR
                            (claim_token IS NOT NULL
                             AND claim_operation IS NOT NULL
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
    ) -> AdaptiveSessionCreateResult:
        self._ensure_schema()
        p = self._placeholder
        with self._transaction(write=True) as connection:
            self._lock_capacity(connection)
            now = self._now()
            if start_key_hash is not None:
                row = self._select_by_start_key(connection, start_key_hash)
                if row is not None:
                    self._validate_start_request(row, start_request_hash)
                    return AdaptiveSessionCreateResult(
                        record=self._row_to_record(row, now),
                        created=False,
                    )

            existing = connection.execute(
                f"SELECT session_id FROM {_TABLE} WHERE session_id = {p}",
                (session_id,),
            ).fetchone()
            if existing is not None:
                raise AdaptiveSessionAlreadyExistsError(session_id)

            self._make_capacity(connection, now)
            expires_at = now + self._ttl_seconds
            connection.execute(
                f"""
                INSERT INTO {_TABLE} (
                    session_id, schema_version, payload_json, status, revision,
                    immutable_hash, start_key_hash, start_request_hash,
                    created_at, updated_at, expires_at
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
                raise RuntimeError("adaptive session disappeared after create")
            return AdaptiveSessionCreateResult(
                record=self._row_to_record(row, now),
                created=True,
            )

    def _find_start_sync(
        self,
        start_key_hash: str,
        start_request_hash: str,
    ) -> StoredAdaptiveSession | None:
        self._ensure_schema()
        with self._transaction() as connection:
            row = self._select_by_start_key(connection, start_key_hash)
            now = self._now()
        if row is None:
            return None
        self._validate_start_request(row, start_request_hash)
        return self._row_to_record(row, now)

    def _inspect_sync(
        self,
        session_id: str,
    ) -> StoredAdaptiveSession | None:
        self._ensure_schema()
        with self._transaction() as connection:
            row = self._select_by_id(connection, session_id)
            now = self._now()
        return None if row is None else self._row_to_record(row, now)

    def _claim_sync(
        self,
        session_id: str,
        operation: str,
        token: str,
    ) -> AdaptiveSessionClaim:
        self._ensure_schema()
        p = self._placeholder
        with self._transaction(write=True) as connection:
            row = self._select_by_id(connection, session_id, for_update=True)
            now = self._now()
            if row is None:
                return AdaptiveSessionClaim(claimed=False, reason="missing")
            # Validate the durable aggregate before making any status decision.
            # Even a terminal or expired corrupt row must fail closed.
            self._row_to_record(row, now)
            if float(row[8]) <= now:
                return AdaptiveSessionClaim(claimed=False, reason="expired")
            if row[3] == "done":
                return AdaptiveSessionClaim(claimed=False, reason="done")
            if row[9] is not None and row[10] is not None and float(row[10]) > now:
                return AdaptiveSessionClaim(claimed=False, reason="in_progress")

            lease_expires = now + self._operation_lease_seconds
            expires_at = self._claimed_expires_at(now)
            cursor = connection.execute(
                f"""
                UPDATE {_TABLE}
                SET claim_token = {p}, claim_operation = {p},
                    claim_expires_at = {p}, updated_at = {p}, expires_at = {p}
                WHERE session_id = {p} AND status = 'active'
                  AND expires_at > {p}
                  AND (claim_token IS NULL OR claim_expires_at <= {p})
                """,
                (
                    token,
                    operation,
                    lease_expires,
                    now,
                    expires_at,
                    session_id,
                    now,
                    now,
                ),
            )
            if cursor.rowcount == 1:
                row = self._select_by_id(connection, session_id)
                if row is None:
                    raise RuntimeError("adaptive session disappeared after claim")
                return AdaptiveSessionClaim(
                    claimed=True,
                    token=token,
                    record=self._row_to_record(row, now),
                )
            return AdaptiveSessionClaim(claimed=False, reason="in_progress")

    def _checkpoint_sync(
        self,
        session_id: str,
        claim_token: str,
        expected_revision: int,
        status: str,
        payload_json: str,
        immutable_hash: str,
    ) -> StoredAdaptiveSession | None:
        self._ensure_schema()
        p = self._placeholder
        with self._transaction(write=True) as connection:
            row = self._select_by_id(connection, session_id, for_update=True)
            now = self._now()
            if row is None:
                return None
            self._row_to_record(row, now)
            lease_expires = now + self._operation_lease_seconds
            expires_at = self._claimed_expires_at(now)
            cursor = connection.execute(
                f"""
                UPDATE {_TABLE}
                SET payload_json = {p}, status = {p}, revision = revision + 1,
                    updated_at = {p}, claim_expires_at = {p}, expires_at = {p}
                WHERE session_id = {p} AND claim_token = {p}
                  AND claim_expires_at > {p} AND expires_at > {p}
                  AND immutable_hash = {p} AND revision = {p}
                  AND (status != 'done' OR {p} = 'done')
                """,
                (
                    payload_json,
                    status,
                    now,
                    lease_expires,
                    expires_at,
                    session_id,
                    claim_token,
                    now,
                    now,
                    immutable_hash,
                    expected_revision,
                    status,
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
        expected_revision: int,
        status: str,
        payload_json: str,
        immutable_hash: str,
    ) -> StoredAdaptiveSession | None:
        self._ensure_schema()
        p = self._placeholder
        with self._transaction(write=True) as connection:
            row = self._select_by_id(connection, session_id, for_update=True)
            now = self._now()
            if row is None:
                return None
            self._row_to_record(row, now)
            expires_at = now + self._ttl_seconds
            cursor = connection.execute(
                f"""
                UPDATE {_TABLE}
                SET payload_json = {p}, status = {p}, revision = revision + 1,
                    updated_at = {p}, expires_at = {p}, claim_token = NULL,
                    claim_operation = NULL, claim_expires_at = NULL
                WHERE session_id = {p} AND claim_token = {p}
                  AND claim_expires_at > {p} AND expires_at > {p}
                  AND immutable_hash = {p} AND revision = {p}
                  AND (status != 'done' OR {p} = 'done')
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
                    expected_revision,
                    status,
                ),
            )
            if cursor.rowcount != 1:
                return None
            row = self._select_by_id(connection, session_id)
            return None if row is None else self._row_to_record(row, now)

    def _release_sync(self, session_id: str, claim_token: str) -> bool:
        self._ensure_schema()
        p = self._placeholder
        with self._transaction(write=True) as connection:
            self._select_by_id(connection, session_id, for_update=True)
            now = self._now()
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

        # TTL cleanup is separate from capacity eviction. An expired row with a
        # still-live claim remains fenced until that claim itself expires.
        connection.execute(
            f"""
            DELETE FROM {_TABLE}
            WHERE expires_at <= {p}
              AND (claim_token IS NULL OR claim_expires_at <= {p})
            """,
            (now, now),
        )
        count = connection.execute(f"SELECT COUNT(*) FROM {_TABLE}").fetchone()[0]
        required = int(count) - self._max_count + 1
        if required <= 0:
            return

        rows = connection.execute(
            f"""
            SELECT session_id FROM {_TABLE}
            WHERE status = 'done'
              AND (claim_token IS NULL OR claim_expires_at <= {p})
            ORDER BY updated_at ASC, session_id ASC
            """,
            (now,),
        ).fetchall()
        if len(rows) < required:
            raise AdaptiveSessionCapacityError("all adaptive session capacity is active")
        for (session_id,) in rows[:required]:
            connection.execute(
                f"DELETE FROM {_TABLE} WHERE session_id = {p}",
                (session_id,),
            )

    def _select_by_id(
        self,
        connection,
        session_id: str,
        *,
        for_update: bool = False,
    ):
        p = self._placeholder
        lock_clause = " FOR UPDATE" if self._postgres and for_update else ""
        return connection.execute(
            f"""
            SELECT session_id, schema_version, payload_json, status, revision,
                   immutable_hash, created_at, updated_at, expires_at,
                   claim_token, claim_expires_at, start_request_hash
            FROM {_TABLE} WHERE session_id = {p}{lock_clause}
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
            raise AdaptiveSessionStartConflictError("payload_mismatch")

    @staticmethod
    def _decode_payload(payload_json: str) -> AdaptiveSessionAggregate:
        try:
            raw = json.loads(payload_json)
            if not isinstance(raw, dict):
                raise TypeError("adaptive session payload is not a JSON object")
            return AdaptiveSessionAggregate.validate_for_persistence(raw)
        except (
            TypeError,
            json.JSONDecodeError,
            ValidationError,
            ValueError,
        ) as exc:
            raise AdaptiveSessionCorruptError("invalid durable adaptive session payload") from exc

    def _row_to_record(self, row, now: float) -> StoredAdaptiveSession:
        if row[1] != _SCHEMA_VERSION:
            raise AdaptiveSessionCorruptError("unsupported adaptive session schema version")
        if row[3] not in _ROW_TO_AGGREGATE_STATUS:
            raise AdaptiveSessionCorruptError("invalid adaptive session status")
        immutable_hash = row[5]
        if not isinstance(immutable_hash, str) or not immutable_hash:
            raise AdaptiveSessionCorruptError("invalid adaptive session immutable hash")
        aggregate = self._decode_payload(row[2])
        if (
            aggregate.adaptive_session_id != row[0]
            or _AGGREGATE_TO_ROW_STATUS.get(aggregate.status) != row[3]
            or self._immutable_hash(aggregate) != immutable_hash
        ):
            raise AdaptiveSessionCorruptError("adaptive session row does not match payload")
        claim_token = row[9]
        claim_expires_at = row[10]
        busy = bool(
            claim_token is not None
            and claim_expires_at is not None
            and float(claim_expires_at) > now
        )
        return StoredAdaptiveSession(
            aggregate=aggregate,
            revision=int(row[4]),
            created_at=float(row[6]),
            updated_at=float(row[7]),
            expires_at=float(row[8]),
            busy=busy,
            expired=float(row[8]) <= now,
        )


adaptive_sessions = AdaptiveSessionStore.from_environment()
