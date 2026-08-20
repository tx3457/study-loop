"""Durable, lease-fenced pause snapshots for the Autonomous HITL loop.

The store persists JSON wire data rather than Python objects. A continuation
owns a short-lived database lease identified by an opaque fencing token. A
clean abandoned lease may be reclaimed, while a lease that crossed the
``progress_started`` barrier becomes a durable ambiguous tombstone and is never
replayed automatically.

Successful continuations are journaled on the predecessor conversation before
an HTTP response is returned. A terminal continuation leaves a completed
tombstone; a pause-to-pause handoff atomically stores that tombstone and the new
pause snapshot.
"""

from __future__ import annotations

import asyncio
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


load_dotenv(Path(__file__).parent.parent / ".env")
logger = logging.getLogger(__name__)

_TABLE = "studyloop_autonomous_sessions"
_PHYSICAL_STATES = {"paused", "in_flight"}
_VISIBLE_STATES = _PHYSICAL_STATES | {"completed", "ambiguous"}
_OUTCOME_SCHEMA_VERSION = 1
_OUTCOME_TOMBSTONE_TOKEN = "__studyloop_outcome_tombstone__"
_POSTGRES_SCHEMA_LOCK_ID = 0x53545544594C4F4F


class SessionAlreadyExistsError(RuntimeError):
    """A pause snapshot already exists for the conversation ID."""


class SessionCapacityError(RuntimeError):
    """No recoverable pause snapshot can be evicted for new capacity."""


class SessionPayloadTooLargeError(ValueError):
    """A serialized snapshot or outcome exceeds the storage boundary."""


@dataclass(frozen=True, slots=True)
class SessionInspection:
    state: str
    payload: dict[str, Any]
    created_at: float
    expires_at: float
    claimed_at: float | None
    progress_started: bool
    claim_expires_at: float | None = None
    continue_fingerprint: str | None = None
    outcome: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class SessionClaim:
    claimed: bool
    reason: str | None = None
    claim_token: str | None = None
    payload: dict[str, Any] | None = None
    outcome: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class SessionDiscardResult:
    discarded: bool
    reason: str


class AutonomousSessionStore:
    """SQLite/PostgreSQL store for versioned Autonomous pause snapshots."""

    def __init__(
        self,
        *,
        database_url: str | None = None,
        sqlite_path: str | None = None,
        ttl_seconds: float = 3600,
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
        if not callable(clock):
            raise TypeError("clock must be callable")

        self._database_url = database_url
        self._sqlite_path = (
            sqlite_path
            or os.getenv("AUTONOMOUS_SESSION_DB_PATH")
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
    def from_environment(cls) -> "AutonomousSessionStore":
        return cls(
            database_url=os.getenv("DATABASE_URL") or None,
            ttl_seconds=float(os.getenv("AUTONOMOUS_SESSION_TTL_SECONDS", "3600")),
            operation_lease_seconds=float(
                os.getenv("AUTONOMOUS_SESSION_OPERATION_LEASE_SECONDS", "600")
            ),
            max_count=int(os.getenv("AUTONOMOUS_SESSION_MAX_COUNT", "200")),
            max_payload_bytes=int(
                os.getenv("AUTONOMOUS_SESSION_MAX_PAYLOAD_BYTES", str(2 * 1024 * 1024))
            ),
            postgres_connect_timeout_seconds=int(
                os.getenv("AUTONOMOUS_SESSION_PG_CONNECT_TIMEOUT_SECONDS", "5")
            ),
            postgres_lock_timeout_ms=int(
                os.getenv("AUTONOMOUS_SESSION_PG_LOCK_TIMEOUT_MS", "5000")
            ),
            postgres_statement_timeout_ms=int(
                os.getenv("AUTONOMOUS_SESSION_PG_STATEMENT_TIMEOUT_MS", "15000")
            ),
            postgres_tcp_user_timeout_ms=int(
                os.getenv("AUTONOMOUS_SESSION_PG_TCP_USER_TIMEOUT_MS", "30000")
            ),
            schema_init_wait_timeout_seconds=float(
                os.getenv(
                    "AUTONOMOUS_SESSION_SCHEMA_INIT_WAIT_TIMEOUT_SECONDS",
                    "30",
                )
            ),
            cancel_drain_timeout_seconds=float(
                os.getenv("AUTONOMOUS_SESSION_CANCEL_DRAIN_TIMEOUT_SECONDS", "20")
            ),
        )

    async def save(self, conversation_id: str, payload: dict[str, Any]) -> None:
        self._validate_conversation_id(conversation_id)
        payload_json = self._serialize_json_object(payload, "session payload")
        await self._run_thread(self._save_sync, conversation_id, payload_json)

    async def handoff(
        self,
        conversation_id: str,
        claim_token: str,
        new_conversation_id: str,
        payload: dict[str, Any],
        response: dict[str, Any],
    ) -> bool:
        """Atomically journal this result and insert its next paused snapshot."""
        self._validate_conversation_id(conversation_id)
        self._validate_conversation_id(new_conversation_id)
        self._validate_claim_token(claim_token)
        if conversation_id == new_conversation_id:
            raise ValueError("handoff requires a new conversation ID")
        payload_json = self._serialize_json_object(payload, "session payload")
        outcome_json = self._serialize_completed_outcome(response)
        return await self._run_thread(
            self._handoff_sync,
            conversation_id,
            claim_token,
            new_conversation_id,
            payload_json,
            outcome_json,
        )

    async def finish(
        self,
        conversation_id: str,
        claim_token: str,
        response: dict[str, Any],
    ) -> bool:
        """Fence and persist a terminal continuation response for replay."""
        self._validate_conversation_id(conversation_id)
        self._validate_claim_token(claim_token)
        return await self._run_thread(
            self._finish_sync,
            conversation_id,
            claim_token,
            self._serialize_completed_outcome(response),
        )

    async def inspect(self, conversation_id: str) -> SessionInspection | None:
        if not conversation_id:
            return None
        return await self._run_thread(self._inspect_sync, conversation_id)

    async def status(self, conversation_id: str) -> str | None:
        if not conversation_id:
            return None
        return await self._run_thread(self._status_sync, conversation_id)

    async def claim(self, conversation_id: str, continue_fingerprint: str) -> SessionClaim:
        """Claim a pause, reclaim a clean stale lease, or replay its outcome."""
        self._validate_conversation_id(conversation_id)
        self._validate_fingerprint(continue_fingerprint)
        claim_token = secrets.token_urlsafe(32)
        worker = asyncio.create_task(
            asyncio.to_thread(
                self._claim_sync,
                conversation_id,
                continue_fingerprint,
                claim_token,
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
                    self._run_thread(
                        self._cancel_sync,
                        conversation_id,
                        claim_token,
                    )
                )
                await self._drain_cancelled_worker(
                    cleanup,
                    deadline=deadline,
                )
            else:
                cleanup = asyncio.create_task(
                    self._release_late_claim(worker, conversation_id, claim_token)
                )
                self._track_background_worker(cleanup)
            raise cancelled

    async def renew(self, conversation_id: str, claim_token: str) -> bool:
        self._validate_claim_token(claim_token)
        return await self._run_thread(self._renew_sync, conversation_id, claim_token)

    async def mark_progress(self, conversation_id: str, claim_token: str) -> bool:
        self._validate_claim_token(claim_token)
        return await self._run_thread(self._mark_progress_sync, conversation_id, claim_token)

    async def cancel(self, conversation_id: str, claim_token: str) -> bool:
        """Return only a live, clean claim to its original paused state."""
        self._validate_claim_token(claim_token)
        return await self._run_thread(self._cancel_sync, conversation_id, claim_token)

    async def release(self, conversation_id: str, claim_token: str) -> bool:
        """Backward-compatible alias for :meth:`cancel`."""
        return await self.cancel(conversation_id, claim_token)

    async def discard_paused(self, conversation_id: str) -> SessionDiscardResult:
        """Delete only an unclaimed pause; never cancel a running owner."""
        self._validate_conversation_id(conversation_id)
        return await self._run_thread(self._discard_paused_sync, conversation_id)

    async def consume(self, conversation_id: str, claim_token: str) -> bool:
        """Delete an invalid claimed snapshot while its lease is still live."""
        self._validate_claim_token(claim_token)
        return await self._run_thread(self._consume_sync, conversation_id, claim_token)

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
        conversation_id: str,
        claim_token: str,
    ) -> None:
        """Cancel only the captured token after a late claim worker settles."""
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            logger.error(
                "late Autonomous claim worker failed: error_type=%s",
                type(exc).__name__,
            )
        try:
            await self._run_thread(
                self._cancel_sync,
                conversation_id,
                claim_token,
            )
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            logger.error(
                "late Autonomous claim cleanup failed: error_type=%s",
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
                "cancelled Autonomous store worker failed: error_type=%s",
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
                    "background Autonomous store worker failed: error_type=%s",
                    type(exc).__name__,
                )

        worker.add_done_callback(on_done)

    @staticmethod
    def _validate_conversation_id(conversation_id: str) -> None:
        if (
            not isinstance(conversation_id, str)
            or not conversation_id
            or len(conversation_id) > 128
        ):
            raise ValueError("invalid conversation_id")

    @staticmethod
    def _validate_claim_token(claim_token: str) -> None:
        if not isinstance(claim_token, str) or not claim_token or len(claim_token) > 256:
            raise ValueError("invalid Autonomous session claim token")

    @staticmethod
    def _validate_fingerprint(continue_fingerprint: str) -> None:
        if (
            not isinstance(continue_fingerprint, str)
            or len(continue_fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in continue_fingerprint)
        ):
            raise ValueError("continue_fingerprint must be a lowercase SHA-256 hex digest")

    def _serialize_json_object(self, payload: dict[str, Any], label: str) -> str:
        if not isinstance(payload, dict):
            raise TypeError(f"{label} must be a JSON object")
        try:
            payload_json = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{label} must be valid JSON") from exc
        payload_bytes = len(payload_json.encode("utf-8"))
        if payload_bytes > self._max_payload_bytes:
            raise SessionPayloadTooLargeError(
                f"{label} is {payload_bytes} bytes; limit is {self._max_payload_bytes}"
            )
        return payload_json

    def _serialize_completed_outcome(self, response: dict[str, Any]) -> str:
        if not isinstance(response, dict):
            raise TypeError("session response must be a JSON object")
        return self._serialize_json_object(
            {
                "schema_version": _OUTCOME_SCHEMA_VERSION,
                "kind": "completed",
                "response": response,
            },
            "session outcome",
        )

    @staticmethod
    def _ambiguous_outcome_json(reason: str = "stale_progress") -> str:
        return json.dumps(
            {
                "schema_version": _OUTCOME_SCHEMA_VERSION,
                "kind": "ambiguous",
                "reason": reason,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _decode_outcome(outcome_json: str) -> tuple[str, dict[str, Any] | None]:
        try:
            envelope = json.loads(outcome_json)
        except (TypeError, json.JSONDecodeError):
            return "ambiguous", None
        if not isinstance(envelope, dict) or envelope.get("schema_version") != 1:
            return "ambiguous", None
        if envelope.get("kind") == "completed" and isinstance(envelope.get("response"), dict):
            return "completed", envelope["response"]
        return "ambiguous", None

    def _now(self) -> float:
        try:
            now = float(self._clock())
        except (TypeError, ValueError) as exc:
            raise ValueError("Autonomous session clock returned an invalid value") from exc
        if not math.isfinite(now) or now < 0:
            raise ValueError("Autonomous session clock returned an invalid value")
        return now

    def _claimed_expires_at(self, now: float) -> float:
        return now + max(self._ttl_seconds, self._operation_lease_seconds)

    def _connect(self):
        if self._database_url:
            import psycopg
            from psycopg.conninfo import conninfo_to_dict

            try:
                connection_parameters = conninfo_to_dict(self._database_url)
            except Exception:
                raise ValueError("PostgreSQL Autonomous session DATABASE_URL is invalid") from None
            explicit_options = connection_parameters.get("options")
            environment_options = os.getenv("PGOPTIONS", "").strip()
            service_configured = bool(
                connection_parameters.get("service") or os.getenv("PGSERVICE")
            )
            if service_configured and explicit_options is None and not environment_options:
                raise ValueError(
                    "PostgreSQL service DSNs must expose connection options "
                    "through DATABASE_URL or PGOPTIONS so Autonomous session "
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
    def operation_lease_seconds(self) -> float:
        """Public heartbeat budget for the continuation coordinator."""
        return self._operation_lease_seconds

    @property
    def _placeholder(self) -> str:
        return "%s" if self._postgres else "?"

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        acquired = self._schema_lock.acquire(timeout=self._schema_init_wait_timeout_seconds)
        if not acquired:
            raise TimeoutError("Autonomous session schema initialization timed out")
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
                        conversation_id TEXT PRIMARY KEY,
                        payload_json TEXT NOT NULL,
                        state TEXT NOT NULL CHECK (state IN ('paused', 'in_flight')),
                        claim_token TEXT,
                        progress_started INTEGER NOT NULL DEFAULT 0
                            CHECK (progress_started IN (0, 1)),
                        created_at DOUBLE PRECISION NOT NULL,
                        updated_at DOUBLE PRECISION NOT NULL,
                        expires_at DOUBLE PRECISION NOT NULL,
                        claimed_at DOUBLE PRECISION,
                        claim_expires_at DOUBLE PRECISION,
                        continue_fingerprint TEXT,
                        outcome_json TEXT,
                        CHECK (
                            (state = 'paused' AND claim_token IS NULL
                             AND claimed_at IS NULL AND progress_started = 0)
                            OR
                            (state = 'in_flight' AND claim_token IS NOT NULL
                             AND claimed_at IS NOT NULL)
                        )
                    )
                    """
                )
                self._migrate_columns(connection)
                connection.execute(
                    f"CREATE INDEX IF NOT EXISTS {_TABLE}_state_expiry_idx "
                    f"ON {_TABLE} (state, expires_at)"
                )
                connection.execute(
                    f"CREATE INDEX IF NOT EXISTS {_TABLE}_state_created_idx "
                    f"ON {_TABLE} (state, created_at)"
                )
                connection.execute(
                    f"CREATE INDEX IF NOT EXISTS {_TABLE}_claim_expiry_idx "
                    f"ON {_TABLE} (claim_expires_at)"
                )
            self._schema_ready = True
        finally:
            self._schema_lock.release()

    def _migrate_columns(self, connection) -> None:
        columns = {
            "claim_expires_at": "DOUBLE PRECISION",
            "continue_fingerprint": "TEXT",
            "outcome_json": "TEXT",
        }
        if self._postgres:
            for name, column_type in columns.items():
                connection.execute(
                    f"ALTER TABLE {_TABLE} ADD COLUMN IF NOT EXISTS {name} {column_type}"
                )
        else:
            existing = {
                row[1] for row in connection.execute(f"PRAGMA table_info({_TABLE})").fetchall()
            }
            for name, column_type in columns.items():
                if name not in existing:
                    connection.execute(f"ALTER TABLE {_TABLE} ADD COLUMN {name} {column_type}")

        # Give legacy in-flight rows one full rollout grace lease. Their absent
        # fingerprint later forces ambiguity instead of automatic replay.
        p = self._placeholder
        connection.execute(
            f"UPDATE {_TABLE} SET claim_expires_at = {p} "
            "WHERE state = 'in_flight' AND claim_expires_at IS NULL "
            "AND outcome_json IS NULL",
            (self._now() + self._operation_lease_seconds,),
        )

    def _save_sync(self, conversation_id: str, payload_json: str) -> None:
        self._ensure_schema()
        with self._transaction(write=True) as connection:
            self._lock_capacity(connection)
            created_at = self._now()
            self._reap_stale_claims(connection, created_at)
            self._purge_expired(connection, created_at)
            if not self._insert_paused_row(connection, conversation_id, payload_json, created_at):
                raise SessionAlreadyExistsError(conversation_id)
            self._enforce_capacity(connection, conversation_id)

    def _handoff_sync(
        self,
        conversation_id: str,
        claim_token: str,
        new_conversation_id: str,
        payload_json: str,
        outcome_json: str,
    ) -> bool:
        self._ensure_schema()
        p = self._placeholder
        with self._transaction(write=True) as connection:
            self._lock_capacity(connection)
            now = self._now()
            self._reap_stale_claims(connection, now)
            self._purge_expired(connection, now)
            completed = connection.execute(
                f"""
                UPDATE {_TABLE}
                SET state = 'in_flight', claim_token = {p}, claimed_at = {p},
                    claim_expires_at = NULL, progress_started = 1,
                    outcome_json = {p}, updated_at = {p}, expires_at = {p}
                WHERE conversation_id = {p} AND state = 'in_flight'
                  AND claim_token = {p} AND claim_expires_at > {p}
                  AND expires_at > {p} AND continue_fingerprint IS NOT NULL
                  AND outcome_json IS NULL
                """,
                (
                    _OUTCOME_TOMBSTONE_TOKEN,
                    now,
                    outcome_json,
                    now,
                    now + self._ttl_seconds,
                    conversation_id,
                    claim_token,
                    now,
                    now,
                ),
            )
            if completed.rowcount != 1:
                return False
            if not self._insert_paused_row(connection, new_conversation_id, payload_json, now):
                raise SessionAlreadyExistsError(new_conversation_id)
            self._enforce_capacity(connection, new_conversation_id)
            return True

    def _finish_sync(
        self,
        conversation_id: str,
        claim_token: str,
        outcome_json: str,
    ) -> bool:
        self._ensure_schema()
        p = self._placeholder
        with self._transaction(write=True) as connection:
            self._select_row(connection, conversation_id, for_update=True)
            now = self._now()
            cursor = connection.execute(
                f"""
                UPDATE {_TABLE}
                SET state = 'in_flight', claim_token = {p}, claimed_at = {p},
                    claim_expires_at = NULL, progress_started = 1,
                    outcome_json = {p}, updated_at = {p}, expires_at = {p}
                WHERE conversation_id = {p} AND state = 'in_flight'
                  AND claim_token = {p} AND claim_expires_at > {p}
                  AND expires_at > {p} AND continue_fingerprint IS NOT NULL
                  AND outcome_json IS NULL
                """,
                (
                    _OUTCOME_TOMBSTONE_TOKEN,
                    now,
                    outcome_json,
                    now,
                    now + self._ttl_seconds,
                    conversation_id,
                    claim_token,
                    now,
                    now,
                ),
            )
            return cursor.rowcount == 1

    def _lock_capacity(self, connection) -> None:
        if self._postgres:
            connection.execute(f"LOCK TABLE {_TABLE} IN SHARE ROW EXCLUSIVE MODE")

    def _purge_expired(self, connection, now: float) -> None:
        p = self._placeholder
        connection.execute(
            f"""
            DELETE FROM {_TABLE}
            WHERE expires_at <= {p}
              AND (state = 'paused' OR claim_expires_at IS NULL
                   OR claim_expires_at <= {p})
            """,
            (now, now),
        )

    def _purge_expired_target(self, connection, conversation_id: str, now: float) -> None:
        p = self._placeholder
        connection.execute(
            f"""
            DELETE FROM {_TABLE}
            WHERE conversation_id = {p} AND expires_at <= {p}
              AND (state = 'paused' OR claim_expires_at IS NULL
                   OR claim_expires_at <= {p})
            """,
            (conversation_id, now, now),
        )

    def _reap_stale_claims(self, connection, now: float) -> None:
        """Fence expired owners before capacity or inspection decisions."""
        p = self._placeholder
        connection.execute(
            f"""
            UPDATE {_TABLE}
            SET state = 'in_flight', claim_token = {p}, claimed_at = {p},
                claim_expires_at = NULL, progress_started = 1,
                outcome_json = {p}, updated_at = {p}
            WHERE state = 'in_flight' AND claim_expires_at <= {p}
              AND expires_at > {p}
              AND (progress_started = 1 OR continue_fingerprint IS NULL)
              AND outcome_json IS NULL
            """,
            (
                _OUTCOME_TOMBSTONE_TOKEN,
                now,
                self._ambiguous_outcome_json(),
                now,
                now,
                now,
            ),
        )
        connection.execute(
            f"""
            UPDATE {_TABLE}
            SET state = 'paused', claim_token = NULL, claimed_at = NULL,
                claim_expires_at = NULL, updated_at = {p}
            WHERE state = 'in_flight' AND claim_expires_at <= {p}
              AND expires_at > {p} AND progress_started = 0
              AND continue_fingerprint IS NOT NULL AND outcome_json IS NULL
            """,
            (now, now, now),
        )

    def _reap_stale_target(self, connection, conversation_id: str, now: float) -> None:
        """Fence only one locked row, avoiding cross-row lock inversion."""
        p = self._placeholder
        connection.execute(
            f"""
            UPDATE {_TABLE}
            SET state = 'in_flight', claim_token = {p}, claimed_at = {p},
                claim_expires_at = NULL, progress_started = 1,
                outcome_json = {p}, updated_at = {p}
            WHERE conversation_id = {p} AND state = 'in_flight'
              AND claim_expires_at <= {p} AND expires_at > {p}
              AND (progress_started = 1 OR continue_fingerprint IS NULL)
              AND outcome_json IS NULL
            """,
            (
                _OUTCOME_TOMBSTONE_TOKEN,
                now,
                self._ambiguous_outcome_json(),
                now,
                conversation_id,
                now,
                now,
            ),
        )
        connection.execute(
            f"""
            UPDATE {_TABLE}
            SET state = 'paused', claim_token = NULL, claimed_at = NULL,
                claim_expires_at = NULL, updated_at = {p}
            WHERE conversation_id = {p} AND state = 'in_flight'
              AND claim_expires_at <= {p} AND expires_at > {p}
              AND progress_started = 0 AND continue_fingerprint IS NOT NULL
              AND outcome_json IS NULL
            """,
            (now, conversation_id, now, now),
        )

    def _insert_paused_row(
        self,
        connection,
        conversation_id: str,
        payload_json: str,
        created_at: float,
    ) -> bool:
        p = self._placeholder
        statement = "INSERT"
        conflict = " ON CONFLICT (conversation_id) DO NOTHING" if self._postgres else ""
        if not self._postgres:
            statement = "INSERT OR IGNORE"
        cursor = connection.execute(
            f"""
            {statement} INTO {_TABLE}
                (conversation_id, payload_json, state, progress_started,
                 created_at, updated_at, expires_at)
            VALUES ({p}, {p}, 'paused', 0, {p}, {p}, {p}){conflict}
            """,
            (
                conversation_id,
                payload_json,
                created_at,
                created_at,
                created_at + self._ttl_seconds,
            ),
        )
        return cursor.rowcount == 1

    def _enforce_capacity(self, connection, protected_id: str) -> None:
        p = self._placeholder
        active_count = connection.execute(
            f"SELECT COUNT(*) FROM {_TABLE} WHERE outcome_json IS NULL"
        ).fetchone()[0]
        excess = int(active_count) - self._max_count
        if excess <= 0:
            return
        evictable = connection.execute(
            f"""
            SELECT conversation_id FROM {_TABLE}
            WHERE state = 'paused' AND outcome_json IS NULL
              AND conversation_id <> {p}
            ORDER BY created_at ASC, conversation_id ASC
            """,
            (protected_id,),
        ).fetchall()
        if len(evictable) < excess:
            raise SessionCapacityError("all session capacity is in flight")
        for (evicted_id,) in evictable[:excess]:
            connection.execute(
                f"DELETE FROM {_TABLE} WHERE conversation_id = {p} "
                "AND state = 'paused' AND outcome_json IS NULL",
                (evicted_id,),
            )

    def _select_row(self, connection, conversation_id: str, *, for_update=False):
        p = self._placeholder
        suffix = " FOR UPDATE" if self._postgres and for_update else ""
        return connection.execute(
            f"""
            SELECT state, payload_json, created_at, expires_at, claimed_at,
                   progress_started, claim_token, claim_expires_at,
                   continue_fingerprint, outcome_json
            FROM {_TABLE} WHERE conversation_id = {p}{suffix}
            """,
            (conversation_id,),
        ).fetchone()

    def _inspect_sync(self, conversation_id: str) -> SessionInspection | None:
        self._ensure_schema()
        with self._transaction(write=True) as connection:
            self._select_row(connection, conversation_id, for_update=True)
            now = self._now()
            self._reap_stale_target(connection, conversation_id, now)
            self._purge_expired_target(connection, conversation_id, now)
            row = self._select_row(connection, conversation_id)
        return None if row is None else self._row_to_inspection(row)

    def _status_sync(self, conversation_id: str) -> str | None:
        self._ensure_schema()
        with self._transaction(write=True) as connection:
            self._select_row(connection, conversation_id, for_update=True)
            now = self._now()
            self._reap_stale_target(connection, conversation_id, now)
            self._purge_expired_target(connection, conversation_id, now)
            row = self._select_row(connection, conversation_id)
        return None if row is None else self._visible_state(row[0], row[9])

    def _row_to_inspection(self, row) -> SessionInspection:
        (
            physical_state,
            payload_json,
            created_at,
            expires_at,
            claimed_at,
            progress,
            _claim_token,
            claim_expires_at,
            continue_fingerprint,
            outcome_json,
        ) = row
        if physical_state not in _PHYSICAL_STATES:
            raise RuntimeError(f"invalid autonomous session state: {physical_state}")
        try:
            payload = json.loads(payload_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("autonomous session payload is invalid JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("autonomous session payload is not a JSON object")
        state = self._visible_state(physical_state, outcome_json)
        outcome = None
        if outcome_json is not None:
            _, outcome = self._decode_outcome(outcome_json)
        return SessionInspection(
            state=state,
            payload=payload,
            created_at=float(created_at),
            expires_at=float(expires_at),
            claimed_at=None if claimed_at is None else float(claimed_at),
            progress_started=bool(progress),
            claim_expires_at=(None if claim_expires_at is None else float(claim_expires_at)),
            continue_fingerprint=continue_fingerprint,
            outcome=outcome,
        )

    @classmethod
    def _visible_state(cls, physical_state: str, outcome_json: str | None) -> str:
        if physical_state not in _PHYSICAL_STATES:
            raise RuntimeError(f"invalid autonomous session state: {physical_state}")
        if outcome_json is None:
            return physical_state
        state, _ = cls._decode_outcome(outcome_json)
        if state not in _VISIBLE_STATES:
            raise RuntimeError(f"invalid autonomous session outcome state: {state}")
        return state

    def _claim_sync(
        self,
        conversation_id: str,
        continue_fingerprint: str,
        claim_token: str,
    ) -> SessionClaim:
        self._ensure_schema()
        p = self._placeholder
        with self._transaction(write=True) as connection:
            row = self._select_row(connection, conversation_id, for_update=True)
            now = self._now()
            if row is None:
                return SessionClaim(claimed=False, reason="missing")
            (
                physical_state,
                payload_json,
                _created_at,
                expires_at,
                _claimed_at,
                progress_started,
                current_token,
                claim_expires_at,
                stored_fingerprint,
                outcome_json,
            ) = row

            if outcome_json is not None:
                if float(expires_at) <= now:
                    connection.execute(
                        f"DELETE FROM {_TABLE} WHERE conversation_id = {p}",
                        (conversation_id,),
                    )
                    return SessionClaim(claimed=False, reason="expired")
                state, response = self._decode_outcome(outcome_json)
                # Legacy in-flight rows intentionally have no request binding.
                # Once fenced they are ambiguous for every caller, but never
                # reveal or replay a completed response.
                if state == "ambiguous" and stored_fingerprint is None:
                    return SessionClaim(claimed=False, reason="ambiguous")
                if stored_fingerprint is None or not secrets.compare_digest(
                    stored_fingerprint, continue_fingerprint
                ):
                    return SessionClaim(claimed=False, reason="payload_mismatch")
                if state == "completed":
                    return SessionClaim(claimed=False, reason="completed", outcome=response)
                return SessionClaim(claimed=False, reason="ambiguous")

            if physical_state == "paused":
                if stored_fingerprint is not None and not secrets.compare_digest(
                    stored_fingerprint, continue_fingerprint
                ):
                    return SessionClaim(claimed=False, reason="payload_mismatch")
                if float(expires_at) <= now:
                    connection.execute(
                        f"DELETE FROM {_TABLE} WHERE conversation_id = {p}",
                        (conversation_id,),
                    )
                    return SessionClaim(claimed=False, reason="expired")
            elif physical_state == "in_flight":
                # Request binding is checked before lease status or any stale
                # transition. A different reply must never mutate, fence, or
                # poison the original continuation's durable row.
                if stored_fingerprint is not None and not secrets.compare_digest(
                    stored_fingerprint, continue_fingerprint
                ):
                    return SessionClaim(claimed=False, reason="payload_mismatch")
                if claim_expires_at is not None and float(claim_expires_at) > now:
                    return SessionClaim(claimed=False, reason="in_progress")
                if float(expires_at) <= now:
                    connection.execute(
                        f"DELETE FROM {_TABLE} WHERE conversation_id = {p} "
                        "AND state = 'in_flight' AND claim_token = {p}",
                        (conversation_id, current_token),
                    )
                    return SessionClaim(claimed=False, reason="expired")
                if bool(progress_started) or stored_fingerprint is None:
                    cursor = connection.execute(
                        f"""
                        UPDATE {_TABLE}
                        SET state = 'in_flight', claim_token = {p},
                            claimed_at = {p}, claim_expires_at = NULL,
                            progress_started = 1, outcome_json = {p},
                            updated_at = {p}, continue_fingerprint = COALESCE(
                                continue_fingerprint, {p}
                            )
                        WHERE conversation_id = {p} AND state = 'in_flight'
                          AND claim_token = {p}
                          AND (claim_expires_at IS NULL OR claim_expires_at <= {p})
                          AND outcome_json IS NULL
                        """,
                        (
                            _OUTCOME_TOMBSTONE_TOKEN,
                            now,
                            self._ambiguous_outcome_json(),
                            now,
                            continue_fingerprint,
                            conversation_id,
                            current_token,
                            now,
                        ),
                    )
                    if cursor.rowcount != 1:
                        return SessionClaim(claimed=False, reason="in_progress")
                    return SessionClaim(claimed=False, reason="ambiguous")
            else:
                raise RuntimeError(f"invalid autonomous session state: {physical_state}")

            lease_expires_at = now + self._operation_lease_seconds
            if physical_state == "paused":
                condition = "state = 'paused' AND outcome_json IS NULL"
                condition_args: tuple[Any, ...] = ()
            else:
                condition = (
                    "state = 'in_flight' AND claim_token = {p} "
                    "AND (claim_expires_at IS NULL OR claim_expires_at <= {p}) "
                    "AND progress_started = 0 AND outcome_json IS NULL"
                ).format(p=p)
                condition_args = (current_token, now)
            cursor = connection.execute(
                f"""
                UPDATE {_TABLE}
                SET state = 'in_flight', claim_token = {p}, claimed_at = {p},
                    claim_expires_at = {p}, continue_fingerprint = {p},
                    updated_at = {p}, expires_at = {p}, progress_started = 0
                WHERE conversation_id = {p} AND {condition}
                """,
                (
                    claim_token,
                    now,
                    lease_expires_at,
                    continue_fingerprint,
                    now,
                    self._claimed_expires_at(now),
                    conversation_id,
                    *condition_args,
                ),
            )
            if cursor.rowcount != 1:
                return SessionClaim(claimed=False, reason="in_progress")
            try:
                payload = json.loads(payload_json)
            except (TypeError, json.JSONDecodeError):
                return SessionClaim(
                    claimed=True,
                    reason="invalid_payload",
                    claim_token=claim_token,
                )
            if not isinstance(payload, dict):
                return SessionClaim(
                    claimed=True,
                    reason="invalid_payload",
                    claim_token=claim_token,
                )
            return SessionClaim(claimed=True, claim_token=claim_token, payload=payload)

    def _renew_sync(self, conversation_id: str, claim_token: str) -> bool:
        self._ensure_schema()
        p = self._placeholder
        with self._transaction(write=True) as connection:
            row = self._select_row(connection, conversation_id, for_update=True)
            now = self._now()
            existing_lease = float(row[7]) if row is not None and row[7] is not None else 0.0
            existing_expiry = float(row[3]) if row is not None else 0.0
            cursor = connection.execute(
                f"""
                UPDATE {_TABLE}
                SET claim_expires_at = {p}, updated_at = {p}, expires_at = {p}
                WHERE conversation_id = {p} AND state = 'in_flight'
                  AND claim_token = {p} AND claim_expires_at > {p}
                  AND expires_at > {p} AND outcome_json IS NULL
                """,
                (
                    max(existing_lease, now + self._operation_lease_seconds),
                    now,
                    max(existing_expiry, self._claimed_expires_at(now)),
                    conversation_id,
                    claim_token,
                    now,
                    now,
                ),
            )
            return cursor.rowcount == 1

    def _mark_progress_sync(self, conversation_id: str, claim_token: str) -> bool:
        self._ensure_schema()
        p = self._placeholder
        with self._transaction(write=True) as connection:
            self._select_row(connection, conversation_id, for_update=True)
            now = self._now()
            cursor = connection.execute(
                f"""
                UPDATE {_TABLE}
                SET progress_started = 1, claim_expires_at = {p},
                    updated_at = {p}, expires_at = {p}
                WHERE conversation_id = {p} AND state = 'in_flight'
                  AND claim_token = {p} AND claim_expires_at > {p}
                  AND expires_at > {p} AND outcome_json IS NULL
                """,
                (
                    now + self._operation_lease_seconds,
                    now,
                    self._claimed_expires_at(now),
                    conversation_id,
                    claim_token,
                    now,
                    now,
                ),
            )
            return cursor.rowcount == 1

    def _cancel_sync(self, conversation_id: str, claim_token: str) -> bool:
        self._ensure_schema()
        p = self._placeholder
        with self._transaction(write=True) as connection:
            self._select_row(connection, conversation_id, for_update=True)
            now = self._now()
            cursor = connection.execute(
                f"""
                UPDATE {_TABLE}
                SET state = 'paused', claim_token = NULL, claimed_at = NULL,
                    claim_expires_at = NULL, continue_fingerprint = NULL,
                    updated_at = {p}
                WHERE conversation_id = {p} AND state = 'in_flight'
                  AND claim_token = {p} AND claim_expires_at > {p}
                  AND expires_at > {p} AND progress_started = 0
                  AND outcome_json IS NULL
                """,
                (now, conversation_id, claim_token, now, now),
            )
            return cursor.rowcount == 1

    def _discard_paused_sync(self, conversation_id: str) -> SessionDiscardResult:
        self._ensure_schema()
        p = self._placeholder
        with self._transaction(write=True) as connection:
            self._select_row(connection, conversation_id, for_update=True)
            now = self._now()
            self._reap_stale_target(connection, conversation_id, now)
            self._purge_expired_target(connection, conversation_id, now)
            row = self._select_row(connection, conversation_id, for_update=True)
            if row is None:
                return SessionDiscardResult(discarded=False, reason="missing")
            physical_state, outcome_json = row[0], row[9]
            if outcome_json is not None:
                outcome_state, _ = self._decode_outcome(outcome_json)
                return SessionDiscardResult(
                    discarded=False,
                    reason=("completed" if outcome_state == "completed" else "ambiguous"),
                )
            if physical_state == "in_flight":
                return SessionDiscardResult(discarded=False, reason="in_progress")
            cursor = connection.execute(
                f"""
                DELETE FROM {_TABLE}
                WHERE conversation_id = {p} AND state = 'paused'
                  AND outcome_json IS NULL
                """,
                (conversation_id,),
            )
            if cursor.rowcount == 1:
                return SessionDiscardResult(discarded=True, reason="canceled")
            return SessionDiscardResult(discarded=False, reason="in_progress")

    def _consume_sync(self, conversation_id: str, claim_token: str) -> bool:
        self._ensure_schema()
        p = self._placeholder
        with self._transaction(write=True) as connection:
            self._select_row(connection, conversation_id, for_update=True)
            now = self._now()
            cursor = connection.execute(
                f"""
                DELETE FROM {_TABLE}
                WHERE conversation_id = {p} AND state = 'in_flight'
                  AND claim_token = {p} AND claim_expires_at > {p}
                  AND expires_at > {p} AND outcome_json IS NULL
                """,
                (conversation_id, claim_token, now, now),
            )
            return cursor.rowcount == 1
