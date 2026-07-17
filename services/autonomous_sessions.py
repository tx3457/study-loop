"""Durable pause snapshots for the standalone Autonomous HITL loop.

The store deliberately persists JSON wire data rather than Python objects.  A
paused session can therefore be reopened by another process, while every
resume is protected by a database compare-and-swap claim.  Claims are never
released by a timeout: only the live owner may release a claim that has not
recorded progress.
"""

from __future__ import annotations

import asyncio
import json
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

_TABLE = "studyloop_autonomous_sessions"
_VALID_STATES = {"paused", "in_flight"}


class SessionAlreadyExistsError(RuntimeError):
    """A pause snapshot already exists for the conversation ID."""


class SessionCapacityError(RuntimeError):
    """No paused snapshot can be evicted without touching an active claim."""


class SessionPayloadTooLargeError(ValueError):
    """A serialized snapshot exceeds the configured storage boundary."""


@dataclass(frozen=True, slots=True)
class SessionInspection:
    state: str
    payload: dict[str, Any]
    created_at: float
    expires_at: float
    claimed_at: float | None
    progress_started: bool


@dataclass(frozen=True, slots=True)
class SessionClaim:
    claimed: bool
    reason: str | None = None
    claim_token: str | None = None
    payload: dict[str, Any] | None = None


class AutonomousSessionStore:
    """SQLite/PostgreSQL store for versioned Autonomous pause snapshots."""

    def __init__(
        self,
        *,
        database_url: str | None = None,
        sqlite_path: str | None = None,
        ttl_seconds: float = 3600,
        max_count: int = 200,
        max_payload_bytes: int = 2 * 1024 * 1024,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if database_url and sqlite_path:
            raise ValueError("database_url and sqlite_path are mutually exclusive")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_count <= 0:
            raise ValueError("max_count must be positive")
        if max_payload_bytes <= 0:
            raise ValueError("max_payload_bytes must be positive")

        self._database_url = database_url
        # Sharing the local database with request receipts reduces operational
        # drift and leaves room for atomic cross-table transitions later.
        self._sqlite_path = (
            sqlite_path
            or os.getenv("AUTONOMOUS_SESSION_DB_PATH")
            or os.getenv("IDEMPOTENCY_DB_PATH", "./.idempotency.sqlite3")
        )
        self._ttl_seconds = ttl_seconds
        self._max_count = max_count
        self._max_payload_bytes = max_payload_bytes
        self._clock = clock
        self._schema_ready = False
        self._schema_lock = threading.Lock()

    @classmethod
    def from_environment(cls) -> "AutonomousSessionStore":
        database_url = os.getenv("DATABASE_URL") or None
        kwargs = {
            "ttl_seconds": float(os.getenv("AUTONOMOUS_SESSION_TTL_SECONDS", "3600")),
            "max_count": int(os.getenv("AUTONOMOUS_SESSION_MAX_COUNT", "200")),
            "max_payload_bytes": int(
                os.getenv("AUTONOMOUS_SESSION_MAX_PAYLOAD_BYTES", str(2 * 1024 * 1024))
            ),
        }
        return cls(database_url=database_url, **kwargs)

    async def save(self, conversation_id: str, payload: dict[str, Any]) -> None:
        """Insert a complete immutable pause snapshot.

        JSON encoding happens before opening the transaction, so malformed
        payloads cannot evict a valid snapshot or partially mutate the table.
        """
        if not conversation_id:
            raise ValueError("conversation_id must not be empty")
        payload_json = self._serialize_payload(payload)
        await self._run_thread(
            self._save_sync, conversation_id, payload_json, self._clock()
        )

    async def handoff(
        self,
        conversation_id: str,
        claim_token: str,
        new_conversation_id: str,
        payload: dict[str, Any],
    ) -> bool:
        """Atomically replace an owned claim with its next paused snapshot."""
        if not conversation_id or not new_conversation_id:
            raise ValueError("conversation IDs must not be empty")
        if conversation_id == new_conversation_id:
            raise ValueError("handoff requires a new conversation ID")
        payload_json = self._serialize_payload(payload)
        return await self._run_thread(
            self._handoff_sync,
            conversation_id,
            claim_token,
            new_conversation_id,
            payload_json,
            self._clock(),
        )

    async def inspect(self, conversation_id: str) -> SessionInspection | None:
        return await self._run_thread(
            self._inspect_sync, conversation_id, self._clock()
        )

    async def status(self, conversation_id: str) -> str | None:
        """Return only resumability state without decoding sensitive payload data."""
        return await self._run_thread(self._status_sync, conversation_id, self._clock())

    async def claim(self, conversation_id: str) -> SessionClaim:
        """Atomically claim a live paused snapshot.

        If the awaiting coroutine is cancelled after the database thread has
        committed, the method waits for the ownership result and conditionally
        releases its own clean claim before propagating cancellation.
        """
        claim_token = secrets.token_urlsafe(32)
        worker = asyncio.create_task(
            asyncio.to_thread(
                self._claim_sync,
                conversation_id,
                claim_token,
                self._clock(),
            )
        )
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError as cancelled:
            try:
                decision = await self._finish_cancelled_worker(worker)
            except BaseException:
                raise cancelled
            if decision.claimed:
                cleanup = asyncio.create_task(
                    self.release(conversation_id, claim_token)
                )
                await self._finish_cancelled_worker(cleanup)
            raise cancelled

    async def mark_progress(self, conversation_id: str, claim_token: str) -> bool:
        """Persist the point after which a claim must never be retried."""
        return await self._run_thread(
            self._mark_progress_sync,
            conversation_id,
            claim_token,
            self._clock(),
        )

    async def release(self, conversation_id: str, claim_token: str) -> bool:
        """Release only this owner's unmodified, unexpired claim."""
        return await self._run_thread(
            self._release_sync,
            conversation_id,
            claim_token,
            self._clock(),
        )

    async def consume(self, conversation_id: str, claim_token: str) -> bool:
        """Delete a claimed snapshot only when the fencing token matches."""
        return await self._run_thread(self._consume_sync, conversation_id, claim_token)

    async def _run_thread(self, function, *args):
        worker = asyncio.create_task(asyncio.to_thread(function, *args))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError as cancelled:
            try:
                await self._finish_cancelled_worker(worker)
            except BaseException:
                raise cancelled
            raise cancelled

    def _serialize_payload(self, payload: dict[str, Any]) -> str:
        if not isinstance(payload, dict):
            raise TypeError("session payload must be a JSON object")
        payload_json = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        payload_bytes = len(payload_json.encode("utf-8"))
        if payload_bytes > self._max_payload_bytes:
            raise SessionPayloadTooLargeError(
                f"session payload is {payload_bytes} bytes; "
                f"limit is {self._max_payload_bytes}"
            )
        return payload_json

    @staticmethod
    async def _finish_cancelled_worker(worker: asyncio.Task):
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
        return worker.result()

    def _connect(self):
        if self._database_url:
            import psycopg

            return psycopg.connect(self._database_url)

        path = Path(self._sqlite_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=10)
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    @contextmanager
    def _transaction(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
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
        with self._schema_lock:
            if self._schema_ready:
                return
            with self._transaction() as connection:
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
                connection.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {_TABLE}_state_expiry_idx
                    ON {_TABLE} (state, expires_at)
                    """
                )
                connection.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {_TABLE}_state_created_idx
                    ON {_TABLE} (state, created_at)
                    """
                )
            self._schema_ready = True

    def _save_sync(
        self, conversation_id: str, payload_json: str, created_at: float
    ) -> None:
        self._ensure_schema()
        with self._transaction() as connection:
            self._lock_capacity(connection)
            self._purge_expired_paused(connection, created_at)
            if not self._insert_paused_row(
                connection, conversation_id, payload_json, created_at
            ):
                raise SessionAlreadyExistsError(conversation_id)
            self._enforce_capacity(connection, conversation_id)

    def _handoff_sync(
        self,
        conversation_id: str,
        claim_token: str,
        new_conversation_id: str,
        payload_json: str,
        created_at: float,
    ) -> bool:
        self._ensure_schema()
        p = self._placeholder
        with self._transaction() as connection:
            self._lock_capacity(connection)
            self._purge_expired_paused(connection, created_at)
            removed = connection.execute(
                f"""
                DELETE FROM {_TABLE}
                WHERE conversation_id = {p} AND state = 'in_flight'
                  AND claim_token = {p}
                """,
                (conversation_id, claim_token),
            )
            if removed.rowcount != 1:
                return False
            if not self._insert_paused_row(
                connection, new_conversation_id, payload_json, created_at
            ):
                raise SessionAlreadyExistsError(new_conversation_id)
            self._enforce_capacity(connection, new_conversation_id)
            return True

    def _lock_capacity(self, connection) -> None:
        if self._postgres:
            # Capacity is a table-wide invariant. Serialize only the short
            # save/evict or handoff section across workers.
            connection.execute(f"LOCK TABLE {_TABLE} IN SHARE ROW EXCLUSIVE MODE")

    def _purge_expired_paused(self, connection, now: float) -> None:
        p = self._placeholder
        connection.execute(
            f"DELETE FROM {_TABLE} WHERE state = 'paused' AND expires_at <= {p}",
            (now,),
        )

    def _insert_paused_row(
        self,
        connection,
        conversation_id: str,
        payload_json: str,
        created_at: float,
    ) -> bool:
        p = self._placeholder
        expires_at = created_at + self._ttl_seconds
        if self._postgres:
            cursor = connection.execute(
                f"""
                INSERT INTO {_TABLE}
                    (conversation_id, payload_json, state, progress_started,
                     created_at, updated_at, expires_at)
                VALUES ({p}, {p}, 'paused', 0, {p}, {p}, {p})
                ON CONFLICT (conversation_id) DO NOTHING
                """,
                (
                    conversation_id,
                    payload_json,
                    created_at,
                    created_at,
                    expires_at,
                ),
            )
        else:
            cursor = connection.execute(
                f"""
                INSERT OR IGNORE INTO {_TABLE}
                    (conversation_id, payload_json, state, progress_started,
                     created_at, updated_at, expires_at)
                VALUES ({p}, {p}, 'paused', 0, {p}, {p}, {p})
                """,
                (
                    conversation_id,
                    payload_json,
                    created_at,
                    created_at,
                    expires_at,
                ),
            )
        return cursor.rowcount == 1

    def _enforce_capacity(self, connection, protected_id: str) -> None:
        p = self._placeholder
        active_count = connection.execute(
            f"SELECT COUNT(*) FROM {_TABLE} WHERE state IN ('paused', 'in_flight')"
        ).fetchone()[0]
        excess = active_count - self._max_count
        if excess <= 0:
            return

        # Only paused rows may be evicted. Never evict the newly inserted row:
        # if all other capacity is in flight, roll the transaction back.
        evictable = connection.execute(
            f"""
            SELECT conversation_id FROM {_TABLE}
            WHERE state = 'paused' AND conversation_id <> {p}
            ORDER BY created_at ASC, conversation_id ASC
            """,
            (protected_id,),
        ).fetchall()
        if len(evictable) < excess:
            raise SessionCapacityError("all session capacity is in flight")
        for (evicted_id,) in evictable[:excess]:
            connection.execute(
                f"DELETE FROM {_TABLE} WHERE conversation_id = {p} AND state = 'paused'",
                (evicted_id,),
            )

    def _inspect_sync(
        self, conversation_id: str, now: float
    ) -> SessionInspection | None:
        self._ensure_schema()
        p = self._placeholder
        with self._transaction() as connection:
            connection.execute(
                f"""
                DELETE FROM {_TABLE}
                WHERE conversation_id = {p} AND state = 'paused' AND expires_at <= {p}
                """,
                (conversation_id, now),
            )
            row = connection.execute(
                f"""
                SELECT state, payload_json, created_at, expires_at, claimed_at,
                       progress_started
                FROM {_TABLE} WHERE conversation_id = {p}
                """,
                (conversation_id,),
            ).fetchone()
        if row is None:
            return None
        state, payload_json, created_at, expires_at, claimed_at, progress = row
        if state not in _VALID_STATES:
            raise RuntimeError(f"invalid autonomous session state: {state}")
        payload = json.loads(payload_json)
        if not isinstance(payload, dict):
            raise RuntimeError("autonomous session payload is not a JSON object")
        return SessionInspection(
            state=state,
            payload=payload,
            created_at=created_at,
            expires_at=expires_at,
            claimed_at=claimed_at,
            progress_started=bool(progress),
        )

    def _status_sync(self, conversation_id: str, now: float) -> str | None:
        self._ensure_schema()
        p = self._placeholder
        with self._transaction() as connection:
            connection.execute(
                f"""
                DELETE FROM {_TABLE}
                WHERE conversation_id = {p} AND state = 'paused' AND expires_at <= {p}
                """,
                (conversation_id, now),
            )
            row = connection.execute(
                f"SELECT state FROM {_TABLE} WHERE conversation_id = {p}",
                (conversation_id,),
            ).fetchone()
        if row is None:
            return None
        state = row[0]
        if state not in _VALID_STATES:
            raise RuntimeError(f"invalid autonomous session state: {state}")
        return state

    def _claim_sync(
        self, conversation_id: str, claim_token: str, now: float
    ) -> SessionClaim:
        self._ensure_schema()
        p = self._placeholder
        with self._transaction() as connection:
            cursor = connection.execute(
                f"""
                UPDATE {_TABLE}
                SET state = 'in_flight', claim_token = {p}, claimed_at = {p},
                    updated_at = {p}, progress_started = 0
                WHERE conversation_id = {p} AND state = 'paused' AND expires_at > {p}
                """,
                (claim_token, now, now, conversation_id, now),
            )
            if cursor.rowcount == 1:
                row = connection.execute(
                    f"""
                    SELECT payload_json FROM {_TABLE}
                    WHERE conversation_id = {p} AND claim_token = {p}
                    """,
                    (conversation_id, claim_token),
                ).fetchone()
                if row is None:
                    raise RuntimeError("autonomous session disappeared after claim")
                try:
                    payload = json.loads(row[0])
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
                return SessionClaim(
                    claimed=True,
                    claim_token=claim_token,
                    payload=payload,
                )

            row = connection.execute(
                f"SELECT state, expires_at FROM {_TABLE} WHERE conversation_id = {p}",
                (conversation_id,),
            ).fetchone()
            if row is None:
                return SessionClaim(claimed=False, reason="missing")
            state, expires_at = row
            if state == "paused" and expires_at <= now:
                connection.execute(
                    f"""
                    DELETE FROM {_TABLE}
                    WHERE conversation_id = {p} AND state = 'paused'
                    """,
                    (conversation_id,),
                )
                return SessionClaim(claimed=False, reason="expired")
            if state == "in_flight":
                return SessionClaim(claimed=False, reason="in_progress")
            raise RuntimeError(f"invalid autonomous session state: {state}")

    def _mark_progress_sync(
        self,
        conversation_id: str,
        claim_token: str,
        now: float,
    ) -> bool:
        self._ensure_schema()
        p = self._placeholder
        with self._transaction() as connection:
            cursor = connection.execute(
                f"""
                UPDATE {_TABLE}
                SET progress_started = 1, updated_at = {p}
                WHERE conversation_id = {p} AND state = 'in_flight'
                  AND claim_token = {p}
                """,
                (now, conversation_id, claim_token),
            )
            return cursor.rowcount == 1

    def _release_sync(
        self,
        conversation_id: str,
        claim_token: str,
        now: float,
    ) -> bool:
        self._ensure_schema()
        p = self._placeholder
        with self._transaction() as connection:
            cursor = connection.execute(
                f"""
                UPDATE {_TABLE}
                SET state = 'paused', claim_token = NULL, claimed_at = NULL,
                    updated_at = {p}
                WHERE conversation_id = {p} AND state = 'in_flight'
                  AND claim_token = {p} AND progress_started = 0
                  AND expires_at > {p}
                """,
                (now, conversation_id, claim_token, now),
            )
            if cursor.rowcount == 1:
                return True
            # An owner may discover that its clean claim expired while it was
            # handling a provider failure. Delete only that exact clean claim.
            connection.execute(
                f"""
                DELETE FROM {_TABLE}
                WHERE conversation_id = {p} AND state = 'in_flight'
                  AND claim_token = {p} AND progress_started = 0
                  AND expires_at <= {p}
                """,
                (conversation_id, claim_token, now),
            )
            return False

    def _consume_sync(self, conversation_id: str, claim_token: str) -> bool:
        self._ensure_schema()
        p = self._placeholder
        with self._transaction() as connection:
            cursor = connection.execute(
                f"""
                DELETE FROM {_TABLE}
                WHERE conversation_id = {p} AND state = 'in_flight'
                  AND claim_token = {p}
                """,
                (conversation_id, claim_token),
            )
            return cursor.rowcount == 1
