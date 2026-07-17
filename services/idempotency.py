"""Durable request receipts for safe retries of tool-using HTTP requests.

The receipt is intentionally separate from the in-memory audit trail.  A
completed request can be replayed without running the agent again; a request
that reached a non-replayable tool is failed closed after an interrupted run.
This is an at-most-once guard, not a distributed exactly-once transaction with
the LangGraph memory store.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


load_dotenv(Path(__file__).parent.parent / ".env")

_TABLE = "studyloop_idempotency_receipts"
_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")


class IdempotencyConflictError(RuntimeError):
    """The supplied key cannot safely start a new execution."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class BeginDecision:
    replayed: bool
    response: dict[str, Any] | None = None


def normalize_idempotency_key(value: object) -> str | None:
    """Return a validated key, treating FastAPI's direct-call default as absent."""
    if not isinstance(value, str):
        return None
    key = value.strip()
    if not _KEY_PATTERN.fullmatch(key):
        raise ValueError(
            "Idempotency-Key 必须为 8-128 位字母、数字或 . _ : -"
        )
    return key


def _fingerprint(operation: str, payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        {"operation": operation, "payload": payload},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class IdempotencyStore:
    """Atomic receipt store backed by SQLite locally or PostgreSQL in production."""

    def __init__(
        self,
        *,
        database_url: str | None = None,
        sqlite_path: str | None = None,
    ) -> None:
        if database_url and sqlite_path:
            raise ValueError("database_url and sqlite_path are mutually exclusive")
        self._database_url = database_url
        self._sqlite_path = sqlite_path or os.getenv(
            "IDEMPOTENCY_DB_PATH", "./.idempotency.sqlite3"
        )
        self._schema_ready = False
        self._schema_lock = threading.Lock()

    @classmethod
    def from_environment(cls) -> "IdempotencyStore":
        database_url = os.getenv("DATABASE_URL") or None
        return cls(database_url=database_url) if database_url else cls()

    async def begin(
        self,
        key: str,
        operation: str,
        payload: dict[str, Any],
    ) -> BeginDecision:
        worker = asyncio.create_task(
            asyncio.to_thread(
                self._begin_sync,
                key,
                operation,
                _fingerprint(operation, payload),
            )
        )
        try:
            # A cancelled await must not cancel the Future proxy while the DB
            # thread can still commit a claim behind it.
            return await asyncio.shield(worker)
        except asyncio.CancelledError as cancelled:
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    # Multiple cancellation sources (for example disconnect
                    # plus shutdown) still cannot orphan a committed claim.
                    continue
            try:
                decision = worker.result()
            except BaseException:
                # No successful ownership decision means there is no claim we
                # can safely identify as ours.
                raise cancelled

            if not decision.replayed:
                cleanup = asyncio.create_task(self.abort(key))
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        continue
                cleanup.result()
            raise cancelled

    async def mark_effect_started(self, key: str, tool_name: str) -> None:
        await asyncio.to_thread(self._mark_effect_started_sync, key, tool_name)

    async def complete(self, key: str, response: dict[str, Any]) -> None:
        response_json = json.dumps(
            response, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        await asyncio.to_thread(self._complete_sync, key, response_json)

    async def abort(self, key: str) -> bool:
        """Release a clean claim or persist ambiguity after an effect started."""
        return await asyncio.to_thread(self._abort_sync, key)

    async def has_effect_started(self, key: str) -> bool:
        return await asyncio.to_thread(self._has_effect_started_sync, key)

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

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._schema_lock:
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
                    created_at DOUBLE PRECISION NOT NULL,
                    updated_at DOUBLE PRECISION NOT NULL
                )
            """
            with self._transaction() as connection:
                connection.execute(statement)
            self._schema_ready = True

    def _begin_sync(
        self, key: str, operation: str, request_fingerprint: str
    ) -> BeginDecision:
        self._ensure_schema()
        now = time.time()
        with self._transaction() as connection:
            if self._postgres:
                cursor = connection.execute(
                    f"""
                    INSERT INTO {_TABLE}
                        (idempotency_key, operation, request_fingerprint, state,
                         created_at, updated_at)
                    VALUES (%s, %s, %s, 'pending', %s, %s)
                    ON CONFLICT (idempotency_key) DO NOTHING
                    """,
                    (key, operation, request_fingerprint, now, now),
                )
                inserted = cursor.rowcount == 1
                row = connection.execute(
                    f"""
                    SELECT operation, request_fingerprint, state, response_json
                    FROM {_TABLE} WHERE idempotency_key = %s
                    """,
                    (key,),
                ).fetchone()
            else:
                cursor = connection.execute(
                    f"""
                    INSERT OR IGNORE INTO {_TABLE}
                        (idempotency_key, operation, request_fingerprint, state,
                         created_at, updated_at)
                    VALUES (?, ?, ?, 'pending', ?, ?)
                    """,
                    (key, operation, request_fingerprint, now, now),
                )
                inserted = cursor.rowcount == 1
                row = connection.execute(
                    f"""
                    SELECT operation, request_fingerprint, state, response_json
                    FROM {_TABLE} WHERE idempotency_key = ?
                    """,
                    (key,),
                ).fetchone()

        if inserted:
            return BeginDecision(replayed=False)
        if row is None:
            raise RuntimeError("idempotency receipt disappeared after claim")

        existing_operation, existing_fingerprint, state, response_json = row
        if (
            existing_operation != operation
            or existing_fingerprint != request_fingerprint
        ):
            raise IdempotencyConflictError("payload_mismatch")
        if state == "completed" and response_json:
            return BeginDecision(replayed=True, response=json.loads(response_json))
        if state == "pending":
            raise IdempotencyConflictError("in_progress")
        raise IdempotencyConflictError("ambiguous")

    def _mark_effect_started_sync(self, key: str, tool_name: str) -> None:
        self._ensure_schema()
        placeholder = "%s" if self._postgres else "?"
        with self._transaction() as connection:
            cursor = connection.execute(
                f"""
                UPDATE {_TABLE}
                SET state = 'effect_started', effect_tool = {placeholder},
                    updated_at = {placeholder}
                WHERE idempotency_key = {placeholder}
                  AND state IN ('pending', 'effect_started')
                """,
                (tool_name, time.time(), key),
            )
            if cursor.rowcount != 1:
                raise IdempotencyConflictError("receipt_not_pending")

    def _complete_sync(self, key: str, response_json: str) -> None:
        self._ensure_schema()
        placeholder = "%s" if self._postgres else "?"
        with self._transaction() as connection:
            cursor = connection.execute(
                f"""
                UPDATE {_TABLE}
                SET state = 'completed', response_json = {placeholder},
                    updated_at = {placeholder}
                WHERE idempotency_key = {placeholder}
                  AND state IN ('pending', 'effect_started')
                """,
                (response_json, time.time(), key),
            )
            if cursor.rowcount != 1:
                raise IdempotencyConflictError("receipt_not_completable")

    def _abort_sync(self, key: str) -> bool:
        self._ensure_schema()
        placeholder = "%s" if self._postgres else "?"
        with self._transaction() as connection:
            row = connection.execute(
                f"SELECT state FROM {_TABLE} WHERE idempotency_key = {placeholder}",
                (key,),
            ).fetchone()
            if row is None:
                return False
            state = row[0]
            if state == "pending":
                connection.execute(
                    f"DELETE FROM {_TABLE} WHERE idempotency_key = {placeholder}",
                    (key,),
                )
                return False
            if state == "effect_started":
                connection.execute(
                    f"""
                    UPDATE {_TABLE} SET state = 'ambiguous', updated_at = {placeholder}
                    WHERE idempotency_key = {placeholder}
                    """,
                    (time.time(), key),
                )
                return True
            return state == "ambiguous"

    def _has_effect_started_sync(self, key: str) -> bool:
        self._ensure_schema()
        placeholder = "%s" if self._postgres else "?"
        with self._transaction() as connection:
            row = connection.execute(
                f"SELECT state FROM {_TABLE} WHERE idempotency_key = {placeholder}",
                (key,),
            ).fetchone()
        return bool(row and row[0] in {"effect_started", "ambiguous"})


request_idempotency = IdempotencyStore.from_environment()
