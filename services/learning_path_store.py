"""Durable immutable Learning Path records.

The generation service is intentionally not coupled to this module. Callers
can look up a completed creation before invoking the model, then atomically
create-or-replay the canonical record after generation. SQLite is used for
local development and PostgreSQL is selected when ``DATABASE_URL`` is set.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import math
import os
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from pydantic import ValidationError

from models.learning_path import LearningPath


load_dotenv(Path(__file__).parent.parent / ".env")

_TABLE = "studyloop_learning_paths"
_SCHEMA_VERSION = 1
_READY_STATUS = "ready"
_POSTGRES_SCHEMA_LOCK_ID = 0x535455444C504154  # ASCII "STUDLPAT"
_PATH_ID_PATTERN = re.compile(r"^lp_[0-9a-f]{32}$")


class LearningPathCreationConflictError(RuntimeError):
    """An idempotency key was reused for a different creation request."""

    def __init__(self, reason: str = "payload_mismatch") -> None:
        self.reason = reason
        super().__init__(reason)


class LearningPathCorruptError(RuntimeError):
    """A durable row failed version, JSON, status, or integrity validation."""


class LearningPathPayloadTooLargeError(ValueError):
    """The canonical LearningPath JSON exceeds the configured byte limit."""


@dataclass(frozen=True, slots=True)
class LearningPathRecord:
    path_id: str
    user_id: str
    document_id: str
    path: LearningPath
    schema_version: int
    created_at: float


class LearningPathStore:
    """Immutable SQLite/PostgreSQL store for completed Learning Paths."""

    def __init__(
        self,
        *,
        database_url: str | None = None,
        sqlite_path: str | None = None,
        max_payload_bytes: int = 1024 * 1024,
        sqlite_busy_timeout_ms: int = 10_000,
        postgres_connect_timeout_seconds: int = 5,
        postgres_lock_timeout_ms: int = 5_000,
        postgres_statement_timeout_ms: int = 15_000,
        clock: Callable[[], float] = time.time,
        path_id_factory: Callable[[], str] | None = None,
    ) -> None:
        if database_url and sqlite_path:
            raise ValueError("database_url and sqlite_path are mutually exclusive")
        if max_payload_bytes <= 0:
            raise ValueError("max_payload_bytes must be positive")
        if sqlite_busy_timeout_ms <= 0:
            raise ValueError("sqlite_busy_timeout_ms must be positive")
        if postgres_connect_timeout_seconds <= 0:
            raise ValueError("postgres_connect_timeout_seconds must be positive")
        if postgres_lock_timeout_ms <= 0:
            raise ValueError("postgres_lock_timeout_ms must be positive")
        if postgres_statement_timeout_ms <= 0:
            raise ValueError("postgres_statement_timeout_ms must be positive")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if path_id_factory is not None and not callable(path_id_factory):
            raise TypeError("path_id_factory must be callable")

        self._database_url = database_url
        self._sqlite_path = (
            sqlite_path
            or os.getenv("LEARNING_PATH_DB_PATH")
            or os.getenv("IDEMPOTENCY_DB_PATH", "./.idempotency.sqlite3")
        )
        self._max_payload_bytes = max_payload_bytes
        self._sqlite_busy_timeout_ms = sqlite_busy_timeout_ms
        self._postgres_connect_timeout_seconds = postgres_connect_timeout_seconds
        self._postgres_lock_timeout_ms = postgres_lock_timeout_ms
        self._postgres_statement_timeout_ms = postgres_statement_timeout_ms
        self._clock = clock
        self._path_id_factory = path_id_factory or (
            lambda: f"lp_{uuid.uuid4().hex}"
        )
        self._schema_ready = False
        self._schema_lock = threading.Lock()

    @classmethod
    def from_environment(cls) -> "LearningPathStore":
        return cls(
            database_url=os.getenv("DATABASE_URL") or None,
            max_payload_bytes=int(
                os.getenv("LEARNING_PATH_MAX_PAYLOAD_BYTES", str(1024 * 1024))
            ),
            sqlite_busy_timeout_ms=int(
                os.getenv("LEARNING_PATH_SQLITE_BUSY_TIMEOUT_MS", "10000")
            ),
            postgres_connect_timeout_seconds=int(
                os.getenv("LEARNING_PATH_PG_CONNECT_TIMEOUT_SECONDS", "5")
            ),
            postgres_lock_timeout_ms=int(
                os.getenv("LEARNING_PATH_PG_LOCK_TIMEOUT_MS", "5000")
            ),
            postgres_statement_timeout_ms=int(
                os.getenv("LEARNING_PATH_PG_STATEMENT_TIMEOUT_MS", "15000")
            ),
        )

    async def create(
        self,
        user_id: str,
        document_id: str,
        path: LearningPath,
        *,
        idempotency_key: str,
        request_fingerprint: str,
    ) -> LearningPathRecord:
        """Create once, or replay the canonical row for the same request.

        A concurrent caller may have generated a different candidate response.
        The idempotency binding, rather than that candidate response, chooses
        the one canonical immutable record.
        """
        user_id = self._normalize_identifier(user_id, "user_id", 128)
        document_id = self._normalize_identifier(document_id, "document_id", 512)
        key_hash = self._key_hash(idempotency_key)
        request_fingerprint = self._validate_fingerprint(request_fingerprint)
        if self._storage_may_have_records():
            existing = await self._run_thread(
                self._find_by_creation_sync,
                key_hash,
                user_id,
                document_id,
                request_fingerprint,
            )
            if existing is not None:
                return existing
        try:
            payload_json = self._serialize_path(path, document_id)
        except (LearningPathPayloadTooLargeError, TypeError, ValueError):
            # A concurrent creator may have committed after the first lookup.
            # A completed idempotent result is authoritative even when this
            # caller's redundant candidate cannot be persisted.
            if self._storage_may_have_records():
                existing = await self._run_thread(
                    self._find_by_creation_sync,
                    key_hash,
                    user_id,
                    document_id,
                    request_fingerprint,
                )
                if existing is not None:
                    return existing
            raise
        path_id = self._new_path_id()
        return await self._run_thread(
            self._create_sync,
            path_id,
            user_id,
            document_id,
            payload_json,
            key_hash,
            request_fingerprint,
        )

    async def find_by_creation(
        self,
        idempotency_key: str,
        user_id: str,
        document_id: str,
        request_fingerprint: str,
    ) -> LearningPathRecord | None:
        """Replay a completed creation before any model call is made."""
        key_hash = self._key_hash(idempotency_key)
        user_id = self._normalize_identifier(user_id, "user_id", 128)
        document_id = self._normalize_identifier(document_id, "document_id", 512)
        request_fingerprint = self._validate_fingerprint(request_fingerprint)
        return await self._run_thread(
            self._find_by_creation_sync,
            key_hash,
            user_id,
            document_id,
            request_fingerprint,
        )

    async def get(self, path_id: str) -> LearningPathRecord | None:
        if not isinstance(path_id, str) or not _PATH_ID_PATTERN.fullmatch(path_id):
            return None
        return await self._run_thread(self._get_sync, path_id)

    async def get_current(
        self,
        user_id: str,
        document_id: str | None = None,
    ) -> LearningPathRecord | None:
        user_id = self._normalize_identifier(user_id, "user_id", 128)
        if document_id is not None:
            document_id = self._normalize_identifier(
                document_id, "document_id", 512
            )
        return await self._run_thread(
            self._get_current_sync,
            user_id,
            document_id,
        )

    async def _run_thread(self, function, *args):
        worker = asyncio.create_task(asyncio.to_thread(function, *args))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError as cancelled:
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    continue
            try:
                worker.result()
            except BaseException:
                raise cancelled
            raise cancelled

    def _storage_may_have_records(self) -> bool:
        return bool(
            self._database_url
            or self._schema_ready
            or Path(self._sqlite_path).exists()
        )

    @staticmethod
    def _normalize_identifier(value: str, field: str, max_length: int) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{field} must be a string")
        normalized = value.strip()
        if not normalized or len(normalized) > max_length:
            raise ValueError(f"invalid {field}")
        return normalized

    @staticmethod
    def _key_hash(idempotency_key: str) -> str:
        if (
            not isinstance(idempotency_key, str)
            or not idempotency_key.strip()
            or len(idempotency_key) > 1024
        ):
            raise ValueError("invalid Idempotency-Key")
        return hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()

    @staticmethod
    def _validate_fingerprint(request_fingerprint: str) -> str:
        if not LearningPathStore._is_sha256(request_fingerprint):
            raise ValueError("request_fingerprint must be a lowercase SHA-256 hex digest")
        return request_fingerprint

    @staticmethod
    def _is_sha256(value) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    def _new_path_id(self) -> str:
        path_id = self._path_id_factory()
        if not isinstance(path_id, str) or not _PATH_ID_PATTERN.fullmatch(path_id):
            raise ValueError("path_id_factory returned an invalid path_id")
        return path_id

    def _serialize_path(self, path: LearningPath, document_id: str) -> str:
        if not isinstance(path, LearningPath):
            raise TypeError("path must be a LearningPath")
        try:
            validated = LearningPath.model_validate(path.model_dump(mode="json"))
        except (TypeError, ValidationError, ValueError) as exc:
            raise ValueError("invalid LearningPath payload") from exc
        if validated.document_id != document_id:
            raise ValueError("LearningPath document_id does not match the row")
        payload_json = self._canonical_json(validated.model_dump(mode="json"))
        payload_bytes = len(payload_json.encode("utf-8"))
        if payload_bytes > self._max_payload_bytes:
            raise LearningPathPayloadTooLargeError(
                f"learning path payload is {payload_bytes} bytes; "
                f"limit is {self._max_payload_bytes}"
            )
        return payload_json

    @staticmethod
    def _canonical_json(value) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @staticmethod
    def _immutable_hash(
        path_id: str,
        user_id: str,
        document_id: str,
        payload_json: str,
        idempotency_key_hash: str,
        request_fingerprint: str,
        created_at: float,
    ) -> str:
        canonical = LearningPathStore._canonical_json(
            {
                "document_id": document_id,
                "created_at": created_at,
                "idempotency_key_hash": idempotency_key_hash,
                "path_id": path_id,
                "payload_json": payload_json,
                "request_fingerprint": request_fingerprint,
                "schema_version": _SCHEMA_VERSION,
                "status": _READY_STATUS,
                "user_id": user_id,
            }
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _now(self) -> float:
        try:
            now = float(self._clock())
        except (TypeError, ValueError) as exc:
            raise ValueError("learning path clock returned an invalid value") from exc
        if not math.isfinite(now) or now < 0:
            raise ValueError("learning path clock returned an invalid value")
        return now

    def _connect(self):
        if self._database_url:
            import psycopg

            return psycopg.connect(
                self._database_url,
                connect_timeout=self._postgres_connect_timeout_seconds,
                options=(
                    f"-c lock_timeout={self._postgres_lock_timeout_ms}ms "
                    f"-c statement_timeout={self._postgres_statement_timeout_ms}ms"
                ),
            )

        path = Path(self._sqlite_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            path,
            timeout=self._sqlite_busy_timeout_ms / 1000,
        )
        connection.execute(f"PRAGMA busy_timeout = {self._sqlite_busy_timeout_ms}")
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
        with self._schema_lock:
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
                        path_id TEXT PRIMARY KEY,
                        schema_version INTEGER NOT NULL
                            CHECK (schema_version = 1),
                        user_id TEXT NOT NULL
                            CHECK (length(user_id) BETWEEN 1 AND 128),
                        document_id TEXT NOT NULL
                            CHECK (length(document_id) BETWEEN 1 AND 512),
                        payload_json TEXT NOT NULL,
                        status TEXT NOT NULL CHECK (status = 'ready'),
                        immutable_hash TEXT NOT NULL
                            CHECK (length(immutable_hash) = 64),
                        idempotency_key_hash TEXT NOT NULL UNIQUE
                            CHECK (length(idempotency_key_hash) = 64),
                        request_fingerprint TEXT NOT NULL
                            CHECK (length(request_fingerprint) = 64),
                        created_at DOUBLE PRECISION NOT NULL
                            CHECK (created_at >= 0)
                    )
                    """
                )
                connection.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {_TABLE}_user_current_idx
                    ON {_TABLE} (user_id, created_at DESC, path_id DESC)
                    """
                )
                connection.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {_TABLE}_user_document_current_idx
                    ON {_TABLE} (
                        user_id, document_id, created_at DESC, path_id DESC
                    )
                    """
                )
            self._schema_ready = True

    def _create_sync(
        self,
        path_id: str,
        user_id: str,
        document_id: str,
        payload_json: str,
        key_hash: str,
        request_fingerprint: str,
    ) -> LearningPathRecord:
        self._ensure_schema()
        p = self._placeholder
        with self._transaction(write=True) as connection:
            existing = self._select_by_creation(connection, key_hash)
            if existing is not None:
                record = self._row_to_record(existing)
                self._validate_creation_binding(
                    existing,
                    key_hash,
                    user_id,
                    document_id,
                    request_fingerprint,
                )
                return record

            now = self._now()
            immutable_hash = self._immutable_hash(
                path_id,
                user_id,
                document_id,
                payload_json,
                key_hash,
                request_fingerprint,
                now,
            )
            cursor = connection.execute(
                f"""
                INSERT INTO {_TABLE} (
                    path_id, schema_version, user_id, document_id,
                    payload_json, status, immutable_hash,
                    idempotency_key_hash, request_fingerprint,
                    created_at
                ) VALUES (
                    {p}, 1, {p}, {p}, {p}, 'ready', {p},
                    {p}, {p}, {p}
                )
                ON CONFLICT (idempotency_key_hash) DO NOTHING
                """,
                (
                    path_id,
                    user_id,
                    document_id,
                    payload_json,
                    immutable_hash,
                    key_hash,
                    request_fingerprint,
                    now,
                ),
            )
            if cursor.rowcount == 1:
                row = self._select_by_id(connection, path_id)
            else:
                row = self._select_by_creation(connection, key_hash)
            if row is None:
                raise RuntimeError("learning path disappeared after create")
            record = self._row_to_record(row)
            self._validate_creation_binding(
                row,
                key_hash,
                user_id,
                document_id,
                request_fingerprint,
            )
            return record

    def _find_by_creation_sync(
        self,
        key_hash: str,
        user_id: str,
        document_id: str,
        request_fingerprint: str,
    ) -> LearningPathRecord | None:
        self._ensure_schema()
        with self._transaction() as connection:
            row = self._select_by_creation(connection, key_hash)
        if row is None:
            return None
        record = self._row_to_record(row)
        self._validate_creation_binding(
            row,
            key_hash,
            user_id,
            document_id,
            request_fingerprint,
        )
        return record

    def _get_sync(self, path_id: str) -> LearningPathRecord | None:
        self._ensure_schema()
        with self._transaction() as connection:
            row = self._select_by_id(connection, path_id)
        return None if row is None else self._row_to_record(row)

    def _get_current_sync(
        self,
        user_id: str,
        document_id: str | None,
    ) -> LearningPathRecord | None:
        self._ensure_schema()
        p = self._placeholder
        where_document = "" if document_id is None else f" AND document_id = {p}"
        parameters = (user_id,) if document_id is None else (user_id, document_id)
        with self._transaction() as connection:
            row = connection.execute(
                f"""
                {self._select_columns()}
                WHERE user_id = {p}{where_document}
                ORDER BY created_at DESC, path_id DESC
                LIMIT 1
                """,
                parameters,
            ).fetchone()
        return None if row is None else self._row_to_record(row)

    @staticmethod
    def _select_columns() -> str:
        return f"""
            SELECT path_id, schema_version, user_id, document_id,
                   payload_json, status, immutable_hash,
                   idempotency_key_hash, request_fingerprint,
                   created_at
            FROM {_TABLE}
        """

    def _select_by_id(self, connection, path_id: str):
        p = self._placeholder
        return connection.execute(
            f"{self._select_columns()} WHERE path_id = {p}",
            (path_id,),
        ).fetchone()

    def _select_by_creation(self, connection, key_hash: str):
        p = self._placeholder
        return connection.execute(
            f"{self._select_columns()} WHERE idempotency_key_hash = {p}",
            (key_hash,),
        ).fetchone()

    @staticmethod
    def _validate_creation_binding(
        row,
        key_hash: str,
        user_id: str,
        document_id: str,
        request_fingerprint: str,
    ) -> None:
        if not (
            hmac.compare_digest(row[7], key_hash)
            and row[2] == user_id
            and row[3] == document_id
            and hmac.compare_digest(row[8], request_fingerprint)
        ):
            raise LearningPathCreationConflictError("payload_mismatch")

    def _decode_payload(self, payload_json: str) -> LearningPath:
        try:
            if not isinstance(payload_json, str):
                raise TypeError("learning path payload is not text")
            if len(payload_json.encode("utf-8")) > self._max_payload_bytes:
                raise ValueError("learning path payload exceeds the durable limit")
            raw = json.loads(payload_json)
            if not isinstance(raw, dict):
                raise TypeError("learning path payload is not a JSON object")
            path = LearningPath.model_validate(raw)
            if self._canonical_json(path.model_dump(mode="json")) != payload_json:
                raise ValueError("learning path payload is not canonical JSON")
            return path
        except (
            TypeError,
            json.JSONDecodeError,
            UnicodeError,
            ValidationError,
            ValueError,
        ) as exc:
            raise LearningPathCorruptError(
                "invalid durable LearningPath payload"
            ) from exc

    def _row_to_record(self, row) -> LearningPathRecord:
        try:
            path_id = self._stored_identifier(row[0], "path_id", 128)
            if not _PATH_ID_PATTERN.fullmatch(path_id):
                raise ValueError("invalid stored path_id format")
            if row[1] != _SCHEMA_VERSION:
                raise ValueError("unsupported schema version")
            user_id = self._stored_identifier(row[2], "user_id", 128)
            document_id = self._stored_identifier(row[3], "document_id", 512)
            if row[5] != _READY_STATUS:
                raise ValueError("invalid learning path status")
            if not self._is_sha256(row[6]):
                raise ValueError("invalid immutable hash")
            if not self._is_sha256(row[7]):
                raise ValueError("invalid idempotency key hash")
            if not self._is_sha256(row[8]):
                raise ValueError("invalid request fingerprint")
            created_at = float(row[9])
            if not math.isfinite(created_at) or created_at < 0:
                raise ValueError("invalid creation timestamp")
            path = self._decode_payload(row[4])
            if path.document_id != document_id:
                raise ValueError("row document_id does not match payload")
            expected_hash = self._immutable_hash(
                path_id,
                user_id,
                document_id,
                row[4],
                row[7],
                row[8],
                created_at,
            )
            if not hmac.compare_digest(row[6], expected_hash):
                raise ValueError("immutable learning path hash mismatch")
        except LearningPathCorruptError:
            raise
        except (TypeError, ValueError, OverflowError) as exc:
            raise LearningPathCorruptError(
                "invalid durable LearningPath record"
            ) from exc

        return LearningPathRecord(
            path_id=path_id,
            user_id=user_id,
            document_id=document_id,
            path=path,
            schema_version=_SCHEMA_VERSION,
            created_at=created_at,
        )

    @staticmethod
    def _stored_identifier(value, field: str, max_length: int) -> str:
        if (
            not isinstance(value, str)
            or not value
            or len(value) > max_length
            or value != value.strip()
        ):
            raise ValueError(f"invalid stored {field}")
        return value


learning_path_store = LearningPathStore.from_environment()


__all__ = [
    "LearningPathCorruptError",
    "LearningPathCreationConflictError",
    "LearningPathPayloadTooLargeError",
    "LearningPathRecord",
    "LearningPathStore",
    "learning_path_store",
]
