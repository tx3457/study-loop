"""Bounded PostgreSQL connection contracts for Quiz sessions.

The five other PostgreSQL-backed stores refuse to start when a service DSN
hides its connection options, so that their lock and statement timeouts cannot
be merged away silently. These tests hold the Quiz session store to the same
contract.
"""

from __future__ import annotations

import os
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

from services.quiz_sessions import QuizSessionStore


def _fake_psycopg(connect: MagicMock, conninfo_to_dict: MagicMock):
    psycopg_module = ModuleType("psycopg")
    psycopg_module.connect = connect
    conninfo_module = ModuleType("psycopg.conninfo")
    conninfo_module.conninfo_to_dict = conninfo_to_dict
    return {"psycopg": psycopg_module, "psycopg.conninfo": conninfo_module}


class TestQuizSessionRuntimeBounds(unittest.TestCase):
    def test_postgres_connection_preserves_options_and_overrides_unsafe_bounds(
        self,
    ) -> None:
        connect = MagicMock(return_value=SimpleNamespace())
        conninfo_to_dict = MagicMock(
            return_value={
                "options": (
                    "-csearch_path=tenant_schema -c lock_timeout=0 "
                    "-c statement_timeout=999999999"
                )
            }
        )
        database_url = (
            "postgresql://example/studyloop"
            "?connect_timeout=999&options=-csearch_path%3Dtenant_schema"
        )
        store = QuizSessionStore(
            database_url=database_url,
            postgres_connect_timeout_seconds=7,
            postgres_lock_timeout_ms=1234,
            postgres_statement_timeout_ms=5678,
        )

        with (
            patch.dict(sys.modules, _fake_psycopg(connect, conninfo_to_dict)),
            patch.dict(
                os.environ,
                {"PGOPTIONS": "-csearch_path=ignored_environment_schema"},
                clear=True,
            ),
        ):
            connection = store._connect()

        self.assertIs(connection, connect.return_value)
        conninfo_to_dict.assert_called_once_with(database_url)
        connect.assert_called_once_with(
            database_url,
            connect_timeout=7,
            options=(
                "-csearch_path=tenant_schema -c lock_timeout=0 "
                "-c statement_timeout=999999999 "
                "-c lock_timeout=1234ms -c statement_timeout=5678ms"
            ),
        )

    def test_postgres_connection_uses_pgoptions_without_dsn_options(self) -> None:
        connect = MagicMock(return_value=SimpleNamespace())
        conninfo_to_dict = MagicMock(return_value={"dbname": "studyloop"})
        database_url = "postgresql://example/studyloop"
        store = QuizSessionStore(
            database_url=database_url,
            postgres_connect_timeout_seconds=7,
            postgres_lock_timeout_ms=1234,
            postgres_statement_timeout_ms=5678,
        )

        with (
            patch.dict(sys.modules, _fake_psycopg(connect, conninfo_to_dict)),
            patch.dict(
                os.environ,
                {"PGOPTIONS": "-csearch_path=environment_schema"},
                clear=True,
            ),
        ):
            store._connect()

        connect.assert_called_once_with(
            database_url,
            connect_timeout=7,
            options=(
                "-csearch_path=environment_schema "
                "-c lock_timeout=1234ms -c statement_timeout=5678ms"
            ),
        )

    def test_hidden_service_options_and_parser_errors_are_sanitized(self) -> None:
        secret = "private-password-that-must-not-leak"
        database_url = f"service=private-service password={secret}"
        connect = MagicMock(return_value=SimpleNamespace())
        conninfo_to_dict = MagicMock(return_value={"service": "private-service"})
        store = QuizSessionStore(database_url=database_url)

        with (
            patch.dict(sys.modules, _fake_psycopg(connect, conninfo_to_dict)),
            patch.dict(os.environ, {"PGOPTIONS": "  \t"}, clear=True),
        ):
            with self.assertRaisesRegex(
                ValueError, "must expose connection options"
            ) as hidden:
                store._connect()

        self.assertNotIn(secret, str(hidden.exception))
        connect.assert_not_called()

        conninfo_to_dict.side_effect = RuntimeError(f"could not parse {database_url}")
        with patch.dict(sys.modules, _fake_psycopg(connect, conninfo_to_dict)):
            with self.assertRaisesRegex(
                ValueError, "PostgreSQL Quiz session DATABASE_URL is invalid"
            ) as invalid:
                store._connect()

        self.assertNotIn(secret, str(invalid.exception))
        self.assertIsNone(invalid.exception.__cause__)
        connect.assert_not_called()

    def test_service_dsn_is_accepted_when_pgoptions_exposes_options(self) -> None:
        connect = MagicMock(return_value=SimpleNamespace())
        conninfo_to_dict = MagicMock(return_value={"service": "private-service"})
        database_url = "service=private-service"
        store = QuizSessionStore(
            database_url=database_url,
            postgres_connect_timeout_seconds=7,
            postgres_lock_timeout_ms=1234,
            postgres_statement_timeout_ms=5678,
        )

        with (
            patch.dict(sys.modules, _fake_psycopg(connect, conninfo_to_dict)),
            patch.dict(
                os.environ, {"PGOPTIONS": "-csearch_path=service_schema"}, clear=True
            ),
        ):
            store._connect()

        connect.assert_called_once_with(
            database_url,
            connect_timeout=7,
            options=(
                "-csearch_path=service_schema "
                "-c lock_timeout=1234ms -c statement_timeout=5678ms"
            ),
        )


if __name__ == "__main__":
    unittest.main()
