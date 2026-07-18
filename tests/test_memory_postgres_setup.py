import unittest
from unittest.mock import patch

from services.memory import (
    _POSTGRES_STORE_SETUP_LOCK_ID,
    _enter_postgres_store,
    _setup_postgres_store,
)


class TestPostgresMemorySetup(unittest.TestCase):
    def test_store_context_closes_when_initialization_fails(self):
        error = RuntimeError("migration failed")
        events = []

        class RecordingContext:
            def __enter__(self):
                events.append("enter")
                return "store"

            def __exit__(self, exc_type, exc, traceback):
                events.append(("exit", exc_type, exc))

        with patch("services.memory._setup_postgres_store", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "migration failed"):
                _enter_postgres_store(RecordingContext(), "postgresql://memory-test")

        self.assertEqual(events, ["enter", ("exit", RuntimeError, error)])

    def test_setup_runs_while_session_lock_is_held(self):
        events = []

        class RecordingCursor:
            def fetchone(self):
                return (True,)

        class RecordingConnection:
            def __enter__(self):
                events.append("enter")
                return self

            def execute(self, statement, params):
                events.append(("lock", statement, params))
                return RecordingCursor()

            def __exit__(self, exc_type, exc, traceback):
                events.append("exit")

        class RecordingStore:
            def setup(self):
                events.append("setup")

        def connect(database_url, *, autocommit):
            events.append(("connect", database_url, autocommit))
            return RecordingConnection()

        with patch("psycopg.connect", side_effect=connect):
            _setup_postgres_store(RecordingStore(), "postgresql://memory-test")

        self.assertEqual(
            events,
            [
                ("connect", "postgresql://memory-test", True),
                "enter",
                (
                    "lock",
                    "SELECT pg_try_advisory_lock(%s)",
                    (_POSTGRES_STORE_SETUP_LOCK_ID,),
                ),
                "setup",
                "exit",
            ],
        )

    def test_lock_connection_closes_when_setup_fails(self):
        events = []

        class RecordingCursor:
            def fetchone(self):
                return (True,)

        class RecordingConnection:
            def __enter__(self):
                events.append("enter")
                return self

            def execute(self, statement, params):
                events.append("lock")
                return RecordingCursor()

            def __exit__(self, exc_type, exc, traceback):
                events.append(("exit", exc_type))

        class FailingStore:
            def setup(self):
                raise RuntimeError("migration failed")

        with patch("psycopg.connect", return_value=RecordingConnection()):
            with self.assertRaisesRegex(RuntimeError, "migration failed"):
                _setup_postgres_store(FailingStore(), "postgresql://memory-test")

        self.assertEqual(events, ["enter", "lock", ("exit", RuntimeError)])

    def test_busy_lock_is_polled_without_blocking(self):
        attempts = iter((False, False, True))

        class RecordingCursor:
            def __init__(self, acquired):
                self.acquired = acquired

            def fetchone(self):
                return (self.acquired,)

        class RecordingConnection:
            def __enter__(self):
                return self

            def execute(self, statement, params):
                return RecordingCursor(next(attempts))

            def __exit__(self, exc_type, exc, traceback):
                return None

        store = unittest.mock.Mock()
        with (
            patch("psycopg.connect", return_value=RecordingConnection()),
            patch("services.memory.time.sleep") as sleep,
        ):
            _setup_postgres_store(store, "postgresql://memory-test")

        self.assertEqual(sleep.call_count, 2)
        sleep.assert_called_with(0.05)
        store.setup.assert_called_once_with()


if __name__ == "__main__":
    unittest.main(verbosity=2)
