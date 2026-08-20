import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from services.memory import (
    PostgresStoreSetupLockTimeoutError,
    _POSTGRES_STORE_SETUP_LOCK_ID,
    _PostgresConnectionConfig,
    _bounded_postgres_conninfo,
    _enter_postgres_store,
    _initialize_postgres_store,
    _postgres_connection_config,
    _setup_postgres_store,
)


class _RecordingStoreContext:
    def __init__(
        self,
        name,
        events,
        *,
        enter_value=None,
        enter_error=None,
        exit_error=None,
    ):
        self.name = name
        self.events = events
        self.enter_value = enter_value if enter_value is not None else f"{name}-store"
        self.enter_error = enter_error
        self.exit_error = exit_error

    def __enter__(self):
        self.events.append((self.name, "enter"))
        if self.enter_error is not None:
            raise self.enter_error
        return self.enter_value

    def __exit__(self, exc_type, exc, traceback):
        self.events.append((self.name, "exit", exc_type, exc))
        if self.exit_error is not None:
            raise self.exit_error


class _RecordingStoreType:
    def __init__(self, contexts, events):
        self.contexts = iter(contexts)
        self.events = events
        self.calls = 0

    def from_conn_string(self, conninfo):
        self.calls += 1
        self.events.append(("from_conn_string", conninfo))
        return next(self.contexts)


class _FakeIdentifier:
    def __init__(self, value):
        self.value = value

    def render(self):
        return f'"{str(self.value).replace(chr(34), chr(34) * 2)}"'


class _FakeSQL(str):
    def format(self, *identifiers):
        rendered = str(self)
        for identifier in identifiers:
            rendered = rendered.replace("{}", identifier.render(), 1)
        return rendered


class TestPostgresMemorySetup(unittest.TestCase):
    def setUp(self):
        fake_psycopg = types.ModuleType("psycopg")
        fake_psycopg.connect = MagicMock()
        fake_conninfo = types.ModuleType("psycopg.conninfo")
        fake_conninfo.conninfo_to_dict = MagicMock(return_value={})
        fake_conninfo.make_conninfo = MagicMock(
            side_effect=lambda database_url, **_kwargs: database_url
        )
        fake_psycopg.conninfo = fake_conninfo
        fake_psycopg.sql = types.SimpleNamespace(
            SQL=_FakeSQL,
            Identifier=_FakeIdentifier,
        )
        self.conninfo_to_dict = fake_conninfo.conninfo_to_dict
        self.make_conninfo = fake_conninfo.make_conninfo
        self.psycopg_module = patch.dict(
            sys.modules,
            {"psycopg": fake_psycopg, "psycopg.conninfo": fake_conninfo},
        )
        self.psycopg_module.start()

    def tearDown(self):
        self.psycopg_module.stop()

    @staticmethod
    def _config(**overrides):
        values = {
            "connect_timeout_seconds": 7.1,
            "lock_timeout_ms": 1_234,
            "statement_timeout_ms": 5_678,
            "setup_statement_timeout_seconds": 9.1,
            "tcp_user_timeout_ms": 12_345,
            "io_wait_timeout_seconds": 2.5,
            "cancel_drain_timeout_seconds": 3.5,
        }
        values.update(overrides)
        return _PostgresConnectionConfig(**values)

    def test_runtime_timeout_environment_is_loaded_exactly(self):
        with patch.dict(
            os.environ,
            {
                "MEMORY_STORE_PG_CONNECT_TIMEOUT_SECONDS": "7.25",
                "MEMORY_STORE_PG_LOCK_TIMEOUT_MS": "1234",
                "MEMORY_STORE_PG_STATEMENT_TIMEOUT_MS": "5678",
                "MEMORY_STORE_PG_SETUP_STATEMENT_TIMEOUT_SECONDS": "91.5",
                "MEMORY_STORE_PG_TCP_USER_TIMEOUT_MS": "9876",
                "MEMORY_STORE_PG_IO_WAIT_TIMEOUT_SECONDS": "4.5",
                "MEMORY_STORE_PG_CANCEL_DRAIN_TIMEOUT_SECONDS": "6.5",
            },
            clear=True,
        ):
            config = _postgres_connection_config()

        self.assertEqual(config.connect_timeout_seconds, 7.25)
        self.assertEqual(config.lock_timeout_ms, 1234)
        self.assertEqual(config.statement_timeout_ms, 5678)
        self.assertEqual(config.setup_statement_timeout_seconds, 91.5)
        self.assertEqual(config.tcp_user_timeout_ms, 9876)
        self.assertEqual(config.io_wait_timeout_seconds, 4.5)
        self.assertEqual(config.cancel_drain_timeout_seconds, 6.5)

    def test_invalid_runtime_timeout_environment_is_rejected(self):
        float_names = (
            "MEMORY_STORE_PG_CONNECT_TIMEOUT_SECONDS",
            "MEMORY_STORE_PG_SETUP_STATEMENT_TIMEOUT_SECONDS",
            "MEMORY_STORE_PG_IO_WAIT_TIMEOUT_SECONDS",
            "MEMORY_STORE_PG_CANCEL_DRAIN_TIMEOUT_SECONDS",
        )
        integer_names = (
            "MEMORY_STORE_PG_LOCK_TIMEOUT_MS",
            "MEMORY_STORE_PG_STATEMENT_TIMEOUT_MS",
            "MEMORY_STORE_PG_TCP_USER_TIMEOUT_MS",
        )
        for name in float_names:
            for value in ("", "0", "-1", "nan", "inf", "not-a-number"):
                with self.subTest(name=name, value=value):
                    with patch.dict(os.environ, {name: value}):
                        with self.assertRaisesRegex(ValueError, name):
                            _postgres_connection_config()
        for name in integer_names:
            for value in ("", "0", "-1", "1.5", "2147483648", "not-an-integer"):
                with self.subTest(name=name, value=value):
                    with patch.dict(os.environ, {name: value}):
                        with self.assertRaisesRegex(ValueError, name):
                            _postgres_connection_config()

    def test_conninfo_preserves_explicit_options_and_original_secret_dsn(self):
        database_url = (
            "postgresql://learner:private-password@db/studyloop"
            "?options=-csearch_path%3Dtenant_schema"
        )
        self.conninfo_to_dict.return_value = {
            "options": "-csearch_path=tenant_schema",
            "password": "private-password",
        }
        self.make_conninfo.side_effect = None
        self.make_conninfo.return_value = "bounded-runtime-conninfo"
        config = self._config()

        with patch.dict(
            os.environ,
            {"PGOPTIONS": "-csearch_path=ignored_environment_schema"},
        ):
            result = _bounded_postgres_conninfo(database_url, config=config)

        self.assertEqual(result, "bounded-runtime-conninfo")
        self.conninfo_to_dict.assert_called_once_with(database_url)
        self.make_conninfo.assert_called_once_with(
            database_url,
            connect_timeout=8,
            tcp_user_timeout=12_345,
            options=("-csearch_path=tenant_schema -c lock_timeout=1234 -c statement_timeout=5678"),
        )

    def test_conninfo_inherits_pgoptions_when_dsn_has_no_options(self):
        database_url = "postgresql://db/studyloop"
        self.conninfo_to_dict.return_value = {"dbname": "studyloop"}
        self.make_conninfo.side_effect = None
        self.make_conninfo.return_value = "bounded-runtime-conninfo"
        config = self._config()

        with patch.dict(
            os.environ,
            {"PGOPTIONS": "-csearch_path=environment_schema"},
        ):
            result = _bounded_postgres_conninfo(database_url, config=config)

        self.assertEqual(result, "bounded-runtime-conninfo")
        self.make_conninfo.assert_called_once_with(
            database_url,
            connect_timeout=8,
            tcp_user_timeout=12_345,
            options=(
                "-csearch_path=environment_schema -c lock_timeout=1234 -c statement_timeout=5678"
            ),
        )

    def test_setup_conninfo_uses_the_separate_migration_statement_budget(self):
        database_url = "postgresql://db/studyloop"
        self.conninfo_to_dict.return_value = {"dbname": "studyloop"}
        self.make_conninfo.side_effect = None
        self.make_conninfo.return_value = "bounded-setup-conninfo"

        result = _bounded_postgres_conninfo(
            database_url,
            setup=True,
            config=self._config(),
        )

        self.assertEqual(result, "bounded-setup-conninfo")
        self.make_conninfo.assert_called_once_with(
            database_url,
            connect_timeout=8,
            tcp_user_timeout=12_345,
            options="-c lock_timeout=1234 -c statement_timeout=9100",
        )

    def test_service_configuration_without_visible_options_fails_closed(self):
        config = self._config()
        cases = (
            ({"service": "private-service"}, {}),
            ({"dbname": "studyloop"}, {"PGSERVICE": "private-service"}),
        )

        for parsed, environment in cases:
            with self.subTest(parsed=parsed, environment=environment):
                self.conninfo_to_dict.reset_mock(return_value=True)
                self.conninfo_to_dict.return_value = parsed
                self.make_conninfo.reset_mock()
                secret_dsn = "service=private-service password=do-not-log"
                with patch.dict(os.environ, environment, clear=True):
                    with self.assertRaisesRegex(
                        ValueError,
                        "must expose connection options",
                    ) as raised:
                        _bounded_postgres_conninfo(secret_dsn, config=config)

                self.assertNotIn("do-not-log", str(raised.exception))
                self.assertNotIn(secret_dsn, str(raised.exception))
                self.make_conninfo.assert_not_called()

    def test_conninfo_parser_and_builder_errors_never_echo_secret_dsn(self):
        secret = "private-password-that-must-not-leak"
        database_url = f"postgresql://learner:{secret}@db/studyloop"
        config = self._config()

        self.conninfo_to_dict.side_effect = RuntimeError(f"could not parse {database_url}")
        with self.assertRaisesRegex(
            ValueError,
            "PostgreSQL learner-memory DATABASE_URL is invalid",
        ) as parse_error:
            _bounded_postgres_conninfo(database_url, config=config)
        self.assertNotIn(secret, str(parse_error.exception))
        self.assertIsNone(parse_error.exception.__cause__)

        self.conninfo_to_dict.side_effect = None
        self.conninfo_to_dict.return_value = {"dbname": "studyloop"}
        self.make_conninfo.side_effect = RuntimeError(f"could not build {database_url}")
        with self.assertRaisesRegex(
            ValueError,
            "PostgreSQL learner-memory DATABASE_URL is invalid",
        ) as build_error:
            _bounded_postgres_conninfo(database_url, config=config)
        self.assertNotIn(secret, str(build_error.exception))
        self.assertIsNone(build_error.exception.__cause__)

    def test_busy_lock_times_out_and_closes_connection(self):
        connection = MagicMock()
        connection.__enter__.return_value = connection
        connection.execute.return_value.fetchone.return_value = (False,)
        store = MagicMock()

        with (
            patch.dict(
                os.environ,
                {"MEMORY_STORE_SETUP_LOCK_TIMEOUT_SECONDS": "1"},
            ),
            patch("psycopg.connect", return_value=connection),
            patch("services.memory.time.monotonic", side_effect=(10.0, 10.4, 11.0)),
            patch("services.memory.time.sleep") as sleep,
        ):
            with self.assertRaisesRegex(
                PostgresStoreSetupLockTimeoutError,
                "Timed out after 1 second waiting for PostgreSQL learner-memory schema lock",
            ) as raised:
                _setup_postgres_store(store, "postgresql://memory-test")

        sleep.assert_called_once_with(0.05)
        self.assertNotIn("memory-test", str(raised.exception))
        store.setup.assert_not_called()
        connection.__exit__.assert_called_once()

    def test_setup_lock_connection_uses_bounded_setup_conninfo(self):
        connection = MagicMock()
        connection.__enter__.return_value = connection
        connection.execute.return_value.fetchone.return_value = (True,)
        store = MagicMock()
        config = self._config()

        with (
            patch(
                "services.memory._bounded_postgres_conninfo",
                return_value="bounded-setup-conninfo",
            ) as bounded_conninfo,
            patch("psycopg.connect", return_value=connection) as connect,
            patch(
                "services.memory._drop_invalid_store_prefix_index",
                return_value=False,
            ),
            patch("services.memory._ensure_valid_store_prefix_index"),
        ):
            _setup_postgres_store(
                store,
                "postgresql://memory-test",
                config=config,
            )

        bounded_conninfo.assert_called_once_with(
            "postgresql://memory-test",
            setup=True,
            config=config,
        )
        connect.assert_called_once_with("bounded-setup-conninfo", autocommit=True)
        store.setup.assert_called_once_with()

    def test_invalid_lock_timeout_configuration_fails_before_connecting(self):
        for value in ("", "0", "-1", "nan", "inf", "not-a-number"):
            with self.subTest(value=value):
                with (
                    patch.dict(
                        os.environ,
                        {"MEMORY_STORE_SETUP_LOCK_TIMEOUT_SECONDS": value},
                    ),
                    patch("psycopg.connect") as connect,
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        "MEMORY_STORE_SETUP_LOCK_TIMEOUT_SECONDS must be a positive number",
                    ):
                        _setup_postgres_store(MagicMock(), "postgresql://memory-test")

                connect.assert_not_called()

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

    def test_initialize_closes_setup_context_before_opening_runtime(self):
        events = []
        setup_context = _RecordingStoreContext("setup", events)
        runtime_store = object()
        runtime_context = _RecordingStoreContext(
            "runtime",
            events,
            enter_value=runtime_store,
        )
        store_type = _RecordingStoreType(
            [setup_context, runtime_context],
            events,
        )
        config = self._config()

        def bounded(_database_url, *, setup=False, config=None):
            return "setup-conninfo" if setup else "runtime-conninfo"

        with (
            patch(
                "services.memory._postgres_connection_config",
                return_value=config,
            ),
            patch(
                "services.memory._bounded_postgres_conninfo",
                side_effect=bounded,
            ),
            patch("services.memory._setup_postgres_store") as setup_store,
        ):
            returned_context, returned_store = _initialize_postgres_store(
                store_type,
                "postgresql://memory-test",
            )

        self.assertIs(returned_context, runtime_context)
        self.assertIs(returned_store, runtime_store)
        setup_store.assert_called_once_with(
            "setup-store",
            "postgresql://memory-test",
            config=config,
        )
        self.assertEqual(
            events,
            [
                ("from_conn_string", "setup-conninfo"),
                ("setup", "enter"),
                ("setup", "exit", None, None),
                ("from_conn_string", "runtime-conninfo"),
                ("runtime", "enter"),
            ],
        )

    def test_initialize_setup_failure_closes_context_and_preserves_error(self):
        events = []
        setup_cleanup_error = RuntimeError("setup cleanup failed")
        setup_context = _RecordingStoreContext(
            "setup",
            events,
            exit_error=setup_cleanup_error,
        )
        store_type = _RecordingStoreType([setup_context], events)
        setup_error = TimeoutError("setup statement timed out")

        with (
            patch(
                "services.memory._bounded_postgres_conninfo",
                return_value="setup-conninfo",
            ),
            patch(
                "services.memory._setup_postgres_store",
                side_effect=setup_error,
            ),
        ):
            with self.assertRaises(TimeoutError) as raised:
                _initialize_postgres_store(
                    store_type,
                    "postgresql://memory-test",
                )

        self.assertIs(raised.exception, setup_error)
        self.assertEqual(store_type.calls, 1)
        self.assertEqual(events[0:2], [("from_conn_string", "setup-conninfo"), ("setup", "enter")])
        self.assertEqual(events[2][0:3], ("setup", "exit", TimeoutError))
        self.assertIs(events[2][3], setup_error)

    def test_initialize_setup_close_failure_never_opens_runtime(self):
        events = []
        close_error = RuntimeError("setup close failed")
        setup_context = _RecordingStoreContext(
            "setup",
            events,
            exit_error=close_error,
        )
        store_type = _RecordingStoreType([setup_context], events)

        with (
            patch(
                "services.memory._bounded_postgres_conninfo",
                return_value="setup-conninfo",
            ),
            patch("services.memory._setup_postgres_store"),
        ):
            with self.assertRaises(RuntimeError) as raised:
                _initialize_postgres_store(
                    store_type,
                    "postgresql://memory-test",
                )

        self.assertIs(raised.exception, close_error)
        self.assertEqual(store_type.calls, 1)
        self.assertEqual(events[-1], ("setup", "exit", None, None))

    def test_initialize_runtime_enter_failure_closes_and_preserves_error(self):
        events = []
        setup_context = _RecordingStoreContext("setup", events)
        runtime_error = ConnectionError("runtime connection failed")
        cleanup_error = RuntimeError("runtime cleanup failed")
        runtime_context = _RecordingStoreContext(
            "runtime",
            events,
            enter_error=runtime_error,
            exit_error=cleanup_error,
        )
        store_type = _RecordingStoreType(
            [setup_context, runtime_context],
            events,
        )

        def bounded(_database_url, *, setup=False, config=None):
            return "setup-conninfo" if setup else "runtime-conninfo"

        with (
            patch(
                "services.memory._bounded_postgres_conninfo",
                side_effect=bounded,
            ),
            patch("services.memory._setup_postgres_store"),
        ):
            with self.assertRaises(ConnectionError) as raised:
                _initialize_postgres_store(
                    store_type,
                    "postgresql://memory-test",
                )

        self.assertIs(raised.exception, runtime_error)
        self.assertEqual(store_type.calls, 2)
        self.assertEqual(
            events[-2:],
            [
                ("runtime", "enter"),
                ("runtime", "exit", ConnectionError, runtime_error),
            ],
        )

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

        with (
            patch("psycopg.connect", side_effect=connect),
            patch(
                "services.memory._drop_invalid_store_prefix_index",
                return_value=False,
            ),
            patch("services.memory._ensure_valid_store_prefix_index"),
        ):
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

        with (
            patch("psycopg.connect", return_value=RecordingConnection()),
            patch(
                "services.memory._drop_invalid_store_prefix_index",
                return_value=False,
            ),
            patch("services.memory._ensure_valid_store_prefix_index"),
        ):
            with self.assertRaisesRegex(RuntimeError, "migration failed"):
                _setup_postgres_store(FailingStore(), "postgresql://memory-test")

        self.assertEqual(events, ["enter", "lock", ("exit", RuntimeError)])

    def test_timed_out_setup_drops_invalid_index_and_next_start_rebuilds_it(self):
        events = []
        setup_error = TimeoutError("setup statement timed out")

        class Cursor:
            def __init__(self, row):
                self.row = row

            def fetchone(self):
                return self.row

        class StatefulConnection:
            def __init__(self):
                self.index_state: tuple[str, str, bool, bool] | None = (
                    "tenant_schema",
                    "store_prefix_idx",
                    True,
                    True,
                )

            def __enter__(self):
                events.append("connection-enter")
                return self

            def execute(self, statement, params=None):
                normalized = " ".join(statement.split())
                if "pg_try_advisory_lock" in normalized:
                    events.append("lock")
                    return Cursor((True,))
                if "pg_catalog.pg_index" in normalized:
                    events.append(("inspect-index", self.index_state, normalized))
                    return Cursor(self.index_state)
                if normalized.startswith("DROP INDEX CONCURRENTLY"):
                    events.append(("drop-invalid-index", normalized))
                    self.index_state = None
                    return Cursor(None)
                if normalized.startswith("CREATE INDEX CONCURRENTLY"):
                    events.append("create-index")
                    self.index_state = (
                        "tenant_schema",
                        "store_prefix_idx",
                        True,
                        True,
                    )
                    return Cursor(None)
                raise AssertionError(f"unexpected SQL: {normalized}")

            def __exit__(self, exc_type, exc, traceback):
                events.append(("connection-exit", exc_type, exc))

        connection = StatefulConnection()

        class TimedOutStore:
            def setup(self):
                events.append("setup-timeout")
                connection.index_state = (
                    "tenant_schema",
                    "store_prefix_idx",
                    False,
                    False,
                )
                raise setup_error

        with patch("psycopg.connect", return_value=connection):
            with self.assertRaises(TimeoutError) as raised:
                _setup_postgres_store(
                    TimedOutStore(),
                    "postgresql://memory-test",
                )

        self.assertIs(raised.exception, setup_error)
        self.assertIsNone(connection.index_state)
        drop_event = next(
            event
            for event in events
            if isinstance(event, tuple) and event[0] == "drop-invalid-index"
        )
        self.assertEqual(
            drop_event[1],
            'DROP INDEX CONCURRENTLY IF EXISTS "tenant_schema"."store_prefix_idx"',
        )
        failure_exit = next(
            event for event in events if isinstance(event, tuple) and event[0] == "connection-exit"
        )
        self.assertEqual(failure_exit[1], TimeoutError)
        self.assertIs(failure_exit[2], setup_error)

        events.clear()

        class RecoveredStore:
            def setup(self):
                events.append("setup-retry")

        with patch("psycopg.connect", return_value=connection):
            _setup_postgres_store(
                RecoveredStore(),
                "postgresql://memory-test",
            )

        self.assertEqual(
            connection.index_state,
            ("tenant_schema", "store_prefix_idx", True, True),
        )
        self.assertIn("create-index", events)
        inspections = [
            event for event in events if isinstance(event, tuple) and event[0] == "inspect-index"
        ]
        self.assertGreaterEqual(len(inspections), 3)
        self.assertTrue(all("to_regclass('store')" in event[2] for event in inspections))
        self.assertTrue(
            all(
                "index_class.relnamespace = table_class.relnamespace" in event[2]
                for event in inspections
            )
        )
        self.assertTrue(
            all("to_regclass('store_prefix_idx')" not in event[2] for event in inspections)
        )
        self.assertEqual(
            inspections[-1][1],
            ("tenant_schema", "store_prefix_idx", True, True),
        )

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
            patch(
                "services.memory._drop_invalid_store_prefix_index",
                return_value=False,
            ),
            patch("services.memory._ensure_valid_store_prefix_index"),
        ):
            _setup_postgres_store(store, "postgresql://memory-test")

        self.assertEqual(sleep.call_count, 2)
        sleep.assert_called_with(0.05)
        store.setup.assert_called_once_with()


if __name__ == "__main__":
    unittest.main(verbosity=2)
