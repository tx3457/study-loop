"""Live PostgreSQL cold-start tests for the learner memory store.

Set TEST_DATABASE_URL to run these tests. The default suite skips them so local
development does not require PostgreSQL.
"""

import os
import subprocess
import sys
import unittest
import uuid
from pathlib import Path

try:
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo
except ModuleNotFoundError:  # Local unit-test environments may omit this extra.
    psycopg = None
    sql = None
    make_conninfo = None


TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(
    TEST_DATABASE_URL and psycopg is not None,
    "TEST_DATABASE_URL or psycopg is not configured",
)
class TestPostgresMemoryStore(unittest.TestCase):
    @staticmethod
    def _admin_connection(*, autocommit=True):
        return psycopg.connect(
            TEST_DATABASE_URL,
            autocommit=autocommit,
            connect_timeout=5,
            options="-c lock_timeout=5000 -c statement_timeout=15000",
        )

    def _initialize_schema_in_child(self, schema):
        database_url = make_conninfo(
            TEST_DATABASE_URL,
            options=f"-csearch_path={schema}",
        )
        child_env = os.environ.copy()
        child_env["DATABASE_URL"] = database_url
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                ("import services.memory as memory; memory._store_ctx.__exit__(None, None, None)"),
            ],
            cwd=PROJECT_ROOT,
            env=child_env,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_two_processes_initialize_a_fresh_schema_concurrently(self):
        schema = f"studyloop_memory_test_{uuid.uuid4().hex}"
        database_url = make_conninfo(
            TEST_DATABASE_URL,
            options=f"-csearch_path={schema}",
        )
        child_code = """
import os
import sys

sys.stdin.readline()
import services.memory as memory

key = str(os.getpid())
memory.store.put(("runtime", "probe"), key, {"ok": True})
assert memory.store.get(("runtime", "probe"), key).value == {"ok": True}
memory._store_ctx.__exit__(None, None, None)
"""
        processes = []

        with self._admin_connection() as connection:
            connection.execute(
                sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema))
            )

        try:
            child_env = os.environ.copy()
            child_env["DATABASE_URL"] = database_url
            for _ in range(2):
                processes.append(
                    subprocess.Popen(
                        [sys.executable, "-c", child_code],
                        cwd=PROJECT_ROOT,
                        env=child_env,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                )
            for process in processes:
                process.stdin.write("\n")
                process.stdin.flush()

            results = [process.communicate(timeout=60) for process in processes]
            self.assertEqual(
                [process.returncode for process in processes],
                [0, 0],
                "\n".join(stderr for _, stderr in results),
            )
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.wait()
            with self._admin_connection() as connection:
                connection.execute(
                    sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                        sql.Identifier(schema)
                    )
                )

    def test_runtime_conninfo_enforces_statement_and_lock_timeouts(self):
        import services.memory as memory

        schema = f"studyloop_memory_timeout_{uuid.uuid4().hex}"
        config = memory._PostgresConnectionConfig(
            connect_timeout_seconds=5,
            lock_timeout_ms=100,
            statement_timeout_ms=150,
            setup_statement_timeout_seconds=30,
            tcp_user_timeout_ms=5_000,
            io_wait_timeout_seconds=1,
            cancel_drain_timeout_seconds=1,
        )
        runtime_conninfo = memory._bounded_postgres_conninfo(
            TEST_DATABASE_URL,
            config=config,
        )

        with psycopg.connect(runtime_conninfo, autocommit=True) as connection:
            with self.assertRaises(psycopg.errors.QueryCanceled):
                connection.execute("SELECT pg_sleep(2)")
            self.assertEqual(connection.execute("SELECT 1").fetchone(), (1,))

        with self._admin_connection() as admin:
            admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
            admin.execute(
                sql.SQL("CREATE TABLE {}.locked_row (id integer PRIMARY KEY)").format(
                    sql.Identifier(schema)
                )
            )
            admin.execute(
                sql.SQL("INSERT INTO {}.locked_row (id) VALUES (1)").format(sql.Identifier(schema))
            )

        blocker = self._admin_connection(autocommit=False)
        try:
            blocker.execute(
                sql.SQL("SELECT id FROM {}.locked_row WHERE id = 1 FOR UPDATE").format(
                    sql.Identifier(schema)
                )
            )
            with psycopg.connect(runtime_conninfo, autocommit=True) as contender:
                with self.assertRaises(psycopg.errors.LockNotAvailable):
                    contender.execute(
                        sql.SQL("UPDATE {}.locked_row SET id = id WHERE id = 1").format(
                            sql.Identifier(schema)
                        )
                    )

                blocker.rollback()
                contender.execute(
                    sql.SQL("UPDATE {}.locked_row SET id = id WHERE id = 1").format(
                        sql.Identifier(schema)
                    )
                )
        finally:
            blocker.rollback()
            blocker.close()
            with self._admin_connection() as admin:
                admin.execute(
                    sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema))
                )

    def test_index_repair_ignores_same_named_index_in_earlier_search_path(self):
        import services.memory as memory

        shadow_schema = f"studyloop_memory_shadow_{uuid.uuid4().hex}"
        store_schema = f"studyloop_memory_store_{uuid.uuid4().hex}"
        with self._admin_connection() as admin:
            admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(shadow_schema)))
            admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(store_schema)))

        try:
            self._initialize_schema_in_child(store_schema)
            with self._admin_connection() as admin:
                admin.execute(
                    sql.SQL("CREATE TABLE {}.decoy (value text)").format(
                        sql.Identifier(shadow_schema)
                    )
                )
                admin.execute(
                    sql.SQL("CREATE INDEX {} ON {}.decoy (value)").format(
                        sql.Identifier("store_prefix_idx"),
                        sql.Identifier(shadow_schema),
                    )
                )
                admin.execute(
                    sql.SQL("DROP INDEX {}.{}").format(
                        sql.Identifier(store_schema),
                        sql.Identifier("store_prefix_idx"),
                    )
                )

            search_conninfo = make_conninfo(
                TEST_DATABASE_URL,
                options=(f"-csearch_path={shadow_schema},{store_schema} -cstatement_timeout=15000"),
            )
            with psycopg.connect(
                search_conninfo,
                autocommit=True,
                connect_timeout=5,
            ) as connection:
                self.assertIsNone(memory._store_prefix_index_state(connection))
                memory._ensure_valid_store_prefix_index(connection)
                self.assertEqual(
                    memory._store_prefix_index_state(connection),
                    (store_schema, "store_prefix_idx", True, True),
                )
                rows = connection.execute(
                    """
                    SELECT namespace.nspname
                    FROM pg_catalog.pg_class AS relation
                    JOIN pg_catalog.pg_namespace AS namespace
                      ON namespace.oid = relation.relnamespace
                    WHERE relation.relname = 'store_prefix_idx'
                      AND namespace.nspname IN (%s, %s)
                    ORDER BY namespace.nspname
                    """,
                    (shadow_schema, store_schema),
                ).fetchall()
                self.assertEqual(
                    [row[0] for row in rows],
                    sorted((shadow_schema, store_schema)),
                )
        finally:
            with self._admin_connection() as admin:
                admin.execute(
                    sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                        sql.Identifier(shadow_schema)
                    )
                )
                admin.execute(
                    sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(store_schema))
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
