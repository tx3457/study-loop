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

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo


TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(TEST_DATABASE_URL, "TEST_DATABASE_URL is not configured")
class TestPostgresMemoryStore(unittest.TestCase):
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

        with psycopg.connect(TEST_DATABASE_URL, autocommit=True) as connection:
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
            with psycopg.connect(TEST_DATABASE_URL, autocommit=True) as connection:
                connection.execute(
                    sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                        sql.Identifier(schema)
                    )
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
