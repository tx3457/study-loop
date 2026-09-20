#!/usr/bin/env python3
"""Run the isolated real-LightRAG contract gate from a clean /tmp cwd."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = REPO_ROOT / "tests" / "lightrag_contract"


def _configure_disposable_postgres() -> None:
    dsn = os.environ.get("TEST_LIGHTRAG_DATABASE_URL")
    if not dsn:
        raise SystemExit(
            "TEST_LIGHTRAG_DATABASE_URL is required and must reference the "
            "disposable LightRAG contract database"
        )
    parsed = urlparse(dsn)
    if parsed.scheme not in {"postgres", "postgresql"}:
        raise SystemExit("TEST_LIGHTRAG_DATABASE_URL must use postgres:// or postgresql://")
    if not parsed.hostname or not parsed.username or not parsed.path.strip("/"):
        raise SystemExit(
            "TEST_LIGHTRAG_DATABASE_URL must include host, user, and database"
        )

    for name in tuple(os.environ):
        if name.startswith("POSTGRES_"):
            os.environ.pop(name, None)
    os.environ.update(
        {
            "POSTGRES_HOST": parsed.hostname,
            "POSTGRES_PORT": str(parsed.port or 5432),
            "POSTGRES_USER": unquote(parsed.username),
            "POSTGRES_PASSWORD": unquote(parsed.password or ""),
            "POSTGRES_DATABASE": unquote(parsed.path.lstrip("/")),
            "POSTGRES_SSL_MODE": "disable",
        }
    )
    os.environ.pop("DATABASE_URL", None)


def main() -> int:
    _configure_disposable_postgres()
    with tempfile.TemporaryDirectory(prefix="study-loop-lightrag-contract-") as cwd:
        os.chdir(cwd)
        import pytest

        return pytest.main(
            [
                "-q",
                "-ra",
                "--strict-markers",
                f"--confcutdir={TEST_ROOT}",
                "-o",
                "markers=lightrag_live: requires the real LightRAG PostgreSQL contract",
                str(TEST_ROOT),
                *sys.argv[1:],
            ]
        )


if __name__ == "__main__":
    raise SystemExit(main())
