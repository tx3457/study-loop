from __future__ import annotations

import os
from pathlib import Path

import pytest

from .support import sdk_contract_violations


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers", "asyncio: execute with pytest-asyncio in the isolated contract venv"
    )
    config.addinivalue_line(
        "markers", "lightrag_live: requires the real LightRAG PostgreSQL contract"
    )


def pytest_collection_modifyitems(config, items) -> None:
    """Block live scenarios after preflight without hiding the gate failure."""
    contract_root = Path(__file__).resolve().parent
    items = [item for item in items if contract_root in Path(item.path).resolve().parents]
    if not os.environ.get("TEST_LIGHTRAG_DATABASE_URL"):
        marker = pytest.mark.skip(
            reason="TEST_LIGHTRAG_DATABASE_URL is not configured for the isolated gate"
        )
        for item in items:
            item.add_marker(marker)
        return

    violations = sdk_contract_violations()
    if not violations:
        return
    reason = "LightRAG contract preflight failed: " + "; ".join(violations)
    marker = pytest.mark.skip(reason=reason)
    for item in items:
        if item.get_closest_marker("lightrag_live") is not None:
            item.add_marker(marker)


@pytest.fixture(scope="session")
def contract_database_url() -> str:
    value = os.environ.get("TEST_LIGHTRAG_DATABASE_URL")
    if not value:
        pytest.fail(
            "TEST_LIGHTRAG_DATABASE_URL must identify a disposable PostgreSQL 16 "
            "+ pgvector database; production DATABASE_URL and .env are forbidden",
            pytrace=False,
        )
    return value
