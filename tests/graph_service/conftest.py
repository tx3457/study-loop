from __future__ import annotations

import os
from pathlib import Path

try:
    import pytest_asyncio
except ImportError:
    import pytest

    async_fixture = pytest.fixture
else:
    async_fixture = pytest_asyncio.fixture

from graph_service.config import Settings


DATABASE_URL = os.environ.get("TEST_GRAPH_SERVICE_DATABASE_URL")


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers", "lightrag_live: uses pinned LightRAG with disposable PostgreSQL"
    )


@async_fixture
async def repository(tmp_path: Path):
    if not DATABASE_URL:
        import pytest

        pytest.skip("TEST_GRAPH_SERVICE_DATABASE_URL is not configured")
    import asyncpg

    from graph_service.repository import Repository

    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=4)
    repo = Repository(pool)
    await repo.migrate()
    cleanup = (
        "TRUNCATE sl_idempotency, sl_corrections, sl_edge_aliases, sl_edges, "
        "sl_entity_aliases, sl_entities, "
        "sl_jobs, sl_source_versions, sl_documents, sl_web_snapshots, "
        "sl_knowledge_bases CASCADE"
    )
    await pool.execute(cleanup)
    try:
        yield repo
    finally:
        await pool.execute(cleanup)
        await pool.close()


@async_fixture
async def settings(tmp_path: Path):
    if not DATABASE_URL:
        import pytest

        pytest.skip("TEST_GRAPH_SERVICE_DATABASE_URL is not configured")
    return Settings(
        database_url=DATABASE_URL,
        internal_token="test-token",
        materials_dir=tmp_path / "materials",
        working_dir=tmp_path / "workspaces",
        provider="test",
        llm_model="deterministic",
        embedding_model="deterministic",
        embedding_dim=64,
    )
