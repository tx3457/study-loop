"""Keep the deterministic test suite isolated from developer data and providers."""

import os
import tempfile
from pathlib import Path

import pytest


_chroma_tmp = tempfile.TemporaryDirectory(prefix="study-loop-pytest-chroma-")
_state_tmp = tempfile.TemporaryDirectory(prefix="study-loop-pytest-state-")

# Set before test modules import application services. This prevents collection-time
# imports from opening a developer's configured Chroma directory or calling providers.
os.environ["CHROMA_DIR"] = _chroma_tmp.name
os.environ["MEMORY_SNAPSHOT_PATH"] = str(Path(_state_tmp.name) / "memory.json")
os.environ["IDEMPOTENCY_DB_PATH"] = str(Path(_state_tmp.name) / "idempotency.sqlite3")
os.environ["QUIZ_SESSION_DB_PATH"] = str(Path(_state_tmp.name) / "quiz-sessions.sqlite3")
os.environ["LEARNING_PATH_DB_PATH"] = str(Path(_state_tmp.name) / "learning-paths.sqlite3")
os.environ["DATABASE_URL"] = ""
os.environ["LLM_API_KEY"] = "test"
os.environ["LLM_BASE_URL"] = "http://127.0.0.1:9/v1"
os.environ["LLM_MODEL"] = "test"
os.environ["STRUCTURED_API_KEY"] = "test"
os.environ["STRUCTURED_BASE_URL"] = "http://127.0.0.1:9/v1"
os.environ["STRUCTURED_MODEL"] = "test"
os.environ["EMBEDDING_API_KEY"] = "test"
os.environ["EMBEDDING_BASE_URL"] = "http://127.0.0.1:9/v1"
os.environ["LLM_EMBEDDING_MODEL"] = "test"
os.environ["RERANKER_ENABLED"] = "false"
os.environ["MCP_LIVE_ENABLED"] = "false"
os.environ["MAS_SUPERVISOR_ENABLED"] = "false"
os.environ["LANGSMITH_TRACING_V2"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGSMITH_API_KEY"] = ""
os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""


@pytest.fixture(autouse=True)
def _open_vectorstore_lifecycle_for_each_test():
    """Tests that call services directly emulate application startup ownership."""
    from services.vectorstore import start_vectorstore_io

    start_vectorstore_io()
    yield


def pytest_sessionfinish(session, exitstatus) -> None:  # noqa: ARG001
    """Release cached Chroma systems before Windows removes temp directories."""
    try:
        from chromadb.api.client import SharedSystemClient

        systems = list(
            getattr(SharedSystemClient, "_identifier_to_system", {}).values()
        )
        for system in systems:
            stop = getattr(system, "stop", None)
            if callable(stop):
                stop()
        SharedSystemClient.clear_system_cache()
    except (ImportError, AttributeError):
        return
