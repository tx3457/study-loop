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
for _knowledge_name in (
    "KNOWLEDGE_LLM_API_KEY",
    "KNOWLEDGE_LLM_BASE_URL",
    "KNOWLEDGE_LLM_MODEL",
    "KNOWLEDGE_LLM_ENABLE_THINKING",
    "KNOWLEDGE_LLM_TEMPERATURE",
    "KNOWLEDGE_LLM_TOP_P",
    "KNOWLEDGE_LLM_TOP_K",
    "KNOWLEDGE_LLM_MIN_P",
    "KNOWLEDGE_LLM_PRESENCE_PENALTY",
    "KNOWLEDGE_QA_MAX_OUTPUT_TOKENS",
    "KNOWLEDGE_QA_MODEL",
    "KNOWLEDGE_QA_ENABLE_THINKING",
    "KNOWLEDGE_QA_TEMPERATURE",
    "KNOWLEDGE_QA_TOP_P",
    "KNOWLEDGE_QA_TOP_K",
    "KNOWLEDGE_QA_MIN_P",
    "KNOWLEDGE_QA_PRESENCE_PENALTY",
    "KNOWLEDGE_QA_TIMEOUT_SECONDS",
    "KNOWLEDGE_HTTP_TIMEOUT_SECONDS",
    "KNOWLEDGE_TOOL_TIMEOUT_SECONDS",
    "KNOWLEDGE_EMBEDDING_API_KEY",
    "KNOWLEDGE_EMBEDDING_BASE_URL",
    "KNOWLEDGE_EMBEDDING_MODEL",
    "KNOWLEDGE_DATABASE_URL",
    "KNOWLEDGE_SERVICE_TOKEN",
):
    # Keep names present-but-empty so python-dotenv cannot refill real local
    # credentials after conftest establishes the deterministic test boundary.
    os.environ[_knowledge_name] = ""
for _knowledge_name, _knowledge_value in {
    "KNOWLEDGE_PROVIDER": "openai",
    "KNOWLEDGE_EMBEDDING_DIM": "1024",
    "KNOWLEDGE_EMBEDDING_MAX_INPUT_TOKENS": "8192",
    "KNOWLEDGE_EMBEDDING_TOKEN_BUDGET": "4096",
    "KNOWLEDGE_EMBEDDING_MAX_UTF8_BYTES": "8000",
    "KNOWLEDGE_CHUNK_TOKENS": "800",
    "KNOWLEDGE_CHUNK_OVERLAP_TOKENS": "100",
    "KNOWLEDGE_EXTRACT_MAX_RECORDS": "40",
    "KNOWLEDGE_EXTRACT_MAX_ENTITIES": "20",
    "KNOWLEDGE_EXTRACT_MAX_GLEANING": "0",
    "KNOWLEDGE_MAX_CACHED_INSTANCES": "8",
    "KNOWLEDGE_LLM_MAX_OUTPUT_TOKENS": "8192",
    "KNOWLEDGE_LLM_MAX_ASYNC": "2",
    "KNOWLEDGE_LLM_TIMEOUT_SECONDS": "180",
    "KNOWLEDGE_LLM_SDK_TIMEOUT_SECONDS": "300",
    "KNOWLEDGE_EMBEDDING_TIMEOUT_SECONDS": "60",
    "KNOWLEDGE_EMBEDDING_SDK_TIMEOUT_SECONDS": "120",
    "KNOWLEDGE_QUERY_TIMEOUT_SECONDS": "90",
    "KNOWLEDGE_MUTATION_TIMEOUT_SECONDS": "900",
    "KNOWLEDGE_INDEX_CONFIG_VERSION": "lightrag-v2",
}.items():
    os.environ[_knowledge_name] = _knowledge_value
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
