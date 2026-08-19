"""Keep the deterministic test suite isolated from developer data and providers."""

import os
import tempfile
from pathlib import Path


_chroma_tmp = tempfile.TemporaryDirectory(prefix="study-loop-pytest-chroma-")
_state_tmp = tempfile.TemporaryDirectory(prefix="study-loop-pytest-state-")

# Set before test modules import application services. This prevents collection-time
# imports from opening a developer's configured Chroma directory or calling providers.
os.environ["CHROMA_DIR"] = _chroma_tmp.name
os.environ["MEMORY_SNAPSHOT_PATH"] = str(Path(_state_tmp.name) / "memory.json")
os.environ["IDEMPOTENCY_DB_PATH"] = str(Path(_state_tmp.name) / "idempotency.sqlite3")
os.environ["QUIZ_SESSION_DB_PATH"] = str(Path(_state_tmp.name) / "quiz-sessions.sqlite3")
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
