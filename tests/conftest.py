"""Keep the deterministic test suite isolated from developer data and providers."""

import os
import tempfile


_chroma_tmp = tempfile.TemporaryDirectory(prefix="study-loop-pytest-chroma-")

# Set before test modules import application services. This prevents collection-time
# imports from opening a developer's configured Chroma directory or calling providers.
os.environ["CHROMA_DIR"] = _chroma_tmp.name
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
