from __future__ import annotations

import pytest
from dataclasses import replace

from graph_service.config import Settings
from graph_service.engine import index_config_hash


RELEVANT_ENV = (
    "KNOWLEDGE_DATABASE_URL",
    "KNOWLEDGE_SERVICE_TOKEN",
    "KNOWLEDGE_PROVIDER",
    "KNOWLEDGE_EMBEDDING_DIM",
    "EMBEDDING_DIM",
    "KNOWLEDGE_MAX_CACHED_INSTANCES",
    "KNOWLEDGE_LLM_TIMEOUT_SECONDS",
    "KNOWLEDGE_QUERY_TIMEOUT_SECONDS",
    "KNOWLEDGE_MUTATION_TIMEOUT_SECONDS",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "EMBEDDING_API_KEY",
    "EMBEDDING_BASE_URL",
    "LLM_EMBEDDING_MODEL",
)


def configured_env(monkeypatch) -> None:
    for name in RELEVANT_ENV:
        monkeypatch.delenv(name, raising=False)
    values = {
        "KNOWLEDGE_DATABASE_URL": "postgresql://graph@127.0.0.1:5432/graph",
        "KNOWLEDGE_SERVICE_TOKEN": "internal-token",
        "KNOWLEDGE_PROVIDER": "openai",
        "KNOWLEDGE_EMBEDDING_DIM": "1024",
        "LLM_API_KEY": "chat-key",
        "LLM_BASE_URL": " http://127.0.0.1:9000/v1 ",
        "LLM_MODEL": "chat-model",
        "LLM_EMBEDDING_MODEL": "embedding-model",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_from_env_inherits_embedding_pair_and_accepts_local_http(monkeypatch) -> None:
    configured_env(monkeypatch)
    settings = Settings.from_env()
    assert settings.embedding_api_key == "chat-key"
    assert settings.embedding_base_url == "http://127.0.0.1:9000/v1"
    assert settings.embedding_dim == 1024


@pytest.mark.parametrize(
    ("name", "value"),
    (("EMBEDDING_API_KEY", "embedding-key"), ("EMBEDDING_BASE_URL", "https://embed.test/v1")),
)
def test_from_env_rejects_partial_embedding_override(monkeypatch, name: str, value: str) -> None:
    configured_env(monkeypatch)
    monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match="must both be set"):
        Settings.from_env()


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("LLM_API_KEY", "replace-with-provider-key"),
        ("LLM_BASE_URL", "https://api.example.com/v1"),
        ("LLM_MODEL", "replace-with-chat-model"),
    ),
)
def test_from_env_rejects_provider_placeholders(monkeypatch, name: str, value: str) -> None:
    configured_env(monkeypatch)
    monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match="placeholder"):
        Settings.from_env()


@pytest.mark.parametrize(
    "url",
    (
        "https://user:secret@provider.test/v1",
        "https://provider.test/v1?token=secret",
        "https://provider.test/v1#secret",
        "ftp://provider.test/v1",
    ),
)
def test_from_env_rejects_unsafe_provider_urls(monkeypatch, url: str) -> None:
    configured_env(monkeypatch)
    monkeypatch.setenv("LLM_BASE_URL", url)
    with pytest.raises(RuntimeError, match="without credentials"):
        Settings.from_env()


@pytest.mark.parametrize(
    "name",
    (
        "KNOWLEDGE_EMBEDDING_DIM",
        "KNOWLEDGE_MAX_CACHED_INSTANCES",
        "KNOWLEDGE_LLM_TIMEOUT_SECONDS",
        "KNOWLEDGE_QUERY_TIMEOUT_SECONDS",
        "KNOWLEDGE_MUTATION_TIMEOUT_SECONDS",
    ),
)
def test_from_env_rejects_nonpositive_limits(monkeypatch, name: str) -> None:
    configured_env(monkeypatch)
    monkeypatch.setenv(name, "0")
    with pytest.raises(RuntimeError, match="positive integer"):
        Settings.from_env()


def test_from_env_rejects_unknown_provider(monkeypatch) -> None:
    configured_env(monkeypatch)
    monkeypatch.setenv("KNOWLEDGE_PROVIDER", "mystery")
    with pytest.raises(RuntimeError, match="unsupported KNOWLEDGE_PROVIDER"):
        Settings.from_env()


def test_index_hash_tracks_provider_origins_and_normalizes_trailing_slash(monkeypatch) -> None:
    configured_env(monkeypatch)
    settings = Settings.from_env()
    first = replace(
        settings,
        llm_base_url="https://one.test/v1/",
        embedding_base_url="https://embed.test/v1/",
    )
    equivalent = replace(
        first,
        llm_base_url="https://one.test/v1",
        embedding_base_url="https://embed.test/v1",
    )
    changed = replace(equivalent, embedding_base_url="https://other.test/v1")
    assert index_config_hash(first) == index_config_hash(equivalent)
    assert index_config_hash(equivalent) != index_config_hash(changed)
