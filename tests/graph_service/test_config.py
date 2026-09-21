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
    "KNOWLEDGE_LLM_API_KEY",
    "KNOWLEDGE_LLM_BASE_URL",
    "KNOWLEDGE_LLM_MODEL",
    "KNOWLEDGE_EMBEDDING_API_KEY",
    "KNOWLEDGE_EMBEDDING_BASE_URL",
    "KNOWLEDGE_EMBEDDING_MODEL",
    "KNOWLEDGE_EMBEDDING_MAX_INPUT_TOKENS",
    "KNOWLEDGE_EMBEDDING_TOKEN_BUDGET",
    "KNOWLEDGE_EMBEDDING_MAX_UTF8_BYTES",
    "KNOWLEDGE_CHUNK_TOKENS",
    "KNOWLEDGE_CHUNK_OVERLAP_TOKENS",
    "KNOWLEDGE_LLM_MAX_OUTPUT_TOKENS",
    "KNOWLEDGE_LLM_ENABLE_THINKING",
    "KNOWLEDGE_LLM_MAX_ASYNC",
    "KNOWLEDGE_LLM_SDK_TIMEOUT_SECONDS",
    "KNOWLEDGE_EMBEDDING_TIMEOUT_SECONDS",
    "KNOWLEDGE_EMBEDDING_SDK_TIMEOUT_SECONDS",
    "KNOWLEDGE_INDEX_CONFIG_VERSION",
    "KNOWLEDGE_EXTRACT_MAX_RECORDS",
    "KNOWLEDGE_EXTRACT_MAX_ENTITIES",
    "KNOWLEDGE_EXTRACT_MAX_GLEANING",
    "KNOWLEDGE_LLM_TEMPERATURE",
    "KNOWLEDGE_LLM_TOP_P",
    "KNOWLEDGE_LLM_TOP_K",
    "KNOWLEDGE_LLM_MIN_P",
    "KNOWLEDGE_LLM_PRESENCE_PENALTY",
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
        "LLM_EMBEDDING_MODEL": "BAAI/bge-m3",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_from_env_inherits_embedding_pair_and_accepts_local_http(monkeypatch) -> None:
    configured_env(monkeypatch)
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_MAX_INPUT_TOKENS", "2048")
    settings = Settings.from_env()
    assert settings.embedding_api_key == "chat-key"
    assert settings.embedding_base_url == "http://127.0.0.1:9000/v1"
    assert settings.embedding_dim == 1024


def test_from_env_prefers_scoped_siliconflow_settings_and_applies_m3_profile(
    monkeypatch,
) -> None:
    configured_env(monkeypatch)
    monkeypatch.setenv("KNOWLEDGE_LLM_API_KEY", "knowledge-chat-key")
    monkeypatch.setenv("KNOWLEDGE_LLM_BASE_URL", "https://api.siliconflow.cn/v1")
    monkeypatch.setenv("KNOWLEDGE_LLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_API_KEY", "knowledge-embedding-key")
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_MODEL", "BAAI/bge-m3")

    settings = Settings.from_env()

    assert settings.llm_api_key == "knowledge-chat-key"
    assert settings.llm_model == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert settings.embedding_api_key == "knowledge-embedding-key"
    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.embedding_max_input_tokens == 8192
    assert settings.embedding_token_budget == 4096
    assert settings.embedding_max_utf8_bytes == 8000
    assert settings.chunk_tokens == 800
    assert settings.chunk_overlap_tokens == 100
    assert settings.llm_max_output_tokens == 8192
    assert settings.llm_temperature == 0.7
    assert settings.llm_top_p == 0.8
    assert settings.llm_top_k == 20
    assert settings.llm_min_p == 0.0
    assert settings.llm_presence_penalty is None
    assert settings.llm_enable_thinking is False
    assert settings.llm_max_async == 2
    assert settings.llm_timeout_seconds == 180
    assert settings.llm_sdk_timeout_seconds == 300
    assert settings.embedding_timeout_seconds == 60
    assert settings.embedding_sdk_timeout_seconds == 120
    assert settings.query_timeout_seconds == 90
    assert settings.mutation_timeout_seconds == 900
    assert settings.extraction_max_records == 40
    assert settings.extraction_max_entities == 20
    assert settings.extraction_max_gleaning == 0
    assert settings.index_config_version == "lightrag-v2"
    assert "knowledge-chat-key" not in repr(settings)
    assert "knowledge-embedding-key" not in repr(settings)


def test_from_env_requires_explicit_input_limit_for_unknown_embedding_model(
    monkeypatch,
) -> None:
    configured_env(monkeypatch)
    monkeypatch.setenv("LLM_EMBEDDING_MODEL", "custom/unknown-embedding")

    with pytest.raises(RuntimeError, match="KNOWLEDGE_EMBEDDING_MAX_INPUT_TOKENS"):
        Settings.from_env()


def test_from_env_profiles_legacy_bge_large_limit(monkeypatch) -> None:
    configured_env(monkeypatch)
    monkeypatch.setenv("LLM_EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")

    settings = Settings.from_env()

    assert settings.embedding_max_input_tokens == 512
    assert settings.embedding_token_budget == 512
    assert settings.embedding_max_utf8_bytes == 480


def test_blank_optional_embedding_limits_use_profile_defaults(monkeypatch) -> None:
    configured_env(monkeypatch)
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_TOKEN_BUDGET", "  ")
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_MAX_UTF8_BYTES", "")

    settings = Settings.from_env()

    assert settings.embedding_token_budget == 4096
    assert settings.embedding_max_utf8_bytes == 8000


def test_unknown_embedding_uses_explicit_limit_minus_safety_margin(monkeypatch) -> None:
    configured_env(monkeypatch)
    monkeypatch.setenv("LLM_EMBEDDING_MODEL", "custom/unknown-embedding")
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_MAX_INPUT_TOKENS", "1000")

    settings = Settings.from_env()

    assert settings.embedding_max_utf8_bytes == 968


def test_scoped_credentials_do_not_mix_with_partial_legacy_pairs(monkeypatch) -> None:
    configured_env(monkeypatch)
    monkeypatch.delenv("LLM_BASE_URL")
    monkeypatch.setenv("KNOWLEDGE_LLM_API_KEY", "scoped-chat-key")
    monkeypatch.setenv("KNOWLEDGE_LLM_BASE_URL", "https://api.siliconflow.cn/v1")
    monkeypatch.setenv("KNOWLEDGE_LLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_API_KEY", "scoped-embedding-key")
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_MODEL", "BAAI/bge-m3")
    monkeypatch.setenv("EMBEDDING_API_KEY", "orphaned-legacy-key")

    settings = Settings.from_env()

    assert settings.llm_api_key == "scoped-chat-key"
    assert settings.embedding_api_key == "scoped-embedding-key"


@pytest.mark.parametrize(
    ("prefix", "missing"),
    (
        ("KNOWLEDGE_LLM", "MODEL"),
        ("KNOWLEDGE_LLM", "API_KEY"),
        ("KNOWLEDGE_EMBEDDING", "MODEL"),
        ("KNOWLEDGE_EMBEDDING", "BASE_URL"),
    ),
)
def test_scoped_provider_triplets_fail_closed_when_partial(
    monkeypatch, prefix: str, missing: str
) -> None:
    configured_env(monkeypatch)
    monkeypatch.setenv(f"{prefix}_API_KEY", "scoped-key")
    monkeypatch.setenv(f"{prefix}_BASE_URL", "https://api.siliconflow.cn/v1")
    model = "Qwen/Qwen3-30B-A3B-Instruct-2507" if prefix == "KNOWLEDGE_LLM" else "BAAI/bge-m3"
    monkeypatch.setenv(f"{prefix}_MODEL", model)
    monkeypatch.delenv(f"{prefix}_{missing}")

    with pytest.raises(RuntimeError, match="must all be set"):
        Settings.from_env()


def test_direct_settings_do_not_force_provider_specific_thinking_option(tmp_path) -> None:
    settings = Settings(
        database_url="postgresql://graph@127.0.0.1:5432/graph",
        internal_token="test-token",
        materials_dir=tmp_path / "materials",
        working_dir=tmp_path / "workspaces",
        provider="openai",
        llm_model="legacy-chat-model",
        embedding_model="BAAI/bge-m3",
        embedding_dim=1024,
    )

    assert settings.llm_enable_thinking is None
    assert settings.llm_temperature == 0.0
    assert settings.llm_top_p is None
    assert settings.llm_top_k is None
    assert settings.llm_min_p is None
    assert settings.llm_presence_penalty is None


def test_qwen35_profile_applies_official_nonthinking_sampling(monkeypatch) -> None:
    configured_env(monkeypatch)
    monkeypatch.setenv("KNOWLEDGE_LLM_API_KEY", "knowledge-chat-key")
    monkeypatch.setenv("KNOWLEDGE_LLM_BASE_URL", "https://api.siliconflow.cn/v1")
    monkeypatch.setenv("KNOWLEDGE_LLM_MODEL", "Qwen/Qwen3.5-35B-A3B")

    settings = Settings.from_env()

    assert settings.llm_enable_thinking is False
    assert settings.llm_temperature == 0.7
    assert settings.llm_top_p == 0.8
    assert settings.llm_top_k == 20
    assert settings.llm_min_p == 0.0
    assert settings.llm_presence_penalty == 1.5


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
        "KNOWLEDGE_LLM_SDK_TIMEOUT_SECONDS",
        "KNOWLEDGE_EMBEDDING_TIMEOUT_SECONDS",
        "KNOWLEDGE_EMBEDDING_SDK_TIMEOUT_SECONDS",
        "KNOWLEDGE_EMBEDDING_MAX_INPUT_TOKENS",
        "KNOWLEDGE_EMBEDDING_TOKEN_BUDGET",
        "KNOWLEDGE_EMBEDDING_MAX_UTF8_BYTES",
        "KNOWLEDGE_CHUNK_TOKENS",
        "KNOWLEDGE_CHUNK_OVERLAP_TOKENS",
        "KNOWLEDGE_LLM_MAX_OUTPUT_TOKENS",
        "KNOWLEDGE_LLM_MAX_ASYNC",
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


@pytest.mark.parametrize(
    ("name", "value", "message"),
    (
        ("KNOWLEDGE_EXTRACT_MAX_RECORDS", "0", "positive integer"),
        ("KNOWLEDGE_EXTRACT_MAX_ENTITIES", "0", "positive integer"),
        ("KNOWLEDGE_EXTRACT_MAX_GLEANING", "-1", "between 0 and 1"),
        ("KNOWLEDGE_EXTRACT_MAX_GLEANING", "2", "between 0 and 1"),
    ),
)
def test_from_env_rejects_invalid_extraction_limits(
    monkeypatch, name: str, value: str, message: str
) -> None:
    configured_env(monkeypatch)
    monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeError, match=message):
        Settings.from_env()


def test_from_env_rejects_more_entities_than_records(monkeypatch) -> None:
    configured_env(monkeypatch)
    monkeypatch.setenv("KNOWLEDGE_EXTRACT_MAX_RECORDS", "10")
    monkeypatch.setenv("KNOWLEDGE_EXTRACT_MAX_ENTITIES", "11")

    with pytest.raises(RuntimeError, match="must not exceed"):
        Settings.from_env()


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("KNOWLEDGE_LLM_TEMPERATURE", "nan"),
        ("KNOWLEDGE_LLM_TEMPERATURE", "2.1"),
        ("KNOWLEDGE_LLM_TOP_P", "0"),
        ("KNOWLEDGE_LLM_TOP_P", "1.1"),
        ("KNOWLEDGE_LLM_TOP_K", "0"),
        ("KNOWLEDGE_LLM_TOP_K", "101"),
        ("KNOWLEDGE_LLM_MIN_P", "-0.1"),
        ("KNOWLEDGE_LLM_MIN_P", "1.1"),
        ("KNOWLEDGE_LLM_PRESENCE_PENALTY", "nan"),
        ("KNOWLEDGE_LLM_PRESENCE_PENALTY", "-2.1"),
        ("KNOWLEDGE_LLM_PRESENCE_PENALTY", "2.1"),
    ),
)
def test_from_env_rejects_invalid_sampling_limits(monkeypatch, name: str, value: str) -> None:
    configured_env(monkeypatch)
    monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeError, match="must be"):
        Settings.from_env()


def test_index_hash_tracks_provider_origins_and_normalizes_trailing_slash(monkeypatch) -> None:
    configured_env(monkeypatch)
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_MAX_INPUT_TOKENS", "2048")
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


def test_index_hash_tracks_semantic_limits_but_not_operational_timeouts(monkeypatch) -> None:
    configured_env(monkeypatch)
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_MAX_INPUT_TOKENS", "2048")
    settings = Settings.from_env()

    semantic_change = replace(settings, chunk_tokens=settings.chunk_tokens + 1)
    extraction_change = replace(
        settings, extraction_max_records=settings.extraction_max_records + 1
    )
    sampling_change = replace(settings, llm_temperature=0.5)
    penalty_change = replace(settings, llm_presence_penalty=1.5)
    timeout_change = replace(settings, llm_timeout_seconds=settings.llm_timeout_seconds + 1)

    assert index_config_hash(settings) != index_config_hash(semantic_change)
    assert index_config_hash(settings) != index_config_hash(extraction_change)
    assert index_config_hash(settings) != index_config_hash(sampling_change)
    assert index_config_hash(settings) != index_config_hash(penalty_change)
    assert index_config_hash(settings) == index_config_hash(timeout_change)
