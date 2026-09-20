from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


def _clean(value: object) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _placeholder(value: str) -> bool:
    lowered = value.lower()
    return lowered.startswith("replace-with-") or lowered.rstrip("/") == (
        "https://api.example.com/v1"
    )


def _provider_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme in {"http", "https"}
        and bool(parsed.hostname)
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
    )


def _positive_int(name: str, raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a positive integer") from error
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class Settings:
    database_url: str
    internal_token: str
    materials_dir: Path
    working_dir: Path
    provider: str
    llm_model: str
    embedding_model: str
    embedding_dim: int
    max_cached_instances: int = 8
    llm_timeout_seconds: int = 60
    query_timeout_seconds: int = 30
    mutation_timeout_seconds: int = 300
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    index_config_version: str = "lightrag-v1"

    @classmethod
    def from_env(cls) -> "Settings":
        database_url = os.environ.get("KNOWLEDGE_DATABASE_URL", "").strip()
        token = os.environ.get("KNOWLEDGE_SERVICE_TOKEN", "").strip()
        if not database_url or not token:
            raise RuntimeError("KNOWLEDGE_DATABASE_URL and KNOWLEDGE_SERVICE_TOKEN are required")
        llm_key = _clean(os.environ.get("LLM_API_KEY"))
        llm_url = _clean(os.environ.get("LLM_BASE_URL"))
        embedding_dim = (
            os.environ.get("KNOWLEDGE_EMBEDDING_DIM")
            or os.environ.get("EMBEDDING_DIM", "")
        ).strip()
        if not embedding_dim:
            raise RuntimeError(
                "KNOWLEDGE_EMBEDDING_DIM is required and must match LLM_EMBEDDING_MODEL"
            )
        provider = (_clean(os.environ.get("KNOWLEDGE_PROVIDER")) or "openai").lower()
        if provider != "openai":
            raise RuntimeError(f"unsupported KNOWLEDGE_PROVIDER: {provider}")
        llm_model = _clean(os.environ.get("LLM_MODEL"))
        embedding_model = _clean(os.environ.get("LLM_EMBEDDING_MODEL"))
        embedding_key = _clean(os.environ.get("EMBEDDING_API_KEY"))
        embedding_url = _clean(os.environ.get("EMBEDDING_BASE_URL"))
        if (embedding_key is None) != (embedding_url is None):
            raise RuntimeError(
                "EMBEDDING_API_KEY and EMBEDDING_BASE_URL must both be set or both be blank"
            )
        if embedding_key is None:
            embedding_key, embedding_url = llm_key, llm_url
        provider_values = {
            "LLM_API_KEY": llm_key,
            "LLM_BASE_URL": llm_url,
            "LLM_MODEL": llm_model,
            "EMBEDDING_API_KEY": embedding_key,
            "EMBEDDING_BASE_URL": embedding_url,
            "LLM_EMBEDDING_MODEL": embedding_model,
        }
        missing = [name for name, value in provider_values.items() if value is None]
        placeholders = [
            name
            for name, value in provider_values.items()
            if value is not None and _placeholder(value)
        ]
        if missing or placeholders:
            raise RuntimeError(
                "invalid provider configuration: "
                + ", ".join([*(f"{name} missing" for name in missing), *(f"{name} placeholder" for name in placeholders)])
            )
        for name, value in (("LLM_BASE_URL", llm_url), ("EMBEDDING_BASE_URL", embedding_url)):
            if not _provider_url(value):
                raise RuntimeError(f"{name} must be an http(s) origin without credentials, query, or fragment")
        dimension = _positive_int("KNOWLEDGE_EMBEDDING_DIM", embedding_dim)
        max_cached = _positive_int(
            "KNOWLEDGE_MAX_CACHED_INSTANCES",
            os.environ.get("KNOWLEDGE_MAX_CACHED_INSTANCES", "8"),
        )
        llm_timeout = _positive_int(
            "KNOWLEDGE_LLM_TIMEOUT_SECONDS",
            os.environ.get("KNOWLEDGE_LLM_TIMEOUT_SECONDS", "60"),
        )
        query_timeout = _positive_int(
            "KNOWLEDGE_QUERY_TIMEOUT_SECONDS",
            os.environ.get("KNOWLEDGE_QUERY_TIMEOUT_SECONDS", "30"),
        )
        mutation_timeout = _positive_int(
            "KNOWLEDGE_MUTATION_TIMEOUT_SECONDS",
            os.environ.get("KNOWLEDGE_MUTATION_TIMEOUT_SECONDS", "300"),
        )
        return cls(
            database_url=database_url,
            internal_token=token,
            materials_dir=Path(
                os.environ.get("KNOWLEDGE_MATERIALS_DIR", "/var/lib/study-loop/materials")
            ),
            working_dir=Path(
                os.environ.get("KNOWLEDGE_WORKING_DIR", "/var/lib/study-loop/workspaces")
            ),
            provider=provider,
            llm_model=llm_model,
            embedding_model=embedding_model,
            embedding_dim=dimension,
            max_cached_instances=max_cached,
            llm_timeout_seconds=llm_timeout,
            query_timeout_seconds=query_timeout,
            mutation_timeout_seconds=mutation_timeout,
            llm_api_key=llm_key,
            llm_base_url=llm_url,
            embedding_api_key=embedding_key,
            embedding_base_url=embedding_url,
            index_config_version=os.environ.get("KNOWLEDGE_INDEX_CONFIG_VERSION", "lightrag-v1"),
        )
