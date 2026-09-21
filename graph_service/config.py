from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
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


def _zero_or_one(name: str, raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer between 0 and 1") from error
    if value not in {0, 1}:
        raise RuntimeError(f"{name} must be an integer between 0 and 1")
    return value


def _bounded_float(
    name: str,
    raw: str | None,
    *,
    default: float | None,
    minimum: float,
    maximum: float,
    include_minimum: bool,
) -> float | None:
    cleaned = _clean(raw)
    if cleaned is None:
        return default
    try:
        value = float(cleaned)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a finite number in range") from error
    lower_valid = value >= minimum if include_minimum else value > minimum
    if not math.isfinite(value) or not lower_valid or value > maximum:
        raise RuntimeError(f"{name} must be a finite number in range")
    return value


def _bounded_optional_int(
    name: str, raw: str | None, *, default: int | None, minimum: int, maximum: int
) -> int | None:
    cleaned = _clean(raw)
    if cleaned is None:
        return default
    try:
        value = int(cleaned)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer in range") from error
    if value < minimum or value > maximum:
        raise RuntimeError(f"{name} must be an integer in range")
    return value


def _optional_bool(name: str, raw: str | None, *, default: bool | None) -> bool | None:
    value = _clean(raw)
    if value is None:
        return default
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean")


def _embedding_profile(model: str) -> tuple[int | None, int | None]:
    model_name = model.lower().rsplit("/", 1)[-1]
    if model_name == "bge-m3":
        return 8192, 8000
    if model_name.startswith("bge-large"):
        return 512, 480
    return None, None


def _credential_pair(
    key_name: str,
    url_name: str,
    *,
    fallback_key: str | None = None,
    fallback_url: str | None = None,
) -> tuple[str | None, str | None]:
    key = _clean(os.environ.get(key_name))
    url = _clean(os.environ.get(url_name))
    if (key is None) != (url is None):
        raise RuntimeError(f"{key_name} and {url_name} must both be set or both be blank")
    if key is None:
        return fallback_key, fallback_url
    return key, url


def _scoped_provider(prefix: str) -> tuple[str, str, str] | None:
    names = (f"{prefix}_API_KEY", f"{prefix}_BASE_URL", f"{prefix}_MODEL")
    values = tuple(_clean(os.environ.get(name)) for name in names)
    present = sum(value is not None for value in values)
    if present == 0:
        return None
    if present != len(names):
        raise RuntimeError(f"{', '.join(names)} must all be set or all be blank")
    key, url, model = values
    assert key is not None and url is not None and model is not None
    return key, url, model


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
    embedding_max_input_tokens: int = 8192
    embedding_token_budget: int = 4096
    embedding_max_utf8_bytes: int = 8000
    chunk_tokens: int = 800
    chunk_overlap_tokens: int = 100
    llm_max_output_tokens: int = 8192
    llm_enable_thinking: bool | None = None
    llm_temperature: float = 0.0
    llm_top_p: float | None = None
    llm_top_k: int | None = None
    llm_min_p: float | None = None
    llm_presence_penalty: float | None = None
    llm_max_async: int = 2
    extraction_max_records: int = 40
    extraction_max_entities: int = 20
    extraction_max_gleaning: int = 0
    llm_timeout_seconds: int = 180
    llm_sdk_timeout_seconds: int = 300
    embedding_timeout_seconds: int = 60
    embedding_sdk_timeout_seconds: int = 120
    query_timeout_seconds: int = 90
    mutation_timeout_seconds: int = 900
    llm_api_key: str | None = field(default=None, repr=False)
    llm_base_url: str | None = None
    embedding_api_key: str | None = field(default=None, repr=False)
    embedding_base_url: str | None = None
    tokenizer_model: str = "gpt-4o-mini"
    index_config_version: str = "lightrag-v2"

    @classmethod
    def from_env(cls) -> "Settings":
        database_url = os.environ.get("KNOWLEDGE_DATABASE_URL", "").strip()
        token = os.environ.get("KNOWLEDGE_SERVICE_TOKEN", "").strip()
        if not database_url or not token:
            raise RuntimeError("KNOWLEDGE_DATABASE_URL and KNOWLEDGE_SERVICE_TOKEN are required")
        llm_override = _scoped_provider("KNOWLEDGE_LLM")
        if llm_override is None:
            llm_key, llm_url = _credential_pair("LLM_API_KEY", "LLM_BASE_URL")
            llm_model = _clean(os.environ.get("LLM_MODEL"))
        else:
            llm_key, llm_url, llm_model = llm_override
        embedding_dim = (
            os.environ.get("KNOWLEDGE_EMBEDDING_DIM") or os.environ.get("EMBEDDING_DIM", "")
        ).strip()
        if not embedding_dim:
            raise RuntimeError(
                "KNOWLEDGE_EMBEDDING_DIM is required and must match LLM_EMBEDDING_MODEL"
            )
        provider = (_clean(os.environ.get("KNOWLEDGE_PROVIDER")) or "openai").lower()
        if provider != "openai":
            raise RuntimeError(f"unsupported KNOWLEDGE_PROVIDER: {provider}")
        embedding_override = _scoped_provider("KNOWLEDGE_EMBEDDING")
        if embedding_override is None:
            embedding_key, embedding_url = _credential_pair(
                "EMBEDDING_API_KEY",
                "EMBEDDING_BASE_URL",
                fallback_key=llm_key,
                fallback_url=llm_url,
            )
            embedding_model = _clean(os.environ.get("LLM_EMBEDDING_MODEL"))
        else:
            embedding_key, embedding_url, embedding_model = embedding_override
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
                + ", ".join(
                    [
                        *(f"{name} missing" for name in missing),
                        *(f"{name} placeholder" for name in placeholders),
                    ]
                )
            )
        assert llm_key is not None
        assert llm_url is not None
        assert llm_model is not None
        assert embedding_key is not None
        assert embedding_url is not None
        assert embedding_model is not None
        for name, value in (("LLM_BASE_URL", llm_url), ("EMBEDDING_BASE_URL", embedding_url)):
            if not _provider_url(value):
                raise RuntimeError(
                    f"{name} must be an http(s) origin without credentials, query, or fragment"
                )
        dimension = _positive_int("KNOWLEDGE_EMBEDDING_DIM", embedding_dim)
        profiled_input_tokens, profiled_utf8_bytes = _embedding_profile(embedding_model)
        raw_input_tokens = _clean(os.environ.get("KNOWLEDGE_EMBEDDING_MAX_INPUT_TOKENS"))
        if raw_input_tokens is None and profiled_input_tokens is None:
            raise RuntimeError(
                "KNOWLEDGE_EMBEDDING_MAX_INPUT_TOKENS is required for unknown embedding models"
            )
        max_input_tokens = _positive_int(
            "KNOWLEDGE_EMBEDDING_MAX_INPUT_TOKENS",
            raw_input_tokens or str(profiled_input_tokens),
        )
        token_budget = _positive_int(
            "KNOWLEDGE_EMBEDDING_TOKEN_BUDGET",
            _clean(os.environ.get("KNOWLEDGE_EMBEDDING_TOKEN_BUDGET"))
            or str(min(4096, max_input_tokens)),
        )
        if token_budget > max_input_tokens:
            raise RuntimeError(
                "KNOWLEDGE_EMBEDDING_TOKEN_BUDGET must not exceed "
                "KNOWLEDGE_EMBEDDING_MAX_INPUT_TOKENS"
            )
        max_utf8_bytes = _positive_int(
            "KNOWLEDGE_EMBEDDING_MAX_UTF8_BYTES",
            _clean(os.environ.get("KNOWLEDGE_EMBEDDING_MAX_UTF8_BYTES"))
            or str(
                profiled_utf8_bytes
                if profiled_utf8_bytes is not None
                else min(8000, max_input_tokens - 32)
            ),
        )
        chunk_tokens = _positive_int(
            "KNOWLEDGE_CHUNK_TOKENS",
            os.environ.get("KNOWLEDGE_CHUNK_TOKENS", "800"),
        )
        chunk_overlap_tokens = _positive_int(
            "KNOWLEDGE_CHUNK_OVERLAP_TOKENS",
            os.environ.get("KNOWLEDGE_CHUNK_OVERLAP_TOKENS", "100"),
        )
        if chunk_overlap_tokens >= chunk_tokens:
            raise RuntimeError(
                "KNOWLEDGE_CHUNK_OVERLAP_TOKENS must be smaller than KNOWLEDGE_CHUNK_TOKENS"
            )
        max_cached = _positive_int(
            "KNOWLEDGE_MAX_CACHED_INSTANCES",
            os.environ.get("KNOWLEDGE_MAX_CACHED_INSTANCES", "8"),
        )
        llm_timeout = _positive_int(
            "KNOWLEDGE_LLM_TIMEOUT_SECONDS",
            os.environ.get("KNOWLEDGE_LLM_TIMEOUT_SECONDS", "180"),
        )
        llm_sdk_timeout = _positive_int(
            "KNOWLEDGE_LLM_SDK_TIMEOUT_SECONDS",
            os.environ.get("KNOWLEDGE_LLM_SDK_TIMEOUT_SECONDS", "300"),
        )
        embedding_timeout = _positive_int(
            "KNOWLEDGE_EMBEDDING_TIMEOUT_SECONDS",
            os.environ.get("KNOWLEDGE_EMBEDDING_TIMEOUT_SECONDS", "60"),
        )
        embedding_sdk_timeout = _positive_int(
            "KNOWLEDGE_EMBEDDING_SDK_TIMEOUT_SECONDS",
            os.environ.get("KNOWLEDGE_EMBEDDING_SDK_TIMEOUT_SECONDS", "120"),
        )
        llm_max_output_tokens = _positive_int(
            "KNOWLEDGE_LLM_MAX_OUTPUT_TOKENS",
            os.environ.get("KNOWLEDGE_LLM_MAX_OUTPUT_TOKENS", "8192"),
        )
        llm_max_async = _positive_int(
            "KNOWLEDGE_LLM_MAX_ASYNC",
            os.environ.get("KNOWLEDGE_LLM_MAX_ASYNC", "2"),
        )
        extraction_max_records = _positive_int(
            "KNOWLEDGE_EXTRACT_MAX_RECORDS",
            _clean(os.environ.get("KNOWLEDGE_EXTRACT_MAX_RECORDS")) or "40",
        )
        extraction_max_entities = _positive_int(
            "KNOWLEDGE_EXTRACT_MAX_ENTITIES",
            _clean(os.environ.get("KNOWLEDGE_EXTRACT_MAX_ENTITIES")) or "20",
        )
        if extraction_max_entities > extraction_max_records:
            raise RuntimeError(
                "KNOWLEDGE_EXTRACT_MAX_ENTITIES must not exceed KNOWLEDGE_EXTRACT_MAX_RECORDS"
            )
        extraction_max_gleaning = _zero_or_one(
            "KNOWLEDGE_EXTRACT_MAX_GLEANING",
            _clean(os.environ.get("KNOWLEDGE_EXTRACT_MAX_GLEANING")) or "0",
        )
        query_timeout = _positive_int(
            "KNOWLEDGE_QUERY_TIMEOUT_SECONDS",
            os.environ.get("KNOWLEDGE_QUERY_TIMEOUT_SECONDS", "90"),
        )
        mutation_timeout = _positive_int(
            "KNOWLEDGE_MUTATION_TIMEOUT_SECONDS",
            os.environ.get("KNOWLEDGE_MUTATION_TIMEOUT_SECONDS", "900"),
        )
        model_name = llm_model.lower().rsplit("/", 1)[-1]
        qwen35_profile = model_name == "qwen3.5-35b-a3b"
        qwen_profile = model_name == "qwen3-30b-a3b-instruct-2507" or qwen35_profile
        thinking_profile = False if qwen_profile else None
        llm_enable_thinking = _optional_bool(
            "KNOWLEDGE_LLM_ENABLE_THINKING",
            os.environ.get("KNOWLEDGE_LLM_ENABLE_THINKING"),
            default=thinking_profile,
        )
        llm_temperature = _bounded_float(
            "KNOWLEDGE_LLM_TEMPERATURE",
            os.environ.get("KNOWLEDGE_LLM_TEMPERATURE"),
            default=0.7 if qwen_profile else 0.0,
            minimum=0.0,
            maximum=2.0,
            include_minimum=True,
        )
        llm_top_p = _bounded_float(
            "KNOWLEDGE_LLM_TOP_P",
            os.environ.get("KNOWLEDGE_LLM_TOP_P"),
            default=0.8 if qwen_profile else None,
            minimum=0.0,
            maximum=1.0,
            include_minimum=False,
        )
        llm_top_k = _bounded_optional_int(
            "KNOWLEDGE_LLM_TOP_K",
            os.environ.get("KNOWLEDGE_LLM_TOP_K"),
            default=20 if qwen_profile else None,
            minimum=1,
            maximum=100,
        )
        llm_min_p = _bounded_float(
            "KNOWLEDGE_LLM_MIN_P",
            os.environ.get("KNOWLEDGE_LLM_MIN_P"),
            default=0.0 if qwen_profile else None,
            minimum=0.0,
            maximum=1.0,
            include_minimum=True,
        )
        llm_presence_penalty = _bounded_float(
            "KNOWLEDGE_LLM_PRESENCE_PENALTY",
            os.environ.get("KNOWLEDGE_LLM_PRESENCE_PENALTY"),
            default=1.5 if qwen35_profile else None,
            minimum=-2.0,
            maximum=2.0,
            include_minimum=True,
        )
        assert llm_temperature is not None
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
            embedding_max_input_tokens=max_input_tokens,
            embedding_token_budget=token_budget,
            embedding_max_utf8_bytes=max_utf8_bytes,
            chunk_tokens=chunk_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
            max_cached_instances=max_cached,
            llm_max_output_tokens=llm_max_output_tokens,
            llm_enable_thinking=llm_enable_thinking,
            llm_temperature=llm_temperature,
            llm_top_p=llm_top_p,
            llm_top_k=llm_top_k,
            llm_min_p=llm_min_p,
            llm_presence_penalty=llm_presence_penalty,
            llm_max_async=llm_max_async,
            extraction_max_records=extraction_max_records,
            extraction_max_entities=extraction_max_entities,
            extraction_max_gleaning=extraction_max_gleaning,
            llm_timeout_seconds=llm_timeout,
            llm_sdk_timeout_seconds=llm_sdk_timeout,
            embedding_timeout_seconds=embedding_timeout,
            embedding_sdk_timeout_seconds=embedding_sdk_timeout,
            query_timeout_seconds=query_timeout,
            mutation_timeout_seconds=mutation_timeout,
            llm_api_key=llm_key,
            llm_base_url=llm_url,
            embedding_api_key=embedding_key,
            embedding_base_url=embedding_url,
            index_config_version=os.environ.get("KNOWLEDGE_INDEX_CONFIG_VERSION", "lightrag-v2"),
        )
