"""Isolated provider profile for knowledge-base autonomous answers."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx

from services.llm import _client as _legacy_client, model as _legacy_model
from services.provider_config import (
    ProviderConfig,
    ProviderConfigurationError,
    build_managed_async_openai,
)


_OVERRIDE_FIELDS = (
    "KNOWLEDGE_LLM_API_KEY",
    "KNOWLEDGE_LLM_BASE_URL",
    "KNOWLEDGE_LLM_MODEL",
)
_QA_ROLE_FIELDS = (
    "KNOWLEDGE_QA_MODEL",
    "KNOWLEDGE_QA_TEMPERATURE",
    "KNOWLEDGE_QA_TOP_P",
    "KNOWLEDGE_QA_TOP_K",
    "KNOWLEDGE_QA_MIN_P",
    "KNOWLEDGE_QA_PRESENCE_PENALTY",
    "KNOWLEDGE_QA_ENABLE_THINKING",
)


def _clean(value: object) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _bounded_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = _clean(env.get(name))
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number between 1 and 300") from None
    if not math.isfinite(value) or not 1 <= value <= 300:
        raise ValueError(f"{name} must be a number between 1 and 300")
    return value


def _bounded_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = _clean(env.get(name))
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer between 1 and 16384") from None
    if not 1 <= value <= 16384:
        raise ValueError(f"{name} must be an integer between 1 and 16384")
    return value


def _boolean(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = _clean(env.get(name))
    if raw is None:
        return default
    normalized = raw.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _optional_float(
    env: Mapping[str, str],
    name: str,
    default: float | None,
    *,
    minimum: float,
    maximum: float,
    minimum_inclusive: bool = True,
) -> float | None:
    raw = _clean(env.get(name))
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a finite number") from None
    minimum_ok = value >= minimum if minimum_inclusive else value > minimum
    if not math.isfinite(value) or not minimum_ok or value > maximum:
        operator = "at least" if minimum_inclusive else "greater than"
        raise ValueError(
            f"{name} must be finite, {operator} {minimum:g}, and at most {maximum:g}"
        )
    return value


def _optional_int(
    env: Mapping[str, str],
    name: str,
    default: int | None,
    *,
    minimum: int,
    maximum: int,
) -> int | None:
    raw = _clean(env.get(name))
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"{name} must be an integer between {minimum} and {maximum}"
        ) from None
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be an integer between {minimum} and {maximum}"
        )
    return value


def _identity_url(value: str) -> str:
    parsed = urlsplit(value)
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{parsed.path.rstrip('/')}"


@dataclass(frozen=True, slots=True)
class KnowledgeLLMProfile:
    client: object = field(repr=False)
    model: str
    max_output_tokens: int | None = None
    request_timeout_seconds: float | None = None
    total_timeout_seconds: float | None = None
    temperature: float | None = None
    top_p: float | None = None
    presence_penalty: float | None = None
    extra_body: dict[str, object] | None = None
    semantic_fingerprint: str = "legacy"
    is_override: bool = False

    @property
    def call_kwargs(self) -> dict[str, object]:
        if not self.is_override:
            return {}
        kwargs: dict[str, object] = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "timeout": self.request_timeout_seconds,
            "total_timeout": self.total_timeout_seconds,
            "extra_body": dict(self.extra_body or {}),
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.presence_penalty is not None:
            kwargs["presence_penalty"] = self.presence_penalty
        return kwargs


def load_knowledge_llm_profile(
    environ: Mapping[str, str] | None = None,
    *,
    legacy_client: object = _legacy_client,
    legacy_model: str = _legacy_model,
    client_builder=build_managed_async_openai,
) -> KnowledgeLLMProfile:
    """Build one process-scoped KB answer profile without credential fallback.

    The dedicated provider is optional. Once any connection field is set, all
    three fields are required so a key can never be sent to a legacy endpoint.
    """

    if environ is None:
        import os

        env: Mapping[str, str] = os.environ
    else:
        env = environ
    configured_values = {name: _clean(env.get(name)) for name in _OVERRIDE_FIELDS}
    present = {name for name, value in configured_values.items() if value is not None}
    qa_role_configured = any(_clean(env.get(name)) is not None for name in _QA_ROLE_FIELDS)
    if not present:
        if qa_role_configured:
            raise ProviderConfigurationError(
                "knowledge",
                ("api_key_missing", "base_url_missing", "model_missing"),
            )
        return KnowledgeLLMProfile(client=legacy_client, model=legacy_model)
    if len(present) != len(_OVERRIDE_FIELDS):
        issues = tuple(
            name.removeprefix("KNOWLEDGE_LLM_").lower() + "_missing"
            for name in _OVERRIDE_FIELDS
            if name not in present
        )
        raise ProviderConfigurationError("knowledge", issues)

    parent_config = ProviderConfig(
        capability="knowledge",
        api_key=configured_values["KNOWLEDGE_LLM_API_KEY"],
        base_url=configured_values["KNOWLEDGE_LLM_BASE_URL"],
        model=configured_values["KNOWLEDGE_LLM_MODEL"],
    )
    if parent_config.issues:
        raise ProviderConfigurationError(parent_config.capability, parent_config.issues)

    selected_model = _clean(env.get("KNOWLEDGE_QA_MODEL")) or parent_config.model
    config = ProviderConfig(
        capability="knowledge",
        api_key=parent_config.api_key,
        base_url=parent_config.base_url,
        model=selected_model,
    )
    if config.issues:
        raise ProviderConfigurationError(config.capability, config.issues)

    models_same = config.model == parent_config.model

    def generation_key(suffix: str) -> str:
        qa_key = f"KNOWLEDGE_QA_{suffix}"
        if _clean(env.get(qa_key)) is not None:
            return qa_key
        if models_same:
            return f"KNOWLEDGE_LLM_{suffix}"
        return qa_key

    timeout_seconds = _bounded_float(env, "KNOWLEDGE_QA_TIMEOUT_SECONDS", 120.0)
    max_tokens = _bounded_int(env, "KNOWLEDGE_QA_MAX_OUTPUT_TOKENS", 4096)
    enable_thinking = _boolean(env, generation_key("ENABLE_THINKING"), False)
    model_name = (config.model or "").lower().rsplit("/", 1)[-1]
    recognized_qwen = model_name in {
        "qwen3-30b-a3b-instruct-2507",
        "qwen3.5-35b-a3b",
    }
    temperature = _optional_float(
        env,
        generation_key("TEMPERATURE"),
        0.7 if recognized_qwen else None,
        minimum=0.0,
        maximum=2.0,
    )
    top_p = _optional_float(
        env,
        generation_key("TOP_P"),
        0.8 if recognized_qwen else None,
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
    )
    top_k = _optional_int(
        env,
        generation_key("TOP_K"),
        20 if recognized_qwen else None,
        minimum=1,
        maximum=100,
    )
    min_p = _optional_float(
        env,
        generation_key("MIN_P"),
        0.0 if recognized_qwen else None,
        minimum=0.0,
        maximum=1.0,
    )
    presence_penalty = _optional_float(
        env,
        generation_key("PRESENCE_PENALTY"),
        1.5 if model_name == "qwen3.5-35b-a3b" else None,
        minimum=-2.0,
        maximum=2.0,
    )
    timeout = httpx.Timeout(
        timeout_seconds,
        connect=min(20.0, timeout_seconds),
        read=timeout_seconds,
        write=min(30.0, timeout_seconds),
        pool=min(5.0, timeout_seconds),
    )
    client = client_builder(config, timeout=timeout, max_retries=0)
    semantics = {
        "provider": _identity_url(config.base_url or ""),
        "model": config.model,
        "enable_thinking": enable_thinking,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "presence_penalty": presence_penalty,
        "max_output_tokens": max_tokens,
        "request_timeout_seconds": timeout_seconds,
    }
    semantic_fingerprint = hashlib.sha256(
        json.dumps(semantics, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return KnowledgeLLMProfile(
        client=client,
        model=config.model or "",
        max_output_tokens=max_tokens,
        request_timeout_seconds=timeout_seconds,
        total_timeout_seconds=timeout_seconds,
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        extra_body={
            "enable_thinking": enable_thinking,
            **({"top_k": top_k} if top_k is not None else {}),
            **({"min_p": min_p} if min_p is not None else {}),
        },
        semantic_fingerprint=semantic_fingerprint,
        is_override=True,
    )


_shared_knowledge_llm_profile: KnowledgeLLMProfile | None = None


def get_knowledge_llm_profile() -> KnowledgeLLMProfile:
    """Load and cache the KB provider only when a KB request needs it."""

    global _shared_knowledge_llm_profile
    if _shared_knowledge_llm_profile is None:
        # Assign only after complete validation. A corrected environment can
        # therefore recover on the next request without retaining bad state.
        profile = load_knowledge_llm_profile()
        _shared_knowledge_llm_profile = profile
    return _shared_knowledge_llm_profile
