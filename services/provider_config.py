"""Provider configuration and OpenAI-compatible client construction.

The application supports separate chat, structured-output, and embedding
providers.  This module is the single place where their environment fallback
rules and safe client defaults are defined.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI


load_dotenv(Path(__file__).parent.parent / ".env")

PROVIDER_TIMEOUT = httpx.Timeout(120.0, connect=20.0)


def _clean(value: object) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _is_placeholder(value: str) -> bool:
    lowered = value.lower()
    return (
        lowered.startswith("replace-with-")
        or lowered.rstrip("/") == "https://api.example.com/v1"
    )


def _valid_base_url(value: str) -> bool:
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


def _identity_url(value: str) -> str:
    parsed = urlsplit(value)
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{path}"


class ProviderConfigurationError(RuntimeError):
    """A model capability was used before its provider was configured."""

    def __init__(self, capability: str, issues: tuple[str, ...]) -> None:
        self.capability = capability
        self.issues = issues
        super().__init__(
            f"provider configuration invalid for {capability}: {', '.join(issues)}"
        )


class _UnconfiguredProviderClient:
    """Lazy failure object that keeps application import and health routes alive."""

    def __init__(self, config: "ProviderConfig", issues: tuple[str, ...]) -> None:
        self._config = config
        self._issues = issues

    def __getattr__(self, name: str):
        raise ProviderConfigurationError(self._config.capability, self._issues)

    async def close(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """Effective configuration for one model capability.

    Secrets are excluded from ``repr`` so accidental debug logging does not
    print API keys.
    """

    capability: str
    api_key: str | None = field(repr=False)
    base_url: str | None = field(repr=False)
    model: str | None
    inherited_fields: tuple[str, ...] = ()

    @property
    def issues(self) -> tuple[str, ...]:
        issues = list(self.connection_issues)
        value = self.model
        if value is None:
            issues.append("model_missing")
        elif _is_placeholder(value):
            issues.append("model_placeholder")
        return tuple(issues)

    @property
    def connection_issues(self) -> tuple[str, ...]:
        issues: list[str] = []
        for field_name, value in (
            ("api_key", self.api_key),
            ("base_url", self.base_url),
        ):
            if value is None:
                issues.append(f"{field_name}_missing")
            elif _is_placeholder(value):
                issues.append(f"{field_name}_placeholder")

        if (
            self.base_url is not None
            and not _is_placeholder(self.base_url)
            and not _valid_base_url(self.base_url)
        ):
            issues.append("base_url_invalid")
        return tuple(issues)

    @property
    def configured(self) -> bool:
        return not self.issues

    @property
    def credential_identity(self) -> tuple[str, str] | None:
        if self.connection_issues or self.base_url is None or self.api_key is None:
            return None
        key_fingerprint = hashlib.sha256(self.api_key.encode("utf-8")).hexdigest()
        return (_identity_url(self.base_url), key_fingerprint)


def load_provider_configs(
    environ: Mapping[str, str] | None = None,
) -> dict[str, ProviderConfig]:
    """Return effective chat, structured, and embedding configurations.

    Structured output and embeddings inherit the chat key and base URL only
    when both dedicated values are blank.  A partial credential override is
    invalid so a secret is never sent to an unintended host.  Structured
    output also inherits the chat model; the embedding model must always be
    configured explicitly.
    """

    env = os.environ if environ is None else environ

    chat_key = _clean(env.get("LLM_API_KEY"))
    chat_url = _clean(env.get("LLM_BASE_URL"))
    chat_model = _clean(env.get("LLM_MODEL"))

    structured_key = _clean(env.get("STRUCTURED_API_KEY"))
    structured_url = _clean(env.get("STRUCTURED_BASE_URL"))
    structured_model = _clean(env.get("STRUCTURED_MODEL"))
    structured_inherited: list[str] = []
    if structured_key is None and structured_url is None:
        structured_key = chat_key
        structured_url = chat_url
        structured_inherited.append("api_key")
        structured_inherited.append("base_url")
    if structured_model is None:
        structured_model = chat_model
        structured_inherited.append("model")

    embedding_key = _clean(env.get("EMBEDDING_API_KEY"))
    embedding_url = _clean(env.get("EMBEDDING_BASE_URL"))
    embedding_inherited: list[str] = []
    if embedding_key is None and embedding_url is None:
        embedding_key = chat_key
        embedding_url = chat_url
        embedding_inherited.append("api_key")
        embedding_inherited.append("base_url")

    return {
        "chat": ProviderConfig(
            capability="chat",
            api_key=chat_key,
            base_url=chat_url,
            model=chat_model,
        ),
        "structured": ProviderConfig(
            capability="structured",
            api_key=structured_key,
            base_url=structured_url,
            model=structured_model,
            inherited_fields=tuple(structured_inherited),
        ),
        "embedding": ProviderConfig(
            capability="embedding",
            api_key=embedding_key,
            base_url=embedding_url,
            model=_clean(env.get("LLM_EMBEDDING_MODEL")),
            inherited_fields=tuple(embedding_inherited),
        ),
    }


def build_async_openai(
    config: ProviderConfig,
    *,
    timeout: float | httpx.Timeout = PROVIDER_TIMEOUT,
    max_retries: int = 2,
    require_model: bool = True,
) -> AsyncOpenAI | _UnconfiguredProviderClient:
    """Build a client without making missing configuration fatal at import.

    An incomplete or placeholder configuration returns a lazy local failure
    object.  This keeps liveness and diagnostics available while ensuring that
    a missing base URL can never fall through to the SDK's public default.
    """

    issues = config.issues if require_model else config.connection_issues
    if issues:
        return _UnconfiguredProviderClient(config, issues)

    api_key = config.api_key
    base_url = config.base_url

    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
    )
