"""Low-cost operational checks for configured model providers."""

from __future__ import annotations

import asyncio
import copy
import inspect
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

import httpx

from services.provider_config import (
    ProviderConfig,
    build_async_openai,
    load_provider_configs,
)


logger = logging.getLogger(__name__)


def _bounded_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return min(maximum, max(minimum, value))


@dataclass(frozen=True, slots=True)
class _CatalogProbe:
    reachable: bool
    model_ids: frozenset[str]


class ProviderHealthChecker:
    """Probe provider catalogs once per credential endpoint and cache results."""

    def __init__(
        self,
        *,
        config_loader: Callable[[], dict[str, ProviderConfig]] = load_provider_configs,
        client_factory=build_async_openai,
        ttl_seconds: float | None = None,
        timeout_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config_loader = config_loader
        self._client_factory = client_factory
        self._ttl_seconds = ttl_seconds or _bounded_float(
            "PROVIDER_HEALTH_TTL_SECONDS", 30.0, 1.0, 300.0
        )
        self._timeout_seconds = timeout_seconds or _bounded_float(
            "PROVIDER_HEALTH_TIMEOUT_SECONDS", 5.0, 0.5, 30.0
        )
        self._clock = clock
        self._lock = asyncio.Lock()
        self._cached_payload: dict | None = None
        self._cache_expires_at = 0.0

    async def check(self) -> dict:
        now = self._clock()
        if self._cached_payload is not None and now < self._cache_expires_at:
            return self._copy_cached()

        async with self._lock:
            now = self._clock()
            if self._cached_payload is not None and now < self._cache_expires_at:
                return self._copy_cached()

            payload = await self._probe()
            self._cached_payload = copy.deepcopy(payload)
            self._cache_expires_at = self._clock() + self._ttl_seconds
            return payload

    def clear_cache(self) -> None:
        self._cached_payload = None
        self._cache_expires_at = 0.0

    def _copy_cached(self) -> dict:
        payload = copy.deepcopy(self._cached_payload)
        payload["cached"] = True
        return payload

    async def _probe(self) -> dict:
        configs = self._config_loader()
        representatives: dict[tuple[str, str], ProviderConfig] = {}
        for config in configs.values():
            identity = config.credential_identity
            if identity is not None:
                representatives.setdefault(identity, config)

        identities = list(representatives)
        probe_results = await asyncio.gather(
            *(self._probe_catalog(representatives[identity]) for identity in identities)
        )
        catalogs = dict(zip(identities, probe_results))

        provider_results = {
            capability: self._provider_result(config, catalogs)
            for capability, config in configs.items()
        }
        ready = all(item["status"] == "ready" for item in provider_results.values())

        return {
            "status": "ready" if ready else "degraded",
            "checked_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "cached": False,
            "probe": "models.list",
            "generation_or_embedding_called": False,
            "providers": provider_results,
        }

    async def _probe_catalog(self, config: ProviderConfig) -> _CatalogProbe:
        timeout = httpx.Timeout(
            self._timeout_seconds,
            connect=min(2.0, self._timeout_seconds),
        )
        client = None
        try:
            client = self._client_factory(
                config,
                timeout=timeout,
                max_retries=0,
                require_model=False,
            )
            page = await asyncio.wait_for(
                client.models.list(), timeout=self._timeout_seconds
            )
            model_ids = frozenset(
                model_id
                for item in getattr(page, "data", ())
                if (model_id := getattr(item, "id", None))
            )
            return _CatalogProbe(reachable=True, model_ids=model_ids)
        except Exception as exc:
            logger.warning(
                "provider catalog probe failed for %s (%s)",
                config.capability,
                type(exc).__name__,
            )
            return _CatalogProbe(reachable=False, model_ids=frozenset())
        finally:
            if client is not None:
                try:
                    close_result = client.close()
                    if inspect.isawaitable(close_result):
                        await asyncio.wait_for(
                            close_result,
                            timeout=min(1.0, self._timeout_seconds),
                        )
                except Exception as exc:
                    logger.debug(
                        "provider health client close failed (%s)", type(exc).__name__
                    )

    @staticmethod
    def _provider_result(
        config: ProviderConfig,
        catalogs: dict[tuple[str, str], _CatalogProbe],
    ) -> dict:
        common = {
            "configured": config.configured,
            "model": config.model,
            "inherited_fields": list(config.inherited_fields),
            "verification": "catalog_only",
            "capability_support": "unverified",
        }

        if not config.configured:
            identity = config.credential_identity
            catalog = catalogs.get(identity) if identity is not None else None
            return {
                **common,
                "status": "misconfigured",
                "code": "configuration_invalid",
                "issues": list(config.issues),
                "catalog_reachable": catalog.reachable if catalog is not None else None,
                "model_visible": None,
            }

        identity = config.credential_identity
        catalog = catalogs[identity]
        if not catalog.reachable:
            return {
                **common,
                "status": "unavailable",
                "code": "catalog_unavailable",
                "issues": [],
                "catalog_reachable": False,
                "model_visible": None,
            }

        model_visible = config.model in catalog.model_ids
        if not model_visible:
            return {
                **common,
                "status": "degraded",
                "code": "model_not_listed",
                "issues": [],
                "catalog_reachable": True,
                "model_visible": False,
            }

        return {
            **common,
            "status": "ready",
            "code": "catalog_ready",
            "issues": [],
            "catalog_reachable": True,
            "model_visible": True,
        }


provider_health_checker = ProviderHealthChecker()
