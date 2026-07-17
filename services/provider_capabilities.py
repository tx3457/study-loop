"""Explicit live probes for every model capability used by StudyLoop.

Unlike the catalog-only health endpoint, these checks can create billable
provider requests. They are kept behind an operator-run CLI and never expose
credentials, endpoints, response bodies, or raw exception messages.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import os
from collections.abc import Callable
from datetime import datetime, timezone

import httpx
from pydantic import BaseModel

from services.provider_config import (
    ProviderConfig,
    build_async_openai,
    load_provider_configs,
)


class _StructuredProbeResponse(BaseModel):
    ok: bool


class _InvalidCapabilityResponse(RuntimeError):
    pass


def _timeout_from_env() -> float:
    try:
        value = float(os.getenv("PROVIDER_CAPABILITY_TIMEOUT_SECONDS", "30"))
    except (TypeError, ValueError):
        value = 30.0
    return min(120.0, max(1.0, value))


def _first_message(response):
    choices = getattr(response, "choices", None) or ()
    if not choices:
        raise _InvalidCapabilityResponse
    message = getattr(choices[0], "message", None)
    if message is None:
        raise _InvalidCapabilityResponse
    return message


class ProviderCapabilityChecker:
    """Send minimal real requests and validate their response contracts."""

    def __init__(
        self,
        *,
        config_loader: Callable[[], dict[str, ProviderConfig]] = load_provider_configs,
        client_factory=build_async_openai,
        timeout_seconds: float | None = None,
    ) -> None:
        self._config_loader = config_loader
        self._client_factory = client_factory
        self._timeout_seconds = (
            _timeout_from_env()
            if timeout_seconds is None
            else max(0.001, timeout_seconds)
        )

    async def check(self) -> dict:
        configs = self._config_loader()
        probes = (
            ("chat", configs["chat"], self._probe_chat),
            ("chat_json_mode", configs["chat"], self._probe_json_mode),
            ("tool_calling_auto", configs["chat"], self._probe_tool_calling),
            (
                "structured_output",
                configs["structured"],
                self._probe_structured_output,
            ),
            ("embedding", configs["embedding"], self._probe_embedding),
        )

        results = {}
        for name, config, probe in probes:
            results[name] = await self._run_probe(config, probe)

        ready = all(item["status"] == "ready" for item in results.values())
        return {
            "status": "ready" if ready else "degraded",
            "checked_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "probe": "live_capability_requests",
            "billable_requests_possible": True,
            "capabilities": results,
        }

    async def _run_probe(self, config: ProviderConfig, probe) -> dict:
        common = {
            "provider_role": config.capability,
            "model": config.model,
            "inherited_fields": list(config.inherited_fields),
        }
        if not config.configured:
            return {
                **common,
                "status": "misconfigured",
                "code": "configuration_invalid",
                "attempted": False,
                "issues": list(config.issues),
            }

        timeout = httpx.Timeout(
            self._timeout_seconds,
            connect=min(5.0, self._timeout_seconds),
        )
        client = None
        try:
            client = self._client_factory(
                config,
                timeout=timeout,
                max_retries=0,
            )
            await asyncio.wait_for(probe(client, config), timeout=self._timeout_seconds)
            return {
                **common,
                "status": "ready",
                "code": "capability_ready",
                "attempted": True,
                "issues": [],
            }
        except asyncio.TimeoutError:
            return {
                **common,
                "status": "failed",
                "code": "request_timeout",
                "attempted": True,
                "issues": [],
            }
        except _InvalidCapabilityResponse:
            return {
                **common,
                "status": "failed",
                "code": "invalid_response",
                "attempted": True,
                "issues": [],
            }
        except Exception as exc:
            return {
                **common,
                "status": "failed",
                "code": "request_failed",
                "attempted": True,
                "error_type": type(exc).__name__,
                "issues": [],
            }
        finally:
            if client is not None:
                await self._close_client(client)

    async def _close_client(self, client) -> None:
        try:
            close_result = client.close()
            if inspect.isawaitable(close_result):
                await asyncio.wait_for(
                    close_result,
                    timeout=min(1.0, max(0.05, self._timeout_seconds)),
                )
        except Exception:
            pass

    @staticmethod
    async def _probe_chat(client, config: ProviderConfig) -> None:
        response = await client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": "Reply with exactly OK."}],
            max_tokens=128,
        )
        content = getattr(_first_message(response), "content", None)
        if not isinstance(content, str) or content.strip() != "OK":
            raise _InvalidCapabilityResponse

    @staticmethod
    async def _probe_json_mode(client, config: ProviderConfig) -> None:
        response = await client.chat.completions.create(
            model=config.model,
            messages=[
                {
                    "role": "user",
                    "content": 'Return one JSON object with exactly {"ok": true}.',
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=128,
        )
        content = getattr(_first_message(response), "content", None)
        if not isinstance(content, str):
            raise _InvalidCapabilityResponse
        try:
            payload = json.loads(content)
        except (TypeError, json.JSONDecodeError) as exc:
            raise _InvalidCapabilityResponse from exc
        if payload != {"ok": True}:
            raise _InvalidCapabilityResponse

    @staticmethod
    async def _probe_tool_calling(client, config: ProviderConfig) -> None:
        tool = {
            "type": "function",
            "function": {
                "name": "capability_probe",
                "description": "Validate provider tool-call support.",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string", "enum": ["ok"]}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            },
        }
        prompts = (
            [
                {
                    "role": "user",
                    "content": (
                        "Call capability_probe with value ok. "
                        "Do not answer with normal text."
                    ),
                }
            ],
            [
                {
                    "role": "system",
                    "content": (
                        "For this compatibility check, respond only by calling "
                        "capability_probe with value ok."
                    ),
                },
                {"role": "user", "content": "Run the compatibility check now."},
            ],
        )
        for messages in prompts:
            response = await client.chat.completions.create(
                model=config.model,
                messages=messages,
                tools=[tool],
                tool_choice="auto",
                max_tokens=128,
            )
            tool_calls = getattr(_first_message(response), "tool_calls", None) or ()
            for tool_call in tool_calls:
                function = getattr(tool_call, "function", None)
                if getattr(function, "name", None) != "capability_probe":
                    continue
                try:
                    arguments = json.loads(getattr(function, "arguments", ""))
                except (TypeError, json.JSONDecodeError):
                    continue
                if arguments == {"value": "ok"}:
                    return
        raise _InvalidCapabilityResponse

    @staticmethod
    async def _probe_structured_output(client, config: ProviderConfig) -> None:
        response = await client.beta.chat.completions.parse(
            model=config.model,
            messages=[
                {
                    "role": "user",
                    "content": "Return a structured response with ok set to true.",
                }
            ],
            response_format=_StructuredProbeResponse,
            max_tokens=128,
        )
        parsed = getattr(_first_message(response), "parsed", None)
        if getattr(parsed, "ok", None) is not True:
            raise _InvalidCapabilityResponse

    @staticmethod
    async def _probe_embedding(client, config: ProviderConfig) -> None:
        response = await client.embeddings.create(
            model=config.model,
            input=["StudyLoop provider capability probe."],
        )
        data = getattr(response, "data", None) or ()
        vector = getattr(data[0], "embedding", None) if data else None
        if not isinstance(vector, (list, tuple)) or not vector:
            raise _InvalidCapabilityResponse
        try:
            valid = all(math.isfinite(float(value)) for value in vector)
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise _InvalidCapabilityResponse
