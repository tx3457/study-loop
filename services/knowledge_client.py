"""Bounded internal HTTP client for the optional knowledge service."""

from __future__ import annotations

import os
import re
from typing import Any
from urllib.parse import urlencode, urlsplit

import httpx
from fastapi import HTTPException
from pydantic import ValidationError

from models.knowledge import KnowledgeScope


_SAFE_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE"})
_SAFE_FORWARD_HEADERS = frozenset({"idempotency-key"})
_SAFE_PATH_ROOTS = frozenset({"knowledge-bases", "knowledge-jobs", "web-snapshots"})
_PUBLIC_ERROR_CODES = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_RESOURCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


class KnowledgeServiceError(HTTPException):
    """Public-safe error from the fixed internal service boundary."""

    def __init__(
        self,
        status_code: int,
        detail: str,
        *,
        error_code: str | None = None,
        request_id: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.detail = detail
        self.error_code = error_code
        self.request_id = request_id
        headers = {"X-Request-ID": request_id} if request_id else None
        super().__init__(status_code=status_code, detail=detail, headers=headers)


class KnowledgeClient:
    def __init__(
        self,
        *,
        service_url: str | None = None,
        service_token: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._service_url = self._validate_service_url(
            service_url if service_url is not None else os.getenv("KNOWLEDGE_SERVICE_URL", "")
        )
        self._service_token = (
            service_token
            if service_token is not None
            else os.getenv("KNOWLEDGE_SERVICE_TOKEN", "")
        ).strip()
        self._transport = transport
        self._timeout = httpx.Timeout(15.0, connect=2.0, read=15.0, write=15.0, pool=2.0)

    @staticmethod
    def _validate_service_url(value: str) -> str:
        value = value.strip().rstrip("/")
        if not value:
            return ""
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError("KNOWLEDGE_SERVICE_URL must be a fixed HTTP service origin")
        return value

    @property
    def configured(self) -> bool:
        return bool(self._service_url and self._service_token)

    async def request(
        self,
        method: str,
        path: str,
        *,
        owner_id: str,
        json: Any = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        normalized_method = method.upper()
        if normalized_method not in _SAFE_METHODS:
            raise ValueError("unsupported knowledge service method")
        parsed_path = urlsplit(path)
        segments = parsed_path.path.split("/")
        if (
            not path.startswith("/")
            or path.startswith("//")
            or parsed_path.scheme
            or parsed_path.fragment
            or len(segments) < 2
            or segments[1] not in _SAFE_PATH_ROOTS
            or any(segment in {".", ".."} for segment in segments)
            or "\\" in parsed_path.path
        ):
            raise ValueError("knowledge service path must be relative to the configured origin")
        if not owner_id or any(ord(char) < 32 for char in owner_id):
            raise ValueError("invalid owner identity")
        if not self.configured:
            raise KnowledgeServiceError(503, "知识库服务暂时不可用")

        forwarded: dict[str, str] = {}
        for key, value in (headers or {}).items():
            if key.lower() not in _SAFE_FORWARD_HEADERS:
                raise ValueError("unsupported forwarded header")
            forwarded[key] = value
        internal_headers = {
            **forwarded,
            "Authorization": f"Bearer {self._service_token}",
            "X-StudyLoop-Subject": owner_id,
        }

        try:
            async with httpx.AsyncClient(
                base_url=self._service_url,
                timeout=self._timeout,
                follow_redirects=False,
                trust_env=False,
                transport=self._transport,
            ) as client:
                response = await client.request(
                    normalized_method,
                    path,
                    json=json,
                    headers=internal_headers,
                )
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError):
            raise KnowledgeServiceError(503, "知识库服务暂时不可用") from None

        if response.is_redirect:
            raise KnowledgeServiceError(503, "知识库服务暂时不可用")
        if response.status_code >= 400:
            raise self._safe_upstream_error(response)
        return response

    @staticmethod
    def _safe_upstream_error(response: httpx.Response) -> KnowledgeServiceError:
        allowed_status = response.status_code if response.status_code in {
            400, 404, 409, 410, 413, 422, 503
        } else 503
        error_code = None
        try:
            body = response.json()
        except (ValueError, UnicodeDecodeError):
            body = None
        if isinstance(body, dict):
            candidate = body.get("error_code")
            if isinstance(candidate, str) and _PUBLIC_ERROR_CODES.fullmatch(candidate):
                error_code = candidate
        details = {
            400: "知识库请求无效",
            404: "知识库资源不存在",
            409: "知识库版本已变化或请求冲突",
            410: "网页快照已过期",
            413: "知识库请求内容过大",
            422: "知识库请求无法处理",
            503: "知识库服务暂时不可用",
        }
        return KnowledgeServiceError(
            allowed_status,
            details[allowed_status],
            error_code=error_code,
            request_id=response.headers.get("X-Request-ID"),
        )

    async def get_scope(self, kb_id: str, owner_id: str) -> KnowledgeScope:
        self._require_resource_id(kb_id)
        response = await self.request(
            "GET", f"/knowledge-bases/{kb_id}/scope", owner_id=owner_id
        )
        try:
            scope = KnowledgeScope.model_validate(response.json())
        except (ValueError, ValidationError):
            raise KnowledgeServiceError(503, "知识库服务返回无效数据") from None
        if scope.knowledge_base_id != kb_id or scope.status != "ready":
            raise KnowledgeServiceError(503, "知识库索引当前不可用")
        return scope

    async def query(
        self,
        kb_id: str,
        owner_id: str,
        query: str,
        revision: int,
        epoch: int,
    ) -> dict[str, Any]:
        self._require_resource_id(kb_id)
        response = await self.request(
            "POST",
            f"/knowledge-bases/{kb_id}/query",
            owner_id=owner_id,
            json={
                "query": query,
                "expected_revision": revision,
                "expected_epoch": epoch,
            },
        )
        data = response.json()
        if not isinstance(data, dict):
            raise KnowledgeServiceError(503, "知识库服务返回无效数据")
        try:
            scope = KnowledgeScope.model_validate(data.get("scope"))
        except (ValueError, ValidationError):
            raise KnowledgeServiceError(503, "知识库服务返回无效数据") from None
        if scope.status != "ready":
            raise KnowledgeServiceError(503, "知识库索引当前不可用")
        if (
            scope.knowledge_base_id != kb_id
            or scope.revision != revision
            or scope.epoch != epoch
        ):
            raise KnowledgeServiceError(409, "知识库版本已变化")
        return data

    async def store_web_snapshot(
        self,
        owner_id: str,
        session_id: str,
        page: Any,
        expires_at: Any,
    ) -> dict[str, Any]:
        payload = {
            "session_id": session_id,
            "url": page.url,
            "title": page.title,
            "text": page.text,
            "content_hash": page.content_hash,
            "fetched_at": page.fetched_at,
            "expires_at": expires_at,
        }
        response = await self.request(
            "POST", "/web-snapshots", owner_id=owner_id, json=payload
        )
        data = response.json()
        if not isinstance(data, dict):
            raise KnowledgeServiceError(503, "知识库服务返回无效数据")
        return data

    async def get_web_snapshot(
        self,
        snapshot_id: str,
        owner_id: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        self._require_resource_id(snapshot_id)
        path = f"/web-snapshots/{snapshot_id}"
        if session_id is not None:
            path += "?" + urlencode({"session_id": session_id})
        response = await self.request("GET", path, owner_id=owner_id)
        data = response.json()
        if not isinstance(data, dict):
            raise KnowledgeServiceError(503, "知识库服务返回无效数据")
        return data

    @staticmethod
    def _require_resource_id(value: str) -> None:
        if not _RESOURCE_ID.fullmatch(value):
            raise ValueError("invalid knowledge resource id")

    async def validate_scope(
        self,
        kb_id: str,
        owner_id: str,
        revision: int,
        epoch: int,
    ) -> KnowledgeScope:
        scope = await self.get_scope(kb_id, owner_id)
        if scope.revision != revision or scope.epoch != epoch:
            raise KnowledgeServiceError(409, "知识库版本已变化")
        return scope


knowledge_client = KnowledgeClient()
