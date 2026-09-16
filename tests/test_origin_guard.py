"""Browser-origin guard contracts at the real ASGI boundary."""

from __future__ import annotations

import asyncio
import json
import re
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

import main
import routers.documents as documents_router
import routers.learning_path as learning_path_router
from services.vectorstore import DEFAULT_DOCUMENT_OWNER
from services.origin_guard import ALLOWED_BROWSER_ORIGINS, OriginGuardMiddleware
from services.request_context import RequestContextMiddleware


_REQUEST_ID_RE = re.compile(r"^req_[0-9a-f]{32}$")
_TRUSTED_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4001",
    "http://127.0.0.1:4001",
    "http://localhost:8001",
    "http://127.0.0.1:8001",
)
_LEARNING_PATH = {
    "document_id": "notes.md",
    "title": "RAG 学习路径",
    "total_stages": 1,
    "stages": [
        {
            "stage": 1,
            "title": "基础",
            "topics": ["RAG"],
            "description": "理解检索增强生成。",
            "estimated_minutes": 20,
        }
    ],
}


class TestOriginGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(main.app, raise_server_exceptions=False)

    def assert_request_id(self, response) -> str:
        request_id = response.headers["x-request-id"]
        self.assertRegex(request_id, _REQUEST_ID_RE)
        return request_id

    def assert_forbidden_without_cors(self, response) -> None:
        self.assertEqual(response.status_code, 403)
        self.assertIsNone(response.headers.get("access-control-allow-origin"))
        request_id = self.assert_request_id(response)
        payload = response.json()
        self.assertEqual(payload.get("detail"), "请求来源不受信任")
        self.assertEqual(payload.get("code"), "origin_not_allowed")
        self.assertEqual(payload.get("request_id"), request_id)

    def test_untrusted_multipart_upload_is_rejected_before_ingestion(self) -> None:
        parse_upload = AsyncMock()
        split_documents = MagicMock()
        deal_document = AsyncMock()

        with (
            patch.object(documents_router, "parse_upload", parse_upload),
            patch.object(
                documents_router.default_chunker,
                "split_documents",
                split_documents,
            ),
            patch.object(documents_router, "deal_document", deal_document),
        ):
            response = self.client.post(
                "/documents/upload",
                headers={"Origin": "https://attacker.example"},
                files={
                    "file": (
                        "notes.md",
                        b"# Untrusted upload\n\nMust never be parsed.",
                        "text/markdown",
                    )
                },
            )

        self.assert_forbidden_without_cors(response)
        parse_upload.assert_not_awaited()
        split_documents.assert_not_called()
        deal_document.assert_not_awaited()

    def test_untrusted_bodyless_post_is_rejected_before_handler(self) -> None:
        request_id = "req_" + "b" * 32
        generate_learning_path = AsyncMock(return_value=_LEARNING_PATH)

        with patch.object(
            learning_path_router,
            "generate_learning_path",
            generate_learning_path,
        ):
            response = self.client.post(
                "/learning-path/notes.md",
                headers={
                    "Origin": "https://attacker.example",
                    "X-Request-ID": request_id,
                },
            )

        self.assert_forbidden_without_cors(response)
        self.assertEqual(response.headers["x-request-id"], request_id)
        self.assertEqual(response.json()["request_id"], request_id)
        generate_learning_path.assert_not_awaited()

    def test_every_unsafe_http_method_is_rejected_before_routing(self) -> None:
        handler = MagicMock()
        boundary_app = FastAPI()

        @boundary_app.api_route(
            "/mutate",
            methods=["POST", "PUT", "PATCH", "DELETE"],
        )
        async def mutate():
            handler()
            return {"status": "mutated"}

        boundary_app.add_middleware(
            CORSMiddleware,
            allow_origins=ALLOWED_BROWSER_ORIGINS,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID"],
        )
        boundary_app.add_middleware(OriginGuardMiddleware)
        boundary_app.add_middleware(RequestContextMiddleware)
        client = TestClient(boundary_app, raise_server_exceptions=False)

        for method in ("POST", "PUT", "PATCH", "DELETE"):
            with self.subTest(method=method):
                response = client.request(
                    method,
                    "/mutate",
                    headers={"Origin": "https://attacker.example"},
                )

                self.assert_forbidden_without_cors(response)

        handler.assert_not_called()

    def test_rejection_does_not_read_request_body_or_call_inner_app(self) -> None:
        request_id = "req_" + "a" * 32
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/mutate",
            "raw_path": b"/mutate",
            "query_string": b"",
            "headers": [
                (b"origin", b"https://attacker.example"),
                (b"x-request-id", request_id.encode()),
                (b"content-length", b"999999999"),
            ],
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
            "state": {},
        }
        inner_app = AsyncMock()
        receive = AsyncMock(side_effect=AssertionError("origin guard must not read the body"))
        sent = []

        async def send(message):
            sent.append(message)

        middleware = RequestContextMiddleware(OriginGuardMiddleware(inner_app))
        asyncio.run(middleware(scope, receive, send))

        receive.assert_not_awaited()
        inner_app.assert_not_awaited()
        start = next(message for message in sent if message["type"] == "http.response.start")
        body = next(message for message in sent if message["type"] == "http.response.body")
        response_headers = dict(start["headers"])
        payload = json.loads(body["body"])
        self.assertEqual(start["status"], 403)
        self.assertEqual(response_headers[b"x-request-id"].decode(), request_id)
        self.assertEqual(payload["code"], "origin_not_allowed")
        self.assertEqual(payload["request_id"], request_id)

    def test_safe_methods_and_non_http_scopes_reach_inner_app(self) -> None:
        cases = (
            {
                "type": "http",
                "method": "HEAD",
                "headers": [(b"origin", b"https://attacker.example")],
            },
            {
                "type": "http",
                "method": "OPTIONS",
                "headers": [(b"origin", b"https://attacker.example")],
            },
            {
                "type": "websocket",
                "headers": [(b"origin", b"https://attacker.example")],
            },
        )

        for scope in cases:
            with self.subTest(scope_type=scope["type"], method=scope.get("method")):
                inner_app = AsyncMock()
                receive = AsyncMock()
                send = AsyncMock()
                middleware = OriginGuardMiddleware(inner_app)

                asyncio.run(middleware(scope, receive, send))

                inner_app.assert_awaited_once_with(scope, receive, send)

    def test_untrusted_simple_content_types_are_rejected_before_handler(self) -> None:
        requests = (
            (None, None),
            ("text/plain", b"ignored"),
            ("application/x-www-form-urlencoded", b"ignored=value"),
        )
        generate_learning_path = AsyncMock(return_value=_LEARNING_PATH)

        with patch.object(
            learning_path_router,
            "generate_learning_path",
            generate_learning_path,
        ):
            for content_type, content in requests:
                headers = {"Origin": "https://attacker.example"}
                if content_type is not None:
                    headers["Content-Type"] = content_type
                with self.subTest(content_type=content_type):
                    response = self.client.post(
                        "/learning-path/notes.md",
                        headers=headers,
                        content=content,
                    )

                    self.assert_forbidden_without_cors(response)

        generate_learning_path.assert_not_awaited()

    def test_all_trusted_origins_can_make_unsafe_requests_with_cors(self) -> None:
        generate_learning_path = AsyncMock(return_value=_LEARNING_PATH)

        with patch.object(
            learning_path_router,
            "generate_learning_path",
            generate_learning_path,
        ):
            for origin in _TRUSTED_ORIGINS:
                with self.subTest(origin=origin):
                    response = self.client.post(
                        "/learning-path/notes.md",
                        headers={"Origin": origin},
                    )

                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(
                        response.headers.get("access-control-allow-origin"),
                        origin,
                    )
                    self.assert_request_id(response)

        self.assertEqual(generate_learning_path.await_count, len(_TRUSTED_ORIGINS))

    def test_cli_post_without_origin_remains_allowed(self) -> None:
        generate_learning_path = AsyncMock(return_value=_LEARNING_PATH)

        with patch.object(
            learning_path_router,
            "generate_learning_path",
            generate_learning_path,
        ):
            response = self.client.post("/learning-path/notes.md")

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.headers.get("access-control-allow-origin"))
        self.assert_request_id(response)
        generate_learning_path.assert_awaited_once_with(
            "notes.md", owner_id=DEFAULT_DOCUMENT_OWNER
        )

    def test_noncanonical_and_lookalike_origins_are_rejected(self) -> None:
        rejected_origins = (
            "null",
            "",
            "not-an-origin",
            "://localhost:5173",
            " http://localhost:5173",
            "http://localhost:5173 ",
            "http://localhost:5173, https://attacker.example",
            "HTTP://localhost:5173",
            "http://LOCALHOST:5173",
            "http://localhost:5173/",
            "http://localhost.evil.example:5173",
            "https://localhost:5173",
            "http://localhost:5174",
        )
        generate_learning_path = AsyncMock(return_value=_LEARNING_PATH)

        with patch.object(
            learning_path_router,
            "generate_learning_path",
            generate_learning_path,
        ):
            for origin in rejected_origins:
                with self.subTest(origin=origin):
                    response = self.client.post(
                        "/learning-path/notes.md",
                        headers={"Origin": origin},
                    )

                    self.assert_forbidden_without_cors(response)

        generate_learning_path.assert_not_awaited()

    def test_duplicate_origin_headers_are_rejected(self) -> None:
        duplicate_headers = (
            [
                ("Origin", "http://localhost:5173"),
                ("Origin", "http://localhost:5173"),
            ],
            [
                ("Origin", "http://localhost:5173"),
                ("Origin", "https://attacker.example"),
            ],
        )
        generate_learning_path = AsyncMock(return_value=_LEARNING_PATH)

        with patch.object(
            learning_path_router,
            "generate_learning_path",
            generate_learning_path,
        ):
            for headers in duplicate_headers:
                with self.subTest(headers=headers):
                    response = self.client.post(
                        "/learning-path/notes.md",
                        headers=headers,
                    )

                    self.assert_forbidden_without_cors(response)

        generate_learning_path.assert_not_awaited()

    def test_untrusted_origin_does_not_block_safe_get_or_gain_cors(self) -> None:
        response = self.client.get(
            "/health/live",
            headers={"Origin": "https://attacker.example"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"name": "StudyLoop", "status": "ok", "auth": "anonymous"})
        self.assertIsNone(response.headers.get("access-control-allow-origin"))
        self.assert_request_id(response)

    def test_cors_preflight_contract_is_unchanged(self) -> None:
        for origin in _TRUSTED_ORIGINS:
            with self.subTest(origin=origin):
                response = self.client.options(
                    "/learning-path/notes.md",
                    headers={
                        "Origin": origin,
                        "Access-Control-Request-Method": "POST",
                    },
                )

                self.assertEqual(response.status_code, 200)
                self.assertEqual(
                    response.headers.get("access-control-allow-origin"),
                    origin,
                )
                self.assert_request_id(response)

        response = self.client.options(
            "/learning-path/notes.md",
            headers={
                "Origin": "https://attacker.example",
                "Access-Control-Request-Method": "POST",
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIsNone(response.headers.get("access-control-allow-origin"))
        self.assert_request_id(response)


if __name__ == "__main__":
    unittest.main()
