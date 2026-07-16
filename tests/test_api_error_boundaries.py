"""HTTP boundaries for provider failures, local CORS, and first-user state."""

import unittest
from unittest.mock import AsyncMock, patch

import httpx

from fastapi.testclient import TestClient
from chromadb.errors import InternalError, NotFoundError
from openai import APIConnectionError

import routers.autonomous as autonomous_router
import routers.documents as documents_router
import routers.learning_path as learning_path_router
import routers.user as user_router
import services.learning_path as learning_path_service
from main import app
from models.learning_path import CompressedReport, PathBrief
from services.retry import RetryExhausted
from services.vectorstore import DocumentAlreadyExistsError


class TestApiErrorBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app, raise_server_exceptions=False)

    def test_dev_cors_allows_localhost_and_loopback_but_not_unknown_origins(self):
        for origin in ("http://localhost:5173", "http://127.0.0.1:5173"):
            with self.subTest(origin=origin):
                response = self.client.options(
                    "/documents",
                    headers={
                        "Origin": origin,
                        "Access-Control-Request-Method": "GET",
                    },
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers.get("access-control-allow-origin"), origin)

        response = self.client.options(
            "/documents",
            headers={
                "Origin": "https://untrusted.example",
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIsNone(response.headers.get("access-control-allow-origin"))

    def test_missing_profile_is_a_normal_empty_state(self):
        origin = "http://127.0.0.1:5173"
        with patch.object(user_router, "get_user_profile", AsyncMock(return_value=None)):
            response = self.client.get(
                "/user/brand-new-user/profile", headers={"Origin": origin}
            )

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json())
        self.assertEqual(response.headers.get("access-control-allow-origin"), origin)

    def test_learning_path_provider_error_is_stable_503_with_cors(self):
        origin = "http://127.0.0.1:5173"
        provider_error = APIConnectionError(
            request=httpx.Request("POST", "https://provider.invalid/v1/chat/completions")
        )
        with patch.object(
            learning_path_router,
            "generate_learning_path",
            AsyncMock(side_effect=provider_error),
        ):
            response = self.client.post(
                "/learning-path/notes.md", headers={"Origin": origin}
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json(),
            {"error": "服务暂时不可用", "detail": "模型服务请求失败"},
        )
        self.assertEqual(response.headers.get("access-control-allow-origin"), origin)

    def test_autonomous_provider_error_is_not_reported_as_max_rounds(self):
        origin = "http://127.0.0.1:5173"
        finish = AsyncMock()
        with patch.object(
            autonomous_router, "check_injection", AsyncMock(return_value=(False, ""))
        ), patch.object(
            autonomous_router,
            "run_tool_round",
            AsyncMock(side_effect=RetryExhausted("provider down")),
        ), patch.object(autonomous_router, "llm_chat", finish):
            response = self.client.post(
                "/agent/autonomous",
                headers={"Origin": origin},
                json={"query": "学RAG", "user_id": "new-user"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json(),
            {"error": "服务暂时不可用", "detail": "模型服务请求失败"},
        )
        self.assertFalse(response.json().get("truncated", False))
        self.assertNotEqual(response.json().get("finalize_reason"), "max_rounds_truncated")
        self.assertEqual(response.headers.get("access-control-allow-origin"), origin)
        finish.assert_not_awaited()

    def test_upload_provider_failure_and_duplicate_have_explicit_statuses(self):
        origin = "http://127.0.0.1:5173"
        files = {"file": ("notes.md", b"# Notes\n\nRAG content", "text/markdown")}

        with patch.object(
            documents_router,
            "deal_document",
            AsyncMock(side_effect=RetryExhausted("embedding unavailable")),
        ):
            response = self.client.post(
                "/documents/upload", headers={"Origin": origin}, files=files
            )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json(),
            {"error": "服务暂时不可用", "detail": "模型服务请求失败"},
        )
        self.assertEqual(response.headers.get("access-control-allow-origin"), origin)

        with patch.object(
            documents_router,
            "deal_document",
            AsyncMock(side_effect=DocumentAlreadyExistsError("文档已存在")),
        ):
            response = self.client.post(
                "/documents/upload", headers={"Origin": origin}, files=files
            )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json(), {"detail": "文档已存在"})

    def test_delete_reports_missing_and_storage_failure(self):
        with patch.object(
            documents_router,
            "delete_document",
            AsyncMock(side_effect=NotFoundError("missing")),
        ):
            response = self.client.delete("/documents/missing.md")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "文档不存在"})

        with patch.object(
            documents_router,
            "delete_document",
            AsyncMock(side_effect=InternalError("storage unavailable")),
        ):
            response = self.client.delete("/documents/notes.md")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "文档存储暂时不可用"})

    def test_document_list_storage_failure_is_json_503(self):
        with patch.object(
            documents_router,
            "get_all_document",
            AsyncMock(side_effect=InternalError("storage unavailable")),
        ):
            response = self.client.get("/documents")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "文档存储暂时不可用"})

    def test_learning_path_retry_exhaustion_hides_provider_details(self):
        origin = "http://127.0.0.1:5173"
        with patch.object(
            learning_path_router,
            "generate_learning_path",
            AsyncMock(side_effect=RetryExhausted("internal-model-42 upstream body")),
        ):
            response = self.client.post(
                "/learning-path/notes.md", headers={"Origin": origin}
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json(),
            {"error": "服务暂时不可用", "detail": "模型服务请求失败"},
        )
        self.assertNotIn("internal-model-42", response.text)
        self.assertEqual(response.headers.get("access-control-allow-origin"), origin)


class TestLearningPathProviderBoundary(unittest.IsolatedAsyncioTestCase):
    async def test_synthesize_uses_retrying_llm_entrypoint(self):
        brief = PathBrief(
            title="RAG 学习路径",
            scope="RAG 基础",
            level="beginner",
            target_count=3,
            keywords=["retrieval", "generation"],
        )
        compressed = CompressedReport(
            summary="RAG combines retrieval and generation.",
            key_concepts=["retrieval", "generation"],
            suggested_stage_count=3,
        )
        parse = AsyncMock(side_effect=RetryExhausted("provider down"))

        with patch.object(learning_path_service, "llm_parse", parse):
            with self.assertRaises(RetryExhausted):
                await learning_path_service.synthesize("notes.md", brief, compressed)

        parse.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
