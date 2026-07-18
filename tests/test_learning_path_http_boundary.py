"""HTTP mappings for grounded learning-path retrieval failures."""

import unittest
from unittest.mock import AsyncMock, patch

from chromadb.errors import InternalError, NotFoundError
from fastapi.testclient import TestClient
from pydantic import ValidationError

from main import app
from models.learning_path import LearningPath
import routers.learning_path as learning_path_router
from services.learning_path import LearningPathEvidenceUnavailableError


class TestLearningPathHttpBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app, raise_server_exceptions=False)

    def test_missing_document_is_404(self) -> None:
        with patch.object(
            learning_path_router,
            "generate_learning_path",
            AsyncMock(side_effect=NotFoundError("missing")),
        ):
            response = self.client.post("/learning-path/missing.md")

        self.assertEqual(response.status_code, 404, response.text)
        self.assertEqual(response.json()["detail"], "文档不存在")

    def test_empty_document_evidence_is_422(self) -> None:
        with patch.object(
            learning_path_router,
            "generate_learning_path",
            AsyncMock(side_effect=LearningPathEvidenceUnavailableError("empty")),
        ):
            response = self.client.post("/learning-path/empty.md")

        self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(response.json()["detail"], "文档没有可用于生成学习路径的内容")

    def test_document_store_failure_is_503(self) -> None:
        with patch.object(
            learning_path_router,
            "generate_learning_path",
            AsyncMock(side_effect=InternalError("database unavailable")),
        ):
            response = self.client.post("/learning-path/notes.md")

        self.assertEqual(response.status_code, 503, response.text)
        self.assertEqual(response.json()["detail"], "文档存储暂时不可用")

    def test_invalid_provider_shape_is_a_redacted_503(self) -> None:
        with self.assertRaises(ValidationError) as raised:
            LearningPath.model_validate({})

        with patch.object(
            learning_path_router,
            "generate_learning_path",
            AsyncMock(side_effect=raised.exception),
        ):
            response = self.client.post("/learning-path/notes.md")

        self.assertEqual(response.status_code, 503, response.text)
        self.assertEqual(response.json()["detail"], "模型返回的学习路径格式无效")
        self.assertNotIn("validation error", response.text.lower())


if __name__ == "__main__":
    unittest.main()
