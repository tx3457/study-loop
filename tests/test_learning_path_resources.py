"""HTTP contracts for durable Learning Path resources."""

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from chromadb.errors import NotFoundError
from fastapi.testclient import TestClient

import routers.learning_path as learning_path_router
from main import app
from models.learning_path import LearningPath
from services.learning_path_store import (
    LearningPathCorruptError,
    LearningPathPayloadTooLargeError,
    LearningPathStore,
)
from services.retry import RetryExhausted


def _path(document_id: str = "notes.md", title: str = "可靠学习路径") -> LearningPath:
    return LearningPath.model_validate(
        {
            "document_id": document_id,
            "title": title,
            "total_stages": 2,
            "stages": [
                {
                    "stage": 1,
                    "title": "理解概念",
                    "topics": ["概念", "边界"],
                    "description": "先建立概念模型。",
                    "estimated_minutes": 20,
                },
                {
                    "stage": 2,
                    "title": "应用检验",
                    "topics": ["练习", "复盘"],
                    "description": "通过练习检验理解。",
                    "estimated_minutes": 30,
                },
            ],
        }
    )


class TestLearningPathResources(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = LearningPathStore(
            sqlite_path=str(Path(self.temp_dir.name) / "paths.sqlite3"),
        )
        self.store_patch = patch.object(
            learning_path_router,
            "learning_path_store",
            self.store,
        )
        self.store_patch.start()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client.close()
        self.store_patch.stop()
        self.temp_dir.cleanup()

    @staticmethod
    def _create_body(document_id: str = "notes.md") -> dict:
        return {"user_id": "default_user", "document_id": document_id}

    def test_create_replay_and_get_return_one_canonical_resource(self) -> None:
        generated = AsyncMock(return_value=_path())
        headers = {"Idempotency-Key": "learning-path-key-01"}
        with patch.object(learning_path_router, "generate_learning_path", generated):
            first = self.client.post(
                "/learning-paths",
                json=self._create_body(),
                headers=headers,
            )
            replay = self.client.post(
                "/learning-paths",
                json=self._create_body(),
                headers=headers,
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(replay.status_code, 200)
        self.assertEqual(replay.json(), first.json())
        self.assertEqual(generated.await_count, 1)
        body = first.json()
        self.assertEqual(body["schema_version"], 1)
        self.assertRegex(body["learning_path_id"], r"^lp_[0-9a-f]{32}$")
        self.assertEqual(body["user_id"], "default_user")
        self.assertEqual(body["path"]["document_id"], "notes.md")
        self.assertIsNone(body["expires_at"])

        loaded = self.client.get(
            f"/learning-paths/{body['learning_path_id']}"
        )
        self.assertEqual(loaded.status_code, 200)
        self.assertEqual(loaded.json(), body)
        current = self.client.get("/learning-paths/current")
        self.assertEqual(current.status_code, 200)
        self.assertEqual(current.json(), body)
        filtered = self.client.get(
            "/learning-paths/current",
            params={"document_id": "notes.md"},
        )
        self.assertEqual(filtered.json(), body)

    def test_replay_precedes_document_lookup_after_material_deletion(self) -> None:
        headers = {"Idempotency-Key": "learning-path-key-02"}
        generator = AsyncMock(return_value=_path())
        with patch.object(learning_path_router, "generate_learning_path", generator):
            created = self.client.post(
                "/learning-paths",
                json=self._create_body(),
                headers=headers,
            )
        self.assertEqual(created.status_code, 200)

        missing = AsyncMock(side_effect=NotFoundError("deleted"))
        with patch.object(learning_path_router, "generate_learning_path", missing):
            replay = self.client.post(
                "/learning-paths",
                json=self._create_body(),
                headers=headers,
            )
            new_request = self.client.post(
                "/learning-paths",
                json=self._create_body(),
                headers={"Idempotency-Key": "learning-path-key-03"},
            )

        self.assertEqual(replay.status_code, 200)
        self.assertEqual(replay.json(), created.json())
        self.assertEqual(new_request.status_code, 404)
        self.assertEqual(missing.await_count, 1)

    def test_concurrent_commit_wins_over_this_call_invalid_provider_candidate(self) -> None:
        headers = {"Idempotency-Key": "learning-path-key-race"}
        with patch.object(
            learning_path_router,
            "generate_learning_path",
            AsyncMock(return_value=_path()),
        ):
            created = self.client.post(
                "/learning-paths",
                json=self._create_body(),
                headers=headers,
            )
        self.assertEqual(created.status_code, 200)
        record = asyncio.run(
            self.store.get(created.json()["learning_path_id"])
        )
        self.assertIsNotNone(record)

        lookup = AsyncMock(side_effect=[None, record])
        invalid = AsyncMock(return_value={"document_id": "notes.md"})
        with (
            patch.object(self.store, "find_by_creation", lookup),
            patch.object(learning_path_router, "generate_learning_path", invalid),
        ):
            replay = self.client.post(
                "/learning-paths",
                json=self._create_body(),
                headers=headers,
            )

        self.assertEqual(replay.status_code, 200)
        self.assertEqual(replay.json(), created.json())
        self.assertEqual(lookup.await_count, 2)

    def test_concurrent_commit_wins_over_unclassified_provider_failure(self) -> None:
        headers = {"Idempotency-Key": "learning-path-key-provider-race"}
        with patch.object(
            learning_path_router,
            "generate_learning_path",
            AsyncMock(return_value=_path()),
        ):
            created = self.client.post(
                "/learning-paths",
                json=self._create_body(),
                headers=headers,
            )
        self.assertEqual(created.status_code, 200)
        record = asyncio.run(
            self.store.get(created.json()["learning_path_id"])
        )
        self.assertIsNotNone(record)

        lookup = AsyncMock(side_effect=[None, record])
        failed = AsyncMock(side_effect=RetryExhausted("private provider body"))
        with (
            patch.object(self.store, "find_by_creation", lookup),
            patch.object(learning_path_router, "generate_learning_path", failed),
        ):
            replay = self.client.post(
                "/learning-paths",
                json=self._create_body(),
                headers=headers,
            )

        self.assertEqual(replay.status_code, 200)
        self.assertEqual(replay.json(), created.json())
        self.assertEqual(lookup.await_count, 2)

    def test_missing_or_mismatched_key_fails_before_generation(self) -> None:
        generated = AsyncMock(return_value=_path())
        with patch.object(learning_path_router, "generate_learning_path", generated):
            missing = self.client.post(
                "/learning-paths",
                json=self._create_body(),
            )
            malformed = self.client.post(
                "/learning-paths",
                json=self._create_body(),
                headers={"Idempotency-Key": "short"},
            )
            first = self.client.post(
                "/learning-paths",
                json=self._create_body(),
                headers={"Idempotency-Key": "learning-path-key-04"},
            )
            mismatch = self.client.post(
                "/learning-paths",
                json=self._create_body("other.md"),
                headers={"Idempotency-Key": "learning-path-key-04"},
            )

        self.assertEqual(missing.status_code, 400)
        self.assertEqual(malformed.status_code, 400)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(mismatch.status_code, 409)
        self.assertEqual(mismatch.json()["code"], "idempotency_conflict")
        self.assertEqual(mismatch.json()["reason"], "payload_mismatch")
        self.assertEqual(generated.await_count, 1)

    def test_custom_web_identity_and_invalid_provider_output_fail_closed(self) -> None:
        custom = self.client.post(
            "/learning-paths",
            json={"user_id": "someone_else", "document_id": "notes.md"},
            headers={"Idempotency-Key": "learning-path-key-05"},
        )
        self.assertEqual(custom.status_code, 422)

        invalid = AsyncMock(return_value={"document_id": "notes.md"})
        with patch.object(learning_path_router, "generate_learning_path", invalid):
            response = self.client.post(
                "/learning-paths",
                json=self._create_body(),
                headers={"Idempotency-Key": "learning-path-key-06"},
            )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "模型返回的学习路径格式无效")

    def test_store_failures_are_stable_and_payload_limit_maps_to_413(self) -> None:
        generated = AsyncMock(return_value=_path())
        with (
            patch.object(
                self.store,
                "find_by_creation",
                AsyncMock(side_effect=RuntimeError("private lookup details")),
            ),
            patch.object(learning_path_router, "generate_learning_path", generated),
        ):
            lookup_failure = self.client.post(
                "/learning-paths",
                json=self._create_body(),
                headers={"Idempotency-Key": "learning-path-key-07"},
            )
        self.assertEqual(lookup_failure.status_code, 503)
        self.assertNotIn("private", lookup_failure.text)
        generated.assert_not_awaited()

        with (
            patch.object(
                self.store,
                "find_by_creation",
                AsyncMock(return_value=None),
            ),
            patch.object(
                self.store,
                "create",
                AsyncMock(side_effect=RuntimeError("private create details")),
            ),
            patch.object(learning_path_router, "generate_learning_path", generated),
        ):
            create_failure = self.client.post(
                "/learning-paths",
                json=self._create_body(),
                headers={"Idempotency-Key": "learning-path-key-08"},
            )
        self.assertEqual(create_failure.status_code, 503)
        self.assertNotIn("private", create_failure.text)

        with (
            patch.object(
                self.store,
                "find_by_creation",
                AsyncMock(return_value=None),
            ),
            patch.object(
                self.store,
                "create",
                AsyncMock(
                    side_effect=LearningPathPayloadTooLargeError("private size")
                ),
            ),
            patch.object(learning_path_router, "generate_learning_path", generated),
        ):
            too_large = self.client.post(
                "/learning-paths",
                json=self._create_body(),
                headers={"Idempotency-Key": "learning-path-key-09"},
            )
        self.assertEqual(too_large.status_code, 413)
        self.assertNotIn("private", too_large.text)

    def test_missing_and_corrupt_records_have_stable_safe_errors(self) -> None:
        current = self.client.get("/learning-paths/current")
        self.assertEqual(current.status_code, 200)
        self.assertIsNone(current.json())

        blank_filter = self.client.get(
            "/learning-paths/current",
            params={"document_id": " "},
        )
        self.assertEqual(blank_filter.status_code, 422)

        malformed = self.client.get("/learning-paths/not-a-path")
        self.assertEqual(malformed.status_code, 404)
        self.assertEqual(malformed.json()["detail"], "学习路径不存在")

        missing = self.client.get(
            "/learning-paths/lp_00000000000000000000000000000000"
        )
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.json()["detail"], "学习路径不存在")

        with patch.object(
            self.store,
            "get",
            AsyncMock(side_effect=LearningPathCorruptError("private row details")),
        ):
            corrupt = self.client.get(
                "/learning-paths/lp_00000000000000000000000000000000"
            )
        self.assertEqual(corrupt.status_code, 503)
        self.assertEqual(corrupt.json()["detail"], "学习路径存储暂时不可用")
        self.assertNotIn("private", corrupt.text)

        with patch.object(
            self.store,
            "get_current",
            AsyncMock(side_effect=LearningPathCorruptError("private row details")),
        ):
            corrupt_current = self.client.get("/learning-paths/current")
        self.assertEqual(corrupt_current.status_code, 503)
        self.assertNotIn("private", corrupt_current.text)


if __name__ == "__main__":
    unittest.main()
