"""Adaptive terminal learning paths are published as durable resources."""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from models.adaptive import NextStepDecision
from models.adaptive_session import AdaptiveSessionAggregate, AdaptiveTurnResponse
from models.learning_path import LearningPath, LearningPathResource
import routers.adaptive as adaptive_router
from services.adaptive_sessions import AdaptiveSessionStore
from services.learning_path_store import (
    LearningPathCorruptError,
    LearningPathCreationConflictError,
    LearningPathPayloadTooLargeError,
    LearningPathStore,
)
from services.quiz_sessions import QuizSessionApiError


def _path(document_id: str = "notes.md") -> LearningPath:
    return LearningPath.model_validate(
        {
            "document_id": document_id,
            "title": "Durable sorting path",
            "total_stages": 1,
            "stages": [
                {
                    "stage": 1,
                    "title": "Stable sorting",
                    "topics": ["merge sort"],
                    "description": "Learn why merge sort is stable.",
                    "estimated_minutes": 20,
                }
            ],
        }
    )


class TestAdaptiveLearningPathPublishing(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        database_path = str(Path(self.tempdir.name) / "publishing.sqlite3")
        self.adaptive_store = AdaptiveSessionStore(sqlite_path=database_path)
        self.path_store = LearningPathStore(sqlite_path=database_path)
        self.adaptive_patch = patch.object(
            adaptive_router,
            "adaptive_sessions",
            self.adaptive_store,
        )
        self.path_patch = patch.object(
            adaptive_router,
            "learning_path_store",
            self.path_store,
        )
        self.adaptive_patch.start()
        self.path_patch.start()

    def tearDown(self) -> None:
        self.path_patch.stop()
        self.adaptive_patch.stop()
        self.tempdir.cleanup()

    async def _terminal_record(
        self,
        *,
        session_id: str = "adapt_legacy_terminal",
        user_id: str = "custom-user",
        document_id: str = "notes.md",
        switch_to_plan: bool = True,
    ):
        action = "switch_to_plan" if switch_to_plan else "finish"
        reason = "switch_to_plan" if switch_to_plan else "agent_finish"
        decision = NextStepDecision(
            action=action,
            topic="sorting",
            reason="use a structured path",
        )
        artifact = adaptive_router._artifact(
            session_id=session_id,
            turn=1,
            decision=decision,
            trajectory=[adaptive_router._turn_from_decision(1, decision)],
            mastery=0.2,
            done=True,
            terminate_reason=reason,
            learning_path=(_path(document_id).model_dump(mode="json") if switch_to_plan else None),
        )
        aggregate = AdaptiveSessionAggregate(
            adaptive_session_id=session_id,
            user_id=user_id,
            document_id=document_id,
            goal="learn sorting",
            status="completed",
            current_artifact=artifact,
        )
        created = await self.adaptive_store.create(aggregate)
        return created.record, artifact

    async def test_legacy_terminal_aggregate_publishes_without_schema_marker(self) -> None:
        record, artifact = await self._terminal_record()

        response = await adaptive_router._response(record, artifact)

        self.assertRegex(response.learning_path_id, r"^lp_[0-9a-f]{32}$")
        self.assertNotIn(
            "learning_path_id",
            record.aggregate.model_dump(mode="json"),
        )
        published = await self.path_store.get(response.learning_path_id)
        self.assertEqual(published.user_id, "custom-user")
        self.assertEqual(published.path, _path())
        resource = LearningPathResource(
            schema_version=published.schema_version,
            learning_path_id=published.path_id,
            user_id=published.user_id,
            path=published.path,
            progress={"revision": 1, "completed_through": 0},
            created_at=published.created_at,
            expires_at=None,
        )
        self.assertEqual(resource.user_id, "custom-user")

    async def test_creation_key_and_fingerprint_are_stable_and_content_bound(self) -> None:
        record, artifact = await self._terminal_record(session_id="adapt_binding_contract")
        find = AsyncMock(wraps=self.path_store.find_by_creation)
        create = AsyncMock(wraps=self.path_store.create)
        fake_store = SimpleNamespace(find_by_creation=find, create=create)

        with patch.object(adaptive_router, "learning_path_store", fake_store):
            response = await adaptive_router._response(record, artifact)

        self.assertIsNotNone(response.learning_path_id)
        expected_key = "adaptive_switch_to_plan_publish_v1:adapt_binding_contract"
        create_kwargs = create.await_args.kwargs
        self.assertEqual(create_kwargs["idempotency_key"], expected_key)
        expected_fingerprint = adaptive_router._canonical_hash(
            {
                "operation": "adaptive_switch_to_plan_publish_v1",
                "adaptive_session_id": "adapt_binding_contract",
                "user_id": "custom-user",
                "document_id": "notes.md",
                "goal": "learn sorting",
                "source_created_at": record.updated_at,
                "learning_path": _path().model_dump(mode="json"),
            }
        )
        self.assertEqual(
            create_kwargs["request_fingerprint"],
            expected_fingerprint,
        )
        self.assertEqual(create_kwargs["source_created_at"], record.updated_at)

    async def test_published_resource_must_keep_the_terminal_source_time(self) -> None:
        record, artifact = await self._terminal_record(session_id="adapt_source_time")
        response = await adaptive_router._response(record, artifact)
        published = await self.path_store.get(response.learning_path_id)

        with self.assertRaises(QuizSessionApiError) as raised:
            adaptive_router._validate_published_path(
                replace(published, created_at=record.updated_at + 1),
                record,
                _path(),
            )

        self.assertEqual(raised.exception.code, "adaptive_learning_path_corrupt")

    async def test_get_and_repeated_responses_replay_one_published_resource(self) -> None:
        record, artifact = await self._terminal_record(session_id="adapt_get_replay")
        first, second = await asyncio.gather(
            adaptive_router._response(record, artifact),
            adaptive_router._response(record, artifact),
        )
        fetched = await adaptive_router.adaptive_snapshot("adapt_get_replay")

        self.assertEqual(
            {first.learning_path_id, second.learning_path_id, fetched.learning_path_id},
            {first.learning_path_id},
        )
        connection = sqlite3.connect(self.path_store._sqlite_path)
        try:
            count = connection.execute("SELECT COUNT(*) FROM studyloop_learning_paths").fetchone()[
                0
            ]
        finally:
            connection.close()
        self.assertEqual(count, 1)

    async def test_commit_then_raise_is_recovered_by_creation_lookup(self) -> None:
        record, artifact = await self._terminal_record(session_id="adapt_lost_create_ack")

        async def commit_then_raise(*args, **kwargs):
            await self.path_store.create(*args, **kwargs)
            raise RuntimeError("lost commit acknowledgement")

        fake_store = SimpleNamespace(
            find_by_creation=self.path_store.find_by_creation,
            create=commit_then_raise,
        )
        with patch.object(adaptive_router, "learning_path_store", fake_store):
            response = await adaptive_router._response(record, artifact)

        self.assertRegex(response.learning_path_id, r"^lp_[0-9a-f]{32}$")

    async def test_failed_first_publish_is_repaired_by_get(self) -> None:
        record, artifact = await self._terminal_record(session_id="adapt_publish_retry")
        unavailable = SimpleNamespace(
            find_by_creation=AsyncMock(return_value=None),
            create=AsyncMock(side_effect=RuntimeError("database unavailable")),
        )
        with (
            patch.object(adaptive_router, "learning_path_store", unavailable),
            self.assertRaises(QuizSessionApiError) as raised,
        ):
            await adaptive_router._response(record, artifact)

        self.assertEqual(
            raised.exception.code,
            "adaptive_learning_path_store_unavailable",
        )
        repaired = await adaptive_router.adaptive_snapshot("adapt_publish_retry")
        self.assertRegex(repaired.learning_path_id, r"^lp_[0-9a-f]{32}$")

    async def test_non_switch_response_does_not_access_path_store(self) -> None:
        record, artifact = await self._terminal_record(
            session_id="adapt_finish_without_path",
            switch_to_plan=False,
        )
        forbidden = SimpleNamespace(
            find_by_creation=AsyncMock(side_effect=AssertionError("unexpected path lookup")),
            create=AsyncMock(side_effect=AssertionError("unexpected path create")),
        )
        with patch.object(adaptive_router, "learning_path_store", forbidden):
            response = await adaptive_router._response(record, artifact)

        self.assertIsNone(response.learning_path_id)
        forbidden.find_by_creation.assert_not_awaited()
        forbidden.create.assert_not_awaited()

    async def test_cancelled_publish_is_not_mapped_to_an_http_error(self) -> None:
        record, artifact = await self._terminal_record(session_id="adapt_cancel_publish")
        cancelled = SimpleNamespace(
            find_by_creation=AsyncMock(return_value=None),
            create=AsyncMock(side_effect=asyncio.CancelledError()),
        )
        with (
            patch.object(adaptive_router, "learning_path_store", cancelled),
            self.assertRaises(asyncio.CancelledError),
        ):
            await adaptive_router._response(record, artifact)

    async def test_known_store_errors_have_fixed_mappings(self) -> None:
        record, artifact = await self._terminal_record(session_id="adapt_publish_errors")
        cases = [
            (
                LearningPathCreationConflictError(),
                "adaptive_learning_path_corrupt",
            ),
            (LearningPathCorruptError(), "adaptive_learning_path_corrupt"),
        ]
        for error, expected_code in cases:
            with self.subTest(error=type(error).__name__):
                failing = SimpleNamespace(
                    find_by_creation=AsyncMock(side_effect=error),
                    create=AsyncMock(),
                )
                with (
                    patch.object(adaptive_router, "learning_path_store", failing),
                    self.assertRaises(QuizSessionApiError) as raised,
                ):
                    await adaptive_router._response(record, artifact)
                self.assertEqual(raised.exception.code, expected_code)

        too_large = SimpleNamespace(
            find_by_creation=AsyncMock(return_value=None),
            create=AsyncMock(side_effect=LearningPathPayloadTooLargeError()),
        )
        with (
            patch.object(adaptive_router, "learning_path_store", too_large),
            self.assertRaises(QuizSessionApiError) as raised,
        ):
            await adaptive_router._response(record, artifact)
        self.assertEqual(raised.exception.status_code, 413)
        self.assertEqual(raised.exception.code, "adaptive_learning_path_too_large")

    def test_network_response_requires_exactly_one_published_path_id(self) -> None:
        valid = {
            "adaptive_session_id": "adapt_response_contract",
            "turn": 1,
            "done": True,
            "turn_type": "quiz",
            "questions": [],
            "lesson": None,
            "decision": NextStepDecision(
                action="switch_to_plan",
                topic="sorting",
                reason="plan",
            ),
            "last_report_score": None,
            "last_report_gaps": [],
            "last_report_feedback": [],
            "mastery": 0.2,
            "trajectory": [
                adaptive_router._turn_from_decision(
                    1,
                    NextStepDecision(
                        action="switch_to_plan",
                        topic="sorting",
                        reason="plan",
                    ),
                )
            ],
            "summary": "Published a path.",
            "terminate_reason": "switch_to_plan",
            "learning_path": _path().model_dump(mode="json"),
            "revision": 1,
            "expires_at": 1.0,
            "busy": False,
        }
        with self.assertRaises(ValidationError):
            AdaptiveTurnResponse.model_validate(valid)

        valid["learning_path_id"] = "lp_" + "a" * 32
        response = AdaptiveTurnResponse.model_validate(valid)
        self.assertEqual(response.learning_path_id, "lp_" + "a" * 32)

        ordinary = dict(valid)
        ordinary.update(
            terminate_reason="agent_finish",
            learning_path=None,
            learning_path_id="lp_" + "b" * 32,
            decision=NextStepDecision(action="finish", topic="sorting"),
        )
        ordinary["trajectory"] = [adaptive_router._turn_from_decision(1, ordinary["decision"])]
        with self.assertRaises(ValidationError):
            AdaptiveTurnResponse.model_validate(ordinary)


if __name__ == "__main__":
    unittest.main()
