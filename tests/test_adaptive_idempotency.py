"""Retry-safety contracts for the stateful adaptive submit endpoint."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from main import app
from models.adaptive import AdaptiveTurn, NextStepDecision
from models.grader import GradingReport
from models.quiz import Question
from models.session import QuizSession
import routers.adaptive as adaptive_router
from services.idempotency import IdempotencyStore


class TestAdaptiveSubmitIdempotency(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = IdempotencyStore(
            sqlite_path=str(Path(self.tempdir.name) / "receipts.sqlite3")
        )
        self.store_patch = patch.object(
            adaptive_router, "request_idempotency", self.store
        )
        self.store_patch.start()
        adaptive_router._sessions.clear()
        adaptive_router.sessions.clear()
        adaptive_router._submit_locks.clear()
        self.client = TestClient(app, raise_server_exceptions=False)

    def tearDown(self) -> None:
        self.store_patch.stop()
        adaptive_router._sessions.clear()
        adaptive_router.sessions.clear()
        adaptive_router._submit_locks.clear()
        self.tempdir.cleanup()

    def _seed_quiz_turn(self) -> str:
        adaptive_session_id = "adapt_retry_contract"
        quiz_session_id = "quiz_retry_contract"
        decision = NextStepDecision(
            action="continue",
            topic="排序",
            count=1,
            reason="继续验证",
        )
        adaptive_router._sessions[adaptive_session_id] = (
            adaptive_router.AdaptiveSession(
                adaptive_session_id=adaptive_session_id,
                user_id="user-1",
                document_id="notes.md",
                goal="学习排序",
                turn=1,
                history=[
                    AdaptiveTurn(
                        turn=1,
                        action="continue",
                        topic="排序",
                        difficulty_score=0.5,
                        reason="继续验证",
                    )
                ],
                current_quiz_session_id=quiz_session_id,
                current_decision=decision,
                current_turn_type="quiz",
            )
        )
        adaptive_router.sessions[quiz_session_id] = QuizSession(
            session_id=quiz_session_id,
            document_id="notes.md",
            user_id="user-1",
            questions=[
                Question(
                    question="归并排序是否稳定？",
                    options=["A. 是", "B. 否"],
                    answer="A",
                    explanation="归并时可以保持相等元素的相对次序。",
                    source="notes.md",
                )
            ],
            user_answers=[],
            status="active",
        )
        return adaptive_session_id

    @staticmethod
    def _report() -> GradingReport:
        return GradingReport(
            session_id="quiz_retry_contract",
            total=1,
            correct=1,
            score=1.0,
            grades=[],
        )

    def test_completed_submit_replays_without_regrading(self) -> None:
        session_id = self._seed_quiz_turn()
        grade = AsyncMock(return_value=self._report())
        finish = NextStepDecision(
            action="finish",
            topic="排序",
            reason="已经掌握",
        )
        headers = {"Idempotency-Key": "adaptive-submit-replay-1"}
        payload = {
            "adaptive_session_id": session_id,
            "turn": 1,
            "answers": ["A. 是"],
        }

        with (
            patch.object(adaptive_router, "_grade_and_update", grade),
            patch.object(adaptive_router, "get_mastery", AsyncMock(return_value=1.0)),
            patch.object(
                adaptive_router, "get_weak_points", AsyncMock(return_value=[])
            ),
            patch.object(
                adaptive_router, "decide_next_step", AsyncMock(return_value=finish)
            ),
            patch.object(
                adaptive_router,
                "should_terminate",
                return_value=(True, "mastery_reached"),
            ),
        ):
            first = self.client.post(
                "/agent/adaptive/submit", json=payload, headers=headers
            )
            replay = self.client.post(
                "/agent/adaptive/submit", json=payload, headers=headers
            )

        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(replay.status_code, 200, replay.text)
        self.assertEqual(replay.json(), first.json())
        self.assertEqual(grade.await_count, 1)

    def test_failure_after_effect_is_ambiguous_and_invalidates_session(self) -> None:
        session_id = self._seed_quiz_turn()
        grade = AsyncMock(side_effect=RuntimeError("provider disconnected"))
        headers = {"Idempotency-Key": "adaptive-submit-ambiguous-1"}
        payload = {
            "adaptive_session_id": session_id,
            "turn": 1,
            "answers": ["A. 是"],
        }

        with patch.object(adaptive_router, "_grade_and_update", grade):
            first = self.client.post(
                "/agent/adaptive/submit", json=payload, headers=headers
            )
            replay = self.client.post(
                "/agent/adaptive/submit", json=payload, headers=headers
            )

        self.assertEqual(first.status_code, 409, first.text)
        self.assertEqual(first.json()["code"], "side_effect_ambiguous")
        self.assertEqual(replay.status_code, 409, replay.text)
        self.assertEqual(replay.json()["reason"], "ambiguous")
        self.assertNotIn(session_id, adaptive_router._sessions)
        self.assertEqual(grade.await_count, 1)

    def test_stale_turn_is_rejected_before_grading(self) -> None:
        session_id = self._seed_quiz_turn()
        grade = AsyncMock(return_value=self._report())

        with patch.object(adaptive_router, "_grade_and_update", grade):
            response = self.client.post(
                "/agent/adaptive/submit",
                json={
                    "adaptive_session_id": session_id,
                    "turn": 2,
                    "answers": ["A. 是"],
                },
                headers={"Idempotency-Key": "adaptive-submit-stale-1"},
            )

        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(grade.await_count, 0)

    def test_abort_store_failure_after_effect_still_invalidates_session(self) -> None:
        session_id = self._seed_quiz_turn()
        grade = AsyncMock(side_effect=RuntimeError("provider disconnected"))

        with (
            patch.object(adaptive_router, "_grade_and_update", grade),
            patch.object(
                self.store,
                "abort",
                AsyncMock(side_effect=RuntimeError("receipt database unavailable")),
            ),
        ):
            response = self.client.post(
                "/agent/adaptive/submit",
                json={
                    "adaptive_session_id": session_id,
                    "turn": 1,
                    "answers": ["A. 是"],
                },
                headers={"Idempotency-Key": "adaptive-abort-failure-1"},
            )

        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["code"], "side_effect_ambiguous")
        self.assertNotIn(session_id, adaptive_router._sessions)

    def test_failure_after_effect_without_key_still_invalidates_session(self) -> None:
        session_id = self._seed_quiz_turn()

        with patch.object(
            adaptive_router,
            "_grade_and_update",
            AsyncMock(side_effect=RuntimeError("provider disconnected")),
        ):
            response = self.client.post(
                "/agent/adaptive/submit",
                json={
                    "adaptive_session_id": session_id,
                    "turn": 1,
                    "answers": ["A. 是"],
                },
            )

        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["code"], "side_effect_ambiguous")
        self.assertNotIn(session_id, adaptive_router._sessions)

    def test_turn_is_required(self) -> None:
        session_id = self._seed_quiz_turn()

        response = self.client.post(
            "/agent/adaptive/submit",
            json={
                "adaptive_session_id": session_id,
                "answers": ["A. 是"],
            },
        )

        self.assertEqual(response.status_code, 422, response.text)
        self.assertIn(session_id, adaptive_router._sessions)


class TestAdaptiveSubmitCancellation(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = IdempotencyStore(
            sqlite_path=str(Path(self.tempdir.name) / "receipts.sqlite3")
        )
        self.store_patch = patch.object(
            adaptive_router,
            "request_idempotency",
            self.store,
        )
        self.store_patch.start()
        adaptive_router._sessions.clear()
        adaptive_router.sessions.clear()
        adaptive_router._submit_locks.clear()
        self.session_id = "adapt_cancel_contract"
        quiz_session_id = "quiz_cancel_contract"
        decision = NextStepDecision(
            action="continue",
            topic="排序",
            count=1,
            reason="继续验证",
        )
        adaptive_router._sessions[self.session_id] = adaptive_router.AdaptiveSession(
            adaptive_session_id=self.session_id,
            user_id="user-1",
            document_id="notes.md",
            goal="学习排序",
            turn=1,
            history=[],
            current_quiz_session_id=quiz_session_id,
            current_decision=decision,
            current_turn_type="quiz",
        )
        adaptive_router.sessions[quiz_session_id] = QuizSession(
            session_id=quiz_session_id,
            document_id="notes.md",
            user_id="user-1",
            questions=[
                Question(
                    question="归并排序是否稳定？",
                    options=["A. 是", "B. 否"],
                    answer="A",
                    explanation="归并可以保持稳定。",
                    source="notes.md",
                )
            ],
            user_answers=[],
            status="active",
        )

    async def asyncTearDown(self) -> None:
        self.store_patch.stop()
        adaptive_router._sessions.clear()
        adaptive_router.sessions.clear()
        adaptive_router._submit_locks.clear()
        self.tempdir.cleanup()

    async def test_cancelled_lock_wait_releases_clean_receipt(self) -> None:
        lock = asyncio.Lock()
        await lock.acquire()
        adaptive_router._submit_locks[self.session_id] = lock
        claimed = asyncio.Event()
        original_begin = self.store.begin

        async def begin(*args, **kwargs):
            decision = await original_begin(*args, **kwargs)
            claimed.set()
            return decision

        request = adaptive_router.AdaptiveSubmitRequest(
            adaptive_session_id=self.session_id,
            turn=1,
            answers=["A. 是"],
        )
        key = "adaptive-submit-cancelled-1"
        try:
            with patch.object(self.store, "begin", side_effect=begin):
                task = asyncio.create_task(
                    adaptive_router.adaptive_submit(
                        request,
                        idempotency_key=key,
                    )
                )
                await asyncio.wait_for(claimed.wait(), timeout=2)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
        finally:
            lock.release()

        retry = await self.store.begin(
            key,
            "agent.adaptive.submit",
            request.model_dump(mode="json"),
        )
        self.assertFalse(retry.replayed)
        await self.store.abort(key)


if __name__ == "__main__":
    unittest.main()
