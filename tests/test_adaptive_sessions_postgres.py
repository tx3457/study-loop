"""Live PostgreSQL contracts for durable Adaptive sessions.

Set TEST_DATABASE_URL to run these tests. The default suite skips them so
local development does not require a PostgreSQL service.
"""

from __future__ import annotations

import asyncio
import os
import unittest
import uuid
from unittest.mock import patch

from models.adaptive import AdaptiveTurn, NextStepDecision
from models.adaptive_session import (
    AdaptiveSessionAggregate,
    AdaptiveTurnArtifact,
)
import services.adaptive_sessions as adaptive_sessions_module
from services.adaptive_sessions import (
    AdaptiveSessionCapacityError,
    AdaptiveSessionStore,
)


TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")


def _aggregate(
    session_id: str,
    *,
    lesson: str = "Compare the midpoint and retain the possible half.",
) -> AdaptiveSessionAggregate:
    return AdaptiveSessionAggregate(
        adaptive_session_id=session_id,
        user_id="postgres-user",
        document_id="notes.md",
        goal="learn binary search",
        status="active",
        current_quiz=None,
        current_artifact=AdaptiveTurnArtifact(
            adaptive_session_id=session_id,
            turn=1,
            turn_type="teach",
            lesson=lesson,
            decision=NextStepDecision(
                action="teach",
                topic="binary search",
                difficulty="medium",
                difficulty_score=0.5,
                question_type="choice",
                count=1,
                reason="Teach before the next quiz.",
            ),
            trajectory=[
                AdaptiveTurn(
                    turn=1,
                    action="teach",
                    topic="binary search",
                    difficulty_score=0.5,
                    reason="Teach before the next quiz.",
                )
            ],
        ),
    )


@unittest.skipUnless(TEST_DATABASE_URL, "TEST_DATABASE_URL is not configured")
class TestPostgresAdaptiveSessionStore(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.now = [1_000.0]
        self.table_name = f"studyloop_adaptive_test_{uuid.uuid4().hex}"
        self.table_patch = patch.object(
            adaptive_sessions_module,
            "_TABLE",
            self.table_name,
        )
        self.table_patch.start()
        self.store = self._new_store()

    def tearDown(self) -> None:
        import psycopg
        from psycopg import sql

        try:
            with psycopg.connect(
                TEST_DATABASE_URL,
                connect_timeout=5,
            ) as connection:
                connection.execute(
                    sql.SQL("DROP TABLE IF EXISTS {}").format(
                        sql.Identifier(self.table_name)
                    )
                )
        finally:
            self.table_patch.stop()

    def _new_store(self, *, max_count: int = 100) -> AdaptiveSessionStore:
        return AdaptiveSessionStore(
            database_url=TEST_DATABASE_URL,
            ttl_seconds=60,
            operation_lease_seconds=10,
            max_count=max_count,
            postgres_connect_timeout_seconds=5,
            postgres_lock_timeout_ms=5_000,
            postgres_statement_timeout_ms=15_000,
            clock=lambda: self.now[0],
        )

    async def test_two_connections_reopen_take_over_and_fence(self) -> None:
        session_id = f"adaptive-{uuid.uuid4().hex}"
        await self.store.create(_aggregate(session_id))
        peer = self._new_store()

        restored = await peer.inspect(session_id)
        self.assertIsNotNone(restored)
        self.assertEqual(restored.aggregate.adaptive_session_id, session_id)

        owner = await self.store.claim(session_id, "submit")
        blocked = await peer.claim(session_id, "submit")
        self.assertTrue(owner.claimed)
        self.assertFalse(blocked.claimed)
        self.assertEqual(blocked.reason, "in_progress")

        self.now[0] += 11
        takeover = await peer.claim(session_id, "submit")
        self.assertTrue(takeover.claimed)
        self.assertNotEqual(owner.token, takeover.token)
        self.assertIsNone(
            await self.store.checkpoint(
                session_id,
                owner.token,
                _aggregate(session_id, lesson="Stale owner."),
                expected_revision=1,
            )
        )

        saved = await peer.complete(
            session_id,
            takeover.token,
            _aggregate(session_id, lesson="Winning owner."),
            expected_revision=1,
        )
        self.assertEqual(saved.revision, 2)
        self.assertFalse(saved.busy)
        self.assertEqual(
            saved.aggregate.current_artifact.lesson,
            "Winning owner.",
        )

    async def test_concurrent_same_start_key_creates_one_session(self) -> None:
        first = self._new_store()
        second = self._new_store()
        first_id = f"adaptive-{uuid.uuid4().hex}"
        second_id = f"adaptive-{uuid.uuid4().hex}"
        start_key = f"adaptive-start-{uuid.uuid4().hex}"
        request = {
            "user_id": "postgres-user",
            "document_id": "notes.md",
            "goal": "learn binary search",
        }

        decisions = await asyncio.gather(
            first.create(
                _aggregate(first_id),
                start_key=start_key,
                start_request=request,
            ),
            second.create(
                _aggregate(second_id),
                start_key=start_key,
                start_request=request,
            ),
        )

        self.assertEqual(sum(decision.created for decision in decisions), 1)
        logical_ids = {
            decision.record.aggregate.adaptive_session_id
            for decision in decisions
        }
        self.assertEqual(len(logical_ids), 1)
        logical_id = logical_ids.pop()
        self.assertIn(logical_id, {first_id, second_id})

    async def test_concurrent_creates_preserve_capacity_limit(self) -> None:
        first = self._new_store(max_count=1)
        second = self._new_store(max_count=1)
        first_id = f"adaptive-{uuid.uuid4().hex}"
        second_id = f"adaptive-{uuid.uuid4().hex}"

        outcomes = await asyncio.gather(
            first.create(_aggregate(first_id)),
            second.create(_aggregate(second_id)),
            return_exceptions=True,
        )

        successes = [
            outcome
            for outcome in outcomes
            if not isinstance(outcome, BaseException)
        ]
        failures = [
            outcome
            for outcome in outcomes
            if isinstance(outcome, BaseException)
        ]
        self.assertEqual(len(successes), 1)
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], AdaptiveSessionCapacityError)


if __name__ == "__main__":
    unittest.main(verbosity=2)
