"""Replay-safety contracts for state-changing tools."""

import asyncio
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import services.tools  # noqa: F401  # 注册内置工具，确保本文件可独立运行
from services.tool_loop import run_tool_round
from services.tool_registry import (
    EffectMode,
    SideEffectAmbiguousError,
    Tool,
    ToolArgumentBinding,
    ToolCallRecord,
    ToolMetadata,
    ToolPolicyViolation,
    tool_registry,
)


class TestToolReplaySafety(unittest.IsolatedAsyncioTestCase):
    async def test_effect_attempt_route_classification_survives_audit_eviction(self):
        profile = tool_registry.get("get_user_profile")
        search = tool_registry.get("search_document")
        self.assertIsNotNone(profile)
        self.assertIsNotNone(search)
        self.assertEqual(profile.metadata.effect_mode, EffectMode.UNKNOWN)
        profile_handler = AsyncMock(return_value='{"user_id":"u"}')
        search_handler = AsyncMock(return_value='{"chunks":[]}')
        effect_run = "unknown-effect-route"
        read_run = "read-only-route"
        original_audit = list(tool_registry._audit_log)

        try:
            with patch.object(profile, "handler", new=profile_handler):
                await tool_registry.invoke(
                    profile.name,
                    {"user_id": "u"},
                    run_id=effect_run,
                    user_id="u",
                    on_before_handler=AsyncMock(),
                )
            with patch.object(search, "handler", new=search_handler):
                await tool_registry.invoke(
                    search.name,
                    {"document_id": "d", "query": "q"},
                    run_id=read_run,
                    on_before_handler=AsyncMock(),
                )

            for index in range(tool_registry._AUDIT_MAX + 1):
                tool_registry._record(ToolCallRecord(
                    tool_call_id=f"noise-{index}",
                    tool_name="noise",
                    arguments={},
                    status="ok",
                    duration_ms=0.0,
                    timestamp="2026-01-01T00:00:00",
                    effect_mode=EffectMode.READ_ONLY.value,
                ))

            self.assertFalse(any(
                record.run_id == effect_run
                for record in tool_registry._audit_log
            ))
            self.assertTrue(tool_registry.has_effect_attempt(effect_run))
            self.assertFalse(tool_registry.has_effect_attempt(read_run))
            tool_registry.clear_run_policy_state(effect_run)
            self.assertFalse(tool_registry.has_effect_attempt(effect_run))
        finally:
            tool_registry.clear_run_policy_state(effect_run)
            tool_registry.clear_run_policy_state(read_run)
            tool_registry._audit_log[:] = original_audit

    def test_security_contract_payload_serializes_all_enforced_metadata(self):
        metadata = ToolMetadata(
            effect_mode=EffectMode.NON_IDEMPOTENT,
            owner_argument="user_id",
            dedupe_within_run=True,
            dedupe_argument_paths=("user_id", "grade_result.score"),
            argument_bindings=(
                ToolArgumentBinding(
                    source_tool="grade_answer",
                    source_path="$",
                    target_argument="grade_result",
                ),
            ),
        )

        payload = metadata.security_contract_payload()

        self.assertEqual(payload, {
            "version": 1,
            "effect_mode": "non_idempotent",
            "owner_argument": "user_id",
            "dedupe_within_run": True,
            "dedupe_argument_paths": ["user_id", "grade_result.score"],
            "dedupe_normalizer_id": None,
            "argument_bindings": [{
                "source_tool": "grade_answer",
                "source_path": "$",
                "target_argument": "grade_result",
            }],
        })
        self.assertEqual(
            json.loads(json.dumps(payload, sort_keys=True)), payload
        )

    def test_named_normalizer_stabilizes_profile_event_reservation(self):
        def normalize(arguments):
            result = arguments["grade_result"]
            score = result.get("score")
            if isinstance(score, (int, float)):
                normalized_score = float(score)
            else:
                normalized_score = 1.0 if result.get("is_correct") else 0.0
            gaps = []
            if result.get("knowledge_gap"):
                gaps.append(str(result["knowledge_gap"]))
            for gap in result.get("knowledge_gaps", []) or []:
                if gap:
                    gaps.append(str(gap))
            return {
                "user_id": arguments["user_id"],
                "document_id": arguments["document_id"],
                "question": str(result.get("question") or ""),
                "user_answer": str(result.get("user_answer") or ""),
                "correct_answer": str(result.get("correct_answer") or ""),
                "score": normalized_score,
                "knowledge_gaps": gaps,
            }

        async def handler(**_arguments):
            return '{"status":"updated"}'

        tool = Tool(
            name="normalized_profile_event_write",
            description="Synthetic normalized profile event.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "document_id": {"type": "string"},
                    "grade_result": {"type": "object"},
                },
                "required": ["user_id", "document_id", "grade_result"],
            },
            handler=handler,
            metadata=ToolMetadata(
                max_retries=0,
                effect_mode=EffectMode.NON_IDEMPOTENT,
                dedupe_within_run=True,
                dedupe_normalizer=normalize,
                dedupe_normalizer_id="profile_event_test_v1",
            ),
        )
        tool_registry.register(tool)
        try:
            base_result = {
                "question": "2+3",
                "user_answer": "5",
                "correct_answer": "5",
            }
            variants = [
                {"score": 1, "is_correct": True},
                {"score": 1.0, "is_correct": False, "knowledge_gap": None},
                {"score": 1, "knowledge_gap": [], "knowledge_gaps": []},
                {"score": 1, "nonce": "ignored"},
            ]
            reservations = {
                tool_registry.semantic_reservation(tool, {
                    "user_id": "u",
                    "document_id": "d",
                    "grade_result": {**base_result, **variant},
                })
                for variant in variants
            }
            self.assertEqual(len(reservations), 1)

            different_question = tool_registry.semantic_reservation(tool, {
                "user_id": "u",
                "document_id": "d",
                "grade_result": {
                    **base_result,
                    "question": "3+3",
                    "score": 1,
                },
            })
            self.assertNotIn(different_question, reservations)
            self.assertEqual(
                tool.metadata.security_contract_payload()["dedupe_normalizer_id"],
                "profile_event_test_v1",
            )
        finally:
            tool_registry.unregister(tool.name, expected_tool=tool)

    def test_dedupe_normalizer_requires_stable_identifier(self):
        async def handler(**_arguments):
            return "{}"

        with self.assertRaises(ValueError):
            Tool(
                name="unnamed_normalizer",
                description="Invalid unnamed normalizer.",
                parameters_schema={"type": "object", "properties": {}},
                handler=handler,
                metadata=ToolMetadata(
                    effect_mode=EffectMode.NON_IDEMPOTENT,
                    dedupe_within_run=True,
                    dedupe_normalizer=lambda arguments: arguments,
                ),
            )

    async def test_owner_policy_fails_closed_before_progress_boundary(self):
        calls = []

        async def handler(user_id):
            calls.append(user_id)
            return '{"status":"unexpected"}'

        tool = Tool(
            name="owner_bound_test_write",
            description="Synthetic owner-bound write.",
            parameters_schema={
                "type": "object",
                "properties": {"user_id": {"type": "string"}},
                "required": ["user_id"],
                "additionalProperties": False,
            },
            handler=handler,
            metadata=ToolMetadata(
                max_retries=0,
                effect_mode=EffectMode.NON_IDEMPOTENT,
                owner_argument="user_id",
                argument_bindings=(
                    ToolArgumentBinding(
                        source_tool="get_user_profile",
                        source_path="$",
                        target_argument="profile",
                    ),
                ),
            ),
        )
        original_audit = list(tool_registry._audit_log)
        tool_registry.register(tool)

        try:
            for trusted_user, expected_reason in [
                (None, "owner_context_missing"),
                ("trusted", "owner_mismatch"),
            ]:
                progress = AsyncMock()
                with self.subTest(expected_reason=expected_reason):
                    with self.assertRaises(ToolPolicyViolation) as raised:
                        await tool_registry.invoke(
                            tool.name,
                            {"user_id": "untrusted"},
                            run_id=f"owner-{expected_reason}",
                            user_id=trusted_user,
                            on_before_handler=progress,
                        )
                    self.assertEqual(raised.exception.reason, expected_reason)
                    progress.assert_not_awaited()
            self.assertEqual(calls, [])
            self.assertTrue(all(
                record.status == "blocked"
                for record in tool_registry._audit_log[len(original_audit):]
            ))
        finally:
            tool_registry.unregister(tool.name, expected_tool=tool)
            tool_registry._audit_log[:] = original_audit

    async def test_semantic_dedupe_ignores_nonce_but_allows_changed_effect(self):
        calls = []

        async def handler(user_id, grade_result, nonce):
            calls.append((user_id, grade_result, nonce))
            return '{"status":"updated"}'

        tool = Tool(
            name="semantic_test_write",
            description="Synthetic semantically deduplicated write.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "grade_result": {"type": "object"},
                    "nonce": {"type": "string"},
                },
                "required": ["user_id", "grade_result", "nonce"],
                "additionalProperties": False,
            },
            handler=handler,
            metadata=ToolMetadata(
                max_retries=0,
                effect_mode=EffectMode.NON_IDEMPOTENT,
                owner_argument="user_id",
                dedupe_within_run=True,
                dedupe_argument_paths=("user_id", "grade_result.score"),
            ),
        )
        run_id = "semantic-dedupe"
        original_audit = list(tool_registry._audit_log)
        tool_registry.register(tool)

        try:
            await tool_registry.invoke(
                tool.name,
                {"user_id": "u", "grade_result": {"score": 0.5}, "nonce": "a"},
                run_id=run_id,
                user_id="u",
            )
            # Policy correctness must not depend on the bounded audit LRU.
            tool_registry._audit_log[:] = original_audit
            with self.assertRaises(ToolPolicyViolation):
                await tool_registry.invoke(
                    tool.name,
                    {"user_id": "u", "grade_result": {"score": 0.5}, "nonce": "b"},
                    run_id=run_id,
                    user_id="u",
                )
            await tool_registry.invoke(
                tool.name,
                {"user_id": "u", "grade_result": {"score": 0.8}, "nonce": "c"},
                run_id=run_id,
                user_id="u",
            )
            self.assertEqual([call[1]["score"] for call in calls], [0.5, 0.8])
        finally:
            tool_registry.clear_run_policy_state(run_id)
            tool_registry.unregister(tool.name, expected_tool=tool)
            tool_registry._audit_log[:] = original_audit

    async def test_policy_snapshot_round_trip_contains_digest_not_arguments(self):
        calls = 0

        async def handler(secret):
            nonlocal calls
            calls += 1
            return '{"status":"updated"}'

        tool = Tool(
            name="digest_snapshot_test_write",
            description="Synthetic digest-only snapshot write.",
            parameters_schema={
                "type": "object",
                "properties": {"secret": {"type": "string"}},
                "required": ["secret"],
                "additionalProperties": False,
            },
            handler=handler,
            metadata=ToolMetadata(
                max_retries=0,
                effect_mode=EffectMode.NON_IDEMPOTENT,
                dedupe_within_run=True,
            ),
        )
        source_run = "digest-source"
        restored_run = "digest-restored"
        original_audit = list(tool_registry._audit_log)
        tool_registry.register(tool)

        try:
            await tool_registry.invoke(
                tool.name, {"secret": "private-answer"}, run_id=source_run
            )
            snapshot = tool_registry.snapshot_run_policy_state(source_run)
            self.assertNotIn("private-answer", json.dumps(snapshot))
            self.assertEqual(len(snapshot), 1)
            self.assertRegex(snapshot[0][1], r"^[0-9a-f]{64}$")

            payload = json.loads(json.dumps(snapshot))
            tool_registry.restore_run_policy_state(restored_run, payload)
            self.assertFalse(tool_registry.has_effect_attempt(restored_run))
            with self.assertRaises(ToolPolicyViolation):
                await tool_registry.invoke(
                    tool.name,
                    {"secret": "private-answer"},
                    run_id=restored_run,
                )
            self.assertFalse(tool_registry.has_effect_attempt(restored_run))
            self.assertEqual(calls, 1)
        finally:
            tool_registry.clear_run_policy_state(source_run)
            tool_registry.clear_run_policy_state(restored_run)
            tool_registry.unregister(tool.name, expected_tool=tool)
            tool_registry._audit_log[:] = original_audit

    async def test_policy_restore_rejects_malformed_or_oversized_payload_atomically(self):
        async def handler(value):
            return '{"status":"updated"}'

        tool = Tool(
            name="validated_digest_restore_write",
            description="Synthetic strict restore validation write.",
            parameters_schema={
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            handler=handler,
            metadata=ToolMetadata(
                max_retries=0,
                effect_mode=EffectMode.NON_IDEMPOTENT,
                dedupe_within_run=True,
            ),
        )
        run_id = "validated-digest-restore"
        original_audit = list(tool_registry._audit_log)
        tool_registry.register(tool)

        try:
            await tool_registry.invoke(tool.name, {"value": 1}, run_id=run_id)
            baseline = tool_registry.snapshot_run_policy_state(run_id)
            valid_digest = "a" * 64
            invalid_payloads = [
                "not-a-sequence",
                [[tool.name]],
                [[tool.name, valid_digest, "extra"]],
                [[1, valid_digest]],
                [["unknown_tool", valid_digest]],
                [[tool.name, "private-answer"]],
                [[tool.name, "A" * 64]],
                [[tool.name, valid_digest]] * (
                    tool_registry._POLICY_MAX_RESERVATIONS + 1
                ),
            ]
            for payload in invalid_payloads:
                with self.subTest(payload_type=type(payload).__name__):
                    with self.assertRaises(ValueError):
                        tool_registry.restore_run_policy_state(run_id, payload)
                    self.assertEqual(
                        tool_registry.snapshot_run_policy_state(run_id), baseline
                    )
        finally:
            tool_registry.clear_run_policy_state(run_id)
            tool_registry.unregister(tool.name, expected_tool=tool)
            tool_registry._audit_log[:] = original_audit

    async def test_failed_progress_boundary_releases_inflight_reservation(self):
        calls = 0

        async def handler(value):
            nonlocal calls
            calls += 1
            return '{"status":"updated"}'

        tool = Tool(
            name="progress_release_test_write",
            description="Synthetic write for progress failure handling.",
            parameters_schema={
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            handler=handler,
            metadata=ToolMetadata(
                max_retries=0,
                effect_mode=EffectMode.NON_IDEMPOTENT,
                dedupe_within_run=True,
            ),
        )
        run_id = "progress-release"
        original_audit = list(tool_registry._audit_log)
        tool_registry.register(tool)

        try:
            with self.assertRaisesRegex(RuntimeError, "progress unavailable"):
                await tool_registry.invoke(
                    tool.name,
                    {"value": 1},
                    run_id=run_id,
                    on_before_handler=AsyncMock(
                        side_effect=RuntimeError("progress unavailable")
                    ),
                )
            self.assertFalse(tool_registry.has_effect_attempt(run_id))
            await tool_registry.invoke(tool.name, {"value": 1}, run_id=run_id)
            self.assertEqual(calls, 1)
            self.assertTrue(tool_registry.has_effect_attempt(run_id))
        finally:
            tool_registry.clear_run_policy_state(run_id)
            tool_registry.unregister(tool.name, expected_tool=tool)
            tool_registry._audit_log[:] = original_audit

    async def test_effect_marker_blocks_concurrent_duplicate_before_progress(self):
        entered = asyncio.Event()
        release = asyncio.Event()
        handler_calls = 0

        async def handler(value):
            nonlocal handler_calls
            handler_calls += 1
            return '{"status":"updated"}'

        async def first_progress():
            entered.set()
            await release.wait()

        second_progress = AsyncMock()
        tool = Tool(
            name="concurrent_effect_marker_write",
            description="Synthetic concurrent effect marker write.",
            parameters_schema={
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            handler=handler,
            metadata=ToolMetadata(
                max_retries=0,
                effect_mode=EffectMode.NON_IDEMPOTENT,
                dedupe_within_run=True,
            ),
        )
        run_id = "concurrent-effect-marker"
        original_audit = list(tool_registry._audit_log)
        tool_registry.register(tool)

        try:
            first = asyncio.create_task(tool_registry.invoke(
                tool.name,
                {"value": 1},
                run_id=run_id,
                on_before_handler=first_progress,
            ))
            await entered.wait()
            with self.assertRaises(ToolPolicyViolation):
                await tool_registry.invoke(
                    tool.name,
                    {"value": 1},
                    run_id=run_id,
                    on_before_handler=second_progress,
                )
            second_progress.assert_not_awaited()
            release.set()
            await first
            self.assertEqual(handler_calls, 1)
        finally:
            release.set()
            tool_registry.clear_run_policy_state(run_id)
            tool_registry.unregister(tool.name, expected_tool=tool)
            tool_registry._audit_log[:] = original_audit

    async def test_round_rejects_tool_replaced_while_waiting_for_model(self):
        original = tool_registry.get("search_document")
        self.assertIsNotNone(original)
        original_audit = list(tool_registry._audit_log)
        writes = []

        async def unsafe_handler(**kwargs):
            writes.append(kwargs)
            return '{"status":"written"}'

        replacement = Tool(
            name="search_document",
            description="Replacement with a different effect contract.",
            parameters_schema=original.parameters_schema,
            handler=unsafe_handler,
            metadata=ToolMetadata(
                timeout_sec=1.0,
                max_retries=0,
                effect_mode=EffectMode.NON_IDEMPOTENT,
            ),
        )
        tool_call = SimpleNamespace(
            id="search-1",
            function=SimpleNamespace(
                name="search_document",
                arguments='{"document_id":"d","query":"q"}',
            ),
        )

        async def replace_then_respond(**kwargs):
            tool_registry.register(replacement)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content=None,
                tool_calls=[tool_call],
            ))])

        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=replace_then_respond)

        try:
            result = await run_tool_round(
                [],
                tools=[original.to_openai_schema()],
                client=client,
                business_tool_allowlist={"search_document"},
            )

            self.assertEqual(writes, [])
            self.assertEqual(result.outcomes[0].kind, "blocked")
            self.assertEqual(
                result.outcomes[0].blocked_reason,
                "replay_safety_binding_changed",
            )
        finally:
            tool_registry.register(original)
            tool_registry._audit_log[:] = original_audit

    async def test_round_allowlist_blocks_nonreplayable_tool(self):
        tool_call = SimpleNamespace(
            id="write-1",
            function=SimpleNamespace(
                name="update_learning_profile",
                arguments=(
                    '{"user_id":"u","document_id":"d",'
                    '"grade_result":{"score":1.0}}'
                ),
            ),
        )
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=None,
                tool_calls=[tool_call],
            ))]
        ))

        with patch("services.tool_loop.dispatch_tool", AsyncMock()) as dispatch:
            result = await run_tool_round(
                [],
                tools=[],
                client=client,
                business_tool_allowlist={"search_document"},
            )

        dispatch.assert_not_awaited()
        self.assertEqual(result.outcomes[0].kind, "blocked")
        self.assertEqual(result.outcomes[0].blocked_reason, "not_in_whitelist")

    async def test_ownership_barrier_runs_before_message_mutation_or_dispatch(self):
        tool_call = SimpleNamespace(
            id="search-barrier",
            function=SimpleNamespace(
                name="search_document",
                arguments='{"document_id":"d","query":"q"}',
            ),
        )
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=None,
                tool_calls=[tool_call],
            ))]
        ))
        messages = []

        async def fail_barrier():
            self.assertEqual(messages, [])
            raise RuntimeError("durable marker unavailable")

        with patch("services.tool_loop.dispatch_tool", AsyncMock()) as dispatch:
            with self.assertRaisesRegex(RuntimeError, "durable marker unavailable"):
                await run_tool_round(
                    messages,
                    tools=[],
                    client=client,
                    on_before_tool_calls=fail_barrier,
                )

        self.assertEqual(messages, [])
        dispatch.assert_not_awaited()

    async def test_invalid_arguments_never_cross_handler_progress_boundary(self):
        tool = tool_registry.get("update_learning_profile")
        self.assertIsNotNone(tool)
        handler = AsyncMock(return_value='{"status":"unexpected"}')
        ownership_barrier = AsyncMock()
        progress_barrier = AsyncMock()
        tool_call = SimpleNamespace(
            id="invalid-write",
            function=SimpleNamespace(
                name="update_learning_profile",
                # grade_result is required by both schema and handler.
                arguments='{"user_id":"u","document_id":"d"}',
            ),
        )
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=None,
                tool_calls=[tool_call],
            ))]
        ))

        with patch.object(tool, "handler", new=handler):
            result = await run_tool_round(
                [],
                tools=[tool.to_openai_schema()],
                client=client,
                on_before_tool_calls=ownership_barrier,
                on_before_tool_dispatch=progress_barrier,
            )

        ownership_barrier.assert_awaited_once()
        progress_barrier.assert_not_awaited()
        handler.assert_not_awaited()
        self.assertEqual(result.outcomes[0].kind, "dispatched")
        self.assertIn("invalid_tool_arguments", result.outcomes[0].result)

    async def test_valid_arguments_cross_progress_boundary_before_handler(self):
        tool = tool_registry.get("search_document")
        self.assertIsNotNone(tool)
        order = []

        async def progress_barrier():
            order.append("progress")

        async def handler(**_kwargs):
            order.append("handler")
            return '{"chunks":[]}'

        tool_call = SimpleNamespace(
            id="valid-read",
            function=SimpleNamespace(
                name="search_document",
                arguments='{"document_id":"d","query":"q"}',
            ),
        )
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=None,
                tool_calls=[tool_call],
            ))]
        ))

        with patch.object(tool, "handler", new=handler):
            await run_tool_round(
                [],
                tools=[tool.to_openai_schema()],
                client=client,
                on_before_tool_dispatch=progress_barrier,
            )

        self.assertEqual(order, ["progress", "handler"])

    def test_profile_read_with_legacy_migration_is_not_replay_safe(self):
        tool = tool_registry.get("get_user_profile")
        self.assertIsNotNone(tool)
        self.assertEqual(tool.metadata.effect_mode, EffectMode.UNKNOWN)
        self.assertEqual(tool.metadata.max_retries, 0)

    async def test_profile_update_is_not_retried_after_ambiguous_timeout(self):
        tool = tool_registry.get("update_learning_profile")
        self.assertIsNotNone(tool)
        handler = AsyncMock(side_effect=[
            asyncio.TimeoutError,
            '{"status":"duplicated"}',
        ])
        original_audit = list(tool_registry._audit_log)

        try:
            with patch.object(tool, "handler", new=handler), patch.object(
                tool.metadata, "base_delay", 0
            ), patch("services.retry.random.uniform", return_value=0):
                with self.assertRaises(SideEffectAmbiguousError):
                    await tool_registry.invoke(
                        "update_learning_profile",
                        {
                            "user_id": "u",
                            "document_id": "d",
                            "grade_result": {"score": 1.0},
                        },
                        user_id="u",
                    )

            self.assertEqual(handler.await_count, 1)
            records = tool_registry._audit_log[len(original_audit):]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].status, "ambiguous")
            self.assertEqual(
                records[0].effect_mode,
                EffectMode.NON_IDEMPOTENT.value,
            )
        finally:
            tool_registry._audit_log[:] = original_audit

    async def test_invalid_profile_arguments_fail_before_effect_is_started(self):
        tool = tool_registry.get("update_learning_profile")
        self.assertIsNotNone(tool)
        handler = AsyncMock(return_value='{"status":"unexpected"}')
        original_audit = list(tool_registry._audit_log)

        try:
            with patch.object(tool, "handler", new=handler):
                result = await tool_registry.invoke(
                    "update_learning_profile",
                    {"user_id": "u", "document_id": "d"},
                    run_id="invalid-profile-args",
                )

            self.assertIn("参数无效", result)
            handler.assert_not_awaited()
            records = tool_registry._audit_log[len(original_audit):]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].status, "error")
            self.assertEqual(records[0].retry_attempts, 0)
            self.assertFalse(
                tool_registry.has_effect_attempt("invalid-profile-args")
            )
        finally:
            tool_registry._audit_log[:] = original_audit

    async def test_closed_schema_rejects_unknown_args_for_kwargs_handler(self):
        calls = []

        async def kwargs_handler(**kwargs):
            calls.append(kwargs)
            return '{"status":"unexpected"}'

        tool = Tool(
            name="closed_schema_kwargs_tool",
            description="MCP-style handler with a closed input schema",
            parameters_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=kwargs_handler,
            metadata=ToolMetadata(
                max_retries=0,
                effect_mode=EffectMode.UNKNOWN,
            ),
        )
        original_audit = list(tool_registry._audit_log)
        tool_registry.register(tool)

        try:
            result = await tool_registry.invoke(
                tool.name,
                {"query": "q", "unexpected": "blocked"},
                run_id="closed-schema-invalid-args",
            )

            self.assertIn("参数无效", result)
            self.assertEqual(calls, [])
            self.assertFalse(
                tool_registry.has_effect_attempt("closed-schema-invalid-args")
            )
        finally:
            tool_registry.unregister(tool.name, expected_tool=tool)
            tool_registry._audit_log[:] = original_audit

    async def test_cancelled_profile_update_is_never_audited_as_success(self):
        tool = tool_registry.get("update_learning_profile")
        self.assertIsNotNone(tool)
        handler = AsyncMock(side_effect=asyncio.CancelledError)
        original_audit = list(tool_registry._audit_log)

        try:
            with patch.object(tool, "handler", new=handler):
                with self.assertRaises(asyncio.CancelledError):
                    await tool_registry.invoke(
                        "update_learning_profile",
                        {
                            "user_id": "u",
                            "document_id": "d",
                            "grade_result": {"score": 1.0},
                        },
                        user_id="u",
                    )

            records = tool_registry._audit_log[len(original_audit):]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].status, "ambiguous")
            self.assertEqual(handler.await_count, 1)
        finally:
            tool_registry._audit_log[:] = original_audit

    async def test_read_only_tool_keeps_transient_retry_behavior(self):
        tool = tool_registry.get("search_document")
        self.assertIsNotNone(tool)
        handler = AsyncMock(side_effect=[
            asyncio.TimeoutError,
            '{"chunks":[]}',
        ])
        original_audit = list(tool_registry._audit_log)

        try:
            with patch.object(tool, "handler", new=handler), patch.object(
                tool.metadata, "base_delay", 0
            ), patch("services.retry.random.uniform", return_value=0):
                result = await tool_registry.invoke(
                    "search_document",
                    {"document_id": "d", "query": "q"},
                )

            self.assertEqual(result, '{"chunks":[]}')
            self.assertEqual(handler.await_count, 2)
            records = tool_registry._audit_log[len(original_audit):]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].status, "ok")
            self.assertEqual(records[0].effect_mode, EffectMode.READ_ONLY.value)
        finally:
            tool_registry._audit_log[:] = original_audit


if __name__ == "__main__":
    unittest.main(verbosity=2)
