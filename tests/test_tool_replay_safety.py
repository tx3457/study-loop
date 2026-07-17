"""Replay-safety contracts for state-changing tools."""

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import services.tools  # 注册内置工具，确保本文件可独立运行
from services.tool_loop import run_tool_round
from services.tool_registry import (
    EffectMode,
    SideEffectAmbiguousError,
    Tool,
    ToolMetadata,
    tool_registry,
)


class TestToolReplaySafety(unittest.IsolatedAsyncioTestCase):
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
