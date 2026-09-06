"""Trusted tool-result lineage and duplicate-call safety for the tool loop."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.tool_loop import run_tool_round
from services.tool_registry import (
    EffectMode,
    Tool,
    ToolArgumentBinding,
    ToolMetadata,
    ToolPolicyViolation,
    tool_registry,
)


def _call(call_id: str, name: str, arguments: str = "{}"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _response(*calls):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content=None,
        tool_calls=list(calls),
    ))])


def _result(call_id: str, name: str, content: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": call_id, "content": content},
    ]


class TestToolArgumentBindings(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.registered: list[Tool] = []

    def tearDown(self):
        for tool in reversed(self.registered):
            tool_registry.unregister(tool.name, expected_tool=tool)

    def register(self, name: str, *bindings: ToolArgumentBinding) -> Tool:
        async def handler(**_kwargs):
            return "{}"

        tool = Tool(
            name=name,
            description="test tool",
            parameters_schema={"type": "object", "properties": {}},
            handler=handler,
            metadata=ToolMetadata(
                effect_mode=EffectMode.READ_ONLY,
                argument_bindings=tuple(bindings),
            ),
        )
        tool_registry.register(tool)
        self.registered.append(tool)
        return tool

    async def test_authoritative_bindings_precede_guard_and_first_conflict_wins(self):
        target = self.register(
            "lineage_target",
            ToolArgumentBinding("grade_source", "$", "last_result"),
            ToolArgumentBinding("profile_source", "mastery.sql", "last_result.score"),
        )
        messages = (
            _result("grade", "grade_source", '{"score":0.9}')
            + _result("profile", "profile_source", '{"mastery":{"sql":0.1}}')
        )
        guard = MagicMock(return_value=None)
        dispatch = AsyncMock(return_value="{}")

        with patch(
            "services.tool_loop.llm_chat",
            AsyncMock(return_value=_response(_call(
                "target",
                target.name,
                '{"last_result":{"score":0.0,"model":"discard"}}',
            ))),
        ), patch("services.tool_loop.dispatch_tool", dispatch):
            result = await run_tool_round(
                messages,
                tools=[],
                business_tool_guard=guard,
            )

        expected = {"last_result": {"score": 0.9}}
        guard.assert_has_calls([
            call(
                target.name,
                {"last_result": {"score": 0.0, "model": "discard"}},
            ),
            call(target.name, expected),
        ])
        self.assertEqual(guard.call_count, 2)
        self.assertEqual(dispatch.await_args.args[1], expected)
        self.assertEqual(result.outcomes[0].arguments, expected)

    async def test_raw_scope_violation_cannot_be_laundered_by_binding(self):
        target = self.register(
            "owner_binding_target",
            ToolArgumentBinding("trusted_source", "user_id", "user_id"),
        )
        messages = _result(
            "trusted",
            "trusted_source",
            '{"user_id":"trusted-user"}',
        )

        def scope_guard(_name, args):
            if args.get("user_id") != "trusted-user":
                return "owner_mismatch"
            return None

        guard = MagicMock(side_effect=scope_guard)
        dispatch = AsyncMock(return_value="{}")
        with patch(
            "services.tool_loop.llm_chat",
            AsyncMock(return_value=_response(_call(
                "target",
                target.name,
                '{"user_id":"other-user"}',
            ))),
        ), patch("services.tool_loop.dispatch_tool", dispatch):
            result = await run_tool_round(
                messages,
                tools=[],
                business_tool_guard=guard,
            )

        guard.assert_called_once_with(target.name, {"user_id": "other-user"})
        dispatch.assert_not_awaited()
        self.assertEqual(result.outcomes[0].arguments, {"user_id": "other-user"})
        self.assertEqual(result.outcomes[0].blocked_reason, "owner_mismatch")

    async def test_current_batch_results_cannot_feed_later_parallel_call(self):
        source = self.register("parallel_source")
        target = self.register(
            "parallel_target",
            ToolArgumentBinding(source.name, "$", "payload"),
        )
        dispatch = AsyncMock(side_effect=['{"value":1}', "{}"])

        with patch(
            "services.tool_loop.llm_chat",
            AsyncMock(return_value=_response(
                _call("source", source.name),
                _call("target", target.name),
            )),
        ), patch("services.tool_loop.dispatch_tool", dispatch):
            await run_tool_round([], tools=[])

        self.assertEqual(dispatch.await_args_list[1].args[1], {})

    async def test_latest_error_or_non_json_clears_stale_success(self):
        target = self.register(
            "stale_target",
            ToolArgumentBinding("stale_source", "$", "payload"),
        )
        for latest in ('{"error":"failed"}', "not-json"):
            messages = (
                _result("old", "stale_source", '{"stale":true}')
                + _result("latest", "stale_source", latest)
            )
            dispatch = AsyncMock(return_value="{}")
            with self.subTest(latest=latest), patch(
                "services.tool_loop.llm_chat",
                AsyncMock(return_value=_response(_call("target", target.name))),
            ), patch("services.tool_loop.dispatch_tool", dispatch):
                await run_tool_round(messages, tools=[])

            self.assertEqual(dispatch.await_args.args[1], {})

    async def test_duplicate_historical_call_id_invalidates_affected_result(self):
        target = self.register(
            "duplicate_history_target",
            ToolArgumentBinding("original_source", "$", "payload"),
        )
        messages = _result(
            "duplicate", "original_source", '{"secret":true}'
        ) + [{
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "duplicate",
                "type": "function",
                "function": {"name": "replacement_source", "arguments": "{}"},
            }],
        }]
        dispatch = AsyncMock(return_value="{}")

        with patch(
            "services.tool_loop.llm_chat",
            AsyncMock(return_value=_response(_call("target", target.name))),
        ), patch("services.tool_loop.dispatch_tool", dispatch):
            await run_tool_round(messages, tools=[])

        self.assertEqual(dispatch.await_args.args[1], {})

    async def test_duplicate_current_call_ids_block_before_any_callback_or_mutation(self):
        source = self.register("duplicate_current_source")
        messages = []
        ownership = AsyncMock()
        progress = AsyncMock()
        guard = MagicMock(return_value=None)
        dispatch = AsyncMock(return_value="{}")

        with patch(
            "services.tool_loop.llm_chat",
            AsyncMock(return_value=_response(
                _call("duplicate", source.name),
                _call("duplicate", source.name),
            )),
        ), patch("services.tool_loop.dispatch_tool", dispatch):
            result = await run_tool_round(
                messages,
                tools=[],
                on_before_tool_calls=ownership,
                on_before_tool_dispatch=progress,
                business_tool_guard=guard,
            )

        self.assertEqual(messages, [])
        ownership.assert_not_awaited()
        progress.assert_not_awaited()
        guard.assert_not_called()
        dispatch.assert_not_awaited()
        self.assertEqual(
            [outcome.blocked_reason for outcome in result.outcomes],
            ["duplicate_tool_call_id", "duplicate_tool_call_id"],
        )

    async def test_non_string_current_call_id_blocks_before_all_side_effects(self):
        source = self.register("invalid_current_id_source")
        for invalid_id in (123, ["unhashable"]):
            messages = []
            ownership = AsyncMock()
            progress = AsyncMock()
            guard = MagicMock(return_value=None)
            dispatch = AsyncMock(return_value="{}")
            with self.subTest(invalid_id=invalid_id), patch(
                "services.tool_loop.llm_chat",
                AsyncMock(return_value=_response(_call(invalid_id, source.name))),
            ), patch("services.tool_loop.dispatch_tool", dispatch):
                result = await run_tool_round(
                    messages,
                    tools=[],
                    on_before_tool_calls=ownership,
                    on_before_tool_dispatch=progress,
                    business_tool_guard=guard,
                )

            self.assertEqual(messages, [])
            ownership.assert_not_awaited()
            progress.assert_not_awaited()
            guard.assert_not_called()
            dispatch.assert_not_awaited()
            self.assertEqual(
                result.outcomes[0].blocked_reason,
                "duplicate_tool_call_id",
            )

    async def test_policy_violation_becomes_blocked_tool_result(self):
        target = self.register("policy_target")
        dispatch = AsyncMock(side_effect=ToolPolicyViolation(
            "owner_mismatch", target.name
        ))

        with patch(
            "services.tool_loop.llm_chat",
            AsyncMock(return_value=_response(_call("target", target.name))),
        ), patch("services.tool_loop.dispatch_tool", dispatch):
            result = await run_tool_round([], tools=[])

        self.assertEqual(result.outcomes[0].kind, "blocked")
        self.assertEqual(result.outcomes[0].blocked_reason, "owner_mismatch")
        self.assertIn("owner_mismatch", result.outcomes[0].result)


if __name__ == "__main__":
    unittest.main()
