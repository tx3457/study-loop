"""Receipt-dependent tool exposure and dispatch policy for ``/chat/tools``."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import routers.chat as chat
import services.tool_registry as registry_module
from models.chat import ToolChatRequest
from services.idempotency import IdempotencyConflictError, IdempotencyStore
from services.tool_registry import (
    EffectMode,
    SideEffectAmbiguousError,
    Tool,
    ToolMetadata,
    tool_registry,
)


def _tool_call(call_id: str, name: str, arguments: str):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _assistant_message(*, content=None, tool_calls=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content=content,
            tool_calls=tool_calls,
        ))]
    )


def _mock_client(responses):
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=responses)
    return client


def _visible_tool_names(client) -> set[str]:
    definitions = client.chat.completions.create.await_args_list[0].kwargs["tools"]
    return {definition["function"]["name"] for definition in definitions}


class TestChatToolReceiptPolicy(unittest.IsolatedAsyncioTestCase):
    async def test_without_receipt_only_read_only_tools_are_visible_and_dispatchable(self):
        idempotent_calls = []

        async def idempotent_handler():
            idempotent_calls.append(True)
            return '{"status":"unexpected"}'

        idempotent_tool = Tool(
            name="test_chat_idempotent_tool",
            description="Synthetic same-argument idempotent write.",
            parameters_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=idempotent_handler,
            metadata=ToolMetadata(
                timeout_sec=1.0,
                max_retries=0,
                effect_mode=EffectMode.IDEMPOTENT,
            ),
        )
        unknown_tool = tool_registry.get("get_user_profile")
        write_tool = tool_registry.get("update_learning_profile")
        search_tool = tool_registry.get("search_document")
        quiz_tool = tool_registry.get("generate_quiz")
        self.assertIsNotNone(unknown_tool)
        self.assertIsNotNone(write_tool)
        self.assertIsNotNone(search_tool)
        self.assertIsNotNone(quiz_tool)
        unknown_handler = AsyncMock(return_value='{"status":"unexpected"}')
        write_handler = AsyncMock(return_value='{"status":"unexpected"}')
        search_handler = AsyncMock(return_value='{"chunks":[],"chunk_ids":[]}')
        quiz_handler = AsyncMock(return_value='{"status":"unexpected"}')
        original_audit = list(tool_registry._audit_log)
        mark_effect_started = AsyncMock()
        run_id = "chat-without-receipt-policy"
        captured_allowlists = []
        real_run_tool_round = chat.run_tool_round

        async def capture_policy(*args, **kwargs):
            allowlist = kwargs.get("business_tool_allowlist")
            captured_allowlists.append(None if allowlist is None else set(allowlist))
            return await real_run_tool_round(*args, **kwargs)

        client = _mock_client([
            _assistant_message(tool_calls=[
                _tool_call(
                    "read",
                    "search_document",
                    '{"user_id":"u","document_id":"d","query":"q"}',
                ),
                _tool_call(
                    "cross-scope-read",
                    "generate_quiz",
                    '{"document_id":"other","topic":"RAG"}',
                ),
                _tool_call("unknown", "get_user_profile", '{"user_id":"u"}'),
                _tool_call(
                    "write",
                    "update_learning_profile",
                    '{"user_id":"other-user","document_id":"other",'
                    '"grade_result":{"score":1}}',
                ),
                _tool_call("idempotent", idempotent_tool.name, "{}"),
            ]),
            _assistant_message(content="已拒绝有状态工具"),
        ])
        tool_registry.register(idempotent_tool)

        try:
            with patch.object(chat, "_client", client), patch.object(
                chat,
                "check_injection",
                AsyncMock(return_value=(False, "")),
            ), patch.object(
                unknown_tool,
                "handler",
                new=unknown_handler,
            ), patch.object(
                write_tool,
                "handler",
                new=write_handler,
            ), patch.object(
                search_tool,
                "handler",
                new=search_handler,
            ), patch.object(
                quiz_tool,
                "handler",
                new=quiz_handler,
            ), patch.object(
                registry_module.request_idempotency,
                "mark_effect_started",
                mark_effect_started,
            ), patch.object(
                chat,
                "run_tool_round",
                new=capture_policy,
            ):
                response = await chat._execute_chat_with_tools(
                    ToolChatRequest(message="尝试有状态操作", user_id="u", document_id="d"),
                    run_id=run_id,
                    idempotency_lease=None,
                )

            visible = _visible_tool_names(client)
            expected_read_only = {
                "search_document",
                "generate_quiz",
                "get_learning_path",
                "grade_answer",
                "plan_next_step",
            }
            self.assertEqual(visible, expected_read_only)
            self.assertTrue(captured_allowlists)
            self.assertTrue(all(
                allowlist == visible
                for allowlist in captured_allowlists
            ))
            self.assertNotIn("get_user_profile", visible)
            self.assertNotIn("update_learning_profile", visible)
            self.assertNotIn(idempotent_tool.name, visible)
            self.assertIn(
                "本请求未提供幂等键，仅可使用本轮暴露的只读工具",
                client.chat.completions.create.await_args_list[0]
                .kwargs["messages"][0]["content"],
            )

            unknown_handler.assert_not_awaited()
            write_handler.assert_not_awaited()
            search_handler.assert_awaited_once()
            quiz_handler.assert_not_awaited()
            mark_effect_started.assert_not_awaited()
            self.assertEqual(idempotent_calls, [])
            new_records = tool_registry._audit_log[len(original_audit):]
            self.assertEqual(
                [record.tool_name for record in new_records],
                ["search_document"],
            )
            self.assertFalse(tool_registry.has_effect_attempt(run_id))
            self.assertEqual(response.tools_called, ["search_document"])

            tool_messages = [
                message
                for message in client.chat.completions.create.await_args_list[1]
                .kwargs["messages"]
                if message.get("role") == "tool"
            ]
            self.assertEqual(len(tool_messages), 5)
            blocked_messages = [
                message for message in tool_messages
                if "工具不在允许列表" in message["content"]
            ]
            self.assertEqual(len(blocked_messages), 3)
            self.assertTrue(all(
                "工具不在允许列表" in message["content"]
                for message in blocked_messages
            ))
            scope_messages = [
                message for message in tool_messages
                if "document_scope_mismatch" in message["content"]
            ]
            self.assertEqual(len(scope_messages), 1)
        finally:
            tool_registry.unregister(idempotent_tool.name, expected_tool=idempotent_tool)
            tool_registry._audit_log[:] = original_audit

    async def test_receipt_enables_stateful_tools_and_same_key_replays_once(self):
        idempotent_calls = []

        async def idempotent_handler():
            idempotent_calls.append(True)
            return '{"status":"idempotent"}'

        idempotent_tool = Tool(
            name="test_chat_receipt_idempotent_tool",
            description="Synthetic idempotent write enabled by a receipt.",
            parameters_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=idempotent_handler,
            metadata=ToolMetadata(
                timeout_sec=1.0,
                max_retries=0,
                effect_mode=EffectMode.IDEMPOTENT,
            ),
        )
        write_tool = tool_registry.get("update_learning_profile")
        profile_tool = tool_registry.get("get_user_profile")
        self.assertIsNotNone(write_tool)
        self.assertIsNotNone(profile_tool)
        original_audit = list(tool_registry._audit_log)
        key = "chat-receipt-write-replay-1"
        marker_seen = []
        captured_allowlists = []
        real_run_tool_round = chat.run_tool_round

        async def capture_policy(*args, **kwargs):
            captured_allowlists.append(kwargs.get("business_tool_allowlist"))
            return await real_run_tool_round(*args, **kwargs)

        with tempfile.TemporaryDirectory() as temp_dir:
            store = IdempotencyStore(
                sqlite_path=str(Path(temp_dir) / "receipts.sqlite3")
            )

            async def write_handler(user_id, document_id, grade_result):
                marker_seen.append((
                    "update_learning_profile",
                    await store.has_effect_started(key),
                ))
                return (
                    '{"status":"updated","user_id":"%s","document_id":"%s"}'
                    % (user_id, document_id)
                )

            async def profile_handler(user_id):
                marker_seen.append((
                    "get_user_profile",
                    await store.has_effect_started(key),
                ))
                return '{"user_id":"%s"}' % user_id

            client = _mock_client([
                _assistant_message(tool_calls=[
                    _tool_call("idempotent", idempotent_tool.name, "{}"),
                    _tool_call("profile", "get_user_profile", '{"user_id":"u"}'),
                    _tool_call(
                        "write",
                        "update_learning_profile",
                        '{"user_id":"u","document_id":"d",'
                        '"grade_result":{"score":1}}',
                    ),
                ]),
                _assistant_message(content="写入完成"),
            ])
            injection_check = AsyncMock(return_value=(False, ""))
            definition_loader = MagicMock(wraps=chat.get_tool_definitions)
            tool_registry.register(idempotent_tool)

            try:
                with patch.object(chat, "_client", client), patch.object(
                    chat,
                    "check_injection",
                    injection_check,
                ), patch.object(
                    chat,
                    "request_idempotency",
                    store,
                ), patch.object(
                    registry_module,
                    "request_idempotency",
                    store,
                ), patch.object(
                    chat,
                    "run_tool_round",
                    new=capture_policy,
                ), patch.object(
                    chat,
                    "get_tool_definitions",
                    definition_loader,
                ), patch.object(
                    profile_tool,
                    "handler",
                    new=profile_handler,
                ), patch.object(write_tool, "handler", new=write_handler):
                    request = ToolChatRequest(
                        message="更新学习状态",
                        user_id="u",
                        document_id="d",
                    )
                    first = await chat.chat_with_tools(request, idempotency_key=key)
                    replay = await chat.chat_with_tools(request, idempotency_key=key)
                    with self.assertRaises(IdempotencyConflictError) as mismatch:
                        await chat.chat_with_tools(
                            request.model_copy(update={"message": "不同请求"}),
                            idempotency_key=key,
                        )

                visible = _visible_tool_names(client)
                self.assertIn(idempotent_tool.name, visible)
                self.assertIn("get_user_profile", visible)
                self.assertIn("update_learning_profile", visible)
                self.assertIn(
                    "本请求已启用幂等收据，可使用本轮暴露的有状态工具",
                    client.chat.completions.create.await_args_list[0]
                    .kwargs["messages"][0]["content"],
                )
                self.assertEqual(idempotent_calls, [True])
                self.assertEqual(marker_seen, [
                    ("get_user_profile", True),
                    ("update_learning_profile", True),
                ])
                self.assertEqual(
                    first.tools_called,
                    [
                        idempotent_tool.name,
                        "get_user_profile",
                        "update_learning_profile",
                    ],
                )
                self.assertEqual(replay, first)
                self.assertEqual(mismatch.exception.reason, "payload_mismatch")
                self.assertEqual(client.chat.completions.create.await_count, 2)
                self.assertEqual(definition_loader.call_count, 2)
                self.assertEqual(injection_check.await_count, 1)
                self.assertTrue(captured_allowlists)
                self.assertTrue(all(
                    allowlist is None
                    for allowlist in captured_allowlists
                ))
                new_records = tool_registry._audit_log[len(original_audit):]
                self.assertEqual(
                    [record.tool_name for record in new_records],
                    [
                        idempotent_tool.name,
                        "get_user_profile",
                        "update_learning_profile",
                    ],
                )
            finally:
                tool_registry.unregister(
                    idempotent_tool.name,
                    expected_tool=idempotent_tool,
                )
                tool_registry._audit_log[:] = original_audit

    async def test_completed_receipt_ack_loss_replays_canonical_response(self):
        write_tool = tool_registry.get("update_learning_profile")
        self.assertIsNotNone(write_tool)
        original_audit = list(tool_registry._audit_log)
        key = "chat-complete-ack-loss-key"
        effects = []

        with tempfile.TemporaryDirectory() as temp_dir:
            store = IdempotencyStore(
                sqlite_path=str(Path(temp_dir) / "receipts.sqlite3")
            )

            async def write_handler(user_id, document_id, grade_result):
                effects.append((user_id, document_id, grade_result))
                return '{"status":"updated"}'

            client = _mock_client([
                _assistant_message(tool_calls=[_tool_call(
                    "write",
                    "update_learning_profile",
                    '{"user_id":"u","document_id":"d",'
                    '"grade_result":{"score":1}}',
                )]),
                _assistant_message(content="canonical response"),
            ])
            real_complete = store.complete

            async def complete_then_lose_ack(lease, response):
                await real_complete(lease, response)
                raise RuntimeError("completion acknowledgement lost")

            try:
                with patch.object(chat, "_client", client), patch.object(
                    chat,
                    "check_injection",
                    AsyncMock(return_value=(False, "")),
                ), patch.object(
                    chat,
                    "request_idempotency",
                    store,
                ), patch.object(
                    registry_module,
                    "request_idempotency",
                    store,
                ), patch.object(
                    store,
                    "complete",
                    new=complete_then_lose_ack,
                ), patch.object(write_tool, "handler", new=write_handler):
                    request = ToolChatRequest(
                        message="更新学习状态",
                        user_id="u",
                        document_id="d",
                    )
                    with self.assertRaises(SideEffectAmbiguousError):
                        await chat.chat_with_tools(request, idempotency_key=key)
                    replay = await chat.chat_with_tools(
                        request,
                        idempotency_key=key,
                    )

                self.assertEqual(len(effects), 1)
                self.assertEqual(replay.response, "canonical response")
                self.assertEqual(replay.tools_called, ["update_learning_profile"])
                self.assertEqual(client.chat.completions.create.await_count, 2)
                new_records = tool_registry._audit_log[len(original_audit):]
                self.assertEqual(len(new_records), 1)
                self.assertEqual(new_records[0].status, "ok")
            finally:
                tool_registry._audit_log[:] = original_audit

    async def test_effect_then_handler_failure_stays_ambiguous_on_retry(self):
        write_tool = tool_registry.get("update_learning_profile")
        self.assertIsNotNone(write_tool)
        original_audit = list(tool_registry._audit_log)
        key = "chat-handler-ambiguous-key"
        effects = []

        with tempfile.TemporaryDirectory() as temp_dir:
            store = IdempotencyStore(
                sqlite_path=str(Path(temp_dir) / "receipts.sqlite3")
            )

            async def write_then_fail(user_id, document_id, grade_result):
                effects.append((user_id, document_id, grade_result))
                raise RuntimeError("provider acknowledgement lost after write")

            client = _mock_client([
                _assistant_message(tool_calls=[_tool_call(
                    "write",
                    "update_learning_profile",
                    '{"user_id":"u","document_id":"d",'
                    '"grade_result":{"score":1}}',
                )]),
            ])

            try:
                with patch.object(chat, "_client", client), patch.object(
                    chat,
                    "check_injection",
                    AsyncMock(return_value=(False, "")),
                ), patch.object(
                    chat,
                    "request_idempotency",
                    store,
                ), patch.object(
                    registry_module,
                    "request_idempotency",
                    store,
                ), patch.object(write_tool, "handler", new=write_then_fail):
                    request = ToolChatRequest(
                        message="更新学习状态",
                        user_id="u",
                        document_id="d",
                    )
                    with self.assertRaises(SideEffectAmbiguousError):
                        await chat.chat_with_tools(request, idempotency_key=key)
                    with self.assertRaises(IdempotencyConflictError) as retry:
                        await chat.chat_with_tools(request, idempotency_key=key)

                self.assertEqual(retry.exception.reason, "ambiguous")
                self.assertEqual(len(effects), 1)
                self.assertEqual(client.chat.completions.create.await_count, 1)
                self.assertTrue(await store.has_effect_started(key))
                new_records = tool_registry._audit_log[len(original_audit):]
                self.assertEqual(len(new_records), 1)
                self.assertEqual(new_records[0].status, "ambiguous")
            finally:
                tool_registry._audit_log[:] = original_audit

    async def test_invalid_idempotency_key_fails_before_receipt_or_model(self):
        begin = AsyncMock()
        injection_check = AsyncMock()
        client = MagicMock()

        with patch.object(chat.request_idempotency, "begin", begin), patch.object(
            chat,
            "check_injection",
            injection_check,
        ), patch.object(chat, "_client", client):
            with self.assertRaises(ValueError):
                await chat.chat_with_tools(
                    ToolChatRequest(message="x", user_id="u"),
                    idempotency_key="short",
                )

        begin.assert_not_awaited()
        injection_check.assert_not_awaited()
        client.chat.completions.create.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
