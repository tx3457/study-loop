import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from models.chat import ToolChatRequest
import routers.chat as chat_router
import services.react_controls as react_controls
import services.tools as tools
from services.tool_registry import tool_registry


class AgentV2ToolHandlerContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_generate_quiz_topic_is_optional_and_uses_document_fallback(self):
        generated = SimpleNamespace(
            model_dump_json=lambda **kwargs: json.dumps(
                {"questions": []}, ensure_ascii=False
            )
        )
        with patch.object(
            tools, "generate_question", AsyncMock(return_value=generated)
        ) as generate:
            await tools._generate_quiz("doc-1", "default_user")

        self.assertEqual(generate.await_args.kwargs["document_id"], "doc-1")
        self.assertEqual(generate.await_args.kwargs["description"], "文档综合内容")
        # 出题要检索文档，属主必须一路传到检索层，否则运行时 TypeError
        self.assertEqual(generate.await_args.kwargs["owner_id"], "default_user")
        schema = tool_registry.get("generate_quiz").parameters_schema
        self.assertNotIn("topic", schema["required"])
        self.assertEqual(schema["properties"]["topic"]["default"], "")

    async def test_profile_exposes_document_or_real_aggregate_topic_mastery(self):
        profile = {
            "topic_mastery": {
                "doc-1": 0.9,
                "doc-2": 0.3,
                "ignored-bool": True,
                "ignored-text": "0.7",
            },
            "weak_points": ["RRF"],
        }
        with patch.object(
            tools, "get_user_profile", AsyncMock(return_value=profile)
        ):
            document = json.loads(await tools._get_user_profile("u", "doc-2"))
            aggregate = json.loads(await tools._get_user_profile("u"))

        self.assertEqual(document["mastery"], 0.3)
        self.assertEqual(document["mastery_scope"], "doc-2")
        self.assertEqual(aggregate["mastery"], 0.6)
        self.assertEqual(aggregate["mastery_scope"], "all_topics")

    async def test_plan_profile_is_optional(self):
        result = json.loads(
            await tools._plan_next_step(last_result={"score": 0.9})
        )

        self.assertEqual(result["action"], "advance")
        schema = tool_registry.get("plan_next_step").parameters_schema
        self.assertEqual(schema["required"], ["last_result"])
        self.assertEqual(schema["properties"]["profile"]["default"], {})

    async def test_profile_update_keeps_durable_snapshot_flush(self):
        with patch.object(
            tools, "update_mastery", AsyncMock(return_value=0.75)
        ), patch.object(
            tools, "append_weak_points", AsyncMock()
        ), patch.object(
            tools, "persist_memory_snapshot", AsyncMock()
        ) as persist:
            await tools._update_learning_profile(
                "u", "doc-1", {"score": 0.75, "knowledge_gap": "RRF"}
            )

        persist.assert_awaited_once_with()


class AgentV2ToolMetadataContractTests(unittest.TestCase):
    def test_owner_dedupe_and_lineage_are_declared_in_tool_metadata(self):
        profile = tool_registry.get("get_user_profile").metadata
        update = tool_registry.get("update_learning_profile").metadata
        plan = tool_registry.get("plan_next_step").metadata

        self.assertEqual(profile.owner_argument, "user_id")
        self.assertEqual(update.owner_argument, "user_id")
        self.assertTrue(update.dedupe_within_run)
        self.assertEqual(
            update.dedupe_normalizer_id,
            "update_learning_profile_event_v1",
        )
        self.assertIs(update.dedupe_normalizer, tools._normalize_profile_update_event)
        self.assertEqual(
            update.security_contract_payload()["dedupe_normalizer_id"],
            "update_learning_profile_event_v1",
        )
        self.assertIn(
            ("grade_answer", "$", "grade_result"),
            {
                (binding.source_tool, binding.source_path, binding.target_argument)
                for binding in update.argument_bindings
            },
        )
        self.assertGreaterEqual(len(plan.argument_bindings), 3)

    def test_profile_update_reservation_normalizes_handler_equivalent_events(self):
        base = {
            "user_id": "u",
            "document_id": "doc",
            "grade_result": {
                "question": "题目",
                "user_answer": "答案",
                "correct_answer": "参考答案",
                "score": 1,
                "is_correct": False,
                "knowledge_gap": None,
                "knowledge_gaps": [],
                "feedback": "ignored",
                "nonce": "first",
            },
        }
        equivalent = {
            "user_id": "u",
            "document_id": "doc",
            "grade_result": {
                "question": "题目",
                "user_answer": "答案",
                "correct_answer": "参考答案",
                "score": 1.0,
                "is_correct": True,
                "nonce": "second",
            },
        }

        first = tool_registry.semantic_reservation(
            "update_learning_profile", base
        )
        second = tool_registry.semantic_reservation(
            "update_learning_profile", equivalent
        )

        self.assertEqual(first, second)

    def test_profile_update_reservation_tracks_real_effect_identity(self):
        base = {
            "user_id": "u",
            "document_id": "doc",
            "grade_result": {
                "question": "题目",
                "user_answer": "答案",
                "correct_answer": "参考答案",
                "score": 0.0,
                "knowledge_gap": "gap-1",
                "knowledge_gaps": [None, "", "gap-2"],
            },
        }
        base_reservation = tool_registry.semantic_reservation(
            "update_learning_profile", base
        )

        mutations = [
            ("user_id", "other"),
            ("document_id", "other-doc"),
            ("question", "另一题"),
            ("user_answer", "另一个答案"),
            ("correct_answer", "另一个参考答案"),
            ("score", 1.0),
            ("knowledge_gap", "another-gap"),
        ]
        for field, value in mutations:
            with self.subTest(field=field):
                changed = json.loads(json.dumps(base, ensure_ascii=False))
                if field in {"user_id", "document_id"}:
                    changed[field] = value
                else:
                    changed["grade_result"][field] = value
                self.assertNotEqual(
                    base_reservation,
                    tool_registry.semantic_reservation(
                        "update_learning_profile", changed
                    ),
                )

    def test_profile_update_normalizer_matches_missing_score_and_gap_semantics(self):
        missing = tools._normalize_profile_update_event({
            "user_id": "u",
            "document_id": "doc",
            "grade_result": {"is_correct": True},
        })
        nulls = tools._normalize_profile_update_event({
            "user_id": "u",
            "document_id": "doc",
            "grade_result": {
                "score": None,
                "is_correct": True,
                "knowledge_gap": None,
                "knowledge_gaps": None,
            },
        })
        empty = tools._normalize_profile_update_event({
            "user_id": "u",
            "document_id": "doc",
            "grade_result": {
                "score": 1.0,
                "is_correct": False,
                "knowledge_gaps": [],
            },
        })

        self.assertEqual(missing, nulls)
        self.assertEqual(nulls, empty)
        self.assertEqual(empty["grade_result"]["score"], 1.0)
        self.assertNotIn("is_correct", empty["grade_result"])
        self.assertEqual(empty["grade_result"]["knowledge_gaps"], [])

        positive_zero = tool_registry.semantic_reservation(
            "update_learning_profile",
            {
                "user_id": "u",
                "document_id": "doc",
                "grade_result": {"score": 0.0},
            },
        )
        negative_zero = tool_registry.semantic_reservation(
            "update_learning_profile",
            {
                "user_id": "u",
                "document_id": "doc",
                "grade_result": {"score": -0.0},
            },
        )
        self.assertEqual(positive_zero, negative_zero)

    def test_react_and_chat_prompts_require_minimal_owner_safe_single_write_behavior(self):
        full_prompt = react_controls.build_react_system_prompt()
        safe_prompt = react_controls.build_react_system_prompt(replay_safe_only=True)

        for prompt in (full_prompt, safe_prompt, chat_router._TOOL_SYSTEM):
            self.assertIn("最小充分工具集", prompt)
            self.assertIn("当前用户", prompt)
        self.assertIn("最多写入一次", full_prompt)
        self.assertIn("最多写入一次", chat_router._TOOL_SYSTEM)
        self.assertIn("不暴露画像写入能力", safe_prompt)


class ChatRunPolicyCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_clears_run_policy_state_on_success(self):
        clear = MagicMock()
        with patch.object(
            chat_router, "_execute_chat_with_tools", AsyncMock(
                return_value=SimpleNamespace(
                    response="ok",
                    tools_called=[],
                    model_dump=lambda **kwargs: {"response": "ok", "tools_called": []},
                )
            )
        ), patch.object(
            chat_router.tool_registry,
            "clear_run_policy_state",
            clear,
            create=True,
        ), patch.object(chat_router.uuid, "uuid4", return_value=SimpleNamespace(hex="a" * 32)):
            await chat_router.chat_with_tools(ToolChatRequest(message="hello"))

        clear.assert_called_once_with("chat_tools_aaaaaaaaaaaa")

    async def test_chat_clears_run_policy_state_on_failure(self):
        clear = MagicMock()
        with patch.object(
            chat_router,
            "_execute_chat_with_tools",
            AsyncMock(side_effect=RuntimeError("boom")),
        ), patch.object(
            chat_router.tool_registry,
            "has_effect_attempt",
            return_value=False,
        ), patch.object(
            chat_router.tool_registry,
            "clear_run_policy_state",
            clear,
            create=True,
        ), patch.object(chat_router.uuid, "uuid4", return_value=SimpleNamespace(hex="b" * 32)):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                await chat_router.chat_with_tools(ToolChatRequest(message="hello"))

        clear.assert_called_once_with("chat_tools_bbbbbbbbbbbb")


if __name__ == "__main__":
    unittest.main()
