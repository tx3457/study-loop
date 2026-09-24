import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from routers import chat as chat_router
from services import tools
from services.react_controls import build_react_system_prompt
from services.tool_registry import tool_registry


def _function_schema(tool_name):
    tool = tool_registry.get(tool_name)
    assert tool is not None
    return tool.to_openai_schema()["function"]


def test_quiz_topic_is_optional_because_generation_reads_the_document():
    schema = _function_schema("generate_quiz")["parameters"]

    assert schema["required"] == ["user_id", "document_id"]
    assert "topic" in schema["properties"]


def test_quiz_handler_uses_nonempty_document_scope_when_topic_is_missing():
    generated = SimpleNamespace(model_dump_json=lambda **_kwargs: '{"ok":true}')
    generate_question = AsyncMock(return_value=generated)
    adaptation = AsyncMock(return_value=(0.65, ["链式法则"]))

    with patch.object(tools, "generate_question", generate_question), \
            patch.object(tools, "adaptive_generation_params", adaptation):
        result = asyncio.run(tools._generate_quiz(document_id="doc_graph", user_id="default_user"))

    assert json.loads(result) == {"ok": True}
    adaptation.assert_awaited_once_with("default_user", "doc_graph", "medium")
    generate_question.assert_awaited_once_with(
        document_id="doc_graph",
        description="文档综合内容",
        count=3,
        difficulty="medium",
        type="choice",
        difficulty_score=0.65,
        weak_points=["链式法则"],
        owner_id="default_user",
    )


def test_quiz_tool_aims_at_the_learners_weak_and_due_points():
    """The Agent's quiz adapts exactly like the Quiz page: difficulty just above
    measured mastery, and points due for review ahead of older weak points."""
    import services.session as session_service

    generated = SimpleNamespace(model_dump_json=lambda **_kwargs: '{"ok":true}')
    generate_question = AsyncMock(return_value=generated)
    profile = {"topic_mastery": {"doc_graph": 0.5}, "weak_points": ["BFS", "DFS"]}

    with patch.object(tools, "generate_question", generate_question), \
            patch.object(session_service, "get_user_profile", AsyncMock(return_value=profile)), \
            patch.object(session_service, "get_due_reviews", AsyncMock(return_value=["DFS", "拓扑排序"])):
        asyncio.run(tools._generate_quiz("doc_graph", "learner"))

    kwargs = generate_question.await_args.kwargs
    assert kwargs["difficulty_score"] == 0.65
    # Due points lead; a point both due and weak appears once.
    assert kwargs["weak_points"] == ["DFS", "拓扑排序", "BFS"]


def test_quiz_tool_still_generates_when_the_profile_store_fails():
    """Targeting is a refinement; an unavailable profile must not stop the quiz."""
    import services.session as session_service

    generated = SimpleNamespace(model_dump_json=lambda **_kwargs: '{"ok":true}')
    generate_question = AsyncMock(return_value=generated)

    with patch.object(tools, "generate_question", generate_question), \
            patch.object(session_service, "get_user_profile",
                         AsyncMock(side_effect=RuntimeError("store down"))), \
            patch.object(session_service, "get_due_reviews", AsyncMock(return_value=[])):
        result = asyncio.run(tools._generate_quiz("doc_graph", "learner", difficulty="hard"))

    assert json.loads(result) == {"ok": True}
    kwargs = generate_question.await_args.kwargs
    assert kwargs["difficulty_score"] == 0.8
    assert kwargs["weak_points"] == []


def test_quiz_handler_preserves_an_explicit_topic():
    generated = SimpleNamespace(model_dump_json=lambda **_kwargs: '{"ok":true}')
    generate_question = AsyncMock(return_value=generated)

    with patch.object(tools, "generate_question", generate_question):
        asyncio.run(tools._generate_quiz("doc_graph", "default_user", topic="BFS"))

    assert generate_question.await_args.kwargs["description"] == "BFS"


def test_plan_profile_is_optional_and_empty_profile_has_a_safe_fallback():
    schema = _function_schema("plan_next_step")["parameters"]

    assert schema["required"] == ["last_result"]
    assert "profile" in schema["properties"]
    result = json.loads(
        asyncio.run(tools._plan_next_step(last_result={"score": 1.0}))
    )
    assert result["action"] == "advance"
    assert result["based_on_score"] == 1.0


def test_profile_tool_exposes_document_and_aggregate_mastery_from_real_shape():
    profile = {
        "user_id": "u",
        "topic_mastery": {"doc_a": 0.2, "doc_b": 0.6},
        "weak_points": [],
        "total_sessions": 2,
    }
    get_profile = AsyncMock(return_value=profile)

    with patch.object(tools, "get_user_profile", get_profile):
        document = json.loads(
            asyncio.run(tools._get_user_profile("u", document_id="doc_b"))
        )
        aggregate = json.loads(asyncio.run(tools._get_user_profile("u")))

    assert document["mastery"] == 0.6
    assert document["mastery_scope"] == "doc_b"
    assert aggregate["mastery"] == 0.4
    assert aggregate["mastery_scope"] == "all_topics"


def test_profile_tools_bind_owner_and_profile_updates_dedupe_within_run():
    profile = tool_registry.get("get_user_profile")
    update = tool_registry.get("update_learning_profile")
    assert profile is not None and update is not None

    assert profile.metadata.owner_argument == "user_id"
    assert update.metadata.owner_argument == "user_id"
    assert update.metadata.dedupe_within_run is True
    # 去重按哪种机制实现（路径白名单还是 normalizer 函数）不在这里断言——
    # 下面的 test_two_distinct_questions_with_the_same_score_are_not_deduplicated
    # 测的是行为，机制换实现时不该连带把它判红。
    assert (
        update.metadata.dedupe_argument_paths
        or update.metadata.dedupe_normalizer is not None
    ), "声明了 dedupe_within_run 却没有任何去重依据"
    assert any(
        item.source_tool == "grade_answer"
        and item.source_path == "$"
        and item.target_argument == "grade_result"
        for item in update.metadata.argument_bindings
    )


def test_two_distinct_questions_with_the_same_score_are_not_deduplicated():
    update = tool_registry.get("update_learning_profile")
    assert update is not None
    handler = AsyncMock(return_value='{"status":"updated"}')
    original_audit = list(tool_registry._audit_log)
    run_id = "two-distinct-grades"

    async def invoke(question):
        return await tool_registry.invoke(
            "update_learning_profile",
            {
                "user_id": "u",
                "document_id": "doc",
                "grade_result": {
                    "question": question,
                    "user_answer": "正确答案",
                    "correct_answer": "正确答案",
                    "score": 1.0,
                    "is_correct": True,
                    "knowledge_gap": None,
                },
            },
            run_id=run_id,
            user_id="u",
        )

    try:
        with patch.object(update, "handler", new=handler):
            asyncio.run(invoke("第一题"))
            asyncio.run(invoke("第二题"))

        assert handler.await_count == 2
    finally:
        tool_registry.clear_run_policy_state(run_id)
        tool_registry._audit_log[:] = original_audit


def test_effect_attempt_marker_survives_audit_eviction_until_run_cleanup():
    update = tool_registry.get("update_learning_profile")
    assert update is not None
    handler = AsyncMock(return_value='{"status":"updated"}')
    original_audit = list(tool_registry._audit_log)
    run_id = "effect-marker-eviction"
    arguments = {
        "user_id": "u",
        "document_id": "doc",
        "grade_result": {
            "question": "1+1",
            "user_answer": "2",
            "correct_answer": "2",
            "score": 1.0,
            "is_correct": True,
        },
    }

    try:
        with patch.object(update, "handler", new=handler):
            asyncio.run(
                tool_registry.invoke(
                    "update_learning_profile",
                    arguments,
                    run_id=run_id,
                    user_id="u",
                )
            )
        tool_registry._audit_log.clear()

        assert tool_registry.has_effect_attempt(run_id) is True
        tool_registry.clear_run_policy_state(run_id)
        assert tool_registry.has_effect_attempt(run_id) is False
    finally:
        tool_registry.clear_run_policy_state(run_id)
        tool_registry._audit_log[:] = original_audit


def test_plan_tool_declares_server_side_result_bindings():
    plan = tool_registry.get("plan_next_step")
    assert plan is not None
    bindings = {
        (item.source_tool, item.source_path, item.target_argument)
        for item in plan.metadata.argument_bindings
    }

    assert ("update_learning_profile", "$", "profile") in bindings
    assert ("get_user_profile", "$", "profile") in bindings
    assert ("grade_answer", "$", "last_result") in bindings
    assert ("get_user_profile", "mastery", "last_result.score") in bindings


def test_react_prompt_requires_minimal_tools_user_binding_and_single_writes():
    prompt = build_react_system_prompt()

    assert "最小充分工具集" in prompt
    assert "不要额外检索文档或读取画像" in prompt
    assert "原子参数" in prompt
    assert "只能访问和更新当前用户" in prompt
    assert "同一批改结果最多写入一次" in prompt


def test_chat_tool_prompt_uses_the_same_safety_and_minimality_boundaries():
    prompt = chat_router._TOOL_SYSTEM

    assert "最小充分工具集" in prompt
    assert "只能访问和更新当前用户" in prompt
    assert "同一批改结果最多写入一次" in prompt
