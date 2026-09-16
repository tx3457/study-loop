import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from services.tool_loop import run_tool_round


def _tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments, ensure_ascii=False),
        ),
    )


def _response(call_id, name, arguments):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[_tool_call(call_id, name, arguments)],
                )
            )
        ]
    )


def _prior_result(call_id, name, result):
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(result, ensure_ascii=False),
        },
    ]


def test_profile_mastery_fills_missing_plan_score_without_model_reconstruction():
    profile = {
        "user_id": "u",
        "mastery": 0.4,
        "weak_points": ["BFS"],
        "history_sessions": 2,
    }
    messages = _prior_result("profile-1", "get_user_profile", profile)
    dispatch = AsyncMock(return_value='{"action":"remediate"}')

    with patch(
        "services.tool_loop.llm_chat",
        AsyncMock(
            return_value=_response(
                "plan-1",
                "plan_next_step",
                {"profile": profile, "last_result": {}},
            )
        ),
    ), patch("services.tool_loop.dispatch_tool", dispatch):
        result = asyncio.run(run_tool_round(messages, tools=[]))

    effective = dispatch.await_args.args[1]
    assert effective["profile"] == profile
    assert effective["last_result"]["score"] == 0.4
    assert result.outcomes[0].arguments == effective


def test_grade_result_fills_missing_profile_update_payload():
    grade = {
        "score": 1.0,
        "is_correct": True,
        "knowledge_gap": None,
    }
    messages = _prior_result("grade-1", "grade_answer", grade)
    dispatch = AsyncMock(return_value='{"status":"updated"}')

    with patch(
        "services.tool_loop.llm_chat",
        AsyncMock(
            return_value=_response(
                "update-1",
                "update_learning_profile",
                {"user_id": "u", "document_id": "doc"},
            )
        ),
    ), patch("services.tool_loop.dispatch_tool", dispatch):
        result = asyncio.run(run_tool_round(messages, tools=[], user_id="u"))

    effective = dispatch.await_args.args[1]
    assert effective["grade_result"] == grade
    assert result.outcomes[0].arguments == effective


def test_existing_four_step_arguments_are_not_overwritten():
    grade = {"score": 0.0, "is_correct": False}
    update = {"status": "updated", "mastery": 0.2}
    messages = (
        _prior_result("grade-1", "grade_answer", grade)
        + _prior_result("update-1", "update_learning_profile", update)
    )
    explicit = {"profile": update, "last_result": grade}
    dispatch = AsyncMock(return_value='{"action":"remediate"}')

    with patch(
        "services.tool_loop.llm_chat",
        AsyncMock(return_value=_response("plan-1", "plan_next_step", explicit)),
    ), patch("services.tool_loop.dispatch_tool", dispatch):
        asyncio.run(run_tool_round(messages, tools=[]))

    assert dispatch.await_args.args[1] == explicit
