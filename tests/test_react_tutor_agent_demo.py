"""
React Tutor Agent demo tests.

These tests run without real model keys. A fake tool-calling client decides the
next action, while assistant_agent still uses the real run_tool_round and
ToolRegistry dispatch path.
"""
import json
import subprocess
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import agents.assistant_agent as aa
import services.tools as tool_module
from agents.assistant_agent import MAX_ASSIST_ROUNDS, assistant_agent
from services.tool_registry import EffectMode, Tool, ToolMetadata, tool_registry


def _tool_call(call_id: str, name: str, args: dict):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(args, ensure_ascii=False),
        ),
    )


def _assistant_message(content: str | None = None, tool_calls: list | None = None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls or [])
            )
        ]
    )


class _FakeCompletions:
    def __init__(self, responses: list):
        self.responses = responses
        self.index = 0

    async def create(self, *_, **__):
        if self.index >= len(self.responses):
            return _assistant_message("fake client exhausted")
        response = self.responses[self.index]
        self.index += 1
        return response


class _FakeClient:
    def __init__(self, responses: list):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=_FakeCompletions(responses).create)
        )


@contextmanager
def _patched_handlers(replacements: dict):
    originals = {}
    for name, handler in replacements.items():
        tool = tool_registry.get(name)
        if tool is None:
            # Some legacy tests clear the singleton registry. Re-register the
            # project defaults so this file is order-independent in full pytest.
            tool_module._register_all()
            tool = tool_registry.get(name)
        if tool is None:
            raise AssertionError(f"missing registered tool: {name}")
        originals[name] = tool.handler
        tool.handler = handler
    try:
        yield
    finally:
        for name, handler in originals.items():
            tool = tool_registry.get(name)
            if tool is not None:
                tool.handler = handler


async def _fake_search_document(document_id: str, query: str) -> str:
    return json.dumps({
        "chunks": ["反向传播依赖链式法则逐层计算梯度。"],
        "document_id": document_id,
        "query": query,
    }, ensure_ascii=False)


async def _fake_empty_search(document_id: str, query: str) -> str:
    return json.dumps({"chunks": [], "document_id": document_id, "query": query}, ensure_ascii=False)


async def _fake_generate_quiz(**kwargs) -> str:
    return json.dumps({
        "questions": [{
            "question": "反向传播依赖什么规则？",
            "answer": "链式法则",
            "explanation": "链式法则用于复合函数求导。",
            "source": "fake",
            "type": "short_answer",
        }]
    }, ensure_ascii=False)


async def _fake_plan_next_step(**kwargs) -> str:
    return json.dumps({
        "action": "advance",
        "recommendation": "继续做多层链式法则题。",
        "focus": ["链式法则"],
    }, ensure_ascii=False)


async def _failing_grade_answer(**_kwargs) -> str:
    raise RuntimeError("grader unavailable")


async def _temporary_runtime_tool() -> str:
    return json.dumps({"ok": True})


class TestReactTutorAgentDemo(unittest.IsolatedAsyncioTestCase):
    async def test_normal_review_loop_uses_only_replay_safe_tools(self):
        grade_result = {
            "question": "反向传播依赖什么规则？",
            "user_answer": "链式法则",
            "correct_answer": "链式法则",
            "is_correct": True,
            "score": 1.0,
        }
        responses = [
            _assistant_message(tool_calls=[_tool_call("c1", "search_document", {
                "document_id": "doc1", "query": "反向传播",
            })]),
            _assistant_message(tool_calls=[_tool_call("c2", "generate_quiz", {
                "document_id": "doc1", "topic": "反向传播", "count": 1,
            })]),
            _assistant_message(tool_calls=[_tool_call("c3", "grade_answer", {
                "question": grade_result["question"],
                "answer": grade_result["user_answer"],
                "correct_answer": grade_result["correct_answer"],
                "question_type": "short_answer",
            })]),
            _assistant_message(tool_calls=[_tool_call("c4", "plan_next_step", {
                "profile": {"weak_points": []}, "last_result": grade_result,
            })]),
            _assistant_message(tool_calls=[_tool_call("c5", "finalize", {
                "final_answer": "复习题已生成并批改，下一步继续练链式法则。",
                "reason": "闭环完成",
            })]),
        ]
        with _patched_handlers({
            "search_document": _fake_search_document,
            "generate_quiz": _fake_generate_quiz,
            "plan_next_step": _fake_plan_next_step,
        }):
            out = await assistant_agent({
                "thread_id": "test-react-normal",
                "goal": "复习反向传播，我的答案是链式法则",
                "user_id": "u1",
                "document_id": "doc1",
                "_client": _FakeClient(responses),
            })

        self.assertEqual(out["final_answer"], "复习题已生成并批改，下一步继续练链式法则。")
        self.assertEqual(out["tools_called"], [
            "search_document",
            "generate_quiz",
            "grade_answer",
            "plan_next_step",
        ])

    async def test_insufficient_evidence_can_ask_user_and_continue(self):
        responses = [
            _assistant_message(tool_calls=[_tool_call("c1", "search_document", {
                "document_id": "doc1", "query": "不存在的知识点",
            })]),
            _assistant_message(tool_calls=[_tool_call("c2", "ask_user", {
                "question": "没有检索到材料，请补充课程材料或换一个知识点。",
            })]),
            _assistant_message(tool_calls=[_tool_call("c3", "finalize", {
                "final_answer": "已收到补充材料，可以继续复习。",
                "reason": "用户补充了材料",
            })]),
        ]
        with _patched_handlers({"search_document": _fake_empty_search}), \
             patch.object(aa, "interrupt", lambda payload: "补充材料：反向传播依赖链式法则"):
            out = await assistant_agent({
                "thread_id": "test-react-ask-user",
                "goal": "复习不存在的知识点",
                "user_id": "u1",
                "document_id": "doc1",
                "_client": _FakeClient(responses),
            })

        self.assertEqual(out["final_answer"], "已收到补充材料，可以继续复习。")
        self.assertTrue(any(
            isinstance(m, dict)
            and m.get("role") == "tool"
            and "User replied" in m.get("content", "")
            for m in out["messages"]
        ))

    async def test_tool_failure_is_recorded_as_observation_then_agent_finalizes(self):
        responses = [
            _assistant_message(tool_calls=[_tool_call("c1", "grade_answer", {
                "question": "q", "answer": "a", "correct_answer": "b",
            })]),
            _assistant_message(tool_calls=[_tool_call("c2", "finalize", {
                "final_answer": "批改工具失败，先给出人工复核建议。",
                "reason": "工具失败后降级收尾",
            })]),
        ]
        with _patched_handlers({"grade_answer": _failing_grade_answer}):
            out = await assistant_agent({
                "thread_id": "test-react-tool-fail",
                "goal": "批改我的答案",
                "user_id": "u1",
                "_client": _FakeClient(responses),
            })

        self.assertEqual(out["final_answer"], "批改工具失败，先给出人工复核建议。")
        self.assertIn("grade_answer", out["tools_called"])
        self.assertTrue(any(
            isinstance(m, dict)
            and m.get("role") == "tool"
            and "grader unavailable" in m.get("content", "")
            for m in out["messages"]
        ))

    async def test_max_rounds_guard_for_never_finalizing_policy(self):
        loop_responses = [
            _assistant_message(tool_calls=[_tool_call(f"c{i}", "search_document", {
                "document_id": "doc1", "query": f"q{i}",
            })])
            for i in range(MAX_ASSIST_ROUNDS)
        ]
        responses = loop_responses + [_assistant_message("达到轮次上限后的总结")]
        with _patched_handlers({"search_document": _fake_search_document}):
            out = await assistant_agent({
                "thread_id": "test-react-max-rounds",
                "goal": "一直检索但不结束",
                "user_id": "u1",
                "document_id": "doc1",
                "_client": _FakeClient(responses),
            })

        self.assertTrue(out["assistant_done"])
        self.assertEqual(out["final_answer"], "达到轮次上限后的总结")
        self.assertEqual(out["tools_called"], ["search_document"])

    async def test_only_replay_safe_runtime_tools_are_visible_to_assistant(self):
        captured_tool_names: list[str] = []

        class CapturingCompletions:
            async def create(self, *_, **kwargs):
                captured_tool_names.extend(
                    t["function"]["name"] for t in kwargs.get("tools", [])
                )
                return _assistant_message(tool_calls=[_tool_call("c1", "finalize", {
                    "final_answer": "done",
                    "reason": "schema captured",
                })])

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=CapturingCompletions().create))
        )
        tool_registry.register(Tool(
            name="runtime_safe_tool",
            description="Read-only tool registered after services.tools import.",
            parameters_schema={"type": "object", "properties": {}},
            handler=_temporary_runtime_tool,
            metadata=ToolMetadata(
                timeout_sec=1.0,
                max_retries=0,
                effect_mode=EffectMode.READ_ONLY,
            ),
        ))
        tool_registry.register(Tool(
            name="runtime_unknown_tool",
            description="Runtime tool without a trusted effect declaration.",
            parameters_schema={"type": "object", "properties": {}},
            handler=_temporary_runtime_tool,
            metadata=ToolMetadata(timeout_sec=1.0, max_retries=0),
        ))
        try:
            out = await assistant_agent({
                "thread_id": "test-dynamic-tool-schema",
                "goal": "capture schema",
                "user_id": "u1",
                "_client": client,
            })
        finally:
            tool_registry._tools.pop("runtime_safe_tool", None)
            tool_registry._tools.pop("runtime_unknown_tool", None)

        self.assertEqual(out["final_answer"], "done")
        self.assertIn("runtime_safe_tool", captured_tool_names)
        self.assertNotIn("runtime_unknown_tool", captured_tool_names)


class TestReactTutorAgentDemoScript(unittest.TestCase):
    def test_demo_script_outputs_tool_trace(self):
        root = Path(__file__).parent.parent
        completed = subprocess.run(
            [sys.executable, "scripts/demo_react_tutor_agent.py"],
            cwd=root,
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("ReactTutorAgent deterministic trace", completed.stdout)
        self.assertIn("tool=search_document", completed.stdout)
        self.assertIn("tool=generate_quiz", completed.stdout)
        self.assertIn("tool=grade_answer", completed.stdout)
        self.assertNotIn("tool=update_learning_profile", completed.stdout)
        self.assertIn("tool=plan_next_step", completed.stdout)
        self.assertIn("action=finalize", completed.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
