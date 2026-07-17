"""
Deterministic React Tutor Agent demo.

This script does not call a real LLM or vector database. It injects a fake
tool-calling client and fake tool handlers, but still exercises the real
assistant_agent -> run_tool_round -> ToolRegistry path.

Run:
  python scripts/demo_react_tutor_agent.py
"""
import asyncio
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Imports create OpenAI-compatible clients, but this deterministic demo injects
# a fake client and never performs network I/O. Defaults keep the demo runnable
# without a real provider key and make accidental calls fail locally.
os.environ.setdefault("LLM_API_KEY", "deterministic-demo-only")
os.environ.setdefault("LLM_BASE_URL", "http://127.0.0.1:9/v1")
os.environ.setdefault("LLM_MODEL", "deterministic-demo-only")
os.environ.setdefault("STRUCTURED_API_KEY", "deterministic-demo-only")
os.environ.setdefault("STRUCTURED_BASE_URL", "http://127.0.0.1:9/v1")
os.environ.setdefault("STRUCTURED_MODEL", "deterministic-demo-only")
os.environ.setdefault("EMBEDDING_API_KEY", "deterministic-demo-only")
os.environ.setdefault("EMBEDDING_BASE_URL", "http://127.0.0.1:9/v1")
os.environ.setdefault("LLM_EMBEDDING_MODEL", "deterministic-demo-only")

from agents.assistant_agent import assistant_agent
from services.tool_registry import tool_registry


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
        self._responses = responses
        self._index = 0

    async def create(self, *_, **__):
        if self._index >= len(self._responses):
            return _assistant_message("DEMO: fake client exhausted.")
        response = self._responses[self._index]
        self._index += 1
        return response


class _FakeClient:
    def __init__(self, responses: list):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=_FakeCompletions(responses).create)
        )


async def _fake_search_document(document_id: str, query: str) -> str:
    return json.dumps(
        {
            "chunks": [
                "反向传播通过链式法则把损失函数梯度从输出层传回每一层参数。",
                "复习时重点区分前向计算、损失计算、梯度回传和参数更新。",
            ],
            "document_id": document_id,
            "query": query,
            "mode": "fake_retriever",
        },
        ensure_ascii=False,
    )


async def _fake_generate_quiz(
    document_id: str,
    topic: str,
    count: int = 1,
    difficulty: str = "medium",
    type: str = "short_answer",
) -> str:
    return json.dumps(
        {
            "questions": [
                {
                    "question": "反向传播主要依赖哪条微积分规则来计算各层梯度？",
                    "options": None,
                    "answer": "链式法则",
                    "explanation": "链式法则可以把复合函数的梯度逐层拆开并向前一层传播。",
                    "source": "fake course material",
                    "type": type,
                }
            ],
            "document_id": document_id,
            "topic": topic,
            "difficulty": difficulty,
            "count": count,
        },
        ensure_ascii=False,
    )


async def _fake_plan_next_step(profile: dict, last_result: dict) -> str:
    return json.dumps(
        {
            "action": "advance",
            "recommendation": "下一步做 2 道包含多层复合函数的梯度推导题。",
            "focus": ["链式法则", "梯度回传"],
            "based_on_score": last_result.get("score", 1.0),
        },
        ensure_ascii=False,
    )


@contextmanager
def _patched_demo_tools():
    replacements = {
        "search_document": _fake_search_document,
        "generate_quiz": _fake_generate_quiz,
        "plan_next_step": _fake_plan_next_step,
    }
    originals = {}
    for name, handler in replacements.items():
        tool = tool_registry.get(name)
        if tool is None:
            continue
        originals[name] = tool.handler
        tool.handler = handler
    try:
        yield
    finally:
        for name, handler in originals.items():
            tool = tool_registry.get(name)
            if tool is not None:
                tool.handler = handler


async def main() -> None:
    run_id = "demo-react-tutor-agent"
    grade_result = {
        "question": "反向传播主要依赖哪条微积分规则来计算各层梯度？",
        "user_answer": "链式法则",
        "correct_answer": "链式法则",
        "is_correct": True,
        "score": 1.0,
        "feedback": "回答正确，可以进入下一步。",
        "knowledge_gap": None,
    }
    profile_snapshot = {
        "user_id": "demo_user",
        "document_id": "demo_doc",
        "score": 1.0,
        "mastery": 0.74,
        "weak_points_added": [],
    }
    responses = [
        _assistant_message(tool_calls=[_tool_call("c1", "search_document", {
            "document_id": "demo_doc",
            "query": "反向传播 链式法则",
        })]),
        _assistant_message(tool_calls=[_tool_call("c2", "generate_quiz", {
            "document_id": "demo_doc",
            "topic": "反向传播",
            "count": 1,
            "difficulty": "medium",
            "type": "short_answer",
        })]),
        _assistant_message(tool_calls=[_tool_call("c3", "grade_answer", {
            "question": grade_result["question"],
            "answer": grade_result["user_answer"],
            "correct_answer": grade_result["correct_answer"],
            "evidence": "反向传播通过链式法则把梯度逐层传回。",
            "explanation": "链式法则用于复合函数求导。",
            "question_type": "short_answer",
        })]),
        _assistant_message(tool_calls=[_tool_call("c4", "plan_next_step", {
            "profile": profile_snapshot,
            "last_result": grade_result,
        })]),
        _assistant_message(tool_calls=[_tool_call("c5", "finalize", {
            "final_answer": (
                "已基于课程材料生成 1 道反向传播复习题，批改结果为正确；"
                "建议下一步练习多层复合函数的梯度推导。"
            ),
            "reason": "检索、复习题、批改和下一步计划均已完成",
        })]),
    ]

    state = {
        "thread_id": run_id,
        "user_id": "demo_user",
        "document_id": "demo_doc",
        "goal": "根据课程材料帮我复习反向传播，并批改我的答案：链式法则。",
        "description": "复习反向传播",
        "messages": [],
        "tools_called": [],
        "_client": _FakeClient(responses),
    }

    with _patched_demo_tools():
        result = await assistant_agent(state)

    print("ReactTutorAgent deterministic trace")
    print("=" * 42)
    records = list(reversed(tool_registry.get_audit(run_id=run_id, limit=20)))
    for index, record in enumerate(records, 1):
        print(f"{index}. action=tool tool={record.tool_name} status={record.status}")
        print(f"   args={json.dumps(record.arguments, ensure_ascii=False)}")
        print(f"   observation={record.output_preview}")
    print(f"{len(records) + 1}. action=finalize tool=finalize status=ok")
    print(f"   final_answer={result.get('final_answer')}")
    print(f"tools_called={result.get('tools_called')}")


if __name__ == "__main__":
    asyncio.run(main())
