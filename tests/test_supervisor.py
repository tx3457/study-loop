"""
TeachingSupervisor 单测（Phase 1 骨架）

覆盖：
  1) LLM 模式下各 next_agent 决策 → Command.goto 正确路由（mock llm_parse 经注入假 client）
  2) _rule_fallback_next 各场景（plan/grade/critic 重出/冷启动/低分降难/高分升难）
  3) _normalize_decision 处理非法 next_agent/action、clamp difficulty_score/count
  4) done / 超 MAX_HANDOFFS / 掌握度达标 → 终止收尾（goto output_guard）
  5) supervisor_enabled / supervisor_mode env 开关
  6) tutor_graph 能 import 且 supervisor 节点已注册（图骨架自检）

全程 mock LLM，纯逻辑验证。跑:
  python -m pytest tests/test_supervisor.py -q
"""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import agents.supervisor as sup
from agents.supervisor import (
    MAX_HANDOFFS,
    SupervisorDecision,
    _normalize_decision,
    _rule_fallback_next,
    supervisor_enabled,
    supervisor_mode,
    teaching_supervisor,
)
from services.adaptive_loop import MASTERY_TARGET


def _fake_client(decision: SupervisorDecision | None = None, raises: bool = False):
    """假 AsyncOpenAI：.beta.chat.completions.parse 返回预设 decision 或抛错（同 test_adaptive_loop 风格）。"""
    async def _parse(**kwargs):
        if raises:
            raise RuntimeError("LLM down")
        msg = type("M", (), {"parsed": decision})()
        return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()
    completions = type("Comp", (), {"parse": staticmethod(_parse)})()
    chat = type("Chat", (), {"completions": completions})()
    beta = type("Beta", (), {"chat": chat})()
    return type("Client", (), {"beta": beta})()


def _report(score: float, gaps: list[str] | None = None) -> dict:
    """造一个 GradingReport.model_dump() 形状的 dict。"""
    gaps = gaps or []
    grades = [
        {"index": i, "question": f"q{i}", "is_correct": i >= len(gaps),
         "knowledge_gap": (gaps[i] if i < len(gaps) else None)}
        for i in range(2)
    ]
    correct = sum(1 for g in grades if g["is_correct"])
    return {"session_id": "s", "total": 2, "correct": correct, "score": score, "grades": grades}


# ═══════════════════════════════════════════════════════════════════════════
# 1) env 开关
# ═══════════════════════════════════════════════════════════════════════════
class TestEnvToggles(unittest.TestCase):
    def test_enabled_default_false(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MAS_SUPERVISOR_ENABLED", None)
            self.assertFalse(supervisor_enabled())

    def test_enabled_true_variants(self):
        for val in ("1", "true", "TRUE", "yes", "Yes"):
            with patch.dict(os.environ, {"MAS_SUPERVISOR_ENABLED": val}):
                self.assertTrue(supervisor_enabled(), val)

    def test_mode_default_llm(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SUPERVISOR_MODE", None)
            self.assertEqual(supervisor_mode(), "llm")

    def test_mode_rule_and_invalid(self):
        with patch.dict(os.environ, {"SUPERVISOR_MODE": "rule"}):
            self.assertEqual(supervisor_mode(), "rule")
        with patch.dict(os.environ, {"SUPERVISOR_MODE": "garbage"}):
            self.assertEqual(supervisor_mode(), "llm")   # 非法值回退 llm


# ═══════════════════════════════════════════════════════════════════════════
# 2) _normalize_decision
# ═══════════════════════════════════════════════════════════════════════════
class TestNormalize(unittest.TestCase):
    def test_normalize_clamps_and_defaults(self):
        # 非法值绕过 pydantic 校验：构造合法实例后再改字段
        d = SupervisorDecision()
        d.next_agent = "蹦迪"
        d.action = "蹦迪"
        d.difficulty = "超难"
        d.question_type = "xxx"
        d.difficulty_score = 9.9
        d.count = 99
        out = _normalize_decision(d)
        self.assertEqual(out.next_agent, "diagnostic")   # 非法 next_agent → diagnostic
        self.assertEqual(out.action, "continue")         # 非法 action → continue
        self.assertEqual(out.difficulty, "medium")
        self.assertEqual(out.question_type, "choice")
        self.assertEqual(out.difficulty_score, 1.0)      # clamp [0,1]
        self.assertEqual(out.count, 10)                  # clamp [1,10]

    def test_finish_implies_done(self):
        out = _normalize_decision(SupervisorDecision(next_agent="finish"))
        self.assertTrue(out.done)
        out2 = _normalize_decision(SupervisorDecision(next_agent="quiz", action="finish"))
        self.assertTrue(out2.done)


# ═══════════════════════════════════════════════════════════════════════════
# 3) _rule_fallback_next
# ═══════════════════════════════════════════════════════════════════════════
class TestRuleFallback(unittest.TestCase):
    def test_action_plan_routes_planner(self):
        d = _rule_fallback_next({"action": "plan", "goal": "图论"})
        self.assertEqual(d.next_agent, "planner")
        self.assertEqual(d.action, "switch_to_plan")

    def test_action_grade_routes_grader(self):
        d = _rule_fallback_next({"action": "grade", "goal": "图论"})
        self.assertEqual(d.next_agent, "grader")

    def test_pending_answers_route_grader(self):
        # 学生交了作答但还没批改 → grader
        d = _rule_fallback_next({"answers": ["A", "B"], "last_report": None, "goal": "图论",
                                 "history": [{"agent": "quiz"}]})
        self.assertEqual(d.next_agent, "grader")

    def test_critic_low_score_routes_reviser(self):
        # Phase 2：有未下发 quiz + critic 低分(overall<0.7) + 未到精修上限 → reviser 精修（不再整轮重出）
        d = _rule_fallback_next({
            "history": [{"agent": "quiz"}],
            "quiz": {"questions": [{"question": "q", "answer": "a"}]},
            "critique_history": [{"overall_score": 0.4, "suggestions": []}],
            "revision_count": 0, "goal": "图论",
        })
        self.assertEqual(d.next_agent, "reviser")
        self.assertIn("critic", d.reason)

    def test_critic_high_severity_routes_reviser(self):
        # Phase 2：含 high severity → reviser 精修
        d = _rule_fallback_next({
            "history": [{"agent": "quiz"}],
            "quiz": {"questions": [{"question": "q", "answer": "a"}]},
            "critique_history": [{"overall_score": 0.9,
                                  "suggestions": [{"severity": "high", "target": "relevance", "action": "x"}]}],
            "revision_count": 0, "goal": "图论",
        })
        self.assertEqual(d.next_agent, "reviser")

    def test_critic_not_yet_run_routes_critic(self):
        # Phase 2：刚出完题（有 quiz、无 critique）→ 先过质量门 critic
        d = _rule_fallback_next({
            "history": [{"agent": "quiz"}],
            "quiz": {"questions": [{"question": "q", "answer": "a"}]},
            "goal": "图论",
        })
        self.assertEqual(d.next_agent, "critic")

    def test_critic_pass_routes_await_answers(self):
        # Phase 2：critic 通过（高分无 high）→ 下发题目等学生作答（await_answers→wait_for_answers）
        d = _rule_fallback_next({
            "history": [{"agent": "quiz"}],
            "quiz": {"questions": [{"question": "q", "answer": "a"}]},
            "critique_history": [{"overall_score": 0.95, "suggestions": []}],
            "critic_passed": True,
            "revision_count": 1, "goal": "图论",
        })
        self.assertEqual(d.next_agent, "await_answers")

    def test_revision_cap_passes_through_to_await(self):
        # Phase 2：精修到上限仍低分 → 放行下发（避免无限 reviser↔critic 循环）
        d = _rule_fallback_next({
            "history": [{"agent": "quiz"}],
            "quiz": {"questions": [{"question": "q", "answer": "a"}]},
            "critique_history": [{"overall_score": 0.4, "suggestions": []}],
            "revision_count": 2, "goal": "图论",
        })
        self.assertEqual(d.next_agent, "await_answers")

    def test_cold_start_routes_diagnostic(self):
        d = _rule_fallback_next({"history": [], "goal": "图论"})
        self.assertEqual(d.next_agent, "diagnostic")

    def test_low_score_remediate(self):
        d = _rule_fallback_next({
            "history": [{"agent": "diagnostic"}],
            "last_report": _report(0.2, ["最短路", "拓扑排序"]),
            "weak_points": ["最短路", "拓扑排序"], "goal": "图论",
        })
        self.assertEqual(d.next_agent, "quiz")
        self.assertEqual(d.action, "remediate")
        self.assertLess(d.difficulty_score, 0.5)
        self.assertEqual(d.target_weak_points, ["最短路", "拓扑排序"])

    def test_high_score_advance(self):
        d = _rule_fallback_next({
            "history": [{"agent": "diagnostic"}],
            "last_report": _report(0.9), "goal": "图论",
        })
        self.assertEqual(d.next_agent, "quiz")
        self.assertEqual(d.action, "advance")
        self.assertGreater(d.difficulty_score, 0.5)

    def test_diagnosed_no_report_opens_quiz(self):
        # 已诊断（有 history）但还没出过题（last_report=None）→ 中等难度开场出题
        d = _rule_fallback_next({
            "history": [{"agent": "diagnostic"}],
            "last_report": None, "goal": "图论",
        })
        self.assertEqual(d.next_agent, "quiz")
        self.assertEqual(d.action, "continue")
        self.assertEqual(d.difficulty, "medium")


# ═══════════════════════════════════════════════════════════════════════════
# 4) teaching_supervisor —— LLM 模式路由 + 终止
# ═══════════════════════════════════════════════════════════════════════════
class TestSupervisorRouting(unittest.IsolatedAsyncioTestCase):

    async def _decide(self, state: dict, decision: SupervisorDecision | None = None, raises=False):
        """在 llm 模式下注入假 client 跑 teaching_supervisor，返回 Command。"""
        st = dict(state)
        st["_client"] = _fake_client(decision, raises=raises)
        with patch.dict(os.environ, {"SUPERVISOR_MODE": "llm"}):
            return await teaching_supervisor(st)

    async def test_llm_routes_to_quiz(self):
        cmd = await self._decide(
            {"goal": "算法", "turn": 1, "history": [{"agent": "diagnostic"}]},
            SupervisorDecision(next_agent="quiz", action="advance", topic="动态规划",
                               difficulty="hard", difficulty_score=0.75, reason="上轮满分"),
        )
        self.assertEqual(cmd.goto, "quiz")
        self.assertEqual(cmd.update["next_agent"], "quiz")
        self.assertEqual(cmd.update["last_action"], "advance")
        self.assertEqual(cmd.update["description"], "动态规划")     # topic 透传给 worker
        self.assertEqual(cmd.update["turn"], 2)                    # turn+1
        self.assertEqual(cmd.update["handoff_count"], 1)           # handoff+1

    async def test_llm_routes_to_grader(self):
        cmd = await self._decide(
            {"goal": "算法", "history": [{"agent": "quiz"}], "answers": ["A"]},
            SupervisorDecision(next_agent="grader", reason="批改"),
        )
        self.assertEqual(cmd.goto, "grader")

    async def test_llm_routes_to_diagnostic(self):
        cmd = await self._decide(
            {"goal": "算法", "history": []},
            SupervisorDecision(next_agent="diagnostic", reason="先诊断"),
        )
        self.assertEqual(cmd.goto, "diagnostic")

    async def test_llm_routes_to_planner(self):
        cmd = await self._decide(
            {"goal": "算法", "history": [{"agent": "quiz"}]},
            SupervisorDecision(next_agent="planner", action="switch_to_plan", reason="缺口系统"),
        )
        self.assertEqual(cmd.goto, "planner")

    async def test_llm_routes_to_tutor(self):
        cmd = await self._decide(
            {"goal": "算法", "history": [{"agent": "quiz"}], "allow_teach": True},
            SupervisorDecision(next_agent="tutor", action="remediate", reason="先讲"),
        )
        self.assertEqual(cmd.goto, "tutor")

    async def test_llm_routes_to_assistant(self):
        cmd = await self._decide(
            {"goal": "算法", "mode": "assist", "history": []},
            SupervisorDecision(next_agent="assistant", reason="开放问答"),
        )
        self.assertEqual(cmd.goto, "assistant")

    async def test_llm_finish_goes_output_guard(self):
        cmd = await self._decide(
            {"goal": "算法", "turn": 3, "history": [{"agent": "quiz"}]},
            SupervisorDecision(next_agent="finish", action="finish", done=True, reason="练够"),
        )
        self.assertEqual(cmd.goto, "output_guard")
        self.assertTrue(cmd.update["done"])
        self.assertEqual(cmd.update["terminate_reason"], "agent_finish")

    async def test_llm_failure_falls_back_to_rule(self):
        # LLM 抛错 → 规则兜底（冷启动 → diagnostic）
        cmd = await self._decide({"goal": "算法", "history": []}, raises=True)
        self.assertEqual(cmd.goto, "diagnostic")
        self.assertIn("兜底", cmd.update["supervisor_reason"])

    async def test_rule_mode_skips_llm(self):
        # SUPERVISOR_MODE=rule：即便注入会抛错的 client 也不调用 LLM，直接规则兜底
        st = {"goal": "算法", "history": [], "_client": _fake_client(raises=True)}
        with patch.dict(os.environ, {"SUPERVISOR_MODE": "rule"}):
            cmd = await teaching_supervisor(st)
        self.assertEqual(cmd.goto, "diagnostic")

    async def test_done_terminates(self):
        cmd = await teaching_supervisor({"done": True, "goal": "算法"})
        self.assertEqual(cmd.goto, "output_guard")
        self.assertEqual(cmd.update["terminate_reason"], "agent_finish")

    async def test_max_handoffs_terminates(self):
        cmd = await teaching_supervisor({"handoff_count": MAX_HANDOFFS, "goal": "算法"})
        self.assertEqual(cmd.goto, "output_guard")
        self.assertEqual(cmd.update["terminate_reason"], "max_handoffs")

    async def test_mastery_reached_terminates(self):
        cmd = await teaching_supervisor({
            "goal": "算法", "turn": 2,
            "last_report": _report(MASTERY_TARGET + 0.05),
        })
        self.assertEqual(cmd.goto, "output_guard")
        self.assertEqual(cmd.update["terminate_reason"], "mastery_reached")

    async def test_normalize_applied_on_llm_output(self):
        # LLM 返回非法 next_agent → 归一化为 diagnostic 并路由过去
        bad = SupervisorDecision()
        bad.next_agent = "wizard"
        cmd = await self._decide({"goal": "算法", "history": []}, bad)
        self.assertEqual(cmd.goto, "diagnostic")


# ═══════════════════════════════════════════════════════════════════════════
# 5) 图骨架自检
# ═══════════════════════════════════════════════════════════════════════════
class TestGraphSkeleton(unittest.TestCase):
    def test_tutor_graph_imports_and_registers_nodes(self):
        from agents.tutor_graph import tutor_graph
        nodes = set(tutor_graph.get_graph().nodes.keys())
        for expected in ("input_guard", "teaching_supervisor", "diagnostic", "quiz",
                         "grader", "critic", "planner", "tutor", "assistant", "output_guard"):
            self.assertIn(expected, nodes, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
