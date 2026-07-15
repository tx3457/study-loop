"""
自适应学习闭环测试(Direction A)

两部分:
  1) 大脑(services.adaptive_loop):归一化 / 规则兜底 / 终止判定 / LLM mock / 失败回退
  2) 路由闭环(routers.adaptive):mock 掉出题/批改/画像,验证
     达标终止、转学习路径、多轮推进 三条路径的端到端编排

全程 mock LLM 与 ChromaDB,纯逻辑验证。跑:
  python -m pytest tests/test_adaptive_loop.py -v
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.adaptive import AdaptiveTurn, NextStepDecision
from models.grader import GradingReport, QuestionGrade
from models.quiz import Question, QuizResponse
import services.adaptive_loop as brain
import routers.adaptive as rt


def _report(score: float, gaps: list[str] | None = None) -> GradingReport:
    gaps = gaps or []
    grades = [
        QuestionGrade(index=i, question=f"q{i}", user_answer="a", correct_answer="a",
                      is_correct=(i >= len(gaps)), knowledge_gap=(gaps[i] if i < len(gaps) else None))
        for i in range(2)
    ]
    correct = sum(1 for g in grades if g.is_correct)
    return GradingReport(session_id="s", total=2, correct=correct, score=score, grades=grades)


def _quiz() -> QuizResponse:
    return QuizResponse(questions=[
        Question(question="Q1", options=["A", "B"], answer="A", explanation="e", source="s", type="choice"),
        Question(question="Q2", options=["A", "B"], answer="B", explanation="e", source="s", type="choice"),
    ])


def _fake_client(decision: NextStepDecision | None = None, raises: bool = False):
    """造一个假的 AsyncOpenAI:.beta.chat.completions.parse 返回预设 decision 或抛错。"""
    async def _parse(**kwargs):
        if raises:
            raise RuntimeError("LLM down")
        msg = type("M", (), {"parsed": decision})()
        return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()
    completions = type("Comp", (), {"parse": staticmethod(_parse)})()
    chat = type("Chat", (), {"completions": completions})()
    beta = type("Beta", (), {"chat": chat})()
    return type("Client", (), {"beta": beta})()


# ═══════════════════════════════════════════════════════════════════════════
# 1) 大脑
# ═══════════════════════════════════════════════════════════════════════════
class TestBrain(unittest.IsolatedAsyncioTestCase):

    def test_normalize_clamps_and_defaults(self):
        d = NextStepDecision(action="蹦迪", difficulty="超难", question_type="xxx",
                             difficulty_score=9.9, count=99, topic="  ")
        out = brain._normalize(d, goal="排序算法")
        self.assertEqual(out.action, "continue")        # 非法 action → continue
        self.assertEqual(out.difficulty, "medium")      # 非法 difficulty → medium
        self.assertEqual(out.question_type, "choice")
        self.assertEqual(out.difficulty_score, 1.0)     # clamp 到 [0,1]
        self.assertEqual(out.count, 10)                 # clamp 到 [1,10]
        self.assertEqual(out.topic, "排序算法")          # 空 topic → goal

    def test_rule_fallback(self):
        opening = brain._rule_fallback(None, "图论", [])
        self.assertEqual(opening.action, "continue")
        self.assertEqual(opening.difficulty, "medium")

        low = brain._rule_fallback(_report(0.2), "图论", ["最短路", "拓扑排序"])
        self.assertEqual(low.action, "remediate")        # 低分 → 降难度补薄弱点
        self.assertLess(low.difficulty_score, 0.5)
        self.assertEqual(low.target_weak_points, ["最短路", "拓扑排序"])

        high = brain._rule_fallback(_report(0.9), "图论", [])
        self.assertEqual(high.action, "advance")         # 高分 → 升难度
        self.assertGreater(high.difficulty_score, 0.5)

    def test_teach_downgraded_when_disallowed(self):
        # 上一步刚讲过(allow_teach=False)→ teach 降级为 remediate,避免连续只讲不练
        out = brain._normalize(NextStepDecision(action="teach", topic="递归"), goal="g", allow_teach=False)
        self.assertEqual(out.action, "remediate")
        keep = brain._normalize(NextStepDecision(action="teach", topic="递归"), goal="g", allow_teach=True)
        self.assertEqual(keep.action, "teach")

    def test_should_terminate_priority(self):
        cont = NextStepDecision(action="continue")
        fin = NextStepDecision(action="finish")
        self.assertEqual(brain.should_terminate(mastery=0.1, turn=1, decision=fin), (True, "agent_finish"))
        self.assertEqual(brain.should_terminate(mastery=0.9, turn=1, decision=cont), (True, "mastery_reached"))
        self.assertEqual(brain.should_terminate(mastery=0.1, turn=brain.MAX_TURNS, decision=cont), (True, "max_turns"))
        self.assertEqual(brain.should_terminate(mastery=0.1, turn=1, decision=cont), (False, ""))

    async def test_decide_with_mock_client(self):
        want = NextStepDecision(action="advance", topic="动态规划", difficulty="hard",
                                difficulty_score=0.75, reason="上轮满分")
        out = await brain.decide_next_step(
            goal="算法", mastery=0.7, weak_points=[], history=[], last_report=_report(1.0),
            client=_fake_client(want),
        )
        self.assertEqual(out.action, "advance")
        self.assertEqual(out.topic, "动态规划")

    async def test_decide_llm_failure_falls_back_to_rule(self):
        out = await brain.decide_next_step(
            goal="算法", mastery=0.3, weak_points=["递归"], history=[], last_report=_report(0.0),
            client=_fake_client(raises=True),
        )
        self.assertEqual(out.action, "remediate")        # LLM 挂 → 规则兜底(低分→remediate)
        self.assertIn("兜底", out.reason)


# ═══════════════════════════════════════════════════════════════════════════
# 2) 路由闭环编排
# ═══════════════════════════════════════════════════════════════════════════
class TestAdaptiveRouterLoop(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        rt._sessions.clear()

    def _patches(self, decision_side, grade_score=0.5, mastery_seq=None):
        """统一打桩:出题/批改/画像/决策/注入全 mock。mastery_seq 控制每次 get_mastery 返回。"""
        self._p = [
            patch.object(rt, "check_injection", AsyncMock(return_value=(False, ""))),
            patch.object(rt, "generate_question", AsyncMock(return_value=_quiz())),
            patch.object(rt, "grade_session", AsyncMock(return_value=_report(grade_score))),
            patch.object(rt, "update_semantic_memory", AsyncMock()),
            patch.object(rt, "write_episodic_memory", AsyncMock()),
            patch.object(rt, "append_decision", AsyncMock()),
            patch.object(rt, "get_weak_points", AsyncMock(return_value=[])),
            patch.object(rt, "get_mastery", AsyncMock(side_effect=mastery_seq) if mastery_seq
                         else AsyncMock(return_value=0.3)),
            patch.object(rt, "decide_next_step", AsyncMock(side_effect=decision_side)
                         if isinstance(decision_side, list) else AsyncMock(return_value=decision_side)),
        ]
        for p in self._p:
            p.start()

    def tearDown(self):
        for p in getattr(self, "_p", []):
            p.stop()

    async def test_start_serves_first_quiz(self):
        self._patches(NextStepDecision(action="continue", topic="排序", reason="开场"))
        resp = await rt.adaptive_start(rt.AdaptiveStartRequest(document_id="doc", goal="排序"))
        self.assertEqual(resp.turn, 1)
        self.assertEqual(len(resp.questions), 2)
        self.assertFalse(resp.done)
        self.assertEqual(len(resp.trajectory), 1)
        self.assertIn(resp.adaptive_session_id, rt._sessions)

    async def test_mastery_reached_terminates(self):
        # get_mastery 调用序:start 1 次 + submit(_grade_and_update 1 次 + submit 本体 1 次)= 3 次
        # 末次(submit 本体)= 0.9 → 达标终止
        self._patches(NextStepDecision(action="continue", topic="排序", reason="continue"),
                      grade_score=0.9, mastery_seq=[0.3, 0.9, 0.9, 0.9])
        start = await rt.adaptive_start(rt.AdaptiveStartRequest(document_id="doc", goal="排序"))
        sid = start.adaptive_session_id
        out = await rt.adaptive_submit(rt.AdaptiveSubmitRequest(adaptive_session_id=sid, answers=["A", "B"]))
        self.assertTrue(out.done)
        self.assertEqual(out.terminate_reason, "mastery_reached")
        self.assertEqual(out.last_report_score, 0.9)

    async def test_switch_to_plan_returns_learning_path(self):
        fake_path = type("P", (), {"model_dump": lambda self: {"stages": ["s1", "s2"]}})()
        self._patches(NextStepDecision(action="switch_to_plan", topic="排序", reason="缺口系统性"),
                      grade_score=0.4, mastery_seq=[0.3, 0.4, 0.4, 0.4])
        with patch.object(rt, "generate_learning_path", AsyncMock(return_value=fake_path)):
            start = await rt.adaptive_start(rt.AdaptiveStartRequest(document_id="doc", goal="排序"))
            out = await rt.adaptive_submit(
                rt.AdaptiveSubmitRequest(adaptive_session_id=start.adaptive_session_id, answers=["A", "B"]))
        self.assertTrue(out.done)
        self.assertEqual(out.terminate_reason, "switch_to_plan")
        self.assertEqual(out.learning_path, {"stages": ["s1", "s2"]})

    async def test_continue_advances_turn(self):
        # 低 mastery + continue 决策 → 不终止,turn 推进到 2,再出一轮题
        self._patches(NextStepDecision(action="continue", topic="排序", reason="巩固"),
                      grade_score=0.5, mastery_seq=[0.3, 0.4, 0.4, 0.4])
        start = await rt.adaptive_start(rt.AdaptiveStartRequest(document_id="doc", goal="排序"))
        out = await rt.adaptive_submit(
            rt.AdaptiveSubmitRequest(adaptive_session_id=start.adaptive_session_id, answers=["A", "B"]))
        self.assertFalse(out.done)
        self.assertEqual(out.turn, 2)
        self.assertEqual(out.turn_type, "quiz")
        self.assertEqual(len(out.questions), 2)
        self.assertEqual(len(out.trajectory), 2)
        self.assertEqual(out.trajectory[0].score, 0.5)   # 第1轮得分已回填
        self.assertEqual(len(out.last_report_feedback), 2)  # 逐题反馈已带出(grader 现成内容)

    async def test_teach_turn_serves_lesson_then_quiz(self):
        # 开场 teach(讲解,不出题)→ 学生点继续(answers=[])→ 不批改 → 推进出题验证
        decisions = [
            NextStepDecision(action="teach", topic="递归", reason="没懂,先讲"),
            NextStepDecision(action="continue", topic="递归", reason="讲完出题验证"),
        ]
        self._patches(decisions, grade_score=0.5, mastery_seq=[0.3, 0.3, 0.3, 0.3])
        with patch.object(rt, "generate_lesson", AsyncMock(return_value="递归就是函数调用自己……")):
            start = await rt.adaptive_start(rt.AdaptiveStartRequest(document_id="doc", goal="递归"))
            self.assertEqual(start.turn_type, "teach")
            self.assertTrue(start.lesson)
            self.assertEqual(len(start.questions), 0)         # 讲解轮没有题

            out = await rt.adaptive_submit(
                rt.AdaptiveSubmitRequest(adaptive_session_id=start.adaptive_session_id, answers=[]))
        self.assertFalse(out.done)
        self.assertEqual(out.turn_type, "quiz")              # 讲解后转出题验证
        self.assertEqual(len(out.questions), 2)

    async def test_answer_count_mismatch_400(self):
        self._patches(NextStepDecision(action="continue", topic="排序", reason="x"))
        start = await rt.adaptive_start(rt.AdaptiveStartRequest(document_id="doc", goal="排序"))
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            await rt.adaptive_submit(
                rt.AdaptiveSubmitRequest(adaptive_session_id=start.adaptive_session_id, answers=["A"]))
        self.assertEqual(ctx.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)
