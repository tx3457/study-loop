"""quiz_agent 出题子图的纯逻辑单测（2026-06-03 补测试网）。

覆盖：
  - services.sufficiency.check_sufficiency 数量/多样性两硬门 + 覆盖软信号(非阻断)
  - quiz_agent._validate_quiz_format 格式校验(降级后的 review 核心)
  - quiz_agent.degrade 降级数值边界
  - quiz_agent._route_after_sufficiency / _should_regenerate 路由表
全部纯函数/纯计算，零 mock。
"""
import asyncio
import unittest

from langgraph.graph import END

from services.sufficiency import check_sufficiency
from agents.quiz_agent import (
    _route_after_sufficiency,
    _should_regenerate,
    _validate_quiz_format,
    degrade,
)


class TestSufficiency(unittest.TestCase):
    def test_too_few_chunks(self):
        ok, reason = check_sufficiency(["只有一个块"])
        self.assertFalse(ok)
        self.assertEqual(reason, "too_few_chunks")

    def test_low_diversity(self):
        # 两个 chunk 但前 30 字完全相同 → 多样性不足
        same = "相同前缀" * 10
        ok, reason = check_sufficiency([same + "A", same + "B"])
        self.assertFalse(ok)
        self.assertEqual(reason, "low_diversity")

    def test_passed_no_weak_points(self):
        ok, reason = check_sufficiency(
            ["机器学习是人工智能的一个分支研究算法", "深度学习使用多层神经网络进行训练"]
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "passed")

    def test_low_coverage_is_soft_signal(self):
        # 覆盖率为非阻断软信号：历史 weak_points 与本次内容无关时，
        # 仍判 sufficient（避免同文档学新方向被误判降级），只标注 passed_low_coverage。
        ok, reason = check_sufficiency(
            ["机器学习是人工智能的一个分支研究算法", "深度学习使用多层神经网络进行训练"],
            weak_points=["量子纠缠", "广义相对论", "光合作用"],
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "passed_low_coverage")

    def test_coverage_hit_passes(self):
        ok, reason = check_sufficiency(
            ["监督学习需要带标签的数据进行训练", "神经网络是深度学习的基础结构"],
            weak_points=["监督学习", "神经网络"],  # 命中率 1.0 ≥ 0.5
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "passed")


class TestValidateQuizFormat(unittest.TestCase):
    def test_valid_choice(self):
        ok, _ = _validate_quiz_format(
            [{"question": "1+1=?", "answer": "2", "options": ["1", "2"]}], "choice"
        )
        self.assertTrue(ok)

    def test_valid_short_answer_no_options(self):
        ok, _ = _validate_quiz_format(
            [{"question": "简述递归", "answer": "函数自我调用"}], "short_answer"
        )
        self.assertTrue(ok)

    def test_empty_questions(self):
        ok, reason = _validate_quiz_format([], "choice")
        self.assertFalse(ok)
        self.assertEqual(reason, "no questions generated")

    def test_empty_answer(self):
        ok, _ = _validate_quiz_format(
            [{"question": "q", "answer": "", "options": ["A", "B"]}], "choice"
        )
        self.assertFalse(ok)

    def test_choice_too_few_options(self):
        ok, _ = _validate_quiz_format(
            [{"question": "q", "answer": "A", "options": ["A"]}], "choice"
        )
        self.assertFalse(ok)


class TestDegrade(unittest.TestCase):
    def test_halves_count_and_lowers_difficulty(self):
        out = asyncio.run(degrade({"count": 6, "difficulty_score": 0.5}))
        self.assertEqual(out["count"], 3)
        self.assertAlmostEqual(out["difficulty_score"], 0.3)
        self.assertTrue(out["insufficient_evidence"])

    def test_count_floor_is_two(self):
        out = asyncio.run(degrade({"count": 2, "difficulty_score": 0.3}))
        self.assertEqual(out["count"], 2)  # max(2//2, 2) = 2

    def test_difficulty_floor_is_point_two(self):
        out = asyncio.run(degrade({"count": 4, "difficulty_score": 0.3}))
        self.assertAlmostEqual(out["difficulty_score"], 0.2)  # max(0.3-0.2, 0.2)


class TestRouting(unittest.TestCase):
    def test_route_sufficient_to_generate(self):
        self.assertEqual(_route_after_sufficiency({"sufficiency_passed": True}), "generate")

    def test_route_insufficient_first_time_rewrites(self):
        self.assertEqual(
            _route_after_sufficiency({"sufficiency_passed": False, "rewrite_count": 0}),
            "rewrite_query",
        )

    def test_route_insufficient_after_rewrite_degrades(self):
        self.assertEqual(
            _route_after_sufficiency({"sufficiency_passed": False, "rewrite_count": 1}),
            "degrade",
        )

    def test_regenerate_pass_ends(self):
        self.assertEqual(_should_regenerate({"review_passed": True}), END)

    def test_regenerate_fail_retries(self):
        self.assertEqual(
            _should_regenerate({"review_passed": False, "generate_count": 1}), "generate"
        )

    def test_regenerate_max_attempts_ends(self):
        self.assertEqual(
            _should_regenerate({"review_passed": False, "generate_count": 2}), END
        )


if __name__ == "__main__":
    unittest.main()
