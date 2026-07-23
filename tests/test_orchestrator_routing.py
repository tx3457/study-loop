"""orchestrator 入口路由 _route 单测。

_route 是整个 orchestrator 最核心的分叉点（按 action 字段确定性硬分流到三条流水线）。
使用确定性规则，测试无需 mock。
"""
import unittest

from agents.orchestrator import _route


class TestOrchestratorRoute(unittest.TestCase):
    def test_quiz_routes_to_adapt_reader(self):
        self.assertEqual(_route({"action": "quiz"}), "adapt_reader")

    def test_grade_routes_to_grader(self):
        self.assertEqual(_route({"action": "grade"}), "grader_agent")

    def test_plan_routes_to_planner(self):
        self.assertEqual(_route({"action": "plan"}), "planner_agent")

    def test_missing_action_defaults_to_quiz_flow(self):
        self.assertEqual(_route({}), "adapt_reader")

    def test_unknown_action_falls_through_to_quiz_flow(self):
        # 未知 action 落到 else 分支（quiz 流）；非法 action 由 input_guard 在更上游拦截
        self.assertEqual(_route({"action": "whatever"}), "adapt_reader")


if __name__ == "__main__":
    unittest.main()
