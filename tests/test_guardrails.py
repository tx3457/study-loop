"""input_guard / output_guard 纯校验函数单测（2026-06-03 补测试网）。

覆盖 fail-fast 输入校验与输出兜底校验。纯函数、零 mock。
注：output_guard 的「题数严格相等」校验即将在健壮性改进中放宽，
故这里只测稳定行为（空输出 / 缺字段），不锁定「严格相等」这一即将变化的细节。
"""
import unittest

from agents.guardrails import (
    GuardrailError,
    _check_action,
    _check_count,
    _check_document_id,
    _check_grade_needs_session,
    _check_plan_output,
    _check_quiz_output,
)


class TestInputGuards(unittest.TestCase):
    def test_action_valid_passes(self):
        for a in ("quiz", "grade", "plan"):
            _check_action({"action": a})  # 不抛即通过

    def test_action_default_quiz_passes(self):
        _check_action({})  # 缺省 action 默认 quiz

    def test_action_invalid_raises(self):
        with self.assertRaises(GuardrailError):
            _check_action({"action": "hack"})

    def test_count_in_range_passes(self):
        _check_count({"action": "quiz", "count": 5})

    def test_count_out_of_range_raises(self):
        for bad in (0, 21, -1):
            with self.assertRaises(GuardrailError):
                _check_count({"action": "quiz", "count": bad})

    def test_count_non_int_raises(self):
        with self.assertRaises(GuardrailError):
            _check_count({"action": "quiz", "count": "5"})

    def test_count_skipped_for_non_quiz(self):
        _check_count({"action": "grade", "count": 999})  # 非 quiz 不校验 count

    def test_grade_without_session_raises(self):
        with self.assertRaises(GuardrailError):
            _check_grade_needs_session({"action": "grade"})

    def test_grade_with_session_passes(self):
        _check_grade_needs_session({"action": "grade", "session_id": "s1"})

    def test_document_id_blank_raises(self):
        for a in ("quiz", "plan"):
            with self.assertRaises(GuardrailError):
                _check_document_id({"action": a, "document_id": "   "})

    def test_document_id_present_passes(self):
        _check_document_id({"action": "quiz", "document_id": "doc1"})

    def test_document_id_none_raises_not_crashes(self):
        # document_id 显式为 None 时应抛 GuardrailError（防御），而不是 AttributeError
        with self.assertRaises(GuardrailError):
            _check_document_id({"action": "quiz", "document_id": None})


class TestOutputGuards(unittest.TestCase):
    def test_quiz_output_valid_passes(self):
        state = {"action": "quiz", "count": 1,
                 "quiz": {"questions": [{"question": "1+1=?", "answer": "2"}]}}
        _check_quiz_output(state)

    def test_quiz_output_empty_raises(self):
        with self.assertRaises(GuardrailError):
            _check_quiz_output({"action": "quiz", "quiz": None})

    def test_quiz_output_fewer_than_expected_passes(self):
        # 降级/重试耗尽导致题数少于请求是可接受的，应放行而非整请求 400
        state = {"action": "quiz", "count": 5,
                 "quiz": {"questions": [{"question": "q", "answer": "a"}]}}
        _check_quiz_output(state)

    def test_quiz_output_more_than_expected_raises(self):
        state = {"action": "quiz", "count": 1,
                 "quiz": {"questions": [{"question": "q1", "answer": "a1"},
                                        {"question": "q2", "answer": "a2"}]}}
        with self.assertRaises(GuardrailError):
            _check_quiz_output(state)

    def test_quiz_output_missing_answer_raises(self):
        state = {"action": "quiz", "count": 1,
                 "quiz": {"questions": [{"question": "q", "answer": ""}]}}
        with self.assertRaises(GuardrailError):
            _check_quiz_output(state)

    def test_quiz_output_skipped_for_non_quiz(self):
        _check_quiz_output({"action": "grade"})  # 非 quiz 直接返回

    def test_plan_output_valid_passes(self):
        _check_plan_output({"action": "plan", "learning_path": {"stages": [{"x": 1}]}})

    def test_plan_output_empty_raises(self):
        with self.assertRaises(GuardrailError):
            _check_plan_output({"action": "plan", "learning_path": {"stages": []}})


if __name__ == "__main__":
    unittest.main()
