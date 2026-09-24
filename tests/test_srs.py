"""
SRS 间隔重复复习调度单测（SM-2，纯逻辑 + InMemoryStore，零 LLM）

覆盖：
  1) sm2_update：答错重置 / 首次/二次/三次答对的间隔推进 / ef 下限 1.3
  2) update_after_session：新错点登记（明天到期）/ 复习通过拉长 / 复习又错重置
  3) get_due_reviews：到期筛选 + 升序 + document_id 过滤
  4) get_due_review_items：界面所需的完整字段（逾期天数 / 归属材料 / 连续答对数）
  5) 闭环：到期的点会进入日常出题，复习不必是一个单独的功能

运行：python -m pytest tests/test_srs.py -q
"""
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.srs import (
    get_due_review_items,
    get_due_reviews,
    sm2_update,
    update_after_session,
)
from services.memory import read_bank_state


class TestSM2(unittest.TestCase):
    def test_wrong_resets(self):
        out = sm2_update({"n": 5, "ef": 2.5, "interval": 30}, 1)
        self.assertEqual(out["n"], 0)
        self.assertEqual(out["interval"], 1)

    def test_first_correct(self):
        out = sm2_update({"n": 0, "ef": 2.5, "interval": 0}, 5)
        self.assertEqual(out["interval"], 1)
        self.assertEqual(out["n"], 1)

    def test_second_correct(self):
        out = sm2_update({"n": 1, "ef": 2.5, "interval": 1}, 5)
        self.assertEqual(out["interval"], 6)
        self.assertEqual(out["n"], 2)

    def test_third_correct_uses_ef(self):
        out = sm2_update({"n": 2, "ef": 2.5, "interval": 6}, 5)
        self.assertEqual(out["interval"], round(6 * 2.5))   # 15
        self.assertEqual(out["n"], 3)

    def test_ef_floor(self):
        s = {"n": 0, "ef": 1.3, "interval": 0}
        for _ in range(5):
            s = sm2_update(s, 3)   # 勉强答对，压低 ef
        self.assertGreaterEqual(s["ef"], 1.3)


class TestSchedule(unittest.IsolatedAsyncioTestCase):
    async def test_session_replay_is_idempotent_and_later_session_persists(self):
        uid, doc = "srs_replay_user", "doc-replay"
        first = await update_after_session(
            uid,
            doc,
            reviewed_points=[],
            wrong_gaps=["递归"],
            today=date(2026, 6, 5),
            session_id="session-1",
        )
        replay = await update_after_session(
            uid,
            doc,
            reviewed_points=[],
            wrong_gaps=["递归"],
            today=date(2026, 6, 5),
            session_id="session-1",
        )
        self.assertEqual(replay, first)

        await update_after_session(
            uid,
            doc,
            reviewed_points=[],
            wrong_gaps=["指针"],
            today=date(2026, 6, 6),
            session_id="session-2",
        )
        stored = await read_bank_state(uid, "review_schedule")
        self.assertEqual(stored["_applied_sessions"], ["session-1", "session-2"])
        self.assertIn(f"{doc}::递归", stored["items"])
        self.assertIn(f"{doc}::指针", stored["items"])

    async def test_new_gap_due_tomorrow(self):
        uid, doc = "srs_u1", "docA"
        today = date(2026, 6, 5)
        await update_after_session(uid, doc, reviewed_points=[], wrong_gaps=["递归"], today=today)
        self.assertNotIn("递归", await get_due_reviews(uid, doc, today=today))           # 今天未到期
        self.assertIn("递归", await get_due_reviews(uid, doc, today=today + timedelta(days=1)))  # 明天到期

    async def test_review_pass_extends(self):
        uid, doc = "srs_u2", "docB"
        today = date(2026, 6, 5)
        await update_after_session(uid, doc, [], ["指针"], today=today)                  # 先错（登记）
        st = await update_after_session(uid, doc, ["指针"], [], today=today + timedelta(days=1))  # 复习答对
        item = st["items"]["docB::指针"]
        self.assertGreaterEqual(item["n"], 1)
        self.assertGreaterEqual(item["interval"], 1)

    async def test_review_fail_resets(self):
        uid, doc = "srs_u3", "docC"
        today = date(2026, 6, 5)
        await update_after_session(uid, doc, [], ["并发"], today=today)
        st = await update_after_session(uid, doc, ["并发"], ["并发"], today=today + timedelta(days=1))  # 复习又错
        item = st["items"]["docC::并发"]
        self.assertEqual(item["n"], 0)
        self.assertEqual(item["interval"], 1)

    async def test_due_sorted(self):
        uid = "srs_u4"
        await update_after_session(uid, "docX", [], ["早"], today=date(2026, 6, 1))   # due 6/2
        await update_after_session(uid, "docX", [], ["晚"], today=date(2026, 6, 5))   # due 6/6
        due = await get_due_reviews(uid, "docX", today=date(2026, 6, 10))
        self.assertEqual(due[:2], ["早", "晚"])   # 按到期日升序

    async def test_document_filter(self):
        uid = "srs_u5"
        await update_after_session(uid, "docM", [], ["医学点"], today=date(2026, 6, 1))
        await update_after_session(uid, "docN", [], ["数学点"], today=date(2026, 6, 1))
        due_m = await get_due_reviews(uid, "docM", today=date(2026, 6, 10))
        self.assertIn("医学点", due_m)
        self.assertNotIn("数学点", due_m)


if __name__ == "__main__":
    unittest.main()


class TestDueReviewItems(unittest.IsolatedAsyncioTestCase):
    """界面视图：get_due_reviews 只给知识点名，界面还要知道逾期多久、属于哪份材料。"""

    async def test_items_carry_what_the_interface_needs(self):
        uid = "srs_items_user"
        # 两份材料各积累一个错点，相隔一天登记，于是到期日不同。
        await update_after_session(
            uid, "doc-a", reviewed_points=[], wrong_gaps=["反向传播"],
            today=date(2026, 6, 1), session_id="s-a",
        )
        await update_after_session(
            uid, "doc-b", reviewed_points=[], wrong_gaps=["梯度下降"],
            today=date(2026, 6, 3), session_id="s-b",
        )

        items = await get_due_review_items(uid, today=date(2026, 6, 6))
        by_point = {row["point"]: row for row in items}
        self.assertEqual(set(by_point), {"反向传播", "梯度下降"})

        # 最该复习的排在前面。
        self.assertEqual(items[0]["point"], "反向传播")

        # 新错点在次日到期，所以 6/2 到期的点在 6/6 逾期 4 天。
        self.assertEqual(by_point["反向传播"]["due"], "2026-06-02")
        self.assertEqual(by_point["反向传播"]["overdue_days"], 4)
        self.assertEqual(by_point["梯度下降"]["overdue_days"], 2)

        # 归属材料带出来了。
        self.assertEqual(by_point["反向传播"]["document_id"], "doc-a")
        self.assertEqual(by_point["梯度下降"]["document_id"], "doc-b")
        # streak 是 SM-2 的连续答对数：刚答错登记的点是 0，界面因此不会显示它。
        self.assertEqual(by_point["反向传播"]["streak"], 0)

        # 对照：既有的面向模型的函数没变，仍然只返回知识点名。
        names = await get_due_reviews(uid, today=date(2026, 6, 6))
        self.assertTrue(all(isinstance(name, str) for name in names))
        self.assertIn("反向传播", names)

    async def test_not_yet_due_is_excluded_and_scope_filters(self):
        uid = "srs_items_scope_user"
        await update_after_session(
            uid, "doc-a", reviewed_points=[], wrong_gaps=["链式法则"],
            today=date(2026, 6, 1), session_id="s-1",
        )
        # 6/1 的错点 6/2 到期；在 6/1 当天查还没到期。
        self.assertEqual(await get_due_review_items(uid, today=date(2026, 6, 1)), [])
        # 对照：次日就到期了。
        self.assertEqual(
            [row["point"] for row in await get_due_review_items(uid, today=date(2026, 6, 2))],
            ["链式法则"],
        )
        # 限定到别的材料就查不到。
        self.assertEqual(
            await get_due_review_items(uid, document_id="doc-other", today=date(2026, 6, 2)),
            [],
        )

    async def test_streak_advances_only_after_a_correct_review(self):
        """streak 用的是 SM-2 的 n：答对才推进，答错清零。界面据此判断要不要显示。"""
        uid = "srs_streak_user"
        await update_after_session(
            uid, "doc-s", reviewed_points=[], wrong_gaps=["矩阵求导"],
            today=date(2026, 6, 1), session_id="s-1",
        )
        first = await get_due_review_items(uid, today=date(2026, 6, 2))
        self.assertEqual([row["streak"] for row in first], [0])

        # 答对一次：streak 推进，间隔拉长到 1 天后。
        await update_after_session(
            uid, "doc-s", reviewed_points=["矩阵求导"], wrong_gaps=[],
            today=date(2026, 6, 2), session_id="s-2",
        )
        after_correct = await get_due_review_items(uid, today=date(2026, 6, 3))
        self.assertEqual([row["streak"] for row in after_correct], [1])

        # 再答错：streak 清零，回到明天重练。
        await update_after_session(
            uid, "doc-s", reviewed_points=["矩阵求导"], wrong_gaps=["矩阵求导"],
            today=date(2026, 6, 3), session_id="s-3",
        )
        after_wrong = await get_due_review_items(uid, today=date(2026, 6, 4))
        self.assertEqual([row["streak"] for row in after_wrong], [0])

    async def test_empty_history_yields_an_empty_queue(self):
        self.assertEqual(
            await get_due_review_items("srs_items_newcomer", today=date(2026, 6, 6)), []
        )


class TestReviewFeedsPractice(unittest.IsolatedAsyncioTestCase):
    """调度不该只喂给模型上下文：日常出题本身就该优先安排到期的点。"""

    async def _prepare(self, uid, document_id="doc-p"):
        from models.session import SessionStartRequest
        from services import session as session_service

        captured = {}

        from models.quiz import Question

        async def fake_generate(*args, **kwargs):
            captured["weak_points"] = kwargs.get("weak_points")
            return SimpleNamespace(questions=[
                Question(
                    question="一道用于验证调度接入的题", options=["A", "B", "C", "D"],
                    answer="A", explanation="解析", source="doc-p", type="choice",
                )
            ])

        with patch.object(session_service, "_ensure_document_available", AsyncMock()), \
                patch.object(session_service, "generate_question", side_effect=fake_generate):
            await session_service.prepare_session(SessionStartRequest(
                document_id=document_id, description="复习", count=1, user_id=uid,
            ))
        return captured.get("weak_points")

    async def test_due_points_reach_question_generation(self):
        uid = "srs_practice_user"
        await update_after_session(
            uid, "doc-p", reviewed_points=[], wrong_gaps=["贝叶斯定理"],
            today=date.today() - timedelta(days=3), session_id="s-1",
        )
        weak_points = await self._prepare(uid)
        self.assertIn("贝叶斯定理", weak_points or [])

    async def test_practice_still_works_without_any_schedule(self):
        """对照：没有复习记录时照常出题，不因为调度为空而失败。"""
        weak_points = await self._prepare("srs_practice_newcomer")
        self.assertEqual(weak_points, [])

    async def test_a_broken_schedule_does_not_block_practice(self):
        """复习调度不可用时练习本身不能被挡住。"""
        from services import session as session_service

        with patch.object(
            session_service, "get_due_reviews", AsyncMock(side_effect=RuntimeError("store down"))
        ):
            weak_points = await self._prepare("srs_practice_degraded")
        self.assertEqual(weak_points, [])
