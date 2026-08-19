"""
SRS 间隔重复复习调度单测（SM-2，纯逻辑 + InMemoryStore，零 LLM）

覆盖：
  1) sm2_update：答错重置 / 首次/二次/三次答对的间隔推进 / ef 下限 1.3
  2) update_after_session：新错点登记（明天到期）/ 复习通过拉长 / 复习又错重置
  3) get_due_reviews：到期筛选 + 升序 + document_id 过滤

运行：python -m pytest tests/test_srs.py -q
"""
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.srs import get_due_reviews, sm2_update, update_after_session
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
