"""
集成测试:LangGraph Durable Checkpointer(借鉴 langgraph checkpoint-sqlite/aio.py:509)

不依赖 StudyLoop 的真实 LLM Agent，构造一个纯计算的最小 graph，验证：
1. checkpointer 写入后 sqlite 文件真实存在
2. 同一 thread_id 第二次启动(close+reopen)能 aget_state 取回上次 state
3. compile_with_checkpointer 出来的 graph 不破坏原有 orchestrator 单例

跑法:
  /path/to/python test/test_checkpointer.py -v
"""
import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).parent.parent))


class _CounterState(TypedDict, total=False):
    counter: int
    trace: list[str]


async def _node_a(state: _CounterState) -> dict:
    return {
        "counter": state.get("counter", 0) + 1,
        "trace": (state.get("trace") or []) + ["a"],
    }


async def _node_b(state: _CounterState) -> dict:
    return {
        "counter": state.get("counter", 0) + 10,
        "trace": (state.get("trace") or []) + ["b"],
    }


def _build_minimal_graph():
    """构造一个最小 2 节点 graph，与 StudyLoop 业务解耦，纯验证 checkpoint 机制。"""
    from langgraph.graph import StateGraph, START, END
    b = StateGraph(_CounterState)
    b.add_node("a", _node_a)
    b.add_node("b", _node_b)
    b.add_edge(START, "a")
    b.add_edge("a", "b")
    b.add_edge("b", END)
    return b


class TestSqliteCheckpointerPersistence(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="checkpoint_test_")
        self.db_path = str(Path(self.tmpdir) / "test.db")

    async def asyncTearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def test_sqlite_file_is_created(self):
        from services.checkpoint import open_sqlite_checkpointer
        b = _build_minimal_graph()

        async with open_sqlite_checkpointer(self.db_path) as cp:
            graph = b.compile(checkpointer=cp)
            config = {"configurable": {"thread_id": "thread_1"}}
            result = await graph.ainvoke({"counter": 0}, config=config)
            self.assertEqual(result["counter"], 11)
            self.assertEqual(result["trace"], ["a", "b"])

        # 关键:checkpointer close 后,sqlite 文件应仍存在
        self.assertTrue(Path(self.db_path).exists(), "sqlite file should persist after close")
        self.assertGreater(Path(self.db_path).stat().st_size, 0)

    async def test_state_survives_reopen(self):
        """关键场景:模拟"进程崩溃→重启",同 thread_id 取回 state"""
        from services.checkpoint import open_sqlite_checkpointer
        b = _build_minimal_graph()
        config = {"configurable": {"thread_id": "shared_thread"}}

        # 第一次:跑完 graph
        async with open_sqlite_checkpointer(self.db_path) as cp:
            graph = b.compile(checkpointer=cp)
            await graph.ainvoke({"counter": 100}, config=config)

        # 第二次:重新打开 checkpointer(模拟新进程),用同 thread_id 取 state
        async with open_sqlite_checkpointer(self.db_path) as cp:
            graph = b.compile(checkpointer=cp)
            snapshot = await graph.aget_state(config)

        # 关键 assertion:state 被恢复,而不是空 dict
        self.assertIsNotNone(snapshot.values)
        self.assertEqual(snapshot.values["counter"], 111)  # 100 + 1 + 10
        self.assertEqual(snapshot.values["trace"], ["a", "b"])

    async def test_different_thread_ids_are_isolated(self):
        """不同 thread_id 的 state 互不影响(多用户并发场景)"""
        from services.checkpoint import open_sqlite_checkpointer
        b = _build_minimal_graph()

        async with open_sqlite_checkpointer(self.db_path) as cp:
            graph = b.compile(checkpointer=cp)
            await graph.ainvoke({"counter": 0}, config={"configurable": {"thread_id": "user_alice"}})
            await graph.ainvoke({"counter": 1000}, config={"configurable": {"thread_id": "user_bob"}})

            alice = await graph.aget_state({"configurable": {"thread_id": "user_alice"}})
            bob = await graph.aget_state({"configurable": {"thread_id": "user_bob"}})

        self.assertEqual(alice.values["counter"], 11)
        self.assertEqual(bob.values["counter"], 1011)


class TestCompileWithCheckpointerFactory(unittest.IsolatedAsyncioTestCase):
    """验证 orchestrator.compile_with_checkpointer 不破坏原单例"""

    async def test_factory_returns_distinct_instance(self):
        from agents.orchestrator import orchestrator, compile_with_checkpointer
        from services.checkpoint import open_sqlite_checkpointer

        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "orch.db")
            async with open_sqlite_checkpointer(db) as cp:
                ckpt_orch = compile_with_checkpointer(cp)
                # 不同对象:新编译的 graph 是带 checkpointer 的副本
                self.assertIsNot(orchestrator, ckpt_orch)


class TestCheckpointEnabledFlag(unittest.TestCase):
    """env 开关契约"""

    def test_default_enabled(self):
        from services import checkpoint
        os.environ.pop("CHECKPOINT_ENABLED", None)
        self.assertTrue(checkpoint.checkpoint_enabled())

    def test_disabled_when_env_false(self):
        from services import checkpoint
        os.environ["CHECKPOINT_ENABLED"] = "false"
        try:
            self.assertFalse(checkpoint.checkpoint_enabled())
        finally:
            os.environ.pop("CHECKPOINT_ENABLED", None)


if __name__ == "__main__":
    unittest.main()
