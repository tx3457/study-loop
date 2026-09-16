"""跨模块调用签名契约。

这些调用在单元测试里全都被 mock 掉了——mock 接受任何参数，所以真实签名
一旦漂移，测试照样全绿，错误要到运行时才出现。这里用 inspect 直接把调用点
的关键字和被调方的签名对起来。

起因是一个真实缺陷：agents/tutor_graph.tutor_node 给 generate_lesson 传
owner_id=，而当时的 generate_lesson 并没有这个参数；所有相关用例都 patch 了
generate_lesson，因此一路绿灯。

跑：python -m pytest tests/test_agent_call_contracts.py -q
"""
import inspect
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _accepts(fn, keyword: str) -> bool:
    params = inspect.signature(fn).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    return keyword in params


class TestLessonCallContract(unittest.TestCase):
    """tutor_node / adaptive 都按关键字调 generate_lesson"""

    def setUp(self):
        from services.adaptive_loop import generate_lesson
        self.fn = generate_lesson

    def test_generate_lesson_accepts_every_keyword_its_callers_pass(self):
        # 调用点实际传的关键字（见 agents/tutor_graph.py 与 routers/adaptive.py）
        for kw in ("document_id", "topic", "weak_points", "owner_id", "last_report"):
            with self.subTest(keyword=kw):
                self.assertTrue(
                    _accepts(self.fn, kw),
                    f"generate_lesson 不接受 {kw}=，但调用点在传它",
                )

    def test_owner_id_is_required_so_retrieval_cannot_be_unscoped(self):
        param = inspect.signature(self.fn).parameters["owner_id"]
        self.assertIs(
            param.default, inspect.Parameter.empty,
            "owner_id 不能有默认值：讲解要检索文档，缺省属主等于跨属主检索",
        )


class TestRetrievalCallContract(unittest.TestCase):
    """检索入口的属主参数是必填的"""

    def test_storage_entry_points_require_an_owner(self):
        import services.vectorstore as vs
        for name in (
            "retrieve_with_rewrite", "hybrid_query_document", "query_document",
            "bm25_only_query_document", "deal_document", "delete_document",
            "get_all_document", "ensure_document_available",
        ):
            with self.subTest(function=name):
                params = inspect.signature(getattr(vs, name)).parameters
                self.assertIn("owner_id", params, f"{name} 少了 owner_id")
                self.assertIs(
                    params["owner_id"].default, inspect.Parameter.empty,
                    f"{name} 的 owner_id 有默认值，属主就成了可遗忘的参数",
                )


if __name__ == "__main__":
    unittest.main()
