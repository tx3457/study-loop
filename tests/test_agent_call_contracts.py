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
    """**kwargs 不算接受。它会静默吞掉拼错的关键字，而这个契约要挡的正是
    签名漂移——被调方必须显式声明每一个调用点传的参数，否则下面的断言会
    在被调方加上 **kwargs 的那天起变成空转。"""
    return keyword in inspect.signature(fn).parameters


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


class TestNoCallSiteOmitsTheOwner(unittest.TestCase):
    """静态扫描全部调用点，而不是只测被 mock 过的那几条路径。

    上面两个类比对的是签名；这个类比对的是**调用点**。所有相关用例都 patch 掉了
    被调方，mock 接受任何参数，所以漏传 owner_id 在测试里是看不见的——
    `_generate_quiz`、`routers/quiz.py` 和 `services/session.py` 三处就是这样
    带着运行时 TypeError 通过了全量测试。后来 LangGraph 规划子图调 `explore`
    也漏了 owner_id——`explore` 当时不在下面这张表里，所以扫描没看到它。
    """

    REQUIRE_OWNER = {
        "retrieve_with_rewrite", "hybrid_query_document", "query_document",
        "bm25_only_query_document", "deal_document", "delete_document",
        "get_all_document", "ensure_document_available",
        "generate_question", "generate_lesson", "generate_learning_path",
        "explore",
    }
    PACKAGES = ("services", "routers", "agents")

    def test_every_call_site_passes_owner_id(self):
        import ast

        root = Path(__file__).parent.parent
        missing = []
        for package in self.PACKAGES:
            for path in sorted((root / package).glob("*.py")):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call):
                        continue
                    name = getattr(node.func, "id", None) or getattr(
                        node.func, "attr", None
                    )
                    if name not in self.REQUIRE_OWNER:
                        continue
                    if any(k.arg is None for k in node.keywords):
                        continue  # **kwargs 转发，静态判断不了
                    if "owner_id" not in {k.arg for k in node.keywords}:
                        missing.append(
                            f"{path.relative_to(root)}:{node.lineno} {name}()"
                        )
        self.assertEqual(
            missing, [],
            "这些调用点没有传 owner_id，运行时会 TypeError：\n" + "\n".join(missing),
        )

    def test_owner_list_covers_every_owner_scoped_service(self):
        """REQUIRE_OWNER 是手写的表，漏在表外的函数等于没被扫描。这里反过来核对：
        services 里凡是把 owner_id 定为必填关键字参数的公开函数，都得在表里。"""
        import ast

        root = Path(__file__).parent.parent
        uncovered = []
        for path in sorted((root / "services").glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if node.name.startswith("_") or node.name in self.REQUIRE_OWNER:
                    continue
                required_keywords = {
                    arg.arg
                    for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults)
                    if default is None
                }
                if "owner_id" in required_keywords:
                    uncovered.append(f"{path.relative_to(root)}:{node.lineno} {node.name}")
        self.assertEqual(
            uncovered, [],
            "这些函数要求 owner_id，却不在 REQUIRE_OWNER 里，调用点不会被扫描：\n"
            + "\n".join(uncovered),
        )

    def test_internal_dispatch_of_owner_bound_tools_passes_the_owner(self):
        """代码自己调 dispatch_tool 时，同样要过注册表的属主校验：参数字典里要有属主
        字段，可信 user_id 还要作为关键字交给注册表。缺了参数，调用在校验阶段就被判为
        参数无效，返回的错误 JSON 很容易被当成"没有结果"吞掉——LangGraph 的三处
        critic 检索就是这样一直拿着空证据在评题。"""
        import ast

        import services.tools  # noqa: F401  触发注册
        from services.tool_registry import tool_registry

        owner_argument = {
            name: tool_registry.get(name).metadata.owner_argument
            for name in tool_registry.list_tools()
            if tool_registry.get(name).metadata.owner_argument
        }
        root = Path(__file__).parent.parent
        problems = []
        for package in self.PACKAGES:
            for path in sorted((root / package).glob("*.py")):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call) or len(node.args) < 2:
                        continue
                    name = getattr(node.func, "id", None) or getattr(
                        node.func, "attr", None
                    )
                    tool = node.args[0]
                    if name != "dispatch_tool" or not (
                        isinstance(tool, ast.Constant) and tool.value in owner_argument
                    ):
                        continue  # 工具名是变量（由模型选）时，靠注册表在运行时把关
                    where = f"{path.relative_to(root)}:{node.lineno} {tool.value}"
                    if "user_id" not in {k.arg for k in node.keywords}:
                        problems.append(f"{where}：没把可信 user_id 交给注册表")
                    arguments = node.args[1]
                    if isinstance(arguments, ast.Dict):
                        keys = {k.value for k in arguments.keys if isinstance(k, ast.Constant)}
                        if owner_argument[tool.value] not in keys:
                            problems.append(f"{where}：参数里没有 {owner_argument[tool.value]}")
        self.assertEqual(problems, [], "\n" + "\n".join(problems))


class TestToolSchemaMatchesItsHandler(unittest.TestCase):
    """注册表按模型给的参数字典调 handler，两边对不上就是运行时错误。

    这一类同样躲得过单测：用例大多直接调 handler，绕开了 schema。
    `generate_quiz` 的 required 里写了 user_id 却没写进 properties，
    `get_learning_path` 的 handler 加了 user_id 而 schema 原封不动，
    两个都是这么溜过去的。
    """

    def _tools(self):
        import services.tools  # noqa: F401  触发注册
        from services.tool_registry import tool_registry
        return tool_registry

    def test_every_handler_argument_is_reachable_from_its_schema(self):
        registry = self._tools()
        problems = []
        for name in registry.list_tools():
            tool = registry.get(name)
            schema = tool.parameters_schema or {}
            properties = set((schema.get("properties") or {}).keys())
            required = set(schema.get("required") or [])
            params = inspect.signature(tool.handler).parameters
            if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
                continue
            handler_required = {
                n for n, p in params.items()
                if p.default is inspect.Parameter.empty
                and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
            }
            for missing in sorted(handler_required - properties):
                problems.append(f"{name}: handler 要 {missing}，schema 里没有这个属性")
            for missing in sorted(handler_required - required):
                problems.append(f"{name}: handler 必填 {missing}，schema 未列入 required")
            for extra in sorted(properties - set(params)):
                problems.append(f"{name}: schema 声明了 {extra}，handler 不接受")
        self.assertEqual(problems, [], "\n" + "\n".join(problems))

    def test_tools_that_touch_user_data_bind_an_owner(self):
        """会读写用户数据的工具必须声明 owner_argument，由注册表比对可信上下文。"""
        registry = self._tools()
        for name in (
            "search_document", "generate_quiz", "get_learning_path",
            "get_user_profile", "update_learning_profile",
        ):
            with self.subTest(tool=name):
                tool = registry.get(name)
                self.assertIsNotNone(tool, f"{name} 未注册")
                self.assertEqual(
                    tool.metadata.owner_argument, "user_id",
                    f"{name} 没有绑定属主，模型可以替别人读写",
                )


if __name__ == "__main__":
    unittest.main()
