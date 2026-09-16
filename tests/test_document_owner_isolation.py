"""文档按属主隔离。

在此之前，collection 的名字只由文件名决定：任何调用方只要知道文件名，
就能读、能删别人的文档。属主现在是存储身份的一部分——换个 owner_id 连
collection 名字都算不出来，所以「忘记校验属主」这种错误不成立。

跑：python -m pytest tests/test_document_owner_isolation.py -q
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import services.vectorstore as vs
from services.vectorstore import DEFAULT_DOCUMENT_OWNER


class TestStorageIdentity(unittest.TestCase):
    """存储 id 由 (属主, 文件名) 共同决定"""

    def test_same_filename_different_owners_never_collide(self):
        a = vs._storage_document_id("notes.md", "alice")
        b = vs._storage_document_id("notes.md", "bob")
        self.assertNotEqual(a, b, "同名文件在不同属主下必须落到不同 collection")

    def test_owner_and_name_boundary_is_unambiguous(self):
        """分隔符必须让 ("a","bc") 和 ("ab","c") 不相撞。"""
        self.assertNotEqual(
            vs._storage_document_id("bc", "a"),
            vs._storage_document_id("c", "ab"),
        )

    def test_default_owner_keeps_legacy_naming(self):
        """既有库没有属主概念，默认属主必须沿用原命名，否则历史数据全部失联。"""
        self.assertEqual(vs._storage_document_id("notes.md", DEFAULT_DOCUMENT_OWNER), "notes.md")

    def test_non_default_owner_never_takes_the_legacy_name(self):
        """否则先建库的属主会占住裸文件名，别人再传同名文件就会撞上它。"""
        self.assertNotEqual(vs._storage_document_id("notes.md", "alice"), "notes.md")

    def test_unicode_filename_still_maps_for_default_owner(self):
        storage = vs._storage_document_id("机器 学习.md", DEFAULT_DOCUMENT_OWNER)
        self.assertTrue(storage.startswith("doc-"))
        self.assertNotIn(" ", storage)


class TestCollectionOwner(unittest.TestCase):
    """读出来的 collection 归谁"""

    def test_collection_without_owner_metadata_is_default_owned(self):
        """属主概念上线前建的库没有 owner_id，必须归默认命名空间而不是无主。"""
        self.assertEqual(vs._collection_owner({}), DEFAULT_DOCUMENT_OWNER)
        self.assertEqual(vs._collection_owner({"owner_id": ""}), DEFAULT_DOCUMENT_OWNER)
        self.assertEqual(vs._collection_owner({"owner_id": None}), DEFAULT_DOCUMENT_OWNER)

    def test_explicit_owner_is_respected(self):
        self.assertEqual(vs._collection_owner({"owner_id": "alice"}), "alice")


class _FakeCollection:
    def __init__(self, name, metadata):
        self.name = name
        self.metadata = metadata


class TestOwnerCheck(unittest.TestCase):
    """取到 collection 之后的第二道闸"""

    def _collection(self, owner):
        return _FakeCollection(
            "storage-name",
            {"source_document_id": "notes.md", "owner_id": owner},
        )

    def test_matching_owner_passes(self):
        col = self._collection("alice")
        self.assertIs(vs._require_public_document_owner(col, "notes.md", "alice"), col)

    def test_other_owner_is_reported_as_missing_not_forbidden(self):
        """报「不存在」而不是「禁止访问」：后者等于一个存在性探测接口，
        调用方可以靠它枚举别人有哪些文档。"""
        col = self._collection("alice")
        with self.assertRaises(vs.DocumentOwnerMismatchError) as caught:
            vs._require_public_document_owner(col, "notes.md", "bob")
        # DocumentOwnerMismatchError 继承 NotFoundError，HTTP 边界因此回 404
        self.assertIsInstance(caught.exception, vs.NotFoundError)
        self.assertNotIn("alice", str(caught.exception))

    def test_internal_alias_still_rejected(self):
        """原有的「拿内部 collection 名冒充公开 id」防护不能因为加属主而失效。"""
        col = self._collection(DEFAULT_DOCUMENT_OWNER)
        with self.assertRaises(vs.DocumentOwnerMismatchError):
            vs._require_public_document_owner(col, "storage-name", DEFAULT_DOCUMENT_OWNER)


class TestBm25CacheIsOwnerScoped(unittest.TestCase):
    """缓存键必须含属主，否则同名文档会串正文"""

    def setUp(self):
        vs.clear_bm25_cache()
        self.addCleanup(vs.clear_bm25_cache)

    def test_same_filename_two_owners_do_not_share_an_entry(self):
        a = vs._storage_document_id("notes.md", "alice")
        b = vs._storage_document_id("notes.md", "bob")
        vs._bm25_cache_store(a, {"all_docs": ["alice 的正文"], "cached_chars": 6})
        vs._bm25_cache_store(b, {"all_docs": ["bob 的正文"], "cached_chars": 4})

        self.assertEqual(vs._bm25_cache_take(a)["all_docs"], ["alice 的正文"])
        self.assertEqual(vs._bm25_cache_take(b)["all_docs"], ["bob 的正文"])

    def test_invalidating_one_owner_leaves_the_other(self):
        a = vs._storage_document_id("notes.md", "alice")
        b = vs._storage_document_id("notes.md", "bob")
        vs._bm25_cache_store(a, {"all_docs": ["a"], "cached_chars": 1})
        vs._bm25_cache_store(b, {"all_docs": ["b"], "cached_chars": 1})

        vs._invalidate_bm25_cache(a)
        self.assertIsNone(vs._bm25_cache_take(a))
        self.assertIsNotNone(vs._bm25_cache_take(b), "失效不能波及另一个属主的同名文档")


if __name__ == "__main__":
    unittest.main()
