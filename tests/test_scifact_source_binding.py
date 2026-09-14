"""Regression tests for the SciFact verifier's two-tier source binding.

背景：`services/vectorstore.py` 原先被哈希锁定，但 `evaluation/scifact_bm25/run.py`
根本不 import 它——锁它的哈希锁的不是被测对象，结果是每次改这个仍在演进的文件都会
让一份冻结实验的校验失败，而失败本身不携带任何关于测量的信息。现在它改为「行为不变量
绑定」，被测路径（tokenization / bm25）仍然哈希锁定。

这里锁定的是：收窄之后校验器**依然会拦截**该拦截的东西，而不是变成了一道空门。
"""
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location(
    "scifact_verify", REPO / "evaluation/scifact_bm25/verify.py"
)
verify = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify)

MANIFEST_PATH = REPO / "evaluation/scifact_bm25/manifest.json"


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


class TestSourceBinding(unittest.TestCase):
    """在临时仓库副本上做破坏性对照，不触碰真实工作树。"""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)
        (self.root / "services").mkdir(parents=True)
        for name in ("tokenization.py", "bm25.py", "vectorstore.py"):
            shutil.copy2(REPO / "services" / name, self.root / "services" / name)
        self._original_root = verify.REPO_ROOT
        verify.REPO_ROOT = self.root
        self.addCleanup(lambda: setattr(verify, "REPO_ROOT", self._original_root))

    def _write(self, relative: str, text: str) -> None:
        (self.root / relative).write_text(text, encoding="utf-8")

    def _read(self, relative: str) -> str:
        return (self.root / relative).read_text(encoding="utf-8")

    def test_current_tree_passes(self):
        verify.verify_sources(_manifest())

    def test_hash_pinned_source_still_fails_on_tamper(self):
        # 被测路径必须保持哈希锁定——这是收窄后最容易被削弱的一点。
        for pinned in ("services/tokenization.py", "services/bm25.py"):
            with self.subTest(source=pinned):
                original = self._read(pinned)
                self._write(pinned, original + "\n# tampered\n")
                with self.assertRaises(ValueError) as raised:
                    verify.verify_sources(_manifest())
                self.assertIn("source hash mismatch", str(raised.exception))
                self._write(pinned, original)

    def test_vectorstore_is_not_hash_pinned(self):
        """无关改动不应再让冻结实验的校验失败——这正是收窄的目的。"""
        original = self._read("services/vectorstore.py")
        self._write("services/vectorstore.py", original + "\n# unrelated feature work\n")
        verify.verify_sources(_manifest())

    def test_broken_vectorstore_invariants_fail(self):
        original = self._read("services/vectorstore.py")
        cases = {
            "vectorstore imports pure BM25 helpers": original.replace(
                "from services.bm25 import build_bm25_index, rank_bm25",
                "from services.bm25 import build_bm25_index",
            ),
            "legacy query character split is absent": (
                original + "\n_legacy = 'get_scores(list(query))'\n"
            ),
        }
        for invariant, mutated in cases.items():
            with self.subTest(invariant=invariant):
                self._write("services/vectorstore.py", mutated)
                with self.assertRaises(ValueError) as raised:
                    verify.verify_sources(_manifest())
                self.assertIn(invariant, str(raised.exception))
                self._write("services/vectorstore.py", original)

    def test_declared_invariant_without_implementation_fails(self):
        manifest = _manifest()
        entry = manifest["studyloop_source"]["invariant_bound_files"]["vectorstore"]
        entry["invariants"] = list(entry["invariants"]) + ["no such invariant"]
        with self.assertRaises(ValueError) as raised:
            verify.verify_sources(manifest)
        self.assertIn("implements no check", str(raised.exception))

    def test_empty_invariant_list_fails(self):
        """否则一个文件可以被「记录为已绑定」却完全不受检查。"""
        manifest = _manifest()
        manifest["studyloop_source"]["invariant_bound_files"]["vectorstore"][
            "invariants"
        ] = []
        with self.assertRaises(ValueError) as raised:
            verify.verify_sources(manifest)
        self.assertIn("declares no invariants", str(raised.exception))

    def test_file_in_both_tiers_fails(self):
        manifest = _manifest()
        manifest["studyloop_source"]["files"]["vectorstore_dup"] = {
            "path": "services/vectorstore.py",
            "sha256": "0" * 64,
        }
        with self.assertRaises(ValueError) as raised:
            verify.verify_sources(manifest)
        self.assertIn("both hash-pinned and invariant-bound", str(raised.exception))


class TestManifestShape(unittest.TestCase):
    def test_evaluated_path_stays_hash_pinned(self):
        pinned = {
            entry["path"]
            for entry in _manifest()["studyloop_source"]["files"].values()
        }
        self.assertEqual(
            pinned, {"services/tokenization.py", "services/bm25.py"}
        )

    def test_narrowing_keeps_the_original_digest_on_record(self):
        """收窄不等于抹掉历史：原绑定哈希必须仍可追溯。"""
        entry = _manifest()["studyloop_source"]["invariant_bound_files"]["vectorstore"]
        self.assertEqual(len(entry["sha256_at_source_binding_commit"]), 64)
        self.assertTrue(entry["reason"])
        self.assertTrue(entry["narrowing_note"])


if __name__ == "__main__":
    unittest.main()
