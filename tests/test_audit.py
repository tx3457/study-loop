"""audit 端点脱敏单测（2026-06-03 安全加固）。

audit 端点无认证，_redact 负责移除可能泄露他人文档正文/查询原文的字段。
"""
import unittest

from routers.audit import _redact


class TestAuditRedact(unittest.TestCase):
    def test_removes_output_preview(self):
        out = _redact({"tool_name": "search_document", "output_preview": "别人文档的正文片段"})
        self.assertNotIn("output_preview", out)
        self.assertEqual(out["tool_name"], "search_document")  # 非敏感元数据保留

    def test_arguments_values_masked(self):
        out = _redact({"arguments": {"query": "用户的隐私查询内容", "document_id": "doc1"}})
        self.assertNotIn("用户的隐私查询内容", str(out["arguments"]))  # 原文不外泄
        self.assertIn("query", out["arguments"])                      # 键名保留(可观测)
        self.assertTrue(out["arguments"]["query"].startswith("<str:"))

    def test_non_dict_arguments_untouched(self):
        out = _redact({"arguments": None})
        self.assertIsNone(out["arguments"])


if __name__ == "__main__":
    unittest.main()
