"""audit 端点脱敏单测。

audit 端点无认证，_redact 负责移除可能泄露他人文档正文/查询原文的字段，
完整 payload 仅在可信环境显式启用。
"""
import os
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from routers.audit import _redact, _serialize


class _Record:
    def __init__(self, payload: dict):
        self.payload = payload

    def to_dict(self) -> dict:
        return dict(self.payload)


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


class TestAuditPayloadGate(unittest.TestCase):
    def setUp(self):
        self.records = [
            _Record({
                "tool_name": "search_document",
                "arguments": {"query": "private query"},
                "output_preview": "private document text",
            })
        ]

    def test_full_payload_is_denied_by_default(self):
        with patch.dict(os.environ, {"AUDIT_PAYLOAD_ENABLED": ""}):
            with self.assertRaises(HTTPException) as ctx:
                _serialize(self.records, include_payload=True)

        self.assertEqual(ctx.exception.status_code, 403)
        self.assertEqual(ctx.exception.detail["error"], "audit_payload_disabled")

    def test_full_payload_requires_explicit_enable(self):
        with patch.dict(os.environ, {"AUDIT_PAYLOAD_ENABLED": "true"}):
            out = _serialize(self.records, include_payload=True)

        self.assertEqual(out[0]["arguments"]["query"], "private query")
        self.assertEqual(out[0]["output_preview"], "private document text")


if __name__ == "__main__":
    unittest.main()
