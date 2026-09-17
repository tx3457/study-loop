"""audit 端点脱敏单测。

audit 端点无认证，_redact 负责移除可能泄露他人文档正文/查询原文的字段，
完整 payload 仅在可信环境显式启用。
"""
import os
import json
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from main import app
from routers.audit import _redact, _serialize
from services.tool_registry import EffectMode, Tool, ToolMetadata, logger, tool_registry
from services.tools import dispatch_tool


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
        out = _redact({
            "tool_name": "search_document",
            "arguments": {
                "query": "用户的隐私查询内容",
                "document_id": "doc1",
            },
        })
        self.assertNotIn("用户的隐私查询内容", str(out["arguments"]))  # 原文不外泄
        self.assertIn("query", out["arguments"])                      # 键名保留(可观测)
        self.assertTrue(out["arguments"]["query"].startswith("<str:"))
        self.assertEqual(out["argument_count"], 2)

    def test_model_supplied_unknown_argument_name_is_not_public(self):
        secret_key = "sk-123456789012345678901234"
        out = _redact({
            "tool_name": "search_document",
            "arguments": {"query": "safe", secret_key: "value"},
        })

        self.assertIn("query", out["arguments"])
        self.assertNotIn(secret_key, str(out))
        self.assertEqual(out["argument_count"], 2)

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


class TestAuditFailureRedaction(unittest.IsolatedAsyncioTestCase):
    async def test_real_handler_secret_is_absent_from_logs_and_default_audit(self):
        secret = "sk-123456789012345678901234"
        tool_name = "audit_secret_failure_test"
        run_id = "audit-secret-failure-run"
        original_audit = list(tool_registry._audit_log)

        async def failing_handler(value: str) -> str:
            raise RuntimeError(f"handler repeated {value}")

        tool = Tool(
            name=tool_name,
            description="test-only failing tool",
            parameters_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            handler=failing_handler,
            metadata=ToolMetadata(
                max_retries=0,
                effect_mode=EffectMode.READ_ONLY,
            ),
        )
        tool_registry.register(tool)
        try:
            with self.assertLogs(logger, level="ERROR") as captured:
                with self.assertRaises(RuntimeError):
                    await dispatch_tool(
                        tool_name,
                        {"value": secret},
                        run_id=run_id,
                    )

            records = tool_registry.get_audit(run_id=run_id)
            public_audit = _serialize(records, include_payload=False)
            serialized = json.dumps(public_audit, ensure_ascii=False)
            self.assertNotIn(secret, "\n".join(captured.output))
            self.assertNotIn(secret, serialized)
            self.assertEqual(
                public_audit[-1]["error_message"],
                "handler_error:RuntimeError",
            )
        finally:
            tool_registry.unregister(tool_name, expected_tool=tool)
            tool_registry._audit_log[:] = original_audit


if __name__ == "__main__":
    unittest.main()


class TestAuditLimitBounds(unittest.TestCase):
    """GET /audit/{run_id} 的 limit 必须和同文件的 GET /audit 落在同一条边界上。

    limit 没有约束时，get_audit 内部的 items[-limit:] 在 limit=0 上退化成
    items[0:]：本该返回 0 条，实际返回全部记录；limit=-1 同理返回除第一条
    外的全部。两者都绕过了调用方以为存在的上限。
    """

    def setUp(self):
        self.client = TestClient(app, raise_server_exceptions=False)

    def test_non_positive_limit_is_rejected(self):
        for limit in (0, -1):
            with self.subTest(limit=limit):
                response = self.client.get(f"/audit/run-x?limit={limit}")
                self.assertEqual(response.status_code, 422)

    def test_limit_above_ceiling_is_rejected(self):
        response = self.client.get("/audit/run-x?limit=501")
        self.assertEqual(response.status_code, 422)

    def test_limit_within_bounds_is_accepted(self):
        response = self.client.get("/audit/run-x?limit=50")
        self.assertEqual(response.status_code, 200)
