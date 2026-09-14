"""HTTP error boundary.

未处理的 ValueError 是内部不变量失败，必须是 500 且不回显内部消息；
只有真正由调用方造成的错误（格式非法的 Idempotency-Key）才是 400；
模型服务重试耗尽必须归类为 provider 故障，而不是被降级成参数错误。
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from agents.grader_agent import _grade
from main import app
from services.retry import RetryExhausted


class TestGraderPropagatesProviderFailure(unittest.IsolatedAsyncioTestCase):
    async def test_retry_exhausted_is_not_downgraded_to_value_error(self):
        """曾经这里转成 ValueError，导致模型服务不可用被报成 400 参数错误。"""
        with patch(
            "agents.grader_agent.with_retry",
            AsyncMock(side_effect=RetryExhausted("provider down")),
        ):
            with self.assertRaises(RetryExhausted):
                await _grade({"session_id": "s1"})


class TestHttpErrorBoundary(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app, raise_server_exceptions=False)

    def test_internal_value_error_is_500_and_does_not_leak_the_message(self):
        marker = "internal-invariant-detail-should-not-escape"
        with patch(
            "routers.user.get_user_profile",
            AsyncMock(side_effect=ValueError(marker)),
        ):
            response = self.client.get("/user/u1/profile")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["code"], "internal_error")
        self.assertNotIn(marker, response.text)

    def test_malformed_idempotency_key_is_still_a_client_error(self):
        response = self.client.post(
            "/agent/autonomous",
            headers={"Idempotency-Key": "short"},
            json={"query": "学 RAG", "user_id": "u"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["code"], "invalid_idempotency_key")


if __name__ == "__main__":
    unittest.main()
