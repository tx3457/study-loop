"""请求身份来自服务端，不是调用方自报。

在此之前 user_id 是请求体/查询串/路径里的普通字段：任何人填别人的 id 就能读别人的
文档、画像和错题本。现在它只来自 Authorization 头解析出的主体。

跑：python -m pytest tests/test_auth_subject.py -q
"""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import HTTPException
from fastapi.testclient import TestClient

import main
from services.auth import (
    AUTH_TOKEN_ENV,
    AuthTokenConfigurationError,
    auth_enabled,
    configured_token,
    require_own_subject,
    resolve_subject,
    verify_configuration,
)
from services.vectorstore import DEFAULT_DOCUMENT_OWNER

_TOKEN = "a-sufficiently-long-token"


class TestTokenConfiguration(unittest.TestCase):
    def test_absent_token_is_anonymous_single_user(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(AUTH_TOKEN_ENV, None)
            self.assertIsNone(configured_token())
            self.assertFalse(auth_enabled())

    def test_blank_token_is_treated_as_absent(self):
        with patch.dict(os.environ, {AUTH_TOKEN_ENV: "   "}):
            self.assertIsNone(configured_token())

    def test_short_token_is_rejected_rather_than_quietly_accepted(self):
        with patch.dict(os.environ, {AUTH_TOKEN_ENV: "short"}):
            with self.assertRaises(AuthTokenConfigurationError):
                configured_token()

    def test_startup_check_fails_fast_on_an_unusable_token(self):
        """不合格的令牌要在启动时就让进程起不来，而不是每个请求各报一次 500。"""
        with patch.dict(os.environ, {AUTH_TOKEN_ENV: "short"}):
            with self.assertRaises(AuthTokenConfigurationError):
                verify_configuration()


class TestSubjectResolution(unittest.TestCase):
    def test_anonymous_mode_resolves_to_the_default_owner(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(AUTH_TOKEN_ENV, None)
            self.assertEqual(resolve_subject(None), DEFAULT_DOCUMENT_OWNER)
            self.assertEqual(resolve_subject("Bearer whatever"), DEFAULT_DOCUMENT_OWNER)

    def test_valid_token_resolves_to_the_default_owner(self):
        with patch.dict(os.environ, {AUTH_TOKEN_ENV: _TOKEN}):
            self.assertEqual(resolve_subject(f"Bearer {_TOKEN}"), DEFAULT_DOCUMENT_OWNER)

    def test_scheme_match_is_case_insensitive(self):
        with patch.dict(os.environ, {AUTH_TOKEN_ENV: _TOKEN}):
            self.assertEqual(resolve_subject(f"bEaReR {_TOKEN}"), DEFAULT_DOCUMENT_OWNER)

    def test_missing_header_is_401_with_a_challenge(self):
        with patch.dict(os.environ, {AUTH_TOKEN_ENV: _TOKEN}):
            with self.assertRaises(HTTPException) as caught:
                resolve_subject(None)
        self.assertEqual(caught.exception.status_code, 401)
        self.assertEqual(caught.exception.headers["WWW-Authenticate"], "Bearer")

    def test_wrong_token_is_401(self):
        with patch.dict(os.environ, {AUTH_TOKEN_ENV: _TOKEN}):
            with self.assertRaises(HTTPException) as caught:
                resolve_subject("Bearer " + "b" * len(_TOKEN))
        self.assertEqual(caught.exception.status_code, 401)

    def test_non_ascii_token_compares_without_raising(self):
        """compare_digest 对非 ASCII 的 str 会抛 TypeError；两侧必须先编码。"""
        token = "令牌-非常长的中文口令-abcdef"
        with patch.dict(os.environ, {AUTH_TOKEN_ENV: token}):
            self.assertEqual(resolve_subject(f"Bearer {token}"), DEFAULT_DOCUMENT_OWNER)
            with self.assertRaises(HTTPException):
                resolve_subject("Bearer 别的口令别的口令别的口令")


class TestOwnSubject(unittest.TestCase):
    def test_own_subject_passes(self):
        self.assertEqual(require_own_subject("me", "me"), "me")

    def test_other_subject_is_404_not_403(self):
        """403 会确认这个用户存在，等于一个用户名枚举接口。"""
        with self.assertRaises(HTTPException) as caught:
            require_own_subject("someone-else", "me")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertNotIn("someone-else", str(caught.exception.detail))


class TestGateCoverage(unittest.TestCase):
    """闸门挂在 include_router 上，豁免名单必须是明确的一小撮"""

    EXEMPT = {"/", "/health/live", "/health/ready"}

    def test_exempt_routes_are_exactly_the_probes(self):
        open_paths = set()
        for route in main.app.routes:
            path = getattr(route, "path", None)
            if path is None or path in {
                "/openapi.json", "/docs", "/redoc", "/docs/oauth2-redirect",
            }:
                continue
            names = set()
            dep = getattr(route, "dependant", None)
            if dep is not None:
                names = {d.call.__name__ for d in dep.dependencies}
            if "require_user_id" not in names:
                open_paths.add(path)
        self.assertEqual(
            open_paths, self.EXEMPT,
            "豁免名单变了。新增端点默认应该落在闸门内侧；"
            "要新增豁免，得先说明为什么它可以不带身份被调用。",
        )

    def test_provider_health_is_gated_because_it_calls_the_provider(self):
        """探针可以开放，但这个端点会真的向 provider 发请求。"""
        self.assertNotIn("/health/providers", self.EXEMPT)


class TestLivenessReportsAuthMode(unittest.TestCase):
    def test_liveness_states_whether_the_gate_is_on(self):
        """匿名模式不是安全边界，部署方要能不先猜出令牌就看到这件事。"""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(AUTH_TOKEN_ENV, None)
            body = TestClient(main.app).get("/health/live").json()
        self.assertEqual(body["auth"], "anonymous")

        with patch.dict(os.environ, {AUTH_TOKEN_ENV: _TOKEN}):
            body = TestClient(main.app).get("/health/live").json()
        self.assertEqual(body["auth"], "token")


class TestGateRejectsUnauthenticatedCalls(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)

    def test_business_endpoint_needs_a_token_when_the_gate_is_on(self):
        with patch.dict(os.environ, {AUTH_TOKEN_ENV: _TOKEN}):
            response = self.client.get("/documents")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers["WWW-Authenticate"], "Bearer")

    def test_probes_stay_open_when_the_gate_is_on(self):
        with patch.dict(os.environ, {AUTH_TOKEN_ENV: _TOKEN}):
            self.assertEqual(self.client.get("/health/live").status_code, 200)
            self.assertEqual(self.client.get("/").status_code, 200)


if __name__ == "__main__":
    unittest.main()
