"""Upload form limits must apply before authentication or file validation."""

import pytest
from fastapi.testclient import TestClient

from main import app


UPLOAD_ROUTES = [
    ("POST", "/documents/upload"),
    ("POST", "/knowledge-bases/test-kb/documents/upload"),
    ("PUT", "/knowledge-bases/test-kb/documents/test-document"),
]


@pytest.mark.parametrize(
    "method,path", UPLOAD_ROUTES,
)
@pytest.mark.parametrize(
    "body,detail",
    [
        (
            "&".join(f"field{i}=value" for i in range(1001)),
            "Too many fields",
        ),
        (
            "field=" + "x" * (1024 * 1024 + 1),
            "maximum size",
        ),
    ],
    ids=["too-many-fields", "oversized-field"],
)
def test_urlencoded_upload_limits_are_enforced_before_auth(
    monkeypatch, method, path, body, detail,
):
    monkeypatch.setenv("STUDYLOOP_AUTH_TOKEN", "form-limit-regression-token")
    response = TestClient(app).request(
        method,
        path,
        content=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    # No Authorization or Origin header: neither auth nor the browser-origin
    # guard may conceal an unbounded parser behind a later 401 or 422 response.
    assert response.status_code == 400
    assert detail in response.json()["detail"]


@pytest.mark.parametrize(
    "method,path", UPLOAD_ROUTES,
)
def test_small_urlencoded_upload_still_reaches_auth(monkeypatch, method, path):
    monkeypatch.setenv("STUDYLOOP_AUTH_TOKEN", "form-limit-regression-token")
    response = TestClient(app).request(method, path, data={"field": "value"})

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
