"""Keep audit exceptions tied to the reviewed embedded-Chroma deployment."""

from pathlib import Path
import json
import shlex
from types import SimpleNamespace
from unittest.mock import AsyncMock

import chromadb
from fastapi.testclient import TestClient
import pytest
import yaml

from main import app
from services.vectorstore import chromadb_client
import services.vectorstore as vectorstore


ROOT = Path(__file__).resolve().parents[1]
REVIEWED_CHROMA_ADVISORIES = {
    "GHSA-36p7-vc44-83pf",
    "GHSA-f4j7-r4q5-qw2c",
    "GHSA-2wm9-hf6c-p5cr",
    "GHSA-xph7-9rjv-w5fr",
}


def test_audit_exempts_only_reviewed_ids_and_keeps_dependency_scanning():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    job = workflow["jobs"]["dependency-audit"]
    step = next(s for s in job["steps"] if s.get("name") == "Audit pinned dependencies")
    args = shlex.split(step["run"].replace("\\\n", " "), comments=True)

    assert args[:5] == ["pip-audit", "-r", "requirements.txt", "--progress-spinner", "off"]
    assert len(args[5:]) == 2 * len(REVIEWED_CHROMA_ADVISORIES)
    assert args[5::2] == ["--ignore-vuln"] * len(REVIEWED_CHROMA_ADVISORIES)
    assert set(args[6::2]) == REVIEWED_CHROMA_ADVISORIES
    assert not job.get("continue-on-error", False)
    assert not step.get("continue-on-error", False)
    assert "if" not in step
    assert "chromadb==1.5.9" in (ROOT / "requirements.txt").read_text().splitlines()
    security_policy = (ROOT / "SECURITY.md").read_text()
    assert all(advisory in security_policy for advisory in REVIEWED_CHROMA_ADVISORIES)


def test_reviewed_chroma_client_remains_embedded_and_persistent():
    settings = chromadb_client.get_settings()
    assert settings.is_persistent is True
    assert settings.chroma_api_impl == "chromadb.api.rust.RustBindingsAPI"
    assert settings.chroma_server_host is None


@pytest.mark.parametrize("method,suffix", [("POST", ""), ("PUT", "/audit-probe")])
def test_application_does_not_serve_chroma_collection_endpoints(method, suffix):
    path = "/api/v2/tenants/default_tenant/databases/default_database/collections" + suffix
    with TestClient(app) as client:
        response = client.request(method, path, json={})
    assert response.status_code == 404


def test_uploaded_configuration_stays_document_text_not_embedding_configuration(tmp_path, monkeypatch):
    local_client = chromadb.PersistentClient(str(tmp_path / "chroma"))
    payload = json.dumps({
        "embedding_function": {
            "type": "huggingface",
            "model_name": "untrusted/audit-model",
            "trust_remote_code": True,
        }
    })

    async def embed(texts):
        return SimpleNamespace(data=[
            SimpleNamespace(embedding=[0.25, 0.5]) for _ in texts
        ])

    cache_before = dict(vectorstore._bm25_cache)
    monkeypatch.setenv("STUDYLOOP_AUTH_TOKEN", "")
    monkeypatch.setattr(vectorstore, "chromadb_client", local_client)
    monkeypatch.setattr(vectorstore, "_embed", AsyncMock(side_effect=embed))
    try:
        with TestClient(app) as client:
            response = client.post(
                "/documents/upload",
                files={"file": ("audit-config.txt", payload.encode(), "text/plain")},
                data={"configuration": payload, "trust_remote_code": "true"},
            )
        assert response.status_code == 200
        collection = local_client.get_collection("audit-config.txt")
        stored = collection.get(include=["documents", "embeddings"])
        assert stored["documents"] == [payload]
        assert stored["embeddings"].tolist() == [[0.25, 0.5]]
        assert "untrusted/audit-model" not in json.dumps(collection.configuration, default=str)
    finally:
        vectorstore._bm25_cache.clear()
        vectorstore._bm25_cache.update(cache_before)
