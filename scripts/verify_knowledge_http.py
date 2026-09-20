#!/usr/bin/env python3
"""Exercise real HTTP gateways + LightRAG/PG, replacing only model providers.

Requires a disposable empty graph database. --hold leaves the verified test
servers alive for manual browser checks until interrupted. Never uses user data.
"""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid

import httpx


ROOT = Path(__file__).resolve().parents[1]
TOKEN = "knowledge-http-contract-token"
USER_TOKEN = "knowledge-http-user-token"


def free_port():
    with socket.socket() as connection:
        connection.bind(("127.0.0.1", 0))
        return connection.getsockname()[1]


def wait_health(url, processes):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if any(process.poll() is not None for process in processes):
            raise RuntimeError("test service exited; inspect process logs")
        try:
            if httpx.get(url + "/health/live", timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise RuntimeError("test service failed to start")


def exercise(backend_url, graph_url):
    with httpx.Client(base_url=backend_url, headers={"Authorization": "Bearer " + USER_TOKEN},
                      timeout=50) as client:
        def request(method, path, **kwargs):
            headers = {"Idempotency-Key": str(uuid.uuid4()), **kwargs.pop("headers", {})}
            response = client.request(method, path, headers=headers, **kwargs)
            if response.status_code >= 400:
                raise RuntimeError(f"{method} {path}: {response.status_code} {response.text[:500]}")
            return response.json()

        def completed(operation):
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                job = request("GET", "/knowledge-jobs/" + operation["job_id"])
                if job["status"] == "succeeded":
                    return job
                if job["status"] == "failed":
                    raise RuntimeError(f"indexing failed: {job.get('error_code')}")
                time.sleep(0.1)
            raise RuntimeError("indexing deadline exceeded")

        def upload(kb, name, text, revision):
            return completed(request("POST", f"/knowledge-bases/{kb}/documents/upload",
                files={"file": (name, text.encode(), "text/plain")},
                data={"expected_revision": str(revision)}))

        capabilities = request("GET", "/knowledge-bases/capabilities")
        assert capabilities["enabled"] and capabilities["available"]
        first = request("POST", "/knowledge-bases", json={"name": "机器学习联调", "description": "Synthetic HTTP fixture"})
        second = request("POST", "/knowledge-bases", json={"name": "医学联调", "description": "Synthetic isolation fixture"})
        a, b = first["id"], second["id"]
        upload(a, "shared.txt", "ENTITY[Shared] ENTITY[Anchor] REL[Shared|Anchor] source_A_marker", 0)
        upload(a, "part-b.txt", "ENTITY[Shared] ENTITY[Context] REL[Shared|Context] source_B_marker", 1)
        upload(b, "shared.txt", "ENTITY[OtherOnly] ENTITY[Anchor] REL[OtherOnly|Anchor] isolated_B_marker", 0)
        graph = request("GET", f"/knowledge-bases/{a}/graph")
        assert graph["nodes"] and graph["edges"], "real graph was empty"
        assert "OtherOnly" not in json.dumps(graph), "knowledge graph leaked another KB"
        node = next(node for node in graph["nodes"] if node["label"] == "Shared")
        completed(request("POST", f"/knowledge-bases/{a}/corrections", json={
            "kind": "rename_entity", "entity_id": node["id"], "label": "Unified",
            "expected_revision": 2,
        }))
        corrected = request("GET", f"/knowledge-bases/{a}/graph")
        assert any(node["label"] == "Unified" for node in corrected["nodes"])
        answer = request("POST", "/agent/autonomous", json={
            "query": "Explain Shared and Anchor", "knowledge_base_id": a,
            "web_enabled": False, "grounding_required": True,
        })
        assert not answer["abstained"], "grounded knowledge answer abstained"
        assert answer["source_citations"] and not answer["citations"]
        assert all(source["knowledge_base_id"] == a for source in answer["source_citations"])
        assert len({source["document_id"] for source in answer["source_citations"]}) == 2

        body = "Synthetic captured web material for an explicit import contract."
        now = datetime.now(timezone.utc)
        snapshot_response = httpx.post(graph_url + "/web-snapshots", headers={
            "Authorization": "Bearer " + TOKEN, "X-StudyLoop-Subject": "default_user",
        }, json={"session_id": "http-fixture", "url": "https://example.org/study-fixture",
                 "title": "Captured fixture", "text": body,
                 "content_hash": hashlib.sha256(body.encode()).hexdigest(),
                 "fetched_at": now.isoformat(), "expires_at": (now + timedelta(days=7)).isoformat()}, timeout=15)
        snapshot_response.raise_for_status()
        snapshot = snapshot_response.json()
        completed(request("POST", f"/knowledge-bases/{a}/web-import", json={
            "snapshot_id": snapshot.get("id") or snapshot["snapshot_id"], "expected_revision": 3,
        }))
        documents = request("GET", f"/knowledge-bases/{a}/documents")["documents"]
        assert any(document["kind"] == "web" for document in documents)
        removed = next(document for document in documents if document["name"] == "part-b.txt")
        completed(request("DELETE", f"/knowledge-bases/{a}/documents/{removed['id']}",
                          json={"expected_revision": 4}))
        remaining = request("GET", f"/knowledge-bases/{a}/graph")
        assert any(node["label"] == "Unified" for node in remaining["nodes"])
        return {"status": "pass", "model_boundary": "deterministic_fakes", "live_model_calls": False,
                "web_snapshot": "synthetic_captured_fixture_not_live_fetch",
                "knowledge_base_id": a, "isolated_knowledge_base_id": b,
                "citation_count": len(answer["source_citations"]),
                "checks": ["authenticated_gateway", "two_kb_isolation", "multipart_ingestion",
                           "real_graph", "rename", "cross_document_agent_citations",
                           "explicit_web_snapshot_import", "shared_source_delete"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--graph-python", required=True)
    parser.add_argument("--backend-python", default=sys.executable)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--hold", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    ports = (free_port(), free_port())
    graph_url, backend_url = (f"http://127.0.0.1:{port}" for port in ports)
    common = {**os.environ, "PYTHONPATH": f"{ROOT}:{ROOT / 'tests'}",
              "KNOWLEDGE_SERVICE_TOKEN": TOKEN, "KNOWLEDGE_SERVICE_URL": graph_url,
              "KNOWLEDGE_BASES_ENABLED": "true", "MCP_LIVE_ENABLED": "false",
              "LANGSMITH_TRACING": "false", "LANGSMITH_TRACING_V2": "false",
              "LANGCHAIN_TRACING_V2": "false", "LANGFUSE_TRACING_ENABLED": "false"}
    graph_env = {**common, "KNOWLEDGE_DATABASE_URL": args.database_url,
                 "KNOWLEDGE_PROVIDER": "test", "LLM_MODEL": "deterministic",
                 "LLM_EMBEDDING_MODEL": "deterministic", "EMBEDDING_DIM": "64",
                 "KNOWLEDGE_EMBEDDING_DIM": "64",
                 "KNOWLEDGE_MATERIALS_DIR": str(output / "materials"),
                 "KNOWLEDGE_WORKING_DIR": str(output / "workspaces")}
    backend_env = {**common, "STUDYLOOP_AUTH_TOKEN": USER_TOKEN, "DATABASE_URL": "",
                   "LLM_EMBEDDING_MODEL": "ci-placeholder",
                   "CHROMA_DIR": str(output / "chroma"), "MEMORY_SNAPSHOT_PATH": str(output / "memory.json"),
                   "IDEMPOTENCY_DB_PATH": str(output / "idempotency.db"),
                   "CHECKPOINT_DB_PATH": str(output / "checkpoints.db")}
    for prefix in ("LLM", "STRUCTURED", "EMBEDDING"):
        backend_env[prefix + "_API_KEY"] = "ci-placeholder"
        backend_env[prefix + "_BASE_URL"] = "http://127.0.0.1:9/v1"
        backend_env[prefix + "_MODEL"] = "ci-placeholder"
    processes, streams = [], []
    try:
        for python, factory, port, env, name in (
            (args.graph_python, "graph_app", ports[0], graph_env, "graph"),
            (args.backend_python, "backend_app", ports[1], backend_env, "backend"),
        ):
            stream = (output / f"{name}.log").open("w")
            streams.append(stream)
            processes.append(subprocess.Popen([
                python, "-m", "uvicorn", f"knowledge_stack_apps:{factory}", "--factory",
                "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning",
            ], cwd=output, env=env, stdout=stream, stderr=subprocess.STDOUT))
        wait_health(graph_url, processes)
        wait_health(backend_url, processes)
        result = exercise(backend_url, graph_url)
        result.update(backend_url=backend_url, graph_url=graph_url)
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        if args.hold:
            print("Test servers held for browser inspection; interrupt to cleanly stop.", flush=True)
            while True:
                time.sleep(1)
    finally:
        for process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for stream in streams:
            stream.close()


if __name__ == "__main__":
    main()
