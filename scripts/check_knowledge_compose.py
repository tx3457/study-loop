#!/usr/bin/env python3
"""Validate the resolved optional topology without starting or touching services."""

import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def main():
    password = "synthetic@pass:word/with#chars"
    environment = {
        **os.environ, "ENV_FILE": ".env.example", "POSTGRES_PASSWORD": password,
        "KNOWLEDGE_SERVICE_TOKEN": "synthetic-contract-token-123456",
        "KNOWLEDGE_POSTGRES_PASSWORD": password, "KNOWLEDGE_EMBEDDING_DIM": "1024",
        "KNOWLEDGE_BASES_ENABLED": "true",
        "LLM_MODEL": "legacy-chat", "LLM_EMBEDDING_MODEL": "legacy-embedding",
    }
    graph_overrides = {
        "KNOWLEDGE_LLM_MODEL": "Qwen/Qwen3.5-35B-A3B",
        "KNOWLEDGE_LLM_API_KEY": "synthetic-knowledge-chat-key",
        "KNOWLEDGE_LLM_BASE_URL": "https://api.siliconflow.cn/v1",
        "KNOWLEDGE_EMBEDDING_MODEL": "BAAI/bge-m3",
        "KNOWLEDGE_EMBEDDING_API_KEY": "synthetic-knowledge-embedding-key",
        "KNOWLEDGE_EMBEDDING_BASE_URL": "https://api.siliconflow.cn/v1",
        "KNOWLEDGE_EMBEDDING_MAX_INPUT_TOKENS": "8192",
        "KNOWLEDGE_EMBEDDING_TOKEN_BUDGET": "4096",
        "KNOWLEDGE_EMBEDDING_MAX_UTF8_BYTES": "8000",
        "KNOWLEDGE_CHUNK_TOKENS": "800", "KNOWLEDGE_CHUNK_OVERLAP_TOKENS": "100",
        "KNOWLEDGE_EXTRACT_MAX_RECORDS": "40", "KNOWLEDGE_EXTRACT_MAX_ENTITIES": "20",
        "KNOWLEDGE_EXTRACT_MAX_GLEANING": "0",
        "KNOWLEDGE_LLM_MAX_OUTPUT_TOKENS": "8192", "KNOWLEDGE_LLM_ENABLE_THINKING": "false",
        "KNOWLEDGE_LLM_TEMPERATURE": "0.7", "KNOWLEDGE_LLM_TOP_P": "0.8",
        "KNOWLEDGE_LLM_TOP_K": "20", "KNOWLEDGE_LLM_MIN_P": "0",
        "KNOWLEDGE_LLM_PRESENCE_PENALTY": "1.5",
        "KNOWLEDGE_LLM_MAX_ASYNC": "2", "KNOWLEDGE_LLM_TIMEOUT_SECONDS": "180",
        "KNOWLEDGE_LLM_SDK_TIMEOUT_SECONDS": "300",
        "KNOWLEDGE_EMBEDDING_TIMEOUT_SECONDS": "60", "KNOWLEDGE_EMBEDDING_SDK_TIMEOUT_SECONDS": "120",
        "KNOWLEDGE_QUERY_TIMEOUT_SECONDS": "90", "KNOWLEDGE_MUTATION_TIMEOUT_SECONDS": "900",
    }
    environment.update(graph_overrides)
    backend_overrides = {
        "KNOWLEDGE_QA_MAX_OUTPUT_TOKENS": "4096", "KNOWLEDGE_QA_TIMEOUT_SECONDS": "120",
        "KNOWLEDGE_HTTP_TIMEOUT_SECONDS": "100", "KNOWLEDGE_TOOL_TIMEOUT_SECONDS": "105",
        "KNOWLEDGE_QA_MODEL": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "KNOWLEDGE_QA_ENABLE_THINKING": "false", "KNOWLEDGE_QA_TEMPERATURE": "0.7",
        "KNOWLEDGE_QA_TOP_P": "0.8", "KNOWLEDGE_QA_TOP_K": "20",
        "KNOWLEDGE_QA_MIN_P": "0", "KNOWLEDGE_QA_PRESENCE_PENALTY": "0.2",
    }
    environment.update(backend_overrides)
    result = subprocess.run([
        "docker", "compose", "--env-file", os.devnull,
        "-f", "docker-compose.yml", "-f", "docker-compose.knowledge.yml",
        "config", "--format", "json",
    ], cwd=ROOT, env=environment, capture_output=True, text=True, check=True)
    services = json.loads(result.stdout)["services"]
    for name in ("knowledge-service", "knowledge-postgres"):
        if services[name].get("ports"):
            raise RuntimeError("knowledge services must not publish host ports")
    graph_env = services["knowledge-service"]["environment"]
    if password in graph_env["KNOWLEDGE_DATABASE_URL"]:
        raise RuntimeError("raw database password must not be interpolated into a URL")
    if graph_env["PGPASSWORD"] != password or graph_env["POSTGRES_PASSWORD"] != password:
        raise RuntimeError("database password lost special characters")
    if services["postgres"]["image"] != "postgres:15":
        raise RuntimeError("legacy database image changed")
    if str(services["backend"]["environment"]["WEB_CONCURRENCY"]) != "1":
        raise RuntimeError("embedded Chroma must remain single-worker")
    if "knowledge-service" in services["backend"].get("depends_on", {}):
        raise RuntimeError("optional knowledge outage must not block legacy startup")
    for key, value in graph_overrides.items():
        if str(graph_env.get(key)) != value:
            raise RuntimeError(f"knowledge-service is missing scoped configuration {key}")
    if any(key in graph_env for key in backend_overrides if key.startswith("KNOWLEDGE_QA_")):
        raise RuntimeError("QA-only settings must not be injected into the graph service")
    backend_env = services["backend"]["environment"]
    for key, value in {
        **backend_overrides,
        **{key: graph_overrides[key] for key in (
            "KNOWLEDGE_LLM_MODEL", "KNOWLEDGE_LLM_API_KEY", "KNOWLEDGE_LLM_BASE_URL",
            "KNOWLEDGE_LLM_ENABLE_THINKING",
            "KNOWLEDGE_LLM_TEMPERATURE", "KNOWLEDGE_LLM_TOP_P",
            "KNOWLEDGE_LLM_TOP_K", "KNOWLEDGE_LLM_MIN_P",
            "KNOWLEDGE_LLM_PRESENCE_PENALTY",
        )},
    }.items():
        if str(backend_env.get(key)) != value:
            raise RuntimeError(f"backend is missing scoped configuration {key}")
    if backend_env.get("LLM_EMBEDDING_MODEL") == graph_overrides["KNOWLEDGE_EMBEDDING_MODEL"]:
        raise RuntimeError("knowledge model must not replace the legacy embedding model")
    fallback_environment = {
        **environment,
        **{key: "" for key in graph_overrides if key.startswith((
            "KNOWLEDGE_LLM_", "KNOWLEDGE_EMBEDDING_",
        ))},
    }
    fallback = subprocess.run(result.args, cwd=ROOT, env=fallback_environment,
                              capture_output=True, text=True, check=True)
    fallback_services = json.loads(fallback.stdout)["services"]
    for name in ("backend", "knowledge-service"):
        if fallback_services[name]["environment"].get("KNOWLEDGE_LLM_ENABLE_THINKING"):
            raise RuntimeError("legacy fallback must not inject provider-specific thinking flags")
    print("knowledge Compose topology verified")


if __name__ == "__main__":
    main()
