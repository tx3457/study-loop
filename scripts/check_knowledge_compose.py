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
    }
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
    print("knowledge Compose topology verified")


if __name__ == "__main__":
    main()
