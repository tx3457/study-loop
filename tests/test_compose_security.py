"""Unit tests for Compose security and health-contract validation."""

from copy import deepcopy
from pathlib import Path

import pytest

from scripts.check_compose_security import (
    SYNTHETIC_PASSWORD,
    ComposeSecurityError,
    _assert_safe_config,
)


def _safe_config() -> dict:
    return {
        "services": {
            "postgres": {"name": "postgres", "ports": []},
            "backend": {
                "name": "backend",
                "ports": [
                    {"target": 8001, "published": "8001", "host_ip": "127.0.0.1"}
                ],
                "environment": {
                    "DATABASE_URL": "postgresql://studyloop@postgres:5432/studyloop",
                    "PGPASSWORD": SYNTHETIC_PASSWORD,
                    "WEB_CONCURRENCY": "1",
                },
                "healthcheck": {
                    "test": ["CMD", "python", "-c", "GET /health/ready"]
                },
                "depends_on": {
                    "backend-volume-init": {
                        "condition": "service_completed_successfully"
                    }
                },
            },
            "backend-volume-init": {
                "name": "backend-volume-init",
                "ports": [],
                "user": "0:0",
                "command": [
                    "sh",
                    "-c",
                    "chown -R 10001:10001 /app/chroma_db",
                ],
            },
            "frontend": {
                "name": "frontend",
                "ports": [
                    {"target": 80, "published": "4001", "host_ip": "127.0.0.1"}
                ],
                "depends_on": {"backend": {"condition": "service_healthy"}},
            },
        }
    }


def test_safe_compose_contract_is_accepted() -> None:
    _assert_safe_config(_safe_config())


def test_backend_healthcheck_is_required() -> None:
    config = deepcopy(_safe_config())
    config["services"]["backend"].pop("healthcheck")

    with pytest.raises(ComposeSecurityError, match="healthcheck"):
        _assert_safe_config(config)


def test_backend_healthcheck_cannot_use_liveness_as_readiness() -> None:
    config = deepcopy(_safe_config())
    config["services"]["backend"]["healthcheck"]["test"][-1] = (
        "GET /health/live"
    )

    with pytest.raises(ComposeSecurityError, match="storage readiness"):
        _assert_safe_config(config)


def test_frontend_must_wait_for_backend_health() -> None:
    config = deepcopy(_safe_config())
    config["services"]["frontend"]["depends_on"]["backend"][
        "condition"
    ] = "service_started"

    with pytest.raises(ComposeSecurityError, match="healthy backend"):
        _assert_safe_config(config)


def test_backend_must_wait_for_volume_ownership_repair() -> None:
    config = deepcopy(_safe_config())
    config["services"]["backend"]["depends_on"].pop("backend-volume-init")

    with pytest.raises(ComposeSecurityError, match="ownership repair"):
        _assert_safe_config(config)


def test_backend_must_force_single_worker_for_embedded_chroma() -> None:
    config = deepcopy(_safe_config())
    config["services"]["backend"]["environment"]["WEB_CONCURRENCY"] = "4"

    with pytest.raises(ComposeSecurityError, match="one worker"):
        _assert_safe_config(config)


def test_backend_image_command_pins_one_worker() -> None:
    dockerfile = Path(__file__).resolve().parents[1] / "Dockerfile"
    cmd_line = next(
        line for line in dockerfile.read_text(encoding="utf-8").splitlines()
        if line.startswith("CMD [")
    )

    assert '"--workers", "1"' in cmd_line


def test_backend_image_healthcheck_uses_storage_readiness() -> None:
    dockerfile = Path(__file__).resolve().parents[1] / "Dockerfile"
    healthcheck = next(
        line for line in dockerfile.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("CMD python -c")
    )

    assert "/health/ready" in healthcheck


def test_frontend_docker_context_excludes_local_environment_files() -> None:
    dockerignore = Path(__file__).resolve().parents[1] / "frontend" / ".dockerignore"
    entries = {
        line.strip()
        for line in dockerignore.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert {".env", ".env.*", "*.local"} <= entries
