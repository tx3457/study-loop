#!/usr/bin/env python3
"""Validate the security-sensitive defaults in docker-compose.yml."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC_PASSWORD = "synthetic@pass:word/with#chars"


class ComposeSecurityError(RuntimeError):
    pass


def _compose(
    *args: str,
    password: str | None,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["ENV_FILE"] = ".env.example"
    if password is None:
        env.pop("POSTGRES_PASSWORD", None)
    else:
        env["POSTGRES_PASSWORD"] = password
    return subprocess.run(
        ["docker", "compose", "--env-file", os.devnull, *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _assert_invalid_password_fails(password: str | None, label: str) -> None:
    result = _compose("config", "--quiet", password=password)
    if result.returncode == 0:
        raise ComposeSecurityError(
            f"docker compose config accepted a {label} POSTGRES_PASSWORD"
        )


def _load_config() -> dict:
    result = _compose("config", "--format", "json", password=SYNTHETIC_PASSWORD)
    if result.returncode != 0:
        raise ComposeSecurityError(
            f"docker compose config failed with a supplied password: {result.stderr.strip()}"
        )
    return json.loads(result.stdout)


def _assert_ports(service: dict, expected: list[tuple[int, str, str]]) -> None:
    actual = [
        (
            int(port["target"]),
            str(port["published"]),
            str(port.get("host_ip", "")),
        )
        for port in service.get("ports", [])
    ]
    if actual != expected:
        raise ComposeSecurityError(
            f"unexpected published ports for {service.get('name', 'service')}: {actual}"
        )


def _assert_safe_config(config: dict) -> None:
    services = config["services"]
    _assert_ports(services["postgres"], [])
    _assert_ports(services["backend"], [(8001, "8001", "127.0.0.1")])
    _assert_ports(services["backend-volume-init"], [])
    _assert_ports(services["frontend"], [(80, "4001", "127.0.0.1")])

    backend_env = services["backend"].get("environment", {})
    database_url = str(backend_env.get("DATABASE_URL", ""))
    if SYNTHETIC_PASSWORD in database_url:
        raise ComposeSecurityError("DATABASE_URL contains the raw PostgreSQL password")
    if backend_env.get("PGPASSWORD") != SYNTHETIC_PASSWORD:
        raise ComposeSecurityError("backend PGPASSWORD does not preserve special characters")

    health_test = services["backend"].get("healthcheck", {}).get("test", [])
    if "/health/live" not in " ".join(str(part) for part in health_test):
        raise ComposeSecurityError("backend healthcheck must use /health/live")

    backend_dependency = services["frontend"].get("depends_on", {}).get("backend", {})
    if backend_dependency.get("condition") != "service_healthy":
        raise ComposeSecurityError("frontend must wait for a healthy backend")

    volume_init = services["backend-volume-init"]
    if str(volume_init.get("user")) not in {"0", "0:0"}:
        raise ComposeSecurityError("backend volume init must run as root")
    init_command = " ".join(str(part) for part in volume_init.get("command", []))
    if "chown -R 10001:10001 /app/chroma_db" not in init_command:
        raise ComposeSecurityError("backend volume init must repair Chroma ownership")
    init_dependency = services["backend"].get("depends_on", {}).get(
        "backend-volume-init", {}
    )
    if init_dependency.get("condition") != "service_completed_successfully":
        raise ComposeSecurityError("backend must wait for Chroma ownership repair")


def main() -> int:
    try:
        _assert_invalid_password_fails(None, "missing")
        _assert_invalid_password_fails("", "blank")
        _assert_safe_config(_load_config())
    except (ComposeSecurityError, json.JSONDecodeError, KeyError, TypeError) as exc:
        print(f"compose security validation failed: {exc}", file=sys.stderr)
        return 1
    print("compose security defaults verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
