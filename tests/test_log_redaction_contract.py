"""Static guard against reintroducing raw exception text into runtime logs."""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOTS = (
    REPO_ROOT / "agents",
    REPO_ROOT / "routers",
    REPO_ROOT / "services",
)


def _runtime_python_files() -> list[Path]:
    files = [REPO_ROOT / "main.py"]
    for root in RUNTIME_ROOTS:
        files.extend(root.rglob("*.py"))
    return sorted(files)


def _logger_method(call: ast.Call) -> str | None:
    func = call.func
    if not isinstance(func, ast.Attribute):
        return None
    if not isinstance(func.value, ast.Name) or func.value.id != "logger":
        return None
    return func.attr


def _contains_name(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(child, ast.Name) and child.id == name
        for child in ast.walk(node)
    )


def test_runtime_logs_never_serialize_caught_exception_text_or_tracebacks() -> None:
    violations: list[str] = []
    for path in _runtime_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for handler in (
            node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)
        ):
            exception_name = handler.name
            for node in ast.walk(handler):
                if not isinstance(node, ast.Call):
                    continue
                method = _logger_method(node)
                if method is None:
                    continue
                relative = path.relative_to(REPO_ROOT)
                location = f"{relative}:{node.lineno}"
                if method == "exception":
                    violations.append(f"{location} uses logger.exception")
                if any(
                    keyword.arg == "exc_info"
                    and not (
                        isinstance(keyword.value, ast.Constant)
                        and keyword.value.value is False
                    )
                    for keyword in node.keywords
                ):
                    violations.append(f"{location} enables exc_info")
                if not exception_name:
                    continue
                if node.args and isinstance(node.args[0], ast.JoinedStr):
                    if _contains_name(node.args[0], exception_name):
                        violations.append(
                            f"{location} interpolates caught exception text"
                        )
                for argument in node.args[1:]:
                    if (
                        isinstance(argument, ast.Name)
                        and argument.id == exception_name
                    ):
                        violations.append(
                            f"{location} passes caught exception as a log argument"
                        )
                    if (
                        isinstance(argument, ast.Call)
                        and isinstance(argument.func, ast.Name)
                        and argument.func.id in {"str", "repr"}
                        and argument.args
                        and isinstance(argument.args[0], ast.Name)
                        and argument.args[0].id == exception_name
                    ):
                        violations.append(
                            f"{location} stringifies caught exception in a log"
                        )

    assert violations == []


def test_documented_servers_disable_raw_url_access_logs() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    development = (REPO_ROOT / "docs" / "DEVELOPMENT.md").read_text(
        encoding="utf-8"
    )
    nginx = (REPO_ROOT / "frontend" / "nginx.conf").read_text(encoding="utf-8")
    api_location = nginx.split("location /api/ {", maxsplit=1)[1].split(
        "}", maxsplit=1
    )[0]

    assert '"--no-access-log"' in dockerfile
    assert (
        "uvicorn main:app --reload --port 8001 --workers 1 --no-access-log"
        in development
    )
    assert "access_log off;" in api_location
