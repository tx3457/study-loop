"""Configuration-level tests for optional tracing integrations."""

from pathlib import Path

import pytest

from services import tracing


_LANGSMITH_FLAGS = (
    "LANGSMITH_TRACING_V2",
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_TRACING",
    "LANGCHAIN_TRACING",
)


def _clear_langsmith_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _LANGSMITH_FLAGS:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    "value",
    [None, "", "0", "false", "FALSE", " no ", "off", "unexpected"],
)
def test_langsmith_is_fail_closed_for_false_or_unknown_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str | None,
) -> None:
    monkeypatch.setattr(tracing, "_LANGSMITH_AVAILABLE", True)
    _clear_langsmith_flags(monkeypatch)
    if value is None:
        monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    else:
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", value)

    assert tracing._langsmith_enabled() is False


def test_langsmith_accepts_exact_lowercase_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tracing, "_LANGSMITH_AVAILABLE", True)
    _clear_langsmith_flags(monkeypatch)
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")

    assert tracing._langsmith_enabled() is True


@pytest.mark.parametrize("value", ["1", "TRUE", "yes", " true "])
def test_langsmith_rejects_values_the_sdk_does_not_treat_as_true(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setattr(tracing, "_LANGSMITH_AVAILABLE", True)
    _clear_langsmith_flags(monkeypatch)
    monkeypatch.setenv("LANGSMITH_TRACING", value)

    assert tracing._langsmith_enabled() is False


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({"LANGSMITH_TRACING_V2": "false", "LANGCHAIN_TRACING_V2": "true"}, False),
        ({"LANGCHAIN_TRACING_V2": "false", "LANGSMITH_TRACING": "true"}, False),
        ({"LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING": "true"}, False),
        ({"LANGSMITH_TRACING_V2": "", "LANGCHAIN_TRACING_V2": "true"}, True),
        ({"LANGCHAIN_TRACING_V2": "   ", "LANGSMITH_TRACING": "true"}, True),
        ({"LANGCHAIN_TRACING": "true"}, True),
    ],
)
def test_langsmith_flag_precedence_matches_the_pinned_sdk(
    monkeypatch: pytest.MonkeyPatch,
    values: dict[str, str],
    expected: bool,
) -> None:
    monkeypatch.setattr(tracing, "_LANGSMITH_AVAILABLE", True)
    _clear_langsmith_flags(monkeypatch)
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    assert tracing._langsmith_enabled() is expected


def test_langsmith_stays_disabled_when_dependency_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tracing, "_LANGSMITH_AVAILABLE", False)
    _clear_langsmith_flags(monkeypatch)
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")

    assert tracing._langsmith_enabled() is False


@pytest.mark.parametrize("blank_key", ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"])
def test_langfuse_requires_two_non_blank_keys(
    monkeypatch: pytest.MonkeyPatch,
    blank_key: str,
) -> None:
    monkeypatch.setattr(tracing, "_LANGFUSE_AVAILABLE", True)
    monkeypatch.delenv("LANGFUSE_TRACING_ENABLED", raising=False)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv(blank_key, "   ")

    assert tracing._langfuse_enabled() is False


def test_langfuse_is_enabled_with_both_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tracing, "_LANGFUSE_AVAILABLE", True)
    monkeypatch.delenv("LANGFUSE_TRACING_ENABLED", raising=False)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    assert tracing._langfuse_enabled() is True


def test_langfuse_explicit_disable_overrides_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tracing, "_LANGFUSE_AVAILABLE", True)
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    assert tracing._langfuse_enabled() is False
    assert tracing.get_langfuse_callback() is None


def test_traceable_is_identity_when_both_integrations_are_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tracing, "_LANGSMITH_AVAILABLE", True)
    monkeypatch.setattr(tracing, "_LANGFUSE_AVAILABLE", True)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")

    def operation() -> str:
        return "ok"

    assert tracing.traceable(name="operation")(operation) is operation


def test_traceable_uses_langsmith_decorator_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    def fake_traceable(**kwargs):
        calls.append(kwargs)

        def decorate(fn):
            return fn

        return decorate

    monkeypatch.setattr(tracing, "_LANGSMITH_AVAILABLE", True)
    monkeypatch.setattr(tracing, "_LANGFUSE_AVAILABLE", False)
    monkeypatch.setattr(tracing, "_ls_traceable", fake_traceable)
    _clear_langsmith_flags(monkeypatch)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")

    def operation() -> str:
        return "ok"

    tracing.traceable(
        name="operation",
        run_type="tool",
        metadata={"source": "test"},
    )(operation)

    assert calls == [
        {
            "name": "operation",
            "run_type": "tool",
            "metadata": {"source": "test"},
        }
    ]


def test_env_example_uses_langfuse_v4_base_url() -> None:
    env_example = Path(__file__).resolve().parents[1] / ".env.example"
    contents = env_example.read_text(encoding="utf-8")

    lines = contents.splitlines()
    assert "LANGSMITH_TRACING=false" in lines
    assert "LANGFUSE_BASE_URL=" in lines
    assert "LANGFUSE_HOST=" not in lines
