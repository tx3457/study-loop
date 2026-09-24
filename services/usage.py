"""Model usage as providers report it, grouped by the feature that caused it.

A self-hosted learner pays for every model call and previously had no way to
see what a quiz or an Agent run cost. This ledger records the token counts the
provider returns and attributes them to a product feature via the request path,
so no call site needs to pass a label.

Boundaries, stated where the numbers are served:
- Token counts only. Prices differ by provider and model and are not returned,
  so no currency amount is estimated.
- A response without usage (streaming, or a provider that omits it) counts as a
  call whose tokens are unknown -- never as zero tokens.
- Chat and embedding tokens stay separate: they differ in price by orders of
  magnitude, so one merged total would mislead a cost judgement.
- In-process and since process start; the separate knowledge-base service runs
  its own model calls and is not included.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from services.request_context import current_request_path


logger = logging.getLogger(__name__)

# (path prefix, stable key, label). A prefix matches itself or a sub-path, so
# "/session" does not swallow an unrelated "/sessions-export".
_OPERATIONS: tuple[tuple[str, str, str], ...] = (
    ("/agent/autonomous", "autonomous", "自主 Agent"),
    ("/agent/adaptive", "adaptive", "自适应辅导"),
    ("/agent/tutor", "tutor_lab", "辅导实验"),
    ("/session", "quiz", "答题练习（含批改）"),
    ("/wrong-questions", "wrong_practice", "错题重练"),
    ("/learning-path", "learning_path", "学习路径"),
    ("/learning-paths", "learning_path", "学习路径"),
    ("/documents", "documents", "文档入库"),
    ("/eval", "evaluation", "评测"),
    ("/chat", "direct_api", "无界面接口"),
    ("/generate", "direct_api", "无界面接口"),
    ("/agent/run", "direct_api", "无界面接口"),
    ("/agent/stream", "direct_api", "无界面接口"),
)
_OTHER = ("other", "其他")
_BACKGROUND = ("background", "后台任务")


def operation_for(path: str | None) -> tuple[str, str]:
    """Map a request path to a (key, label) pair; no path means background work."""
    if not path:
        return _BACKGROUND
    for prefix, key, label in _OPERATIONS:
        if path == prefix or path.startswith(f"{prefix}/"):
            return key, label
    return _OTHER


def _count(usage, name: str) -> int | None:
    value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


@dataclass
class _Bucket:
    label: str
    calls: int = 0
    calls_without_usage: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0

    def as_dict(self, key: str) -> dict:
        return {
            "key": key,
            "label": self.label,
            "calls": self.calls,
            "calls_without_usage": self.calls_without_usage,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "chat_tokens": self.prompt_tokens + self.completion_tokens,
            "embedding_tokens": self.embedding_tokens,
        }


class UsageLedger:
    def __init__(self, clock=time.time) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._buckets: dict[str, _Bucket] = {}
        self._started_at = clock()

    def record(self, kind: str, response) -> None:
        """Record one provider response. Never raises: accounting must not fail a call."""
        try:
            self._record(kind, response)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("usage accounting skipped: error_type=%s", type(exc).__name__)

    def _record(self, kind: str, response) -> None:
        key, label = operation_for(current_request_path())
        usage = getattr(response, "usage", None)
        with self._lock:
            bucket = self._buckets.setdefault(key, _Bucket(label=label))
            bucket.calls += 1
            if usage is None:
                bucket.calls_without_usage += 1
                return
            if kind == "embedding":
                tokens = _count(usage, "total_tokens")
                if tokens is None:
                    tokens = _count(usage, "prompt_tokens")
                if tokens is None:
                    bucket.calls_without_usage += 1
                    return
                bucket.embedding_tokens += tokens
                return
            prompt = _count(usage, "prompt_tokens")
            completion = _count(usage, "completion_tokens")
            if prompt is None and completion is None:
                bucket.calls_without_usage += 1
                return
            bucket.prompt_tokens += prompt or 0
            bucket.completion_tokens += completion or 0

    def snapshot(self) -> dict:
        with self._lock:
            operations = [bucket.as_dict(key) for key, bucket in self._buckets.items()]
            started_at = self._started_at
        operations.sort(
            key=lambda row: (row["chat_tokens"], row["embedding_tokens"], row["calls"]),
            reverse=True,
        )
        totals = {
            name: sum(row[name] for row in operations)
            for name in (
                "calls", "calls_without_usage", "prompt_tokens",
                "completion_tokens", "chat_tokens", "embedding_tokens",
            )
        }
        return {
            "since": datetime.fromtimestamp(started_at, timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "unit": "tokens",
            "currency_estimated": False,
            "excludes": ["knowledge_service"],
            "operations": operations,
            "totals": totals,
        }

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()
            self._started_at = self._clock()


usage_ledger = UsageLedger()
