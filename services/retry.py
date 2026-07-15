"""
Retry + per-call Timeout（Phase 8 P1-2 升级）

with_retry(fn, ..., timeout=...)：指数退避重试，带 jitter，per-call asyncio.wait_for 超时。

错误分类：
  Transient（暂时性）  : RateLimitError / APITimeoutError / APIConnectionError → 重试
  Server Error（服务端）: APIStatusError 5xx → 重试
  Client Error（客户端）: APIStatusError 4xx → 不重试，直接抛出
  asyncio.TimeoutError : per-call 超时 → 重试（用户配置 A）
  其他异常             : 不重试，直接抛出

为什么要业务层 timeout（而不是只信 SDK timeout）：
  - SDK timeout 只管单次 HTTP 请求，管不到 ChromaDB / BM25 / 本地计算
  - SDK timeout 管不到 retry 之间的 backoff，无法控总预算
  - SDK timeout 流式场景下只管 chunk 间隔，不管总时长
  - 业务层 with_retry(timeout=) 用 asyncio.wait_for 包整个 coroutine，
    超时后 cancel 整个调用链，资源不漏

为什么 fn 必须是 callable 而不是 coroutine？
  coroutine 只能 await 一次，重试需要每次创建新 coroutine。
  正确写法：with_retry(lambda: some_async_func(args), timeout=30)
"""
import asyncio
import logging
import random

from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

logger = logging.getLogger(__name__)

# 需要重试的暂时性错误
_TRANSIENT = (RateLimitError, APITimeoutError, APIConnectionError)


class RetryExhausted(Exception):
    """所有重试均失败后抛出，由调用方决定是 fallback 还是报错。"""
    pass


async def with_retry(
    fn,
    max_retries: int = 3,
    base_delay: float = 1.0,
    timeout: float | None = None,
):
    """指数退避重试（带 jitter）+ per-call timeout。

    重试间隔：base_delay * 2^i + random(0, 0.5) 秒
      i=0: ~1.0s  i=1: ~2.0s  i=2: ~4.0s

    Args:
        fn: 每次调用返回新 coroutine 的 callable，不能传 coroutine 对象本身。
        max_retries: 最多重试次数（不含首次调用）。
        base_delay: 首次重试前等待秒数。
        timeout: 单次调用最长允许秒数。None 表示不限（保持旧行为）。
                 触发时 cancel 整个 coroutine（cooperative cancel）。
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            if timeout is not None:
                return await asyncio.wait_for(fn(), timeout=timeout)
            return await fn()

        except asyncio.TimeoutError:
            # per-call 超时：cancel 已传到底层 coroutine
            last_exc = TimeoutError(f"call exceeded {timeout}s")
            logger.warning(
                "[retry] attempt %d/%d timed out (>%.1fs)",
                attempt + 1, max_retries + 1, timeout,
            )

        except _TRANSIENT as exc:
            last_exc = exc

        except APIStatusError as exc:
            if exc.status_code >= 500:   # 服务端错误，可重试
                last_exc = exc
            else:                         # 客户端错误（4xx），不重试
                raise

        except Exception:
            raise  # 非 API 错误，不重试

        # ── 是否还有重试机会 ──────────────────────────────
        if attempt == max_retries:
            break

        delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
        logger.warning(
            "[retry] attempt %d/%d failed (%s). retrying in %.1fs",
            attempt + 1, max_retries + 1, type(last_exc).__name__, delay,
        )
        await asyncio.sleep(delay)

    raise RetryExhausted(f"重试 {max_retries} 次均失败：{last_exc}") from last_exc
