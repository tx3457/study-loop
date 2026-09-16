from typing import Optional

from openai import AsyncOpenAI

from services.provider_config import (
    PROVIDER_REQUEST_DEADLINE_SECONDS,
    build_managed_async_openai,
    load_provider_configs,
    run_with_provider_deadline,
)
from services.retry import with_retry

_provider_configs = load_provider_configs()
_chat_config = _provider_configs["chat"]
_structured_config = _provider_configs["structured"]

api_key = _chat_config.api_key
base_url = _chat_config.base_url
model = _chat_config.model

# 模块级共享 client：统一 LLM 入口默认用它；调用方可注入自己的 client（测试 mock / 多租户）。
_client = build_managed_async_openai(_chat_config)

# ── 结构化输出（json_schema）供应商分离 ──────────────────────────────────────
# DeepSeek 等厂商不支持 OpenAI 的 json_schema response_format（实测 400:
# "This response_format type is unavailable now"）。chat 切换厂商时，所有
# beta.chat.completions.parse 调用仍走 STRUCTURED_*（未配置则跟随 LLM_*）。
# SiliconFlow 高峰期 TLS 建连可达 4-6s，超过 openai SDK 默认 connect=5.0s →
# 全部请求在握手阶段就 APITimeoutError。放宽 connect 超时。
structured_model = _structured_config.model
structured_client = build_managed_async_openai(_structured_config)


# ═══════════════════════════════════════════════════════════════════════════
# 统一 LLM 入口：为经本模块发起的调用应用退避重试和总时间预算。
#
# 设计：
#   - llm_chat  → chat.completions.create（普通对话 / function calling）
#   - llm_parse → beta.chat.completions.parse（结构化输出）
#   - 两者都 client 注入：默认用模块级 _client，传入 client 则用传入的，
#     保留 adaptive / routers 既有的 mock 能力。
#   - with_retry 要求 fn 是「无参 callable 返回新 coroutine」，故用 lambda 包裹。
#   - max_retries 默认 2；调用方可传 max_retries=0 禁用重试。
# ═══════════════════════════════════════════════════════════════════════════
async def llm_chat(
    messages: list,
    *,
    client: Optional[AsyncOpenAI] = None,
    max_retries: int = 2,
    base_delay: float = 1.0,
    timeout: Optional[float] = None,
    total_timeout: Optional[float] = PROVIDER_REQUEST_DEADLINE_SECONDS,
    **kwargs,
):
    """统一 chat.completions.create 入口，带重试和端到端时间预算。

    透传 model 之外的全部 kwargs（tools / tool_choice / temperature / max_tokens / stream...）。
    model 默认取模块级配置，可被 kwargs 覆盖。
    """
    use_client = client or _client
    kwargs.setdefault("model", model)
    return await run_with_provider_deadline(
        lambda: with_retry(
            lambda: use_client.chat.completions.create(messages=messages, **kwargs),
            max_retries=max_retries,
            base_delay=base_delay,
            timeout=timeout,
        ),
        total_timeout=total_timeout,
    )


async def llm_parse(
    messages: list,
    response_format,
    *,
    client: Optional[AsyncOpenAI] = None,
    max_retries: int = 2,
    base_delay: float = 1.0,
    timeout: Optional[float] = None,
    total_timeout: Optional[float] = PROVIDER_REQUEST_DEADLINE_SECONDS,
    **kwargs,
):
    """统一结构化输出入口，带重试和端到端时间预算。

    默认走 structured_client / structured_model：chat 与结构化输出可分属不同厂商。
    """
    use_client = client or structured_client
    kwargs.setdefault("model", structured_model)
    return await run_with_provider_deadline(
        lambda: with_retry(
            lambda: use_client.beta.chat.completions.parse(
                messages=messages, response_format=response_format, **kwargs
            ),
            max_retries=max_retries,
            base_delay=base_delay,
            timeout=timeout,
        ),
        total_timeout=total_timeout,
    )


async def chat_stream(message: str):
    response = await llm_chat(
        [{"role": "user", "content": message}],
        stream=True,
    )

    async def iter_events():
        async for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield f"data: {delta}\n\n"

    return iter_events()


async def chat_history(messages: list):
    response = await llm_chat(messages)
    return response.choices[0].message.content
