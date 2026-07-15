"""
Query Rewriting:HyDE + Multi-query(Phase 9 P0-5)

借鉴 llama_index 两个独立技术(我们组合使用):
  - HyDE(Hypothetical Document Embeddings, arXiv:2212.10496)
    :indices/query/query_transform/base.py:96-120
    用 LLM 生成"假设性答案",再用这段答案做向量召回。
    适用场景:用户 query 太短/太抽象,直接向量召回精度低。
  - Multi-query / QueryFusion
    :retrievers/fusion_retriever.py:15-21
    用 LLM 生成 N 个 query 变体,分别检索后用 RRF 合并。
    适用场景:用户 query 表达单一,容易漏召回同义/近义文档。

两者关系:
  HyDE 是 "改写 query → 加密"(语义扩展)
  Multi-query 是 "拆 query → 多角度召回"(覆盖扩展)
  本模块同时提供两个函数,调用方按场景选择或组合。

env 开关:
  QUERY_REWRITE_ENABLED=false   全关
  HYDE_ENABLED=false             单独关 HyDE
  MULTIQUERY_ENABLED=false       单独关 multi-query
  MULTIQUERY_N=3                 multi-query 生成几个变体(默认 3)
"""
import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)
load_dotenv(Path(__file__).parent.parent / ".env")

_client = AsyncOpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL"))
_model = os.getenv("LLM_MODEL")


# ── env 开关 ─────────────────────────────────────────────────────────────────
def query_rewrite_enabled() -> bool:
    return os.getenv("QUERY_REWRITE_ENABLED", "true").lower() in ("1", "true", "yes")


def hyde_enabled() -> bool:
    return query_rewrite_enabled() and os.getenv("HYDE_ENABLED", "true").lower() in ("1", "true", "yes")


def multiquery_enabled() -> bool:
    return query_rewrite_enabled() and os.getenv("MULTIQUERY_ENABLED", "true").lower() in ("1", "true", "yes")


def multiquery_n() -> int:
    try:
        return max(1, int(os.getenv("MULTIQUERY_N", "3")))
    except ValueError:
        return 3


# ── HyDE ────────────────────────────────────────────────────────────────────
HYDE_SYSTEM_PROMPT = (
    "你是 Hypothetical Document Embeddings(HyDE)助手。"
    "给定用户的问题,请用 2-3 句话写出一个**假设性的、可能存在于知识库中的答案段落**。"
    "答案应该尽量像教科书或文档原文,使用专业术语,而不是口语化解释。"
    "若问题过于宽泛,生成最有可能存在的标准答案;不要追问、不要拒绝、不要加任何元说明。"
)


async def hyde_rewrite(query: str, *, client: Optional[AsyncOpenAI] = None) -> str:
    """HyDE:用 LLM 生成"假设性答案"段落,作为后续向量召回的 query embedding 来源。

    Args:
        query: 原始用户 query
        client: 可注入测试用 mock client;不传则用模块 _client

    Returns:
        生成的假设性答案文本(失败时降级返回原 query)
    """
    if not query.strip():
        return query

    use_client = client or _client
    try:
        resp = await use_client.chat.completions.create(
            model=_model,
            max_tokens=256,
            temperature=0.3,
            messages=[
                {"role": "system", "content": HYDE_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
        )
        hypothesis = (resp.choices[0].message.content or "").strip()
        if not hypothesis:
            logger.warning("[query_rewriter] HyDE empty response, fallback to original query")
            return query
        logger.info(f"[query_rewriter] HyDE expanded: {len(query)} → {len(hypothesis)} chars")
        return hypothesis
    except Exception as e:
        logger.warning(f"[query_rewriter] HyDE failed, fallback to original: {e}")
        return query


# ── Multi-query ──────────────────────────────────────────────────────────────
MULTIQUERY_SYSTEM_PROMPT = (
    "你是检索 query 改写助手。"
    "给定一个用户 query,生成 {n} 个**不同表达、但意图相同**的查询变体,"
    "用于覆盖召回中可能因措辞差异漏掉的文档。\n\n"
    "要求:\n"
    "- 每行一个变体,不带编号、不带其他符号\n"
    "- 不要复述原 query\n"
    "- 不要漂移到不相关主题\n"
    "- 优先用同义词、术语替换、不同角度提问"
)


async def multi_query_rewrite(
    query: str,
    n: Optional[int] = None,
    *,
    client: Optional[AsyncOpenAI] = None,
) -> list[str]:
    """Multi-query:生成 N 个 query 变体,与原 query 一起用于多路并行检索后 RRF 合并。

    Returns:
        包含原 query 的列表 [original, variant_1, variant_2, ...]
        失败时只返回 [original]
    """
    if not query.strip():
        return [query]

    n_variants = n if n is not None else multiquery_n()
    use_client = client or _client

    try:
        resp = await use_client.chat.completions.create(
            model=_model,
            max_tokens=512,
            temperature=0.5,
            messages=[
                {"role": "system", "content": MULTIQUERY_SYSTEM_PROMPT.format(n=n_variants)},
                {"role": "user", "content": query},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        variants = [
            line.strip().lstrip("-*•0123456789.) ").strip()
            for line in raw.splitlines()
            if line.strip()
        ]
        variants = [v for v in variants if v and v != query][:n_variants]
        logger.info(f"[query_rewriter] multi-query generated {len(variants)} variants")
        return [query] + variants
    except Exception as e:
        logger.warning(f"[query_rewriter] multi-query failed, fallback to [original]: {e}")
        return [query]


# ── RRF 合并(给 multi-query 检索结果用)─────────────────────────────────────
def rrf_merge_ranked_lists(
    ranked_lists: list[list[tuple[str, str]]],
    top_k: int = 5,
    k: int = 60,
) -> list[tuple[str, str, float]]:
    """RRF(Reciprocal Rank Fusion)合并多个排序后的检索结果。

    Args:
        ranked_lists: 多个 (doc_id, doc_text) 列表,每个列表已按相关性降序
        top_k: 合并后返回前 K 条
        k: RRF 常数(60 是工业默认)

    Returns:
        [(doc_id, doc_text, rrf_score), ...] 按 rrf_score 降序
    """
    rrf_scores: dict[str, float] = {}
    id_to_doc: dict[str, str] = {}

    for ranked in ranked_lists:
        for rank, (doc_id, doc_text) in enumerate(ranked):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            id_to_doc[doc_id] = doc_text

    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
    return [(d_id, id_to_doc[d_id], rrf_scores[d_id]) for d_id in sorted_ids]
