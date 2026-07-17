"""
Cross-Encoder Reranker(第二阶段精排)

借鉴 kotaemon/libs/kotaemon/kotaemon/rerankings/cohere.py:35-66 的接口设计,
但用本地 BGE-reranker-v2-m3(免 API key)替代 Cohere API。

设计原则:
  1. 懒加载:首次调用才下载模型,启动不阻塞(~2.27GB 一次)
  2. 单例缓存:lru_cache 装饰 _get_reranker(),进程内复用
  3. env 开关:RERANKER_ENABLED=false 时跳过(ablation 对照用)
  4. GPU 自适应:RERANKER_DEVICE env 可显式指定;未指定则 torch.cuda 自动检测
                 GPU(fp32)推理比 CPU 快约 10x,16GB 显存绰绰有余(模型仅占 ~2.5GB)
  5. 失败安全:模型加载或推理失败 → raise RerankerUnavailable,上游决定降级

为什么是 cross-encoder 不是 RRF?
  - RRF 只融合 rank 而忽略 query-document 语义相似度
  - cross-encoder 真正读 (query, doc) 对,给出 0-1 相关性分数,精度显著高
  - 工业实践:bi-encoder/BM25 召回 top-N → cross-encoder 精排 top-K(N>>K)
"""
import asyncio
import logging
import os
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)


class RerankerUnavailable(Exception):
    """模型未加载或推理失败,上游应降级到 RRF-only"""


def reranker_enabled() -> bool:
    """env 控制开关,ablation 实验时设 false"""
    return os.getenv("RERANKER_ENABLED", "false").lower() in ("1", "true", "yes")


def _reranker_model_name() -> str:
    return os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")


def _reranker_device() -> str:
    """决定 CrossEncoder 跑在哪个设备。

    优先级:
      1. RERANKER_DEVICE env(显式指定,如 'cuda' / 'cuda:0' / 'cpu' / 'mps')
      2. torch.cuda 自动检测 → 有 GPU 用 cuda,否则 cpu
      3. 兜底 'cpu'(torch 未装或检测异常)
    """
    forced = os.getenv("RERANKER_DEVICE")
    if forced:
        return forced
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        # Apple Silicon
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception as e:
        logger.debug(f"[reranker] torch device probe failed: {e}, fallback cpu")
    return "cpu"


@lru_cache(maxsize=1)
def _get_reranker():
    """懒加载 + 进程内单例。首次调用会下载模型(~2.27GB)。"""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        raise RerankerUnavailable(f"sentence-transformers not installed: {e}")

    model_name = _reranker_model_name()
    device = _reranker_device()
    logger.info(
        f"[reranker] loading CrossEncoder model: {model_name} on device={device} "
        "(first run may download ~2.27GB)"
    )
    try:
        return CrossEncoder(model_name, device=device)
    except Exception as e:
        raise RerankerUnavailable(f"failed to load {model_name} on {device}: {e}")


async def rerank_docs(
    query: str,
    docs: list[str],
    ids: Optional[list[str]] = None,
    top_k: int = 5,
) -> list[tuple[str, float, Optional[str]]]:
    """对 (query, doc) 对批量打分并按分数降序返回 top_k。

    Args:
        query: 用户原始查询
        docs:  RRF 召回的候选 chunks(建议传 top_k * 2~3 倍数量给精排做选择)
        ids:   与 docs 对齐的 chunk_id 列表(可选,用于审计 / 回溯)
        top_k: 返回精排后的前 K 条

    Returns:
        [(doc, score, id_or_None), ...],按 score 降序

    Raises:
        RerankerUnavailable: 模型加载或推理失败,上游应降级到 RRF-only
    """
    if not docs:
        return []
    if len(docs) <= top_k:
        # 候选不够多,精排意义有限,但仍打分以保证输出格式一致
        pass

    model = _get_reranker()
    pairs = [(query, d) for d in docs]

    try:
        # CrossEncoder.predict 是 CPU/GPU 同步推理,放线程池避免阻塞 event loop
        scores = await asyncio.to_thread(model.predict, pairs)
    except Exception as e:
        raise RerankerUnavailable(f"predict failed: {e}")

    # scores 可能是 numpy.ndarray 或 list[float]
    scored = [(docs[i], float(scores[i]), ids[i] if ids else None) for i in range(len(docs))]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
