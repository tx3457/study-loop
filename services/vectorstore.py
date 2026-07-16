import chromadb
from openai import AsyncOpenAI
import logging
import os
import time
import uuid
from pathlib import Path
from dotenv import load_dotenv
import asyncio
from chromadb.errors import NotFoundError
from services.retry import with_retry
from services.tracing import traceable
from services.reranker import RerankerUnavailable, rerank_docs, reranker_enabled
from services.bm25 import build_bm25_index, rank_bm25
from services.query_rewriter import (
    hyde_enabled,
    hyde_rewrite,
    multi_query_rewrite,
    multiquery_enabled,
    rrf_merge_ranked_lists,
)
from services.tokenization import BM25_TOKENIZER_ID

logger = logging.getLogger(__name__)
load_dotenv(Path(__file__).parent.parent / ".env")

embedding_model = os.getenv("LLM_EMBEDDING_MODEL")
# embedding 供应商可与 chat 分离（DeepSeek 无 embedding 接口）：
# EMBEDDING_* 未配置则跟随 LLM_*
# connect 超时放宽：SiliconFlow 高峰期 TLS 建连超过 SDK 默认 5s（见 services/llm.py）
from services.llm import PROVIDER_TIMEOUT

client = AsyncOpenAI(
    api_key=os.getenv("EMBEDDING_API_KEY") or os.getenv("LLM_API_KEY"),
    base_url=os.getenv("EMBEDDING_BASE_URL") or os.getenv("LLM_BASE_URL"),
    timeout=PROVIDER_TIMEOUT,
)

# 绝对路径,避免不同启动目录(systemd/docker/测试)各自指向不同的 ./chroma_db
_CHROMA_DIR = os.getenv("CHROMA_DIR") or str(Path(__file__).parent.parent / "chroma_db")
chromadb_client = chromadb.PersistentClient(_CHROMA_DIR)

# 单次 embedding 请求最多 chunk 数:大文档分批,避免撞厂商单请求 input 上限
EMBED_BATCH_SIZE = 64
_STAGING_PREFIX = "studyloop-staging-"
_STAGING_TTL_SECONDS = max(
    60, int(os.getenv("STAGING_COLLECTION_TTL_SECONDS", "3600"))
)
_active_staging_names: set[str] = set()


class DocumentAlreadyExistsError(Exception):
    """A published collection already owns this document id."""


async def _embed(texts: list[str]):
    """embedding 统一入口：接 with_retry（厂商偶发连接抖动/超时 → 指数退避重试）。

    此前 3 处 embeddings.create 裸调用是 resilience 链的盲区——LLM 调用全有
    retry，embedding 一抖整条 RAG 链直接 500。
    """
    return await with_retry(
        lambda: client.embeddings.create(model=embedding_model, input=texts),
        max_retries=3,
        base_delay=1.0,
        timeout=30,
    )


async def _collection_exists(name: str) -> bool:
    try:
        await asyncio.to_thread(chromadb_client.get_collection, name=name)
    except NotFoundError:
        return False
    return True


async def _get_collection_if_exists(name: str):
    try:
        return await asyncio.to_thread(chromadb_client.get_collection, name=name)
    except NotFoundError:
        return None


def _staging_is_stale(metadata: dict, now: float | None = None) -> bool:
    created_at = metadata.get("created_at")
    if not isinstance(created_at, (int, float)):
        return True
    return (now or time.time()) - created_at >= _STAGING_TTL_SECONDS


async def _ensure_document_slot_available(document_id: str) -> None:
    """Reject real duplicates while migrating legacy empty ghost collections."""
    existing = await _get_collection_if_exists(document_id)
    if existing is None:
        return

    metadata = existing.metadata if isinstance(existing.metadata, dict) else {}
    status = metadata.get("ingest_status")
    count = await asyncio.to_thread(existing.count)
    removable = (
        status == "indexing" and _staging_is_stale(metadata)
    ) or (
        status != "indexed" and count == 0
    )
    if removable:
        try:
            await asyncio.to_thread(chromadb_client.delete_collection, name=document_id)
        except NotFoundError:
            pass
        logger.info("[vectorstore] removed incomplete collection before upload: %s", document_id)
        return

    raise DocumentAlreadyExistsError(f"文档 '{document_id}' 已存在，请先删除后再上传")


async def _run_blocking_to_completion(func, /, *args, **kwargs):
    """Do not abandon an in-flight Chroma thread when the HTTP task is cancelled."""
    task = asyncio.create_task(asyncio.to_thread(func, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError as cancellation:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if task.done() and not task.cancelled():
            try:
                task.result()
            except Exception:
                pass
        raise cancellation


async def deal_document(document_id: str, filename: str, chunks: list[str]):
    # 同名重传不能继续用 add：Chroma 会忽略重复 id，导致接口成功但正文仍是旧版本。
    await _ensure_document_slot_available(document_id)

    # 先完成所有外部 embedding 调用，provider 失败时不产生任何 Chroma 写入。
    # 分批 embed:几百 chunk 一次性 embed 会撞 API input 上限(多数厂商 ~2048 条/8k token)
    embeddings: list = []
    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        resp = await _embed(chunks[start:start + EMBED_BATCH_SIZE])
        embeddings.extend(item.embedding for item in resp.data)

    # 写入唯一 staging collection；只有 add 完整成功后才 rename 发布为 document_id。
    # 即使 cleanup 失败，列表也会过滤 staging/indexing collection，不会宣称上传成功。
    staging_name = f"{_STAGING_PREFIX}{uuid.uuid4().hex}"
    collection = None
    try:
        # create 同步执行，确保取消信号到达时已经拿到本次 staging 的所有权，
        # 外层 cleanup 不会因后台线程稍后才创建成功而漏删。
        collection = chromadb_client.create_collection(
            name=staging_name,
            metadata={
                "ingest_status": "indexing",
                "source_filename": filename,
                "created_at": int(time.time()),
            },
        )
        _active_staging_names.add(staging_name)
        await _run_blocking_to_completion(
            collection.add,
            documents=chunks,
            embeddings=embeddings,
            ids=[f"{document_id}_chunk_{i}" for i in range(len(chunks))],
            metadatas=[
                {"source": chunks[i][:50], "doc_id": document_id, "chunk_index": i}
                for i in range(len(chunks))
            ],
        )

        # 防止两个并发上传在首次检查后同时发布同一个 document_id。
        await _ensure_document_slot_available(document_id)
        try:
            await _run_blocking_to_completion(
                collection.modify,
                name=document_id,
                metadata={"ingest_status": "indexed", "source_filename": filename},
            )
        except Exception as publish_error:
            # Chroma 的 rename 冲突没有稳定的专用异常类型；以正式 collection
            # 是否已出现判定并发同名上传，统一返回 409 而不是误报存储故障。
            if await _collection_exists(document_id):
                raise DocumentAlreadyExistsError(
                    f"文档 '{document_id}' 已存在，请先删除后再上传"
                ) from publish_error
            raise
    except BaseException:
        if collection is not None:
            cleanup_name = getattr(collection, "name", staging_name) or staging_name
            try:
                # 同步 cleanup，避免请求已经取消时第二个 await 再次中断清理。
                chromadb_client.delete_collection(name=cleanup_name)
            except Exception as cleanup_error:
                logger.error(
                    "[vectorstore] staging cleanup failure for %s: %s",
                    cleanup_name,
                    cleanup_error,
                    exc_info=True,
                )
        raise
    finally:
        _active_staging_names.discard(staging_name)

    _bm25_cache.pop(document_id, None)   # 文档内容已变,BM25 缓存失效
    return len(chunks)


# ── BM25 索引缓存(按 document_id),避免每次查询全量 collection.get + 重建索引 ──────
_bm25_cache: dict[str, dict] = {}


async def _get_bm25_index(collection, document_id: str) -> dict:
    """取或构建某文档的 BM25 索引,带进程内缓存。返回 {bm25, all_docs, all_ids}。

    缓存在 deal_document / delete_document 时按 document_id 失效。
    分词由 services.tokenization 统一提供，确保索引和查询使用同一规则。
    """
    cached = _bm25_cache.get(document_id)
    if cached is not None:
        return cached
    all_results = await asyncio.to_thread(collection.get, include=["documents"])
    all_docs = all_results["documents"]
    all_ids = all_results["ids"]
    bm25 = build_bm25_index(all_docs)
    entry = {
        "bm25": bm25,
        "all_docs": all_docs,
        "all_ids": all_ids,
        "tokenizer_id": BM25_TOKENIZER_ID,
    }
    _bm25_cache[document_id] = entry
    return entry


def clear_bm25_cache() -> None:
    """清空 BM25 缓存（测试用 / 手动失效）。"""
    _bm25_cache.clear()


async def query_document(document_id: str, query: str):
    query_vec = await _embed([query])
    collection = chromadb_client.get_collection(name=document_id)
    return await asyncio.to_thread(collection.query,query_embeddings=[query_vec.data[0].embedding],n_results=5)


@traceable(
    name="hybrid_retrieval",
    run_type="retriever",
    metadata={"strategy": "bm25+vector+rrf+rerank", "k": 60},
)
async def hybrid_query_document(
    document_id: str,
    query: str,
    n_results: int = 5,
    enable_rerank: bool | None = None,
) -> dict:
    """
    Hybrid 检索：向量 + BM25 → RRF 融合 → Cross-Encoder 精排。

    两阶段检索(借鉴 kotaemon rerankings/cohere.py:35 工业实践):
      召回阶段:vec + BM25,RRF 取 top n_results * 3(给精排足够候选)
      精排阶段:CrossEncoder 真正读 (query, doc) 对打分,取 top n_results

    enable_rerank:
      None  → 用 RERANKER_ENABLED 环境变量(默认 true)
      True  → 强制开启(评测对照组用)
      False → 强制关闭(评测对照组用)
    """
    use_rerank = reranker_enabled() if enable_rerank is None else enable_rerank
    # 精排阶段需要更多候选,召回阶段取 3x;不精排时保持原有 2x 行为以最小化变化
    recall_multiplier = 3 if use_rerank else 2
    recall_n = n_results * recall_multiplier

    # 1. 向量检索
    query_vec = await _embed([query])
    collection = chromadb_client.get_collection(name=document_id)
    vec_results = await asyncio.to_thread(
        collection.query,
        query_embeddings=[query_vec.data[0].embedding],
        n_results=recall_n
    )
    vec_docs = vec_results["documents"][0]      # list[str]
    vec_ids = vec_results["ids"][0]             # list[str]

    # 2. BM25 检索:索引按 document_id 缓存(文档增删时失效),避免每次全量重建
    idx = await _get_bm25_index(collection, document_id)
    all_docs, all_ids, bm25 = idx["all_docs"], idx["all_ids"], idx["bm25"]
    if bm25 is None:
        bm25_ids, bm25_docs = [], []
    else:
        bm25_ranked = rank_bm25(bm25, query, recall_n)
        bm25_ids = [all_ids[i] for i, _ in bm25_ranked]
        bm25_docs = [all_docs[i] for i, _ in bm25_ranked]

    # 3. RRF 融合(召回融合,k=60)
    K = 60
    rrf_scores: dict[str, float] = {}
    id_to_doc: dict[str, str] = {}

    for rank, doc_id in enumerate(vec_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (K + rank + 1)
        id_to_doc[doc_id] = vec_docs[rank]

    for rank, doc_id in enumerate(bm25_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (K + rank + 1)
        id_to_doc[doc_id] = bm25_docs[rank]

    rrf_top_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:recall_n]
    rrf_top_docs = [id_to_doc[doc_id] for doc_id in rrf_top_ids]

    # 4. Cross-Encoder 精排(可降级)
    if use_rerank and len(rrf_top_docs) > 1:
        try:
            reranked = await rerank_docs(query, rrf_top_docs, rrf_top_ids, top_k=n_results)
            top_docs = [r[0] for r in reranked]
            top_ids = [r[2] for r in reranked]
            return {"documents": [top_docs], "ids": [top_ids]}
        except RerankerUnavailable as e:
            logger.warning(f"[hybrid] reranker unavailable, fallback to RRF-only: {e}")
            # 降级:用 RRF top n_results

    # 5. 不精排 / 精排失败 → 用 RRF top n_results
    top_ids = rrf_top_ids[:n_results]
    top_docs = [id_to_doc[doc_id] for doc_id in top_ids]
    return {"documents": [top_docs], "ids": [top_ids]}


@traceable(
    name="retrieve_with_rewrite",
    run_type="retriever",
    metadata={"strategy": "hyde+multiquery+hybrid+rrf"},
)
async def retrieve_with_rewrite(document_id: str, query: str, n_results: int = 5) -> dict:
    """生产检索入口：按 env 决定是否做 HyDE / Multi-query 改写，再 hybrid 检索 + RRF 合并。

    组合矩阵（与 test/run_eval_v3_phase9.py 同一套逻辑，保证生产 == 评测）：
      Multi-query 关 + HyDE 关 → 直接 hybrid(query)
      Multi-query 开 + HyDE 关 → N 个变体各 hybrid → RRF 合并
      Multi-query 关 + HyDE 开 → hybrid(HyDE 改写后的假设答案)
      Multi-query 开 + HyDE 开 → N 变体 → 各 HyDE 改写 → 各 hybrid → RRF 合并

    env 开关见 services/query_rewriter.py：
      QUERY_REWRITE_ENABLED（总开关）/ HYDE_ENABLED / MULTIQUERY_ENABLED / MULTIQUERY_N
      全关时行为与旧版 hybrid_query_document 完全一致（零行为变化）。

    生产加固（相对评测版）：
      ① query 去重——HyDE/multi-query 改写失败会 fallback 回原 query，可能产生重复，去重避免重复检索与 RRF 偏置
      ② 全失败兜底——所有改写后的子查询都检索失败时，退回单 query hybrid，绝不返回空 chunks 饿死出题
    """
    queries: list[str] = [query]
    if multiquery_enabled():
        queries = await multi_query_rewrite(query)
    if hyde_enabled():
        queries = list(await asyncio.gather(*[hyde_rewrite(q) for q in queries]))

    # ① 去重（保序）
    seen: set[str] = set()
    deduped: list[str] = []
    for q in queries:
        if q and q not in seen:
            seen.add(q)
            deduped.append(q)
    queries = deduped or [query]

    # 单 query：直接走 hybrid，省去无意义的 RRF 合并
    if len(queries) == 1:
        return await hybrid_query_document(document_id, queries[0], n_results=n_results)

    # 多 query：各自 hybrid 检索 → RRF 合并
    results = await asyncio.gather(
        *[hybrid_query_document(document_id, q, n_results=n_results) for q in queries],
        return_exceptions=True,
    )
    ranked_lists: list[list[tuple[str, str]]] = []
    for r in results:
        if isinstance(r, Exception):
            logger.warning(f"[retrieve_with_rewrite] one sub-query failed, skipped: {r}")
            continue
        docs = r.get("documents", [[]])[0] or []
        ids = r.get("ids", [[]])[0] or []
        ranked_lists.append([(ids[i], docs[i]) for i in range(min(len(docs), len(ids)))])

    # ② 全失败兜底：退回单 query hybrid
    if not ranked_lists:
        logger.warning("[retrieve_with_rewrite] all rewritten sub-queries failed, fallback to original query")
        return await hybrid_query_document(document_id, query, n_results=n_results)

    merged = rrf_merge_ranked_lists(ranked_lists, top_k=n_results)
    return {
        "documents": [[m[1] for m in merged]],
        "ids":       [[m[0] for m in merged]],
    }


async def bm25_only_query_document(document_id: str, query: str, n_results: int = 5) -> dict:
    """纯 BM25 检索（用于评测对照组，与 hybrid 和 naive 三组对比）"""
    collection = chromadb_client.get_collection(name=document_id)
    idx = await _get_bm25_index(collection, document_id)
    all_docs, all_ids, bm25 = idx["all_docs"], idx["all_ids"], idx["bm25"]
    if bm25 is None:
        return {"documents": [[]], "ids": [[]]}
    ranked = rank_bm25(bm25, query, n_results)
    top_ids = [all_ids[i] for i, _ in ranked]
    top_docs = [all_docs[i] for i, _ in ranked]
    return {"documents": [top_docs], "ids": [top_ids]}


async def get_all_document():
    collections = await asyncio.to_thread(chromadb_client.list_collections)
    visible = []
    now = time.time()
    for collection in collections:
        metadata = collection.metadata if isinstance(collection.metadata, dict) else {}
        status = metadata.get("ingest_status")
        if status == "indexing":
            if (
                collection.name not in _active_staging_names
                and _staging_is_stale(metadata, now)
            ):
                try:
                    await asyncio.to_thread(
                        chromadb_client.delete_collection, name=collection.name
                    )
                except NotFoundError:
                    pass
                except Exception as cleanup_error:
                    logger.error(
                        "[vectorstore] stale staging cleanup failure for %s: %s",
                        collection.name,
                        cleanup_error,
                    )
            continue

        # 升级前的 embedding ghost 没有 metadata 且 count=0：隐藏并迁移清理。
        if status != "indexed" and await asyncio.to_thread(collection.count) == 0:
            try:
                await asyncio.to_thread(
                    chromadb_client.delete_collection, name=collection.name
                )
            except NotFoundError:
                pass
            except Exception as cleanup_error:
                logger.error(
                    "[vectorstore] legacy ghost cleanup failure for %s: %s",
                    collection.name,
                    cleanup_error,
                )
            continue
        visible.append(collection)
    return visible

async def delete_document(document_id: str):
    await asyncio.to_thread(chromadb_client.delete_collection,name=document_id)
    _bm25_cache.pop(document_id, None)   # 仅在存储删除成功后失效
