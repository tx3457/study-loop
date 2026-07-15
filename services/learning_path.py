"""
Learning Path 多阶段流水线（Phase 8 P4，借鉴 DeepTutor SourceExplorer + open_deep_research compress）

升级前：单步——取全文 join → LLM → LearningPath
  问题：长文档塞爆 context；planning 没结构化输入；失败无 fallback

升级后：4 阶段流水线 + 可选 critique-revise
  brief_extraction → explore → compress → synthesize → (critique → revise)?

每个阶段都是纯函数：方便单测 + LangGraph 节点直接复用。

面试讲点：
  - 解决长文档塞 LLM 爆 context 的硬伤（compress 阶段）
  - brief 改写让 synthesize 输入稳定（借 open_deep_research write_research_brief）
  - explore 并行多 query RAG sweep（借 DeepTutor SourceExplorer）
  - critique-revise 把"双门"模式从出题扩展到规划（借 DeepTutor SpineSynthesizer）
"""
import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from models.learning_path import (
    CompressedReport,
    ExplorationReport,
    LearningPath,
    PathBrief,
    PathCritique,
)
from services.vectorstore import chromadb_client, hybrid_query_document

load_dotenv(Path(__file__).parent.parent / ".env")

logger = logging.getLogger(__name__)
# 路径流水线各阶段均为 json_schema 结构化输出 → 走 structured 供应商
from services.llm import structured_client as _client, structured_model as _model


# ── 阶段 A：brief_extraction ─────────────────────────────────────────────
_BRIEF_SYSTEM = (
    "你是学习路径规划的 brief 提取专家。任务：基于文档标题和用户意图，"
    "输出结构化 PathBrief，包含规划标题、学习范围、目标水平、阶段数、"
    "3-6 个核心关键词（这些关键词会用作并行 RAG 检索的 query）。"
)


async def extract_brief(document_id: str, user_intent: str = "") -> PathBrief:
    """A 阶段：从文档 id + 可选用户意图 → 结构化 brief。"""
    intent = user_intent.strip() or f"为文档 {document_id} 生成通用学习路径"
    try:
        resp = await _client.beta.chat.completions.parse(
            model=_model,
            messages=[
                {"role": "system", "content": _BRIEF_SYSTEM},
                {"role": "user", "content": f"文档 ID：{document_id}\n用户意图：{intent}"},
            ],
            response_format=PathBrief,
        )
        brief = resp.choices[0].message.parsed
        # 安全限位：阶段数和关键词数
        brief.target_count = max(3, min(brief.target_count, 6))
        brief.keywords = brief.keywords[:6] or [document_id]
        return brief
    except Exception as e:
        logger.warning(f"[planner] brief 提取失败，fallback: {e}")
        return PathBrief(
            title=f"{document_id} 学习路径",
            scope="基于文档内容的通用学习",
            level="intermediate",
            target_count=4,
            keywords=[document_id],
        )


# ── 阶段 B：explore ──────────────────────────────────────────────────────
async def explore(document_id: str, brief: PathBrief) -> ExplorationReport:
    """B 阶段：用 brief.keywords 并行 hybrid 检索，汇总 chunks 并去重。"""
    queries = brief.keywords or [document_id]
    # 并行 RAG sweep
    results = await asyncio.gather(
        *[hybrid_query_document(document_id, q, n_results=5) for q in queries],
        return_exceptions=True,
    )

    seen: set = set()
    merged_chunks: list[str] = []
    for q, r in zip(queries, results):
        if isinstance(r, Exception):
            logger.warning(f"[planner] explore query='{q}' failed: {r}")
            continue
        for doc in (r.get("documents", [[]])[0] or []):
            # 用前 80 字做 dedupe key（chunk 之间可能高度重复）
            key = doc[:80]
            if key not in seen:
                seen.add(key)
                merged_chunks.append(doc)

    return ExplorationReport(
        queries_used=queries,
        chunks=merged_chunks,
        candidate_concepts=[],  # 由 compress 阶段填
    )


# ── 阶段 C：compress ─────────────────────────────────────────────────────
_COMPRESS_SYSTEM = (
    "你是文档压缩助手。任务：把多个 chunks 压缩成 500-1000 字摘要 + 核心概念列表。\n"
    "要求：\n"
    "1. 保留主要主题和层级关系\n"
    "2. 去除冗余、重复段落\n"
    "3. 给出 suggested_stage_count（根据内容自然分段建议，3-6 之间）\n"
    "4. key_concepts 按重要度从高到低排序"
)


async def compress(report: ExplorationReport, brief: PathBrief) -> CompressedReport:
    """C 阶段：把 N 个 chunks 压成单一 summary，防 token 爆炸。"""
    # 安全限位：最多取 30 个 chunk 拼成 user 消息，避免 compress 本身爆 context
    chunks_text = "\n\n---\n\n".join(report.chunks[:30])
    if not chunks_text.strip():
        return CompressedReport(
            summary="（未检索到任何内容）",
            key_concepts=[],
            suggested_stage_count=brief.target_count,
        )

    try:
        resp = await _client.beta.chat.completions.parse(
            model=_model,
            messages=[
                {"role": "system", "content": _COMPRESS_SYSTEM},
                {"role": "user", "content": (
                    f"【学习意图】{brief.scope} (level={brief.level})\n\n"
                    f"【chunks 共 {len(report.chunks)} 条，取前 30】\n{chunks_text}"
                )},
            ],
            response_format=CompressedReport,
        )
        return resp.choices[0].message.parsed
    except Exception as e:
        logger.warning(f"[planner] compress 失败，fallback 直接截断: {e}")
        return CompressedReport(
            summary=chunks_text[:2000],
            key_concepts=brief.keywords,
            suggested_stage_count=brief.target_count,
        )


# ── 阶段 D：synthesize ───────────────────────────────────────────────────
_SYNTHESIZE_SYSTEM = (
    "你是学习路径规划专家。基于压缩摘要 + brief，生成分阶段学习计划。\n"
    "要求：\n"
    "- 阶段数取 brief.target_count 和 compressed.suggested_stage_count 的折中\n"
    "- 每阶段：序号 / 标题 / 知识点列表 / 阶段描述 / 预计学习分钟数（10-30）\n"
    "- 阶段从易到难，前后递进\n"
    "- document_id 原样返回不修改"
)


async def synthesize(
    document_id: str,
    brief: PathBrief,
    compressed: CompressedReport,
    revise_hint: str = "",
) -> LearningPath:
    """D 阶段：综合 brief + compressed → LearningPath。revise_hint 非空时表示是重生。"""
    user_msg = (
        f"【brief】title={brief.title}, scope={brief.scope}, level={brief.level}, "
        f"target_count={brief.target_count}\n\n"
        f"【compressed summary】\n{compressed.summary}\n\n"
        f"【key_concepts】{compressed.key_concepts}\n\n"
        f"【suggested_stage_count】{compressed.suggested_stage_count}\n\n"
        f"document_id={document_id}"
    )
    if revise_hint:
        user_msg += f"\n\n【重要：修订建议（必须遵守）】\n{revise_hint}"

    resp = await _client.beta.chat.completions.parse(
        model=_model,
        messages=[
            {"role": "system", "content": _SYNTHESIZE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        response_format=LearningPath,
    )
    path = resp.choices[0].message.parsed
    path.document_id = document_id
    return path


# ── 阶段 E：critique (可选) ──────────────────────────────────────────────
_CRITIQUE_SYSTEM = (
    "你是学习路径评估官。任务：给出 0-1 评分 + 问题列表 + 是否需要重写。\n"
    "评估维度：\n"
    "1. 阶段是否真的从易到难（递进性）\n"
    "2. 是否覆盖 key_concepts 中的核心概念\n"
    "3. 各阶段 topics 是否非空且具体\n"
    "4. estimated_minutes 是否合理（10-30 区间）\n"
    "若 overall_score >= 0.7 → needs_revision=False；否则 True 并给出 revision_hints。"
)


async def critique(path: LearningPath, compressed: CompressedReport) -> PathCritique:
    """E 阶段：critique LearningPath，决定是否触发 revise。"""
    path_dump = path.model_dump_json()
    try:
        resp = await _client.beta.chat.completions.parse(
            model=_model,
            messages=[
                {"role": "system", "content": _CRITIQUE_SYSTEM},
                {"role": "user", "content": (
                    f"【待评估 LearningPath】\n{path_dump}\n\n"
                    f"【参考 key_concepts】{compressed.key_concepts}"
                )},
            ],
            response_format=PathCritique,
        )
        return resp.choices[0].message.parsed
    except Exception as e:
        logger.warning(f"[planner] critique 失败，默认通过: {e}")
        return PathCritique(
            overall_score=1.0, issues=[], needs_revision=False, revision_hints="",
        )


# ── 顶层入口（向后兼容旧 API）────────────────────────────────────────────
async def generate_learning_path(
    document_id: str,
    user_intent: str = "",
    enable_critique: bool = True,
) -> LearningPath:
    """完整流水线入口。enable_critique=False 可关掉 critique-revise（ablation 用）。

    pipeline:
      A brief → B explore → C compress → D synthesize → E critique →（不通过则 revise）
    """
    logger.info(f"[planner] start pipeline for doc={document_id}, intent={user_intent!r}")

    # A
    brief = await extract_brief(document_id, user_intent)
    logger.info(f"[planner] brief: {brief.title}, keywords={brief.keywords}")

    # B
    report = await explore(document_id, brief)
    logger.info(f"[planner] explored {len(report.chunks)} unique chunks")

    # C
    compressed = await compress(report, brief)
    logger.info(f"[planner] compressed → {len(compressed.summary)} chars summary, "
                f"{len(compressed.key_concepts)} concepts")

    # D
    path = await synthesize(document_id, brief, compressed)
    logger.info(f"[planner] synthesized {path.total_stages} stages")

    if not enable_critique:
        return path

    # E - critique
    crit = await critique(path, compressed)
    logger.info(f"[planner] critique: score={crit.overall_score:.2f}, "
                f"needs_revision={crit.needs_revision}")

    # E' - revise (max 1 轮)
    if crit.needs_revision and crit.revision_hints:
        logger.info(f"[planner] revising with hint: {crit.revision_hints[:80]}")
        path = await synthesize(document_id, brief, compressed, revise_hint=crit.revision_hints)
        logger.info(f"[planner] revised → {path.total_stages} stages")

    return path
