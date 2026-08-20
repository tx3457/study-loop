"""
A/B 评测服务

── 架构 ─────────────────────────────────────────────────────────────────────
两层设计：
  1. LLM-as-Judge：对单道题目多维度打分（JudgeVerdict schema）
  2. A/B Runner ：控制变量，生成两组题目 → 并发 Judge → 聚合对比

── 支持的实验 ───────────────────────────────────────────────────────────────
  experiment="ce"  → 同一检索结果，对比有/无 Context Engineering
  experiment="rag" → 同一 query，对比纯向量 vs Hybrid 检索

"""
import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv

from models.eval import (
    ABConfig, ABResult, JudgeScore, JudgeVerdict,
    VariantMetrics, VariantResult,
)
from models.quiz import QuizResponse
from services.rag import generate_question_from_chunks
from services.llm import (
    llm_parse,
    structured_client as _client,
    structured_model as _model,
)
from services.vectorstore import hybrid_query_document, query_document
from services.tracing import traceable

load_dotenv(Path(__file__).parent.parent / ".env")
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# 1. LLM-as-Judge
# ═══════════════════════════════════════════════════════════════════════════

JUDGE_SYSTEM = (
    "你是题目质量评审员。根据原文内容和用户薄弱知识点，对题目进行多维度评分。\n"
    "评分标准：\n"
    "- relevance (1-5)：题目是否与检索内容相关\n"
    "- clarity (1-5)：题目表述是否清晰无歧义\n"
    "- difficulty_feel：感知难度 easy / medium / hard / expert\n"
    "- covers_weak_point：是否覆盖用户薄弱知识点列表中的某个\n"
    "- faithfulness：题目和答案是否忠实于原文，无编造内容\n"
    "- reasoning：一句话评分理由"
)


@traceable(name="llm_judge", run_type="chain")
async def judge_question(
    question: str,
    answer: str,
    source: str,
    weak_points: list[str],
) -> JudgeVerdict:
    """LLM-as-Judge：对单道题目多维度评分。

    用 structured output 强制输出 JudgeScore schema，
    避免自由文本解析错误和幻觉评分。
    """
    try:
        resp = await llm_parse(
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": (
                    f"薄弱知识点列表：{weak_points}\n\n"
                    f"原文片段：{source}\n\n"
                    f"题目：{question}\n"
                    f"答案：{answer}\n\n"
                    "请按评分标准输出结构化评分。"
                )},
            ],
            response_format=JudgeScore,
            client=_client,
            model=_model,
            max_tokens=512,
        )
        score = resp.choices[0].message.parsed
        if score is None:
            raise ValueError("judge returned no parsed score")
        score = JudgeScore.model_validate(score)
        return JudgeVerdict.model_validate(score.model_dump())
    except Exception as exc:
        logger.warning(
            "[judge] 评分失败，标记为无效样本: error_type=%s",
            type(exc).__name__,
        )
        return JudgeVerdict(
            status="error",
            reasoning="judge 调用失败，本样本不计入质量指标",
            error_type=type(exc).__name__,
            error_message=None,
        )


async def judge_batch(
    quiz: QuizResponse,
    weak_points: list[str],
) -> list[JudgeVerdict]:
    """并发评估一组题目，返回逐题 JudgeVerdict 列表。"""
    tasks = [
        judge_question(q.question, q.answer, q.source, weak_points)
        for q in quiz.questions
    ]
    return await asyncio.gather(*tasks)


# ═══════════════════════════════════════════════════════════════════════════
# 2. 指标聚合
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_metrics(verdicts: list[JudgeVerdict]) -> VariantMetrics:
    """将有效逐题评分聚合为组级指标。

    Judge 失败是运行健康信号，不是题目质量评分，因此不进入
    relevance/clarity/faithfulness 的分子或分母。全部失败时保留
    旧数值字段为 0.0，并由 ``valid_count=0`` 明确表示不可评估。
    """
    valid = [v for v in verdicts if v.status == "valid"]
    total_count = len(verdicts)
    valid_count = len(valid)
    failed_count = total_count - valid_count
    n = valid_count or 1
    difficulty_dist: dict[str, int] = {}
    for v in valid:
        # valid verdicts are model-validated to contain all score fields.
        difficulty_dist[v.difficulty_feel] = difficulty_dist.get(v.difficulty_feel, 0) + 1

    return VariantMetrics(
        weak_point_coverage=sum(1 for v in valid if v.covers_weak_point) / n,
        avg_relevance=sum(v.relevance for v in valid) / n,
        avg_clarity=sum(v.clarity for v in valid) / n,
        faithfulness_rate=sum(1 for v in valid if v.faithfulness) / n,
        difficulty_dist=difficulty_dist,
        total_count=total_count,
        valid_count=valid_count,
        failed_count=failed_count,
        judge_success_rate=valid_count / total_count if total_count else 0.0,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 3. A/B 实验运行器
# ═══════════════════════════════════════════════════════════════════════════

@traceable(name="ab_experiment", run_type="chain")
async def run_ab_experiment(config: ABConfig) -> ABResult:
    """运行一次 A/B 实验，返回两组对比结果。

    流程：
      1. 获取检索结果（控制变量）
      2. 并发生成 Baseline + Treatment 两组题目
      3. 并发 LLM-as-Judge 评估两组
      4. 聚合指标 + 计算差异
    """
    if config.experiment == "ce":
        result = await _run_ce_experiment(config)
    else:
        result = await _run_rag_experiment(config)
    return result


async def _run_ce_experiment(config: ABConfig) -> ABResult:
    """CE 实验：同一检索结果，对比有/无 Context Engineering。"""
    # 1. 共用检索结果（控制变量）
    retrieval = await hybrid_query_document(config.document_id, config.query)
    chunks = retrieval["documents"][0]

    # 2. 并发生成两组题目
    baseline_quiz, treatment_quiz = await asyncio.gather(
        generate_question_from_chunks(
            chunks, config.count, "medium", config.type,
        ),
        generate_question_from_chunks(
            chunks, config.count, "medium", config.type,
            difficulty_score=config.difficulty_score,
            weak_points=config.weak_points,
        ),
    )

    # 3. 并发评估
    baseline_verdicts, treatment_verdicts = await asyncio.gather(
        judge_batch(baseline_quiz, config.weak_points),
        judge_batch(treatment_quiz, config.weak_points),
    )

    # 4. 聚合 + 对比
    return _build_result(
        config=config,
        baseline_label="无 CE（Baseline）",
        treatment_label=f"有 CE（score={config.difficulty_score}, weak_points 注入）",
        baseline_verdicts=baseline_verdicts,
        treatment_verdicts=treatment_verdicts,
    )


async def _run_rag_experiment(config: ABConfig) -> ABResult:
    """RAG 实验：同一 query，对比纯向量 vs Hybrid 检索。"""
    # 1. 并发执行两种检索
    vec_result, hybrid_result = await asyncio.gather(
        query_document(config.document_id, config.query),
        hybrid_query_document(config.document_id, config.query),
    )
    vec_chunks = vec_result["documents"][0]
    hybrid_chunks = hybrid_result["documents"][0]

    # 2. 并发生成题目（CE 参数相同，只变检索策略）
    gen_kwargs = dict(
        count=config.count,
        difficulty="medium",
        type=config.type,
        difficulty_score=config.difficulty_score,
        weak_points=config.weak_points,
    )
    baseline_quiz, treatment_quiz = await asyncio.gather(
        generate_question_from_chunks(vec_chunks, **gen_kwargs),
        generate_question_from_chunks(hybrid_chunks, **gen_kwargs),
    )

    # 3. 并发评估
    baseline_verdicts, treatment_verdicts = await asyncio.gather(
        judge_batch(baseline_quiz, config.weak_points),
        judge_batch(treatment_quiz, config.weak_points),
    )

    # 4. 聚合 + 对比
    return _build_result(
        config=config,
        baseline_label="纯向量检索（Baseline）",
        treatment_label="Hybrid BM25+Vector+RRF（Treatment）",
        baseline_verdicts=baseline_verdicts,
        treatment_verdicts=treatment_verdicts,
    )


def _build_result(
    *,
    config: ABConfig,
    baseline_label: str,
    treatment_label: str,
    baseline_verdicts: list[JudgeVerdict],
    treatment_verdicts: list[JudgeVerdict],
) -> ABResult:
    """聚合两组评分，计算差异。"""
    bm = aggregate_metrics(baseline_verdicts)
    tm = aggregate_metrics(treatment_verdicts)

    quality_comparable = bm.valid_count > 0 and tm.valid_count > 0
    delta = {
        "weak_point_coverage": (
            round(tm.weak_point_coverage - bm.weak_point_coverage, 3)
            if quality_comparable else None
        ),
        "avg_relevance": (
            round(tm.avg_relevance - bm.avg_relevance, 2)
            if quality_comparable else None
        ),
        "avg_clarity": (
            round(tm.avg_clarity - bm.avg_clarity, 2)
            if quality_comparable else None
        ),
        "faithfulness_rate": (
            round(tm.faithfulness_rate - bm.faithfulness_rate, 3)
            if quality_comparable else None
        ),
        "judge_success_rate": round(tm.judge_success_rate - bm.judge_success_rate, 3),
    }

    return ABResult(
        experiment=config.experiment,
        config=config,
        baseline=VariantResult(
            variant="baseline", label=baseline_label,
            verdicts=baseline_verdicts, metrics=bm,
        ),
        treatment=VariantResult(
            variant="treatment", label=treatment_label,
            verdicts=treatment_verdicts, metrics=tm,
        ),
        delta=delta,
    )
