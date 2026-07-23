"""
A/B 评测数据模型
"""
from typing import Literal

from pydantic import BaseModel, Field, model_validator


# ── LLM-as-Judge 单题评分 ──────────────────────────────────────────────────

class JudgeScore(BaseModel):
    """LLM 成功返回时的结构化评分 schema。"""

    relevance: int = Field(ge=1, le=5, description="题目与检索内容的相关性 1-5")
    clarity: int = Field(ge=1, le=5, description="题目表述清晰度 1-5")
    difficulty_feel: str = Field(description="题目感知难度: easy / medium / hard / expert")
    covers_weak_point: bool = Field(description="是否覆盖用户薄弱知识点")
    matched_point: str = Field(description="匹配的具体知识点，无则填'无'")
    faithfulness: bool = Field(description="题目和答案是否忠实于原文，无编造")
    reasoning: str = Field(description="Judge 的评分理由（一句话）")


class JudgeVerdict(BaseModel):
    """Judge 单题结果：成功时带评分，失败时显式标记 error。

    评分字段对 error 结果为 ``None``，避免用伪造默认分数表示
    Judge 失败。旧的成功结果形状仍保留，只增加状态字段。
    """

    relevance: int | None = Field(default=None, ge=1, le=5, description="题目与检索内容的相关性 1-5")
    clarity: int | None = Field(default=None, ge=1, le=5, description="题目表述清晰度 1-5")
    difficulty_feel: str | None = Field(default=None, description="题目感知难度")
    covers_weak_point: bool | None = Field(default=None, description="是否覆盖用户薄弱知识点")
    matched_point: str | None = Field(default=None, description="匹配的具体知识点")
    faithfulness: bool | None = Field(default=None, description="题目和答案是否忠实于原文")
    reasoning: str = Field(description="Judge 的评分理由（一句话）")
    status: Literal["valid", "error"] = Field(default="valid", description="Judge 调用状态")
    error_type: str | None = Field(default=None, description="Judge 失败异常类型")
    error_message: str | None = Field(default=None, description="Judge 失败摘要")

    @model_validator(mode="after")
    def validate_status_payload(self):
        score_fields = (
            self.relevance,
            self.clarity,
            self.difficulty_feel,
            self.covers_weak_point,
            self.matched_point,
            self.faithfulness,
        )
        if self.status == "valid" and any(value is None for value in score_fields):
            raise ValueError("valid judge verdict requires all score fields")
        if self.status == "error":
            if not self.error_type:
                raise ValueError("error judge verdict requires error_type")
            if any(value is not None for value in score_fields):
                raise ValueError("error judge verdict cannot contain score fields")
        return self


# ── A/B 实验配置 ────────────────────────────────────────────────────────────

class ABConfig(BaseModel):
    """A/B 实验请求参数。

    experiment 字段决定对比维度：
      "ce"  → 对比有/无 Context Engineering（difficulty_score + weak_points）
      "rag" → 对比纯向量检索 vs Hybrid（BM25 + Vector + RRF）
    """
    document_id: str
    query: str = "全部内容"
    count: int = 5
    type: str = "choice"
    experiment: str = Field(
        pattern="^(ce|rag)$",
        description="实验类型: ce（Context Engineering）| rag（检索策略）",
    )
    # CE 实验参数（experiment="ce" 时使用）
    weak_points: list[str] = ["高速推理", "产品功能"]
    difficulty_score: float = Field(default=0.65, ge=0.0, le=1.0)


# ── 单组评估结果 ────────────────────────────────────────────────────────────

class VariantMetrics(BaseModel):
    """单个变体的聚合指标"""
    weak_point_coverage: float = Field(description="薄弱知识点覆盖率 0-1")
    avg_relevance: float = Field(description="平均相关性 1-5")
    avg_clarity: float = Field(description="平均清晰度 1-5")
    faithfulness_rate: float = Field(description="忠实率 0-1")
    difficulty_dist: dict[str, int] = Field(description="难度分布计数")
    total_count: int = Field(default=0, ge=0, description="总 Judge 样本数")
    valid_count: int = Field(default=0, ge=0, description="有效 Judge 样本数")
    failed_count: int = Field(default=0, ge=0, description="Judge 失败样本数")
    judge_success_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Judge 有效样本占比")


class VariantResult(BaseModel):
    """单个变体（Baseline 或 Treatment）的完整结果"""
    variant: str          # "baseline" | "treatment"
    label: str            # 人类可读标签
    verdicts: list[JudgeVerdict]
    metrics: VariantMetrics


# ── A/B 实验最终结果 ────────────────────────────────────────────────────────

class ABResult(BaseModel):
    """A/B 实验完整输出，包含两组对比 + 差异"""
    experiment: str
    config: ABConfig
    baseline: VariantResult
    treatment: VariantResult
    delta: dict = Field(description="Treatment - Baseline 各指标差异")
