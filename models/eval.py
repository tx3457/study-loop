"""
A/B 评测数据模型（Phase 7 工程补洞 #3）

── 面试表述 ─────────────────────────────────────────────────────────────────
"我设计了一套 LLM-as-Judge 评测流水线，用结构化输出让 LLM 对出题质量打分，
 评估维度包括知识点覆盖率、难度对齐度、题目清晰度。A/B 桩支持切换检索策略
 和 CE 参数，实验结果以 JSON 返回，便于接入 dashboard 或写入数据库做长期追踪。"
"""
from pydantic import BaseModel, Field


# ── LLM-as-Judge 单题评分 ──────────────────────────────────────────────────

class JudgeVerdict(BaseModel):
    """LLM Judge 对单道题目的结构化评分。

    用 structured output（response_format）强制 LLM 输出此 schema，
    避免自由文本解析错误。
    """
    relevance: int = Field(ge=1, le=5, description="题目与检索内容的相关性 1-5")
    clarity: int = Field(ge=1, le=5, description="题目表述清晰度 1-5")
    difficulty_feel: str = Field(description="题目感知难度: easy / medium / hard / expert")
    covers_weak_point: bool = Field(description="是否覆盖用户薄弱知识点")
    matched_point: str = Field(description="匹配的具体知识点，无则填'无'")
    faithfulness: bool = Field(description="题目和答案是否忠实于原文，无编造")
    reasoning: str = Field(description="Judge 的评分理由（一句话）")


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
