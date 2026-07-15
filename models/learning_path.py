"""
Learning Path 数据模型（Phase 8 P4：planner 多阶段流水线升级）

新增 3 个中间数据类，对应 brief → explore → compress → synthesize 四阶段：
  - PathBrief         : brief_extraction 阶段产出（结构化用户意图）
  - ExplorationReport : explore 阶段产出（并行 RAG 召回的 chunks + 候选概念）
  - CompressedReport  : compress 阶段产出（压缩到 ~1000 token 的摘要）
  - LearningPath      : synthesize 阶段产出（保留原结构兼容下游）
  - PathCritique      : critique 阶段产出（决定是否 revise）
"""
from pydantic import BaseModel, Field


# ── 现有 schema（不动，下游兼容）────────────────────────────────────────────
class LearningStage(BaseModel):
    stage: int
    title: str
    topics: list[str]
    description: str
    estimated_minutes: int


class LearningPath(BaseModel):
    document_id: str
    title: str
    total_stages: int
    stages: list[LearningStage]


# ── 新增：多阶段中间数据 ────────────────────────────────────────────────────
class PathBrief(BaseModel):
    """brief_extraction 阶段：把用户 query 改写成结构化意图。"""
    title: str = Field(description="规划标题，例如 '掌握 RAG 系统基础'")
    scope: str = Field(description="学习范围一句话描述")
    level: str = Field(description="目标水平：beginner / intermediate / advanced")
    target_count: int = Field(description="期望阶段数 3-6")
    keywords: list[str] = Field(description="3-6 个核心关键词，用于 explore 阶段并行检索")


class ExplorationReport(BaseModel):
    """explore 阶段：并行 RAG sweep 后汇总的原始 chunks + 候选概念。"""
    queries_used: list[str]
    chunks: list[str] = Field(description="去重后的 chunks，按 query 分组拼接")
    candidate_concepts: list[str] = Field(default_factory=list, description="LLM 从 chunks 抽出的候选概念，供 synthesize 排序")


class CompressedReport(BaseModel):
    """compress 阶段：把 ExplorationReport 压缩到固定 token 内。"""
    summary: str = Field(description="500-1000 字的整体内容摘要")
    key_concepts: list[str] = Field(description="按重要度排序的核心概念")
    suggested_stage_count: int = Field(description="建议阶段数，与 brief.target_count 可能不同")


class PathCritique(BaseModel):
    """critique 阶段：评估 LearningPath 质量，决定是否 revise。"""
    overall_score: float = Field(ge=0.0, le=1.0, description="整体评分")
    issues: list[str] = Field(default_factory=list, description="发现的问题清单")
    needs_revision: bool = Field(description="是否需要重新生成")
    revision_hints: str = Field(default="", description="给 revise 阶段的具体改进建议")
