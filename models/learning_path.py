"""
Learning Path 数据模型

数据模型对应 brief → explore → compress → synthesize 流水线：
  - PathBrief         : brief_extraction 阶段产出（结构化用户意图）
  - ExplorationReport : explore 阶段产出（并行 RAG 召回的 chunks + 候选概念）
  - CompressedReport  : compress 阶段产出（压缩到 ~1000 token 的摘要）
  - LearningPath      : synthesize 阶段产出（保留原结构兼容下游）
  - PathCritique      : critique 阶段产出（决定是否 revise）
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ── 对外 schema（保持下游兼容）──────────────────────────────────────────────
class LearningStage(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        revalidate_instances="always",
        str_strip_whitespace=True,
    )

    stage: int = Field(ge=1, le=12, strict=True)
    title: str = Field(min_length=1, max_length=200)
    topics: list[str] = Field(min_length=1, max_length=20)
    description: str = Field(min_length=1, max_length=4000)
    estimated_minutes: int = Field(ge=1, le=480, strict=True)

    @field_validator("topics")
    @classmethod
    def validate_topics(cls, topics: list[str]) -> list[str]:
        normalized = [topic.strip() for topic in topics]
        if any(not topic or len(topic) > 200 for topic in normalized):
            raise ValueError("topics must contain non-empty strings up to 200 characters")
        if len({topic.casefold() for topic in normalized}) != len(normalized):
            raise ValueError("topics must be unique within a stage")
        return normalized


class LearningPath(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        revalidate_instances="always",
        str_strip_whitespace=True,
    )

    document_id: str = Field(min_length=1, max_length=512)
    title: str = Field(min_length=1, max_length=200)
    total_stages: int = Field(ge=1, le=12, strict=True)
    stages: list[LearningStage] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def validate_stage_sequence(self) -> "LearningPath":
        if self.total_stages != len(self.stages):
            raise ValueError("total_stages must equal the number of stages")
        expected = list(range(1, self.total_stages + 1))
        actual = [stage.stage for stage in self.stages]
        if actual != expected:
            raise ValueError("stages must be ordered and numbered from 1")
        return self


class CreateLearningPathRequest(BaseModel):
    """创建持久学习路径资源时使用的受限 Web 请求。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    user_id: Literal["default_user"] = "default_user"
    document_id: str = Field(min_length=1, max_length=512)


class LearningPathProgress(BaseModel):
    """Authoritative stage progress derived from durable completion events."""

    model_config = ConfigDict(extra="forbid", revalidate_instances="always")

    revision: int = Field(ge=1, strict=True)
    completed_through: int = Field(ge=0, le=12, strict=True)


class LearningPathResource(BaseModel):
    """生成后不可变、可跨刷新恢复的学习路径资源。"""

    model_config = ConfigDict(
        extra="forbid",
        revalidate_instances="always",
        str_strip_whitespace=True,
    )

    schema_version: Literal[1] = 1
    learning_path_id: str = Field(pattern=r"^lp_[0-9a-f]{32}$")
    user_id: Literal["default_user"]
    path: LearningPath
    progress: LearningPathProgress
    created_at: float = Field(ge=0, allow_inf_nan=False)
    expires_at: None = None

    @model_validator(mode="after")
    def validate_progress(self) -> "LearningPathResource":
        completed = self.progress.completed_through
        if completed > self.path.total_stages:
            raise ValueError("learning path progress exceeds the stage count")
        if self.progress.revision != completed + 1:
            raise ValueError("learning path progress revision is inconsistent")
        return self


# Provider wire schema intentionally contains only JSON types, required fields,
# and additionalProperties=false. Some OpenAI-compatible providers reject
# min/max JSON-Schema keywords even though the local SDK accepts them. The
# strict domain model above is therefore applied after parsing the wire shape.
class LearningStageWire(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    stage: int
    title: str
    topics: list[str]
    description: str
    estimated_minutes: int


class LearningPathWire(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    document_id: str
    title: str
    total_stages: int
    stages: list[LearningStageWire]


# ── 多阶段中间数据 ──────────────────────────────────────────────────────────
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
