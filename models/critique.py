"""
Critic Agent 评估 schema（Phase 7 美团 JD 改造 T2）

── 设计目标 ────────────────────────────────────────────────────────────────
升级 ReviewResult（passed: bool）为结构化 3 维度评分 + 改进建议列表，
让 Tutor↔Critic 双 agent 反思循环有真实的协作信息流通：
  - Tutor 出题 → Critic 多维度评分 → Tutor 看建议改进
  - 不再是单次 bool 拒绝/通过

── 与 ReviewResult 的边界 ──────────────────────────────────────────────────
ReviewResult 保留作为 QuizAgent 内部的"快速通过/拒绝"信号；
CritiqueReport 是更上层、可被 Orchestrator 看到的"反思证据链"。
"""
from typing import Literal
from pydantic import BaseModel, Field


class DimensionScore(BaseModel):
    """单维度评分（0.0-1.0）"""
    score: float = Field(ge=0.0, le=1.0, description="该维度得分，0=很差 1=完美")
    reasoning: str = Field(description="为什么打这个分，引用题目原文")


class CritiqueSuggestion(BaseModel):
    """单条改进建议，可被 Tutor 直接消费"""
    target: Literal["difficulty", "relevance", "coverage", "general"] = Field(
        description="建议针对的维度，general=跨维度泛建议"
    )
    severity: Literal["low", "medium", "high"] = Field(
        description="严重程度：high 必改，medium 建议改，low 可忽略"
    )
    action: str = Field(description="具体改进动作，如『把第 3 题选项 B 改成更具迷惑性的近似项』")


class CritiqueReport(BaseModel):
    """Critic Agent 输出，写入 OrchestratorState.critique_history"""

    difficulty: DimensionScore = Field(description="难度匹配度（vs 用户画像 difficulty_score）")
    relevance: DimensionScore = Field(description="题目与检索 chunks 的相关性")
    coverage: DimensionScore = Field(description="题目对薄弱知识点的覆盖度")

    overall_score: float = Field(
        ge=0.0, le=1.0,
        description="三维度加权平均（默认等权 1/3）"
    )

    suggestions: list[CritiqueSuggestion] = Field(
        default_factory=list,
        description="按 severity 降序排列的改进建议"
    )

    triggered_search: bool = Field(
        default=False,
        description="是否触发 dispatch_tool('search_document') 重新检索证据"
    )

    evidence_chunks_count: int = Field(
        default=0,
        description="二次检索拿到的 chunks 数量（用于 trajectory eval 算 self_correction_rate）"
    )

    def needs_revision(self, threshold: float = 0.7) -> bool:
        """判定是否需要 Tutor 重出。overall_score < threshold 或存在 high severity 建议则需要。"""
        if self.overall_score < threshold:
            return True
        if any(s.severity == "high" for s in self.suggestions):
            return True
        return False
