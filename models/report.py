from pydantic import BaseModel


class TopicMastery(BaseModel):
    topic: str
    mastery_pct: float    # 0-100
    question_count: int
    correct_count: int


class LearningReport(BaseModel):
    session_id: str
    document_id: str
    overall_score: float
    topic_mastery: list[TopicMastery]
    strengths: list[str]       # 掌握良好的知识点（mastery ≥ 70%）
    weaknesses: list[str]      # 薄弱知识点（mastery < 70%）
    recommendations: list[str] # 3-5 条个性化学习建议
    summary: str               # 总体评语


class _ReportCore(BaseModel):
    """LLM 只填这部分，session/document/score 由代码注入"""
    topic_mastery: list[TopicMastery]
    strengths: list[str]
    weaknesses: list[str]
    recommendations: list[str]
    summary: str
