"""
自适应学习闭环数据模型

NextStepDecision 由 LLM 根据具体错题和知识盲点产出下一步教学动作：
补薄弱点、升难度、同水平巩固、转学习路径或结束，并给出可解释理由。
"""
from pydantic import BaseModel, Field


class NextStepDecision(BaseModel):
    """自适应辅导 agent 对"下一步"的结构化决策(LLM 推理产出,非固定公式)。

    action 取值(代码层会归一化兜底,防 LLM 乱填):
      advance        学生表现好 → 升难度 / 进阶
      remediate      学生卡在薄弱点 → 降难度 + 针对盲点重练
      continue       同水平巩固
      switch_to_plan 知识缺口系统性 → 转学习路径规划
      finish         已达标 / 练够了 → 结束
    """
    action: str = Field(default="continue", description="advance/remediate/continue/switch_to_plan/finish")
    topic: str = Field(default="", description="下一轮出题主题(中文关键词)")
    difficulty: str = Field(default="medium", description="easy/medium/hard")
    difficulty_score: float = Field(default=0.5, description="0-1 连续难度")
    question_type: str = Field(default="choice", description="choice/true_false/short_answer")
    count: int = Field(default=3, description="题目数量")
    target_weak_points: list[str] = Field(default_factory=list, description="本轮重点针对的薄弱知识点")
    reason: str = Field(default="", description="为什么这么决定(pedagogical reasoning,可解释)")


class AdaptiveTurn(BaseModel):
    """单轮记录，串联后形成学生的难度与掌握度轨迹。"""
    turn: int
    action: str                                 # 触发本轮的 agent 决策动作
    topic: str
    difficulty_score: float
    reason: str = ""
    score: float | None = None                  # 本轮答题得分(批改后回填)
    mastery_after: float | None = None          # 本轮批改后的 EMA 掌握度
    knowledge_gaps: list[str] = Field(default_factory=list)
