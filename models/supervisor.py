"""
Supervisor-based Multi-Agent 决策模型（Phase 1 骨架）

把现有"规则路由 workflow"（orchestrator._route 的 if/else + _should_revise 阈值）
升级为 TeachingSupervisor（LLM 动态编排）+ 专职 worker 的真 Multi-Agent。

和 NextStepDecision（services/adaptive_loop.decide_next_step 产出）的关系：
  - NextStepDecision 只回答"下一步教什么"（action / difficulty / topic ...）；
  - SupervisorDecision 在此之上还回答"派哪个 worker 去执行"（next_agent），
    把"教学决策"与"agent 编排"合并成一次 LLM 推理，是 supervisor 范式的核心。

设计：字段尽量与 NextStepDecision 对齐（action / topic / difficulty / count ...），
方便 worker 之间透传，也方便 _rule_fallback_next 复用 adaptive_loop 的难度逻辑。
"""
from pydantic import BaseModel, Field


class SupervisorDecision(BaseModel):
    """教学主管（TeachingSupervisor）对"下一步派谁、做什么"的结构化决策。

    next_agent 取值（代码层 _normalize_decision 会兜底，防 LLM 乱填）：
      diagnostic  诊断学情：读用户画像 / mastery / weak_points（对应 adapt_reader）
      planner     生成系统学习路径（知识缺口系统性时）
      quiz        出题（hybrid 检索 + 生成 + 审核）
      grader      批改学生作答 → grading_report
      tutor       纯讲解（teach：学生没懂，先讲清概念再出题验证）
      assistant   开放式问答 / 闲聊 / 工具调用（assist 模式）
      finish      已达标 / 练够 / 会话结束 → 收尾出图

    action 取值（与 NextStepDecision 的 5 动作对齐，语义同 adaptive_loop）：
      advance        学生表现好 → 升难度 / 进阶主题
      remediate      卡在薄弱点 → 降难度 + 针对盲点重练
      continue       同水平巩固
      switch_to_plan 知识缺口系统性 → 转学习路径规划
      finish         已达标 / 练够了 → 结束
    """

    next_agent: str = Field(
        default="diagnostic",
        description="下一步派谁执行：diagnostic/planner/quiz/grader/tutor/assistant/finish",
    )
    action: str = Field(
        default="continue",
        description="教学动作：advance/remediate/continue/switch_to_plan/finish",
    )
    topic: str = Field(default="", description="下一步主题（中文关键词）")
    difficulty: str = Field(default="medium", description="easy/medium/hard")
    difficulty_score: float = Field(
        default=0.5, ge=0.0, le=1.0, description="0-1 连续难度（最近发展区）",
    )
    question_type: str = Field(default="choice", description="choice/true_false/short_answer")
    count: int = Field(default=3, ge=1, le=10, description="题目数量（1-10）")
    target_weak_points: list[str] = Field(
        default_factory=list, description="本步重点针对的薄弱知识点（从学生已暴露盲点里选）",
    )
    reason: str = Field(default="", description="为什么这么编排（可解释教学/调度理由，一两句话）")
    done: bool = Field(default=False, description="会话是否应当结束（与 next_agent=finish 等价兜底）")
