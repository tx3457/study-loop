"""
Orchestrator 路由

POST /agent/run  →  根据 action 路由到对应 Agent 流水线：
  action="quiz"  → AdaptAgent(读画像) → QuizAgent(检索+出题+审核)
  action="grade" → GraderAgent(AI批改) → AdaptAgent(写回画像)
  action="plan"  → PlannerAgent(学习路径)
"""
from fastapi import Depends, APIRouter, HTTPException
from pydantic import BaseModel
from agents.orchestrator import orchestrator
from services.auth import require_user_id
from services.tracing import build_trace_config

router = APIRouter(prefix="/agent", tags=["agent"])


class RunRequest(BaseModel):
    action: str                      # "quiz" | "grade" | "plan"
    user_id: str = "default_user"
    document_id: str = ""
    description: str = ""           # 出题主题（quiz 时使用）
    count: int = 5
    difficulty: str = "medium"      # easy / medium / hard（无画像时 fallback）
    type: str = "choice"
    session_id: str | None = None   # grade 时必填（对应已完成的答题会话）


@router.post("/run")
async def run_agent(req: RunRequest, subject: str = Depends(require_user_id)):
    """Multi-Agent 编排入口。

    Quiz 流（action='quiz'）：
      1. AdaptAgent 读取用户画像，计算 difficulty_score 和 weak_points
      2. QuizAgent 执行 Hybrid 检索 + CE 出题 + 质量审核

    Grade 流（action='grade'）：
      需先通过 POST /session/{id}/answer 提交全部答案，再调用此接口。
      1. GraderAgent 调用 AI 批改，生成 grading_report
      2. AdaptAgent 将结果写回情节记忆并更新掌握度画像

    Plan 流（action='plan'）：
      PlannerAgent 读取文档全量内容，生成分阶段学习路径。
    """
    # 请求体里自报的 user_id 不作数：身份只来自 Authorization 头解析出的主体。
    req.user_id = subject
    if req.action == "grade" and not req.session_id:
        raise HTTPException(status_code=422, detail="grade action requires session_id")

    result = await orchestrator.ainvoke(
        req.model_dump(),
        config=build_trace_config(
            user_id=req.user_id,
            # grade 流传入的 session_id 对应已完成答题会话；其他流用 document_id 作兜底
            session_id=req.session_id or req.document_id or None,
            tags=[f"action:{req.action}", "endpoint:agent_run"],
        ),
    )

    if req.action == "quiz":
        return {
            "quiz": result.get("quiz"),
            "difficulty_score": result.get("difficulty_score"),
            "weak_points": result.get("weak_points"),
        }
    elif req.action == "grade":
        return {"grading_report": result.get("grading_report")}
    elif req.action == "plan":
        return {"learning_path": result.get("learning_path")}
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {req.action}")
