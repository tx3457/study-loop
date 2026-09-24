from fastapi import APIRouter, Depends, Query

from services.auth import require_own_subject, require_user_id
from services.memory import get_user_profile, get_user_sessions
from services.srs import get_due_review_items

router = APIRouter(prefix="/user")


@router.get("/{user_id}/profile")
async def profile(user_id: str, subject: str = Depends(require_user_id)):
    """查看用户语义记忆（画像：掌握度 + 薄弱知识点）

    路径里的 user_id 必须就是调用方自己。在此之前它是一个不受约束的路径参数，
    换个名字就能读别人的画像。
    """
    user_id = require_own_subject(user_id, subject)
    # 首次使用时没有画像是正常空状态；200 + null 让 Dashboard 显示空态，
    # 避免 React StrictMode 的两次开发态读取都产生 404 控制台错误。
    return await get_user_profile(user_id)


@router.get("/{user_id}/sessions")
async def session_history(user_id: str, subject: str = Depends(require_user_id)):
    """查看用户情节记忆（历史会话摘要，按日期倒序）"""
    user_id = require_own_subject(user_id, subject)
    return await get_user_sessions(user_id)


@router.get("/{user_id}/reviews/due")
async def due_reviews(
    user_id: str,
    document_id: str | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    subject: str = Depends(require_user_id),
):
    """今天该复习的知识点，最该复习的在前。

    The schedule already existed and already drove the model's context; it just
    had no way out to the person doing the learning. Read-only: reviewing is
    what updates the schedule, not looking at it.
    """
    user_id = require_own_subject(user_id, subject)
    items = await get_due_review_items(user_id, document_id, limit=limit)
    return {"items": items, "total": len(items)}
