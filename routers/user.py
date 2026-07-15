from fastapi import APIRouter, HTTPException
from services.memory import get_user_profile, get_user_sessions

router = APIRouter(prefix="/user")


@router.get("/{user_id}/profile")
async def profile(user_id: str):
    """查看用户语义记忆（画像：掌握度 + 薄弱知识点）"""
    data = await get_user_profile(user_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"No profile found for user '{user_id}'")
    return data


@router.get("/{user_id}/sessions")
async def session_history(user_id: str):
    """查看用户情节记忆（历史会话摘要，按日期倒序）"""
    return await get_user_sessions(user_id)
