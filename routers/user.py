from fastapi import APIRouter
from services.memory import get_user_profile, get_user_sessions

router = APIRouter(prefix="/user")


@router.get("/{user_id}/profile")
async def profile(user_id: str):
    """查看用户语义记忆（画像：掌握度 + 薄弱知识点）"""
    # 首次使用时没有画像是正常空状态；200 + null 让 Dashboard 显示空态，
    # 避免 React StrictMode 的两次开发态读取都产生 404 控制台错误。
    return await get_user_profile(user_id)


@router.get("/{user_id}/sessions")
async def session_history(user_id: str):
    """查看用户情节记忆（历史会话摘要，按日期倒序）"""
    return await get_user_sessions(user_id)
