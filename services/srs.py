"""
间隔重复复习调度（SM-2）——把跨会话记忆从"记得你"闭成"主动安排你复习该复习的"。

SM-2（SuperMemo-2，Anki 同源算法）：每个知识点维护 (复习次数 n, 难度因子 ef,
间隔 interval 天, 下次到期 due)。复习质量 quality<3（又错）→ 重置近期反复出现；
>=3（答对）→ 间隔按 ef 指数拉长、逐渐淡出。开场时 supervisor 优先安排"到期"知识点
复习——这正是遗忘曲线下的最优复习时机，是 Anki / 多邻国级学习产品的核心。

与现有记忆的衔接：复习项来自 error_log / weak_points 暴露的 knowledge_gap；
存储用 memory Store 的 review_schedule bank（key='current'），全程纯规则、零 LLM。

差异化：知码是"代码沙盒判题"，本项目这条是"学习者侧的遗忘曲线复习调度"，互不重叠。
"""
from datetime import date, timedelta

from services.memory import (
    persist_memory_snapshot,
    read_bank_state,
    write_bank_state,
)

_REVIEW_BANK = "review_schedule"
_DEFAULT_EF = 2.5
_MIN_EF = 1.3
# 答错（含复习又错）的质量分：<3 触发 SM-2 重置
_QUALITY_WRONG = 2


def sm2_update(state: dict, quality: int) -> dict:
    """SM-2 核心：根据复习质量 quality(0-5) 更新 {n, ef, interval}。纯函数、可单测。

    quality<3（没答上来）→ 复习次数清零、间隔回到 1 天（近期反复出现）；
    quality>=3（答上来了）→ 间隔按 n 推进（1 → 6 → interval*ef），难度因子 ef 同步微调。
    """
    ef = state.get("ef", _DEFAULT_EF)
    n = state.get("n", 0)
    interval = state.get("interval", 0)
    quality = max(0, min(5, int(quality)))

    if quality < 3:
        n = 0
        interval = 1
    else:
        if n == 0:
            interval = 1
        elif n == 1:
            interval = 6
        else:
            interval = max(1, round(interval * ef))
        n += 1
    # ef 更新（SM-2 标准公式），下限 1.3 防间隔塌缩
    ef = ef + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    ef = max(_MIN_EF, round(ef, 3))
    return {"n": n, "ef": ef, "interval": interval}


def _quality_correct(score: float | None = None) -> int:
    """答对时的质量分：默认 4，整体表现很好（>=0.85）给 5。"""
    if score is None:
        return 4
    return 5 if score >= 0.85 else 4


def _key(document_id: str | None, point: str) -> str:
    return f"{document_id or '_'}::{point}"


def _apply(items: dict, document_id: str | None, point: str, quality: int, today: date) -> None:
    """对单个知识点跑一次 SM-2 并写回 items（带 due / last 时间戳）。"""
    k = _key(document_id, point)
    cur = items.get(k, {"n": 0, "ef": _DEFAULT_EF, "interval": 0})
    upd = sm2_update(cur, quality)
    items[k] = {
        **upd,
        "point": point,
        "document_id": document_id,
        "due": (today + timedelta(days=upd["interval"])).isoformat(),
        "last": today.isoformat(),
    }


async def get_due_reviews(user_id: str, document_id: str | None = None,
                          today: date | None = None, limit: int = 5) -> list[str]:
    """返回到期（due <= today）该复习的知识点，按到期日升序（最该复习的在前）。"""
    today = today or date.today()
    items = (await read_bank_state(user_id, _REVIEW_BANK) or {}).get("items", {})
    due = []
    for it in items.values():
        if document_id is not None and it.get("document_id") not in (document_id, None):
            continue
        if it.get("due", "") <= today.isoformat():
            due.append(it)
    due.sort(key=lambda x: x.get("due", ""))
    return [it["point"] for it in due if it.get("point")][:limit]


async def update_after_session(user_id: str, document_id: str | None,
                               reviewed_points: list[str], wrong_gaps: list[str],
                               today: date | None = None) -> dict:
    """批改后更新调度（零 LLM）：

      - reviewed_points（本轮 supervisor 针对复习的点）中未再错的 → 复习通过（间隔拉长）；又错的 → 重置
      - wrong_gaps 中不在复习集的新错点 → 新增 / 重置（明天到期）

    返回更新后的 review_schedule state（便于测试断言）。
    """
    today = today or date.today()
    state = await read_bank_state(user_id, _REVIEW_BANK) or {"items": {}}
    items = dict(state.get("items", {}))
    wrong_set = set(wrong_gaps)
    reviewed_set = set(reviewed_points)

    # 1. 本轮针对复习的点：未再错 → 通过；又错 → 重置
    for p in reviewed_points:
        if not p:
            continue
        quality = _QUALITY_WRONG if p in wrong_set else _quality_correct()
        _apply(items, document_id, p, quality, today)

    # 2. 本轮新暴露的错点（不在复习集）：register / reset
    for g in wrong_gaps:
        if not g or g in reviewed_set:
            continue
        _apply(items, document_id, g, _QUALITY_WRONG, today)

    state["items"] = items
    state["last_updated"] = today.isoformat()
    await write_bank_state(user_id, _REVIEW_BANK, state)
    await persist_memory_snapshot()
    return state
