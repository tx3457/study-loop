"""
偏好学习（启发式 consolidation，零 LLM）。

填补 services/memory.py 的 update_preferences "零调用方"缺口：preferences bank 一直只读
不写（adapt_reader 读它做 difficulty fallback，但从没人写 → 永远空、fallback 永不命中）。
本模块从批改报告 + 历史轨迹用纯规则推断用户学习偏好，供 grader_worker 在会话结束写回。

为什么用规则而非 LLM：
  偏好信号是统计性的（得分趋势 / 题型表现 / 盲点复现），规则可解释、确定性、零成本，
  比让 LLM "猜偏好"更稳。

推断三个偏好（保守：样本不足或信号弱 → 不写该字段，避免污染 preferences）：
  - preferred_difficulty：近期得分趋势 → easy/medium/hard（均分 <0.5 易 / >=0.8 难 / 居中）
  - preferred_type：题型表现 EMA 累积 → 偏向正确率高的题型（需该题型 >=2 次样本）
  - needs_explanation：同一 knowledge_gap 在历史里反复出现（>=2 次）→ True（supervisor 倾向先讲）
"""
from collections import Counter

from models.grader import GradingReport

# 得分趋势阈值
_HIGH = 0.8
_LOW = 0.5
# 至少需要多少轮历史才推断难度（防 1 个样本的噪声）
_MIN_HISTORY_FOR_DIFFICULTY = 2
# 近期窗口（只看最近 N 轮趋势，老数据已被 mastery EMA 吸收）
_RECENT_WINDOW = 3
# 同一盲点出现多少次算"反复错"
_REPEAT_GAP_THRESHOLD = 2

_TYPES = {"choice", "true_false", "short_answer"}


def _recent_scores(history: list[dict]) -> list[float]:
    """取最近 _RECENT_WINDOW 轮的 score（history 由 grader_worker 回填，末条=本轮）。"""
    scores = [
        h.get("score") for h in history
        if isinstance(h, dict) and isinstance(h.get("score"), (int, float))
    ]
    return scores[-_RECENT_WINDOW:]


def _infer_difficulty(history: list[dict]) -> str | None:
    """近期得分趋势 → 偏好难度。样本不足返回 None（不写）。

    history 已含本轮（grader_worker 先 append 再调 consolidate），故不再额外并入 report.score，
    否则单轮会被重复计数。
    """
    scores = _recent_scores(history)
    if len(scores) < _MIN_HISTORY_FOR_DIFFICULTY:
        return None
    avg = sum(scores) / len(scores)
    if avg >= _HIGH:
        return "hard"
    if avg < _LOW:
        return "easy"
    return "medium"


def _repeated_gap(history: list[dict]) -> bool:
    """同一 knowledge_gap 在历史里反复出现（>=2 次）→ 倾向需要先讲解。

    只统计 history（已含本轮），不再并入 report，避免单轮盲点被算成"反复"。
    """
    gaps: list[str] = []
    for h in history:
        if isinstance(h, dict):
            gaps += [g for g in (h.get("knowledge_gaps") or []) if g]
    counts = Counter(gaps)
    return any(v >= _REPEAT_GAP_THRESHOLD for v in counts.values())


def infer_preferences(
    report: GradingReport,
    history: list[dict] | None = None,
    current_prefs: dict | None = None,
    question_type: str | None = None,
) -> dict:
    """从批改报告 + 历史轨迹推断偏好 patch（只含要更新的字段，空 dict = 不更新）。

    Args:
        report:        本轮 GradingReport（提供本轮 score 用于题型 EMA）
        history:       轨迹列表（每条含 score / knowledge_gaps，末条=本轮）
        current_prefs: 当前 preferences（用于 type_perf 增量累积）
        question_type: 本轮出题题型（QuestionGrade 不带题型，故由 state["type"] 传入）

    Returns:
        patch dict，如 {"preferred_difficulty": "hard", "preferred_type": "choice",
                       "type_perf": {...}, "needs_explanation": True}
    """
    history = history or []
    current_prefs = current_prefs or {}
    patch: dict = {}

    # 1. preferred_difficulty：近期得分趋势
    diff = _infer_difficulty(history)
    if diff:
        patch["preferred_difficulty"] = diff

    # 2. preferred_type：题型表现 EMA 累积（new = old*0.6 + score*0.4，对齐 mastery EMA）
    if question_type in _TYPES:
        type_perf = {k: dict(v) for k, v in (current_prefs.get("type_perf") or {}).items()}
        prev = type_perf.get(question_type)
        if prev:
            n = prev.get("n", 0) + 1
            new_avg = round(prev.get("avg", report.score) * 0.6 + report.score * 0.4, 3)
        else:
            n = 1
            new_avg = round(float(report.score), 3)
        type_perf[question_type] = {"n": n, "avg": new_avg}
        patch["type_perf"] = type_perf
        # 选 avg 最高且样本 >=2 的题型作为偏好（单样本不算数）
        eligible = {t: d for t, d in type_perf.items() if d.get("n", 0) >= 2}
        if eligible:
            patch["preferred_type"] = max(eligible.items(), key=lambda x: x[1]["avg"])[0]

    # 3. needs_explanation：盲点反复出现
    if _repeated_gap(history):
        patch["needs_explanation"] = True

    return patch
