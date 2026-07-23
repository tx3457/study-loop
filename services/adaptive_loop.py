"""
自适应学习闭环的 Agent 决策逻辑

decide_next_step 读取本轮逐题对错、知识盲点及历史难度/得分轨迹，
推理下一步动作（补薄弱点、升难度、巩固、转规划或结束）并给出理由。
闭环:出题 → 作答 → 批改 → decide_next_step 推理 → 下一步,直到达标/练够。

健壮性:LLM 决策失败时回退规则(score<0.5→remediate 降难度,否则 advance 升难度),
以 fail-soft 方式保持辅导会话可继续执行。
"""
import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

from models.adaptive import AdaptiveTurn, NextStepDecision
from models.grader import GradingReport
from services.llm import llm_chat, llm_parse
from services.vectorstore import retrieve_with_rewrite

logger = logging.getLogger(__name__)
load_dotenv(Path(__file__).parent.parent / ".env")

# LLM 调用统一走 services.llm.llm_chat / llm_parse；默认 client 在那里管理。
# 这里只保留 _model 给 llm_chat/llm_parse 显式透传。
_model = os.getenv("LLM_MODEL")

# ── 闭环参数 ─────────────────────────────────────────────────────────────────
MASTERY_TARGET = 0.85          # 掌握度达到即结束
MAX_TURNS = 5                  # 最多 5 轮,防无限循环(和 autonomous 的 max_rounds 同精神)

_ACTIONS = {"advance", "remediate", "teach", "continue", "switch_to_plan", "finish"}
_DIFF_WORDS = {"easy", "medium", "hard"}
_TYPES = {"choice", "true_false", "short_answer"}


_DECIDE_SYSTEM = (
    "你是一个自适应学习辅导 agent。根据学生本轮答题表现、暴露的知识盲点,以及历史"
    "难度/得分轨迹,决定下一步该怎么教。你不是只会加难度的公式,要像真人老师一样判断:\n"
    "- 学生连续答得好(得分高)→ advance:升难度或进阶主题\n"
    "- 学生在某知识点反复错、光换难度没用 → teach:先给一段讲解(检索材料+举例)把概念讲清,下一轮再用同概念出题验证\n"
    "- 学生得分偏低但只是手生 → remediate:降难度,针对薄弱点重练\n"
    "- 表现中等、需要巩固 → continue:同水平换题再练\n"
    "- 知识缺口很系统、零散补救无效 → switch_to_plan:转系统学习路径规划\n"
    "- 已掌握或练习充分 → finish:结束\n\n"
    "teach 与 remediate 的区别:teach 是『讲』(学生没懂,要先讲解),remediate 是『再练』(学生懂但不熟)。\n"
    "必须给出 reason(你的教学理由,一两句话)。topic 用中文关键词;"
    "target_weak_points 从学生已暴露的盲点里选;difficulty_score 用 0-1 连续值。"
)


def _format_history(history: list[AdaptiveTurn]) -> str:
    """把轨迹压成'第N轮: 动作/主题/难度/得分/掌握度'。"""
    lines = []
    for t in history:
        score = f"{t.score:.2f}" if t.score is not None else "未答"
        mastery = f"{t.mastery_after:.2f}" if t.mastery_after is not None else "-"
        lines.append(
            f"  第{t.turn}轮: {t.action} | 主题「{t.topic}」 | 难度{t.difficulty_score:.2f} "
            f"| 得分{score} | 掌握度{mastery}"
        )
    return "\n".join(lines)


def _format_report(report: GradingReport) -> str:
    """把本轮批改压成'得分 + 逐题对错 + 盲点'。"""
    lines = [f"  得分: {report.correct}/{report.total} = {report.score:.2f}"]
    for g in report.grades:
        mark = "✓" if g.is_correct else "✗"
        gap = f"(盲点: {g.knowledge_gap})" if (not g.is_correct and g.knowledge_gap) else ""
        lines.append(f"  {mark} {g.question[:40]} {gap}")
    return "\n".join(lines)


def _normalize(d: NextStepDecision, goal: str, allow_teach: bool = True) -> NextStepDecision:
    """归一化非法枚举 + clamp 数值 + 填默认,防 LLM 乱填把下游搞崩。
    allow_teach=False(上一步刚讲过)时,把 teach 降级为 remediate,避免连续只讲不练。"""
    if d.action not in _ACTIONS:
        d.action = "continue"
    if d.action == "teach" and not allow_teach:
        d.action = "remediate"
    if d.difficulty not in _DIFF_WORDS:
        d.difficulty = "medium"
    if d.question_type not in _TYPES:
        d.question_type = "choice"
    try:
        d.difficulty_score = max(0.0, min(1.0, float(d.difficulty_score)))
    except (TypeError, ValueError):
        d.difficulty_score = 0.5
    d.count = max(1, min(10, int(d.count) if d.count else 3))
    if not (d.topic or "").strip():
        d.topic = goal
    return d


def _rule_fallback(last_report: Optional[GradingReport], goal: str,
                   weak_points: list[str]) -> NextStepDecision:
    """LLM 决策失败时的规则兜底(fail-soft):
       开场 → medium 巩固;有结果 → 得分<0.5 降难度补薄弱点,否则升难度。"""
    if last_report is None:
        return NextStepDecision(
            action="continue", topic=goal, difficulty="medium", difficulty_score=0.5,
            count=3, reason="冷启动,中等难度开场(规则兜底)",
        )
    if last_report.score < 0.5:
        return NextStepDecision(
            action="remediate", topic=goal, difficulty="easy", difficulty_score=0.3,
            target_weak_points=weak_points[:3], count=3,
            reason="本轮得分偏低,降难度并针对薄弱点重练(规则兜底)",
        )
    return NextStepDecision(
        action="advance", topic=goal, difficulty="hard", difficulty_score=0.75, count=3,
        reason="本轮得分良好,升难度进阶(规则兜底)",
    )


async def decide_next_step(
    *,
    goal: str,
    mastery: Optional[float],
    weak_points: list[str],
    history: list[AdaptiveTurn],
    last_report: Optional[GradingReport],
    allow_teach: bool = True,
    client: Optional[AsyncOpenAI] = None,
) -> NextStepDecision:
    """LLM 推理下一步教学动作。

    last_report=None 表示开场(还没答过题)→ agent 据目标和掌握度定开场策略。
    allow_teach=False(上一步刚讲过)→ 禁止再次 teach,强制进入出题验证。
    任何异常都回退规则兜底,保证闭环不中断。
    """
    parts = [
        f"学习目标: {goal}",
        f"当前掌握度(EMA): {f'{mastery:.2f}' if mastery is not None else '无历史(冷启动)'}",
    ]
    if weak_points:
        parts.append(f"已知薄弱点: {', '.join(weak_points[:8])}")
    if history:
        parts.append("历史轨迹:\n" + _format_history(history))
    if last_report is not None:
        parts.append("本轮答题结果:\n" + _format_report(last_report))
    else:
        parts.append("这是第一轮,还没有答题数据,请据目标和掌握度给出开场出题策略。")
    if not allow_teach:
        parts.append("注意:上一步已经给学生讲解过了,这一步请用出题验证学习效果,不要再选 teach。")
    user_msg = "\n\n".join(parts)

    try:
        # 统一 LLM 入口：带退避重试 + client 注入（保留测试 mock 能力）。
        resp = await llm_parse(
            [
                {"role": "system", "content": _DECIDE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            response_format=NextStepDecision,
            client=client,
            model=_model,
            max_tokens=512,
        )
        decision = resp.choices[0].message.parsed
        if decision is None:
            raise ValueError("parsed decision is None")
        logger.info(f"[adaptive] decide: {decision.action} | {decision.topic} | "
                    f"diff={decision.difficulty_score:.2f} | {decision.reason[:50]}")
        return _normalize(decision, goal, allow_teach)
    except Exception as e:
        logger.warning(f"[adaptive] decide LLM failed, rule fallback: {e}")
        return _rule_fallback(last_report, goal, weak_points)


def should_terminate(
    *, mastery: Optional[float], turn: int, decision: NextStepDecision,
) -> tuple[bool, str]:
    """终止判定(优先级:agent 主动结束 > 掌握度达标 > 轮数上限)。"""
    if decision.action == "finish":
        return True, "agent_finish"
    if mastery is not None and mastery >= MASTERY_TARGET:
        return True, "mastery_reached"
    if turn >= MAX_TURNS:
        return True, "max_turns"
    return False, ""


# ── teach:纯讲解(检索材料 + 讲清概念 + 举例 + 点误区,不出题、不苏格拉底)──────────
_LESSON_SYSTEM = (
    "你是讲解老师。根据提供的资料,针对学生的薄弱知识点,给一段清晰的**纯讲解**:\n"
    "1. 先用两三句话讲清核心概念;\n"
    "2. 再举一个具体例子说明;\n"
    "3. 最后点明学生在这个点上常见的误区。\n"
    "要求:直接讲解,**不要用反问或苏格拉底式提问,不要出题**,不要客套话。"
)


async def generate_lesson(
    *,
    document_id: str,
    topic: str,
    weak_points: list[str],
    last_report: Optional[GradingReport] = None,
    client: Optional[AsyncOpenAI] = None,
) -> str:
    """teach 动作的内容:检索该知识点材料 → LLM 生成纯讲解。失败返回降级文本(不中断闭环)。"""
    try:
        result = await retrieve_with_rewrite(document_id, topic, n_results=4)
        chunks = result.get("documents", [[]])[0] or []
    except Exception as e:
        logger.warning(f"[adaptive] lesson retrieval failed: {e}")
        chunks = []
    material = "\n\n".join(chunks[:4]) if chunks else "(未检索到资料,凭通用知识讲解)"

    gaps = "、".join(weak_points[:5]) if weak_points else topic
    wrong = ""
    if last_report is not None:
        wrong_qs = [g.question for g in last_report.grades if not g.is_correct]
        if wrong_qs:
            wrong = "学生刚做错的题:\n" + "\n".join(f"- {q[:60]}" for q in wrong_qs[:3])

    user_msg = f"知识点 / 薄弱点:{gaps}\n\n参考资料:\n{material}\n\n{wrong}"
    try:
        # 统一 LLM 入口：带退避重试 + client 注入。
        resp = await llm_chat(
            [
                {"role": "system", "content": _LESSON_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            client=client,
            model=_model,
            max_tokens=600,
            temperature=0.4,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text or f"关于「{gaps}」的讲解生成为空,请重试。"
    except Exception as e:
        logger.warning(f"[adaptive] lesson generation failed: {e}")
        return f"关于「{gaps}」的讲解暂时不可用(LLM 调用失败)。"
