"""
Reviser Agent(Phase 9 P0-6:LangGraph 多 agent 范式 - reviewer↔reviser 循环子图)

借鉴 gpt-researcher/multi_agents/agents/editor.py:126-144 的图拓扑设计:
  reviewer --no_pass--> reviser --> reviewer (循环)
  reviewer --pass--> END

早期实现(线性回环):
  critic_adapter --no_pass--> quiz_agent (整轮重跑:重检索 + 重出全部题)
                                ↓
                              critic_adapter

当前实现(reviser 精修):
  critic_adapter --no_pass--> reviser (只修改有问题的题,保留好的)
                                ↓
                              critic_adapter

为什么 reviser 比"整轮重跑"更好?
  1. 速度:不再重新检索,只调一次 LLM 改题
  2. 稳定性:保留 critic 已认可的题目,避免好题被改坏(对照 aider 的 patch-style 思想)
  3. 成本:省一次 embedding + BM25 + RRF 全套
  4. 与 Task 1 reflection 回灌互补:reflected_message 是 prompt 层反思,
     reviser 是图拓扑层精修,两层叠加

设计选择:
  - reviser 不读 chunks 自己改;它信任 critic 给的 suggestions,只针对低分维度精修
  - 输出契约与 quiz_agent.generate 一致(QuizResponse.model_dump()),
    critic_adapter 重审时无感知节点切换
"""
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from agents.state import OrchestratorState
from models.quiz import QuizResponse
from services.tracing import traceable

load_dotenv(Path(__file__).parent.parent / ".env")
logger = logging.getLogger(__name__)

# 题目修订用 json_schema 结构化输出 → 走 structured 供应商
from services.llm import structured_client as _client, structured_model as _model


_REVISER_SYSTEM = (
    "你是题目精修专家。给定一组题目和审稿人的具体反馈,"
    "你的任务是**最小化修改**:只改审稿人指出的问题,完全保留没被指出问题的题目。\n\n"
    "改的时候:\n"
    "- 优先针对 high severity 的 suggestion 改\n"
    "- 改答案要从原文 chunks 里找证据,不能凭空造\n"
    "- 改难度要朝 difficulty_score 靠拢\n"
    "- 题目数量不变,题目顺序不变\n"
    "- explanation 重新写,体现修改后的依据"
)


@traceable(name="reviser_agent.run", run_type="llm")
async def reviser_agent(state: OrchestratorState) -> dict:
    """Reviser 节点:根据 critic 的最新 critique 精修题目。

    输入(从 OrchestratorState 读):
      - quiz: 当前待精修的题目(critic 刚审过的那一份)
      - critique_history[-1]: 最新一次评分 + suggestions
      - chunks(可选):原始证据,通过 reflected_message 间接传递

    输出(写回 OrchestratorState):
      - quiz: 精修后的新版本
      - revision_count: +1
    """
    quiz = state.get("quiz") or {}
    history = state.get("critique_history", [])
    latest_critique = history[-1] if history else {}
    reflected = state.get("reflected_message", "") or ""

    if not quiz or not latest_critique:
        logger.warning("[reviser] empty quiz or no critique, skip revision")
        return {}

    quiz_text = json.dumps(quiz, ensure_ascii=False)[:3000]
    critique_text = json.dumps(latest_critique, ensure_ascii=False)[:1500]

    user_msg = (
        f"【当前题目(JSON)】\n{quiz_text}\n\n"
        f"【审稿人最新评分(JSON)】\n{critique_text}\n\n"
        f"【审稿人格式化反馈】\n{reflected}\n\n"
        "请输出精修后的完整题目(同 QuizResponse 结构)。"
    )

    try:
        resp = await _client.beta.chat.completions.parse(
            model=_model,
            messages=[
                {"role": "system", "content": _REVISER_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            response_format=QuizResponse,
        )
        revised_quiz = resp.choices[0].message.parsed
        # 保留题型(critic_adapter 后续审 difficulty 时需要)
        original_type = state.get("type", "choice")
        for q in revised_quiz.questions:
            if not getattr(q, "type", None):
                q.type = original_type
        new_quiz_dict = revised_quiz.model_dump()
        logger.info(
            f"[reviser] revised {len(new_quiz_dict.get('questions', []))} questions "
            f"(round {state.get('revision_count', 0)})"
        )
    except Exception as e:
        logger.warning(f"[reviser] LLM call failed: {e}, return original quiz unchanged")
        new_quiz_dict = quiz  # 失败降级:保持原题不变,让 critic 决定是否退出

    return {
        "quiz": new_quiz_dict,
    }
