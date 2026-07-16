"""
检索充分性判定 + Query 改写（Phase 8 升级 P1-1）

Pre-generation 质量门：retrieve 之后、generate 之前判断检索结果是否够生成高质题。

3 个 heuristic 信号（不调 LLM，省成本）：
  1. 数量信号 (硬门 gating)：chunks 数量必须 >= MIN_CHUNKS
  2. 多样性信号 (硬门)：chunks 前 30 字唯一数 > 1（防止重复段拼接被当 2 个有效 chunk）
  3. 覆盖信号 (软信号/非阻断)：weak_points 在合并 chunks 中的命中率 >= 0.5；
     低于门槛只标注 passed_low_coverage，不再让 sufficiency 失败。

判定逻辑：数量 AND 多样性（两条硬门全过才 sufficient）；覆盖率仅作诊断。
为什么硬门用 AND：
  - 用户最大痛点是 LLM 幻觉——chunks 与意图不相关时硬编答案。
  - OR 太松：只要"题数够"就过，但内容可能完全跑题。
  - AND 严格但安全：任一硬信号差就触发改写/降级，宁可降级出 2 题，不出 5 题幻觉。

为什么覆盖率从硬门降为软信号（2026-06-03）：
  - 覆盖率拿的是该文档的历史 weak_points，未按本次请求的目标方向 scope。
  - 学生在同一文档探索「新方向」时，新方向与历史薄弱点无关，覆盖率必然偏低；
    若硬 gating 会把合法请求误判为证据不足并白白降级（减题、降难度）。
  - 故降级为非阻断诊断：低覆盖只标注，相关性把关交给后置 critic（避免假阴性降级）。

为什么不用 LLM judge：
  - 已经有 post-generation 的 critic_agent 做 LLM 评估（双门设计，前置门 heuristic 后置门 LLM）
  - 这里再调 LLM 重复且贵
  - 80% 的"不充分"情况靠 heuristic 就能识别

Rewrite Query：LLM 把过窄/口语化 query 改写得更宽泛，加同义词、去具体限定。
"""
import logging
from typing import Literal

from services.llm import _client, model as _model

logger = logging.getLogger(__name__)


# ── 配置常量（写成模块级，便于面试讲点 + chunk_size sweep 调）─────────────────
MIN_CHUNKS = 2                  # 数量信号：少于 2 个 chunk 一律不充分
DIVERSITY_SAMPLE_CHARS = 30     # 多样性信号：取每 chunk 前 N 字判唯一
COVERAGE_THRESHOLD = 0.5        # 覆盖信号：weak_points 命中率门槛
MAX_REWRITES = 1                # query 改写上限（再不行就 degrade）

SufficiencyReason = Literal["passed", "passed_low_coverage", "too_few_chunks", "low_diversity"]


# ── 判定逻辑（纯函数，便于单测）──────────────────────────────────────────────
def check_sufficiency(
    chunks: list[str],
    weak_points: list[str] | None = None,
) -> tuple[bool, SufficiencyReason]:
    """判断检索 chunks 是否够生成高质量题。

    硬门 = 数量 AND 多样性（任一不过 → 不充分，触发改写/降级）。
    覆盖率为非阻断软信号：低覆盖仍 sufficient，只标注 passed_low_coverage。

    Returns:
        (is_sufficient, reason_code)
        reason_code 同时是 metadata 的诊断标签
    """
    # 1. 数量信号（硬门 gating）
    if len(chunks) < MIN_CHUNKS:
        return False, "too_few_chunks"

    # 2. 多样性信号（硬门）
    unique_heads = {(c or "")[:DIVERSITY_SAMPLE_CHARS] for c in chunks}
    if len(unique_heads) <= 1:
        return False, "low_diversity"

    # 3. 覆盖信号（软信号，仅诊断，不再 gating）
    #    覆盖率拿的是该文档的历史 weak_points，未按本次请求目标方向 scope；
    #    同文档探索新方向时覆盖率必然偏低，硬 gating 会误判合法请求为证据不足。
    #    故低覆盖只标注 passed_low_coverage，相关性把关交给后置 critic。
    if weak_points:
        combined = "".join(chunks).lower()
        hit = sum(1 for wp in weak_points if wp.strip() and wp.lower() in combined)
        coverage_ratio = hit / len(weak_points)
        if coverage_ratio < COVERAGE_THRESHOLD:
            return True, "passed_low_coverage"

    return True, "passed"


# ── Query 改写（LLM 调用，失败 fallback 返回原 query）────────────────────────
_REWRITE_SYSTEM = (
    "你是 RAG 检索 query 改写助手。任务：把用户的检索 query 改写得更宽泛、"
    "加同义词、去掉过于具体的限定，提升向量库召回率。\n"
    "规则：\n"
    "1. 保留核心主题，不要改变意图\n"
    "2. 加同义词或上位概念（如 'RAG' → 'RAG, 检索增强生成, retrieval augmented'）\n"
    "3. 去掉时间/数字/格式等过窄限定\n"
    "4. 直接输出改写后的 query 字符串，不要解释、不要 JSON、不要 markdown"
)


async def rewrite_query(
    description: str,
    weak_points: list[str] | None = None,
) -> str:
    """LLM 改写 query。失败时返回原 query（fallback），不抛异常。"""
    user_msg = f"原始 query：{description.strip()}"
    if weak_points:
        wp_str = "、".join(weak_points[:3])
        user_msg += f"\n用户薄弱点（改写时可参考）：{wp_str}"

    try:
        resp = await _client.chat.completions.create(
            model=_model,
            max_tokens=128,
            messages=[
                {"role": "system", "content": _REWRITE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
        )
        rewritten = (resp.choices[0].message.content or "").strip()
        # 简单清洗：去引号、去标点尾巴
        rewritten = rewritten.strip('"\'"').strip()
        if not rewritten or rewritten == description.strip():
            logger.info("[sufficiency] rewrite returned identical/empty, fallback to original")
            return description
        logger.info(f"[sufficiency] rewrote '{description[:40]}' → '{rewritten[:40]}'")
        return rewritten
    except Exception as e:
        logger.warning(f"[sufficiency] rewrite_query LLM call failed: {e}, fallback to original")
        return description
