"""
A/B 评测端点

POST /eval/ab  →  运行一次 A/B 实验，返回两组对比 + LLM-as-Judge 评分

── 使用示例 ─────────────────────────────────────────────────────────────────
  # CE 实验：有/无 Context Engineering 对比
  curl -X POST http://localhost:8001/eval/ab -H 'Content-Type: application/json' -d '{
    "document_id": "product.txt",
    "query": "产品功能",
    "experiment": "ce",
    "weak_points": ["高速推理", "Qwen"],
    "difficulty_score": 0.65
  }'

  # RAG 实验：纯向量 vs Hybrid 检索对比
  curl -X POST http://localhost:8001/eval/ab -H 'Content-Type: application/json' -d '{
    "document_id": "product.txt",
    "query": "产品功能",
    "experiment": "rag"
  }'
"""
import logging
from fastapi import APIRouter
from models.eval import ABConfig, ABResult
from services.eval import run_ab_experiment

router = APIRouter(prefix="/eval", tags=["eval"])
logger = logging.getLogger(__name__)


@router.post("/ab", response_model=ABResult)
async def ab_experiment(config: ABConfig):
    """运行 A/B 评测实验。

    流程：检索 → 并发生成两组题目 → 并发 LLM-as-Judge → 聚合对比。
    调用依赖外部 LLM，前端应显示 loading 状态。
    """
    logger.info("[eval] A/B experiment started: experiment=%s", config.experiment)
    return await run_ab_experiment(config)
