"""
Audit 查询端点（Phase 8 P1-2；2026-06-03 加脱敏）

暴露 ToolRegistry 的内存 audit_log 给调试 / 评估使用。

端点：
  GET /audit/{run_id}          按 run_id 查工具调用全链路
  GET /audit?user_id=...       按 user_id 查（可选 tool_name 过滤）
  GET /audit/summary/overview  整体摘要：平均时长、错误率、p95

⚠️ 安全：该端点当前无认证。output_preview(检索到的文档正文片段) 与 arguments(query/topic 原文)
   若直接返回会造成跨用户信息泄露,故**默认脱敏**;排障时由可信环境用 include_payload=true 显式打开。
"""
from typing import Optional

from fastapi import APIRouter, Query

from services.tool_registry import tool_registry

router = APIRouter(prefix="/audit", tags=["audit"])


def _redact(d: dict) -> dict:
    """脱敏:移除可能含他人文档正文/查询原文的字段,只保留可观测元数据。"""
    safe = dict(d)
    safe.pop("output_preview", None)
    args = safe.get("arguments")
    if isinstance(args, dict):
        # 保留键名,值用「类型:长度」占位,避免泄露 query/topic 等原文
        safe["arguments"] = {k: f"<{type(v).__name__}:{len(str(v))}chars>" for k, v in args.items()}
    return safe


def _serialize(records, include_payload: bool) -> list[dict]:
    return [r.to_dict() if include_payload else _redact(r.to_dict()) for r in records]


@router.get("/{run_id}")
def get_audit_by_run(run_id: str, limit: int = 50, include_payload: bool = False):
    """按 run_id 查全链路工具调用。默认脱敏；include_payload=true 返回完整 payload（仅限可信环境）。"""
    records = tool_registry.get_audit(run_id=run_id, limit=limit)
    return {
        "run_id": run_id,
        "count": len(records),
        "calls": _serialize(records, include_payload),
    }


@router.get("")
def query_audit(
    user_id: Optional[str] = Query(None),
    tool_name: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    include_payload: bool = Query(False),
):
    """按 user_id / tool_name 灵活过滤。默认脱敏。"""
    records = tool_registry.get_audit(user_id=user_id, tool_name=tool_name, limit=limit)
    return {
        "filters": {"user_id": user_id, "tool_name": tool_name, "limit": limit},
        "count": len(records),
        "calls": _serialize(records, include_payload),
    }


@router.get("/summary/overview")
def audit_summary():
    """整体摘要：哪个 tool 最慢、错误率最高（不含敏感 payload）。"""
    return tool_registry.audit_summary()
