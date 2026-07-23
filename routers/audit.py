"""
Audit 查询端点

暴露 ToolRegistry 的内存 audit_log 给调试 / 评估使用。

端点：
  GET /audit/{run_id}          按 run_id 查工具调用全链路
  GET /audit?user_id=...       按 user_id 查（可选 tool_name 过滤）
  GET /audit/summary/overview  整体摘要：平均时长、错误率、p95

⚠️ 安全：该端点当前无认证。output_preview(检索到的文档正文片段) 与 arguments(query/topic 原文)
   若直接返回会造成跨用户信息泄露，故默认脱敏。完整 payload 还要求可信环境显式设置
   AUDIT_PAYLOAD_ENABLED=true；该开关不替代认证。
"""
import os
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status

from services.tool_registry import tool_registry

router = APIRouter(prefix="/audit", tags=["audit"])
_AUDIT_PAYLOAD_ENABLED_ENV = "AUDIT_PAYLOAD_ENABLED"
_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


def _redact(d: dict) -> dict:
    """脱敏:移除可能含他人文档正文/查询原文的字段,只保留可观测元数据。"""
    safe = dict(d)
    safe.pop("output_preview", None)
    args = safe.get("arguments")
    if isinstance(args, dict):
        # 保留键名,值用「类型:长度」占位,避免泄露 query/topic 等原文
        safe["arguments"] = {k: f"<{type(v).__name__}:{len(str(v))}chars>" for k, v in args.items()}
    return safe


def _full_payload_enabled() -> bool:
    return (
        os.getenv(_AUDIT_PAYLOAD_ENABLED_ENV, "").strip().casefold()
        in _TRUTHY_ENV_VALUES
    )


def _serialize(records, include_payload: bool) -> list[dict]:
    if include_payload and not _full_payload_enabled():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "audit_payload_disabled",
                "message": (
                    "完整审计 payload 默认关闭；仅可在受信环境显式启用 "
                    f"{_AUDIT_PAYLOAD_ENABLED_ENV}=true"
                ),
            },
        )
    return [r.to_dict() if include_payload else _redact(r.to_dict()) for r in records]


@router.get("/{run_id}")
def get_audit_by_run(run_id: str, limit: int = 50, include_payload: bool = False):
    """按 run_id 查询工具调用；完整 payload 受环境开关保护。"""
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
    """按 user_id / tool_name 过滤；完整 payload 同时要求查询参数与环境开关。"""
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
