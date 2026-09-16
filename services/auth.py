"""可信的请求主体。

在此之前，`user_id` 是请求体里的一个普通字段：任何调用方都可以声称自己是任何人，
所以文档归属、学习画像和审计记录全都建立在一个客户端自称的值上。存储层已经按
owner 隔离（见 services/vectorstore.py），但只要主体本身可以伪造，隔离就没有意义。

本模块是请求身份的唯一来源。产品定位是单用户自托管，所以这里不做注册、口令散列
或 JWT，只做一个共享访问令牌：

  - 设置 STUDYLOOP_AUTH_TOKEN 后，所有业务端点都要求
    `Authorization: Bearer <token>`，否则 401；
  - 未设置时是匿名单用户模式，主体固定为 DEFAULT_DOCUMENT_OWNER。这保留了本地
    开发的零配置体验，但它**不是**安全边界：启动时告警，/health/live 也如实报告。

将来若引入真正的多用户，只需替换这里的主体解析，存储层不必改动。
"""
import hmac
import logging
import os
from typing import Optional

from fastapi import Header, HTTPException, status

from services.vectorstore import DEFAULT_DOCUMENT_OWNER

logger = logging.getLogger(__name__)

AUTH_TOKEN_ENV = "STUDYLOOP_AUTH_TOKEN"
_BEARER_PREFIX = "bearer "
# 过短的令牌挡不住猜测。拒绝启动好过给出虚假的安全感。
_MIN_TOKEN_LENGTH = 16


class AuthTokenConfigurationError(RuntimeError):
    """STUDYLOOP_AUTH_TOKEN 已设置但不可用。"""


def configured_token() -> Optional[str]:
    """返回已配置的令牌；未配置返回 None；配置了但不合格则抛错。"""
    raw = os.getenv(AUTH_TOKEN_ENV)
    if raw is None:
        return None
    token = raw.strip()
    if not token:
        return None
    if len(token) < _MIN_TOKEN_LENGTH:
        raise AuthTokenConfigurationError(
            f"{AUTH_TOKEN_ENV} 至少需要 {_MIN_TOKEN_LENGTH} 个字符"
        )
    return token


def auth_enabled() -> bool:
    return configured_token() is not None


def verify_configuration() -> None:
    """启动时调用：配置错误就让进程起不来，而不是每个请求各报一次 500。"""
    if auth_enabled():
        logger.info("access token gate enabled")
    else:
        logger.warning(
            "%s is not set: every caller is served as the single default user. "
            "This is convenient for local development and is not an access control boundary.",
            AUTH_TOKEN_ENV,
        )


def resolve_subject(authorization: Optional[str]) -> str:
    """返回本次请求的可信身份，失败抛 401。

    做成纯函数是为了让测试不必构造 FastAPI 依赖。
    """
    token = configured_token()
    if token is None:
        return DEFAULT_DOCUMENT_OWNER

    header = (authorization or "").strip()
    if header[: len(_BEARER_PREFIX)].lower() != _BEARER_PREFIX:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="需要 Authorization: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # 常数时间比较，避免按字符比较泄露令牌前缀。
    # compare_digest 对非 ASCII 的 str 会抛 TypeError，所以两侧都先编码。
    presented = header[len(_BEARER_PREFIX):].strip().encode("utf-8")
    if not hmac.compare_digest(presented, token.encode("utf-8")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="访问令牌无效",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return DEFAULT_DOCUMENT_OWNER


async def require_user_id(authorization: Optional[str] = Header(None)) -> str:
    """FastAPI 依赖：请求身份的唯一出处。"""
    return resolve_subject(authorization)


def require_own_subject(claimed: str, subject: str) -> str:
    """URL 里点名的身份必须就是调用方自己。

    只用于身份写在路径里的端点（/users/{user_id}/...）。这类请求里 URL 就是
    请求本身，静默换成别的身份会让返回内容和 URL 说的对不上；其余端点一律
    直接用 subject 覆盖请求体里自报的值。

    不匹配报 404 而不是 403：403 会确认这个用户确实存在，等于一个用户名枚举接口。
    """
    if claimed != subject:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在",
        )
    return subject
