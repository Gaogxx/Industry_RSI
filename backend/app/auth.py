"""Bearer Token 鉴权（最小权限：读写分离）。"""

import hmac

from fastapi import HTTPException, Request

from .config import get_settings


def _token_ok(header: str | None) -> bool:
    key = get_settings().api_key
    if not key:
        return True
    if not header or not header.lower().startswith("bearer "):
        return False
    return hmac.compare_digest(header[7:].strip(), key)


def require_key(request: Request) -> None:
    if not _token_ok(request.headers.get("authorization")):
        raise HTTPException(401, "缺少或错误的 API Key", headers={"WWW-Authenticate": "Bearer"})


def read_access(request: Request) -> None:
    if get_settings().dashboard_public:
        return
    require_key(request)


def actor_of(request: Request, default: str = "api") -> str:
    """审计用的操作者：反向代理（SSO）可通过 X-Kaizen-User 头传入登录用户名。"""
    name = request.headers.get("x-kaizen-user")
    return f"user:{name}" if name else default

