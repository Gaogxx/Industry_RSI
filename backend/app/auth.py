"""Bearer Token 鉴权（最小权限：读写分离）。"""

import hmac

from fastapi import HTTPException, Request
from starlette.types import ASGIApp, Receive, Scope, Send

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


class BearerASGI:
    """包裹 MCP 的 ASGI 应用（流式响应友好）。"""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
            if not _token_ok(headers.get("authorization")):
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"content-type", b"application/json"), (b"www-authenticate", b"Bearer")]})
                await send({"type": "http.response.body", "body": b'{"detail":"unauthorized"}'})
                return
        await self.app(scope, receive, send)
