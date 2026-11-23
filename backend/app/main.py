"""Kaizen 服务入口。

    /            Kaizen 控制台（零构建 SPA：对话为主界面，左侧栏打开案例 / CAPA / 自我进化 …）
    /agui        AG-UI 1.0 端点（控制台对话与 RSI 直播）
    /api/*       REST API
    /mcp         可选：MCP Streamable HTTP（Claude / VS Code / Cursor 等 MCP 客户端调用工具）
    /healthz     健康检查
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.routing import Route

from . import __version__, durable
from .agui import router as agui_router
from .api import router as api_router
from .auth import BearerASGI
from .config import get_settings
from .db import init_db, session_scope
from .mcp_server import mcp
from .seed import prepare_demo

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("kaizen")

mcp_http = mcp.streamable_http_app(streamable_http_path="/mcp", host="0.0.0.0")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    init_db()
    durable.init_dbos()
    durable.launch()  # 恢复上次中断的分析工作流
    task = None
    if s.seed_demo:
        if s.seed_background:  # 控制台先启动，演示数据在后台生成（首次约 1 分钟），界面显示进度
            task = asyncio.create_task(prepare_demo(s.rsi_on_startup))
        else:
            await prepare_demo(s.rsi_on_startup)
    from .agent.workflow import ensure_schedule

    await ensure_schedule()
    log.info("LLM 模式：%s ｜ 编排：DBOS 持久化工作流 ｜ 夜间学习：%s",
             "离线（统计 + 贝叶斯内核）" if s.offline else f"{s.llm_model} @ {s.llm_base_url}", s.nightly_cron or "关闭")
    async with mcp.session_manager.run():
        yield
    if task and not task.done():
        task.cancel()
    durable.shutdown()


app = FastAPI(title="Kaizen", version=__version__, lifespan=lifespan,
              description="芯片制造质量根因分析与 RSI 递归自我改进：统计因果内核 + 可进化记忆 + 数字孪生 + DBOS + Pydantic AI + AG-UI")
app.include_router(api_router)
app.include_router(agui_router)
if get_settings().enable_mcp:
    for route in mcp_http.routes:  # 挂载 /mcp（带 Bearer 鉴权）
        if isinstance(route, Route):
            app.router.routes.append(Route(route.path, endpoint=BearerASGI(route.endpoint), methods=route.methods))


@app.get("/healthz")
def healthz():
    return {"ok": True, "version": __version__}


class _RevalidatingStatic(StaticFiles):
    """看板静态文件每次按 ETag 重新校验，升级后浏览器不会继续用旧脚本。"""

    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


app.mount("/", _RevalidatingStatic(directory=Path(__file__).parent / "static", html=True), name="static")
