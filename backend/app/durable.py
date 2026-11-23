"""DBOS 初始化：持久化工作流引擎在应用进程内运行，系统库默认 SQLite（无需额外服务）。"""

from __future__ import annotations

from pathlib import Path

from dbos import DBOS

from . import __version__
from .config import get_settings

_inited = False


def init_dbos() -> None:
    global _inited
    if _inited:
        return
    url = get_settings().dbos_database_url
    if url.startswith("sqlite:///"):
        Path(url.split("///", 1)[-1]).parent.mkdir(parents=True, exist_ok=True)
    from .agent import workflow  # noqa: F401  注册工作流与步骤

    DBOS(config={"name": "kaizen-agent", "system_database_url": url, "application_version": __version__,
                 "log_level": "WARNING", "console_log_level": "WARNING"})
    _inited = True


def launch() -> None:
    DBOS.launch()  # 同时恢复上次崩溃时未完成的工作流


def shutdown() -> None:
    global _inited
    DBOS.destroy(destroy_registry=False)
    _inited = False
