from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from sqlalchemy import event
from sqlmodel import Session, SQLModel, create_engine

from .config import get_settings

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        url = get_settings().database_url
        kwargs: dict = {}
        if url.startswith("sqlite"):
            path = url.split("///", 1)[-1]
            if path and path != ":memory:":
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            kwargs["connect_args"] = {"check_same_thread": False}
        _engine = create_engine(url, **kwargs)
        if url.startswith("sqlite"):

            @event.listens_for(_engine, "connect")
            def _sqlite_pragmas(dbapi_conn, _):  # WAL：看板读与 Agent 写互不阻塞
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA foreign_keys=ON")
                cur.close()

    return _engine


def reset_engine() -> None:
    """测试用：切换 DATABASE_URL 后重建连接。"""
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = None


def init_db() -> None:
    from . import models  # noqa: F401  确保表已注册

    _backup_legacy_sqlite()
    SQLModel.metadata.create_all(get_engine())


def _backup_legacy_sqlite() -> None:
    """旧版本的 SQLite 演示库结构不兼容（缺列）：自动改名备份后重建，不做破坏性删除。生产环境请用迁移工具。"""
    from sqlalchemy import inspect

    from . import models  # noqa: F401  确保全部表已注册

    url = get_settings().database_url
    if not url.startswith("sqlite:///") or url.endswith(":memory:"):
        return
    path = Path(url.split("///", 1)[-1])
    if not path.exists():
        return
    insp = inspect(get_engine())
    stale = any(insp.has_table(t.name) and not {c.name for c in t.columns} <= {c["name"] for c in insp.get_columns(t.name)}
                for t in SQLModel.metadata.sorted_tables)
    if stale:
        reset_engine()
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        paths = [path]
        dbos_url = get_settings().dbos_database_url or ""
        if dbos_url.startswith("sqlite:///"):  # 业务库重建后，旧的工作流 ID（如 rsi-session-1）不能复用
            paths.append(Path(dbos_url.split("///", 1)[-1]))
        for base in paths:
            for suffix in ("", "-wal", "-shm"):
                p = Path(str(base) + suffix)
                if p.exists():
                    p.rename(Path(f"{base}{suffix}.{stamp}.bak"))


@contextmanager
def session_scope() -> Iterator[Session]:
    with Session(get_engine(), expire_on_commit=False) as session:
        yield session


def get_session() -> Iterator[Session]:
    """FastAPI 依赖。"""
    with Session(get_engine(), expire_on_commit=False) as session:
        yield session
