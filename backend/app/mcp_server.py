"""MCP 服务器（规范 2026-07-28，Streamable HTTP，含 MCP Apps 扩展）。

可选组件：Claude / VS Code / Cursor 等 MCP 客户端接入后，任意模型都能调用案例库、DBOS 分析工作流、
关案学习与学习曲线；`show_kaizen_board` 通过 MCP Apps 在支持的客户端里渲染交互看板（含学习曲线）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from mcp.server.apps import Apps
from mcp.server.mcpserver import MCPServer
from sqlmodel import col, select

from . import __version__
from .db import session_scope
from .fab.flow import CAUSES, STAGES, SYMPTOMS
from .fab.learning import edges_view
from .improvement import compute_kpis, learning_snapshot, run_cycle, tool_hotspots
from .models import ACTION_STATUSES, Action, Analysis, BacktestRun, Case, utcnow
from .retrieval import search_cases as _search
from .services import NotFound, analyze_case as _analyze, close_case as _close, create_case as _create
from .services import update_action as _update

BOARD_URI = "ui://kaizen/board.html"
apps = Apps()
apps.add_html_resource(BOARD_URI, (Path(__file__).parent / "static" / "mcp_board.html").read_text("utf-8"),
                       name="Kaizen 改进与学习看板", description="KPI、学习曲线、知识更新与进行中案例", prefers_border=True)


def _brief(c: Case) -> dict:
    return {"id": c.id, "code": c.code, "title": c.title, "stage": STAGES.get(c.stage, c.stage),
            "symptom": SYMPTOMS[c.symptom].name if c.symptom in SYMPTOMS else c.symptom, "status": c.status,
            "severity": c.severity, "root_cause": c.root_cause, "cause_code": c.cause_code}


# 注意：Apps 扩展在 MCPServer 构造时登记工具，UI 工具必须定义在构造之前
@apps.tool(resource_uri=BOARD_URI, name="show_kaizen_board",
           description="在聊天中显示 Kaizen 交互看板：KPI、学习曲线（持续学习 vs 冻结先验）、知识更新、进行中案例。")
def show_kaizen_board() -> dict:
    with session_scope() as db:
        k = compute_kpis(db)
        run = db.exec(select(BacktestRun).order_by(col(BacktestRun.id).desc())).first()
        open_cases = db.exec(select(Case).where(Case.status != "closed").order_by(col(Case.created_at).desc())).all()
        L = learning_snapshot(db)
        pts = run.points if run else []
        step = max(1, len(pts) // 40)
        return {"kpis": {key: k[key] for key in ("cases_total", "cases_open", "mttr_days", "capa_overdue",
                                                 "capa_effectiveness", "recurrence_rate")},
                "learning": {"summary": run.summary if run else None,
                             "curve": [{"i": p["i"], "l": p["cum_learning"], "f": p["cum_frozen"]} for p in pts[::step]],
                             "shifts": L["top_shifts"][:4]},
                "open_cases": [_brief(c) for c in open_cases[:6]]}


mcp = MCPServer(
    name="kaizen",
    title="Kaizen 芯片质量根因与持续学习",
    version=__version__,
    instructions=(
        "芯片制造（前道晶圆、2.5D 先进封装、HBM、FT/BI/SLT、整机柜）质量根因分析工具集。"
        "已有案例用 analyze_case（含共性分析、晶圆图、FDC、贝叶斯排序）；新问题先 search_cases 再 create_case；"
        "工程师确认根因后用 confirm_root_cause 触发持续学习；询问整体态势或学习效果用 show_kaizen_board / get_learning_status。"
    ),
    extensions=[apps],
)


@mcp.tool(description="检索相似的历史芯片质量案例（BM25 + 可选向量），返回症状、状态与已确认根因。")
async def search_cases(query: str, stage: Optional[str] = None, limit: int = 5) -> dict:
    with session_scope() as db:
        return {"results": [{**_brief(h.case), "score": h.score} for h in await _search(db, query, stage=stage, limit=limit)]}


@mcp.tool(description="获取案例详情：关键数据、最新分析的统计证据、贝叶斯根因排序（含若不学习的对比）与 CAPA。")
def get_case(case_id: int) -> dict:
    with session_scope() as db:
        c = db.get(Case, case_id)
        if not c:
            return {"error": f"案例 {case_id} 不存在"}
        an = db.exec(select(Analysis).where(Analysis.case_id == case_id).order_by(col(Analysis.id).desc())).first()
        acts = db.exec(select(Action).where(Action.case_id == case_id)).all()
        return {"case": {**_brief(c), "description": c.description, "data": c.data},
                "candidates": [{"code": x.code, "name": x.name} for x in CAUSES.get(c.symptom, [])],
                "analysis": {"id": an.id, "score": an.critique.get("score"),
                             "suspects": an.evidence.get("suspects", [])[:3], "wafer_pattern": an.evidence.get("wafer_pattern"),
                             "hypotheses": [{k: h[k] for k in ("cause_code", "name", "posterior")} for h in an.hypotheses],
                             "learning_changed_top": an.counterfactual.get("changed"),
                             "textbook_top": an.counterfactual.get("top_name")} if an else None,
                "actions": [{"id": a.id, "kind": a.kind, "title": a.title, "status": ACTION_STATUSES[a.status]} for a in acts]}


@mcp.tool(description="新建芯片质量案例。symptom 可省略（自动识别）；data 为现场关键指标键值对。")
def create_case(title: str, description: str, symptom: Optional[str] = None, product: str = "", severity: str = "S2",
                site: str = "", equipment: str = "", data: Optional[dict] = None) -> dict:
    with session_scope() as db:
        c = _create(db, {"title": title, "description": description, "symptom": symptom, "product": product,
                         "severity": severity, "site": site, "equipment": equipment, "data": data or {}, "source": "mcp"},
                    actor="mcp")
        return {"created": _brief(c)}


@mcp.tool(description="运行根因分析（DBOS 持久化工作流：统计证据 → 贝叶斯排序 → 经验手册 → 5-Why/CAPA → 评审），返回 8D 报告。")
async def analyze_case(case_id: int) -> dict:
    try:
        done, wf = None, None
        async for ev in _analyze(case_id, actor="mcp"):
            if ev["type"] == "workflow":
                wf = ev["workflow_id"]
            elif ev["type"] == "done":
                done = ev
        return {"code": done["code"], "workflow_id": wf, "score": done["score"], "top_cause": done["top"],
                "learning_changed_top": done["learning_changed_top"], "report_markdown": done["report_md"]}
    except NotFound as e:
        return {"error": str(e)}


@mcp.tool(description="工程师确认根因并关案，触发持续学习：更新贝叶斯知识图谱与经验手册、回放学习曲线，返回本次学到了什么。"
                      "cause_code 取 get_case 返回的 candidates；新根因可填任意新 code 并提供 root_cause 描述。")
async def confirm_root_cause(case_id: int, cause_code: str, root_cause: str = "") -> dict:
    try:
        r = await _close(case_id, cause_code=cause_code, root_cause=root_cause or None, actor="mcp")
        return {"closed": r["case"]["code"], "rank_in_analysis": r["rank_in_analysis"],
                "learned": [{"kind": e["kind"], "detail": e["detail"], "before": e["before"], "after": e["after"]}
                            for e in r["learning_events"]],
                "learning_curve": r["learning_summary"]}
    except (NotFound, ValueError) as e:
        return {"error": str(e)}


@mcp.tool(description="列出 CAPA 措施。status 可选 todo/doing/done/verified/ineffective；overdue_only=true 只看逾期。")
def list_actions(status: Optional[str] = None, overdue_only: bool = False, case_id: Optional[int] = None) -> dict:
    now = utcnow()
    with session_scope() as db:
        out = []
        for a in db.exec(select(Action).order_by(col(Action.due_date))).all():
            od = a.status in ("todo", "doing") and bool(a.due_date and a.due_date < now)
            if (status and a.status != status) or (overdue_only and not od) or (case_id and a.case_id != case_id):
                continue
            out.append({"id": a.id, "case_id": a.case_id, "kind": a.kind, "title": a.title, "status": a.status,
                        "due_date": str(a.due_date)[:10], "overdue": od})
        return {"actions": out[:50], "total": len(out)}


@mcp.tool(description="更新 CAPA 状态与验证结果。status=ineffective 时系统会把该措施针对的根因记为“被现场否定”（负向学习）。")
def update_action(action_id: int, status: str, result: str = "") -> dict:
    with session_scope() as db:
        try:
            a = _update(db, action_id, status=status, result=result or None, actor="mcp")
            return {"updated": {"id": a.id, "status": a.status, "result": a.result}}
        except (NotFound, ValueError) as e:
            return {"error": str(e)}


@mcp.tool(description="持续学习状态：学习曲线（持续学习 vs 冻结先验的首位命中率）、知识图谱最大变化、机台热点、经验手册规模。")
def get_learning_status() -> dict:
    with session_scope() as db:
        return {**learning_snapshot(db), "hotspots": tool_hotspots(db)[:5],
                "edges": [{k: e[k] for k in ("symptom_name", "name", "prior", "posterior", "confirmed", "refuted")}
                          for e in edges_view(db) if e["confirmed"] or e["refuted"]]}


@mcp.tool(description="获取 KPI：MTTR、CAPA 按期率/有效率、复发率、逾期措施等。")
def get_kpis() -> dict:
    with session_scope() as db:
        return {k: v for k, v in compute_kpis(db).items() if k != "trend"}


@mcp.tool(description="运行一次持续改进周期：KPI 体检、帕累托、学习曲线回放、关键发现。")
async def run_improvement_cycle() -> dict:
    cycle = await run_cycle(trigger="mcp")
    return {"cycle_id": cycle.id, "findings": cycle.findings, "summary_markdown": cycle.summary_md}
