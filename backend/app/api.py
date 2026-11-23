"""REST API（看板 / MES·YMS 集成 / 第三方调用）。"""

from __future__ import annotations

import importlib.metadata as md
import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlmodel import Session, col, select

from . import __version__
from .agent.workflow import workflow_steps
from .auth import actor_of, read_access, require_key
from .config import get_settings
from .db import get_session
from .fab.flow import CAUSES, STAGES, STEPS, SYMPTOMS
from .fab.learning import edges_view, run_backtest
from .improvement import cause_pareto, compute_kpis, compute_pareto, learning_snapshot, run_cycle, tool_hotspots
from .models import (ACTION_KINDS, ACTION_STATUSES, BULLET_STATUSES, CASE_STATUSES, MEMORY_KINDS, SEVERITIES, SIX_M, SOURCES,
                     Action, Analysis, AuditLog, BacktestRun, Case, FabDataset, Feedback, ImprovementCycle, LearningEvent, MemoryItem,
                     utcnow)
from .retrieval import search_cases
from .services import NotFound, analyze_case, audit, case_dict, close_case, create_case, set_memory_status, update_action

router = APIRouter(prefix="/api")
R = [Depends(read_access)]
W = [Depends(require_key)]


def _ver(p: str) -> str:
    try:
        return md.version(p)
    except md.PackageNotFoundError:
        return "-"


class CaseIn(BaseModel):
    title: str = Field(min_length=2, max_length=200)
    description: str = ""
    symptom: Optional[str] = None
    product: str = ""
    severity: str = "S2"
    site: str = ""
    line: str = ""
    equipment: str = ""
    data: dict = {}
    cost_impact: float = 0
    downtime_hours: float = 0
    reporter: str = ""
    source: str = "api"


class ActionPatch(BaseModel):
    status: Optional[str] = None
    result: Optional[str] = None


class CloseIn(BaseModel):
    cause_code: str = Field(min_length=2)
    root_cause: Optional[str] = None


class BulletStatus(BaseModel):
    active: bool


class FeedbackIn(BaseModel):
    analysis_id: int
    rating: int = Field(ge=1, le=5)
    comment: str = ""


def _demo_state() -> dict:
    from .seed import STATE

    return dict(STATE) if get_settings().seed_demo else {"state": "off"}


@router.get("/meta", dependencies=R)
def meta():
    s = get_settings()
    return {
        "version": __version__,
        "llm": {"offline": s.offline, "model": "offline-rules" if s.offline else s.llm_model},
        "stack": {"ag_ui": _ver("ag-ui-protocol"), "dbos": _ver("dbos"), "pydantic_ai": _ver("pydantic-ai-slim"),
                  "pydantic_ai_harness": _ver("pydantic-ai-harness"), "mcp": _ver("mcp"), "fastapi": _ver("fastapi")},
        "stages": STAGES,
        "symptoms": {k: {"name": v.name, "stage": v.stage, "metric": v.metric} for k, v in SYMPTOMS.items()},
        "causes": {k: [{"code": c.code, "name": c.name, "step": c.step, "six_m": c.six_m} for c in v] for k, v in CAUSES.items()},
        "steps": {k: v.name for k, v in STEPS.items()},
        "severities": SEVERITIES, "case_statuses": CASE_STATUSES, "six_m": SIX_M, "action_kinds": ACTION_KINDS,
        "action_statuses": ACTION_STATUSES, "bullet_statuses": BULLET_STATUSES, "memory_kinds": MEMORY_KINDS, "sources": SOURCES,
        "auth_required_for_write": bool(s.api_key), "demo": _demo_state(),
    }


@router.get("/kpis", dependencies=R)
def kpis(db: Session = Depends(get_session)):
    last = db.exec(select(ImprovementCycle).order_by(col(ImprovementCycle.id).desc())).first()
    return {"kpis": compute_kpis(db), "pareto": compute_pareto(db), "causes": cause_pareto(db),
            "hotspots": tool_hotspots(db), "learning": learning_snapshot(db),
            "latest_cycle": last.model_dump() if last else None}


@router.get("/cases", dependencies=R)
def list_cases(stage: str | None = None, status: str | None = None, symptom: str | None = None, q: str | None = None,
               limit: int = 300, db: Session = Depends(get_session)):
    stmt = select(Case).order_by(col(Case.created_at).desc())
    for field, val in (("stage", stage), ("status", status), ("symptom", symptom)):
        if val:
            stmt = stmt.where(getattr(Case, field) == val)
    cases = db.exec(stmt).all()
    if q:
        cases = [c for c in cases if q.lower() in f"{c.title}{c.description}{c.equipment}{c.root_cause}".lower()]
    acts = db.exec(select(Action)).all()
    now = utcnow()
    out = []
    for c in cases[:limit]:
        d = case_dict(c)
        d.pop("evidence", None)
        mine = [a for a in acts if a.case_id == c.id]
        d["actions_total"] = len(mine)
        d["actions_open"] = sum(1 for a in mine if a.status in ("todo", "doing"))
        d["actions_overdue"] = sum(1 for a in mine if a.status in ("todo", "doing") and a.due_date and a.due_date < now)
        out.append(d)
    return out


@router.post("/cases", dependencies=W, status_code=201)
def new_case(body: CaseIn, request: Request, db: Session = Depends(get_session)):
    return case_dict(create_case(db, body.model_dump(), actor=actor_of(request)))


@router.get("/cases/{case_id}", dependencies=R)
def get_case(case_id: int, db: Session = Depends(get_session)):
    c = db.get(Case, case_id)
    if not c:
        raise HTTPException(404, "案例不存在")
    analyses = db.exec(select(Analysis).where(Analysis.case_id == case_id).order_by(col(Analysis.id).desc())).all()
    actions = db.exec(select(Action).where(Action.case_id == case_id).order_by(col(Action.id))).all()
    fbs = db.exec(select(Feedback).where(col(Feedback.analysis_id).in_([a.id for a in analyses] or [0]))).all()
    events = db.exec(select(LearningEvent).where(LearningEvent.case_id == case_id).order_by(col(LearningEvent.id))).all()
    ds = db.get(FabDataset, case_id)
    dataset = None
    if ds:
        d = ds.data
        ev = (analyses[0].evidence if analyses else None) or c.evidence or {}
        top = (ev.get("suspects") or [{}])[0]
        fdc_tool = top.get("value") if top and "#" not in top.get("factor", "#") and top.get("score", 0) >= 1.3 else None
        dataset = {"lots": len(d["lots"]), "fails": sum(1 for x in d["lots"] if x["fail"]), "route": d["route"],
                   "metric": d.get("metric"), "events": d.get("events", []), "wafers": d.get("wafers", []),
                   "fdc_tool": fdc_tool, "fdc": (d.get("fdc", {}).get(fdc_tool) or [])[-80:] if fdc_tool else [],
                   "series": [{"time": x["time"], "v": x.get("yield", x.get("fail_rate")), "fail": x["fail"],
                               "hit": bool(fdc_tool and x["steps"].get(top.get("factor")) == fdc_tool)} for x in d["lots"]]}
    return {"case": case_dict(c), "analyses": [a.model_dump() for a in analyses], "actions": [a.model_dump() for a in actions],
            "feedback": [f.model_dump() for f in fbs], "learning_events": [e.model_dump() for e in events],
            "dataset": dataset, "candidates": [{"code": x.code, "name": x.name} for x in CAUSES.get(c.symptom, [])]}


@router.post("/cases/{case_id}/analyze", dependencies=W)
async def run_analysis(case_id: int, request: Request, stream: bool = True):
    actor = actor_of(request)
    if not stream:
        try:
            result = None
            async for ev in analyze_case(case_id, actor=actor):
                if ev["type"] == "done":
                    result = ev
            return result
        except NotFound as e:
            raise HTTPException(404, str(e))

    async def gen():
        try:
            async for ev in analyze_case(case_id, actor=actor):
                yield f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"
        except NotFound as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/cases/{case_id}/close", dependencies=W)
async def close(case_id: int, body: CloseIn, request: Request):
    try:
        return await close_case(case_id, cause_code=body.cause_code, root_cause=body.root_cause, actor=actor_of(request))
    except NotFound as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))


@router.get("/actions", dependencies=R)
def list_actions(status: str | None = None, overdue: bool = False, db: Session = Depends(get_session)):
    stmt = select(Action).order_by(col(Action.due_date).desc())
    if status:
        stmt = stmt.where(Action.status == status)
    now = utcnow()
    cases = {c.id: c for c in db.exec(select(Case)).all()}
    out = []
    for a in db.exec(stmt).all():
        od = a.status in ("todo", "doing") and bool(a.due_date and a.due_date < now)
        if overdue and not od:
            continue
        c = cases.get(a.case_id)
        out.append({**a.model_dump(), "overdue": od, "case_code": c.code if c else "", "case_title": c.title if c else ""})
    return out


@router.patch("/actions/{action_id}", dependencies=W)
def patch_action(action_id: int, body: ActionPatch, request: Request, db: Session = Depends(get_session)):
    try:
        return update_action(db, action_id, status=body.status, result=body.result, actor=actor_of(request))
    except NotFound as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))


# ── 持续学习 ──────────────────────────────────────────────
@router.get("/learning/curve", dependencies=R)
def learning_curve(db: Session = Depends(get_session)):
    runs = db.exec(select(BacktestRun).order_by(col(BacktestRun.id).desc()).limit(20)).all()
    if not runs:
        return {"latest": None, "history": []}
    return {"latest": runs[0].model_dump(),
            "history": [{"id": r.id, "trigger": r.trigger, "created_at": r.created_at, **r.summary} for r in reversed(runs)]}


@router.post("/learning/backtest", dependencies=W)
def learning_backtest(request: Request, db: Session = Depends(get_session)):
    run = run_backtest(db, trigger=actor_of(request, "manual"))
    return run.model_dump()


@router.get("/learning/edges", dependencies=R)
def learning_edges(db: Session = Depends(get_session)):
    return edges_view(db)


@router.get("/learning/events", dependencies=R)
def learning_events(limit: int = 100, db: Session = Depends(get_session)):
    cases = {c.id: c.code for c in db.exec(select(Case)).all()}
    rows = db.exec(select(LearningEvent).order_by(col(LearningEvent.id).desc()).limit(limit)).all()
    return [{**e.model_dump(), "case_code": cases.get(e.case_id, ""),
             "symptom_name": SYMPTOMS[e.symptom].name if e.symptom in SYMPTOMS else e.symptom} for e in rows]


@router.get("/memory", dependencies=R)
def memory(db: Session = Depends(get_session)):
    rows = db.exec(select(MemoryItem).order_by(col(MemoryItem.updated_at).desc())).all()
    return [{**b.model_dump(), "symptom_name": SYMPTOMS[b.symptom].name if b.symptom in SYMPTOMS else b.symptom} for b in rows]


@router.post("/memory/{item_id}/status", dependencies=W)
def memory_status(item_id: int, body: BulletStatus, request: Request, db: Session = Depends(get_session)):
    try:
        return set_memory_status(db, item_id, body.active, actor=actor_of(request))
    except NotFound as e:
        raise HTTPException(404, str(e))


@router.get("/memory/hotspots", dependencies=R)
def memory_hotspots(db: Session = Depends(get_session)):
    from .fab.learning import hotspots_view

    return hotspots_view(db)


# ── RSI 递归自我改进 ─────────────────────────────────────
@router.get("/rsi/sessions", dependencies=R)
def rsi_sessions(db: Session = Depends(get_session)):
    from .models import RSISession

    return [{k: v for k, v in r.model_dump().items() if k != "curve"} | {"points": len(r.curve)}
            for r in db.exec(select(RSISession).order_by(col(RSISession.id).desc()).limit(30)).all()]


@router.post("/rsi/sessions", dependencies=W)
async def rsi_start(request: Request):
    from .rsi.workflow import start_session

    sid, wf = await start_session(trigger=actor_of(request, "manual"))
    return {"session_id": sid, "workflow_id": wf}


@router.get("/rsi/sessions/{sid}", dependencies=R)
def rsi_session(sid: int, db: Session = Depends(get_session)):
    from .models import RSISession, RSITask

    sess = db.get(RSISession, sid)
    if not sess:
        raise HTTPException(404, "会话不存在")
    tasks = db.exec(select(RSITask).where(RSITask.session_id == sid).order_by(col(RSITask.id))).all()

    def slim(t: RSITask) -> dict:  # 证据原文只用于训练，不下发；校准题量大，只给摘要
        d = t.model_dump(exclude={"evidence"})
        if t.phase == "CAL":
            d["attempts"] = [{"cause_code": a["cause_code"], "verdict": {"passed": a["verdict"]["passed"]}} for a in t.attempts]
        return d

    return {"session": sess.model_dump(), "tasks": [slim(t) for t in tasks]}


@router.get("/learning/weights", dependencies=R)
def learning_weights(db: Session = Depends(get_session)):
    """证据权重（各类证据的可信度）的版本历史：每一版都经过验收门，标明来自孪生练习还是产线确认。"""
    from .fab.kg import FACTOR_NAMES
    from .models import EvidenceModel

    rows = db.exec(select(EvidenceModel).order_by(col(EvidenceModel.id).desc()).limit(60)).all()
    by_source: dict[str, int] = {}
    for src in db.exec(select(EvidenceModel.source)).all():
        by_source[src] = by_source.get(src, 0) + 1
    return {"factors": list(FACTOR_NAMES), "current": rows[0].model_dump() if rows else None,
            "history": [r.model_dump() for r in reversed(rows)], "by_source": by_source}


@router.get("/rsi/benchmark", dependencies=R)
def rsi_benchmark(db: Session = Depends(get_session)):
    from .models import RSIBenchmark

    b = db.exec(select(RSIBenchmark).order_by(col(RSIBenchmark.id).desc())).first()
    return b.model_dump() if b else None


@router.post("/rsi/benchmark", dependencies=W)
async def rsi_benchmark_run(request: Request, db: Session = Depends(get_session)):
    from .models import RSIBenchmark
    from .rsi.workflow import run_benchmark

    bid = await run_benchmark(trigger=actor_of(request, "manual"))
    return db.get(RSIBenchmark, bid).model_dump()


@router.get("/rsi/gaps", dependencies=R)
def rsi_gaps():
    from .rsi.workflow import gaps_view

    return gaps_view()


@router.get("/rsi/twin", dependencies=R)
def rsi_twin():
    """孪生的说明信息（不暴露真实分布与热点；这些只在事后审计的单题记录里出现）。"""
    from .rsi.twin import DIFFICULTY, EVAL_PER_SYMPTOM, SITE_TRUTH

    return {"symptoms": list(SITE_TRUTH), "difficulty": DIFFICULTY, "eval_per_symptom": EVAL_PER_SYMPTOM,
            "eval_size": EVAL_PER_SYMPTOM * len(SITE_TRUTH)}


@router.get("/workflows/{wf_id}/steps", dependencies=R)
async def wf_steps(wf_id: str):
    return await workflow_steps(wf_id)


# ── 改进周期 / 其它 ───────────────────────────────────────
@router.post("/improvement/run", dependencies=W)
async def improvement_run(request: Request):
    return (await run_cycle(trigger=actor_of(request, "manual"))).model_dump()


@router.get("/improvement/cycles", dependencies=R)
def cycles(limit: int = 20, db: Session = Depends(get_session)):
    return db.exec(select(ImprovementCycle).order_by(col(ImprovementCycle.id).desc()).limit(limit)).all()


@router.post("/feedback", dependencies=W, status_code=201)
def feedback(body: FeedbackIn, request: Request, db: Session = Depends(get_session)):
    if not db.get(Analysis, body.analysis_id):
        raise HTTPException(404, "分析不存在")
    fb = Feedback(**body.model_dump())
    db.add(fb)
    audit(db, actor_of(request), "analysis.feedback", f"analysis:{body.analysis_id}", rating=body.rating)
    db.commit()
    db.refresh(fb)
    return fb


@router.get("/search", dependencies=R)
async def search(q: str, stage: str | None = None, limit: int = 5, db: Session = Depends(get_session)):
    out = []
    for h in await search_cases(db, q, stage=stage, limit=limit):
        d = case_dict(h.case)
        d.pop("evidence", None)
        out.append({**d, "score": h.score})
    return out


@router.get("/audit", dependencies=R)
def audit_log(limit: int = 100, db: Session = Depends(get_session)):
    return db.exec(select(AuditLog).order_by(col(AuditLog.created_at).desc(), col(AuditLog.id).desc()).limit(limit)).all()
