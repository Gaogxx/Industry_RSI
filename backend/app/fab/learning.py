"""产线侧学习（真实环境反馈）：工程师确认根因 / CAPA 验证结果 → 合并进可进化记忆（权重 1.0）。

与孪生侧（app/rsi）共用同一套记忆与合并逻辑；另外提供真实案例的预序回放（学习曲线）与只读视图。
"""

from __future__ import annotations

from sqlmodel import Session, col, select

from ..models import BacktestRun, Case, CausalEdge, LearningEvent, MemoryItem, ToolHotspot
from ..rsi.memory import consolidate_fail, consolidate_pass, ensure_textbook, load_memory, save_memory
from .flow import CAUSES, STEPS, SYMPTOMS
from .kg import prequential, prior_edges, rank


def ensure_edges(db: Session) -> None:
    ensure_textbook(db)
    db.commit()


def rank_case(db: Session, evidence: dict, symptom: str) -> tuple[list, list]:
    """返回 (用当前记忆的排序, 教科书先验的排序)。分析时记忆是冻结的（只读）。"""
    mem = load_memory(db)
    return rank(evidence, symptom, mem.edges, mem.hotspot_weights(), mem.weights), rank(evidence, symptom, prior_edges())


def _store(db: Session, events: list[dict], case_id: int | None, when=None, rsi_task_id: int | None = None) -> list[LearningEvent]:
    rows = []
    for e in events:
        ev = LearningEvent(case_id=case_id, rsi_task_id=rsi_task_id, **{k: v for k, v in e.items() if k in LearningEvent.model_fields})
        if when:
            ev.created_at = when
        db.add(ev)
        rows.append(ev)
    return rows


def learn_from_closure(db: Session, case: Case, cause_code: str, ranked_codes: list[str], when=None) -> list[LearningEvent]:
    mem = load_memory(db)
    events = consolidate_pass(mem, case.symptom, cause_code, ranked_codes, case.evidence or {}, source="real", ref=case.id,
                              cause_name=case.root_cause, six_m=case.root_cause_category)
    save_memory(db, mem, when)
    rows = _store(db, events, case.id, when)
    db.commit()
    return rows


def refute_cause(db: Session, case: Case, cause_code: str, when=None) -> list[LearningEvent]:
    mem = load_memory(db)
    events = consolidate_fail(mem, case.symptom, cause_code, case.evidence or {}, source="real", ref=case.id)
    save_memory(db, mem, when)
    rows = _store(db, events, case.id, when)
    db.commit()
    return rows


def memory_for(db: Session, symptom: str, limit: int = 5) -> list[MemoryItem]:
    rows = db.exec(select(MemoryItem).where(MemoryItem.symptom == symptom, MemoryItem.status == "active")).all()
    return sorted(rows, key=lambda b: -(b.passes - b.fails))[:limit]


def run_backtest(db: Session, trigger: str = "manual", when=None) -> BacktestRun:
    cases = db.exec(select(Case).where(Case.status == "closed", Case.cause_code != "").order_by(col(Case.closed_at))).all()
    payload = [{"code": c.code, "time": c.closed_at.isoformat() if c.closed_at else "", "symptom": c.symptom,
                "cause_code": c.cause_code, "evidence": c.evidence or {}} for c in cases if c.symptom in CAUSES]
    from ..models import AuditLog

    changes = [(r.created_at.isoformat(), r.target) for r in db.exec(select(AuditLog).where(AuditLog.action == "twin.update")).all()]
    result = prequential(payload, changes)
    run = BacktestRun(trigger=trigger, points=result["points"], summary=result["summary"])
    if when:
        run.created_at = when
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def edges_view(db: Session) -> list[dict]:
    rows = db.exec(select(CausalEdge)).all()
    tot, tot0 = {}, {}
    for r in rows:
        tot[r.symptom] = tot.get(r.symptom, 0) + r.alpha
        tot0[r.symptom] = tot0.get(r.symptom, 0) + r.alpha0
    out = []
    for r in rows:
        mean, prior = r.alpha / (tot[r.symptom] or 1), r.alpha0 / (tot0[r.symptom] or 1)
        out.append({"id": r.id, "symptom": r.symptom, "symptom_name": SYMPTOMS[r.symptom].name if r.symptom in SYMPTOMS else r.symptom,
                    "cause_code": r.cause_code, "name": r.name, "step": r.step,
                    "step_name": STEPS[r.step].name if r.step in STEPS else r.step, "six_m": r.six_m,
                    "prior": round(prior, 3), "posterior": round(mean, 3), "delta": round(mean - prior, 3),
                    "confirmed": r.confirmed, "confirmed_sandbox": r.confirmed_sandbox, "refuted": r.refuted,
                    "learned": r.learned, "alpha": round(r.alpha, 2), "updated_at": r.updated_at})
    return sorted(out, key=lambda x: (x["symptom"], -x["posterior"]))


def hotspots_view(db: Session) -> list[dict]:
    names = {c.code: c.name for cs in CAUSES.values() for c in cs}
    rows = db.exec(select(ToolHotspot)).all()
    return sorted(({"cause_code": h.cause_code, "cause": names.get(h.cause_code, h.cause_code), "tool": h.tool,
                    "real": h.real, "sandbox": h.sandbox} for h in rows), key=lambda x: -(x["real"] + x["sandbox"]))
