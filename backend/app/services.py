"""业务服务层：案例生命周期、根因分析调度、CAPA、关案学习、审计。REST / OpenAI 兼容 / MCP 三个入口共用。"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from sqlmodel import Session, col, select

from .agent.workflow import run_rca
from .db import session_scope
from .fab.flow import SYMPTOMS, cause_by_code, guess_symptom
from .fab.learning import learn_from_closure, refute_cause, run_backtest
from .llm import LLM
from .models import ACTION_STATUSES, Action, Analysis, AuditLog, Case, LearningEvent, MemoryItem, utcnow


class NotFound(Exception):
    pass


def audit(db: Session, actor: str, action: str, target: str = "", **detail) -> None:
    db.add(AuditLog(actor=actor, action=action, target=target, detail=detail))


def case_dict(c: Case) -> dict:
    d = c.model_dump()
    d["code"] = c.code
    d["symptom_name"] = SYMPTOMS[c.symptom].name if c.symptom in SYMPTOMS else ""
    return d


_METRIC = re.compile(r"([一-鿿A-Za-z][一-鿿A-Za-z0-9()（）/%·µ²\-]{0,15}?)\s*(?:[:：=]|为|达到|降至|升至|低于|高于|由)\s*"
                     r"(-?\d+(?:\.\d+)?\s*(?:%|ppm|PPM|DPPM|nm|µm|um|mV|mm/s|N|℃|个/cm²|h|小时|天|片|颗|批|次)?)")


def extract_metrics(text: str, limit: int = 8) -> dict:
    out: dict = {}
    for k, v in _METRIC.findall(text):
        k = k.strip("，。、 的")
        if k and k not in out and not k.isdigit():
            out[k] = v.strip()
        if len(out) >= limit:
            break
    return out


_SITE = re.compile(r"(Fab ?[A-Z0-9]+|[一-鿿]{1,8}(?:厂|基地))")
_EQUIP = re.compile(r"\b([A-Z]{2,5}-\d{2}[A-Z]?)\b")


def case_from_text(text: str) -> dict:
    text = text.strip()
    clauses = [c for c in re.split(r"[。\n！!？?，,；;]", text) if c.strip()]
    title = clauses[0].strip() if clauses else text
    if len(title) < 8 and len(clauses) > 1:
        title = f"{title}，{clauses[1].strip()}"
    site, equip = _SITE.search(text), _EQUIP.search(text)
    sym = guess_symptom(text)
    return {"title": title[:40] + ("…" if len(title) > 40 else ""), "description": text, "data": extract_metrics(text),
            "site": site.group(1) if site else "", "equipment": equip.group(1) if equip else "",
            "symptom": sym, "stage": SYMPTOMS[sym].stage}


def create_case(db: Session, payload: dict, actor: str = "api") -> Case:
    fields = {k: v for k, v in payload.items() if k in Case.model_fields and k not in ("id", "status", "created_at", "cause_code")}
    if fields.get("symptom") not in SYMPTOMS:
        fields["symptom"] = guess_symptom(f"{payload.get('title', '')} {payload.get('description', '')}")
    fields["stage"] = SYMPTOMS[fields["symptom"]].stage
    case = Case(**fields)
    db.add(case)
    db.commit()
    db.refresh(case)
    audit(db, actor, "case.create", case.code, title=case.title, source=case.source)
    db.commit()
    return case


async def analyze_case(case_id: int, actor: str = "api", llm: LLM | None = None) -> AsyncIterator[dict]:
    """启动 DBOS 持久化工作流；事件：workflow / step / done / error。落库在工作流的 persist 步骤内完成。"""
    with session_scope() as db:
        case = db.get(Case, case_id)
        if not case:
            raise NotFound(f"案例 {case_id} 不存在")
        case.status = "analyzing"
        db.add(case)
        db.commit()
    async for ev in run_rca(case_id, actor=actor, llm=llm):
        yield ev


def update_action(db: Session, action_id: int, *, status: str | None = None, result: str | None = None,
                  actor: str = "api") -> Action:
    act = db.get(Action, action_id)
    if not act:
        raise NotFound(f"措施 {action_id} 不存在")
    if status:
        if status not in ACTION_STATUSES:
            raise ValueError(f"非法状态 {status}，可选：{list(ACTION_STATUSES)}")
        act.status = status
        if status in ("done", "verified", "ineffective") and not act.completed_at:
            act.completed_at = utcnow()
    if result is not None:
        act.result = result
    db.add(act)
    case = db.get(Case, act.case_id)
    siblings = db.exec(select(Action).where(Action.case_id == act.case_id)).all()
    if case and case.status == "capa" and siblings and all(a.status in ("done", "verified") for a in siblings):
        case.status = "verifying"
        db.add(case)
    db.commit()
    if case and status == "ineffective":  # 现场否定 → 回到 PDCA 的 Do，并让图谱“记住被误导过”
        case.status = "capa"
        case.closed_at = None
        db.add(case)
        if act.cause_code:
            refute_cause(db, case, act.cause_code)
    audit(db, actor, "action.update", f"A-{action_id}", status=status, result=result)
    db.commit()
    db.refresh(act)
    return act


async def close_case(case_id: int, *, cause_code: str, root_cause: str | None = None, actor: str = "api") -> dict:
    """关案 = 给系统一个经过现场验证的标签，触发学习。返回本次学到了什么。"""
    with session_scope() as db:
        case = db.get(Case, case_id)
        if not case:
            raise NotFound(f"案例 {case_id} 不存在")
        if case.status == "closed":
            raise ValueError("案例已关闭")
        an = db.exec(select(Analysis).where(Analysis.case_id == case_id).order_by(col(Analysis.id).desc())).first()
        ranked = [h["cause_code"] for h in (an.hypotheses if an else [])]
        found = cause_by_code(cause_code)
        if not found and not root_cause:
            raise ValueError("新根因需要填写 root_cause 描述")
        case.cause_code = cause_code
        case.root_cause = root_cause or found[1].name
        case.root_cause_category = found[1].six_m if found else (case.root_cause_category or "method")
        case.status = "closed"
        case.closed_at = utcnow()
        db.add(case)
        db.commit()
        rank_before = ranked.index(cause_code) + 1 if cause_code in ranked else None
        events = learn_from_closure(db, case, cause_code, ranked)
        from .rsi.memory import load_memory, refit_and_store

        w = refit_and_store(db, load_memory(db), source="production")  # 产线确认也是证据权重的训练标签
        run = run_backtest(db, trigger=f"close:{case.code}")
        audit(db, actor, "case.close", case.code, cause_code=cause_code, rank_in_analysis=rank_before,
              learning_events=len(events), weights_accepted=w["accepted"])
        db.commit()
        return {"case": case_dict(case), "rank_in_analysis": rank_before,
                "learning_events": [e.model_dump() for e in events], "learning_summary": run.summary,
                "weights": {"accepted": w["accepted"], "weights": w["weights"], "n": w["n"], "reason": w.get("reason", "")}}


def set_memory_status(db: Session, item_id: int, active: bool, actor: str = "api") -> MemoryItem:
    """人工启用/停用一条记忆（人在回路）。"""
    b = db.get(MemoryItem, item_id)
    if not b:
        raise NotFound(f"记忆条目 {item_id} 不存在")
    b.status = "active" if active else "retired"
    b.updated_at = utcnow()
    db.add(b)
    db.add(LearningEvent(kind="memory_manual", symptom=b.symptom, cause_code=b.cause_code,
                         detail=f"人工{'启用' if active else '停用'}记忆条目 #{b.id}"))
    audit(db, actor, "memory.activate" if active else "memory.retire", f"M-{item_id}")
    db.commit()
    db.refresh(b)
    return b
