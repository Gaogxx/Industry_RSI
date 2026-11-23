"""持续改进周期（PDCA 的 Check + Act），由 DBOS 夜间定时、看板按钮、kaizen-coach 或 MCP 触发。

一次周期：KPI 体检 → 帕累托（6M / 根因 / 机台）→ 学习曲线回放（持续学习 vs 冻结先验）→ 关键发现。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import timedelta

from sqlmodel import Session, col, select

from .db import session_scope
from .fab.flow import STAGES, SYMPTOMS, cause_by_code
from .fab.learning import edges_view, run_backtest
from .models import (CASE_STATUSES, SIX_M, Action, Analysis, Case, Feedback, ImprovementCycle, LearningEvent,
                     MemoryItem, utcnow)
from .services import audit


def _cause_tool(c: Case) -> str | None:
    """已确认根因所在工序上的显著机台（用于机台热点与复发统计）。"""
    found = cause_by_code(c.cause_code)
    if not found:
        return None
    for s in (c.evidence or {}).get("suspects", []):
        if s["step"] == found[1].step and s.get("score", 0) >= 1.3 and "#" not in s["factor"]:
            return s["value"]
    return None


def compute_kpis(db: Session, now=None) -> dict:
    cases = db.exec(select(Case)).all()
    actions = db.exec(select(Action)).all()
    analyses = db.exec(select(Analysis)).all()
    feedback = db.exec(select(Feedback)).all()
    now = now or utcnow()
    closed = [c for c in cases if c.status == "closed" and c.closed_at]
    mttr = sum((c.closed_at - c.created_at).total_seconds() for c in closed) / 86400 / len(closed) if closed else 0
    finished = [a for a in actions if a.completed_at and a.due_date]
    on_time = sum(1 for a in finished if a.completed_at <= a.due_date)
    overdue = [a for a in actions if a.status in ("todo", "doing") and a.due_date and a.due_date < now]
    verified = sum(1 for a in actions if a.status == "verified")
    ineffective = sum(1 for a in actions if a.status == "ineffective")
    recurrent, seen = 0, defaultdict(list)
    for c in sorted(closed, key=lambda c: c.created_at):
        key = (c.cause_code, _cause_tool(c))
        if key[1] and any((c.created_at - t) <= timedelta(days=180) for t in seen[key]):
            recurrent += 1
        seen[key].append(c.created_at)
    scores = [a.critique.get("score") for a in analyses if isinstance(a.critique.get("score"), (int, float))]
    months = []
    for i in range(11, -1, -1):
        y, m = now.year, now.month - i
        while m <= 0:
            y, m = y - 1, m + 12
        months.append(f"{y}-{m:02d}")
    trend = {k: {"opened": 0, "closed": 0} for k in months}
    for c in cases:
        if c.created_at.strftime("%Y-%m") in trend:
            trend[c.created_at.strftime("%Y-%m")]["opened"] += 1
        if c.closed_at and c.closed_at.strftime("%Y-%m") in trend:
            trend[c.closed_at.strftime("%Y-%m")]["closed"] += 1
    return {
        "cases_total": len(cases), "cases_open": sum(1 for c in cases if c.status != "closed"), "cases_closed": len(closed),
        "by_status": {CASE_STATUSES[k]: sum(1 for c in cases if c.status == k) for k in CASE_STATUSES},
        "by_stage": dict(Counter(STAGES.get(c.stage, c.stage) for c in cases).most_common()),
        "mttr_days": round(mttr, 1), "capa_total": len(actions),
        "capa_on_time_rate": round(on_time / len(finished), 3) if finished else None,
        "capa_overdue": len(overdue),
        "capa_effectiveness": round(verified / (verified + ineffective), 3) if (verified + ineffective) else None,
        "recurrence_rate": round(recurrent / len(closed), 3) if closed else 0,
        "avg_agent_score": round(sum(scores) / len(scores), 1) if scores else None,
        "avg_feedback": round(sum(f.rating for f in feedback) / len(feedback), 2) if feedback else None,
        "cost_impact_wan": round(sum(c.cost_impact for c in cases), 1),
        "downtime_hours": round(sum(c.downtime_hours for c in cases), 1),
        "trend": [{"month": k, **v} for k, v in trend.items()],
    }


def compute_pareto(db: Session) -> list[dict]:
    cases = [c for c in db.exec(select(Case).where(Case.status == "closed")).all() if c.root_cause_category]
    counts = Counter(c.root_cause_category for c in cases).most_common()
    total, cum, out = sum(n for _, n in counts) or 1, 0, []
    for k, n in counts:
        cum += n
        out.append({"key": k, "label": SIX_M.get(k, k), "count": n, "cum_pct": round(cum / total, 3)})
    return out


def cause_pareto(db: Session, top: int = 8) -> list[dict]:
    cases = [c for c in db.exec(select(Case).where(Case.status == "closed")).all() if c.cause_code]
    agg: dict[str, dict] = {}
    for c in cases:
        a = agg.setdefault(c.cause_code, {"cause_code": c.cause_code, "name": c.root_cause, "count": 0, "cost_wan": 0.0,
                                          "symptom": SYMPTOMS[c.symptom].name if c.symptom in SYMPTOMS else ""})
        a["count"] += 1
        a["cost_wan"] = round(a["cost_wan"] + c.cost_impact, 1)
    return sorted(agg.values(), key=lambda x: (-x["count"], -x["cost_wan"]))[:top]


def tool_hotspots(db: Session) -> list[dict]:
    hits: dict[tuple, list[str]] = defaultdict(list)
    for c in db.exec(select(Case).where(Case.status == "closed")).all():
        t = _cause_tool(c)
        if t:
            hits[(t, c.cause_code, c.root_cause)].append(c.code)
    return sorted(({"tool": t, "cause_code": cc, "cause": name, "count": len(v), "cases": v[-5:]}
                   for (t, cc, name), v in hits.items() if len(v) >= 2), key=lambda x: -x["count"])


def learning_snapshot(db: Session, now=None) -> dict:
    from .models import BacktestRun

    run = db.exec(select(BacktestRun).order_by(col(BacktestRun.id).desc())).first()
    edges = edges_view(db)
    shifts = sorted((e for e in edges if e["confirmed"] or e["learned"]), key=lambda e: -abs(e["delta"]))[:5]
    since = (now or utcnow()) - timedelta(days=30)
    return {
        "backtest": run.summary if run else None,
        "backtest_at": run.created_at.isoformat() if run else None,
        "top_shifts": [{k: e[k] for k in ("symptom_name", "name", "prior", "posterior", "delta", "confirmed")} for e in shifts],
        "events_30d": len(db.exec(select(LearningEvent).where(LearningEvent.created_at >= since)).all()),
        "events_total": len(db.exec(select(LearningEvent)).all()),
        "edges_learned_new": sum(1 for e in edges if e["learned"]),
        "memory_active": len(db.exec(select(MemoryItem).where(MemoryItem.status == "active")).all()),
        "memory_retired": len(db.exec(select(MemoryItem).where(MemoryItem.status == "retired")).all()),
        "memory_sandbox": len(db.exec(select(MemoryItem).where(MemoryItem.source != "real")).all()),
    }


def _findings(db: Session, kpis: dict, pareto: list[dict], learning: dict) -> list[str]:
    out = []
    bt = learning.get("backtest") or {}
    if bt.get("cases"):
        gain = (bt["hit1_learning_recent"] - bt["hit1_frozen_recent"]) * 100
        out.append(f"持续学习：近半程首位根因命中率 {bt['hit1_learning_recent']:.0%}，冻结在教科书先验则为 "
                   f"{bt['hit1_frozen_recent']:.0%}（{gain:+.0f} 个百分点，基于 {bt['cases']} 个已验证案例的预序评估）。")
    for s in learning.get("top_shifts", [])[:2]:
        out.append(f"知识更新：「{s['symptom_name']}」→「{s['name']}」教科书 {s['prior']:.0%} → 本厂 {s['posterior']:.0%}"
                   f"（{s['confirmed']} 次确认）。")
    for h in tool_hotspots(db)[:2]:
        out.append(f"机台热点：{h['tool']} 已 {h['count']} 次因「{h['cause'][:16]}」导致问题（{'、'.join(h['cases'][-3:])}），"
                   "建议改为基于状态的预测性维护并收紧 FDC 联锁。")
    if pareto:
        vital = [p for p in pareto if p["cum_pct"] <= 0.8] or pareto[:1]
        out.append("帕累托：已确认根因集中在「" + "、".join(p["label"] for p in vital) + f"」（{vital[-1]['cum_pct']:.0%}）。")
    if kpis["capa_overdue"]:
        out.append(f"有 {kpis['capa_overdue']} 条 CAPA 已逾期，需在周例会上逐条确认责任人。")
    open_ids = {c.id for c in db.exec(select(Case).where(Case.status != "closed")).all()}
    low = [a for a in db.exec(select(Analysis)).all()
           if a.case_id in open_ids and a.hypotheses and a.hypotheses[0].get("posterior", 1) < 0.5]
    if low:
        out.append(f"进行中案例有 {len(low)} 份分析首位根因后验 < 50%：证据不足时优先执行系统建议的鉴别验证，并补齐 FDC/物料批次数据。")
    if learning.get("memory_retired"):
        out.append(f"可进化记忆已自动停用 {learning['memory_retired']} 条被多次否定的条目（失败次数超过成功次数 2 次以上）。")
    return out


def build_cycle(db: Session, trigger: str, now=None, actor: str | None = None) -> ImprovementCycle:
    """一次改进周期（now 用于按时间顺序回放演示历史：只看当时已经发生的数据）。"""
    when = now or utcnow()
    run_backtest(db, trigger=f"cycle:{trigger}", when=now)
    kpis, pareto, learning = compute_kpis(db, now), compute_pareto(db), learning_snapshot(db, now)
    findings = _findings(db, kpis, pareto, learning)
    fmt = lambda v: "-" if v is None else (f"{v:.0%}" if isinstance(v, float) and v <= 1 else str(v))  # noqa: E731
    bt = learning.get("backtest") or {}
    md = [
        f"# 持续改进周期复盘（{when:%Y-%m-%d %H:%M} UTC，触发：{trigger}）", "",
        "| 指标 | 数值 |", "|---|---|",
        f"| 案例总数 / 未关闭 | {kpis['cases_total']} / {kpis['cases_open']} |",
        f"| 平均关闭周期 MTTR | {kpis['mttr_days']} 天 |",
        f"| CAPA 按期率 / 有效率 | {fmt(kpis['capa_on_time_rate'])} / {fmt(kpis['capa_effectiveness'])} |",
        f"| 逾期 CAPA | {kpis['capa_overdue']} |",
        f"| 复发率（同根因同机台 180 天内） | {fmt(kpis['recurrence_rate'])} |",
        f"| 首位根因命中率：持续学习 / 冻结先验 | {fmt(bt.get('hit1_learning'))} / {fmt(bt.get('hit1_frozen'))} |",
        f"| 近 30 天学习事件 | {learning['events_30d']} |",
        "", "## 关键发现", *[f"{i}. {f}" for i, f in enumerate(findings, 1)],
    ]
    cycle = ImprovementCycle(trigger=trigger, kpis=kpis, pareto=pareto, learning=learning, findings=findings,
                             summary_md="\n".join(md), created_at=when)
    db.add(cycle)
    audit(db, actor or ("system" if trigger == "schedule" else trigger), "improvement.run", "", when=now, findings=len(findings))
    db.commit()
    db.refresh(cycle)
    return cycle


async def run_cycle(trigger: str = "manual") -> ImprovementCycle:
    with session_scope() as db:
        return build_cycle(db, trigger)
