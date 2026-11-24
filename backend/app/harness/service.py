"""Harness 层的调度与查询：夜间任务、手动触发、控制台总览。

更新时机一览（也写在 docs/02 §2.8）：
  每次分析     写收据；慢脑做完内层递归立刻按孪生探测结果更新“近邻练习”经验
  每次关案     收据写入现场结果；经验库按成败更新（答错追责）；推翻关案时撤回
  夜间         上线观察期检查（必要时回滚）→ 满足触发条件才开可靠度模型训练窗口 → 距上次超过 7 天才跑一次 RRSI 策略进化
"""

from __future__ import annotations

from datetime import timedelta

from sqlmodel import Session, col, func, select

from ..models import HarnessBenchmark, HarnessExperience, HarnessPolicy, HarnessReceipt, ReliabilityModel, utcnow
from ..rsi.memory import load_memory
from . import policy
from .evolve import evolve_run, harness_benchmark, probation_check
from .experience import p_success, trials
from .train import active_model, pending_triggers, train_window

EVOLVE_EVERY = timedelta(days=7)


async def nightly(db: Session, when=None, force_evolve: bool = False) -> dict:
    when = when or utcnow()
    out = {"probation": probation_check(db, when)}
    trig = pending_triggers(db)
    if trig:
        row = await train_window(db, load_memory(db), ",".join(trig), when)
        out["train"] = {"id": row.id, "status": row.status, "trigger": ",".join(trig)}
    last = db.exec(select(HarnessPolicy).order_by(col(HarnessPolicy.id).desc())).first()
    if force_evolve or last is None or when - last.created_at.replace(tzinfo=when.tzinfo) >= EVOLVE_EVERY:
        res = await evolve_run(db, load_memory(db), when=when)
        out["evolve"] = {"run_id": res["run_id"], "promoted": res["promoted"], "candidates": len(res["candidates"])}
    return out


async def benchmark(db: Session, trigger: str = "manual", when=None) -> HarnessBenchmark:
    _, params = policy.active(db)
    res = await harness_benchmark(load_memory(db), active_model(db), params)
    row = HarnessBenchmark(trigger=trigger, results=res, created_at=when or utcnow())
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _wilson(k: int, n: int) -> list[float] | None:
    if not n:
        return None
    import math

    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0, c - r), 3), round(min(1, c + r), 3)]


def overview(db: Session) -> dict:
    pid, params = policy.active(db)
    rows = db.exec(select(HarnessReceipt)).all()
    verified = [r for r in rows if r.verified_at]
    fast_v = [r for r in verified if r.route == "fast"]
    fast_ok = sum(1 for r in fast_v if (r.outcome or {}).get("fast_match"))
    ttt = [r for r in rows if (r.inner or {}).get("ttt")]
    adopted = [r for r in ttt if r.inner["ttt"].get("adopt")]
    in_set = [r for r in verified if "in_set" in (r.outcome or {})]
    by_month: dict[str, dict] = {}
    for r in rows:
        m = by_month.setdefault(r.created_at.strftime("%Y-%m"), {"month": r.created_at.strftime("%Y-%m"), "fast": 0, "slow": 0, "ttt": 0})
        m[r.route] += 1
        m["ttt"] += bool((r.inner or {}).get("ttt"))
    exps = db.exec(select(HarnessExperience).order_by(col(HarnessExperience.updated_at).desc())).all()
    archive = db.exec(select(HarnessPolicy).order_by(col(HarnessPolicy.id).desc()).limit(40)).all()
    models = db.exec(select(ReliabilityModel).order_by(col(ReliabilityModel.id).desc()).limit(12)).all()
    bench = db.exec(select(HarnessBenchmark).order_by(col(HarnessBenchmark.id).desc())).first()
    reasons: dict[str, int] = {}
    for r in rows:
        for x in r.reasons or []:
            reasons[x] = reasons.get(x, 0) + 1
    return {
        "policy": {"id": pid, "params": params, "components": {k: {"default": v[0], "min": v[2], "max": v[3], "desc": v[4]}
                                                                for k, v in policy.COMPONENTS.items()}},
        "receipts": {"total": len(rows), "fast": sum(1 for r in rows if r.route == "fast"), "slow": sum(1 for r in rows if r.route == "slow"),
                     "verified": len(verified), "fast_verified": len(fast_v), "fast_correct": fast_ok,
                     "fast_precision_ci95": _wilson(fast_ok, len(fast_v)),
                     "final_correct": sum(1 for r in verified if (r.outcome or {}).get("top_match")),
                     "ttt": len(ttt), "ttt_adopted": len(adopted),
                     "coverage": round(sum(1 for r in in_set if r.outcome["in_set"]) / len(in_set), 3) if in_set else None,
                     "reasons": reasons, "by_month": sorted(by_month.values(), key=lambda m: m["month"])},
        "experiences": {"counts": {f"{t}:{s}": sum(1 for x in exps if x.track == t and x.status == s)
                                   for t in ("fact", "tip") for s in ("candidate", "active", "stable", "retired")},
                        "items": [{"id": x.id, "track": x.track, "status": x.status, "text": x.text, "p": round(p_success(x), 3),
                                   "n": int(trials(x)), "retired_reason": x.retired_reason, "blame": x.blame[-2:],
                                   "updated_at": x.updated_at} for x in exps[:40]]},
        "archive": [p.model_dump() for p in archive],
        "models": [m.model_dump() for m in models],
        "triggers": pending_triggers(db),
        "benchmark": bench.model_dump() if bench else None,
    }
