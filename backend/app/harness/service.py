"""调度与查询：夜间检查、手动触发、控制台总览。

更新时机一览（也写在 docs/02 §2.8），全部事件触发、不定时：
  每次分析     写收据；慢脑做完立刻按探测结果更新改进经验库
  每次关案     收据写入现场结果；改进经验库按成败更新（答错追责）；推翻关案时撤回
  夜间检查     ① 上线观察期（必要时回滚 harness 配置）
               ② RSI-Opt 外层：有触发才进化训练策略（还没有策略 / 连续 2 个训练窗口失败）
               ③ 快脑训练窗口：有触发才开窗（A-RSI：新收据、产线变更、样本池饱和、慢脑频繁推翻快脑、产线命中率下滑）
               ④ RRSI：有触发才进化 harness 配置（改进库有新规律、快脑错误上升、新模型上线、产线变更、慢脑饱和）
"""

from __future__ import annotations

from sqlmodel import Session, col, select

from ..fastbrain import optimize, playbook, windows
from ..models import FastBrainModel, HarnessBenchmark, HarnessExperience, HarnessPolicy, HarnessReceipt, TrainingPolicy, utcnow
from ..rsi.memory import load_memory
from ..stats import wilson
from . import policy
from .evolve import evolve_run, harness_benchmark, pending_triggers as rrsi_triggers, probation_check
from .experience import library, p_success, trials


async def nightly(db: Session, when=None, *, force: tuple[str, ...] = (), twin=None) -> dict:
    when = when or utcnow()
    out: dict = {"probation": probation_check(db, when)}
    trig = optimize.pending_triggers(db) or (["forced"] if "rsi_opt" in force else [])
    if trig:
        res = await optimize.evolve_training(db, load_memory(db), when=when, twin=twin, trigger=",".join(trig))
        out["rsi_opt"] = {"run_id": res["run_id"], "promoted": res["promoted"], "candidates": len(res["candidates"]), "trigger": res["trigger"]}
    trig = windows.pending_triggers(db) or (["forced"] if "window" in force else [])
    if trig:
        row = await windows.train_window(db, load_memory(db), ",".join(trig), when, twin=twin)
        out["window"] = {"id": row.id, "status": row.status, "trigger": ",".join(trig)}
    trig = rrsi_triggers(db) or (["forced"] if "rrsi" in force else [])
    if trig:
        res = await evolve_run(db, load_memory(db), when=when, twin=twin, trigger=",".join(trig))
        out["rrsi"] = {"run_id": res["run_id"], "promoted": res["promoted"], "candidates": len(res["candidates"]), "trigger": res["trigger"]}
    return out


async def benchmark(db: Session, trigger: str = "manual", when=None) -> HarnessBenchmark:
    _, params = policy.active(db)
    res = await harness_benchmark(db, load_memory(db), params)
    row = HarnessBenchmark(trigger=trigger, results=res, created_at=when or utcnow())
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _model_view(m: FastBrainModel) -> dict:
    d = m.model_dump()
    d["params"] = {k: v for k, v in (m.params or {}).items() if k != "theta"}
    return d


def fastbrain_overview(db: Session) -> dict:
    models = db.exec(select(FastBrainModel).order_by(col(FastBrainModel.id).desc()).limit(16)).all()
    pols = db.exec(select(TrainingPolicy).order_by(col(TrainingPolicy.id).desc()).limit(30)).all()
    pid, pol = optimize.active_policy(db)
    act = windows.active_row(db)
    return {"active": _model_view(act) if act else None, "windows": [_model_view(m) for m in models],
            "policy": {"id": pid, "params": pol, "space": {k: {"options": v[0], "default": v[1], "desc": v[2]} for k, v in optimize.SPACE.items()}},
            "policies": [p.model_dump() for p in pols], "notes": playbook.overview(db),
            "triggers": {"window": windows.pending_triggers(db), "rsi_opt": optimize.pending_triggers(db)},
            "overrules": windows.overrules(db)}


def overview(db: Session) -> dict:
    pid, params = policy.active(db)
    rows = db.exec(select(HarnessReceipt)).all()
    verified = [r for r in rows if r.verified_at]
    fast_v = [r for r in verified if r.route == "fast"]
    fast_ok = sum(1 for r in fast_v if (r.outcome or {}).get("fast_match"))
    slow = [r for r in rows if r.route == "slow" and (r.inner or {}).get("hits")]
    chosen: dict[str, int] = {}
    for r in slow:
        c = ((r.inner or {}).get("arbiter") or {}).get("chosen", "fast")
        chosen[c] = chosen.get(c, 0) + 1
    in_set = [r for r in verified if "in_set" in (r.outcome or {})]
    by_month: dict[str, dict] = {}
    for r in rows:
        m = by_month.setdefault(r.created_at.strftime("%Y-%m"), {"month": r.created_at.strftime("%Y-%m"), "fast": 0, "slow": 0, "overrule": 0})
        m[r.route] += 1
        m["overrule"] += r.route == "slow" and r.top != r.top_fast
    exps = db.exec(select(HarnessExperience).order_by(col(HarnessExperience.updated_at).desc())).all()
    archive = db.exec(select(HarnessPolicy).order_by(col(HarnessPolicy.id).desc()).limit(40)).all()
    bench = db.exec(select(HarnessBenchmark).order_by(col(HarnessBenchmark.id).desc())).first()
    reasons: dict[str, int] = {}
    for r in rows:
        for x in r.reasons or []:
            reasons[x] = reasons.get(x, 0) + 1
    slow_v = [r for r in verified if r.route == "slow" and r.top != r.top_fast]
    return {
        "policy": {"id": pid, "params": params, "components": {k: {"default": v[0], "min": v[2], "max": v[3], "desc": v[4]}
                                                                for k, v in policy.COMPONENTS.items()}},
        "receipts": {"total": len(rows), "fast": sum(1 for r in rows if r.route == "fast"), "slow": sum(1 for r in rows if r.route == "slow"),
                     "verified": len(verified), "fast_verified": len(fast_v), "fast_correct": fast_ok,
                     "fast_precision_ci95": wilson(fast_ok, len(fast_v)),
                     "final_correct": sum(1 for r in verified if (r.outcome or {}).get("top_match")),
                     "with_model": sum(1 for r in rows if r.fast_model), "chosen": chosen,
                     "overrule_verified": len(slow_v), "overrule_correct": sum(1 for r in slow_v if (r.outcome or {}).get("top_match")),
                     "overrule_fast_correct": sum(1 for r in slow_v if (r.outcome or {}).get("fast_match")),
                     "coverage": round(sum(1 for r in in_set if r.outcome["in_set"]) / len(in_set), 3) if in_set else None,
                     "reasons": reasons, "by_month": sorted(by_month.values(), key=lambda m: m["month"])},
        "experiences": {"counts": {f"tip:{s}": sum(1 for x in exps if x.status == s) for s in ("active", "stable", "retired")},
                        "library": library(db)[:24],
                        "items": [{"id": x.id, "track": x.track, "status": x.status, "text": x.text, "p": round(p_success(x), 3),
                                   "n": int(trials(x)), "retired_reason": x.retired_reason, "blame": x.blame[-2:],
                                   "updated_at": x.updated_at} for x in exps[:40]]},
        "archive": [p.model_dump() for p in archive],
        "triggers": rrsi_triggers(db),
        "benchmark": bench.model_dump() if bench else None,
    }
