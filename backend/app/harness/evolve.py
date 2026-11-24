"""跨 run 的 harness 策略进化（双层递归之上的第三层：改进“怎么用快慢脑和近邻练习”的旋钮）。

参照：
- RRSI（arXiv:2609.24972，Regularized Recursive Self-Improvement of Agent Harnesses）：
  提议端——每个候选能捆绑的改动数随代数退火（3 → 2 → 1）；记录每次改动的组件、假设、实测效果，后续提议参考它；
          进展停滞时优先尝试没改过的组件。
  选择端——稳定性下限：候选不能比当前最好低于噪声带；复杂度感知：改进落在噪声带内，只有降低运行成本才接受；
          剪枝：长期没有贡献、又增加成本的组件改回默认；白名单 + 越界检查防止候选夹带别的东西。
- RSI-Opt（arXiv:2610.32026）与 Darwin Gödel Machine：策略存档跨 run 继承，所有候选（包括被拒的）都留档；
  父代采样 ∝ sigmoid(得分 − 中位数) / (1 + 子代数)。RSI-Opt 没有验收门、留出任务在第 1 代后就不再提升，
  所以这里保留验收门，并把孪生题分成互不重叠的三份：search（搜索）/ select（选择）/ report（只用于最终报告与晋升）。
- Agnes Harness：字典序目标——先保证快脑答错数不增加，再看命中率，最后看成本；上线后有观察期，
  快脑错误率显著上升就自动回滚到上一版。
"""

from __future__ import annotations

import asyncio
import math
import random
import uuid

from sqlmodel import Session, col, select

from ..fab.analytics import analyze_dataset
from ..fab.kg import rank
from ..models import AuditLog, HarnessPolicy, HarnessReceipt, utcnow
from ..rsi.memory import MemoryState
from ..rsi.twin import FabTwin, TaskSpec, observed_symptom
from . import policy
from .brain import conformal_set, feature_vector, reliability, route
from .experience import signature
from .train import active_model, paired_ci
from .ttt import test_time_training

SPLITS = {"search": (6_100_000, 4), "select": (6_200_000, 6), "report": (6_300_000, 8)}
NON_INFERIOR = -0.02  # 命中率配对差的 95% 区间下界不低于 −2 个百分点，视为“不退步”
HYPOTHESIS = {
    ("fast_min_reliability", -1): "放宽快脑门槛：更多案例直接给结论以省算力，检验命中率是否不降",
    ("fast_min_reliability", 1): "收紧快脑门槛：减少快脑答错",
    ("audit_every", -1): "更频繁地强制慢脑抽查", ("audit_every", 1): "减少抽查以省算力",
    ("ttt_tasks", 1): "加大近邻练习题量：探测更稳、采纳更准", ("ttt_tasks", -1): "减少近邻练习题量以省算力",
    ("ttt_prior_weight", 1): "近邻练习对临时先验的影响更大", ("ttt_prior_weight", -1): "近邻练习对临时先验的影响更小",
    ("ttt_keep", 1): "近邻练习时多保留一些线上经验", ("ttt_keep", -1): "近邻练习时更相信孪生的现状",
    ("adopt_min_gain", 1): "提高采纳门槛：探测提升更明显才改结论", ("adopt_min_gain", -1): "降低采纳门槛",
    ("use_experience", 1): "按经验库跳过无效练习、提前转慢脑", ("use_experience", -1): "不按经验库调整（对照）",
}


# 三套评估题各带两种“产线刚变更、记忆还没跟上”的压力症状（互不相同），防止进化只适应平常场景、一味压缩近邻练习
STRESS = {"search": ["WM_CENTER", "PKG_WARP"], "select": ["WM_SCRATCH", "HBM_OPEN"], "report": ["WM_EDGE", "TEST_VMIN"]}


def stressed_truth(symptoms: list[str]) -> dict:
    """假想变更：把这些症状的第一、第二大根因份额对调（孪生已同步，线上记忆还停在变更前）。"""
    truth = {k: dict(v) for k, v in FabTwin().truth.items()}
    for sym in symptoms:
        d = truth[sym]
        a, b = sorted(d, key=d.get, reverse=True)[:2]
        d[a], d[b] = d[b], d[a]
    return truth


async def make_split(mem: MemoryState, name: str, twin: FabTwin | None = None, symptoms: list[str] | None = None,
                     per_symptom: int | None = None, stress: bool = False) -> list[dict]:
    seed0, per = SPLITS[name]
    per = per_symptom or per
    twin = twin or FabTwin()
    jobs = [(sym, si, None, twin) for si, sym in enumerate(symptoms or list(twin.truth))]
    if stress:
        st = FabTwin(truth=stressed_truth(STRESS[name]))
        jobs += [(sym, 50 + si, name, st) for si, sym in enumerate(STRESS[name])]
    out = []
    for sym, si, key, tw in jobs:
        for j in range(per * (2 if key else 1)):
            seed = seed0 + si * 1000 + j
            t = tw.make_task(TaskSpec(sym, ("medium", "hard", "easy", "hard")[j % 4], decoy_pm=j % 3 == 1), seed)
            ev = analyze_dataset(t.observation)
            obs = observed_symptom(sym, ev)
            hyps = [h.as_dict() for h in rank(ev, obs, mem.edges, mem.hotspot_weights(), mem.weights)]
            out.append({"seed": seed, "evidence": ev, "symptom": obs, "truth": t.hidden["cause"], "hyps": hyps, "twin": key})
    return out


def twins_for(name: str) -> dict:
    return {name: FabTwin(truth=stressed_truth(STRESS[name]))}


async def _ttt(cache: dict, task: dict, mem: MemoryState, params: dict, twin: FabTwin | None, twins: dict | None = None) -> dict:
    key = (task["seed"], params["ttt_tasks"], params["ttt_prior_weight"], params["ttt_keep"])
    if key not in cache:
        tw = (twins or {}).get(task.get("twin")) or twin
        r = await test_time_training(task["evidence"], task["symptom"], mem, params, seed=task["seed"] % 1_000_000 + 40_000_000, twin=tw)
        cache[key] = {k: r[k] for k in ("probe_n", "probe_before", "probe_after", "top_after", "tasks")}
    return cache[key]


async def evaluate(params: dict, tasks: list[dict], mem: MemoryState, cache: dict, *, twin: FabTwin | None = None,
                   twins: dict | None = None, model=None, slow_brain: bool = True, all_fast: bool = False) -> dict:
    q = (model.conformal or {}).get("q") if model is not None else None
    decisions, streak = [], {}
    for t in tasks:
        sig = signature(t["evidence"], t["hyps"])
        fv = feature_vector(t["hyps"], sig, {"facts": []})
        rel = reliability(fv, model)
        cset = conformal_set(t["hyps"], q)
        reasons = [] if all_fast else route(fv, rel, cset, params, data_ok=True, severity="S2", verified=1,
                                            streak=streak.get(t["symptom"], 0), exp={})
        streak[t["symptom"]] = 0 if reasons else streak.get(t["symptom"], 0) + 1
        decisions.append((t, reasons, cset))
    slow = [t for t, reasons, _ in decisions if reasons and slow_brain]
    for i in range(0, len(slow), 8):  # 多个案例的近邻练习并行跑（练习题本身也在进程池里并行）
        await asyncio.gather(*[_ttt(cache, t, mem, params, twin, twins) for t in slow[i:i + 8]])
    correct, fast_n, fast_wrong, cost, adopted, covered, sizes = [], 0, 0, 0, 0, 0, 0
    for t, reasons, cset in decisions:
        top = t["hyps"][0]["cause_code"] if t["hyps"] else ""
        if not reasons:
            fast_n += 1
            fast_wrong += top != t["truth"]
        elif slow_brain:
            r = cache[(t["seed"], params["ttt_tasks"], params["ttt_prior_weight"], params["ttt_keep"])]
            cost += r["tasks"]
            gain = (r["probe_after"] - r["probe_before"]) / max(1, r["probe_n"])
            if r["probe_after"] > r["probe_before"] and gain >= params["adopt_min_gain"]:
                top, adopted = r["top_after"], adopted + 1
        correct.append(1.0 if top == t["truth"] else 0.0)
        covered += t["truth"] in cset
        sizes += len(cset)
    n = max(1, len(tasks))
    stress_idx = [i for i, (t, _, _) in enumerate(decisions) if t.get("twin")]
    return {"n": len(tasks), "hit1": round(100 * sum(correct) / n, 2), "fast_rate": round(fast_n / n, 3),
            "hit1_stress": round(100 * sum(correct[i] for i in stress_idx) / len(stress_idx), 2) if stress_idx else None,
            "cost_per_task": round(cost / n, 2),
            "fast_wrong": fast_wrong, "fast_precision": round((fast_n - fast_wrong) / fast_n, 3) if fast_n else None,
            "ttt_tasks": cost, "adopted": adopted, "coverage": round(covered / n, 3), "set_size": round(sizes / n, 2),
            "_correct": correct}


def _public(ev: dict) -> dict:
    return {k: v for k, v in ev.items() if not k.startswith("_")}


def _propose(parent: dict, max_edits: int, rng: random.Random, gains: dict, stalled: bool, tried: set) -> tuple[dict, list[dict]]:
    names = list(policy.COMPONENTS)
    weights = [(2.0 if gains.get(n, 0) > 0 else 1.0) * (1.5 if stalled and n not in tried else 1.0) for n in names]
    picked = []
    while len(picked) < max_edits:
        n = rng.choices(names, weights=weights)[0]
        if n not in picked:
            picked.append(n)
    cand, edits = dict(parent), []
    for n in picked:
        _, step, lo, hi, _ = policy.COMPONENTS[n]
        d = rng.choice([-1, 1])
        v = cand[n] + d * step
        if not lo <= v <= hi:
            d, v = -d, cand[n] - d * step
        cand[n] = v
        edits.append({"component": n, "from": parent[n], "to": round(v, 3) if isinstance(v, float) else v,
                      "hypothesis": HYPOTHESIS.get((n, d), "")})
    return policy.clamp(cand), edits


def _sample_parent(archive: list[dict], rng: random.Random) -> dict:
    med = sorted(a["score"] for a in archive)[len(archive) // 2]
    w = [1 / (1 + math.exp(-(a["score"] - med))) / (1 + a["children"]) for a in archive]
    return rng.choices(archive, weights=w)[0]


async def evolve_run(db: Session, mem: MemoryState, *, generations: int = 3, children: int = 2, when=None, seed: int = 11,
                     twin: FabTwin | None = None) -> dict:
    when = when or utcnow()
    run_id = f"rrsi-{when:%Y%m%d%H%M}-{uuid.uuid4().hex[:4]}"
    rng = random.Random(seed)
    model = active_model(db)
    inc_id, inc = policy.active(db)
    search, select_ = await make_split(mem, "search", twin, stress=True), await make_split(mem, "select", twin, stress=True)
    tw = {**twins_for("search"), **twins_for("select")}
    cache: dict = {}
    ev_inc = {"search": await evaluate(inc, search, mem, cache, twin=twin, twins=tw, model=model),
              "select": await evaluate(inc, select_, mem, cache, twin=twin, twins=tw, model=model)}
    past = db.exec(select(HarnessPolicy).where(col(HarnessPolicy.status).in_(["accepted", "active", "retired"]))).all()
    gains: dict = {}
    for p in db.exec(select(HarnessPolicy)).all():  # 读历次改动的实测效果（RRSI：后续提议以此为条件）
        for e in p.edits or []:
            gains[e["component"]] = gains.get(e["component"], 0) + (e.get("delta_hit1") or 0)
    archive = [{"params": inc, "score": ev_inc["search"]["hit1"], "children": 0, "id": inc_id}]
    archive += [{"params": policy.clamp(p.params), "score": (p.scores.get("search") or {}).get("hit1", 0), "children": 0, "id": p.id}
                for p in past if p.id != inc_id][-6:]
    best = {"params": inc, "ev": ev_inc, "id": inc_id}
    log, tried, stalled = [], set(), False
    for gen in range(generations):
        max_edits = max(1, 3 - gen)  # 改动预算退火
        accepted_gen = False
        for _ in range(children):
            parent = _sample_parent(archive, rng)
            parent["children"] += 1
            cand, edits = _propose(parent["params"], max_edits, rng, gains, stalled, tried)
            tried.update(e["component"] for e in edits)
            if cand == parent["params"] or any(k not in policy.COMPONENTS for k in cand):
                continue
            ev_s = await evaluate(cand, search, mem, cache, twin=twin, twins=tw, model=model)
            status, note, ev_sel = "rejected", "", None
            if ev_s["fast_wrong"] > best["ev"]["search"]["fast_wrong"] + 1:
                note = "搜索集：快脑答错数增加"
            else:
                ev_sel = await evaluate(cand, select_, mem, cache, twin=twin, twins=tw, model=model)
                mean, lo, hi = paired_ci(ev_sel["_correct"], best["ev"]["select"]["_correct"])
                cheaper = ev_sel["cost_per_task"] < 0.9 * best["ev"]["select"]["cost_per_task"]  # 实测的孪生练习题数
                safe = ev_sel["fast_wrong"] <= best["ev"]["select"]["fast_wrong"]
                if not safe:
                    note = "选择集：快脑答错数增加（安全优先，直接拒绝）"
                elif lo > 0:
                    status, note = "accepted", f"选择集命中率显著提高（配对差 95% 区间 [{lo * 100:+.1f}, {hi * 100:+.1f}] 个百分点）"
                elif lo >= NON_INFERIOR and cheaper:
                    status, note = "accepted", "改进在噪声带内，但孪生练习题数减少 10% 以上（复杂度感知接受）"
                else:
                    band = f"[{lo * 100:+.1f}, {hi * 100:+.1f}] 个百分点"
                    if hi < 0:
                        note = f"命中率显著退步（{band}）" + ("，虽然更省" if cheaper else "")
                    elif lo < NON_INFERIOR:
                        note = f"可能退步（{band}，下界低于 −2 个百分点）" + ("，虽然更省" if cheaper else "")
                    else:
                        note = f"改进在噪声带内（{band}）且没有降低成本"
                for e in edits:
                    e["delta_hit1"] = round(ev_sel["hit1"] - best["ev"]["select"]["hit1"], 2)
            row = HarnessPolicy(run_id=run_id, parent_id=parent["id"], params=cand, edits=edits, status=status, note=note,
                                scores={"search": _public(ev_s), "select": _public(ev_sel) if ev_sel else None,
                                        "generation": gen, "max_edits": max_edits}, created_at=when)
            db.add(row)
            db.flush()
            log.append({"id": row.id, "status": status, "edits": edits, "note": note})
            if status == "accepted":
                accepted_gen = True
                best = {"params": cand, "ev": {"search": ev_s, "select": ev_sel}, "id": row.id}
                archive.append({"params": cand, "score": ev_s["hit1"], "children": 0, "id": row.id})
        stalled = not accepted_gen
    # 剪枝：偏离默认值、又没有带来实测收益的组件改回默认（RRSI 的 Lasso 式剪枝）
    prune = {k: policy.DEFAULT[k] for k in policy.COMPONENTS if best["params"][k] != policy.DEFAULT[k] and gains.get(k, 0) <= 0}
    if prune and best["id"] != inc_id:
        cand = policy.clamp({**best["params"], **prune})
        if cand != best["params"]:
            ev_sel = await evaluate(cand, select_, mem, cache, twin=twin, twins=tw, model=model)
            mean, lo, hi = paired_ci(ev_sel["_correct"], best["ev"]["select"]["_correct"])
            ok = lo >= NON_INFERIOR and ev_sel["fast_wrong"] <= best["ev"]["select"]["fast_wrong"]
            edits = [{"component": k, "from": best["params"][k], "to": v, "hypothesis": "剪枝：没有实测收益，改回默认"} for k, v in prune.items()]
            row = HarnessPolicy(run_id=run_id, parent_id=best["id"], params=cand, edits=edits, status="accepted" if ok else "rejected",
                                note="剪枝" + ("通过" if ok else "会让命中率退步，保留原值"), scores={"select": _public(ev_sel)}, created_at=when)
            db.add(row)
            db.flush()
            if ok:
                best = {"params": cand, "ev": {**best["ev"], "select": ev_sel}, "id": row.id}
    promoted = False
    report = None
    if best["id"] != inc_id:  # 只在最后用一次封存的 report 集决定是否上线
        rep_tasks = await make_split(mem, "report", twin, stress=True)
        rt = twins_for("report")
        r_inc = await evaluate(inc, rep_tasks, mem, cache, twin=twin, twins=rt, model=model)
        r_new = await evaluate(best["params"], rep_tasks, mem, cache, twin=twin, twins=rt, model=model)
        mean, lo, hi = paired_ci(r_new["_correct"], r_inc["_correct"])
        promoted = lo >= NON_INFERIOR and r_new["fast_wrong"] <= r_inc["fast_wrong"]
        report = {"incumbent": _public(r_inc), "candidate": _public(r_new), "diff": {"mean": mean, "ci95": [lo, hi]}}
        row = db.get(HarnessPolicy, best["id"])
        row.scores = {**row.scores, "report": report}
        if promoted:
            for old in db.exec(select(HarnessPolicy).where(HarnessPolicy.status == "active")).all():
                old.status = "retired"
                db.add(old)
            row.status, row.note = "active", (row.note + "；report 集不退步，晋升上线").lstrip("；")
        db.add(row)
    db.add(AuditLog(actor="system:rrsi", action="harness.evolve", target=run_id,
                    detail={"candidates": len(log), "accepted": sum(1 for x in log if x["status"] == "accepted"), "promoted": promoted},
                    created_at=when))
    db.commit()
    return {"run_id": run_id, "incumbent": {"id": inc_id, "params": inc, "search": _public(ev_inc["search"]), "select": _public(ev_inc["select"])},
            "candidates": log, "best": {"id": best["id"], "params": best["params"]}, "report": report, "promoted": promoted}


def probation_check(db: Session, when=None) -> dict:
    """上线观察期（Agnes：新版本先观察，错误率显著上升自动回滚）：该版本下已确认的快脑结论里，答错 ≥ 3 次且 ≥ 20% 即回滚。"""
    act = db.exec(select(HarnessPolicy).where(HarnessPolicy.status == "active").order_by(col(HarnessPolicy.id).desc())).first()
    if act is None:
        return {"status": "default_policy"}
    rows = db.exec(select(HarnessReceipt).where(HarnessReceipt.policy_version == act.id, HarnessReceipt.route == "fast",
                                                col(HarnessReceipt.verified_at).is_not(None))).all()
    wrong = sum(1 for r in rows if not (r.outcome or {}).get("fast_match"))
    if len(rows) >= 10 and wrong >= 3 and wrong / len(rows) >= 0.2:
        parent = db.get(HarnessPolicy, act.parent_id) if act.parent_id else None
        act.status = "rolled_back"
        db.add(act)
        back = HarnessPolicy(run_id=act.run_id, parent_id=act.id, params=parent.params if parent else dict(policy.DEFAULT),
                             status="active", note=f"观察期回滚：快脑确认 {len(rows)} 次中答错 {wrong} 次", created_at=when or utcnow())
        db.add(back)
        db.add(AuditLog(actor="system:rrsi", action="harness.rollback", target=act.run_id, detail={"fast": len(rows), "wrong": wrong},
                        created_at=when or utcnow()))
        db.commit()
        return {"status": "rolled_back", "fast": len(rows), "wrong": wrong}
    return {"status": "ok", "fast": len(rows), "wrong": wrong}


async def harness_benchmark(mem: MemoryState, model=None, params: dict | None = None) -> dict:
    """孪生对照：同一批 report 题，只用快脑 vs 快慢双脑；以及“产线刚变更、记忆还没跟上”的压力场景。"""
    params = params or dict(policy.DEFAULT)
    cache: dict = {}
    tasks = await make_split(mem, "report", None)
    fast = await evaluate(params, tasks, mem, cache, model=model, all_fast=True)
    dual = await evaluate(params, tasks, mem, cache, model=model)
    no_ttt = await evaluate(params, tasks, mem, cache, model=model, slow_brain=False)
    stress = STRESS["report"]
    twin = FabTwin(truth=stressed_truth(stress))
    s_tasks = await make_split(mem, "report", twin, symptoms=stress, per_symptom=20)
    s_cache: dict = {}
    s_fast = await evaluate(params, s_tasks, mem, s_cache, twin=twin, model=model, all_fast=True)
    s_dual = await evaluate(params, s_tasks, mem, s_cache, twin=twin, model=model)
    d_mean, d_lo, d_hi = paired_ci(dual["_correct"], fast["_correct"])
    s_mean, s_lo, s_hi = paired_ci(s_dual["_correct"], s_fast["_correct"])
    return {"params": params, "fast_only": _public(fast), "dual": _public(dual), "dual_no_ttt": _public(no_ttt),
            "dual_vs_fast": {"mean": d_mean, "ci95": [d_lo, d_hi]},
            "stress": {"symptoms": stress, "change": "第一、第二大根因份额对调（孪生已同步，线上记忆还停在变更前）",
                       "fast_only": _public(s_fast), "dual": _public(s_dual),
                       "diff": {"mean": s_mean, "ci95": [s_lo, s_hi]}}}
