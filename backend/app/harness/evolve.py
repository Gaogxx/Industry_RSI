"""RRSI（arXiv:2609.24972，Regularized Recursive Self-Improvement of Agent Harnesses）用在 harness 上：
进化“基础 harness”——组件的开关与旋钮 + 条件规则。

三层关系：
  内层（Harness² 的任务级）  每个案例从基础 harness 出发，慢脑 提议 → 探测 → 合成（slow.py）
  外层（Harness² 的经验）    反思写入改进经验库（experience.py）
  RRSI（跨 run）            从改进库汇总的规律提议修改基础 harness，用正则化的验收决定是否采纳

正则化（RRSI 的 R）：
  - 改动预算退火：每个候选能捆绑的改动数随代数 3 → 2 → 1；
  - 稳定性下限 / 噪声带：select 集命中率配对差的 95% 区间下界 ≥ −2 个百分点才算不退步；
  - 复杂度感知：改进落在噪声带内时，只有实测成本（孪生仿真次数）降低 ≥ 10%、或配置更简单且不更贵才接受；
  - 剪枝：没有贡献的规则删掉、没有实测收益的旋钮改回默认（Lasso 式）；
  - 防泄漏审查：规则只能以可观察的条件字段为条件、来源必须是改进库（产线收据或线上探测），
    并且在 search 与 select 集上都要有足够的支持题，否则无法验证；
  - 字典序目标（Agnes）：先保证快脑答错数不增加，再看命中率，最后看成本。

何时跑（参照 A-RSI 的“何时更新”，事件触发，不定时）：
  bootstrap        还没有任何配置存档
  library:N        上次之后改进库里有 ≥ 3 条经验变为稳定或停用（有新的规则可提）
  fast_errors      上次之后产线确认的快脑结论里答错 ≥ 3 次且 ≥ 15%
  new_model        上次之后有新的快脑模型晋升（路由门槛要按新模型重新调）
  drift            孪生同步了产线变更
  slow_saturated   慢脑“赢过快脑”的比例跌到两次之前的 85% 以下（慢脑组件可能可以省掉）
  退避：最近连续 2 次都没有接受任何候选时，只响应 drift / new_model / fast_errors 这类强信号（A-RSI 的 halt）。
最后是否上线（RRSI 自己的门，与快脑训练窗口的健康门 / 晋升门是两套东西）：
  ① 白名单 + 越界检查：配置合法、规则过防泄漏审查；
  ② search 集初筛：快脑答错数不比当前最好多 1 个以上（便宜的快速淘汰）；
  ③ select 集验收：命中率配对差的区间按本次已评估的候选数做 Bonferroni 校正（候选越多区间越宽，防止“从一堆候选里
     挑出靠运气最好的那个”）；下界 > 0 才算显著提高；落在噪声带内只有更便宜或更简单才接受；
  ④ report 集上线门：每次进化都让孪生生成一批全新的 report 题（永不复用，避免封存集被反复使用后也被“调”进去）；
     要求快脑答错不增加、命中率配对差下界 ≥ −2 个百分点且平均差 ≥ 0（独立确认方向）、成本变高时必须显著更好；
  ⑤ 产线回放：用最近已确认的产线案例重放候选的快慢路由，快脑答错数不得多于在用配置（孪生之外的真实标签）；
  ⑥ 上线观察期：新配置下已确认的快脑结论，错误率 Wilson 区间下界超过上一版的错误率（且超过路由门槛允许的
     错误率）就自动回滚——和上一版比的统计检验，而不是拍一个固定阈值。
"""

from __future__ import annotations

import asyncio
import math
import random
import uuid

from sqlmodel import Session, col, func, select

from ..fab.analytics import analyze_dataset
from ..fab.flow import SYMPTOMS
from ..fab.kg import rank
from ..fastbrain.infer import model_rank
from ..fastbrain.model import FastModel
from ..fastbrain.windows import load_active
from ..models import AuditLog, FastBrainModel, HarnessExperience, HarnessPolicy, HarnessReceipt, utcnow
from ..rsi.memory import MemoryState, condition_of
from ..rsi.twin import FabTwin, TaskSpec, observed_symptom
from ..stats import paired_ci
from . import policy
from .brain import conformal_set, route
from .experience import library
from .slow import arbitrate, components

SPLITS = {"search": (6_100_000, 3), "select": (6_200_000, 5), "report": (6_300_000, 6)}
REPORT_BASE = 20_000_000  # 上线门的 report 题每次换新种子：20M + 运行序号 × 60k（不与其它题集重叠）
NON_INFERIOR = -0.02
STRESS = {"search": ["WM_CENTER", "PKG_WARP"], "select": ["WM_SCRATCH", "HBM_OPEN"], "report": ["WM_EDGE", "TEST_VMIN"]}
LIB_CHANGES, FAST_ERR_N, FAST_ERR_RATE, SAT_DROP = 3, 3, 0.15, 0.85
STRONG = ("bootstrap", "drift", "new_model", "fast_errors")
HYPOTHESIS = {
    ("router.fast_min_conf", -1): "放宽快脑门槛：更多案例直接给结论以省算力，检验命中率是否不降",
    ("router.fast_min_conf", 1): "收紧快脑门槛：减少快脑答错",
    ("router.audit_every", -1): "更频繁地强制慢脑抽查", ("router.audit_every", 1): "减少抽查以省算力",
    ("slow.ttt", 1): "开启近邻训练", ("slow.ttt", -1): "关闭近邻训练以省算力",
    ("slow.ttt_tasks", 1): "加大近邻题量：探测更稳", ("slow.ttt_tasks", -1): "减少近邻题量以省算力",
    ("slow.ttt_keep", 1): "近邻练习时多保留线上经验", ("slow.ttt_keep", -1): "近邻练习时更相信孪生现状",
    ("arbiter.min_gain", 1): "提高改判门槛：探测优势更明显才改判", ("arbiter.min_gain", -1): "降低改判门槛",
    ("experience.use", 1): "按改进库调整", ("experience.use", -1): "不按改进库调整（对照）",
    ("investigator.mode", 1): "大模型调查员只在 S1 或探测打平时调用", ("investigator.mode", -1): "大模型调查员调用更频繁",
}
EVOLVED = [k for k in policy.COMPONENTS if k != "investigator.mode"]  # 调查员要真实大模型才能评估，孪生里不进化它


def stressed_truth(symptoms: list[str]) -> dict:
    """假想变更：把这些症状的第一、第二大根因份额对调（孪生已同步，快脑模型和记忆都还停在变更前）。"""
    truth = {k: dict(v) for k, v in FabTwin().truth.items()}
    for sym in symptoms:
        d = truth[sym]
        a, b = sorted(d, key=d.get, reverse=True)[:2]
        d[a], d[b] = d[b], d[a]
    return truth


def twins_for(*names: str) -> dict:
    return {n: FabTwin(truth=stressed_truth(STRESS[n])) for n in names}


async def make_split(mem: MemoryState, name: str, twin: FabTwin | None = None, symptoms: list[str] | None = None,
                     per_symptom: int | None = None, stress: bool = False, twins: dict | None = None, seed0: int | None = None) -> list[dict]:
    base, per = SPLITS[name]
    seed0 = base if seed0 is None else seed0
    per = per_symptom or per
    twin = twin or FabTwin()
    jobs = [(sym, si, None, twin) for si, sym in enumerate(symptoms or list(twin.truth))]
    if stress:
        st = (twins or {}).get(name) or FabTwin(truth=stressed_truth(STRESS[name]))
        jobs += [(sym, 50 + si, name, st) for si, sym in enumerate(STRESS[name])]
    out = []
    for sym, si, key, tw in jobs:
        for j in range(per * (2 if key else 1)):
            seed = seed0 + si * 1000 + j
            t = tw.make_task(TaskSpec(sym, ("medium", "hard", "easy", "hard")[j % 4], decoy_pm=j % 3 == 1), seed)
            ev = analyze_dataset(t.observation)
            obs = observed_symptom(sym, ev)
            hyps = [h.as_dict() for h in rank(ev, obs, mem.edges, mem.hotspot_weights(), mem.weights)]
            out.append({"seed": seed, "evidence": ev, "symptom": obs, "truth": t.hidden["cause"], "hyps": hyps, "twin": key,
                        "cond": condition_of(ev)})
    return out


def _fast(t: dict, model: FastModel, mem: MemoryState) -> list[dict]:
    key = ("_fast", id(model))
    if key not in t:
        t[key] = model_rank(model, t["evidence"], t["symptom"], mem, t["hyps"])
    return t[key]


async def evaluate(params: dict, tasks: list[dict], mem: MemoryState, model: FastModel, q: float | None, cache: dict, *,
                   twin: FabTwin | None = None, twins: dict | None = None, all_fast: bool = False, slow: bool = True) -> dict:
    decisions, streak = [], {}
    for t in tasks:
        fr = _fast(t, model, mem)
        conf = fr[0]["posterior"] if fr else 0.0
        cset = conformal_set(fr, q)
        prm, _ = policy.effective(params, t["symptom"], t["cond"])
        reasons = [] if all_fast else route(conf, cset, prm, data_ok=True, severity="S2", verified=1,
                                            streak=streak.get(t["symptom"], 0), exp={})
        streak[t["symptom"]] = 0 if reasons else streak.get(t["symptom"], 0) + 1
        decisions.append((t, prm, reasons, cset, fr))
    slow_jobs = [(t, prm) for t, prm, reasons, _, _ in decisions if reasons and slow]
    res: dict[int, dict] = {}

    async def one(t: dict, prm: dict) -> None:
        tw = (twins or {}).get(t.get("twin")) or twin or FabTwin()
        res[id(t)] = await components(t["evidence"], t["symptom"], mem, model, prm, t["seed"] % 1_000_000 + 40_000_000, tw,
                                      cache=cache, want_rankings=False)
    for i in range(0, len(slow_jobs), 8):  # 多个案例的慢脑并行跑（练习题本身也在进程池里并行）
        await asyncio.gather(*[one(t, prm) for t, prm in slow_jobs[i:i + 8]])
    correct, fast_n, fast_wrong, cost, overrule, covered, sizes = [], 0, 0, 0, 0, 0, 0
    for t, prm, reasons, cset, fr in decisions:
        top = fr[0]["cause_code"] if fr else ""
        if not reasons:
            fast_n += 1
            fast_wrong += top != t["truth"]
        elif slow:
            r = res[id(t)]
            cost += r["cost"]
            arb = arbitrate(r["hits"], r["probe_n"], float(prm["arbiter.min_gain"]))
            if arb["chosen"] != "fast":
                top = r["tops"].get(arb["chosen"]) or {"bayes": t["hyps"][0]["cause_code"] if t["hyps"] else top}.get(arb["chosen"], top)
                overrule += 1
        correct.append(1.0 if top == t["truth"] else 0.0)
        covered += t["truth"] in cset
        sizes += len(cset)
    n = max(1, len(tasks))
    stress_idx = [i for i, (t, *_rest) in enumerate(decisions) if t.get("twin")]
    slow_n = sum(1 for _, _, reasons, _, _ in decisions if reasons)
    return {"n": len(tasks), "hit1": round(100 * sum(correct) / n, 2), "fast_rate": round(fast_n / n, 3),
            "hit1_stress": round(100 * sum(correct[i] for i in stress_idx) / len(stress_idx), 2) if stress_idx else None,
            "cost_per_task": round(cost / n, 2), "fast_wrong": fast_wrong,
            "fast_precision": round((fast_n - fast_wrong) / fast_n, 3) if fast_n else None,
            "overrule_rate": round(overrule / slow_n, 3) if slow_n else 0.0, "coverage": round(covered / n, 3),
            "set_size": round(sizes / n, 2), "_correct": correct}


def _public(ev: dict | None) -> dict | None:
    return {k: v for k, v in ev.items() if not k.startswith("_")} if ev else None


# ── 提议端 ──
def rule_proposals(db: Session, params: dict) -> list[dict]:
    """从改进库（跨产线汇总）提议条件规则：近邻训练反复赢不过快脑的条件下关掉；稳定有效而全局关闭时在该条件下打开；
    快脑反复答错的条件强制转慢脑。"""
    out, have = [], {policy.rule_id(r) for r in params.get("rules", [])}
    for a in library(db):
        if a["symptom"] not in SYMPTOMS:
            continue
        cond = {"symptom": a["symptom"], "evidence": a["evidence"]}
        src = f"library:{a['component']}:{a['symptom']}:{a['evidence']}"
        label = f"{SYMPTOMS[a['symptom']].name}（{'共性显著' if a['evidence'] == 'strong' else '证据模糊'}）"
        if a["component"] == "ttt":
            comp, name = "slow.ttt", "近邻训练"
            if a["n"] >= 4 and a["p"] < 0.45:
                out.append({"if": cond, "set": {comp: 0}, "source": src, "ids": a["ids"],
                            "why": f"改进库：{label}下{name} {a['n']} 次只赢过快脑 {a['ok']:.0f} 次（P={a['p']:.0%}），关掉以省孪生算力"})
            elif a["n"] >= 3 and a["p"] >= 0.72 and not params.get(comp):
                out.append({"if": cond, "set": {comp: 1}, "source": src, "ids": a["ids"],
                            "why": f"改进库：{label}下{name}稳定有效（P={a['p']:.0%}），全局关闭时在该条件下保留"})
        elif a["component"] == "fast" and a["n"] >= 4 and a["p"] < 0.45:
            out.append({"if": cond, "set": {"router.force_slow": 1}, "source": src, "ids": a["ids"],
                        "why": f"改进库：{label}下快脑 {a['n']} 次只对 {a['ok']:.0f} 次（P={a['p']:.0%}），该条件一律转慢脑"})
    return [r for r in out if policy.rule_id(r) not in have]


def critic(db: Session, rule: dict, search: list[dict], select_: list[dict]) -> str:
    """防泄漏审查：返回空串表示通过，否则返回拒绝原因。"""
    bad = policy.valid_rule(rule)
    if bad:
        return bad
    if not str(rule.get("source", "")).startswith("library:"):
        return "来源不是改进经验库"
    for x in db.exec(select(HarnessExperience).where(col(HarnessExperience.id).in_(rule.get("ids") or [-1]))).all():
        if any(set(e) - {"receipt", "case", "probe"} for e in x.evidence):
            return "来源经验里混有评估集样本"
    s = sum(policy.matches(rule, t["symptom"], t["cond"]) for t in search)
    v = sum(policy.matches(rule, t["symptom"], t["cond"]) for t in select_)
    if s < 3 or v < 3:
        return f"支持题不足（search {s} 题、select {v} 题），无法验证"
    return ""


def _propose(parent: dict, max_edits: int, rng: random.Random, gains: dict, stalled: bool, tried: set,
             proposals: list[dict]) -> tuple[dict, list[dict]]:
    ops = [("num", n) for n in EVOLVED] + [("add", i) for i in range(len(proposals))] + [("del", i) for i in range(len(parent["rules"]))]
    w = []
    for kind, x in ops:
        key = x if kind == "num" else "rule"
        base = 2.5 if kind == "add" else (0.8 if kind == "del" else 1.0)
        w.append(base * (2.0 if gains.get(key, 0) > 0 else 1.0) * (1.5 if stalled and key not in tried else 1.0))
    picked: list = []
    while len(picked) < min(max_edits, len(ops)):
        op = rng.choices(ops, weights=w)[0]
        if op not in picked:
            picked.append(op)
    cand = {**parent, "rules": list(parent["rules"])}
    edits = []
    for kind, x in picked:
        if kind == "num":
            _, step, lo, hi, _ = policy.COMPONENTS[x]
            d = rng.choice([-1, 1])
            v = cand[x] + d * step
            if not lo <= v <= hi:
                d, v = -d, cand[x] - d * step
            edits.append({"component": x, "from": cand[x], "to": round(v, 3) if isinstance(v, float) else v,
                          "hypothesis": HYPOTHESIS.get((x, d), "")})
            cand[x] = v
        elif kind == "add":
            r = proposals[x]
            cand["rules"].append(r)
            edits.append({"component": "rule", "op": "add", "rule": {k: r[k] for k in ("if", "set", "why")}, "hypothesis": r["why"]})
        else:
            r = parent["rules"][x]
            cand["rules"] = [y for y in cand["rules"] if policy.rule_id(y) != policy.rule_id(r)]
            edits.append({"component": "rule", "op": "remove", "rule": {k: r[k] for k in ("if", "set", "why")},
                          "hypothesis": "删掉这条规则（检验它是否还有贡献）"})
    out = policy.clamp(cand)
    ids = {policy.rule_id(r) for r in out["rules"]}  # clamp 会丢掉附带字段，ids 留给审查用
    out["rules"] = [r for r in cand["rules"] if policy.rule_id(r) in ids][:policy.MAX_RULES]
    return out, edits


def _sample_parent(archive: list[dict], rng: random.Random) -> dict:
    med = sorted(a["score"] for a in archive)[len(archive) // 2]
    w = [1 / (1 + math.exp(-(a["score"] - med))) / (1 + a["children"]) for a in archive]
    return rng.choices(archive, weights=w)[0]


def _store(params: dict) -> dict:
    return {**{k: v for k, v in params.items() if k != "rules"},
            "rules": [{k: r[k] for k in ("if", "set", "why", "source") if k in r} for r in params.get("rules", [])]}


def _verdict(ev: dict, best: dict, cand: dict, best_params: dict, k: int = 1) -> tuple[str, str]:
    alpha = 0.05 / max(1, k)  # Bonferroni：本次已评估 k 个候选
    mean, lo, hi = paired_ci(ev["_correct"], best["_correct"], alpha=alpha)
    band = f"[{lo * 100:+.1f}, {hi * 100:+.1f}] 个百分点（{(1 - alpha) * 100:.1f}% 区间，已按 {k} 个候选校正）"
    cheaper = ev["cost_per_task"] < 0.9 * best["cost_per_task"]
    simpler = policy.complexity(cand) < policy.complexity(best_params) and ev["cost_per_task"] <= best["cost_per_task"]
    if ev["fast_wrong"] > best["fast_wrong"]:
        return "rejected", "选择集：快脑答错数增加（安全优先，直接拒绝）"
    if lo > 0:
        return "accepted", f"选择集命中率显著提高（配对差 95% 区间 {band}）"
    if lo >= NON_INFERIOR and (cheaper or simpler):
        return "accepted", f"改进在噪声带内（{band}），但{'孪生仿真次数减少 10% 以上' if cheaper else '配置更简单且不更贵'}（复杂度感知接受）"
    if hi < 0:
        return "rejected", f"命中率显著退步（{band}）" + ("，虽然更省" if cheaper else "")
    if lo < NON_INFERIOR:
        return "rejected", f"可能退步（{band}，下界低于 −2 个百分点）" + ("，虽然更省" if cheaper else "")
    return "rejected", f"改进在噪声带内（{band}）且没有降低成本或复杂度"


# ── 何时跑 ──
def _runs(db: Session) -> list[str]:
    ids = db.exec(select(HarnessPolicy.run_id).where(HarnessPolicy.run_id != "").order_by(col(HarnessPolicy.id).desc()).limit(200)).all()
    return list(dict.fromkeys(ids))


def _overrule_share(db: Session, since=None) -> tuple[float | None, int]:
    q = select(HarnessReceipt).where(HarnessReceipt.route == "slow")
    if since is not None:
        q = q.where(col(HarnessReceipt.created_at) > since)
    rows = [r for r in db.exec(q).all() if (r.inner or {}).get("hits")]
    if not rows:
        return None, 0
    won = sum(1 for r in rows if max(v for k, v in r.inner["hits"].items() if k != "fast") > r.inner["hits"]["fast"])
    return round(won / len(rows), 3), len(rows)


def pending_triggers(db: Session) -> list[str]:
    last = db.exec(select(HarnessPolicy).order_by(col(HarnessPolicy.id).desc())).first()
    if last is None:
        return ["bootstrap"]
    since = last.created_at
    out = []
    n_lib = db.exec(select(func.count(col(HarnessExperience.id))).where(col(HarnessExperience.updated_at) > since,
                                                                       col(HarnessExperience.status).in_(["stable", "retired"]))).one()
    if n_lib >= LIB_CHANGES:
        out.append(f"library:{n_lib}")
    fast = db.exec(select(HarnessReceipt).where(HarnessReceipt.route == "fast", col(HarnessReceipt.verified_at) > since)).all()
    wrong = sum(1 for r in fast if not (r.outcome or {}).get("fast_match"))
    if wrong >= FAST_ERR_N and wrong / max(1, len(fast)) >= FAST_ERR_RATE:
        out.append(f"fast_errors:{wrong}/{len(fast)}")
    if db.exec(select(FastBrainModel).where(col(FastBrainModel.created_at) > since,
                                            col(FastBrainModel.status).in_(["active", "retired"]))).first():
        out.append("new_model")
    if db.exec(select(AuditLog).where(AuditLog.action == "twin.update", col(AuditLog.created_at) > since)).first():
        out.append("drift")
    share, n = _overrule_share(db, since)
    hist = [a.detail.get("overrule_share") for a in db.exec(select(AuditLog).where(AuditLog.action == "harness.evolve")
                                                            .order_by(col(AuditLog.id).desc()).limit(2)).all()]
    if share is not None and n >= 20 and len(hist) == 2 and hist[-1] and share <= SAT_DROP * hist[-1]:
        out.append(f"slow_saturated:{share:.0%}")
    runs = _runs(db)[:2]
    if len(runs) == 2 and all(not db.exec(select(HarnessPolicy).where(HarnessPolicy.run_id == r, col(HarnessPolicy.status).in_(
            ["accepted", "active", "rolled_back", "retired"]))).first() for r in runs):
        out = [t for t in out if t.split(":")[0] in STRONG]  # 退避：连续两次没有接受任何候选，只响应强信号
    return out


async def evolve_run(db: Session, mem: MemoryState, *, generations: int = 3, children: int = 2, when=None, seed: int = 11,
                     twin: FabTwin | None = None, trigger: str = "manual") -> dict:
    when = when or utcnow()
    run_id = f"rrsi-{when:%Y%m%d%H%M}-{uuid.uuid4().hex[:4]}"
    rng = random.Random(seed)
    row_m, model = load_active(db)
    model = model or FastModel.bayes(mem.weights)
    q = (row_m.conformal or {}).get("q") if row_m else None
    inc_id, inc = policy.active(db)
    tw = twins_for("search", "select", "report")
    search = await make_split(mem, "search", twin, stress=True, twins=tw)
    select_ = await make_split(mem, "select", twin, stress=True, twins=tw)
    cache: dict = {}
    kw = {"twin": twin, "twins": tw}
    ev_inc = {"search": await evaluate(inc, search, mem, model, q, cache, **kw), "select": await evaluate(inc, select_, mem, model, q, cache, **kw)}
    if inc_id == 0:  # 第一次：默认配置作为基线存档并启用
        base = HarnessPolicy(run_id=run_id, params=_store(inc), status="active", note="初始 harness 配置（基线）",
                             scores={"search": _public(ev_inc["search"]), "select": _public(ev_inc["select"])}, created_at=when)
        db.add(base)
        db.flush()
        inc_id = base.id
    proposals = rule_proposals(db, inc)
    reviews = []
    ok_props = []
    for r in proposals:
        why = critic(db, r, search, select_)
        reviews.append({"rule": {k: r[k] for k in ("if", "set", "why")}, "verdict": "通过" if not why else "拒绝", "reason": why})
        if not why:
            ok_props.append(r)
    gains: dict = {}
    for p in db.exec(select(HarnessPolicy)).all():  # 读历次改动的实测效果（RRSI：后续提议以此为条件）
        for e in p.edits or []:
            gains[e.get("component", "")] = gains.get(e.get("component", ""), 0) + (e.get("delta_hit1") or 0)
    past = db.exec(select(HarnessPolicy).where(col(HarnessPolicy.status).in_(["accepted", "active", "retired"]))).all()
    archive = [{"params": inc, "score": ev_inc["search"]["hit1"], "children": 0, "id": inc_id}]
    archive += [{"params": policy.clamp(p.params), "score": (p.scores.get("search") or {}).get("hit1", 0), "children": 0, "id": p.id}
                for p in past if p.id != inc_id][-6:]
    best = {"params": inc, "ev": ev_inc, "id": inc_id}
    log, tried, stalled, n_eval = [], set(), False, 0
    for gen in range(generations):
        max_edits = max(1, 3 - gen)  # 改动预算退火
        accepted_gen = False
        for _ in range(children):
            parent = _sample_parent(archive, rng)
            parent["children"] += 1
            cand, edits = _propose(parent["params"], max_edits, rng, gains, stalled, tried, ok_props)
            tried.update(e["component"] for e in edits)
            if not edits or _store(cand) == _store(parent["params"]):
                continue
            invalid = [policy.valid_rule(r) for r in cand["rules"] if policy.valid_rule(r)]
            ev_s, ev_sel = None, None
            if invalid:
                status, note = "rejected", "白名单检查：配置不合法（" + "；".join(invalid) + "）"
            else:
                ev_s = await evaluate(cand, search, mem, model, q, cache, **kw)
                if ev_s["fast_wrong"] > best["ev"]["search"]["fast_wrong"] + 1:
                    status, note = "rejected", "搜索集：快脑答错数增加"
                else:
                    ev_sel = await evaluate(cand, select_, mem, model, q, cache, **kw)
                    n_eval += 1
                    status, note = _verdict(ev_sel, best["ev"]["select"], cand, best["params"], n_eval)
                    for e in edits:
                        e["delta_hit1"] = round(ev_sel["hit1"] - best["ev"]["select"]["hit1"], 2)
            row = HarnessPolicy(run_id=run_id, parent_id=parent["id"], params=_store(cand), edits=edits, status=status, note=note,
                                scores={"search": _public(ev_s), "select": _public(ev_sel), "generation": gen, "max_edits": max_edits,
                                        "complexity": policy.complexity(cand)}, created_at=when)
            db.add(row)
            db.flush()
            log.append({"id": row.id, "status": status, "edits": edits, "note": note})
            if status == "accepted":
                accepted_gen = True
                best = {"params": cand, "ev": {"search": ev_s, "select": ev_sel}, "id": row.id}
                archive.append({"params": cand, "score": ev_s["hit1"], "children": 0, "id": row.id})
        stalled = not accepted_gen
    # 剪枝（Lasso 式）：逐条删规则、把没有实测收益的旋钮改回默认，不退步、不更贵就采纳
    prune_ops = [("rule", r) for r in best["params"]["rules"]]
    reset = {k: policy.DEFAULT[k] for k in EVOLVED if best["params"][k] != policy.DEFAULT[k] and gains.get(k, 0) <= 0}
    if reset:
        prune_ops.append(("reset", reset))
    for kind, x in prune_ops:
        if best["id"] == inc_id and kind == "reset":
            break
        cand = {**best["params"], "rules": [r for r in best["params"]["rules"] if kind != "rule" or policy.rule_id(r) != policy.rule_id(x)]}
        if kind == "reset":
            cand.update(x)
        ev_sel = await evaluate(cand, select_, mem, model, q, cache, **kw)
        mean, lo, hi = paired_ci(ev_sel["_correct"], best["ev"]["select"]["_correct"])
        ok = lo >= NON_INFERIOR and ev_sel["fast_wrong"] <= best["ev"]["select"]["fast_wrong"] \
            and ev_sel["cost_per_task"] <= best["ev"]["select"]["cost_per_task"] * 1.05
        edits = ([{"component": "rule", "op": "remove", "rule": {k: x[k] for k in ("if", "set", "why")}, "hypothesis": "剪枝：检验这条规则是否还有贡献"}]
                 if kind == "rule" else
                 [{"component": k, "from": best["params"][k], "to": v, "hypothesis": "剪枝：没有实测收益，改回默认"} for k, v in x.items()])
        row = HarnessPolicy(run_id=run_id, parent_id=best["id"], params=_store(cand), edits=edits, status="accepted" if ok else "rejected",
                            note="剪枝" + ("通过：去掉后不退步、不更贵" if ok else "会让命中率退步或更贵，保留"),
                            scores={"select": _public(ev_sel), "complexity": policy.complexity(cand)}, created_at=when)
        db.add(row)
        db.flush()
        if ok:
            best = {"params": cand, "ev": {**best["ev"], "select": ev_sel}, "id": row.id}
    promoted, report = False, None
    if best["id"] != inc_id:  # report 集上线门：每次都是一批全新的孪生题（永不复用）+ 产线回放
        k_run = len(_runs(db))
        rep = await make_split(mem, "report", twin, stress=True, twins=tw, seed0=REPORT_BASE + (k_run % 300) * 60_000)
        r_inc = await evaluate(inc, rep, mem, model, q, cache, **kw)
        r_new = await evaluate(best["params"], rep, mem, model, q, cache, **kw)
        mean, lo, hi = paired_ci(r_new["_correct"], r_inc["_correct"])
        line = {"incumbent": replay(db, inc, mem, model, q), "candidate": replay(db, best["params"], mem, model, q)}
        costlier = r_new["cost_per_task"] > r_inc["cost_per_task"] * 1.05
        checks = {"fast_wrong": r_new["fast_wrong"] <= r_inc["fast_wrong"], "non_inferior": lo >= NON_INFERIOR, "direction": mean >= 0,
                  "cost": (not costlier) or lo > 0, "line_replay": line["candidate"]["fast_wrong"] <= line["incumbent"]["fast_wrong"]}
        promoted = all(checks.values())
        failed = [{"fast_wrong": "快脑答错增加", "non_inferior": f"命中率下界 {lo * 100:+.1f} 个百分点", "direction": f"平均差 {mean * 100:+.1f} 个百分点 < 0",
                   "cost": "成本变高却没有显著更好", "line_replay": "产线回放里快脑答错更多"}[c] for c, v in checks.items() if not v]
        report = {"incumbent": _public(r_inc), "candidate": _public(r_new), "diff": {"mean": mean, "ci95": [lo, hi]},
                  "fresh_seed": REPORT_BASE + (k_run % 300) * 60_000, "line_replay": line, "checks": checks}
        row = db.get(HarnessPolicy, best["id"])
        row.scores = {**row.scores, "report": report}
        if promoted:
            for old in db.exec(select(HarnessPolicy).where(HarnessPolicy.status == "active")).all():
                old.status = "retired"
                db.add(old)
            row.status, row.note = "active", (row.note + "；上线门全部通过（全新 report 题 + 产线回放），上线").lstrip("；")
        else:
            row.note += "；上线门未通过：" + "、".join(failed)
        db.add(row)
    share, _ = _overrule_share(db)
    db.add(AuditLog(actor="system:rrsi", action="harness.evolve", target=run_id,
                    detail={"trigger": trigger, "candidates": len(log), "accepted": sum(1 for x in log if x["status"] == "accepted"),
                            "promoted": promoted, "overrule_share": share, "reviews": reviews}, created_at=when))
    db.commit()
    return {"run_id": run_id, "trigger": trigger, "incumbent": {"id": inc_id, "params": _store(inc), "search": _public(ev_inc["search"]),
                                                                "select": _public(ev_inc["select"])},
            "reviews": reviews, "candidates": log, "best": {"id": best["id"], "params": _store(best["params"])},
            "report": report, "promoted": promoted}


def replay(db: Session, params: dict, mem: MemoryState, model: FastModel, q: float | None, limit: int = 200) -> dict:
    """产线回放：用最近已确认的产线案例（证据 + 现场确认的根因）重放某个配置的快慢路由，数快脑直接给结论时答错几次。"""
    from ..models import Case

    rows = db.exec(select(HarnessReceipt, Case).where(HarnessReceipt.case_id == Case.id, col(HarnessReceipt.verified_at).is_not(None))
                   .order_by(col(HarnessReceipt.verified_at).desc()).limit(limit)).all()
    n = fast = wrong = 0
    for rc, case in rows:
        truth = (rc.outcome or {}).get("cause_code")
        if not truth or not case.evidence or not rc.symptom:
            continue
        hyps = [h.as_dict() for h in rank(case.evidence, rc.symptom, mem.edges, mem.hotspot_weights(), mem.weights)]
        fr = model_rank(model, case.evidence, rc.symptom, mem, hyps)
        if not fr:
            continue
        prm, _ = policy.effective(params, rc.symptom, condition_of(case.evidence))
        reasons = route(fr[0]["posterior"], conformal_set(fr, q), prm, data_ok=True, severity=case.severity, verified=1, streak=0, exp={})
        n += 1
        if not reasons:
            fast += 1
            wrong += fr[0]["cause_code"] != truth
    return {"n": n, "fast": fast, "fast_wrong": wrong}


def _error_rate(db: Session, version: int) -> tuple[int, int]:
    rows = db.exec(select(HarnessReceipt).where(HarnessReceipt.policy_version == version, HarnessReceipt.route == "fast",
                                                col(HarnessReceipt.verified_at).is_not(None))).all()
    return sum(1 for r in rows if not (r.outcome or {}).get("fast_match")), len(rows)


def probation_check(db: Session, when=None) -> dict:
    """上线观察期（Agnes：新版本先观察，错误率显著上升自动回滚）。

    统计检验而不是固定阈值：新配置下已确认的快脑结论（≥ 10 条），错误率 Wilson 95% 区间的下界
    同时超过“上一版的错误率”和“路由门槛允许的错误率（1 − 快脑门槛）”就回滚——即有把握说它比上一版差、也比承诺的差。
    """
    from ..stats import wilson

    act = db.exec(select(HarnessPolicy).where(HarnessPolicy.status == "active").order_by(col(HarnessPolicy.id).desc())).first()
    if act is None:
        return {"status": "default_policy"}
    wrong, n = _error_rate(db, act.id)
    pw, pn = _error_rate(db, act.parent_id) if act.parent_id else (0, 0)
    baseline = max(pw / pn if pn else 0.0, 1 - float(policy.clamp(act.params)["router.fast_min_conf"]))
    lower = (wilson(wrong, n) or [0.0])[0]
    if n >= 10 and lower > baseline:
        parent = db.get(HarnessPolicy, act.parent_id) if act.parent_id else None
        act.status = "rolled_back"
        db.add(act)
        back = HarnessPolicy(run_id=act.run_id, parent_id=act.id, params=parent.params if parent else _store(policy.DEFAULT),
                             status="active", created_at=when or utcnow(),
                             note=f"观察期回滚：快脑确认 {n} 次中答错 {wrong} 次，错误率下界 {lower:.0%} 高于基准 {baseline:.0%}")
        db.add(back)
        db.add(AuditLog(actor="system:rrsi", action="harness.rollback", target=act.run_id,
                        detail={"fast": n, "wrong": wrong, "lower": lower, "baseline": baseline}, created_at=when or utcnow()))
        db.commit()
        return {"status": "rolled_back", "fast": n, "wrong": wrong, "lower": lower, "baseline": round(baseline, 3)}
    return {"status": "ok", "fast": n, "wrong": wrong, "lower": lower, "baseline": round(baseline, 3)}


async def harness_benchmark(db: Session, mem: MemoryState, params: dict | None = None) -> dict:
    """孪生对照：同一批 report 题——只用贝叶斯 / 只用快脑模型 / 快慢双脑；以及“产线刚变更、模型和记忆都没跟上”的压力场景。"""
    params = params or policy.clamp({})
    row_m, model = load_active(db)
    bayes = FastModel.bayes(mem.weights)
    model = model or bayes
    q = (row_m.conformal or {}).get("q") if row_m else None
    cache: dict = {}
    tasks = await make_split(mem, "report", None)
    b_only = await evaluate(params, tasks, mem, bayes, None, cache, all_fast=True)
    m_only = await evaluate(params, tasks, mem, model, q, cache, all_fast=True)
    dual = await evaluate(params, tasks, mem, model, q, cache)
    stress = STRESS["report"]
    twin = FabTwin(truth=stressed_truth(stress))
    s_tasks = await make_split(mem, "report", twin, symptoms=stress, per_symptom=20)
    s_cache: dict = {}
    s_b = await evaluate(params, s_tasks, mem, bayes, None, s_cache, twin=twin, all_fast=True)
    s_m = await evaluate(params, s_tasks, mem, model, q, s_cache, twin=twin, all_fast=True)
    s_d = await evaluate(params, s_tasks, mem, model, q, s_cache, twin=twin)
    ci = lambda a, b: dict(zip(("mean", "ci95"), (lambda m, lo, hi: (m, [lo, hi]))(*paired_ci(a["_correct"], b["_correct"]))))  # noqa: E731
    return {"params": _store(params), "fast_model": row_m.id if row_m else 0,
            "bayes_only": _public(b_only), "model_only": _public(m_only), "dual": _public(dual),
            "model_vs_bayes": ci(m_only, b_only), "dual_vs_model": ci(dual, m_only),
            "stress": {"symptoms": stress, "change": "第一、第二大根因份额对调（孪生已同步，快脑模型和记忆都还停在变更前）",
                       "bayes_only": _public(s_b), "model_only": _public(s_m), "dual": _public(s_d), "diff": ci(s_d, s_m)}}
