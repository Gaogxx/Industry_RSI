"""快慢双脑。参照 Agnes Harness 技术报告（2026-11）第 7 节的 System 1 / System 2 路由：

- 快脑（System 1）= 统计内核 + 可进化记忆：毫秒级给出排序，不调大模型、不跑孪生；
- 慢脑（System 2）= 双层递归的内层：证据族对照 + 近邻练习（测试时训练），可选再加大模型调查员；
- 必须交给慢脑的情形（Agnes：每个任务第一步、上一步失败、设备检查、连续多步后强制接管）在 Kaizen 对应为：
  缺数据、S1 重大、该产线该症状还没有任何现场确认过的先例、经验库记录快脑在同条件下答错过、
  同症状连续多次快脑后的周期抽查、校准可靠度低于阈值、保形集合里不止一个根因。
- 交给慢脑时附上快脑的排序和可靠度，慢脑在此基础上探测与合成。
"""

from __future__ import annotations

import math

from sqlmodel import Session, col, func, select

from ..models import HarnessReceipt, utcnow
from . import policy
from .experience import p_success, relevant, reflect_inner, scope_of, signature, view
from .recursion import contrast_probes, inner_loop

FEATURES = ("bias", "top_post", "margin", "entropy", "strong", "n_signals", "after_pm", "fact_support", "fast_tip")
MIN_LOTS, MIN_FAIL_LOTS = 30, 3


def feature_vector(hyps: list[dict], sig: dict, exp: dict) -> dict:
    posts = [h["posterior"] for h in hyps] or [0.0]
    top = posts[0]
    ent = -sum(p * math.log(p) for p in posts if p > 0) / math.log(max(2, len(posts)))
    top_code = hyps[0]["cause_code"] if hyps else ""
    return {"bias": 1.0, "top_post": top, "margin": top - (posts[1] if len(posts) > 1 else 0.0), "entropy": ent,
            "strong": 1.0 if sig.get("evidence") == "strong" else 0.0, "n_signals": float(len(sig.get("signals", []))),
            "after_pm": 1.0 if sig.get("after_pm") else 0.0,
            "fact_support": 1.0 if any((f.action or {}).get("cause") == top_code for f in exp.get("facts", [])) else 0.0,
            "fast_tip": p_success(exp["fast"]) if exp.get("fast") is not None else 0.5}


def reliability(fv: dict, model) -> float:
    """首位根因是真因的概率：有已晋升的可靠度模型就用它（校准过），否则用后验近似。"""
    if model is not None and model.coef:
        z = sum(model.coef.get(k, 0.0) * fv[k] for k in FEATURES)
        return 1 / (1 + math.exp(-max(-30.0, min(30.0, z))))
    return fv["top_post"]


def conformal_set(hyps: list[dict], q: float | None) -> list[str]:
    """保形预测集合：分数 = 1 − 真因后验；阈值 q 来自孪生校准集。没有校准时退回到“累计后验 ≥ 90%”。"""
    if not hyps:
        return []
    if q is None:
        out, mass = [], 0.0
        for h in hyps:
            out.append(h["cause_code"])
            mass += h["posterior"]
            if mass >= 1 - policy.CONFORMAL_ALPHA:
                break
        return out
    return [h["cause_code"] for h in hyps if h["posterior"] >= 1 - q] or [hyps[0]["cause_code"]]


def _counters(db: Session, scope: str, symptom: str, before) -> tuple[int, int]:
    verified = db.exec(select(func.count(col(HarnessReceipt.id))).where(
        HarnessReceipt.scope == scope, HarnessReceipt.symptom == symptom, col(HarnessReceipt.verified_at).is_not(None),
        col(HarnessReceipt.verified_at) <= before)).one()
    recent = db.exec(select(HarnessReceipt.route).where(HarnessReceipt.scope == scope, HarnessReceipt.symptom == symptom,
                                                        col(HarnessReceipt.created_at) <= before)
                     .order_by(col(HarnessReceipt.created_at).desc()).limit(12)).all()
    streak = 0
    for r in recent:
        if r != "fast":
            break
        streak += 1
    return int(verified), streak


def route(fv: dict, rel: float, cset: list[str], params: dict, *, data_ok: bool, severity: str, verified: int,
          streak: int, exp: dict) -> list[str]:
    reasons = []
    if not data_ok:
        reasons.append("missing_data")
    if severity == "S1":
        reasons.append("critical")
    if verified == 0:
        reasons.append("first_seen")
    if params.get("use_experience") and exp.get("fast") is not None and exp["fast"].status == "retired":
        reasons.append("prior_error")
    if streak >= int(params["audit_every"]) - 1:
        reasons.append("audit")
    if rel < float(params["fast_min_reliability"]):
        reasons.append("low_reliability")
    if len(cset) > 1:
        reasons.append("ambiguous")
    return reasons


REASON_TEXT = {"missing_data": "缺少批次数据", "critical": "S1 重大", "first_seen": "该产线该症状还没有现场确认过的先例",
               "prior_error": "经验库：同条件下快脑答错过", "audit": "周期抽查（连续快脑后强制慢脑）",
               "low_reliability": "可靠度低于阈值", "ambiguous": "保形集合不止一个根因"}


async def decide(db: Session, case: dict, evidence: dict, symptom: str, hyps: list[dict], mem, *, seed: int,
                 twin=None, when=None, model=None, params: tuple[int, dict] | None = None, slow_brain: bool = True) -> dict:
    """返回 {"receipt": HarnessReceipt（未落库）, "final": 最终排序, "answer": 给工程师看的结论状态}。"""
    from .train import active_model

    when = when or utcnow()
    pid, prm = params or policy.active(db)
    model = model if model is not None else active_model(db)
    scope = scope_of(case)
    sig = signature(evidence, hyps)
    exp = relevant(db, scope, symptom, sig)
    q = (model.conformal or {}).get("q") if model is not None else None
    fv = feature_vector(hyps, sig, exp)
    rel = reliability(fv, model)
    cset = conformal_set(hyps, q)
    summary = (evidence or {}).get("summary") or {}
    data_ok = summary.get("lots", 0) >= MIN_LOTS and summary.get("fail_lots", 0) >= MIN_FAIL_LOTS
    verified, streak = _counters(db, scope, symptom, when)
    reasons = route(fv, rel, cset, prm, data_ok=data_ok, severity=case.get("severity", ""), verified=verified,
                    streak=streak, exp=exp)
    path = "slow" if reasons else "fast"
    final, inner = hyps, {}
    if path == "slow" and slow_brain:
        inner = await inner_loop(evidence, symptom, hyps, mem, prm, exp, seed, twin)
        final = inner.pop("final")
        reflect_inner(db, scope, symptom, sig, inner, when)
    elif path == "slow":
        inner = {"probes": contrast_probes(hyps), "proposals": [], "compose": {"adopted": "原排序", "why": "未运行慢脑"}}
    sig_f = signature(evidence, final)
    fv_f = feature_vector(final, sig_f, exp)
    rel_f = reliability(fv_f, model)
    cset_f = conformal_set(final, q)
    used = [x for x in (exp.get("fast"), exp.get("ttt"), *exp.get("facts", [])) if x is not None]
    receipt = HarnessReceipt(case_id=case.get("id"), scope=scope, symptom=symptom, signature={**sig, "fv": fv}, route=path, reasons=reasons,
                             policy_version=pid, reliability=round(rel_f, 4), reliability_model=model.id if model is not None else 0,
                             conformal={"alpha": policy.CONFORMAL_ALPHA, "q": q, "set": cset_f,
                                        "method": "split-conformal" if q is not None else "posterior-mass"},
                             inner=inner, experiences=[x.id for x in used],
                             top_fast=hyps[0]["cause_code"] if hyps else "", top=final[0]["cause_code"] if final else "",
                             created_at=when)
    sensitive = (inner.get("compose") or {}).get("sensitive", [])
    if path == "fast" and len(cset_f) == 1:
        status = "supported"
        text = f"快脑结论：可靠度 {rel_f:.0%}，90% 把握的根因集合只有首位"
    else:
        status = "needs_verification"
        text = (f"{'慢脑' if path == 'slow' else '快脑'}结论：可靠度 {rel_f:.0%}；90% 把握的根因集合含 {len(cset_f)} 个根因，需现场鉴别"
                + (f"；结论对「{'、'.join(sensitive)}」敏感" if sensitive else ""))
    return {"receipt": receipt, "final": final,
            "answer": {"status": status, "text": text, "route": path, "reasons": [REASON_TEXT.get(r, r) for r in reasons],
                       "reliability": round(rel_f, 4), "set": cset_f, "experiences": [view(x) for x in used],
                       "inner": inner}}
