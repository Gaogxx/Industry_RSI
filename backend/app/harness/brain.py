"""快慢双脑。参照 Agnes Harness 技术报告（2026-11）第 7 节的 System 1 / System 2：

- 快脑（System 1）= 训练出来的小模型（app/fastbrain，Wide & Deep & Cross 排序网络），毫秒级给出根因排序和
  校准概率，不跑孪生、不调大模型。它通过训练窗口的晋升门（留出集上不比在用版本和贝叶斯差）之后才拿到裁决权；
  还没有通过晋升门的模型时，快脑退回贝叶斯排序。
- 慢脑（System 2）= 大模型调查员（有 Key 时）+ 近邻训练（临时记忆 + 快脑模型重排）+ 贝叶斯意见，由探测题裁决（见 slow.py）。
- 必须交给慢脑的情形（Agnes：任务第一步、上一步失败、设备检查、连续多步后强制接管）在 Kaizen 对应为：
  缺数据、S1 重大、该产线该症状还没有现场确认过的先例、改进库记录快脑在同条件下答错过、
  同症状连续多次快脑后的周期抽查、快脑模型的校准概率低于阈值、保形集合不止一个根因、条件规则要求转慢脑。
"""

from __future__ import annotations

from sqlmodel import Session, col, func, select

from ..fastbrain.infer import model_rank
from ..fastbrain.model import FastModel
from ..fastbrain.windows import CONFORMAL_ALPHA, load_active
from ..models import HarnessReceipt, utcnow
from . import policy
from .experience import reflect_inner, relevant, scope_of, signature, view
from .slow import contrast_probes, slow_brain

MIN_LOTS, MIN_FAIL_LOTS = 30, 3


def conformal_set(ranking: list[dict], q: float | None) -> list[str]:
    """保形预测集合：分数 = 1 − 模型给真因的概率；阈值 q 来自孪生校准集。没有校准时退回到“累计概率 ≥ 90%”。"""
    if not ranking:
        return []
    if q is None:
        out, mass = [], 0.0
        for h in ranking:
            out.append(h["cause_code"])
            mass += h["posterior"]
            if mass >= 1 - CONFORMAL_ALPHA:
                break
        return out
    return [h["cause_code"] for h in ranking if h["posterior"] >= 1 - q] or [ranking[0]["cause_code"]]


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


def route(conf: float, cset: list[str], params: dict, *, data_ok: bool, severity: str, verified: int, streak: int, exp: dict,
          rules: list[str] | None = None) -> list[str]:
    reasons = []
    if not data_ok:
        reasons.append("missing_data")
    if severity == "S1":
        reasons.append("critical")
    if verified == 0:
        reasons.append("first_seen")
    if params.get("experience.use") and exp.get("fast") is not None and exp["fast"].status == "retired":
        reasons.append("prior_error")
    if streak >= int(params["router.audit_every"]) - 1:
        reasons.append("audit")
    if conf < float(params["router.fast_min_conf"]):
        reasons.append("low_conf")
    if len(cset) > 1:
        reasons.append("ambiguous")
    if params.get("router.force_slow"):
        reasons.append("rule")
    return reasons


REASON_TEXT = {"missing_data": "缺少批次数据", "critical": "S1 重大", "first_seen": "该产线该症状还没有现场确认过的先例",
               "prior_error": "改进库：同条件下快脑答错过", "audit": "周期抽查（连续快脑后强制慢脑）",
               "low_conf": "快脑模型的校准概率低于阈值", "ambiguous": "保形集合不止一个根因", "rule": "条件规则要求转慢脑"}


def fast_brain(db: Session, evidence: dict, symptom: str, hyps: list[dict], mem, model_pack=None) -> dict:
    """快脑：有已晋升的模型就用模型排序（校准概率 + 保形阈值），否则退回贝叶斯。"""
    row, model = model_pack if model_pack is not None else load_active(db)
    if model is None:
        model = FastModel.bayes(mem.weights)
        ranking = model_rank(model, evidence, symptom, mem, hyps)
        return {"model": model, "row": None, "ranking": ranking, "q": None, "kind": "bayes"}
    return {"model": model, "row": row, "ranking": model_rank(model, evidence, symptom, mem, hyps),
            "q": (row.conformal or {}).get("q"), "kind": "model"}


async def decide(db: Session, case: dict, evidence: dict, symptom: str, hyps: list[dict], mem, *, seed: int,
                 twin=None, when=None, model_pack=None, params: tuple[int, dict] | None = None, slow: bool = True) -> dict:
    """返回 {"receipt": HarnessReceipt（未落库）, "final": 最终排序, "answer": 给工程师看的结论状态}。"""
    when = when or utcnow()
    pid, base = params or policy.active(db)
    scope = scope_of(case)
    sig = signature(evidence, hyps)
    prm, rule_hits = policy.effective(base, symptom, sig)
    exp = relevant(db, scope, symptom, sig)
    fb = fast_brain(db, evidence, symptom, hyps, mem, model_pack)
    fast_rank, q = fb["ranking"], fb["q"]
    conf = fast_rank[0]["posterior"] if fast_rank else 0.0
    cset = conformal_set(fast_rank, q)
    summary = (evidence or {}).get("summary") or {}
    data_ok = summary.get("lots", 0) >= MIN_LOTS and summary.get("fail_lots", 0) >= MIN_FAIL_LOTS
    verified, streak = _counters(db, scope, symptom, when)
    reasons = route(conf, cset, prm, data_ok=data_ok, severity=case.get("severity", ""), verified=verified, streak=streak, exp=exp)
    path = "slow" if reasons else "fast"
    final, inner = fast_rank, {}
    if path == "slow" and slow:
        inner = await slow_brain(evidence, symptom, mem, fb["model"], fast_rank, hyps, prm, exp, seed, twin)
        final = inner.pop("final")
        reflect_inner(db, scope, symptom, sig, inner, when)
    elif path == "slow":
        inner = {"probes": contrast_probes(hyps), "proposals": [], "compose": {"adopted": "快脑模型", "why": "未运行慢脑"}}
    inner["rules"] = rule_hits
    chosen = (inner.get("arbiter") or {}).get("chosen", "fast")
    rel = final[0]["posterior"] if final else 0.0
    cset_f = conformal_set(final, q if chosen in ("fast", "ttt") else None)
    tie = bool((inner.get("arbiter") or {}).get("tie"))
    mode = int(prm.get("investigator.mode", 1))
    investigate = path == "slow" and (mode == 1 or (mode == 2 and (case.get("severity") == "S1" or tie)))
    used = [x for x in exp.values() if x is not None]
    receipt = HarnessReceipt(case_id=case.get("id"), scope=scope, symptom=symptom, signature={**sig, "conf": round(conf, 4)},
                             route=path, reasons=reasons, policy_version=pid, reliability=round(rel, 4),
                             fast_model=fb["row"].id if fb["row"] is not None else 0,
                             conformal={"alpha": CONFORMAL_ALPHA, "q": q, "set": cset_f,
                                        "method": "split-conformal" if q is not None else "posterior-mass"},
                             inner=inner, experiences=[x.id for x in used],
                             top_fast=fast_rank[0]["cause_code"] if fast_rank else "", top=final[0]["cause_code"] if final else "",
                             created_at=when)
    sensitive = (inner.get("compose") or {}).get("sensitive", [])
    who = "快脑模型" if fb["kind"] == "model" else "快脑（贝叶斯：还没有通过晋升门的模型）"
    if path == "fast" and len(cset_f) == 1:
        status, text = "supported", f"{who}结论：校准概率 {rel:.0%}，90% 把握的根因集合只有首位"
    else:
        status = "needs_verification"
        text = (f"{'慢脑' if path == 'slow' else who}结论（{(inner.get('compose') or {}).get('adopted', who)}）："
                f"概率 {rel:.0%}；90% 把握的根因集合含 {len(cset_f)} 个根因" + ("，需现场鉴别" if len(cset_f) > 1 else "")
                + (f"；结论对「{'、'.join(sensitive)}」敏感" if sensitive else ""))
    return {"receipt": receipt, "final": final,
            "answer": {"status": status, "text": text, "route": path, "reasons": [REASON_TEXT.get(r, r) for r in reasons],
                       "reliability": round(rel, 4), "set": cset_f, "experiences": [view(x) for x in used], "inner": inner,
                       "fast_kind": fb["kind"], "fast_model": receipt.fast_model, "investigate": investigate}}
