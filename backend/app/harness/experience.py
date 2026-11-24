"""双轨任务经验库与外层递归的经验反思。

参照：
- TTSE（arXiv:2609.24289）：经验分两轨——fact（可核实的产线事实）与 tip（“在什么条件下怎么做”）；
  失败先追责到具体条目，被追责的条目停用但留档；拿不准时记为 tip，不把未经核实的说法升格为事实。
- Bayesian-Agent（arXiv:2606.08348）：每条经验是一个假设，用 Beta 后验记成败；
  P(成功) < 0.45 且失败 ≥ 4 次停用，P ≥ 0.72 且 ≥ 3 次标为稳定。
- Harness²（arXiv:2610.14224）外层递归：任务做完后由反思模块写下“什么条件下哪种做法修好了什么”，
  只有执行对比支持的经验才入库；下一个任务仍从干净的基础流程出发，改进的是“改进流程的能力”。
- Agnes Harness 技术报告：只从已验证的任务里学；标签被推翻时相关经验自动撤销。

Kaizen 的经验只来自两种可验证的信号：孪生近邻练习的探测结果（干预反馈判定）与工程师关案确认的根因。
模型自己的高置信结论从不当作标签。
"""

from __future__ import annotations

import hashlib

from sqlmodel import Session, col, select

from ..fab.flow import SYMPTOMS, cause_by_code
from ..models import Case, HarnessExperience, HarnessReceipt, utcnow
from ..rsi.memory import condition_of

RETIRE_P, RETIRE_FAILS = 0.45, 4
STABLE_P, STABLE_N = 0.72, 3
FACT_MIN_VERIFIED = 3  # 事实轨：至少 3 次独立的现场确认才生效


def scope_of(case: dict) -> str:
    site, product = (" ".join(str(case.get(k) or "").split()) for k in ("site", "product"))
    if not site:
        return ""
    return "s-" + hashlib.sha1(f"{site}|{product}".encode()).hexdigest()[:10]


def signature(evidence: dict, hyps: list[dict]) -> dict:
    cond = condition_of(evidence or {})
    names = {f["name"] for f in (hyps[0]["factors"] if hyps else []) if f.get("x", 1) > 1.05}
    return {**cond, "top": hyps[0]["cause_code"] if hyps else "", "signals": sorted(names - {"先验知识"})}


def _cond_text(symptom: str, cond: dict) -> str:
    return "；".join([SYMPTOMS[symptom].name if symptom in SYMPTOMS else symptom,
                     "共性分析显著" if cond.get("evidence") == "strong" else "证据模糊",
                     {"tool": "富集在机台", "batch": "富集在物料批次", "none": "无富集"}.get(cond.get("factor"), "无富集"),
                     "PM 后突变" if cond.get("after_pm") else "与 PM 无关"])


def p_success(x: HarnessExperience) -> float:
    return x.alpha / (x.alpha + x.beta)


def trials(x: HarnessExperience) -> float:
    return x.alpha + x.beta - 2


def _settle(x: HarnessExperience) -> None:
    p, fails = p_success(x), x.beta - 1
    if x.status != "retired" and p < RETIRE_P and fails >= RETIRE_FAILS:
        x.status, x.retired_reason = "retired", f"成功率 {p:.0%}，失败 {fails:.0f} 次（低于 {RETIRE_P:.0%} 且失败 ≥ {RETIRE_FAILS} 次）"
    elif x.status == "active" and p >= STABLE_P and trials(x) >= STABLE_N:
        x.status = "stable"
    elif x.status == "stable" and p < STABLE_P:
        x.status = "active"


def _get(db: Session, key: str, **init) -> HarnessExperience:
    x = db.exec(select(HarnessExperience).where(HarnessExperience.key == key)).first()
    if x is None:
        x = HarnessExperience(key=key, **init)
        db.add(x)
        db.flush()
    return x


def _record(x: HarnessExperience, ok: bool, evidence: dict, when=None, blame: str = "") -> None:
    if ok:
        x.alpha += 1
    else:
        x.beta += 1
        if blame:
            x.blame = (list(x.blame) + [{**evidence, "why": blame}])[-20:]
    x.evidence = (list(x.evidence) + [evidence])[-30:]
    x.updated_at = when or utcnow()
    _settle(x)


def relevant(db: Session, scope: str, symptom: str, cond: dict) -> dict:
    """检索与当前任务条件相同的经验：fast tip、近邻练习 tip、生效的事实。"""
    ev = cond.get("evidence", "weak")
    keys = [f"tip:fast:{scope}:{symptom}:{ev}", f"tip:ttt:{scope}:{symptom}:{ev}"]
    rows = {x.key: x for x in db.exec(select(HarnessExperience).where(col(HarnessExperience.key).in_(keys))).all()}
    facts = db.exec(select(HarnessExperience).where(HarnessExperience.track == "fact", HarnessExperience.symptom == symptom,
                                                    col(HarnessExperience.scope).in_([scope, ""]),
                                                    col(HarnessExperience.status).in_(["active", "stable"]))).all()
    return {"fast": rows.get(keys[0]), "ttt": rows.get(keys[1]), "facts": list(facts)}


def view(x: HarnessExperience | None) -> dict | None:
    if x is None:
        return None
    return {"id": x.id, "track": x.track, "text": x.text, "status": x.status, "p": round(p_success(x), 3),
            "n": int(trials(x)), "action": x.action, "retired_reason": x.retired_reason}


# ── 外层递归 ①：内层做完立刻反思（证据 = 孪生近邻练习的探测对比，有真值）──
def reflect_inner(db: Session, scope: str, symptom: str, cond: dict, inner: dict, when=None) -> HarnessExperience | None:
    ttt = inner.get("ttt")
    if not ttt or not ttt.get("probe_n"):
        return None
    gain = ttt["probe_after"] - ttt["probe_before"]
    if gain == 0 and ttt["top_before"] == ttt["top_after"]:
        return None  # 没有可观察的行为差异：Harness² 的门控——证据不足就不新增条目
    ev = cond.get("evidence", "weak")
    x = _get(db, f"tip:ttt:{scope}:{symptom}:{ev}", track="tip", scope=scope, symptom=symptom, condition=cond,
             action={"do": "ttt"}, text=f"在［{_cond_text(symptom, cond)}］下先做近邻练习（测试时训练）再下结论", created_at=when or utcnow())
    _record(x, gain > 0, {"probe": {"before": ttt["probe_before"], "after": ttt["probe_after"], "n": ttt["probe_n"]}}, when,
            blame="近邻练习没有提高探测题命中率" if gain <= 0 else "")
    return x


# ── 外层递归 ②：关案后反思（证据 = 工程师确认的根因）──
def reflect_verified(db: Session, case: Case, when=None) -> dict:
    rc = db.exec(select(HarnessReceipt).where(HarnessReceipt.case_id == case.id, col(HarnessReceipt.verified_at).is_(None))
                 .order_by(col(HarnessReceipt.id).desc())).first()
    if rc is None or not case.cause_code:
        return {}
    when = when or case.closed_at or utcnow()
    truth, cond, sym, scope = case.cause_code, rc.signature or {}, rc.symptom, rc.scope
    ok = rc.top == truth
    rc.outcome = {"cause_code": truth, "top_match": ok, "fast_match": rc.top_fast == truth,
                  "in_set": truth in (rc.conformal or {}).get("set", [])}
    rc.verified_at = when
    db.add(rc)
    ev = cond.get("evidence", "weak")
    touched = []
    if rc.route == "fast":  # 快脑直接给的结论：对了加分，错了追责（之后同条件先转慢脑）
        x = _get(db, f"tip:fast:{scope}:{sym}:{ev}", track="tip", scope=scope, symptom=sym, condition=cond, action={"do": "fast"},
                 text=f"在［{_cond_text(sym, cond)}］下快脑可以直接给结论", created_at=when)
        _record(x, ok, {"receipt": rc.id, "case": case.code}, when, blame=f"{case.code} 快脑首位与现场确认不一致")
        touched.append(x)
    elif ((rc.inner or {}).get("ttt") or {}).get("adopt"):
        x = _get(db, f"tip:ttt:{scope}:{sym}:{ev}", track="tip", scope=scope, symptom=sym, condition=cond, action={"do": "ttt"},
                 text=f"在［{_cond_text(sym, cond)}］下先做近邻练习（测试时训练）再下结论", created_at=when)
        _record(x, ok, {"receipt": rc.id, "case": case.code}, when, blame=f"{case.code} 采纳近邻练习后的首位仍与现场确认不一致")
        touched.append(x)
    found = cause_by_code(truth)
    tool = next((s["value"] for s in (case.evidence or {}).get("suspects", [])
                 if found and s["step"] == found[1].step and "#" not in s["factor"] and s.get("p_adj", 1) < 0.05), None)
    if tool:  # 事实轨：机台 ↔ 根因的关联，只记现场确认过的；满 3 次才生效（之前是候选）
        x = _get(db, f"fact:tool:{scope}:{sym}:{tool}:{truth}", track="fact", scope=scope, symptom=sym,
                 condition={"tool": tool}, action={"cause": truth, "tool": tool}, status="candidate",
                 text=f"{tool} 与「{found[1].name}」相关（现场确认）", created_at=when)
        _record(x, True, {"receipt": rc.id, "case": case.code}, when)
        if x.status == "candidate" and trials(x) >= FACT_MIN_VERIFIED:
            x.status = "active"
        touched.append(x)
    db.flush()
    return {"receipt": rc.id, "top_match": ok, "experiences": [view(x) for x in touched]}


def revoke_for_case(db: Session, case_id: int) -> int:
    """标签被推翻（重新打开案例）时，撤回由该案例带来的经验计数（Agnes：标签被推翻自动撤销）。"""
    rc = db.exec(select(HarnessReceipt).where(HarnessReceipt.case_id == case_id, col(HarnessReceipt.verified_at).is_not(None))
                 .order_by(col(HarnessReceipt.id).desc())).first()
    if rc is None:
        return 0
    n = 0
    for x in db.exec(select(HarnessExperience)).all():
        mine = [e for e in x.evidence if e.get("receipt") == rc.id]
        if not mine:
            continue
        failed = any(b.get("receipt") == rc.id for b in x.blame)
        x.evidence = [e for e in x.evidence if e.get("receipt") != rc.id]
        x.blame = [b for b in x.blame if b.get("receipt") != rc.id]
        if failed:
            x.beta = max(1.0, x.beta - 1)
        else:
            x.alpha = max(1.0, x.alpha - 1)
        if x.status == "retired" and p_success(x) >= RETIRE_P:
            x.status, x.retired_reason = "active", ""
        _settle(x)
        db.add(x)
        n += 1
    rc.verified_at, rc.outcome = None, {}
    db.add(rc)
    return n
