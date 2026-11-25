"""harness 的改进经验库（双层递归的外层）与经验反思。

参照：
- Harness²（arXiv:2610.14224）：任务做完后由反思模块写下“在什么条件下哪个组件 / 做法有效”，只有执行对比支持的
  经验才入库（没有可观察的行为差异就不新增条目）；下一个任务仍从干净的基础 harness 出发——改进库积累的是
  “改进 harness 的能力”。Kaizen 的执行对比是孪生探测题（有程序给出的真值），比纯行为对比更强。
- Agnes Harness 技术报告：只从已验证的任务里学；标签被推翻（重新打开案例）时相关经验自动撤销。
- Beta 账本（见 app/ledger.py）：P<0.45 且失败≥4 次停用（留档可查），P≥0.72 且≥3 次标为稳定。

条目都是“做法”（tip）：
  tip:fast:{产线}:{症状}:{证据强弱}         在该条件下快脑可以直接给结论（关案时按快脑首位对错记成败）
  tip:ttt:{产线}:{症状}:{证据强弱}         在该条件下近邻训练能赢过快脑（探测题对比）
产线事实（机台 ↔ 根因）由 RSIAgent 的 TTSE 事实轨管理（rsi/memory.py），这里不重复记。
RRSI 在触发时从这里汇总跨产线的规律，提议条件规则（见 evolve.rule_proposals）。
"""

from __future__ import annotations

import hashlib

from sqlmodel import Session, col, select

from ..fab.flow import SYMPTOMS
from ..ledger import RETIRE_P, p_success, record as _record, settle as _settle, trials  # noqa: F401
from ..models import Case, HarnessExperience, HarnessReceipt, utcnow
from ..rsi.memory import condition_of

COMPONENTS = ("fast", "ttt")
TEXT = {"fast": "快脑可以直接给结论", "ttt": "近邻训练能赢过快脑"}


def scope_of(case: dict) -> str:
    site, product = (" ".join(str(case.get(k) or "").split()) for k in ("site", "product"))
    if not site:
        return ""
    return "s-" + hashlib.sha1(f"{site}|{product}".encode()).hexdigest()[:10]


def signature(evidence: dict, hyps: list[dict]) -> dict:
    cond = condition_of(evidence or {})
    names = {f["name"] for f in (hyps[0]["factors"] if hyps else []) if f.get("x", 1) > 1.05}
    return {**cond, "top": hyps[0]["cause_code"] if hyps else "", "signals": sorted(names - {"先验知识"})}


def cond_text(symptom: str, cond: dict) -> str:
    return "；".join([SYMPTOMS[symptom].name if symptom in SYMPTOMS else symptom,
                     "共性分析显著" if cond.get("evidence") == "strong" else "证据模糊",
                     {"tool": "富集在机台", "batch": "富集在物料批次", "none": "无富集"}.get(cond.get("factor"), "无富集"),
                     "PM 后突变" if cond.get("after_pm") else "与 PM 无关"])


def key_of(comp: str, scope: str, symptom: str, cond: dict) -> str:
    return f"tip:{comp}:{scope}:{symptom}:{cond.get('evidence', 'weak')}"


def _get(db: Session, comp: str, scope: str, symptom: str, cond: dict, when) -> HarnessExperience:
    key = key_of(comp, scope, symptom, cond)
    x = db.exec(select(HarnessExperience).where(HarnessExperience.key == key)).first()
    if x is None:
        x = HarnessExperience(key=key, track="tip", scope=scope, symptom=symptom, condition=cond, action={"component": comp},
                              text=f"在［{cond_text(symptom, cond)}］下{TEXT[comp]}", created_at=when, updated_at=when)
        db.add(x)
        db.flush()
    return x


def relevant(db: Session, scope: str, symptom: str, cond: dict) -> dict:
    keys = {c: key_of(c, scope, symptom, cond) for c in COMPONENTS}
    rows = {x.key: x for x in db.exec(select(HarnessExperience).where(col(HarnessExperience.key).in_(list(keys.values())))).all()}
    return {c: rows.get(k) for c, k in keys.items()}


def view(x: HarnessExperience | None) -> dict | None:
    if x is None:
        return None
    return {"id": x.id, "track": x.track, "text": x.text, "status": x.status, "p": round(p_success(x), 3),
            "n": int(trials(x)), "action": x.action, "retired_reason": x.retired_reason}


# ── 外层反思 ①：慢脑做完立刻反思（证据 = 同一批孪生探测题上的命中对比）──
def reflect_inner(db: Session, scope: str, symptom: str, cond: dict, inner: dict, when=None) -> list[HarnessExperience]:
    when = when or utcnow()
    hits, tops, out = inner.get("hits") or {}, inner.get("tops") or {}, []
    for comp in ("ttt",):
        if comp not in hits:
            continue
        if hits[comp] == hits["fast"] and tops.get(comp) == tops.get("fast"):
            continue  # 没有可观察的行为差异：证据不足就不新增条目（Harness² 的门控）
        x = _get(db, comp, scope, symptom, cond, when)
        _record(x, hits[comp] > hits["fast"], {"probe": {"comp": hits[comp], "fast": hits["fast"], "n": inner.get("probe_n")}}, when,
                blame=f"探测题上没有赢过快脑（{hits[comp]} vs {hits['fast']} / {inner.get('probe_n')}）")
        out.append(x)
    return out


# ── 外层反思 ②：关案后反思（证据 = 工程师确认的根因）──
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
    touched = []
    chosen = ((rc.inner or {}).get("arbiter") or {}).get("chosen")
    if rc.route == "fast":  # 快脑直接给的结论：对了加分，错了追责（之后同条件先转慢脑）
        x = _get(db, "fast", scope, sym, cond, when)
        _record(x, ok, {"receipt": rc.id, "case": case.code}, when, blame=f"{case.code} 快脑首位与现场确认不一致")
        touched.append(x)
    elif chosen == "ttt":  # 慢脑改判了：改判对不对，追责到做出改判的那个组件
        x = _get(db, chosen, scope, sym, cond, when)
        _record(x, ok, {"receipt": rc.id, "case": case.code}, when, blame=f"{case.code} 改判后的首位仍与现场确认不一致")
        touched.append(x)
    db.flush()
    return {"receipt": rc.id, "top_match": ok, "fast_match": rc.outcome["fast_match"], "experiences": [view(x) for x in touched]}


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


def library(db: Session) -> list[dict]:
    """跨产线汇总改进库：按（组件, 症状, 证据强弱）合并成败计数，供 RRSI 提议条件规则。"""
    agg: dict[tuple, dict] = {}
    for x in db.exec(select(HarnessExperience).where(HarnessExperience.track == "tip")).all():
        comp = (x.action or {}).get("component")
        if comp not in COMPONENTS:
            continue
        k = (comp, x.symptom, (x.condition or {}).get("evidence", "weak"))
        a = agg.setdefault(k, {"component": comp, "symptom": x.symptom, "evidence": k[2], "ok": 0.0, "fail": 0.0,
                               "retired": 0, "stable": 0, "ids": []})
        a["ok"] += x.alpha - 1
        a["fail"] += x.beta - 1
        a["retired"] += x.status == "retired"
        a["stable"] += x.status == "stable"
        a["ids"].append(x.id)
    for a in agg.values():
        n = a["ok"] + a["fail"]
        a["n"], a["p"] = int(n), round((a["ok"] + 1) / (n + 2), 3)
    return sorted(agg.values(), key=lambda a: -a["n"])
