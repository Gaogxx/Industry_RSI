"""可进化记忆（执行 agent 独占写入，且只在验证 agent 判定之后写入）。

三类内容（对应 RSIAgent 论文）：
  环境知识  CausalEdge（症状→根因份额）+ ToolHotspot（根因多发机台）
  可复用流程 MemoryItem(kind=procedure)：在什么证据条件下先查哪个根因
  失败教训  MemoryItem(kind=lesson)：在什么条件下某个整改被环境否定

产线验证（real）权重 1.0，孪生验证（sandbox）权重默认 0.5：孪生学到的东西有用，但必须让位于真实产线的确认。
同一套合并逻辑同时服务于“真实关案学习”和“孪生自主练习”，并且可以纯内存运行（用于冷启动基准/消融）。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from sqlmodel import Session, select

from ..fab.flow import CAUSES, SYMPTOMS, cause_by_code
from ..fab.kg import EdgeState, learn, prior_edges, refute, share, suspect_tool
from ..models import CausalEdge, EvidenceModel, MemoryItem, ToolHotspot, utcnow

SANDBOX_WEIGHT = 0.5


MIN_EXAMPLES = 20
GATE_TOLERANCE = 1.0  # 面板首位命中率允许的抖动（百分点，约 1 题）；对数损失必须不变差


@dataclass
class MemoryState:
    edges: dict[str, dict[str, EdgeState]]
    hotspots: dict[tuple[str, str], dict] = field(default_factory=dict)  # (cause, tool) -> {"real": n, "sandbox": n}
    items: dict[str, dict] = field(default_factory=dict)  # key -> 条目
    weights: dict[str, float] | None = None  # 学到的证据权重（None = 专家设定，全部为 1）
    examples: list[tuple[dict, str, str]] = field(default_factory=list)  # (证据, 症状, 已验证根因)：证据权重的训练样本
    era: int | None = None  # 面对哪个时段的孪生（None = 现在）：决定留出测试集与验收门面板

    def refit(self, gate: bool = True) -> dict:
        """用全部已验证样本重新拟合证据权重；候选权重必须通过验收门（封存面板上不退步）才会被采纳。"""
        from .calibration import feature_rows, fit, gate_check

        hw = self.hotspot_weights()
        res = fit(feature_rows(self.examples, self.edges, hw), start=self.weights)
        res["accepted"] = False
        if res["n"] < MIN_EXAMPLES:  # 样本太少时保持现状
            res["reason"] = f"已验证样本 {res['n']} 个，不足 {MIN_EXAMPLES} 个"
            return res
        if gate:
            before, after = gate_check(self, self.weights), gate_check(self, res["weights"])
            res["gate"] = {"before": before, "after": after}
            if after["logloss"] > before["logloss"] or after["hit1"] < before["hit1"] - GATE_TOLERANCE:
                res["reason"] = (f"验收门拒绝：面板首位命中率 {before['hit1']} → {after['hit1']}，"
                                 f"对数损失 {before['logloss']} → {after['logloss']}")
                return res
        self.weights, res["accepted"] = res["weights"], True
        return res

    @classmethod
    def textbook(cls) -> "MemoryState":
        return cls(edges=prior_edges())

    def clone(self) -> "MemoryState":
        """作答用的快照：训练样本只读、量大，不做深拷贝。"""
        examples, self.examples = self.examples, []
        try:
            out = copy.deepcopy(self)
        finally:
            self.examples = examples
        out.examples = list(examples)
        return out

    def hotspot_weights(self, sandbox_weight: float = SANDBOX_WEIGHT) -> dict[tuple[str, str], float]:
        return {k: v.get("real", 0) + sandbox_weight * v.get("sandbox", 0) for k, v in self.hotspots.items()}

    def active_items(self, symptom: str) -> list[dict]:
        rows = [i for i in self.items.values() if i["symptom"] == symptom and i["status"] == "active"]
        return sorted(rows, key=lambda i: -(i["passes"] - i["fails"]))

    def size(self) -> int:
        return sum(1 for i in self.items.values() if i["status"] == "active")


# ── 条件抽取：记忆必须带上“在什么条件下” ─────────────────────
def condition_of(evidence: dict, cause_code: str | None = None) -> dict:
    sus = [s for s in evidence.get("suspects", []) if s.get("score", 0) >= 1.3]
    cp = evidence.get("change_point") or {}
    factor = "none"
    if sus:
        factor = "batch" if "#" in sus[0]["factor"] else "tool"
    return {"evidence": "strong" if sus else "weak", "after_pm": bool(cp.get("after_pm")), "factor": factor}


def _cond_text(c: dict) -> str:
    return "；".join([
        "共性分析显著" if c["evidence"] == "strong" else "共性分析不显著（证据模糊）",
        {"tool": "富集在机台", "batch": "富集在物料批次", "none": "无富集"}[c["factor"]],
        "PM 后突变" if c["after_pm"] else "与 PM 无关",
    ])


def _touch_item(mem: MemoryState, key: str, *, kind: str, symptom: str, cause_code: str, cond: dict, text: str,
                source: str, ref: int | None, passed: bool) -> tuple[dict, bool]:
    item = mem.items.get(key)
    created = item is None
    if created:
        item = mem.items[key] = {"id": None, "kind": kind, "key": key, "symptom": symptom, "cause_code": cause_code,
                                 "condition": cond, "text": text, "passes": 0, "fails": 0, "source": source,
                                 "source_cases": [], "source_tasks": [], "status": "active"}
    if passed:
        item["passes"] += 1
    else:
        item["fails"] += 1
    if item["source"] != source:
        item["source"] = "both"
    if ref is not None:
        item["source_cases" if source == "real" else "source_tasks"].append(ref)
    if item["fails"] >= item["passes"] + 2 and item["status"] == "active":
        item["status"] = "retired"
    elif item["status"] == "retired" and item["passes"] > item["fails"] + 1:
        item["status"] = "active"
    item["_dirty"] = True
    return item, created


def consolidate_pass(mem: MemoryState, symptom: str, cause_code: str, ranked: list[str], evidence: dict, *,
                     source: str = "real", ref: int | None = None, weight: float | None = None,
                     cause_name: str = "", six_m: str = "") -> list[dict]:
    """验证通过：知识（份额）+ 机台热点 + 可复用流程。"""
    w = weight if weight is not None else (1.0 if source == "real" else SANDBOX_WEIGHT)
    found = cause_by_code(cause_code)
    events = [dict(e, source=source) for e in learn(mem.edges, symptom, cause_code, ranked, confirmed_name=cause_name or cause_code,
                                                     confirmed_step=found[1].step if found else "", six_m=six_m,
                                                     weight=w, source=source)]
    edge = mem.edges[symptom][cause_code]
    tool = suspect_tool(evidence, edge)
    if tool:
        h = mem.hotspots.setdefault((cause_code, tool), {"real": 0.0, "sandbox": 0.0})
        h[source if source in h else "real"] += 1
        h["_dirty"] = True
        events.append({"kind": "hotspot", "source": source, "symptom": symptom, "cause_code": cause_code, "before": 0, "after": 0,
                       "detail": f"机台热点：{tool} ↔ {edge.name[:16]}（产线 {h['real']:.0f} 次 · 孪生 {h['sandbox']:.0f} 次）"})
    cond = condition_of(evidence)
    name = found[1].name if found else (cause_name or cause_code)
    text = f"条件［{SYMPTOMS[symptom].name if symptom in SYMPTOMS else symptom}；{_cond_text(cond)}］→ 优先排查「{name}」"
    if found and found[1].capa[2]:
        text += f"；验证有效的预防措施：{found[1].capa[2]}"
    item, created = _touch_item(mem, f"proc:{symptom}:{cause_code}:{cond['evidence']}", kind="procedure", symptom=symptom,
                                cause_code=cause_code, cond=cond, text=text, source=source, ref=ref, passed=True)
    events.append({"kind": "memory_add" if created else "memory_pass", "source": source, "symptom": symptom, "cause_code": cause_code,
                   "before": item["passes"] - 1, "after": item["passes"],
                   "detail": ("新增流程记忆：" if created else f"流程记忆再次验证通过（{item['passes']}/{item['passes'] + item['fails']}）：") + text[:60]})
    return events


def consolidate_fail(mem: MemoryState, symptom: str, cause_code: str, evidence: dict, *, source: str = "real",
                     ref: int | None = None, weight: float | None = None) -> list[dict]:
    """验证失败：知识扣减 + 失败教训 + 同条件流程记忆记一次失败（失败多于成功会自动停用）。"""
    w = weight if weight is not None else (1.0 if source == "real" else SANDBOX_WEIGHT)
    events = []
    r = refute(mem.edges, symptom, cause_code, weight=w)
    if r:
        events.append(dict(r, source=source))
    cond = condition_of(evidence)
    found = cause_by_code(cause_code)
    name = found[1].name if found else cause_code
    text = f"条件［{SYMPTOMS[symptom].name if symptom in SYMPTOMS else symptom}；{_cond_text(cond)}］下按「{name}」整改未能恢复指标"
    item, created = _touch_item(mem, f"lesson:{symptom}:{cause_code}:{cond['evidence']}", kind="lesson", symptom=symptom,
                                cause_code=cause_code, cond=cond, text=text, source=source, ref=ref, passed=False)
    events.append({"kind": "memory_fail", "source": source, "symptom": symptom, "cause_code": cause_code, "before": item["fails"] - 1,
                   "after": item["fails"], "detail": ("新增失败教训：" if created else "失败教训再次出现：") + text[:60]})
    proc = mem.items.get(f"proc:{symptom}:{cause_code}:{cond['evidence']}")
    if proc and proc["status"] == "active":
        _touch_item(mem, proc["key"], kind="procedure", symptom=symptom, cause_code=cause_code, cond=cond, text=proc["text"],
                    source=source, ref=ref, passed=False)
        if proc["status"] == "retired":
            events.append({"kind": "memory_retire", "source": source, "symptom": symptom, "cause_code": cause_code, "before": 1,
                           "after": 0, "detail": f"流程记忆被多次否定，已自动停用：{proc['text'][:50]}"})
    return events


def forget(mem: MemoryState, symptom: str, label: str, keep: float | None = None) -> dict | None:
    """产线变更后的部分遗忘：受影响症状下学到的份额只保留 keep（向教科书先验收缩），让变更后的新证据更快生效。"""
    table = mem.edges.get(symptom)
    if not table:
        return None
    from ..fab.kg import FORGET_KEEP, forget_table

    keep = FORGET_KEEP if keep is None else keep
    top = max(table.values(), key=lambda e: e.alpha)
    before = share(table, top.cause_code)
    forget_table(table, keep)
    return {"kind": "drift", "source": "real", "symptom": symptom, "cause_code": top.cause_code, "before": round(before, 4),
            "after": round(share(table, top.cause_code), 4),
            "detail": f"产线变更：{label}。该症状的旧经验降权到 {keep:.0%}，变更后的新证据会更快生效"}


# ── 数据库同步 ──────────────────────────────────────────
def ensure_textbook(db: Session) -> None:
    if db.exec(select(CausalEdge)).first():
        return
    save_memory(db, MemoryState.textbook())


def load_memory(db: Session) -> MemoryState:
    meta = {c.code: c for causes in CAUSES.values() for c in causes}
    edges: dict[str, dict[str, EdgeState]] = {}
    for r in db.exec(select(CausalEdge)).all():
        m = meta.get(r.cause_code)
        edges.setdefault(r.symptom, {})[r.cause_code] = EdgeState(
            r.symptom, r.cause_code, r.name, r.step, r.six_m, r.alpha, r.beta, r.alpha0, r.beta0,
            m.fdc if m else (), m.pm_related if m else False, m.tool_specific if m else False,
            r.confirmed, r.refuted, r.learned, r.confirmed_sandbox)
    if not edges:
        edges = prior_edges()
    hotspots = {(h.cause_code, h.tool): {"real": h.real, "sandbox": h.sandbox} for h in db.exec(select(ToolHotspot)).all()}
    items = {}
    for i in db.exec(select(MemoryItem)).all():
        items[i.key] = {"id": i.id, "kind": i.kind, "key": i.key, "symptom": i.symptom, "cause_code": i.cause_code,
                        "condition": i.condition, "text": i.text, "passes": i.passes, "fails": i.fails, "source": i.source,
                        "source_cases": list(i.source_cases), "source_tasks": list(i.source_tasks), "status": i.status}
    em = db.exec(select(EvidenceModel).order_by(EvidenceModel.id.desc())).first()
    return MemoryState(edges=edges, hotspots=hotspots, items=items, weights=em.weights if em else None)


MAX_TWIN_EXAMPLES = 1500  # 只用最近的孪生样本（滑动窗口），产线样本全部保留


def training_examples(db: Session) -> list[tuple[dict, str, str]]:
    """全部已验证样本：孪生中验证通过的练习题（最近 1500 道）+ 产线上工程师确认的案例。"""
    from ..models import Case, RSITask

    rows = db.exec(select(RSITask).where(RSITask.verdict == "PASS", RSITask.symptom != "")
                   .order_by(RSITask.id.desc()).limit(MAX_TWIN_EXAMPLES)).all()
    out = [(t.evidence, t.symptom, t.attempts[-1]["cause_code"]) for t in rows if t.evidence and t.attempts]
    out += [(c.evidence, c.symptom, c.cause_code) for c in db.exec(select(Case).where(Case.status == "closed", Case.cause_code != "")).all()
            if c.evidence]
    return out


def refit_and_store(db: Session, mem: MemoryState, source: str = "rsi", when=None) -> dict:
    """从数据库收集全部已验证样本（孪生 PASS 练习 + 产线确认案例）重拟合证据权重；通过验收门才保存新版本。"""
    mem.examples = training_examples(db)
    res = mem.refit()
    if res["accepted"]:
        row = EvidenceModel(weights=res["weights"], n=res["n"], avg_loglik=res["avg_loglik"], source=source,
                            gate=res.get("gate", {}))
        if when:
            row.created_at = when
        db.add(row)
        db.flush()
    return res


def save_memory(db: Session, mem: MemoryState, when=None) -> None:
    now = when or utcnow()
    rows = {(r.symptom, r.cause_code): r for r in db.exec(select(CausalEdge)).all()}
    for sym, table in mem.edges.items():
        for e in table.values():
            r = rows.get((sym, e.cause_code)) or CausalEdge(symptom=sym, cause_code=e.cause_code, name=e.name, step=e.step,
                                                            six_m=e.six_m, alpha=e.alpha, beta=e.beta, alpha0=e.alpha0,
                                                            beta0=e.beta0, learned=e.learned)
            new = (e.alpha, e.confirmed, e.refuted, e.confirmed_sandbox)
            if r.id is None or (r.alpha, r.confirmed, r.refuted, r.confirmed_sandbox) != new:
                r.alpha, r.confirmed, r.refuted, r.confirmed_sandbox = new
                r.updated_at = now
                db.add(r)
    hs = {(h.cause_code, h.tool): h for h in db.exec(select(ToolHotspot)).all()}
    for (cause, tool), v in mem.hotspots.items():
        if v.pop("_dirty", False) or (cause, tool) not in hs:
            h = hs.get((cause, tool)) or ToolHotspot(cause_code=cause, tool=tool)
            h.real, h.sandbox, h.updated_at = v.get("real", 0), v.get("sandbox", 0), now
            db.add(h)
    by_key = {i.key: i for i in db.exec(select(MemoryItem)).all()}
    for key, it in mem.items.items():
        if not it.pop("_dirty", False) and key in by_key:
            continue
        row = by_key.get(key) or MemoryItem(key=key, symptom=it["symptom"], text=it["text"], created_at=now)
        by_key[key] = row
        for f in ("kind", "cause_code", "condition", "text", "passes", "fails", "source", "source_cases", "source_tasks", "status"):
            setattr(row, f, it[f])
        row.updated_at = now
        db.add(row)
    db.flush()
    for key, it in mem.items.items():
        if it.get("id") is None and key in by_key:
            it["id"] = by_key[key].id


def item_view(mem: MemoryState) -> list[dict]:
    return [{k: v for k, v in i.items() if not k.startswith("_")} for i in mem.items.values()]


assert share  # noqa: S101
