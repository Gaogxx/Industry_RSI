"""贝叶斯因果知识图谱：每个症状下的候选根因服从 Dirichlet(α₁…αₖ) 分布（多项-狄利克雷共轭）。

排序（推理）：P(根因 | 证据) ∝ P(根因 | 症状) × 共性分析似然 × FDC 似然 × PM 时序似然
             其中 P(根因 | 症状) = α_c / Σα，是“本厂经验”的部分，会随案例学习
学习（更新）：工程师确认根因并关案时 ——
  · 确认的根因  α_c += 1：份额上升，其余根因份额随之下降（排在它前面的误导假设记录为 false_lead 事件）
  · CAPA 验证无效的根因  α_c −= 0.5（不低于初始值的 25%）：现场否定的负证据
  · 知识图谱里没有的新根因 → 新增一个根因节点（图谱自己长出来）
初始 α 由“教科书先验 × KAPPA”给出，KAPPA 越小，本厂数据越快压过教科书。

纯函数实现，既能作用于数据库中的边，也能在内存里做“预序评估”(prequential) 回放出学习曲线。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .flow import CAUSES, STEPS

KAPPA = 4.0  # 先验强度：相当于 4 个“虚拟历史案例”


@dataclass
class EdgeState:
    symptom: str
    cause_code: str
    name: str
    step: str
    six_m: str
    alpha: float
    beta: float
    alpha0: float
    beta0: float
    fdc: tuple[str, ...] = ()
    pm_related: bool = False
    tool_specific: bool = True
    confirmed: int = 0
    refuted: int = 0
    learned: bool = False  # 由案例学习新增的边
    confirmed_sandbox: int = 0  # 在数字孪生中经验证 agent 判定通过的次数



def share(table: dict[str, "EdgeState"], code: str, initial: bool = False) -> float:
    total = sum((e.alpha0 if initial else e.alpha) for e in table.values()) or 1.0
    e = table[code]
    return (e.alpha0 if initial else e.alpha) / total


def prior_edges() -> dict[str, dict[str, EdgeState]]:
    out: dict[str, dict[str, EdgeState]] = {}
    for sym, causes in CAUSES.items():
        out[sym] = {}
        for c in causes:
            a0 = KAPPA * c.prior
            out[sym][c.code] = EdgeState(sym, c.code, c.name, c.step, c.six_m, a0, 0.0, a0, 0.0,
                                         c.fdc, c.pm_related, c.tool_specific)
    return out


@dataclass
class Hypothesis:
    cause_code: str
    name: str
    step: str
    six_m: str
    prior: float  # 当前（已学习的）先验均值
    textbook_prior: float
    posterior: float = 0.0
    factors: list[dict] = field(default_factory=list)
    confirmed: int = 0
    refuted: int = 0
    learned: bool = False

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


FACTOR_NAMES = ("先验知识", "共性分析", "FDC", "PM 时序", "机台热点")


def rank(evidence: dict, symptom: str, edges: dict[str, dict[str, EdgeState]],
         hotspots: dict[tuple[str, str], float] | None = None,
         weights: dict[str, float] | None = None) -> list[Hypothesis]:
    """hotspots: {(cause_code, tool): 权重}，来自产线与孪生中验证过的“根因多发于哪台机台”。

    weights: 各类证据的可信度指数（学习得到）。后验 ∝ Π 因子^权重；None 表示全部为 1（专家手工模型）。
    """
    table = edges.get(symptom, {})
    hotspots = hotspots or {}
    suspects = [s for s in evidence.get("suspects", []) if s.get("score", 0) >= 1.3]
    strong_elsewhere = max((s["score"] for s in suspects), default=0)
    cp = evidence.get("change_point") or {}
    fdc = evidence.get("fdc", [])
    hyps = []
    for e in table.values():
        prior, textbook = share(table, e.cause_code), share(table, e.cause_code, initial=True)
        h = Hypothesis(e.cause_code, e.name, e.step, e.six_m, prior, textbook, confirmed=e.confirmed,
                       refuted=e.refuted, learned=e.learned)
        lik = 1.0
        h.factors.append({"name": "先验知识", "x": round(prior, 3),
                          "detail": f"教科书 {textbook:.0%} → 已学习 {prior:.0%}（产线确认 {e.confirmed} · 孪生确认 {e.confirmed_sandbox}"
                                    f" · 否定 {e.refuted}）"})
        same_step = [s for s in suspects if s["step"] == e.step]
        if same_step:
            s = same_step[0]
            is_batch = "#" in s["factor"]
            match = (is_batch and not e.tool_specific) or (not is_batch and e.tool_specific)
            f = 1 + s["score"] / (2 if match else 5)
            lik *= f
            h.factors.append({"name": "共性分析", "x": round(f, 2),
                              "detail": f"{STEPS[e.step].name if e.step in STEPS else e.step} · {s['value']} "
                                        f"失效率 {s['rate_with']:.0%} vs {s['rate_without']:.0%}，p_adj={s['p_adj']}"})
            if not is_batch and e.tool_specific and hotspots:
                hf = _hotspot_factor(table, e, s["value"], hotspots)
                if abs(hf - 1) > 0.02:
                    lik *= hf
                    h.factors.append({"name": "机台热点", "x": round(hf, 2),
                                      "detail": f"已验证记忆：{s['value']} 上该根因的相对频次 ×{hf:.2f}"})
        elif strong_elsewhere >= 3:
            lik *= 0.5
            h.factors.append({"name": "共性分析", "x": 0.5, "detail": "显著机台/批次不在该工序"})
        hit = [x for x in fdc if x["param"] in e.fdc and abs(x["z"]) >= 2]
        if hit:
            f = 1 + abs(hit[0]["z"]) / 2
            lik *= f
            h.factors.append({"name": "FDC", "x": round(f, 2), "detail": f"{hit[0]['tool']} {hit[0]['param']} 偏移 {hit[0]['z']:+.1f}σ"})
        if cp.get("after_pm"):
            f = 2.0 if e.pm_related else 0.8
            lik *= f
            h.factors.append({"name": "PM 时序", "x": f, "detail": f"{cp['value']} 在 PM 后开始恶化"
                              + ("，符合该根因机理" if e.pm_related else "，与该根因机理无关")})
        h.posterior = prior * lik
        hyps.append(h)
    if weights:
        logs = []
        for h in hyps:
            ls = 0.0
            for f in h.factors:
                f["w"] = round(weights.get(f["name"], 1.0), 2)
                ls += weights.get(f["name"], 1.0) * math.log(max(f["x"], 1e-9))
            logs.append(ls)
        m = max(logs) if logs else 0.0
        for h, ls in zip(hyps, logs):
            h.posterior = math.exp(ls - m)
    z = sum(h.posterior for h in hyps) or 1.0
    for h in hyps:
        h.posterior /= z
    return sorted(hyps, key=lambda h: -h.posterior)


def _hotspot_factor(table: dict[str, EdgeState], e: EdgeState, tool: str, hotspots: dict[tuple[str, str], float]) -> float:
    """P(机台 | 根因) 的相对似然：同一工序的候选根因之间比较，在该机台上出现过的次数（Laplace 平滑）。"""
    peers = [x for x in table.values() if x.step == e.step and x.tool_specific]
    if len(peers) < 2:
        return 1.0
    counts = {x.cause_code: hotspots.get((x.cause_code, tool), 0.0) for x in peers}
    total = sum(counts.values())
    if total <= 0:
        return 1.0
    k = len(peers)
    return k * (counts[e.cause_code] + 0.5) / (total + 0.5 * k)


def learn(edges: dict[str, dict[str, EdgeState]], symptom: str, confirmed: str, ranked_codes: list[str],
          confirmed_name: str = "", confirmed_step: str = "", six_m: str = "", weight: float = 1.0,
          source: str = "real") -> list[dict]:
    """用一个已验证的案例更新图谱，返回学习事件（before/after 为该根因在此症状下的份额）。"""
    table = edges.setdefault(symptom, {})
    events = []
    if confirmed not in table:
        table[confirmed] = EdgeState(symptom, confirmed, confirmed_name or confirmed, confirmed_step or "?", six_m or "method",
                                     0.0, 0.0, 0.0, 0.0, learned=True)
        events.append({"kind": "edge_added", "symptom": symptom, "cause_code": confirmed, "before": 0.0, "after": 0.0,
                       "detail": f"知识图谱新增根因：{table[confirmed].name}"})
    before = {c: share(table, c) for c in table}
    table[confirmed].alpha += weight
    if source == "sandbox":
        table[confirmed].confirmed_sandbox += 1
    else:
        table[confirmed].confirmed += 1
    after = {c: share(table, c) for c in table}
    e = table[confirmed]
    events.append({"kind": "confirm", "symptom": symptom, "cause_code": confirmed, "before": round(before[confirmed], 4),
                   "after": round(after[confirmed], 4), "detail": f"确认根因：{e.name}"})
    for code in ranked_codes:
        if code == confirmed:
            break
        if code in table:
            events.append({"kind": "false_lead", "symptom": symptom, "cause_code": code, "before": round(before[code], 4),
                           "after": round(after[code], 4), "detail": f"排在真因之前的误导假设：{table[code].name}"})
    if events and events[0]["kind"] == "edge_added":
        events[0]["after"] = round(after[confirmed], 4)
    return events


def refute(edges: dict[str, dict[str, EdgeState]], symptom: str, code: str, weight: float = 1.0) -> dict | None:
    table = edges.get(symptom, {})
    e = table.get(code)
    if not e:
        return None
    b = share(table, code)
    e.alpha = max(e.alpha - 0.5 * weight, 0.25 * e.alpha0) if e.alpha0 else max(e.alpha - 0.5 * weight, 0.1)
    e.refuted += 1
    return {"kind": "refute", "symptom": symptom, "cause_code": code, "before": round(b, 4),
            "after": round(share(table, code), 4), "detail": f"CAPA 验证无效，否定：{e.name}"}


def prequential(cases: list[dict]) -> dict:
    """预序评估（先预测、后学习）：按关案时间回放，比较“持续学习”与“冻结在教科书先验”的命中率。

    cases: [{"code", "time", "symptom", "cause_code", "evidence"}]，按时间排序。
    """
    learned, frozen = prior_edges(), prior_edges()
    hotspots: dict[tuple[str, str], float] = {}
    points, hits_l, hits_f, rr_l, rr_f = [], [], [], [], []
    for i, c in enumerate(cases, 1):
        rl = [h.cause_code for h in rank(c["evidence"], c["symptom"], learned, hotspots)]
        rf = [h.cause_code for h in rank(c["evidence"], c["symptom"], frozen)]
        truth = c["cause_code"]
        pos_l = rl.index(truth) + 1 if truth in rl else 99
        pos_f = rf.index(truth) + 1 if truth in rf else 99
        hits_l.append(pos_l == 1)
        hits_f.append(pos_f == 1)
        rr_l.append(1 / pos_l)
        rr_f.append(1 / pos_f)
        w = 10
        points.append({
            "i": i, "case": c["code"], "time": c["time"], "symptom": c["symptom"],
            "hit_learning": pos_l == 1, "hit_frozen": pos_f == 1,
            "cum_learning": round(sum(hits_l) / i, 3), "cum_frozen": round(sum(hits_f) / i, 3),
            "win_learning": round(sum(hits_l[-w:]) / len(hits_l[-w:]), 3),
            "win_frozen": round(sum(hits_f[-w:]) / len(hits_f[-w:]), 3),
        })
        learn(learned, c["symptom"], truth, rl)
        tool = suspect_tool(c["evidence"], learned.get(c["symptom"], {}).get(truth))
        if tool:
            hotspots[(truth, tool)] = hotspots.get((truth, tool), 0.0) + 1.0
    n = len(cases)
    half = n // 2
    return {
        "points": points,
        "summary": {
            "cases": n,
            "hit1_learning": round(sum(hits_l) / n, 3) if n else None,
            "hit1_frozen": round(sum(hits_f) / n, 3) if n else None,
            "mrr_learning": round(sum(rr_l) / n, 3) if n else None,
            "mrr_frozen": round(sum(rr_f) / n, 3) if n else None,
            "hit1_learning_recent": round(sum(hits_l[half:]) / max(1, n - half), 3) if n else None,
            "hit1_frozen_recent": round(sum(hits_f[half:]) / max(1, n - half), 3) if n else None,
        },
    }


def suspect_tool(evidence: dict, edge: "EdgeState | None") -> str | None:
    """证据中与该根因同工序、显著的机台（用于学习机台热点）。"""
    if not edge or not edge.tool_specific:
        return None
    for s in evidence.get("suspects", []):
        if s["step"] == edge.step and s.get("score", 0) >= 1.3 and "#" not in s["factor"]:
            return s["value"]
    return None


def entropy(hyps: list[Hypothesis]) -> float:
    """假设分布的熵（越低越笃定），用于展示“学得越多越确定”。"""
    return round(-sum(h.posterior * math.log2(h.posterior) for h in hyps if h.posterior > 0), 3)
