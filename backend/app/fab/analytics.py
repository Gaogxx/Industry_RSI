"""良率工程的确定性分析方法（与 LLM 无关、可审计、可复现）：

1. 共性分析（Commonality）：失效批次在哪台机台/腔体/物料批次上显著富集 —— Fisher 精确检验 + Bonferroni 校正；
2. 变点检测：可疑机台从哪天开始变坏，是否紧跟 PM 事件；
3. 晶圆图空间特征分类：边缘环 / 中心 / 划痕 / 簇状 / 随机；
4. FDC 参数偏移：变点前后传感器均值偏移多少个 σ。

输出“证据摘要”，供贝叶斯因果图谱做根因排序。
"""

from __future__ import annotations

import math
from statistics import mean, pstdev

from .flow import STEPS


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_greater(a: int, b: int, c: int, d: int) -> float:
    """单侧 Fisher 精确检验：该机台上的失效比例是否显著高于其它机台。"""
    n_total, fails, n_tool = a + b + c + d, a + c, a + b
    denom = _log_comb(n_total, n_tool)
    p = sum(math.exp(_log_comb(fails, x) + _log_comb(n_total - fails, n_tool - x) - denom)
            for x in range(a, min(fails, n_tool) + 1))
    return min(1.0, p)


def commonality(lots: list[dict], top: int = 6) -> list[dict]:
    factors = sorted({k for l in lots for k in l["steps"]})
    tests = []
    total_fail = sum(1 for l in lots if l["fail"])
    for f in factors:
        values = sorted({l["steps"][f] for l in lots if f in l["steps"]})
        for v in values:
            a = sum(1 for l in lots if l["steps"].get(f) == v and l["fail"])
            n = sum(1 for l in lots if l["steps"].get(f) == v)
            b, c = n - a, total_fail - a
            d = len(lots) - n - c
            if n < 3:
                continue
            p = fisher_greater(a, b, c, d)
            rate_with = a / n
            rate_without = c / max(1, c + d)
            tests.append({"factor": f, "step": f.split("#")[0], "value": v, "n": n, "fails": a,
                          "rate_with": round(rate_with, 3), "rate_without": round(rate_without, 3),
                          "lift": round(rate_with / max(rate_without, 0.01), 2), "p": p})
    m = max(1, len(tests))
    for t in tests:
        t["p_adj"] = min(1.0, t["p"] * m)
        t["score"] = round(min(12.0, -math.log10(max(t["p_adj"], 1e-12))), 2)
        t["p"] = float(f"{t['p']:.3g}")
        t["p_adj"] = float(f"{t['p_adj']:.3g}")
    tests.sort(key=lambda t: (-t["score"], -t["lift"]))
    return tests[:top]


def change_point(lots: list[dict], factor: str, value: str, events: list[dict]) -> dict | None:
    seq = sorted((l for l in lots if l["steps"].get(factor) == value), key=lambda l: l["time"])
    if len(seq) < 8:
        return None
    best = None
    for i in range(3, len(seq) - 3):
        before, after = seq[:i], seq[i:]
        rb = sum(l["fail"] for l in before) / len(before)
        ra = sum(l["fail"] for l in after) / len(after)
        gap = ra - rb
        if best is None or gap > best[0]:
            best = (gap, i, rb, ra)
    if not best or best[0] < 0.2:
        return None
    t = seq[best[1]]["time"]
    pm = [e for e in events if e["tool"] == value and e["type"] == "PM" and e["time"] <= t]
    last_pm = pm[-1] if pm else None
    after_pm = bool(last_pm and _hours(last_pm["time"], t) <= 72)
    return {"factor": factor, "value": value, "since": t, "rate_before": round(best[2], 3),
            "rate_after": round(best[3], 3), "after_pm": after_pm, "pm_time": last_pm["time"] if last_pm else None}


def _hours(t1: str, t2: str) -> float:
    from datetime import datetime

    return abs((datetime.fromisoformat(t2) - datetime.fromisoformat(t1)).total_seconds()) / 3600


def wafer_signature(rows: list[str]) -> dict:
    n = len(rows)
    c, R = (n - 1) / 2, n / 2
    dies, fails = [], []
    for i, row in enumerate(rows):
        for j, ch in enumerate(row):
            if ch == ".":
                continue
            r = math.hypot(j - c, i - c) / R
            dies.append((i, j, r))
            if ch == "1":
                fails.append((i, j, r))
    if not dies:
        return {"pattern": "unknown", "confidence": 0.0}
    rate = len(fails) / len(dies)
    fail_set = {(i, j) for i, j, _ in fails}
    comps = _components(fail_set)
    big = [c for c in comps if len(c) >= 5]  # 连通失效区域（随机背景失效很少能连成 5 颗以上）
    if rate < 0.04 and not big:
        return {"pattern": "good", "confidence": 0.9, "fail_rate": round(rate, 3)}

    def zone_rate(lo, hi):
        z = [d for d in dies if lo <= d[2] < hi]
        return sum(1 for d in z if rows[d[0]][d[1]] == "1") / max(1, len(z))

    edge, center = zone_rate(0.8, 2), zone_rate(0, 0.36)
    largest = max(comps, key=len) if comps else []
    linear = _linearity(largest) if len(largest) >= 6 else 0.0
    clustered = sum(len(c) for c in big) / max(1, len(fails))
    feats = {"fail_rate": round(rate, 3), "edge_rate": round(edge, 3), "center_rate": round(center, 3),
             "clustered": round(clustered, 2), "linearity": round(linear, 2), "regions": len(big)}
    if edge > 2.0 * rate and edge > 0.3:
        return {"pattern": "edge", "confidence": round(min(0.95, edge / (edge + rate)), 2), **feats}
    if center > 2.5 * rate and center > 0.3:
        return {"pattern": "center", "confidence": round(min(0.95, center / (center + rate)), 2), **feats}
    if linear > 0.88:
        return {"pattern": "scratch", "confidence": round(min(0.95, linear), 2), **feats}
    if big and clustered > 0.3:
        return {"pattern": "cluster", "confidence": round(min(0.95, 0.4 + clustered), 2), **feats}
    return {"pattern": "random", "confidence": 0.5, **feats}


def _components(cells: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    seen, comps = set(), []
    for start in cells:
        if start in seen:
            continue
        stack, comp = [start], []
        seen.add(start)
        while stack:
            i, j = stack.pop()
            comp.append((i, j))
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    nb = (i + di, j + dj)
                    if nb in cells and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        comps.append(comp)
    return comps


def _linearity(cells: list[tuple[int, int]]) -> float:
    xs, ys = [j for _, j in cells], [i for i, _ in cells]
    mx, my = mean(xs), mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs) / len(xs)
    syy = sum((y - my) ** 2 for y in ys) / len(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / len(xs)
    tr, det = sxx + syy, sxx * syy - sxy ** 2
    l1 = tr / 2 + math.sqrt(max(tr * tr / 4 - det, 0))
    return l1 / tr if tr else 0.0


PATTERN_TO_SYMPTOM = {"edge": "WM_EDGE", "center": "WM_CENTER", "scratch": "WM_SCRATCH", "cluster": "WM_CLUSTER"}


def fdc_shift(fdc: dict[str, list[dict]], tool: str, since: str | None) -> list[dict]:
    rows = fdc.get(tool) or []
    if len(rows) < 6:
        return []
    params = [k for k in rows[0] if k not in ("lot", "time")]
    out = []
    for p in params:
        if since:
            before = [r[p] for r in rows if r["time"] < since]
            after = [r[p] for r in rows if r["time"] >= since]
        else:
            half = len(rows) // 2
            before, after = [r[p] for r in rows[:half]], [r[p] for r in rows[half:]]
        if len(before) < 3 or len(after) < 3:
            continue
        sd = pstdev(before) or 1e-9
        out.append({"tool": tool, "param": p, "z": round((mean(after) - mean(before)) / sd, 2),
                    "before": round(mean(before), 3), "after": round(mean(after), 3)})
    return sorted(out, key=lambda x: -abs(x["z"]))


def analyze_dataset(ds: dict) -> dict:
    """原始数据 → 证据摘要。"""
    lots = ds["lots"]
    suspects = commonality(lots)
    top = suspects[0] if suspects else None
    cp = change_point(lots, top["factor"], top["value"], ds.get("events", [])) if top and top["score"] >= 1.3 else None
    fdc = fdc_shift(ds.get("fdc", {}), top["value"], cp["since"] if cp else None) if top and "#" not in top["factor"] else []
    sigs = [dict(wafer_signature(w["map"]), wafer=w["wafer"]) for w in ds.get("wafers", []) if w.get("fail")]
    votes: dict[str, float] = {}
    for s in sigs:
        votes[s["pattern"]] = votes.get(s["pattern"], 0) + s["confidence"]
    pattern = max(votes, key=votes.get) if votes else None
    fails = [l for l in lots if l["fail"]]
    summary = {"lots": len(lots), "fail_lots": len(fails)}
    if lots and "yield" in lots[0]:
        summary["mean_yield"] = round(mean(l["yield"] for l in lots), 2)
        summary["fail_mean_yield"] = round(mean(l["yield"] for l in fails), 2) if fails else None
    return {
        "summary": summary,
        "suspects": suspects,
        "change_point": cp,
        "fdc": [f for f in fdc if abs(f["z"]) >= 1.5][:4],
        "wafer_pattern": pattern,
        "wafer_pattern_conf": round(votes[pattern] / max(1, len(sigs)), 2) if pattern else None,
        "wafer_signatures": sigs,
        "step_names": {k: STEPS[k].name for k in {s["step"] for s in suspects} if k in STEPS},
    }
