"""良率工程的确定性分析方法（与 LLM 无关、可审计、可复现）：

1. 共性分析（Commonality）：失效批次在哪台机台/腔体/物料批次上显著富集 —— Fisher 精确检验 + Bonferroni 校正；
2. 变点检测：可疑机台从哪天开始变坏，是否紧跟 PM 事件；
3. 晶圆图空间特征分类：边缘环 / 中心 / 划痕 / 簇状 / 随机；
4. FDC 参数偏移：变点前后传感器均值偏移多少个 σ；
5. 原始信号（v0.7，给快脑小模型用）：用连续的良率 / 失效率而不是“是否失效”的二值，先找整体变点，
   再在变点之后的时间窗里按工序逐台机台做 Welch t 检验；同时看该机台变点之前是否本来就差、变点前是否刚做过 PM。
   只用每道工序都有的机台因子（物料批次字段在孪生里只给材料类根因生成，用它会泄露答案）。

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


def _welch(a: list[float], b: list[float]) -> float:
    if len(a) < 3 or len(b) < 3:
        return 0.0
    ma, mb = mean(a), mean(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    se = math.sqrt(va / len(a) + vb / len(b)) or 1e-9
    return (ma - mb) / se


def raw_signals(ds: dict) -> dict:
    """原始信号摘要：整体变点 + 变点后时间窗内逐工序、逐机台的连续指标检验（坏的方向为正）。"""
    lots = sorted(ds.get("lots") or [], key=lambda l: l["time"])
    if len(lots) < 24:
        return {}
    key = "yield" if "yield" in lots[0] else "fail_rate"
    bad = [(-l[key] if key == "yield" else l[key]) for l in lots]
    n = len(bad)
    pre = [0.0]
    for v in bad:
        pre.append(pre[-1] + v)
    sd = pstdev(bad) or 1e-9
    best_k, best_z = n // 2, 0.0
    for k in range(10, n - 10):  # 整体变点：变坏幅度的标准化差最大处（两侧各至少 10 批）
        m1, m2 = pre[k] / k, (pre[n] - pre[k]) / (n - k)
        z = (m2 - m1) / sd * math.sqrt(k * (n - k) / n)
        if z > best_z:
            best_k, best_z = k, z
    t_star = lots[best_k]["time"]
    pms = [e for e in ds.get("events", []) if e.get("type") == "PM"]
    steps = sorted({f for l in lots for f in l["steps"] if f in STEPS})
    out = {}
    for f in steps:
        tools = sorted({l["steps"][f] for l in lots if f in l["steps"]})
        if len(tools) < 2:
            continue
        rows = []
        for t in tools:
            post_in = [b for l, b in zip(lots[best_k:], bad[best_k:]) if l["steps"].get(f) == t]
            post_out = [b for l, b in zip(lots[best_k:], bad[best_k:]) if l["steps"].get(f) not in (None, t)]
            pre_in = [b for l, b in zip(lots[:best_k], bad[:best_k]) if l["steps"].get(f) == t]
            pre_out = [b for l, b in zip(lots[:best_k], bad[:best_k]) if l["steps"].get(f) not in (None, t)]
            rows.append((_welch(post_in, post_out), _welch(pre_in, pre_out), t))
        rows.sort(key=lambda r: -r[0])
        t_best, t_pre, tool = rows[0]
        pm = any(e["tool"] == tool and -72 <= _hours(e["time"], t_star) <= 24 for e in pms)
        out[f] = {"tool": tool, "t": round(t_best, 2), "gap": round(t_best - rows[1][0], 2), "pre": round(t_pre, 2), "pm": pm}
    return {"t_star": t_star, "z_star": round(best_z, 2), "steps": out}


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
        "raw_signals": raw_signals(ds),
    }
