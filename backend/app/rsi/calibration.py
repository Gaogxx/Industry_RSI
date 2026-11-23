"""证据权重学习（RSI v2 的核心改进）。

诊断发现：只学“本厂常见什么根因”（先验）不够——手工设定的证据因子偏保守，先验一旦变得有把握，强证据反而压不过它
（留出集强证据题从 88.5% 跌到 78.2%）。所以 RSI 还要学“每类证据该信多少”：

    log P(根因 c | 证据) = Σ_k w_k · log 因子_k(c) − log Z        （带可学习指数的专家乘积 / 条件 logit）

w 在所有症状间共享，一道验证过的练习题就能同时改进所有症状的判断，样本效率远高于逐症状计数。
拟合：以专家设定 w=1 为中心的高斯先验（岭回归），牛顿法求最大后验，5 个参数十几次迭代即收敛。
训练标签只来自“已验证”的结论：孪生中验证 agent 判定 PASS 的整改对象，以及产线上工程师确认的根因。
"""

from __future__ import annotations

import math

from ..fab.kg import FACTOR_NAMES, EdgeState, rank

PRIOR_LAMBDA = 2.0  # 越大越相信专家设定；样本少时自动回到 w≈1


def features(evidence: dict, symptom: str, edges: dict[str, dict[str, EdgeState]],
             hotspots: dict[tuple[str, str], float]) -> list[tuple[str, list[float]]]:
    """每个候选根因的 log 因子向量（顺序同 FACTOR_NAMES；同名因子相加）。"""
    rows = []
    for h in rank(evidence, symptom, edges, hotspots):
        f = [0.0] * len(FACTOR_NAMES)
        for x in h.factors:
            if x["name"] in FACTOR_NAMES:
                f[FACTOR_NAMES.index(x["name"])] += math.log(max(x["x"], 1e-9))
        rows.append((h.cause_code, f))
    return rows


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for i in range(n):
        piv = max(range(i, n), key=lambda r: abs(m[r][i]))
        m[i], m[piv] = m[piv], m[i]
        if abs(m[i][i]) < 1e-12:
            continue
        for r in range(n):
            if r != i:
                fct = m[r][i] / m[i][i]
                for c in range(i, n + 1):
                    m[r][c] -= fct * m[i][c]
    return [m[i][n] / m[i][i] if abs(m[i][i]) > 1e-12 else 0.0 for i in range(n)]


def fit(examples: list[tuple[list[tuple[str, list[float]]], str]], lam: float = PRIOR_LAMBDA,
        iters: int = 25) -> dict:
    """examples: [(候选根因及其特征, 已验证的根因)]。返回权重、样本数与训练集对数似然。"""
    k = len(FACTOR_NAMES)
    w = [1.0] * k
    data = [(rows, truth) for rows, truth in examples if any(c == truth for c, _ in rows) and len(rows) > 1]
    for _ in range(iters):
        g = [-lam * (wi - 1.0) for wi in w]
        hmat = [[(-lam if i == j else 0.0) for j in range(k)] for i in range(k)]
        for rows, truth in data:
            s = [sum(wi * fi for wi, fi in zip(w, f)) for _, f in rows]
            mx = max(s)
            ex = [math.exp(v - mx) for v in s]
            z = sum(ex)
            p = [e / z for e in ex]
            mean = [sum(p[j] * rows[j][1][i] for j in range(len(rows))) for i in range(k)]
            for j, (c, f) in enumerate(rows):
                y = 1.0 if c == truth else 0.0
                for i in range(k):
                    g[i] += (y - p[j]) * f[i]
                    for l in range(k):
                        hmat[i][l] -= p[j] * (f[i] - mean[i]) * (f[l] - mean[l])
        step = _solve([[-x for x in row] for row in hmat], g)
        w = [wi + si for wi, si in zip(w, step)]
        if max(abs(si) for si in step) < 1e-4:
            break
    ll = 0.0
    for rows, truth in data:
        s = [sum(wi * fi for wi, fi in zip(w, f)) for _, f in rows]
        mx = max(s)
        z = sum(math.exp(v - mx) for v in s)
        ll += next(sv for (c, _), sv in zip(rows, s) if c == truth) - mx - math.log(z)
    return {"weights": {n: round(v, 3) for n, v in zip(FACTOR_NAMES, w)}, "n": len(data),
            "avg_loglik": round(ll / max(1, len(data)), 4)}


def gate_check(mem, weights: dict | None) -> dict:
    """在验收门面板上评估一组证据权重：首位命中率与平均对数损失（越低越好）。"""
    from .twin import gate_panel

    hw = mem.hotspot_weights()
    hits, loss, panel = 0, 0.0, gate_panel()
    for e in panel:
        hyps = rank(e["evidence"], e["symptom"], mem.edges, hw, weights)
        hits += bool(hyps and hyps[0].cause_code == e["truth"])
        loss -= math.log(max(next((h.posterior for h in hyps if h.cause_code == e["truth"]), 0.0), 1e-6))
    return {"hit1": round(100 * hits / len(panel), 2), "logloss": round(loss / len(panel), 4)}
