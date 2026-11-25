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
from collections import OrderedDict

from ..fab.kg import FACTOR_NAMES, EdgeState, rank

PRIOR_LAMBDA = 2.0  # 越大越相信专家设定；样本少时自动回到 w≈1


def features(evidence: dict, symptom: str, edges: dict[str, dict[str, EdgeState]],
             hotspots: dict[tuple[str, str], float]) -> list[tuple[str, list[float]]]:
    """每个候选根因的 log 因子向量（顺序同 FACTOR_NAMES；同名因子相加），按该症状下根因的登记顺序排列。

    rank() 产生的因子都在 FACTOR_NAMES 之内，所以 softmax(Σ w·f) 与 rank(..., weights) 的后验完全相同。
    """
    rows = {}
    for h in rank(evidence, symptom, edges, hotspots):
        f = [0.0] * len(FACTOR_NAMES)
        for x in h.factors:
            f[FACTOR_NAMES.index(x["name"])] += math.log(max(x["x"], 1e-9))
        rows[h.cause_code] = (h.cause_code, f)
    order = list(edges.get(symptom, {}))
    return [rows[c] for c in order if c in rows] + [r for c, r in rows.items() if c not in order]


def score(rows: list[tuple[str, list[float]]], weights: dict | None, truth: str) -> tuple[bool, float]:
    """用缓存的特征计算（首位是否命中, 真因后验），与 rank(..., weights) 的结果一致（并列时取登记顺序靠前者）。"""
    if not rows:
        return False, 0.0
    w = [float((weights or {}).get(n, 1.0)) for n in FACTOR_NAMES]
    s = [sum(wi * fi for wi, fi in zip(w, f)) for _, f in rows]
    m = max(s)
    ex = [math.exp(v - m) for v in s]
    z = sum(ex)
    best = s.index(m)
    post = next((e / z for (c, _), e in zip(rows, ex) if c == truth), 0.0)
    return rows[best][0] == truth, post


# 特征缓存：一道题的特征只取决于它的证据、该症状下各根因的份额与相关机台热点。
# 校准波次之间大部分症状的份额不变，所以重拟合时绝大多数样本可以直接复用上次算好的特征（结果完全一致）。
_FEATURES: OrderedDict = OrderedDict()  # LRU：常用的（测试集、验收门、训练样本）留在缓存里
_FEATURES_MAX = 60_000
_FP_IDS: dict[tuple, int] = {}


def _fingerprints(edges: dict[str, dict[str, EdgeState]], hotspots: dict[tuple[str, str], float]) -> dict[str, int]:
    by_cause: dict[str, list] = {}
    for (cause, tool), v in hotspots.items():
        by_cause.setdefault(cause, []).append((tool, v))
    out = {}
    for sym, table in edges.items():
        fp = (sym, tuple((c, e.alpha, e.alpha0) for c, e in table.items()),
              tuple((c, tuple(sorted(by_cause.get(c, [])))) for c in table))
        if fp not in _FP_IDS:
            if len(_FP_IDS) > 200_000:
                _FP_IDS.clear()
                _FEATURES.clear()
            _FP_IDS[fp] = len(_FP_IDS)
        out[sym] = _FP_IDS[fp]
    return out


def feature_rows(examples: list[tuple[dict, str, str]], edges: dict[str, dict[str, EdgeState]],
                 hotspots: dict[tuple[str, str], float]) -> list[tuple[list, str]]:
    """[(证据, 症状, 已验证根因)] → fit() 的输入；未变化的样本复用缓存。"""
    fps = _fingerprints(edges, hotspots)
    out = []
    for ev, sym, cause in examples:
        key = (id(ev), fps.get(sym))
        hit = _FEATURES.get(key)
        if hit is None or hit[0] is not ev:
            hit = _FEATURES[key] = (ev, features(ev, sym, edges, hotspots))
            if len(_FEATURES) > _FEATURES_MAX:
                _FEATURES.popitem(last=False)
        else:
            _FEATURES.move_to_end(key)
        out.append((hit[1], cause))
    return out


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
        iters: int = 25, start: dict | None = None, anchor: dict | None = None) -> dict:
    """examples: [(候选根因及其特征, 已验证的根因)]。返回权重、样本数与训练集对数似然。

    start：从上一版权重热启动（目标函数是凸的，最优解不变，只是少走几步牛顿迭代）。
    anchor：岭惩罚的中心（默认是专家设定 w=1；测试时训练用线上权重做中心，只允许小幅偏离）。
    """
    k = len(FACTOR_NAMES)
    w = [float((start or {}).get(n, 1.0)) for n in FACTOR_NAMES]
    center = [float((anchor or {}).get(n, 1.0)) for n in FACTOR_NAMES]
    data = [(rows, truth) for rows, truth in examples if any(c == truth for c, _ in rows) and len(rows) > 1]
    rng = range(k)
    for _ in range(iters):
        g = [-lam * (wi - ci) for wi, ci in zip(w, center)]
        hmat = [[(-lam if i == j else 0.0) for j in rng] for i in rng]
        for rows, truth in data:
            feats = [f for _, f in rows]
            s = [sum(wi * fi for wi, fi in zip(w, f)) for f in feats]
            mx = max(s)
            ex = [math.exp(v - mx) for v in s]
            z = sum(ex)
            p = [e / z for e in ex]
            mean = [sum(pj * f[i] for pj, f in zip(p, feats)) for i in rng]
            for (c, f), pj in zip(rows, p):
                r = (1.0 if c == truth else 0.0) - pj
                d = [fi - mi for fi, mi in zip(f, mean)]
                g = [gi + r * fi for gi, fi in zip(g, f)]
                hmat = [[h - pd * dl for h, dl in zip(row, d)] for row, pd in zip(hmat, [pj * di for di in d])]
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


def gate_check(mem, weights: dict | None, round_: int | None = None) -> dict:
    """在验收门面板上评估一组证据权重：首位命中率、平均对数损失（越低越好），以及逐题结果与分症状对数损失（配对比较用）。"""
    from .twin import gate_panel

    hw = mem.hotspot_weights()
    panel = gate_panel(getattr(mem, "era", None), getattr(mem, "gate_round", 0) if round_ is None else round_)
    hits, lls = [], []
    if not weights:  # 专家设定（无权重）时 rank() 用因子乘积，逐题计算以保证一致
        for e in panel:
            hyps = rank(e["evidence"], e["symptom"], mem.edges, hw, weights)
            hits.append(1.0 if hyps and hyps[0].cause_code == e["truth"] else 0.0)
            lls.append(-math.log(max(next((h.posterior for h in hyps if h.cause_code == e["truth"]), 0.0), 1e-6)))
    else:
        rows = feature_rows([(e["evidence"], e["symptom"], e["truth"]) for e in panel], mem.edges, hw)
        for r, truth in rows:
            hit, post = score(r, weights, truth)
            hits.append(1.0 if hit else 0.0)
            lls.append(-math.log(max(post, 1e-6)))
    by_sym: dict[str, list[float]] = {}
    for e, ll in zip(panel, lls):
        by_sym.setdefault(e["symptom"], []).append(ll)
    return {"hit1": round(100 * sum(hits) / len(panel), 2), "logloss": round(sum(lls) / len(panel), 4), "n": len(panel),
            "by_symptom": {k: round(sum(v) / len(v), 4) for k, v in by_sym.items()}, "_hit": hits, "_ll": lls}
