"""快脑小模型：Wide & Deep & Cross 排序网络（纯 Python，几百个参数，毫秒级推理）。

结构参照工业界的排序模型：Wide & Deep（Cheng et al. 2016，Google Play 推荐）与 DCN-v2（Wang et al. 2021，
Google 的大规模排序系统，显式特征交叉）；损失用列表式 softmax 交叉熵（ListNet / 列表式 learning-to-rank）；
概率用温度缩放校准（Guo et al. 2017）。

  x0 = [候选特征 x ; 根因 ID 嵌入 e_c]
  分数 = a·x                          宽（线性）部分：初始化为贝叶斯证据权重
       + v·tanh(W·x0 + b)             深部分：隐层
       + w_c·(x0 ⊙ (U·Vᵀ·x0 + c) + x0)  交叉部分（DCN-v2 低秩交叉层）：自动学特征之间的乘积交互
       + 偏置[c]                        根因偏置层：本厂当前各根因多常见的残差
  概率 = softmax(分数 / T)

- 初始化时 v、w_c、偏置都为 0：未训练的模型 = 贝叶斯排序。训练学的是“在贝叶斯之上的校正”，
  晋升门要求它在留出题上不比贝叶斯差，才拿到裁决权。
- DCN-v2 交叉层固定开启（加了原始信号特征后，命中率不变、概率校准更好：ECE 0.021 → 0.014）；
  隐层宽度、根因嵌入维度由训练策略决定，RSI-Opt 外层可以按实测结果改它们。
- 训练 = Adam + L2（宽部分锚定在贝叶斯权重，全部参数锚定在上一版，连续训练时不遗忘）；
  训练循环里有控制器（RSI-Opt 的内层）：每个 epoch 末决定继续 / 降学习率 / 回滚 / 提前停止。
"""

from __future__ import annotations

import math
import random
from operator import mul

from .features import ALL_CAUSES, LABELS, NAMES, bayes_init

D = len(NAMES)
CIDX = {c: i for i, c in enumerate(ALL_CAUSES)}
NC = len(ALL_CAUSES)


def _softmax(s: list[float], T: float = 1.0) -> list[float]:
    m = max(s)
    e = [math.exp((v - m) / T) for v in s]
    z = sum(e)
    return [v / z for v in e]


class FastModel:
    def __init__(self, hidden: int = 8, theta: list[float] | None = None, init_weights: dict | None = None,
                 seed: int = 0, T: float = 1.0, rank: int = 4, emb: int = 4):
        self.H, self.R, self.K = int(hidden), int(rank), int(emb)
        self.dx = D + self.K
        o, n = {}, 0
        for name, size in (("a", D), ("E", NC * self.K), ("W", self.H * self.dx), ("b", self.H), ("v", self.H),
                           ("U", self.dx * self.R), ("V", self.dx * self.R), ("c", self.dx if self.R else 0),
                           ("wc", self.dx if self.R else 0), ("bias", NC)):
            o[name] = (n, n + size)
            n += size
        self.o, self.size = o, n
        if theta is None:
            rng = random.Random(seed)
            th = [0.0] * n
            th[0:D] = bayes_init(init_weights)
            for name, sd in (("E", 0.1), ("W", 0.3), ("U", 0.1), ("V", 0.1)):
                lo, hi = o[name]
                for i in range(lo, hi):
                    th[i] = rng.gauss(0, sd)
            theta = th
        self.theta = [float(v) for v in theta]
        self.T = float(T)
        assert len(self.theta) == self.size  # noqa: S101

    @property
    def arch(self) -> dict:
        return {"hidden": self.H, "rank": self.R, "emb": self.K}

    @property
    def n_params(self) -> int:
        return self.size

    @classmethod
    def bayes(cls, weights: dict | None) -> "FastModel":
        """与贝叶斯排序完全等价的模型（深、交叉部分输出为 0，偏置为 0）。"""
        return cls(1, init_weights=weights, seed=0, rank=0, emb=0)

    def clone(self) -> "FastModel":
        return FastModel(self.H, list(self.theta), T=self.T, rank=self.R, emb=self.K)

    def to_dict(self) -> dict:
        return {**self.arch, "d": D, "T": round(self.T, 4), "theta": [round(v, 5) for v in self.theta]}

    @classmethod
    def from_dict(cls, d: dict | None) -> "FastModel | None":
        """特征维度变了（例如新增了原始信号特征）的旧模型不能再用：返回 None，快脑退回贝叶斯，等下一个训练窗口。"""
        if not d or not d.get("theta") or d.get("d", 19) != D:
            return None
        return cls(d["hidden"], d["theta"], T=d.get("T", 1.0), rank=d.get("rank", 0), emb=d.get("emb", 0))

    def bias_of(self, code: str) -> float:
        i = CIDX.get(code)
        return self.theta[self.o["bias"][0] + i] if i is not None else 0.0

    def _sl(self, th: list[float], name: str) -> list[float]:
        lo, hi = self.o[name]
        return th[lo:hi]

    def _unpack(self, th: list[float] | None = None) -> dict:
        th = th if th is not None else self.theta
        dx, K, R = self.dx, self.K, self.R
        Wf, Ef, Uf, Vf = self._sl(th, "W"), self._sl(th, "E"), self._sl(th, "U"), self._sl(th, "V")
        return {"a": self._sl(th, "a"), "E": [Ef[i * K:(i + 1) * K] for i in range(NC)],
                "W": [Wf[h * dx:(h + 1) * dx] for h in range(self.H)], "b": self._sl(th, "b"), "v": self._sl(th, "v"),
                "U": [Uf[i * R:(i + 1) * R] for i in range(dx)], "Vt": [[Vf[i * R + j] for i in range(dx)] for j in range(R)],
                "c": self._sl(th, "c"), "wc": self._sl(th, "wc"), "bias": self._sl(th, "bias")}

    def _cand(self, x: list[float], c: str, P: dict) -> tuple:
        ci = CIDX.get(c)
        x0 = x + (P["E"][ci] if ci is not None and self.K else [0.0] * self.K)
        t = [math.tanh(bh + sum(map(mul, wh, x0))) for wh, bh in zip(P["W"], P["b"])]
        wide = sum(map(mul, P["a"], x))
        deep = sum(map(mul, P["v"], t))
        q = m = x1 = None
        cross = 0.0
        if self.R:
            q = [sum(map(mul, vj, x0)) for vj in P["Vt"]]
            m = [sum(map(mul, ui, q)) + ci_ for ui, ci_ in zip(P["U"], P["c"])]
            x1 = [x0i * mi + x0i for x0i, mi in zip(x0, m)]
            cross = sum(map(mul, P["wc"], x1))
        bias = P["bias"][ci] if ci is not None else 0.0
        return wide + deep + cross + bias, (ci, x0, t, q, m, x1), (wide, deep, cross, bias)

    def forward(self, X: list[list[float]], codes: list[str], th: list[float] | None = None, parts: dict | None = None):
        P = parts or self._unpack(th)
        scores, caches = [], []
        for x, c in zip(X, codes):
            s, cache, _ = self._cand(x, c, P)
            scores.append(s)
            caches.append(cache)
        return scores, caches

    def probs(self, X: list[list[float]], codes: list[str]) -> list[float]:
        if not X:
            return []
        return _softmax(self.forward(X, codes)[0], self.T)

    def explain(self, X: list[list[float]], codes: list[str], k: int = 3) -> list[list[dict]]:
        """每个候选：相对其它候选的平均水平，哪几项把它的分数推高/压低（softmax 只看相对差）。"""
        P = self._unpack()
        terms = []
        for x, c in zip(X, codes):
            _, _, (wide, deep, cross, bias) = self._cand(x, c, P)
            row = {NAMES[j]: P["a"][j] * x[j] for j in range(D)}
            row.update({"_deep": deep, "_cross": cross, "_bias": bias})
            terms.append(row)
        keys = list(terms[0]) if terms else []
        mean = {key: sum(t[key] for t in terms) / len(terms) for key in keys}
        label = {**LABELS, "_deep": "证据交互（深层）", "_cross": "特征交叉（交叉层）", "_bias": "根因偏置（本厂残差）"}
        out = []
        for t in terms:
            rel = sorted(((key, t[key] - mean[key]) for key in keys), key=lambda kv: -abs(kv[1]))
            out.append([{"name": label[key], "key": key, "delta": round(d, 3)} for key, d in rel[:k] if abs(d) > 0.05])
        return out

    # ── 训练 ──
    def grad(self, batch: list[dict], th: list[float], reg: dict) -> tuple[float, list[float]]:
        P = self._unpack(th)
        o, dx, K, R, H = self.o, self.dx, self.K, self.R, self.H
        g = [0.0] * self.size
        oa, oE, oW, ob, ov = o["a"][0], o["E"][0], o["W"][0], o["b"][0], o["v"][0]
        oU, oV, oc, owc, obias = o["U"][0], o["V"][0], o["c"][0], o["wc"][0], o["bias"][0]
        loss, wsum = 0.0, 0.0
        for s in batch:
            if s["y"] is None:
                continue
            w = s.get("w", 1.0)
            scores, caches = self.forward(s["X"], s["codes"], parts=P)
            p = _softmax(scores)
            loss -= w * math.log(max(p[s["y"]], 1e-12))
            wsum += w
            for i, (x, (ci, x0, t, q, m, x1)) in enumerate(zip(s["X"], caches)):
                gi = w * (p[i] - (1.0 if i == s["y"] else 0.0))
                if abs(gi) < 1e-9:
                    continue
                for j in range(D):
                    g[oa + j] += gi * x[j]
                if ci is not None:
                    g[obias + ci] += gi
                dx0 = [0.0] * dx if (K and ci is not None) else None
                vv = P["v"]
                for h in range(H):
                    g[ov + h] += gi * t[h]
                    gz = gi * vv[h] * (1 - t[h] * t[h])
                    if gz:
                        g[ob + h] += gz
                        base = oW + h * dx
                        wh = P["W"][h]
                        for j in range(dx):
                            g[base + j] += gz * x0[j]
                        if dx0 is not None:
                            for j in range(D, dx):
                                dx0[j] += gz * wh[j]
                if R:
                    wc = P["wc"]
                    gm = [gi * wc[j] * x0[j] for j in range(dx)]
                    gq = [0.0] * R
                    for j in range(dx):
                        g[owc + j] += gi * x1[j]
                        g[oc + j] += gm[j]
                        if gm[j]:
                            row = P["U"][j]
                            base = oU + j * R
                            for r in range(R):
                                g[base + r] += gm[j] * q[r]
                                gq[r] += row[r] * gm[j]
                    for j in range(dx):
                        base = oV + j * R
                        for r in range(R):
                            g[base + r] += x0[j] * gq[r]
                    if dx0 is not None:
                        Vt = P["Vt"]
                        for j in range(D, dx):
                            dx0[j] += gi * wc[j] * (m[j] + 1) + sum(Vt[r][j] * gq[r] for r in range(R))
                if dx0 is not None:
                    base = oE + ci * K
                    for kk in range(K):
                        g[base + kk] += dx0[D + kk]
        n = max(wsum, 1e-9)
        g = [x / n for x in g]
        loss /= n
        # 正则：宽部分锚定在贝叶斯权重；深 / 交叉 / 嵌入 L2；根因偏置 L2；全部参数锚定在上一版（连续训练不遗忘）
        l2, bl2, anc, a0, prev = reg["l2"], reg["bias_l2"], reg["anchor"], reg["a0"], reg["prev"]
        for j in range(D):
            g[oa + j] += 2 * l2 * (th[oa + j] - a0[j])
        for i in range(o["E"][0], obias):
            g[i] += 2 * l2 * th[i]
        for i in range(obias, self.size):
            g[i] += 2 * bl2 * th[i]
        if anc and prev is not None:
            for i in range(self.size):
                g[i] += 2 * anc * (th[i] - prev[i])
        return loss, g


def nll(model: FastModel, data: list[dict], T: float | None = None) -> float:
    T = model.T if T is None else T
    tot, n = 0.0, 0.0
    for s in data:
        if s["y"] is None:
            continue
        p = _softmax(model.forward(s["X"], s["codes"])[0], T)
        tot -= s.get("w", 1.0) * math.log(max(p[s["y"]], 1e-12))
        n += s.get("w", 1.0)
    return tot / max(n, 1e-9)


def fit(model: FastModel, train: list[dict], val: list[dict], policy: dict, *, a0: list[float], seed: int = 0,
        controller: dict | None = None) -> dict:
    """Adam 训练 + 内层控制器（RSI-Opt：每个 epoch 末是一个决策点）。返回训练日志；model 原地更新为最好的一轮。

    controller：{"decay_on_plateau": bool, "rollback_on_rise": bool}（由 TTSE 训练手册里仍有效的做法决定）。
    """
    ctl = {"decay_on_plateau": True, "rollback_on_rise": True, **(controller or {})}
    rng = random.Random(seed)
    th = list(model.theta)
    prev = list(model.theta)
    reg = {"l2": policy["l2"], "bias_l2": policy["bias_l2"], "anchor": policy["anchor"], "a0": a0,
           "prev": prev if len(prev) == model.size else None}
    lr, b1, b2, eps = policy["lr"], 0.9, 0.999, 1e-8
    m, v = [0.0] * model.size, [0.0] * model.size
    step = 0
    usable = [s for s in train if s["y"] is not None]
    model.theta = th
    best_val, best_th = nll(model, val, 1.0), list(th)
    start_val = best_val
    epochs, decisions, stall, decays = [], [], 0, 0
    bs = int(policy.get("batch", 32))
    for ep in range(int(policy["epochs"])):
        rng.shuffle(usable)
        tl, nb = 0.0, 0
        for i in range(0, len(usable), bs):
            loss, g = model.grad(usable[i:i + bs], th, reg)
            step += 1
            for k in range(model.size):
                m[k] = b1 * m[k] + (1 - b1) * g[k]
                v[k] = b2 * v[k] + (1 - b2) * g[k] * g[k]
                th[k] -= lr * (m[k] / (1 - b1 ** step)) / (math.sqrt(v[k] / (1 - b2 ** step)) + eps)
            tl += loss
            nb += 1
        model.theta = th
        vl = nll(model, val, 1.0)
        epochs.append({"epoch": ep + 1, "train": round(tl / max(1, nb), 4), "val": round(vl, 4), "lr": round(lr, 5)})
        for d in decisions:  # 回看之前的干预：之后两轮内验证损失是否创新低（TTSE 训练手册据此记成败）
            if d.get("helped") is None and ep + 1 - d["epoch"] >= 1 and vl < d["val_at"] - 1e-4:
                d["helped"] = True
            elif d.get("helped") is None and ep + 1 - d["epoch"] >= 2:
                d["helped"] = False
        if not math.isfinite(vl):
            th[:] = best_th
            decisions.append({"epoch": ep + 1, "action": "halt", "why": "验证损失不是有限值，回滚并停止", "val_at": best_val, "helped": None})
            break
        if vl < best_val - 1e-4:
            best_val, best_th, stall = vl, list(th), 0
            continue
        if ctl["rollback_on_rise"] and vl > best_val * 1.05:
            th[:] = best_th
            lr *= 0.5
            m, v = [0.0] * model.size, [0.0] * model.size
            decisions.append({"epoch": ep + 1, "action": "rollback", "why": f"验证损失回升到 {vl:.3f}（最好 {best_val:.3f}）：回滚到最好的一轮、学习率减半",
                              "val_at": best_val, "helped": None})
            stall = 0
            continue
        stall += 1
        if stall >= int(policy["patience"]):
            if ctl["decay_on_plateau"] and decays < 2:
                lr *= 0.5
                decays += 1
                stall = 0
                decisions.append({"epoch": ep + 1, "action": "lr_decay", "why": f"连续 {policy['patience']} 轮验证损失没有创新低：学习率减半",
                                  "val_at": best_val, "helped": None})
            else:
                decisions.append({"epoch": ep + 1, "action": "early_stop", "why": "验证损失进入平台期，提前停止", "val_at": best_val, "helped": None})
                break
    model.theta = best_th
    for d in decisions:
        d.pop("val_at", None)
    return {"epochs": epochs, "decisions": decisions, "val_start": round(start_val, 4), "val_best": round(best_val, 4),
            "steps": step, "samples": len(usable)}


def calibrate_temperature(model: FastModel, data: list[dict]) -> float:
    """温度缩放：在校准集上找使对数损失最小的 T（黄金分割搜索 log T ∈ [−1.2, 1.2]）。"""
    lo, hi = -1.2, 1.2
    gr = (math.sqrt(5) - 1) / 2
    f = lambda lt: nll(model, data, math.exp(lt))  # noqa: E731
    c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    fc, fd = f(c), f(d)
    for _ in range(18):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo)
            fc = f(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo)
            fd = f(d)
    return round(math.exp((lo + hi) / 2), 4)


def evaluate(model: FastModel, data: list[dict]) -> dict:
    """首位命中、对数损失、ECE（首位概率 vs 是否命中），以及逐题结果（用于配对 bootstrap）。"""
    correct, ll, conf = [], [], []
    for s in data:
        if not s["X"]:
            continue
        p = model.probs(s["X"], s["codes"])
        top = max(range(len(p)), key=p.__getitem__)
        y = s["y"]
        correct.append(1.0 if y is not None and top == y else 0.0)
        ll.append(-math.log(max(p[y], 1e-12)) if y is not None else -math.log(1e-4))
        conf.append(p[top])
    n = max(1, len(correct))
    bins: dict[int, list] = {}
    for c, ok in zip(conf, correct):
        bins.setdefault(min(9, int(c * 10)), []).append((c, ok))
    ece = sum(len(b) * abs(sum(c for c, _ in b) / len(b) - sum(o for _, o in b) / len(b)) for b in bins.values()) / n
    return {"n": len(correct), "hit1": round(100 * sum(correct) / n, 2), "logloss": round(sum(ll) / n, 4), "ece": round(ece, 4),
            "_correct": correct, "_ll": ll, "_conf": conf}
