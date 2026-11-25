"""用快脑模型给一个案例排序（推理）。"""

from __future__ import annotations

from ..rsi.memory import MemoryState
from .features import candidate_rows
from .model import FastModel


def model_rank(model: FastModel, evidence: dict, symptom: str, mem: MemoryState, hyps: list[dict] | None = None) -> list[dict]:
    """返回按模型校准概率排序的候选：posterior = 模型概率；bayes_posterior = 原贝叶斯后验；model_why = 主要依据。"""
    hyps, X = candidate_rows(evidence, symptom, mem, hyps)
    if not hyps:
        return []
    codes = [h["cause_code"] for h in hyps]
    p = model.probs(X, codes)
    why = model.explain(X, codes)
    out = [{**h, "bayes_posterior": h.get("bayes_posterior", h["posterior"]), "posterior": round(pi, 4), "model_why": w}
           for h, pi, w in zip(hyps, p, why)]
    return sorted(out, key=lambda h: -h["posterior"])


def top_hits(model: FastModel, samples: list[dict]) -> int:
    n = 0
    for s in samples:
        if s["y"] is None or not s["X"]:
            continue
        p = model.probs(s["X"], s["codes"])
        n += max(range(len(p)), key=p.__getitem__) == s["y"]
    return n
