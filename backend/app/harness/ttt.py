"""测试时训练（TTT）：慢脑里针对当前案例，在数字孪生中做一组近邻练习，只改一份临时记忆来判这一案。

参照 Agnes Sys1 技术报告的 TTT 设计（逐请求、自出可验证的辅助题、只改少量参数、对基线做锚定、预测完即丢弃、
构造不出可靠样本或结果变差就回退到基线）与 Sun et al. 2020 的 test-time training。不同之处：
- Kaizen 没有神经网络权重，临时更新的是该症状的根因份额、机台热点和证据权重（证据权重以线上版本为中心做强锚定）；
- 辅助题由课程 agent 出（Curriculum.test_time），标签来自孪生的干预反馈（验证 agent 判定），程序可验证；
- 留出三分之一的题做“探测”：同一批题，练习前后各答一次（每题只给一次机会），命中多了才采纳练习后的结论。
"""

from __future__ import annotations

import random

from ..fab.kg import forget_table, rank
from ..rsi.agents import Curriculum, Executor, Verifier
from ..rsi.calibration import feature_rows, fit
from ..rsi.engine import consolidate, practice_many
from ..rsi.memory import MemoryState
from ..rsi.twin import FabTwin

ANCHOR = 8.0  # 临时证据权重的锚定强度（岭惩罚中心 = 线上权重）


async def test_time_training(evidence: dict, symptom: str, mem: MemoryState, params: dict, seed: int,
                             twin: FabTwin | None = None) -> dict:
    n = int(params["ttt_tasks"])
    n_probe = max(4, n // 3)
    rng = random.Random(seed)
    specs = Curriculum().test_time(symptom, evidence, n_probe + (n - n_probe), rng)
    probe, train = specs[:n_probe], specs[n_probe:]
    twin, ex, ver = twin or FabTwin(), Executor(), Verifier()
    probe_seeds = [seed * 1000 + i for i in range(n_probe)]
    train_seeds = [seed * 1000 + 500 + i for i in range(len(train))]
    before = await practice_many(twin, mem, probe, probe_seeds, ex, ver, max_attempts=1)
    scratch = mem.clone()
    scratch.examples = []
    if symptom in scratch.edges:  # 测试时适应：临时记忆里该症状的旧份额先降权，让近邻题的结果说了算（是否采纳由探测题把关）
        forget_table(scratch.edges[symptom], float(params["ttt_keep"]))
    results = await practice_many(twin, scratch, train, train_seeds, ex, ver)
    for r in results:  # 只改临时记忆：份额、热点、条目（权重 = 策略里的 ttt_prior_weight）
        consolidate(scratch, r, weight=float(params["ttt_prior_weight"]))
    refit = None
    if len(scratch.examples) >= 12:  # 证据权重：只用近邻题、强锚定在线上版本附近
        res = fit(feature_rows(scratch.examples, scratch.edges, scratch.hotspot_weights()), lam=ANCHOR,
                  start=mem.weights, anchor=mem.weights)
        if res["n"] >= 12:
            scratch.weights, refit = res["weights"], res["weights"]
    after = await practice_many(twin, scratch, probe, probe_seeds, ex, ver, max_attempts=1)
    b = sum(r["verdict"] == "PASS" for r in before)
    a = sum(r["verdict"] == "PASS" for r in after)
    hyps = [h.as_dict() for h in rank(evidence, symptom, scratch.edges, scratch.hotspot_weights(), scratch.weights)]
    base = rank(evidence, symptom, mem.edges, mem.hotspot_weights(), mem.weights)
    gain = (a - b) / n_probe
    adopt = a > b and gain >= float(params["adopt_min_gain"])
    return {"tasks": len(specs) + n_probe, "probe_n": n_probe, "probe_before": b, "probe_after": a,
            "train_passes": sum(r["verdict"] == "PASS" for r in results), "weights": refit,
            "top_before": base[0].cause_code if base else "", "top_after": hyps[0]["cause_code"] if hyps else "",
            "posterior_after": hyps[0]["posterior"] if hyps else 0.0, "adopt": adopt, "ranking": hyps,
            "reason": ("采纳：探测题命中 {b}/{n} → {a}/{n}" if adopt else
                       "回退到线上记忆：探测题命中 {b}/{n} → {a}/{n}，没有可靠提升").format(b=b, a=a, n=n_probe)}
