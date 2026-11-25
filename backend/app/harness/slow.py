"""慢脑（System 2）：快脑判不准时，针对这一个案例做的事。双层递归的内层（参照 Harness²：提议 → 探测 → 合成）。

参照 Agnes Harness 技术报告第 7 节：快脑是训练出来的小模型，慢脑负责深度推理；慢脑的结论回流到快脑的训练。

慢脑的组件（每个都是 harness 组件，可以被 RRSI 开关、被条件规则覆盖、被改进经验库跳过）：
  探测题        课程 agent 出一批与本案相近的孪生题（Curriculum.test_time），三分之一留作探测题，对错由孪生程序判定
  近邻训练  另外三分之二由执行 agent 作答、验证 agent 判定（RSIAgent 的方式），只更新一份临时贝叶斯记忆
                （该症状旧份额先降权，再按验证结果学份额、机台热点）；快脑模型参数冻结，读入临时记忆重新算特征、
                重新排序；用完即弃（参照 Sun et al. 2020 与 Agnes 的逐请求 TTT：可验证的辅助题、对基线锚定、变差就回退）
                v0.7 早期还有一个“只调快脑根因偏置层”的挑战者：演示历史 48 次里只赢过快脑 1 次，孪生压力题上还拖后腿，
                已经并入这里并去掉偏置调整
  贝叶斯意见   原始贝叶斯排序：一个结构完全不同、零成本的独立意见
  证据族对照      去掉某一族证据，首位会不会变（找出只靠单一证据的结论，标为需现场鉴别）
  大模型调查员    配了 Key 时由大模型写 Python、在沙箱里切分本案数据核查假设（在工作流里接在慢脑之后）
裁决：各候选在同一批探测题上的命中数比较；挑战者必须比快脑多答对 ≥ max(1, ⌈min_gain × 探测题数⌉) 道才改判，
     否则沿用快脑（打平时快脑优先）。慢脑不改线上快脑——它的作用通过收据回流：推翻快脑的次数是训练窗口的触发条件，
     经工程师确认的案例进入下一次训练。
"""

from __future__ import annotations

import math
import random

from ..fab.kg import forget_table, rank
from ..fastbrain.data import observe_many, to_sample
from ..fastbrain.infer import model_rank, top_hits
from ..fastbrain.model import FastModel
from ..rsi.agents import Curriculum, Executor, Verifier
from ..rsi.calibration import feature_rows, fit
from ..rsi.engine import consolidate, practice_many
from ..rsi.memory import MemoryState
from ..rsi.twin import FabTwin

ANCHOR = 8.0  # 临时证据权重的锚定强度（岭惩罚中心 = 线上权重）
PRIORITY = {"fast": 0, "ttt": 1, "bayes": 2}
NAMES = {"fast": "快脑模型", "ttt": "近邻训练", "bayes": "贝叶斯意见"}
FAMILIES = {"commonality": ({"共性分析", "机台热点"}, "机台或批次共性"), "fdc": ({"FDC"}, "FDC 偏移"),
            "maintenance": ({"PM 时序"}, "维护时序")}


def contrast_probes(hyps: list[dict]) -> list[dict]:
    """证据族对照：保持数据与候选集合不变，挨个去掉一族证据，看首位是否改变。"""
    if not hyps:
        return []
    top, out = hyps[0]["cause_code"], []
    for fam, (names, label) in FAMILIES.items():
        if not any(f["name"] in names for h in hyps for f in h["factors"]):
            continue
        scores = [sum(f.get("w", 1.0) * math.log(max(f["x"], 1e-9)) for f in h["factors"] if f["name"] not in names) for h in hyps]
        challenger = hyps[max(range(len(scores)), key=scores.__getitem__)]
        out.append({"family": fam, "label": label, "challenger": challenger["cause_code"], "name": challenger["name"],
                    "changed": challenger["cause_code"] != top})
    return out


def _specs(symptom: str, evidence: dict, n: int, seed: int):
    n_probe = max(4, n // 3)
    specs = Curriculum().test_time(symptom, evidence, n, random.Random(seed))
    probe, train = specs[:n_probe], specs[n_probe:]
    return probe, [seed * 1000 + i for i in range(len(probe))], train, [seed * 1000 + 500 + i for i in range(len(train))]


async def memory_ttt(symptom: str, mem: MemoryState, keep: float, train, train_seeds, probe_obs: list[dict], twin: FabTwin) -> dict:
    """近邻练习：执行 agent 作答、验证 agent 判定，只更新一份临时贝叶斯记忆（RSIAgent 的方式）。"""
    scratch = mem.clone()
    scratch.examples = []
    if symptom in scratch.edges:  # 测试时适应：临时记忆里该症状的旧份额先降权，让近邻题的结果说了算（是否采纳由探测题把关）
        forget_table(scratch.edges[symptom], keep)
    results = await practice_many(twin, scratch, train, train_seeds, Executor(), Verifier())
    for r in results:
        consolidate(scratch, r, weight=1.0)
    refit = None
    if len(scratch.examples) >= 12:
        res = fit(feature_rows(scratch.examples, scratch.edges, scratch.hotspot_weights()), lam=ANCHOR, start=mem.weights, anchor=mem.weights)
        if res["n"] >= 12:
            scratch.weights, refit = res["weights"], res["weights"]
    hw = scratch.hotspot_weights()
    hits = 0  # 对照：贝叶斯直接用临时记忆在探测题上的命中（只记录，不参与裁决）
    for o in probe_obs:
        h = rank(o["evidence"], o["symptom"], scratch.edges, hw, scratch.weights)
        hits += bool(h and h[0].cause_code == o["truth"])
    return {"scratch": scratch, "hits": hits, "train_passes": sum(r["verdict"] == "PASS" for r in results), "weights": refit,
            "cost": sum(1 + len(r["attempts"]) for r in results)}


async def components(evidence: dict, symptom: str, mem: MemoryState, model: FastModel, prm: dict, seed: int,
                     twin: FabTwin | None = None, cache: dict | None = None, want_rankings: bool = True) -> dict:
    """跑探测题与启用的慢脑组件，返回各候选的探测命中、对本案的首位与成本。cache 供 RRSI 评估复用。"""
    twin = twin or FabTwin()
    n = int(prm["slow.ttt_tasks"])
    probe, ps_seeds, train, tr_seeds = _specs(symptom, evidence, n, seed)
    cache = cache if cache is not None else {}
    pk = ("probe", seed, n, id(twin), id(model))
    if pk not in cache:
        obs = await observe_many(twin, probe, ps_seeds)
        samples = [to_sample(o, mem) for o in obs]
        cache[pk] = {"obs": obs, "samples": samples, "fast": top_hits(model, samples),
                     "bayes": top_hits(FastModel.bayes(mem.weights), samples)}
    P = cache[pk]
    out = {"probe_n": len(probe), "hits": {"fast": P["fast"], "bayes": P["bayes"]}, "cost": len(probe), "rankings": {}, "detail": {}}
    if prm.get("slow.ttt"):
        key = ("ttt", seed, n, float(prm["slow.ttt_keep"]), id(twin), id(model))
        if key not in cache:
            mt = await memory_ttt(symptom, mem, float(prm["slow.ttt_keep"]), train, tr_seeds, P["obs"], twin)
            sc = mt["scratch"]
            r = model_rank(model, evidence, symptom, sc)  # 快脑模型参数冻结，读临时记忆重排
            cache[key] = {"hits": top_hits(model, [to_sample(o, sc) for o in P["obs"]]), "top": r[0]["cause_code"] if r else "",
                          "ranking": r, "cost": mt["cost"], "train_passes": mt["train_passes"], "memory_hits": mt["hits"]}
        c = cache[key]
        out["hits"]["ttt"] = c["hits"]
        out["cost"] += c["cost"]
        out["rankings"]["ttt"] = c["ranking"]
        out["detail"]["ttt"] = {"tasks": len(train), "train_passes": c["train_passes"], "top": c["top"],
                                "bayes_on_scratch_hits": c["memory_hits"]}
    if not want_rankings:
        out["tops"] = {k: (v[0]["cause_code"] if v else "") for k, v in out.pop("rankings").items()}
    return out


def arbitrate(hits: dict, probe_n: int, min_gain: float) -> dict:
    need = max(1, math.ceil(min_gain * probe_n))
    best = sorted(hits, key=lambda k: (-hits[k], PRIORITY[k]))[0]
    chosen = best if best != "fast" and hits[best] - hits["fast"] >= need else "fast"
    tie = sum(1 for k in hits if hits[k] == hits[best]) > 1
    if chosen == "fast":
        why = (f"快脑在探测题上不输（{hits['fast']}/{probe_n}）" if best == "fast" else
               f"{NAMES[best]} 只多答对 {hits[best] - hits['fast']} 道（需 ≥ {need}），沿用快脑")
    else:
        why = f"{NAMES[chosen]} 探测命中 {hits[chosen]}/{probe_n}，比快脑多 {hits[chosen] - hits['fast']} 道（需 ≥ {need}）：改判"
    return {"chosen": chosen, "need": need, "best": best, "tie": tie, "why": why}


async def slow_brain(evidence: dict, symptom: str, mem: MemoryState, model: FastModel, fast_rank: list[dict], bayes_hyps: list[dict],
                     prm: dict, exp: dict, seed: int, twin: FabTwin | None = None) -> dict:
    proposals = [{"kind": "invariant", "name": "贝叶斯意见", "why": "结构完全不同的独立意见，作为挑战者参与裁决"},
                 {"kind": "domain", "name": "证据族对照", "why": "检查结论是否只靠单一证据族"}]
    run = dict(prm)
    tip = exp.get("ttt")
    label = "近邻训练"
    if not prm.get("slow.ttt"):
        proposals.append({"kind": "invariant", "name": label, "skipped": True, "why": "harness 配置里该组件关闭（或被条件规则关闭）"})
    elif prm.get("experience.use") and tip is not None and tip.status == "retired":
        run["slow.ttt"] = 0
        proposals.append({"kind": "invariant", "name": label, "skipped": True,
                          "why": f"改进经验库：该条件下这个组件已停用（{tip.retired_reason}），跳过以节省孪生算力"})
    else:
        proposals.append({"kind": "invariant", "name": label,
                          "why": "记忆和快脑模型都可能没跟上产线现状：临时记忆在可验证的近邻题上学，快脑模型读它重排" +
                                 ("（改进经验库：该条件下稳定有效）" if tip is not None and tip.status == "stable" else "")})
    res = await components(evidence, symptom, mem, model, run, seed, twin)
    arb = arbitrate(res["hits"], res["probe_n"], float(prm["arbiter.min_gain"]))
    rankings = {"fast": fast_rank, "bayes": bayes_hyps, **res["rankings"]}
    final = rankings[arb["chosen"]]
    probes = contrast_probes(bayes_hyps)
    sensitive = [p["label"] for p in probes if p["changed"]]
    tops = {k: (v[0]["cause_code"] if v else "") for k, v in rankings.items()}
    return {"reference": tops["fast"], "proposals": proposals, "probes": probes, "probe_n": res["probe_n"], "hits": res["hits"],
            "tops": tops, "arbiter": arb, "cost": res["cost"], "components": res["detail"],
            "compose": {"adopted": NAMES[arb["chosen"]], "why": arb["why"], "sensitive": sensitive}, "final": final}
