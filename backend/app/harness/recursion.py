"""双层递归的内层（任务级）：提议 → 探测 → 合成。参照 Harness²（arXiv:2610.14224）。

Harness² 在没有标准答案的开放任务里，只能靠“行为对比”判断候选改动是否有效；Kaizen 有数字孪生，
探测可以用有真值的近邻练习题来做（命中率前后对比），这是比纯行为对比更强的信号。

- 提议：通用候选（invariant）= 近邻练习 / 测试时训练，针对“记忆可能已经过时”这一通用弱点；
        领域候选（domain）= 证据族对照，针对本案的证据结构（去掉某一族证据，首位会不会变）。
        经验库会调整提议：某条件下近邻练习反复无效（经验已停用）就跳过，省下孪生算力。
- 探测：近邻练习在留出的探测题上前后各答一次；证据族对照在本案上重算排序。
- 合成：探测证明有效才采纳练习后的排序，否则保留原排序；对证据敏感的结论标为“需现场鉴别”。
外层（经验反思）见 experience.py。
"""

from __future__ import annotations

import math

from .ttt import test_time_training

FAMILIES = {"commonality": ({"共性分析", "机台热点"}, "机台或批次共性"), "fdc": ({"FDC"}, "FDC 偏移"),
            "maintenance": ({"PM 时序"}, "维护时序")}


def contrast_probes(hyps: list[dict]) -> list[dict]:
    """证据族对照：保持数据与候选集合不变，挨个去掉一族证据，看首位是否改变（找出驱动结论的单点证据）。"""
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


async def inner_loop(evidence: dict, symptom: str, hyps: list[dict], mem, params: dict, exp: dict, seed: int,
                     twin=None) -> dict:
    ttt_tip = exp.get("ttt")
    skip = bool(params.get("use_experience")) and ttt_tip is not None and ttt_tip.status == "retired"
    proposals = [{"kind": "domain", "name": "证据族对照", "why": "检查结论是否只靠单一证据族"}]
    if skip:
        proposals.append({"kind": "invariant", "name": "近邻练习（测试时训练）", "skipped": True,
                          "why": f"经验库：该条件下近邻练习已停用（{ttt_tip.retired_reason}），跳过以节省孪生算力"})
    else:
        proposals.append({"kind": "invariant", "name": "近邻练习（测试时训练）",
                          "why": "记忆可能没跟上产线现状：针对本案在孪生里做一组可验证的近邻题" +
                                 ("（经验库：该条件下近邻练习稳定有效）" if ttt_tip is not None and ttt_tip.status == "stable" else "")})
    probes = contrast_probes(hyps)
    ttt = None if skip else await test_time_training(evidence, symptom, mem, params, seed, twin)
    final = ttt["ranking"] if ttt and ttt["adopt"] else hyps
    sensitive = [p["label"] for p in probes if p["changed"]]
    compose = {"adopted": "近邻练习后的排序" if ttt and ttt["adopt"] else "原排序",
               "why": ttt["reason"] if ttt else "未做近邻练习",
               "sensitive": sensitive}
    slim = {k: v for k, v in (ttt or {}).items() if k != "ranking"} or None
    return {"reference": hyps[0]["cause_code"] if hyps else "", "proposals": proposals, "probes": probes, "ttt": slim,
            "compose": compose, "final": final}
