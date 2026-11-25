"""快脑小模型的输入：每个候选根因一行特征。

两类特征：
- 贝叶斯证据因子（5 个，取对数）：先验份额、共性分析、FDC、PM 时序、机台热点。模型初始化时这 5 个的线性权重
  等于 RSIAgent 学到的证据权重，所以未训练的模型和贝叶斯排序完全一致；训练只学“在此之上该怎么改”。
- 贝叶斯没用上的结构信号：未达显著的机台信号 × 机台热点记忆、变点是否就在该根因的工序、PM 后突变是否发生在
  本工序的机台、FDC 偏移落在哪个参数、显著共性的类型是否与根因类型吻合……这些交互是乘积模型表达不了的。

- 原始信号（v0.7）：统计内核从批次原始数据算出的连续指标检验（见 fab/analytics.raw_signals）——变点之后的时间窗里，
  本工序最差机台的 Welch t 值、它领先第二差机台多少、变点之前是否本来就差、变点前是否刚做过 PM，以及别的工序和全案的
  最强信号。二值失效的共性分析在弱异常时常常一个显著机台都找不到，连续指标在同样的数据里仍然看得出来。

刻意不用的信号（防止学到仿真伪迹、在产线上失效）：
- 未达显著的“物料批次”共性：孪生只给材料类根因生成批次字段，弱批次信号的“存在本身”就泄露了答案；
- 未达 2σ 的 FDC 参数：孪生只监控可疑工序的参数，参数名的出现本身也会泄露工序；
- 晶圆图空间特征：孪生里只由症状决定，与根因无关。
每道工序的机台都参与共性分析，所以“未达显著的机台信号”不受这个问题影响。
"""

from __future__ import annotations

import math

from ..fab.flow import CAUSES, cause_by_code
from ..fab.kg import FACTOR_NAMES, _hotspot_factor, rank
from ..rsi.memory import MemoryState

SIG = 1.3  # 与 kg.rank 相同的显著性门槛（共性分析得分）

FEATURES: tuple[tuple[str, str], ...] = (
    ("lf_prior", "先验份额（记忆）"),
    ("lf_common", "共性分析因子"),
    ("lf_fdc", "FDC 因子"),
    ("lf_pm", "PM 时序因子"),
    ("lf_hot", "机台热点因子"),
    ("sus", "本工序显著共性（类型吻合）"),
    ("sus_mis", "本工序显著共性（类型不符）"),
    ("weak", "本工序未达显著的机台信号"),
    ("hot", "该机台的热点记忆"),
    ("weak_hot", "弱机台信号 × 热点记忆"),
    ("elsewhere", "显著共性在别的工序"),
    ("fdc_own", "FDC 偏移在该根因的参数"),
    ("fdc_other", "FDC 偏移在别的参数"),
    ("cp_here", "变点就在本工序"),
    ("pm_here", "本工序机台 PM 后突变"),
    ("pm_rel_after", "PM 相关根因 × PM 后突变"),
    ("tool_sp", "机台类根因"),
    ("tool_x_batch", "机台类根因 × 首位共性是物料批次"),
    ("strong_any", "本案有显著共性"),
    ("rs_t", "原始信号：变点后本工序最差机台"),
    ("rs_gap", "原始信号：领先本工序第二差机台"),
    ("rs_pre", "原始信号：该机台变点前就偏差"),
    ("rs_hot", "原始信号最差机台 × 热点记忆"),
    ("rs_pm", "原始信号：该机台变点前刚做过 PM"),
    ("rs_other", "原始信号：别的工序的最强机台"),
    ("rs_any", "原始信号：全案最强机台"),
    ("rs_z", "原始信号：整体变点强度"),
)
NAMES = tuple(k for k, _ in FEATURES)
LABELS = dict(FEATURES)
BAYES_INDEX = {n: NAMES.index(k) for n, k in zip(FACTOR_NAMES, ("lf_prior", "lf_common", "lf_fdc", "lf_pm", "lf_hot"))}
ALL_CAUSES: tuple[str, ...] = tuple(c.code for causes in CAUSES.values() for c in causes)


def bayes_init(weights: dict | None) -> list[float]:
    """线性层的初值 = 贝叶斯的证据权重（其余特征为 0）。"""
    a = [0.0] * len(NAMES)
    for name, i in BAYES_INDEX.items():
        a[i] = float((weights or {}).get(name, 1.0))
    return a


def candidate_rows(evidence: dict, symptom: str, mem: MemoryState, hyps: list[dict] | None = None) -> tuple[list[dict], list[list[float]]]:
    """返回（贝叶斯排序的候选 hyps, 每个候选的特征行），两者顺序一致。"""
    hw = mem.hotspot_weights()
    table = mem.edges.get(symptom, {})
    if hyps is None:
        hyps = [h.as_dict() for h in rank(evidence, symptom, mem.edges, hw, mem.weights)]
    sus = evidence.get("suspects", []) or []
    sig = [s for s in sus if s.get("score", 0) >= SIG]
    top_batch = 1.0 if sig and "#" in sig[0]["factor"] else 0.0
    cp = evidence.get("change_point") or {}
    cp_step = (cp.get("factor") or "").split("#")[0]
    after_pm = 1.0 if cp.get("after_pm") else 0.0
    fdc = evidence.get("fdc", []) or []
    raw = evidence.get("raw_signals") or {}
    rsteps = raw.get("steps") or {}
    clip = lambda v, lo, hi: min(hi, max(lo, v))  # noqa: E731
    r_any = clip(max((v["t"] for v in rsteps.values()), default=0.0) / 6, -1.0, 2.0)
    r_z = clip(raw.get("z_star", 0.0) / 6, 0.0, 2.0)
    rows = []
    for h in hyps:
        found = cause_by_code(h["cause_code"])
        cause = found[1] if found else None
        step = h["step"]
        tool_sp = bool(cause and cause.tool_specific)
        lf = [0.0] * 5
        for f in h["factors"]:
            if f["name"] in BAYES_INDEX:
                lf[FACTOR_NAMES.index(f["name"])] += math.log(max(f["x"], 1e-9))
        own = [s for s in sig if s["step"] == step]
        match = max((s["score"] for s in own if ("#" in s["factor"]) != tool_sp), default=0.0)
        mism = max((s["score"] for s in own if ("#" in s["factor"]) == tool_sp), default=0.0)
        tools_here = [s for s in sus if s["step"] == step and "#" not in s["factor"]] if tool_sp else []
        weak = max((min(max(s["score"], 0.0), SIG) for s in tools_here), default=0.0) / SIG
        hot = 0.0
        if tools_here and hw and h["cause_code"] in table:
            best = max(tools_here, key=lambda s: s["score"])
            hot = math.log(max(_hotspot_factor(table, table[h["cause_code"]], best["value"], hw), 1e-6))
        params = set(cause.fdc) if cause else set()
        own_fdc = max((abs(x["z"]) for x in fdc if x["param"] in params and abs(x["z"]) >= 2), default=0.0)
        other_fdc = max((abs(x["z"]) for x in fdc if x["param"] not in params and abs(x["z"]) >= 2), default=0.0)
        cp_here = 1.0 if cp and cp_step == step else 0.0
        pm_rel = 1.0 if cause and cause.pm_related else 0.0
        rs = rsteps.get(step) if tool_sp else None
        r_t = clip(rs["t"] / 6, -1.0, 2.0) if rs else 0.0
        r_hot = 0.0
        if rs and hw and h["cause_code"] in table:
            r_hot = clip(rs["t"] / 4, 0.0, 1.0) * math.log(max(_hotspot_factor(table, table[h["cause_code"]], rs["tool"], hw), 1e-6))
        r_other = clip(max((v["t"] for k, v in rsteps.items() if k != step), default=0.0) / 6, -1.0, 2.0)
        raw_feats = [r_t, clip(rs["gap"] / 6, 0.0, 2.0) if rs else 0.0, clip(rs["pre"] / 6, -2.0, 2.0) if rs else 0.0, r_hot,
                     (1.0 if rs and rs["pm"] else 0.0) * pm_rel, r_other, r_any, r_z]
        rows.append([*lf, min(2.0, match / 4), min(2.0, mism / 4), weak, hot, weak * hot,
                     min(2.0, max((s["score"] for s in sig if s["step"] != step), default=0.0) / 4),
                     min(2.0, own_fdc / 4), min(2.0, other_fdc / 4), cp_here, cp_here * after_pm, pm_rel * after_pm,
                     1.0 if tool_sp else 0.0, (1.0 if tool_sp else 0.0) * top_batch, 1.0 if sig else 0.0, *raw_feats])
    return hyps, rows
