"""数字孪生（RSI 的“新环境”）：一座虚构 Fab / 封测厂的可交互模型。

它内部藏着本厂的真实规律——三个 agent 都看不到：
  · SITE_TRUTH：每个症状背后各根因的真实发生频率（与教科书先验不同）
  · HOTSPOTS  ：某个根因在本厂多发于哪台机台 / 腔体
探索时课程 agent 只能指定“症状 + 难度 + 是否加干扰”，孪生按真实规律抽取根因并生成 MES/FDC/晶圆图数据；
验证时只返回“按提议整改后同一时段的指标”（干预反馈），从不直接给出答案。

产线不是一成不变的：CHANGES 记录了过去一年的产线变更（换供应商、新机台导入、PM 程序调整…），
变更之前对应症状的根因分布不同。孪生按“时段”（era）同步这些变更，所以每个月的 RSI 会话都有新东西可学；
在线会话总是面对“现在”的孪生（CURRENT_ERA）。

真实部署时应当用本厂历史数据校准这些表（或接入工艺仿真），否则学到的是孪生而不是产线——
因此孪生验证的记忆权重低于产线验证（见 rsi/memory.py）。
"""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from statistics import mean

from ..fab.analytics import PATTERN_TO_SYMPTOM, analyze_dataset
from ..fab.flow import CAUSES, STEPS, SYMPTOMS, Scenario
from ..fab.simulate import simulate

SITE_TRUTH = {
    "WM_EDGE": {"ETCH_FOCUS_RING": .65, "CMP_RETAINING_RING": .15, "ALD_EDGE_TEMP": .12, "LITHO_EBR": .08},
    "WM_CENTER": {"WET_CENTER_RESIDUE": .6, "CVD_SHOWERHEAD": .28, "IMP_DOSE_CENTER": .12},
    "WM_SCRATCH": {"ROBOT_SCRATCH": .6, "CMP_SLURRY_LPC": .25, "CMP_DISK": .15},
    "WM_CLUSTER": {"PM_SEASONING": .65, "CVD_FLAKE": .2, "RETICLE_HAZE": .15},
    "OVL_DRIFT": {"SCANNER_MATCHING": .45, "LENS_HEATING": .3, "R2R_MODEL": .15, "WAFER_WARP": .1},
    "PKG_WARP": {"SUBSTRATE_CTE": .6, "BOND_TEMP_UNIFORM": .2, "CURE_PROFILE": .2},
    "PKG_VOID": {"UF_DISPENSE": .55, "REFLOW_PROFILE": .3, "FLUX_RESIDUE": .15},
    "HBM_OPEN": {"TCB_COPLANARITY": .6, "NCF_LOT": .25, "HBM_INCOMING": .15},
    "TEST_VMIN": {"SOCKET_CONTACT": .65, "PROCESS_SLOW_CORNER": .22, "TEST_PROGRAM": .13},
    "BI_EARLY": {"BI_SOCKET_HEAT": .6, "GATE_OXIDE": .25, "ESD_HANDLING": .15},
    "SLT_HANG": {"POWER_SEQ": .55, "HBM_TRAINING": .3, "TIM_COVERAGE": .15},
    "RACK_LEAK": {"FITTING_TORQUE": .6, "QD_ORING": .25, "MANIFOLD_WELD": .15},
}

# 根因 → (多发机台, 概率)；其余概率均匀落在同工序其它机台
HOTSPOTS = {
    "ETCH_FOCUS_RING": ("ETC-02B", .7), "CMP_RETAINING_RING": ("CMP-01", .6), "ALD_EDGE_TEMP": ("DEP-02", .6),
    "CVD_SHOWERHEAD": ("DEP-03", .65), "IMP_DOSE_CENTER": ("IMP-02", .6), "WET_CENTER_RESIDUE": ("WET-01", .7),
    "CMP_DISK": ("CMP-03", .6), "ROBOT_SCRATCH": ("WET-02", .7), "PM_SEASONING": ("ETC-01A", .6),
    "RETICLE_HAZE": ("SCN-02", .6), "CVD_FLAKE": ("DEP-01", .6), "LENS_HEATING": ("SCN-01", .65),
    "SCANNER_MATCHING": ("SCN-03", .85), "WAFER_WARP": ("DEP-02", .5), "BOND_TEMP_UNIFORM": ("CWS-02", .6),
    "CURE_PROFILE": ("UF-01", .6), "REFLOW_PROFILE": ("FCB-02", .6), "UF_DISPENSE": ("UF-02", .65),
    "TCB_COPLANARITY": ("HBA-02", .7), "SOCKET_CONTACT": ("FTT-02", .75), "BI_SOCKET_HEAT": ("BI-01", .65),
    "POWER_SEQ": ("SLT-03", .6), "TIM_COVERAGE": ("UF-01", .5), "FITTING_TORQUE": ("RCK-02", .65),
}

DIFFICULTY = {"easy": (0.7, 1.0), "medium": (0.25, 0.45), "hard": (0.08, 0.2)}


@dataclass(frozen=True)
class Change:
    """一次产线变更：days_ago 天前生效；生效之前该症状的根因分布是 before（之后是 SITE_TRUTH 中的“现在”）。"""

    days_ago: int
    label: str
    symptom: str
    before: dict


# 过去一年的产线变更（虚构），按时间先后排列。days_ago 相对于演示数据写入的日期。
CHANGES = [
    Change(320, "测试厂 B 更换 FT 测试座供应商，接触电阻偏高", "TEST_VMIN",
           {"SOCKET_CONTACT": .25, "PROCESS_SLOW_CORNER": .55, "TEST_PROGRAM": .20}),
    Change(290, "CVD 腔体 DEP-03 换新喷淋头，中心偏厚问题减少", "WM_CENTER",
           {"WET_CENTER_RESIDUE": .35, "CVD_SHOWERHEAD": .50, "IMP_DOSE_CENTER": .15}),
    Change(262, "封测厂 B 切换 ABF 基板供应商（CTE 偏大）", "PKG_WARP",
           {"SUBSTRATE_CTE": .25, "BOND_TEMP_UNIFORM": .45, "CURE_PROFILE": .30}),
    Change(230, "底部填充胶换型号，点胶窗口变窄", "PKG_VOID",
           {"UF_DISPENSE": .25, "REFLOW_PROFILE": .50, "FLUX_RESIDUE": .25}),
    Change(205, "CMP 保持环改用长寿命型号后，刻蚀聚焦环成为边缘失效的主因", "WM_EDGE",
           {"ETCH_FOCUS_RING": .40, "CMP_RETAINING_RING": .35, "ALD_EDGE_TEMP": .15, "LITHO_EBR": .10}),
    Change(170, "整机 BMC 固件更新上电时序", "SLT_HANG",
           {"POWER_SEQ": .25, "HBM_TRAINING": .55, "TIM_COVERAGE": .20}),
    Change(143, "HBM3E 改为 12 层堆叠，TCB 共面度窗口收紧", "HBM_OPEN",
           {"TCB_COPLANARITY": .15, "NCF_LOT": .60, "HBM_INCOMING": .25}),
    Change(105, "液冷快接头换供应商，管路接头扭矩规范调整", "RACK_LEAK",
           {"FITTING_TORQUE": .30, "QD_ORING": .55, "MANIFOLD_WELD": .15}),
    Change(78, "老化板 BI-01 换用新型插座", "BI_EARLY",
           {"BI_SOCKET_HEAT": .25, "GATE_OXIDE": .50, "ESD_HANDLING": .25}),
    Change(41, "Fab D 新导入一台国产 DUV 扫描机 SCN-03", "OVL_DRIFT",
           {"SCANNER_MATCHING": .05, "LENS_HEATING": .55, "R2R_MODEL": .25, "WAFER_WARP": .15}),
    Change(11, "清洗机 WET-02 更换机械手后对位偏移", "WM_SCRATCH",
           {"ROBOT_SCRATCH": .30, "CMP_SLURRY_LPC": .45, "CMP_DISK": .25}),
]
CURRENT_ERA = len(CHANGES)


def era_at(days_ago: float) -> int:
    """days_ago 天前处在第几个时段（= 已生效的变更条数）。"""
    return sum(1 for c in CHANGES if days_ago <= c.days_ago)


def truth_for_era(era: int | None = None) -> dict[str, dict[str, float]]:
    """某个时段的真实根因分布：尚未生效的变更，对应症状用变更前的分布。"""
    era = CURRENT_ERA if era is None else era
    t = {s: dict(d) for s, d in SITE_TRUTH.items()}
    for c in reversed(CHANGES[era:]):
        t[c.symptom] = dict(c.before)
    return t


def _cause_obj(symptom: str, code: str):
    return next(c for c in CAUSES[symptom] if c.code == code)


def hotspot_tool(cause_code: str, symptom: str, rng: random.Random) -> str | None:
    c = _cause_obj(symptom, cause_code)
    if not c.tool_specific:
        return None
    tools = STEPS[c.step].tools
    hot = HOTSPOTS.get(cause_code)
    if hot and rng.random() < hot[1]:
        return hot[0]
    others = [t for t in tools if not hot or t != hot[0]] or list(tools)
    return rng.choice(others)


@dataclass
class TaskSpec:
    symptom: str
    difficulty: str = "medium"
    decoy_pm: bool = False
    variant: str = "信息量变体"  # 前置技能 / 信息量变体 / 失败驱动 / 压力测试
    rationale: str = ""

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Task:
    spec: TaskSpec
    seed: int
    observation: dict  # MES/YMS/FDC 数据（三个 agent 都可以看）
    symptom: str  # 观测到的症状（晶圆图特征可修正）
    requirements: dict  # 验收要求（验证 agent 依据）
    hidden: dict = field(default_factory=dict)  # 孪生内部：真因、机台、强度（任何 agent 不可见）


class FabTwin:
    def __init__(self, now: datetime | None = None, era: int | None = None, truth: dict | None = None):
        self.now = now or datetime.now(timezone.utc)
        self.era = CURRENT_ERA if era is None else era
        self.truth = truth or truth_for_era(self.era)  # truth：压力测试用的假想变更（孪生已同步、记忆还没跟上）

    def make_task(self, spec: TaskSpec, seed: int, force_cause: str | None = None) -> Task:
        """force_cause 仅供构造分层测试集使用；练习时根因永远按本厂真实分布抽取。"""
        rng = random.Random(seed * 7919 + 17)
        dist = self.truth[spec.symptom]
        cause = rng.choices(list(dist), weights=list(dist.values()))[0]
        if force_cause in dist:
            cause = force_cause
        tool = hotspot_tool(cause, spec.symptom, rng)
        lo, hi = DIFFICULTY[spec.difficulty]
        strength = rng.uniform(lo, hi)
        extras = {"tool": tool, "decoy_pm": spec.decoy_pm}
        ds = simulate(Scenario(spec.symptom, cause, strength=strength, seed=seed, n_lots=100, extras=extras), self.now)
        observed = spec.symptom
        metric_key = "yield" if SYMPTOMS[spec.symptom].stage == "fe" else "fail_rate"
        lots = ds["lots"]
        k = len(lots) * 2 // 5
        baseline = mean(l[metric_key] for l in lots[:k])
        return Task(spec=spec, seed=seed, observation=ds, symptom=observed,
                    requirements={"metric": metric_key, "baseline": round(baseline, 3), "higher_is_better": metric_key == "yield",
                                  "min_recovery": 0.6, "capa_kinds": ["containment", "corrective", "preventive"]},
                    hidden={"cause": cause, "tool": tool, "strength": round(strength, 3), "extras": extras})

    def apply_fix(self, task: Task, cause_code: str) -> dict:
        """干预反馈：按提议的根因在孪生中整改，重跑同一时段，返回整改前后最近窗口的指标（不返回答案）。"""
        fixed = cause_code == task.hidden["cause"]
        ds = simulate(Scenario(task.spec.symptom, task.hidden["cause"], strength=task.hidden["strength"], seed=task.seed,
                               n_lots=100, extras={**task.hidden["extras"], "fixed": fixed}), self.now)
        key = task.requirements["metric"]
        k = len(ds["lots"]) * 3 // 5
        before = mean(l[key] for l in task.observation["lots"][k:])
        after = mean(l[key] for l in ds["lots"][k:])
        rng = random.Random(task.seed + zlib.crc32(cause_code.encode()) % 1000)
        after += rng.gauss(0, 0.04 if key == "yield" else 0.01)  # 量测噪声：验证本身也有不确定性
        return {"metric": key, "before": round(before, 3), "after": round(after, 3), "baseline": task.requirements["baseline"]}


def observed_symptom(task_symptom: str, evidence: dict) -> str:
    if SYMPTOMS[task_symptom].wafer_pattern and evidence.get("wafer_pattern") in PATTERN_TO_SYMPTOM:
        return PATTERN_TO_SYMPTOM[evidence["wafer_pattern"]]
    return task_symptom


EVAL_PER_SYMPTOM = 40  # 测试集 480 题：p≈0.6 时标准误约 2.2 个百分点
GATE_PER_SYMPTOM = 10  # 验收门面板 120 题
_CACHE: dict[str, list[dict]] = {}


def _stratified(per: int, seed0: int, era: int | None = None) -> list[dict]:
    twin, out = FabTwin(datetime(2026, 11, 1, tzinfo=timezone.utc), era=era), []
    for si, (sym, dist) in enumerate(twin.truth.items()):
        quota = {c: max(1, round(per * p)) for c, p in dist.items()}
        j = 0
        for cause, n in quota.items():
            for _ in range(n):
                diff = "medium" if j % 2 == 0 else "hard"
                t = twin.make_task(TaskSpec(sym, diff), seed=seed0 + si * 100 + j, force_cause=cause)
                ev = analyze_dataset(t.observation)
                out.append({"symptom": observed_symptom(sym, ev), "truth": cause, "evidence": ev, "difficulty": diff})
                j += 1
    return out


def eval_set(era: int | None = None) -> list[dict]:
    """留出测试集（分层）：每个症状 40 道题，根因按本厂真实分布配比，中等/困难各半；种子与练习、验收门都不重叠。

    分层是为了让分数代表“在本厂真实分布下的期望命中率”，而不是少数几道随机题的运气。测试集只用于报告，
    任何学习步骤（包括验收门）都看不到它。
    """
    key = f"eval:{CURRENT_ERA if era is None else era}"
    if key not in _CACHE:
        _CACHE[key] = _stratified(EVAL_PER_SYMPTOM, 900_000, era)
    return _CACHE[key]


def gate_panel(era: int | None = None) -> list[dict]:
    """验收门面板（封存的验证集，与测试集不重叠）：每次想更新证据权重，都必须在这里“不退步”才会被采纳。

    依据：OpsHarness（arXiv 2608.25661）中去掉验收门的自我进化版本比不进化还差；SEAL（2607.24300）同理。
    """
    key = f"gate:{CURRENT_ERA if era is None else era}"
    if key not in _CACHE:
        _CACHE[key] = _stratified(GATE_PER_SYMPTOM, 960_000, era)
    return _CACHE[key]


assert all(set(c.before) == set(SITE_TRUTH[c.symptom]) for c in CHANGES)  # noqa: S101
assert [c.days_ago for c in CHANGES] == sorted((c.days_ago for c in CHANGES), reverse=True)  # noqa: S101
