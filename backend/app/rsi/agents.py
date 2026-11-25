"""RSIAgent 的三个角色（职责分离，互相制衡）：

  课程 agent（Curriculum）  独占“决定练什么”：看知识缺口与上一轮结果出题，判断何时停止
  执行 agent（Executor）    独占“写记忆”：用当前记忆做根因分析并提出整改；只在验证判定后合并经验
  验证 agent（Verifier）    独占“判真假”：只看环境反馈（整改前后指标）与验收要求，看不到执行 agent 的推理与记忆

三者都按 TTSE 使用经验（不改参数、在测试时靠经验进化）：执行 agent 作答时检索“稳定”的做法条目（tip），
课程 agent 读自己“失败后加练是否有用”的做法条目，验证 agent 是唯一的裁判——条目的成败只由它的判定决定，
失败追责到当时实际用到的条目（见 rsi/memory.py）。

三者都有确定性内核（离线可用）。配置大模型后，课程 agent 可由与执行不同的模型担任（论文：GLM-5.3 执行，
Kimi-K3 负责课程与验证），以降低相关性错误；验证 agent 刻意保持为基于环境反馈的确定性判定——
论文指出“验证不完整”是递归自我改进的第一大失败来源，而本系统的环境反馈是数值化的，规则判定比 LLM 更可靠。
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field

from ..fab.analytics import analyze_dataset
from ..fab.flow import CAUSES, STAGES, SYMPTOMS, cause_by_code
from ..fab.kg import rank, share
from ..llm import LLM
from .memory import MemoryState
from .twin import SITE_TRUTH, Task, TaskSpec, observed_symptom


# ── 课程 agent ───────────────────────────────────────────
class CurriculumPick(BaseModel):
    symptom: str
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    decoy_pm: bool = False
    variant: Literal["前置技能", "信息量变体", "失败驱动", "压力测试", "证据校准", "变更校准", "近邻练习", "训练池"] = "信息量变体"
    rationale: str = ""


class CurriculumOut(BaseModel):
    tasks: list[CurriculumPick] = Field(default_factory=list)
    stop: bool = False
    reason: str = ""


def _compositions(n: int, k: int):
    if k == 1:
        yield (n,)
        return
    for i in range(n + 1):
        for rest in _compositions(n - i, k - 1):
            yield (i, *rest)


def knowledge_gradient(table: dict, weight: float = 0.5, horizons: tuple[int, ...] = (1, 2, 4, 8)) -> float:
    """知识梯度 KG(*)：再练 n 道该症状的题，“按先验选最可能根因”的期望正确率平均每题提高多少，取 n 上的最大值。

    只看 n=1 不够：一个样本常常翻不动当前最优选择，KG₁=0，但几道题累积起来就可能翻转（信息价值的非凹性，
    Frazier & Powell 2010）。期望在 Dirichlet-多项后验预测下精确枚举计算。
    """
    if not table:
        return 0.0
    alpha = [e.alpha for e in table.values()]
    a0 = sum(alpha) or 1.0
    now = max(alpha) / a0
    best = 0.0
    for n in horizons:
        exp_after = 0.0
        for counts in _compositions(n, len(alpha)):  # 狄利克雷-多项分布：n 道题里各根因出现的次数
            lp = math.lgamma(n + 1) + math.lgamma(a0) - math.lgamma(a0 + n)
            for c, a in zip(counts, alpha):
                lp += math.lgamma(c + a) - math.lgamma(a) - math.lgamma(c + 1) if a > 0 else (0.0 if c == 0 else -1e9)
            exp_after += math.exp(lp) * max((a + weight * c) / (a0 + weight * n) for a, c in zip(alpha, counts))
        best = max(best, (exp_after - now) / n)
    return max(0.0, best)


def weak_rate(history: list[dict], sym: str) -> float:
    """该症状练习题中“证据模糊”的比例（证据越模糊，先验越重要）；没练过时取 0.6。"""
    mine = [t for t in history if t["spec"]["symptom"] == sym and "weak" in t]
    return sum(t["weak"] for t in mine) / len(mine) if mine else 0.6


def knowledge_gaps(mem: MemoryState, history: list[dict]) -> list[dict]:
    out = []
    for sym in SITE_TRUTH:
        table = mem.edges.get(sym, {})
        shares = [share(table, c) for c in table] if table else [1.0]
        h = -sum(p * math.log(p) for p in shares if p > 0) / math.log(max(2, len(shares)))
        n_real = sum(e.confirmed for e in table.values())
        n_sb = sum(e.confirmed_sandbox for e in table.values())
        mine = [t for t in history if t["spec"]["symptom"] == sym]
        fails = sum(1 for t in mine if t["verdict"] != "PASS")
        kg = knowledge_gradient(table)
        voi = weak_rate(history, sym) * kg  # 期望命中率增益 ≈ 证据模糊的概率 × 知识梯度
        gap = voi * 100 + 0.2 / (1 + len(mine)) + 0.4 * (fails / len(mine) if mine else 0)
        out.append({"symptom": sym, "name": SYMPTOMS[sym].name, "stage": SYMPTOMS[sym].stage, "entropy": round(h, 3),
                    "real": n_real, "sandbox": n_sb, "practiced": len(mine), "fails": fails,
                    "kg": round(kg, 4), "voi": round(voi * 100, 3), "gap": round(gap, 3)})
    return sorted(out, key=lambda g: -g["gap"])


class Curriculum:
    def __init__(self, llm: LLM | None = None, max_depth: int = 12, patience: int = 3):
        self.llm, self.max_depth, self.patience = llm, max_depth, patience
        self.stop_reason = ""

    async def breadth(self, mem: MemoryState, history: list[dict], n: int = 8) -> list[TaskSpec]:
        gaps = knowledge_gaps(mem, history)
        by_stage: dict[str, list[dict]] = {}
        for g in gaps:
            by_stage.setdefault(g["stage"], []).append(g)
        order, picked = list(by_stage), []
        while len(picked) < n and any(by_stage.values()):  # 按工段轮转，保证广度覆盖
            for st in sorted(order, key=lambda s: -(by_stage[s][0]["gap"] if by_stage[s] else -1)):
                if by_stage[st] and len(picked) < n:
                    picked.append(by_stage[st].pop(0))
        specs = []
        for g in picked:
            first = g["real"] + g["sandbox"] == 0
            specs.append(TaskSpec(g["symptom"], "easy" if first else "medium", False, "前置技能" if first else "信息量变体",
                                  f"{g['name']}：不确定性 {g['entropy']:.2f}，产线确认 {g['real']} 次、孪生确认 {g['sandbox']} 次"))
        return await self._llm_refine(specs, gaps, "breadth", n) or specs

    async def depth(self, mem: MemoryState, history: list[dict], step: int) -> TaskSpec | None:
        drs = [t for t in history if t["phase"] == "DRS"]
        if step >= self.max_depth:
            self.stop_reason = f"已达到深度探索预算上限（{self.max_depth} 题）"
            return None
        if len(drs) >= self.patience and all(t["delta"] < 0.02 and t["new_items"] == 0 for t in drs[-self.patience:]):
            self.stop_reason = f"连续 {self.patience} 道题没有带来实质新知识，课程 agent 判断探索结束"
            return None  # 论文的停止条件
        last = history[-1] if history else None
        drill = last and last["verdict"] != "PASS"
        if drill:
            tip = mem.items.get(f"tip:curriculum:drill:{last['spec']['symptom']}")
            drill = tip is None or tip["status"] != "retired"  # TTSE：这条做法在该症状上停用了就不再加练，改按信息价值出题
        if drill:
            sym = last["spec"]["symptom"]
            spec = TaskSpec(sym, "hard" if last["spec"]["difficulty"] != "hard" else "medium", False, "失败驱动",
                            f"上一题「{SYMPTOMS[sym].name}」未通过验证，针对同一症状加练")
        else:
            g = knowledge_gaps(mem, history)[0]
            decoy = step % 2 == 1
            spec = TaskSpec(g["symptom"], "hard", decoy, "压力测试",
                            f"{g['name']} 的信息价值最高（期望命中率增益 {g['voi']:.2f} 个百分点/题）"
                            + ("；加入无关机台 PM 干扰" if decoy else "；证据更弱"))
        refined = await self._llm_refine([spec], knowledge_gaps(mem, history), "depth", 1)
        return refined[0] if refined else spec

    def calibration(self, mem: MemoryState, history: list[dict], n: int, rng, focus: set[str] | None = None) -> list[TaskSpec]:
        """校准批：按信息价值分配症状，难度混合（易 2 : 中 4 : 难 4），主要用于学习“每类证据该信多少”。

        focus：上次会话之后发生产线变更的症状，出题比例提高到 4 倍（这部分校准题也会更新先验）。
        """
        gaps = knowledge_gaps(mem, history)
        weights = [(g["voi"] + 0.5) * (4 if focus and g["symptom"] in focus else 1) for g in gaps]
        specs = []
        for i in range(n):
            g = rng.choices(gaps, weights=weights)[0]
            diff = rng.choices(["easy", "medium", "hard"], weights=[2, 4, 4])[0]
            changed = bool(focus and g["symptom"] in focus)
            specs.append(TaskSpec(g["symptom"], diff, rng.random() < 0.2, "变更校准" if changed else "证据校准",
                                  f"{'产线变更后重新校准' if changed else '校准批：学习各类证据的可信度'}（{g['name']}，信息价值 {g['voi']:.2f}）"))
        return specs

    def test_time(self, symptom: str, evidence: dict, n: int, rng) -> list[TaskSpec]:
        """测试时出题（与 TTT 互补）：针对眼前这个案例出“近邻练习”——同一症状、相近的证据强度与干扰。

        常规课程决定长期该练什么（面向全部症状的知识缺口）；近邻练习只为当前这一案服务，练完即弃，留下的是经验。
        """
        from .memory import condition_of

        cond = condition_of(evidence or {})
        weights = [3, 4, 3] if cond["evidence"] == "strong" else [1, 4, 5]
        out = []
        for i in range(n):
            diff = rng.choices(["easy", "medium", "hard"], weights=weights)[0]
            decoy = cond["after_pm"] and rng.random() < 0.5
            out.append(TaskSpec(symptom, diff, decoy, "近邻练习",
                                f"测试时训练：与当前案例相近的「{SYMPTOMS[symptom].name if symptom in SYMPTOMS else symptom}」"
                                f"（{'共性分析显著' if cond['evidence'] == 'strong' else '证据模糊'}{'，PM 后突变' if cond['after_pm'] else ''}）"))
        return out

    def training_pool(self, per_symptom: int, mix: dict, rng, focus: set[str] | None = None,
                      symptoms: list[str] | None = None) -> list[TaskSpec]:
        """快脑小模型的训练池（参照 A-RSI：按题族固定配额出题，标签由孪生程序给出）。

        每个症状固定 per_symptom 道题；难度与干扰比例来自训练策略（mix: hard_share / decoy_share）；
        focus = 训练手册里“产线变更后分布变了”的症状，配额加倍。与常规课程不同，这里不按信息价值挑症状——
        固定配额保证每个窗口的样本池可比，换池与否由“仍有信息量的样本占比”决定。
        """
        hard, decoy = float(mix.get("hard_share", 0.5)), float(mix.get("decoy_share", 0.33))
        out = []
        for sym in symptoms or list(SITE_TRUTH):
            n = per_symptom * (2 if focus and sym in focus else 1)
            for i in range(n):
                diff = "hard" if rng.random() < hard else rng.choice(["easy", "medium", "medium"])
                out.append(TaskSpec(sym, diff, rng.random() < decoy, "训练池",
                                    f"快脑训练池：{SYMPTOMS[sym].name}" + ("（产线变更后加倍配额）" if focus and sym in focus else "")))
        return out

    async def _llm_refine(self, specs: list[TaskSpec], gaps: list[dict], phase: str, n: int) -> list[TaskSpec] | None:
        if not (self.llm and self.llm.available):
            return None
        out = await self.llm.structured(CurriculumOut, "你是良率工程的课程设计者，为根因分析智能体安排数字孪生练习题。只能从给定症状中选择。",
                                        f"阶段：{phase}（{'广度：多方向并行' if phase == 'breadth' else '深度：针对缺口与失败加难'}）\n"
                                        f"知识缺口表：{gaps}\n规则给出的候选：{[s.as_dict() for s in specs]}\n请给出 {n} 道题。")
        if not out or not out.tasks:
            return None
        valid = [TaskSpec(t.symptom, t.difficulty, t.decoy_pm, t.variant, t.rationale) for t in out.tasks if t.symptom in SITE_TRUTH]
        return valid[:n] or None


# ── 执行 agent ───────────────────────────────────────────
class Executor:
    TIP_P, TIP_RATIO = 0.8, 0.5

    def pick(self, hyps: list, mem: MemoryState, sym: str, ev: dict, exclude: set[str]) -> tuple:
        """TTSE 的测试时检索：排在前三、后验不低于首位一半的候选里，若有“同条件下稳定有效”的做法条目（P ≥ 0.8），
        且比首位自己的做法条目更可靠，就先试它；否则按排序。返回（候选, 用到的条目）。"""
        from .memory import condition_of, item_p

        cands = [h for h in hyps if h.cause_code not in exclude] or hyps
        ev_cond = condition_of(ev)["evidence"]
        tip_of = lambda h: mem.items.get(f"proc:{sym}:{h.cause_code}:{ev_cond}")  # noqa: E731
        top = cands[0]
        base = tip_of(top)
        base_p = item_p(base) if base and base["status"] == "stable" else 0.0
        best = None
        for h in cands[1:3]:
            it = tip_of(h)
            if it and it["status"] == "stable" and item_p(it) >= self.TIP_P and item_p(it) > base_p \
                    and h.posterior >= self.TIP_RATIO * top.posterior and (best is None or item_p(it) > item_p(best[1])):
                best = (h, it)
        if best:
            return best[0], {"key": best[1]["key"], "text": best[1]["text"], "p": round(item_p(best[1]), 3)}
        return top, None

    def attempt(self, task: Task, mem: MemoryState, exclude: set[str], evidence: dict | None = None) -> dict:
        ev = evidence or analyze_dataset(task.observation)
        sym = observed_symptom(task.spec.symptom, ev)
        hyps = rank(ev, sym, mem.edges, mem.hotspot_weights(), mem.weights)
        pick, tip = self.pick(hyps, mem, sym, ev, exclude)
        found = cause_by_code(pick.cause_code)
        tpl = found[1].capa if found else ("隔离可疑批次", "针对根因整改", "纳入 FMEA")
        capa = [{"kind": k, "title": t} for k, t in zip(("containment", "corrective", "preventive"), tpl)]
        return {"symptom": sym, "evidence": ev, "cause_code": pick.cause_code, "cause": pick.name,
                "posterior": round(pick.posterior, 3), "ranking": [(h.cause_code, round(h.posterior, 3)) for h in hyps[:4]],
                "capa": capa, "memory_used": [i["key"] for i in mem.active_items(sym)[:3]], "tip": tip}


# ── 验证 agent ───────────────────────────────────────────
class Verifier:
    """只接收：验收要求 + 环境反馈 + 措施类别。看不到执行 agent 的排序、理由和记忆。"""

    def judge(self, requirements: dict, feedback: dict, capa_kinds: set[str]) -> dict:
        hi = requirements["higher_is_better"]
        b, a, base = feedback["before"], feedback["after"], feedback["baseline"]
        gap = (base - b) if hi else (b - base)
        gain = (a - b) if hi else (b - a)
        floor = 0.08 if hi else 0.02  # 低于量测噪声的改善无法确认
        recovered = gain / gap if gap > floor else 0.0
        capa_ok = set(requirements["capa_kinds"]) <= capa_kinds
        conclusive = gap > floor
        passed = conclusive and recovered >= requirements["min_recovery"] and capa_ok
        unit = "%" if hi else "%"
        evidence = [f"整改前 {b:.2f}{unit} → 整改后 {a:.2f}{unit}（基线 {base:.2f}{unit}）",
                    f"恢复比例 {recovered:.0%}（要求 ≥ {requirements['min_recovery']:.0%}）" if conclusive
                    else "异常幅度低于量测噪声，无法确认恢复 → 不予通过（避免固化未验证经验）",
                    "围堵/纠正/预防措施齐全" if capa_ok else "措施类别不全"]
        return {"passed": passed, "recovered": round(recovered, 3), "conclusive": conclusive, "evidence": evidence}



assert CAUSES and STAGES  # noqa: S101
