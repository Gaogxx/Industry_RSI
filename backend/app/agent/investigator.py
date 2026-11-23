"""代码模式调查员（Pydantic AI + pydantic-ai-harness CodeMode + Monty 沙箱）。

大模型不再一次次发起工具调用，而是写一段 Python，在隔离沙箱里调用下面这些“类型化数据工具”，
自由地切分批次履历、比较 FDC、核对晶圆图 —— 大表格不进上下文，只有结论回到模型。
沙箱无文件系统、无网络、有资源上限；调查结论以“有界因子”(×1.3 / ×0.7) 透明地并入贝叶斯排序。

仅在配置了大模型时运行；离线模式下由统计内核独立完成分析。
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev

from pydantic_ai import Agent, RunContext
from pydantic_ai_harness import CodeMode

from ..llm import LLM
from . import prompts
from .schemas import InvestigationOut


@dataclass
class FabDeps:
    data: dict
    hypotheses: list[dict]


def build_investigator(llm: LLM) -> Agent[FabDeps, InvestigationOut]:
    agent: Agent[FabDeps, InvestigationOut] = Agent(
        llm.model(), deps_type=FabDeps, output_type=InvestigationOut, retries=2,
        instructions="你是半导体良率调查员，只根据工具返回的数据下结论。", capabilities=[CodeMode(max_tool_calls=40)])

    @agent.tool
    def hypotheses(ctx: RunContext[FabDeps]) -> list[dict]:
        """系统当前的贝叶斯根因假设：cause_code、name、step（工序）、posterior（后验概率）。"""
        return [{"cause_code": h["cause_code"], "name": h["name"], "step": h["step"], "posterior": h["posterior"]}
                for h in ctx.deps.hypotheses]

    @agent.tool
    def factors(ctx: RunContext[FabDeps]) -> list[str]:
        """批次履历中可切分的因子（工序 code，或“工序#物料批次”）。"""
        return sorted({k for lot in ctx.deps.data["lots"] for k in lot["steps"]})

    @agent.tool
    def fail_rate_by(ctx: RunContext[FabDeps], factor: str) -> list[dict]:
        """按某因子（如 'ETCH'）统计每台机台/批次的批次数 n、失效数 fails、失效率 rate。"""
        groups: dict[str, list[bool]] = {}
        for lot in ctx.deps.data["lots"]:
            if factor in lot["steps"]:
                groups.setdefault(lot["steps"][factor], []).append(lot["fail"])
        return sorted(({"value": v, "n": len(x), "fails": sum(x), "rate": round(sum(x) / len(x), 3)} for v, x in groups.items()),
                      key=lambda r: -r["rate"])

    @agent.tool
    def fail_rate_before_after(ctx: RunContext[FabDeps], factor: str, value: str, split_time: str) -> dict:
        """某机台/批次在 split_time（ISO 时间）前后的失效率，用于验证“PM 后突变”等时序假设。"""
        seq = [lot for lot in ctx.deps.data["lots"] if lot["steps"].get(factor) == value]
        before = [lot["fail"] for lot in seq if lot["time"] < split_time]
        after = [lot["fail"] for lot in seq if lot["time"] >= split_time]
        rate = lambda x: round(sum(x) / len(x), 3) if x else None  # noqa: E731
        return {"n_before": len(before), "rate_before": rate(before), "n_after": len(after), "rate_after": rate(after)}

    @agent.tool
    def fdc_summary(ctx: RunContext[FabDeps], tool: str) -> list[dict]:
        """某机台各 FDC 参数前半段 vs 后半段的均值与偏移（σ）。"""
        rows = ctx.deps.data.get("fdc", {}).get(tool) or []
        if len(rows) < 6:
            return []
        out = []
        half = len(rows) // 2
        for p in (k for k in rows[0] if k not in ("lot", "time")):
            a, b = [r[p] for r in rows[:half]], [r[p] for r in rows[half:]]
            sd = pstdev(a) or 1e-9
            out.append({"param": p, "mean_first": round(mean(a), 3), "mean_second": round(mean(b), 3),
                        "shift_sigma": round((mean(b) - mean(a)) / sd, 2)})
        return out

    @agent.tool
    def pm_events(ctx: RunContext[FabDeps]) -> list[dict]:
        """PM（预防性维护）事件：tool、time。"""
        return ctx.deps.data.get("events", [])

    @agent.tool
    def wafer_patterns(ctx: RunContext[FabDeps]) -> list[dict]:
        """失效晶圆的空间特征分类结果（edge/center/scratch/cluster/random）及特征量。"""
        from ..fab.analytics import wafer_signature

        return [{"wafer": w["wafer"], **wafer_signature(w["map"])} for w in ctx.deps.data.get("wafers", []) if w.get("fail")]

    return agent


async def investigate(llm: LLM, case: dict, symptom_name: str, data: dict, hypotheses: list[dict]) -> InvestigationOut | None:
    if not llm.available or not data:
        return None
    try:
        agent = build_investigator(llm)
        result = await agent.run(prompts.INVESTIGATE.format(title=case["title"], symptom=symptom_name),
                                 deps=FabDeps(data=data, hypotheses=hypotheses))
        llm.usage.add(result.usage)
        return result.output
    except Exception as e:
        llm.usage.errors.append(f"investigator: {str(e)[:160]}")
        return None


def fuse(hypotheses: list[dict], inv: InvestigationOut) -> list[dict]:
    """把调查结论作为有界似然因子并入后验（×1.3 支持 / ×0.7 反驳），并重新归一化。"""
    codes = {h["cause_code"] for h in hypotheses}
    sup, con = set(inv.supports) & codes, set(inv.contradicts) & codes
    for h in hypotheses:
        f = 1.3 if h["cause_code"] in sup else 0.7 if h["cause_code"] in con else 1.0
        if f != 1.0:
            h["posterior"] *= f
            h["factors"].append({"name": "调查 Agent", "x": f, "detail": "代码模式数据核查" + ("支持" if f > 1 else "反驳")})
    z = sum(h["posterior"] for h in hypotheses) or 1
    for h in hypotheses:
        h["posterior"] = round(h["posterior"] / z, 4)
    return sorted(hypotheses, key=lambda h: -h["posterior"])
