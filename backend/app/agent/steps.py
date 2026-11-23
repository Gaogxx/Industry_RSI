"""分析流程的各个步骤（与编排框架无关的纯异步函数）。

    受理 → 取数与统计证据 → 贝叶斯根因推断（含“若不学习”反事实）→ 调取经验手册
        → 解释（LLM / 规则）→ CAPA → 评审（不过则带意见重写解释，最多 2 轮）→ 8D 报告

每步接收/返回普通 dict，可被 DBOS 持久化工作流、测试或任何调度器复用。
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable

from sqlmodel import col, select

from ..db import session_scope
from ..fab.analytics import PATTERN_TO_SYMPTOM, analyze_dataset
from ..fab.flow import CAUSES, STAGES, STEPS, SYMPTOMS, cause_by_code, domain_context, guess_symptom
from ..fab.kg import entropy
from ..fab.learning import memory_for, rank_case
from ..llm import LLM
from ..models import ACTION_KINDS, SIX_M, Case, FabDataset
from . import prompts
from .schemas import CapaOut, CriticOut, ExplainOut, IntakeOut

Emit = Callable[[str, str, str], None]


def _j(o) -> str:
    return json.dumps(o, ensure_ascii=False, default=str)


def _fmt_time(v) -> str:
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo

    if isinstance(v, str):
        try:
            v = datetime.fromisoformat(v)
        except ValueError:
            return v
    if isinstance(v, datetime):
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M")
    return "待补充"


# ── 1. 受理 ───────────────────────────────────────────────
async def intake(state: dict, llm: LLM, emit: Emit) -> dict:
    c = state["case"]
    emit("intake", "受理案例", f"整理 5W2H、判定症状：{c['title']}")
    symptom = c.get("symptom") if c.get("symptom") in SYMPTOMS else guess_symptom(f"{c['title']} {c['description']}")
    res = await llm.structured(IntakeOut, prompts.SYSTEM_BASE, prompts.INTAKE.format(
        title=c["title"], description=c["description"], site=c.get("site"), product=c.get("product"),
        data=_j(c.get("data") or {}), symptoms="\n".join(f"{k}：{v.name}" for k, v in SYMPTOMS.items())))
    out = res.model_dump() if res else None
    data = c.get("data") or {}
    problem = {
        "problem_statement": f"{c.get('site') or '现场'}「{c['title']}」：{SYMPTOMS[symptom].metric}异常，需定位根因并圈定影响范围。",
        "what": c["title"], "where": " / ".join(x for x in [c.get("site"), c.get("line"), c.get("equipment")] if x) or "待补充",
        "when": _fmt_time(c.get("created_at")), "who": c.get("reporter") or "良率/质量工程师",
        "how_many": "；".join(f"{k}={v}" for k, v in list(data.items())[:3]) or "待统计",
        "how_detected": "YMS / SPC / FDC 报警", "impact": f"损失约 {c.get('cost_impact', 0)} 万元",
    }
    if out:
        problem.update({k: v for k, v in out.items() if k in problem and v})
        if out.get("symptom") in SYMPTOMS and not c.get("symptom"):
            symptom = out["symptom"]
    return {"problem": problem, "symptom": symptom, "severity": (out or {}).get("severity") or c.get("severity")}


# ── 2. 取数与统计证据 ─────────────────────────────────────
async def evidence(state: dict, emit: Emit) -> dict:
    c = state["case"]
    with session_scope() as db:
        ds = db.get(FabDataset, c["id"])
        data = ds.data if ds else None
    if not data:
        emit("evidence", "统计证据", "该案例未关联 MES/YMS/FDC 数据，仅基于描述与历史知识推断")
        return {"evidence": c.get("evidence") or {}, "symptom": state["symptom"]}
    emit("evidence", "取数", f"批次 {len(data['lots'])} 个 · 晶圆图 {len(data.get('wafers', []))} 张 · "
                             f"FDC 机台 {len(data.get('fdc', {}))} 台 · PM 事件 {len(data.get('events', []))} 条")
    ev = analyze_dataset(data)
    symptom = state["symptom"]
    if ev.get("wafer_pattern") in PATTERN_TO_SYMPTOM and SYMPTOMS[symptom].wafer_pattern:
        symptom = PATTERN_TO_SYMPTOM[ev["wafer_pattern"]]  # 晶圆图空间特征优先于文字描述
        emit("evidence", "晶圆图特征", f"{SYMPTOMS[symptom].name}（置信 {ev['wafer_pattern_conf']:.0%}）")
    top = (ev.get("suspects") or [{}])[0]
    if top.get("score", 0) >= 1.3:
        emit("evidence", "共性分析", f"{top['value']}（{STEPS.get(top['step']).name if top['step'] in STEPS else top['step']}）"
                                 f"失效率 {top['rate_with']:.0%} vs 其余 {top['rate_without']:.0%}，p_adj={top['p_adj']}")
    else:
        emit("evidence", "共性分析", "无显著机台/批次富集 —— 证据模糊，将更多依赖已学习的知识")
    if ev.get("change_point"):
        cp = ev["change_point"]
        emit("evidence", "变点", f"{cp['value']} 自 {_fmt_time(cp['since'])} 恶化" + ("，紧随 PM" if cp["after_pm"] else ""))
    for f in ev.get("fdc", [])[:2]:
        emit("evidence", "FDC", f"{f['tool']} {f['param']} 偏移 {f['z']:+.1f}σ")
    return {"evidence": ev, "symptom": symptom}


# ── 3. 贝叶斯根因推断 ─────────────────────────────────────
def _eig(hyps: list[dict]) -> list[dict]:
    """若对某个假设做一次决定性验证，期望能减少多少不确定性（比特）。"""
    ps = [h["posterior"] for h in hyps]
    H = -sum(p * math.log2(p) for p in ps if p > 0)
    out = []
    for i, h in enumerate(hyps[:3]):
        rest = [p for j, p in enumerate(ps) if j != i]
        z = sum(rest) or 1
        h_rest = -sum((p / z) * math.log2(p / z) for p in rest if p > 0)
        out.append({"cause_code": h["cause_code"], "name": h["name"], "eig": round(H - (1 - h["posterior"]) * h_rest, 3)})
    return sorted(out, key=lambda x: -x["eig"])


async def infer(state: dict, emit: Emit) -> dict:
    with session_scope() as db:
        learned, frozen = rank_case(db, state["evidence"], state["symptom"])
    hyps = [h.as_dict() for h in learned]
    top, ftop = learned[0], frozen[0]
    emit("infer", "贝叶斯根因排序", "；".join(f"{h.name[:14]} {h.posterior:.0%}" for h in learned[:3]))
    cf = {"top": ftop.cause_code, "top_name": ftop.name, "top_posterior": round(ftop.posterior, 3),
          "learned_top_posterior_if_frozen": round(next(h.posterior for h in frozen if h.cause_code == top.cause_code), 3),
          "changed": ftop.cause_code != top.cause_code,
          "entropy_learned": entropy(learned), "entropy_frozen": entropy(frozen),
          "ranking_frozen": [{"cause_code": h.cause_code, "name": h.name, "posterior": round(h.posterior, 3)} for h in frozen]}
    if cf["changed"]:
        emit("infer", "持续学习的影响", f"若只用教科书先验，首位会是「{ftop.name}」({ftop.posterior:.0%})；"
                                       f"结合本厂已验证案例，首位为「{top.name}」({top.posterior:.0%})")
    else:
        emit("infer", "持续学习的影响", f"首位一致；已学习知识使其概率 {cf['learned_top_posterior_if_frozen']:.0%} → {top.posterior:.0%}")
    tests = _eig(hyps)
    if top.posterior < 0.6:
        emit("infer", "建议鉴别验证", f"根因尚不确定，优先验证「{tests[0]['name']}」（预期信息增益 {tests[0]['eig']} bit）")
    return {"hypotheses": hyps, "counterfactual": cf, "tests": tests}


# ── 4. 经验手册与相似案例 ─────────────────────────────────
async def knowledge(state: dict, emit: Emit) -> dict:
    sym = state["symptom"]
    with session_scope() as db:
        bullets = [{"id": b.id, "text": b.text, "helpful": b.passes, "harmful": b.fails, "cause_code": b.cause_code,
                    "kind": b.kind, "source": b.source} for b in memory_for(db, sym)]
        similar = db.exec(select(Case).where(Case.symptom == sym, Case.status == "closed", Case.id != state["case"]["id"])
                          .order_by(col(Case.closed_at).desc()).limit(5)).all()
        sims = [{"id": s.id, "code": s.code, "title": s.title, "cause_code": s.cause_code, "root_cause": s.root_cause}
                for s in similar]
    emit("knowledge", "调取记忆", f"可进化记忆 {len(bullets)} 条（含孪生验证 {sum(1 for b in bullets if b['source'] != 'real')} 条）· 同症状已关闭案例 {len(sims)} 个")
    return {"playbook": bullets, "similar": sims}


# ── 5. 解释 ──────────────────────────────────────────────
def _offline_explain(state: dict) -> dict:
    ev, hyps, sym = state["evidence"], state["hypotheses"], SYMPTOMS[state["symptom"]]
    top = hyps[0]
    found = cause_by_code(top["cause_code"])
    summary = ev.get("summary", {})
    whys = [f"{sym.name}：{sym.metric} 异常（失效批次 {summary.get('fail_lots', '?')}/{summary.get('lots', '?')}）"]
    s = next((x for x in ev.get("suspects", []) if x["step"] == top["step"] and x["score"] >= 1.3), None)
    if s:
        whys.append(f"失效集中在 {s['value']}（{STEPS[s['step']].name if s['step'] in STEPS else s['step']}），"
                    f"失效率 {s['rate_with']:.0%} vs 其余 {s['rate_without']:.0%}")
    else:
        whys.append("各机台/批次失效率无显著差异，依据本厂已验证案例的先验判断")
    cp = ev.get("change_point")
    if cp:
        whys.append(f"自 {_fmt_time(cp['since'])} 起恶化" + ("，紧随该机台 PM" if cp["after_pm"] else ""))
    f = next((x for x in ev.get("fdc", []) if found and x["param"] in found[1].fdc), None)
    whys.append(f"FDC：{f['param']} 偏移 {f['z']:+.1f}σ，指向「{top['name']}」" if f else f"机理：{top['name']}")
    if found and found[1].capa[2]:
        whys.append(f"系统性根因：现行管控未覆盖「{found[1].capa[2]}」，问题未能提前预警")
    ish = {k: [] for k in SIX_M}
    for h in hyps:
        ish.setdefault(h["six_m"], []).append(f"{h['name']}（{h['posterior']:.0%}）")
    narrative = f"综合统计证据与已学习知识，首位根因为「{top['name']}」，后验概率 {top['posterior']:.0%}。"
    return {"five_whys": whys, "ishikawa": ish, "narrative": narrative, "risks": []}


async def explain(state: dict, llm: LLM, emit: Emit) -> dict:
    it = state.get("iteration", 0) + 1
    emit("explain", f"解释与 5-Why（第 {it} 轮）", "LLM 解释证据链" if llm.available else "规则引擎生成证据链")
    base = _offline_explain(state)
    fb = ""
    if it > 1 and state.get("critique", {}).get("issues"):
        fb = "\n## 上一轮评审意见（请逐条修正）\n" + "\n".join(f"- {i}" for i in state["critique"]["issues"])
    res = await llm.structured(ExplainOut, prompts.SYSTEM_BASE, prompts.EXPLAIN.format(
        domain=domain_context(state["symptom"]), problem=_j(state["problem"]),
        evidence=_j({k: v for k, v in state["evidence"].items() if k != "wafer_signatures"}),
        hypotheses=_j([{k: h[k] for k in ("cause_code", "name", "posterior", "factors")} for h in state["hypotheses"]]),
        playbook="\n".join(f"- #{b['id']} {b['text']}" for b in state.get("playbook", [])) or "无", feedback=fb))
    out = res.model_dump() if res else None
    if out:
        if isinstance(out.get("five_whys"), list) and len(out["five_whys"]) >= 3:
            base["five_whys"] = [str(w) for w in out["five_whys"]]
        for k, v in (out.get("ishikawa_extra") or {}).items():
            if k in base["ishikawa"] and isinstance(v, list):
                base["ishikawa"][k] += [str(x) for x in v if x]
        base["narrative"] = out.get("narrative") or base["narrative"]
        base["risks"] = [str(r) for r in out.get("risks") or []]
    top = state["hypotheses"][0]
    root_causes = [{"cause_code": h["cause_code"], "cause": h["name"], "category": h["six_m"],
                    "evidence": "；".join(f["detail"] for f in h["factors"] if f["name"] != "先验知识") or "依据本厂先验",
                    "confidence": h["posterior"]} for h in state["hypotheses"][:2] if h is top or h["posterior"] >= 0.2]
    found = cause_by_code(top["cause_code"])
    containment = [found[1].capa[0]] if found else []
    cp = state["evidence"].get("change_point")
    if cp:
        containment.append(f"以 {_fmt_time(cp['since'])} 为界圈定可疑批次 (bracketing)，在 MES 中 hold 并加测")
    return {**base, "root_causes": root_causes, "containment": containment, "iteration": it}


# ── 6. CAPA ─────────────────────────────────────────────
async def capa(state: dict, llm: LLM, emit: Emit) -> dict:
    top = state["hypotheses"][0]
    found = cause_by_code(top["cause_code"])
    emit("capa", "制定 CAPA", f"针对「{top['name']}」")
    tpl = found[1].capa if found else ("隔离可疑批次", "针对根因整改", "纳入 FMEA 与 FDC 监控")
    metric = found[1].metric if found else "失效率"
    acts = [{"kind": "containment", "title": tpl[0], "owner_role": "生产/良率主管", "due_days": 1,
             "verify_metric": "可疑批次 hold 率", "baseline": "-", "target": "100%"},
            {"kind": "corrective", "title": tpl[1], "owner_role": f"{STEPS[top['step']].name if top['step'] in STEPS else ''}工程师",
             "due_days": 7, "verify_metric": metric, "baseline": "当前值", "target": "恢复基线并稳定 2 周"},
            {"kind": "preventive", "title": tpl[2], "owner_role": "模块经理", "due_days": 30,
             "verify_metric": "同类问题复发次数", "baseline": "-", "target": "90 天 0 复发"}]
    if top["posterior"] < 0.6 and state.get("tests"):
        t = state["tests"][0]
        acts.insert(1, {"kind": "containment", "title": f"鉴别验证：针对「{t['name']}」做对照实验（腔体/批次对调或分批验证）",
                        "owner_role": "良率工程师", "due_days": 2, "verify_metric": "实验组 vs 对照组失效率",
                        "baseline": "-", "target": "确认或排除该根因"})
    res = await llm.structured(CapaOut, prompts.SYSTEM_BASE, prompts.CAPA.format(
        cause=top["name"], evidence=_j(state["evidence"].get("suspects", [])[:3]), template=_j(acts)))
    refined = [a.model_dump() for a in res.actions] if res else []
    if len({a["kind"] for a in refined}) == 3:
        for a in refined:
            try:
                a["due_days"] = int(a.get("due_days") or 7)
            except (TypeError, ValueError):
                a["due_days"] = 7
        acts = refined
    for a in acts:
        a["cause_code"] = top["cause_code"]
    return {"capa": acts}


# ── 7. 评审 ──────────────────────────────────────────────
def rubric(state: dict) -> tuple[int, list[str]]:
    score, issues = 0, []
    ev, hyps = state["evidence"], state["hypotheses"]
    data_backed = any(s["score"] >= 1.3 for s in ev.get("suspects", [])) or ev.get("fdc") or ev.get("wafer_pattern")
    if data_backed:
        score += 25
    else:
        issues.append("缺少数据证据（无显著共性/FDC/晶圆图特征），需补充 MES/YMS/FDC 数据")
    p1 = hyps[0]["posterior"]
    score += round(20 * min(1, p1 / 0.6))
    margin = p1 - (hyps[1]["posterior"] if len(hyps) > 1 else 0)
    if margin >= 0.15:
        score += 10
    else:
        issues.append(f"前两位假设概率接近（差 {margin:.0%}），应先做鉴别验证再大规模整改")
    whys = state.get("five_whys", [])
    score += 15 if len(whys) >= 4 else 5 * len(whys) // 2
    if len(whys) < 4:
        issues.append("5-Why 深度不足")
    kinds = {a["kind"] for a in state.get("capa", [])}
    score += 5 * len(kinds & set(ACTION_KINDS))
    issues += [f"缺少{ACTION_KINDS[k]}" for k in set(ACTION_KINDS) - kinds]
    if state.get("capa") and all(a.get("verify_metric") for a in state["capa"]):
        score += 10
    if state.get("playbook") is not None:
        score += 5
    return min(100, score), issues


async def review(state: dict, llm: LLM, emit: Emit, threshold: int) -> dict:
    r_score, r_issues = rubric(state)
    res = await llm.structured(CriticOut, prompts.SYSTEM_BASE, prompts.CRITIC.format(
        evidence=_j({"suspects": state["evidence"].get("suspects", [])[:3], "hypotheses": state["hypotheses"][:3]}),
        analysis=_j({"five_whys": state["five_whys"], "root_causes": state["root_causes"]}), capa=_j(state["capa"])), temperature=0.0)
    out = res.model_dump() if res else None
    if out:
        score, issues = round(0.5 * r_score + 0.5 * float(out["score"])), r_issues + [str(i) for i in out.get("issues") or []]
    else:
        score, issues = r_score, r_issues
    passed = score >= threshold
    emit("review", f"评审得分 {score}/100", "通过" if passed else "未通过：" + "；".join(issues[:2]))
    return {"critique": {"score": score, "rubric": r_score, "issues": issues, "passed": passed,
                         "suggestions": [str(s) for s in (out or {}).get("suggestions") or []]}}


def describe_stage(symptom: str) -> str:
    s = SYMPTOMS.get(symptom)
    return STAGES.get(s.stage, "") if s else ""


assert CAUSES  # noqa: S101
