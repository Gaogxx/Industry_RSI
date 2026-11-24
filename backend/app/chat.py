"""对话技能：Kaizen 智能体在对话里能做的事（AG-UI 端点 /agui 调用）。

- rca    描述问题 → 建案 → DBOS 工作流分析（步骤作为“思考过程”流式显示）→ 8D 报告
         「分析 KZ-0061」分析已有案例；「确认 KZ-0061 根因是聚焦环磨损」先请求人工批准，批准后关案并展示学到了什么
- coach  基于 KPI、学习曲线、知识图谱回答问题；「运行改进周期」触发复盘
每个技能都是 (kind, chunk) 异步流：kind = reasoning（思考过程）/ content（回答）/ activity（面板卡片）。
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator

from sqlmodel import col, select

from .agent import prompts
from .db import session_scope
from .fab.flow import CAUSES, SYMPTOMS, cause_by_code
from .improvement import cause_pareto, compute_kpis, learning_snapshot, run_cycle, tool_hotspots
from .llm import LLM
from .models import ACTION_KINDS, ACTION_STATUSES, Action, Analysis, Case, ImprovementCycle, utcnow
from .services import NotFound, analyze_case, case_from_text, close_case, create_case, extract_metrics

HELP = """👋 我是 **Kaizen 芯片质量根因分析智能体**。三种用法：

1. **分析已有案例**（带 MES/YMS/FDC 数据，最完整）：`分析 KZ-0061`
2. **描述新问题**：例如「Fab A 刻蚀层 CP 良率下降，失效集中在晶圆外圈，良率=86%」
3. **确认根因（触发持续学习）**：`确认 KZ-0061 根因是聚焦环磨损` —— 批准后系统会立刻告诉你它学到了什么

分析流程（DBOS 持久化工作流）：受理 → 共性分析 / 晶圆图 / FDC / 变点 → 贝叶斯根因排序（对比“若不学习”）
→ 经验手册 → [代码模式调查员] → 5-Why 与 CAPA → 评审 → 8D 报告。"""

_QUESTION = re.compile(r"[?？]\s*$|^(这个|该|它|请问|为什么|怎么|如何|哪些|什么|是否|有没有|能否|列出|查看|当前|还有)")


def _context_case_id(messages: list[dict]) -> int | None:
    for m in reversed(messages[:-1]):
        if m.get("role") == "assistant":
            hit = re.search(r"KZ-(\d{4})", str(m.get("content") or ""))
            if hit:
                return int(hit.group(1))
    return None


def _match_cause(symptom: str, text: str, hypotheses: list[dict]) -> str | None:
    cands = [(c.code, c.name) for c in CAUSES.get(symptom, [])] + [(h["cause_code"], h["name"]) for h in hypotheses]
    up = text.upper()
    for code, _ in cands:
        if code in up:
            return code
    best, score = None, 0
    for code, name in cands:  # 中文名二元组重叠度
        grams = {name[i:i + 2] for i in range(len(name) - 1)}
        s = sum(1 for g in grams if g in text)
        if s > score:
            best, score = code, s
    return best if score >= 2 else None


CONFIRM_RE = re.compile(r"(?:确认|关闭|关案)\s*(?:KZ-)?(\d{1,6}).*?根因", re.I)


def plan_confirm(case_id: int, text: str) -> dict:
    """解析“确认 KZ-xxxx 根因是 …”：返回待批准的关案计划，或 {"error": 提示}。不做任何写入。"""
    with session_scope() as db:
        case = db.get(Case, case_id)
        if not case:
            return {"error": f"未找到案例 KZ-{case_id:04d}。"}
        if case.status == "closed":
            return {"error": f"{case.code} 已经关闭。"}
        an = db.exec(select(Analysis).where(Analysis.case_id == case_id).order_by(col(Analysis.id).desc())).first()
        hyps = an.hypotheses if an else []
        symptom, code = case.symptom, case.code
    tail = re.split(r"根因(?:是|为|：|:)?", text, maxsplit=1)[-1]
    cause = _match_cause(symptom, tail, hyps)
    if not cause:
        opts = "、".join(f"`{c.code}` {c.name}" for c in CAUSES.get(symptom, []))
        return {"error": f"没能识别根因。可选：{opts}。也可以写新根因：`确认 KZ-{case_id:04d} 根因是 <新根因描述>`（会作为新知识加入图谱）。"}
    ranked = [h["cause_code"] for h in hyps]
    found = cause_by_code(cause)
    return {"case_id": case_id, "code": code, "cause_code": cause, "cause_name": found[1].name if found else cause,
            "rank": ranked.index(cause) + 1 if cause in ranked else None, "symptom": symptom}


async def confirm(case_id: int, text: str, actor: str, cause_code: str | None = None) -> AsyncIterator[tuple[str, object]]:
    if cause_code:
        with session_scope() as db:
            case = db.get(Case, case_id)
            symptom = case.symptom if case else ""
        code = cause_code
    else:
        plan = plan_confirm(case_id, text)
        if "error" in plan:
            yield "content", plan["error"]
            return
        code, symptom = plan["cause_code"], plan["symptom"]
    yield "reasoning", "▸ **关案并学习** 更新贝叶斯知识图谱与可进化记忆 → 重拟合证据权重（经验收门）→ 重新回放学习曲线\n"
    try:
        r = await close_case(case_id, cause_code=code, actor=actor)
    except (NotFound, ValueError) as e:
        yield "content", f"❌ {e}"
        return
    c, s = r["case"], r["learning_summary"]
    rank = r["rank_in_analysis"]
    lines = [f"✅ 已关闭 **{c['code']}**，确认根因「{c['root_cause']}」" + (f"（分析时排第 {rank} 位）" if rank else "（分析时未列出 → 作为新知识加入）"),
             "", "**🧠 本次学到了什么**"]
    for e in r["learning_events"]:
        if e["kind"] in ("confirm", "false_lead", "refute", "edge_added"):
            lines.append(f"- {e['detail']}：在「{SYMPTOMS[symptom].name}」下的份额 {e['before']:.0%} → **{e['after']:.0%}**")
        else:
            lines.append(f"- {e['detail']}")
    w = r.get("weights") or {}
    if w.get("n", 0) >= 20:
        lines.append(f"- 证据权重重拟合（{w['n']} 个已验证样本）：" + ("✅ 通过验收门，已采纳新版本" if w["accepted"] else f"保持原版本（{w['reason']}）"))
    lines += ["", f"**📈 学习曲线**（{s['cases']} 个已验证案例的预序评估：每个案例先预测、后学习）",
              f"- 首位根因命中率：持续学习 **{s['hit1_learning']:.0%}** vs 冻结在教科书先验 {s['hit1_frozen']:.0%}",
              f"- 近半程：持续学习 **{s['hit1_learning_recent']:.0%}** vs {s['hit1_frozen_recent']:.0%}",
              "", "🔗 [打开「自我进化」页面](#/evolution?tab=continual)"]
    yield "content", "\n".join(lines)
    yield "activity", {"type": "kaizen.learning", "route": f"/cases/{case_id}", "open": True,
                       "title": f"{c['code']} 已关案 · 本次学到了什么",
                       "rows": [e["detail"] for e in r["learning_events"]][:6],
                       "stats": {"持续学习命中率": f"{s['hit1_learning']:.0%}", "冻结先验": f"{s['hit1_frozen']:.0%}"}}


async def followup(case_id: int, messages: list[dict], text: str, actor: str) -> AsyncIterator[tuple[str, str]]:
    if re.search(r"改进周期|复盘|所有|全部|整体|全厂|学习曲线", text):
        async for part in coach(messages, text, actor):
            yield part
        return
    with session_scope() as db:
        case = db.get(Case, case_id)
        if not case:
            yield "content", f"未找到案例 KZ-{case_id:04d}。"
            return
        an = db.exec(select(Analysis).where(Analysis.case_id == case_id).order_by(col(Analysis.id).desc())).first()
        acts = db.exec(select(Action).where(Action.case_id == case_id)).all()
        now = utcnow()
        ctx = {"case": {"code": case.code, "title": case.title, "status": case.status, "data": case.data},
               "evidence": {k: v for k, v in (an.evidence if an else {}).items() if k != "wafer_signatures"},
               "hypotheses": [{k: h[k] for k in ("cause_code", "name", "posterior", "factors")} for h in (an.hypotheses if an else [])],
               "counterfactual": an.counterfactual if an else {},
               "actions": [{"id": a.id, "kind": a.kind, "title": a.title, "status": a.status,
                            "overdue": a.status in ("todo", "doing") and bool(a.due_date and a.due_date < now)} for a in acts]}
    yield "reasoning", f"▸ **读取案例上下文** {ctx['case']['code']}：统计证据、贝叶斯排序、CAPA\n"
    llm = LLM()
    if llm.available:
        async for tok in llm.stream(prompts.SYSTEM_BASE + "\n" + prompts.FOLLOWUP.format(
                code=ctx["case"]["code"], context=json.dumps(ctx, ensure_ascii=False, default=str)),
                [m for m in messages if m.get("role") in ("user", "assistant")][-6:]):
            yield "content", tok
        return
    code, cf, hyps = ctx["case"]["code"], ctx["counterfactual"], ctx["hypotheses"]
    if re.search(r"学习|不学习|先验|教科书", text) and hyps:
        rows = "\n".join(f"| {h['name']} | {h['posterior']:.0%} | " +
                         next((f"{x['posterior']:.0%}" for x in cf.get("ranking_frozen", []) if x["cause_code"] == h["cause_code"]), "-") + " |"
                         for h in hyps)
        out = (f"**{code}：持续学习 vs 冻结先验**\n\n| 根因 | 已学习 | 教科书先验 |\n|---|---|---|\n{rows}\n\n" +
               ("首位假设**因学习而改变**。" if cf.get("changed") else "首位假设一致，学习改变了置信度。"))
    elif re.search(r"证据|数据|共性|机台|FDC|晶圆", text, re.I):
        ev = ctx["evidence"]
        sus = "\n".join(f"- {s['value']}（{s['step']}）失效率 {s['rate_with']:.0%} vs {s['rate_without']:.0%}，p_adj={s['p_adj']}"
                        for s in ev.get("suspects", [])[:3]) or "- 无显著共性"
        out = f"**{code} 统计证据**\n\n共性分析：\n{sus}\n\n晶圆图特征：{ev.get('wafer_pattern') or '-'}；FDC：" + \
              ("；".join(f"{f['tool']} {f['param']} {f['z']:+.1f}σ" for f in ev.get("fdc", [])) or "无显著偏移")
    elif re.search(r"CAPA|措施|逾期|进度", text, re.I):
        out = f"**{code} CAPA**\n\n" + ("\n".join(f"- A-{a['id']} [{ACTION_KINDS[a['kind']]}] {a['title']} —— "
                                                f"{ACTION_STATUSES[a['status']]}" + (" ⚠️逾期" if a["overdue"] else "")
                                                for a in ctx["actions"]) or "- 暂无")
    else:
        out = f"**{code} 根因排序**\n\n" + "\n".join(f"{i}. {h['name']}（{h['posterior']:.0%}）" for i, h in enumerate(hyps, 1))
    yield "content", out + f"\n\n> 离线模式：配置大模型 Key 后可自由追问。\n\n🔗 [打开案例页面](#/cases/{case_id})"
    yield "activity", {"type": "kaizen.case", "route": f"/cases/{case_id}", "open": False, "title": f"{code} 案例面板"}


async def rca(messages: list[dict], text: str, actor: str) -> AsyncIterator[tuple[str, str]]:
    t = text.strip()
    if len(t) < 6 or t.lower() in ("help", "帮助", "你好", "hi", "hello"):
        yield "content", HELP
        return
    m_confirm = CONFIRM_RE.search(t)
    if m_confirm:
        async for part in confirm(int(m_confirm.group(1)), t, actor):
            yield part
        return
    m = re.search(r"(?:分析|analy[sz]e)\s*(?:KZ-)?(\d{1,6})\b", t, re.I) if len(t) < 40 else None
    ctx_case = _context_case_id(messages)
    if not m and ctx_case and _QUESTION.search(t) and not extract_metrics(t):
        async for part in followup(ctx_case, messages, t, actor):
            yield part
        return
    with session_scope() as db:
        if m:
            case = db.get(Case, int(m.group(1)))
            if not case:
                yield "content", f"未找到案例 KZ-{int(m.group(1)):04d}。"
                return
        else:
            case = create_case(db, {**case_from_text(t), "source": "chat", "reporter": actor}, actor=actor)
        case_id, code = case.id, case.code
    yield "reasoning", f"📁 {'载入' if m else '已建案'} {code}\n"
    done = None
    async for ev in analyze_case(case_id, actor=actor):
        if ev["type"] == "workflow":
            yield "reasoning", f"⚙️ DBOS 持久化工作流 `{ev['workflow_id']}`（每步检查点，中断可恢复）\n"
        elif ev["type"] == "step":
            yield "reasoning", f"▸ **{ev['label']}** {ev.get('detail', '')}\n"
        elif ev["type"] == "done":
            done = ev
        elif ev["type"] == "error":
            yield "content", f"❌ 工作流出错：{ev['message']}"
    if done:
        tip = ("" if m else "\n\n> ℹ️ 该案例由聊天创建，未关联 MES/YMS/FDC 数据，结论主要依据已学习的本厂知识；"
               "接入数据后可做共性分析与晶圆图识别。")
        yield "content", (done["report_md"] + tip + f"\n\n🔗 [打开 {code} 案例页面](#/cases/{case_id})"
                          f"　·　确认根因后回复：`确认 {code} 根因是 …`")
        with session_scope() as db:
            an = db.exec(select(Analysis).where(Analysis.case_id == case_id).order_by(col(Analysis.id).desc())).first()
            top = [{"name": h["name"], "posterior": h["posterior"], "cause_code": h["cause_code"]} for h in (an.hypotheses if an else [])[:3]]
            hz = (an.counterfactual or {}).get("harness") if an else None
        yield "activity", {"type": "kaizen.case", "route": f"/cases/{case_id}", "open": True, "title": f"{code} 根因分析",
                           "top": top, "changed": bool(done.get("learning_changed_top")),
                           "harness": {"route": hz["route"], "reliability": hz["reliability"], "set_n": len(hz["set"]),
                                       "ttt": (hz.get("inner") or {}).get("ttt")} if hz else None}


async def coach(messages: list[dict], text: str, actor: str) -> AsyncIterator[tuple[str, str]]:
    if re.search(r"改进周期|复盘|run.?cycle|周报|月报", text, re.I):
        yield "reasoning", "▸ **运行改进周期** KPI 体检 → 帕累托 → 学习曲线回放 → 关键发现\n"
        cycle = await run_cycle(trigger=actor)
        yield "content", cycle.summary_md + "\n\n🔗 [打开「改进周期」页面](#/cycles)"
        yield "activity", {"type": "kaizen.cycle", "route": "/cycles", "open": True, "title": f"改进周期 #{cycle.id}",
                           "rows": list(cycle.findings)[:4]}
        return
    with session_scope() as db:
        last = db.exec(select(ImprovementCycle).order_by(col(ImprovementCycle.id).desc())).first()
        open_cases = db.exec(select(Case).where(Case.status != "closed").order_by(col(Case.created_at).desc())).all()
        ctx = {"kpis": {k: v for k, v in compute_kpis(db).items() if k != "trend"}, "causes": cause_pareto(db),
               "hotspots": tool_hotspots(db), "learning": learning_snapshot(db),
               "latest_findings": last.findings if last else [],
               "open_cases": [{"code": c.code, "title": c.title, "status": c.status} for c in open_cases[:10]]}
    llm = LLM()
    if llm.available:
        async for tok in llm.stream(prompts.COACH.format(context=json.dumps(ctx, ensure_ascii=False, default=str)),
                                    [m for m in messages if m.get("role") in ("user", "assistant")][-8:]):
            yield "content", tok
        return
    k, L = ctx["kpis"], ctx["learning"]
    bt = L.get("backtest") or {}
    lines = ["**当前改进与学习态势**（离线模式：配置大模型 Key 后可自由问答）", "",
             f"- 案例 {k['cases_total']} 个，未关闭 {k['cases_open']}，MTTR {k['mttr_days']} 天，逾期 CAPA {k['capa_overdue']}",
             f"- 学习曲线：首位命中率 持续学习 {bt.get('hit1_learning', 0):.0%} vs 冻结先验 {bt.get('hit1_frozen', 0):.0%}"
             f"（近半程 {bt.get('hit1_learning_recent', 0):.0%} vs {bt.get('hit1_frozen_recent', 0):.0%}）",
             *[f"- 知识更新：「{s['symptom_name']}」→「{s['name'][:18]}」{s['prior']:.0%} → {s['posterior']:.0%}" for s in L["top_shifts"][:3]],
             *[f"- 机台热点：{h['tool']} × {h['count']} 次（{h['cause'][:16]}）" for h in ctx["hotspots"][:2]],
             "", "**最近一次改进周期**", *([f"- {f}" for f in ctx["latest_findings"]] or ["- 暂无，输入「运行改进周期」"])]
    yield "content", "\n".join(lines)
    yield "activity", {"type": "kaizen.overview", "route": "/evolution" if re.search(r"学习|进化|RSI|效果", text, re.I) else "/overview",
                       "open": True, "title": "自我进化" if re.search(r"学习|进化|RSI|效果", text, re.I) else "质量总览"}
