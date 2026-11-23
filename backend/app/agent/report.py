"""8D 报告渲染（由结构化结果确定性生成，格式统一、可追溯）。"""

from __future__ import annotations

from ..fab.flow import STAGES, STEPS, SYMPTOMS
from ..models import ACTION_KINDS, SEVERITIES, SIX_M


def _c(v) -> str:
    return str(v).replace("|", "/").replace("\n", " ")


def render_8d(case: dict, state: dict, analysis_id: int | None = None) -> str:
    p, ev, crit = state.get("problem") or {}, state.get("evidence") or {}, state.get("critique") or {}
    sym = SYMPTOMS.get(state.get("symptom") or case.get("symptom"))
    hyps, cf = state.get("hypotheses") or [], state.get("counterfactual") or {}
    code = f"KZ-{case['id']:04d}" if case.get("id") else "KZ-NEW"
    sev = state.get("severity") or case.get("severity", "")
    L = [
        f"# 8D 分析报告 · {code} {case.get('title', '')}",
        "",
        f"> 工段：**{STAGES.get(sym.stage, '-') if sym else '-'}** ｜ 症状：{sym.name if sym else '-'} ｜ 严重度：{sev} {SEVERITIES.get(sev, '')}"
        f" ｜ 产品：{case.get('product') or '-'}",
        f"> 评审得分：**{crit.get('score', '-')}/100** ｜ 解释迭代：{state.get('iteration', 1)} 轮 ｜ 模型：{state.get('model', '-')}",
        "",
        "## D1 团队",
        "良率工程（组长）、模块工艺（光刻/刻蚀/薄膜/CMP 或封装）、设备工程、测试工程、质量/SQE",
        "",
        "## D2 问题描述（5W2H）",
        f"**{p.get('problem_statement', '')}**",
        "",
        "| 维度 | 内容 |",
        "|---|---|",
    ]
    for k, label in [("what", "What"), ("where", "Where"), ("when", "When"), ("who", "Who"),
                     ("how_many", "How many"), ("how_detected", "How detected"), ("impact", "影响")]:
        L.append(f"| {label} | {_c(p.get(k, '-'))} |")
    L += ["", "## D3 围堵措施"] + [f"- {c}" for c in state.get("containment", [])]

    L += ["", "## D4 根因分析", "", "### 统计证据（MES / YMS / FDC）"]
    s = ev.get("summary") or {}
    if s:
        L.append(f"- 批次：{s.get('lots')} 个，失效 {s.get('fail_lots')} 个" +
                 (f"；平均 CP 良率 {s['mean_yield']}%，失效批次 {s.get('fail_mean_yield')}%" if s.get("mean_yield") else ""))
    if ev.get("wafer_pattern"):
        L.append(f"- 晶圆图空间特征：**{ev['wafer_pattern']}**（置信 {ev.get('wafer_pattern_conf', 0):.0%}）")
    if ev.get("suspects"):
        L += ["", "| 共性分析：工序 · 机台/批次 | 失效率（该机台 vs 其余） | 提升倍数 | p（Bonferroni 校正） |", "|---|---|---|---|"]
        for x in ev["suspects"][:4]:
            L.append(f"| {STEPS[x['step']].name if x['step'] in STEPS else x['step']} · {x['value']} | "
                     f"{x['rate_with']:.0%} vs {x['rate_without']:.0%} | {x['lift']} | {x['p_adj']} |")
    if ev.get("change_point"):
        cp = ev["change_point"]
        L.append(f"- 变点：{cp['value']} 失效率 {cp['rate_before']:.0%} → {cp['rate_after']:.0%}" + ("，**紧随 PM**" if cp["after_pm"] else ""))
    for f in ev.get("fdc", [])[:3]:
        L.append(f"- FDC：{f['tool']} {f['param']} {f['before']} → {f['after']}（{f['z']:+.1f}σ）")

    if hyps:
        L += ["", "### 贝叶斯根因排序", "", "| # | 根因 | 6M | 后验 | 先验（教科书 → 本厂已学习） | 证据贡献 |", "|---|---|---|---|---|---|"]
        for i, h in enumerate(hyps[:4], 1):
            prior_f = next((f for f in h["factors"] if f["name"] == "先验知识"), {})
            contrib = "；".join(f"{f['name']}×{f['x']}" for f in h["factors"] if f["name"] != "先验知识") or "-"
            L.append(f"| {i} | {_c(h['name'])} | {SIX_M.get(h['six_m'], h['six_m'])} | **{h['posterior']:.0%}** | "
                     f"{_c(prior_f.get('detail', ''))} | {_c(contrib)} |")
    if cf:
        if cf.get("changed"):
            L += ["", f"> 🧠 **持续学习的影响**：若只用教科书先验，首位假设会是「{cf['top_name']}」（{cf['top_posterior']:.0%}）；"
                      f"结合本厂已验证案例，首位为「{hyps[0]['name']}」（{hyps[0]['posterior']:.0%}）。"]
        elif hyps:
            L += ["", f"> 🧠 **持续学习的影响**：首位假设一致，已学习知识使其概率从 {cf['learned_top_posterior_if_frozen']:.0%} 提高到 "
                      f"{hyps[0]['posterior']:.0%}（不确定性 {cf['entropy_frozen']} → {cf['entropy_learned']} bit）。"]
    if state.get("tests") and hyps and hyps[0]["posterior"] < 0.6:
        t = state["tests"][0]
        L.append(f"> 🔬 根因尚不确定：建议优先鉴别「{t['name']}」，预期信息增益 {t['eig']} bit。")

    L += ["", "### 5-Why"] + [f"{i}. {w}" for i, w in enumerate(state.get("five_whys", []), 1)]
    if state.get("narrative"):
        L += ["", f"**结论**：{state['narrative']}"]
    ish = state.get("ishikawa") or {}
    if any(ish.values()):
        L += ["", "### 鱼骨图（6M）", "", "| 维度 | 候选因素 |", "|---|---|"]
        for k, label in SIX_M.items():
            if ish.get(k):
                L.append(f"| {label} | {_c('；'.join(ish[k]))} |")

    acts = state.get("capa") or []
    for kind, title in [("corrective", "D5/D6 纠正措施与验证"), ("preventive", "D7 预防再发（横向展开）")]:
        L += ["", f"## {title}", "", "| 措施 | 责任角色 | 期限 | 验证指标 | 目标 |", "|---|---|---|---|---|"]
        for a in (a for a in acts if a.get("kind") == kind):
            L.append(f"| {_c(a['title'])} | {a.get('owner_role', '')} | {a.get('due_days', '')} 天 | "
                     f"{_c(a.get('verify_metric', ''))} | {_c(a.get('target', ''))} |")

    L += ["", "## D8 经验与引用"]
    for b in state.get("playbook", [])[:3]:
        L.append(f"- 记忆 #{b['id']}（{'产线' if b.get('source') == 'real' else '孪生' if b.get('source') == 'sandbox' else '产线+孪生'}验证 通过 {b['helpful']} / 失败 {b['harmful']}）：{b['text']}")
    if state.get("similar"):
        L.append("- 同症状历史案例：" + "；".join(f"{x['code']}（{x['root_cause'][:16]}）" for x in state["similar"][:3]))
    if crit.get("issues"):
        L += ["", "**评审意见（待跟进）**"] + [f"- {i}" for i in crit["issues"][:5]]
    L += ["", "---", "关案时请确认真实根因：系统会据此更新可进化记忆（知识、机台热点、流程与教训），下一次同类分析立即用上"
          + (f"（分析 #{analysis_id}）。" if analysis_id else "。")]
    return "\n".join(L)


assert ACTION_KINDS  # noqa: S101
