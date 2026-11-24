"""DBOS 持久化工作流（Durable Execution）：替代 LangGraph 的编排层。

每个步骤的输出都检查点到 DBOS 系统库（SQLite），因此：
- 服务在分析中途崩溃/重启后，`DBOS.launch()` 会从最后完成的步骤继续，不重复调用大模型；
- 每次分析都有完整的“执行轨迹”（步骤、输入输出、耗时），可在看板中查看，也可从任一步骤 fork 重放；
- 进度通过 DBOS Stream 推送，AG-UI 对话里实时显示。
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from datetime import timedelta

from dbos import DBOS, SetWorkflowID
from sqlmodel import select

from ..config import get_settings
from ..db import session_scope
from ..fab.flow import SYMPTOMS
from ..llm import LLM
from ..models import Action, Analysis, AuditLog, Case, FabDataset, HarnessReceipt, utcnow
from . import steps
from .investigator import fuse, investigate
from .report import render_8d

STREAM = "progress"
_LLM_OVERRIDES: dict[str, LLM] = {}  # 测试/调用方按 workflow_id 注入模型


def _llm() -> LLM:
    return _LLM_OVERRIDES.get(DBOS.workflow_id or "") or LLM()


class _Run:
    """步骤内的事件收集 + token 用量差分。"""

    def __init__(self, node: str):
        self.node, self.events, self.llm = node, [], _llm()
        self.u0 = (self.llm.usage.prompt_tokens, self.llm.usage.completion_tokens, self.llm.usage.calls)
        self.t0 = time.perf_counter()

    def emit(self, node: str, label: str, detail: str = "") -> None:
        self.events.append({"type": "step", "node": node, "label": label, "detail": detail})

    async def finish(self, out: dict) -> dict:
        for e in self.events:
            await DBOS.write_stream_async(STREAM, e)
        u = self.llm.usage
        out["_trace"] = {"node": self.node, "ms": round((time.perf_counter() - self.t0) * 1000),
                         "prompt_tokens": u.prompt_tokens - self.u0[0], "completion_tokens": u.completion_tokens - self.u0[1],
                         "llm_calls": u.calls - self.u0[2], "errors": u.errors[-2:]}
        return out


@DBOS.step()
async def load_case(case_id: int) -> dict:
    with session_scope() as db:
        c = db.get(Case, case_id)
        d = c.model_dump()
        d["code"] = c.code
        return d


@DBOS.step()
async def s_intake(state: dict) -> dict:
    r = _Run("intake")
    return await r.finish(await steps.intake(state, r.llm, r.emit))


@DBOS.step()
async def s_evidence(state: dict) -> dict:
    r = _Run("evidence")
    return await r.finish(await steps.evidence(state, r.emit))


@DBOS.step()
async def s_infer(state: dict) -> dict:
    r = _Run("infer")
    return await r.finish(await steps.infer(state, r.emit))


@DBOS.step()
async def s_harness(state: dict) -> dict:
    """快慢双脑：快脑可靠就直接给结论；否则慢脑做双层递归的内层（证据族对照 + 近邻练习 / 测试时训练）。"""
    from ..harness.brain import decide
    from ..rsi.memory import load_memory

    r = _Run("harness")
    case = {**state["case"], "severity": state.get("severity") or state["case"].get("severity", "")}
    with session_scope() as db:
        mem = load_memory(db)
        n_prev = len(db.exec(select(Analysis).where(Analysis.case_id == case["id"])).all())
        out = await decide(db, case, state.get("evidence") or {}, state["symptom"], state["hypotheses"], mem,
                           seed=case["id"] * 7919 + n_prev)
        db.commit()
    ans, rc = out["answer"], out["receipt"]
    receipt = {k: v for k, v in rc.model_dump().items() if k not in ("id", "created_at", "verified_at", "analysis_id")}
    ttt = (ans["inner"] or {}).get("ttt")
    if ans["route"] == "fast":
        r.emit("harness", "快脑 · 直接给结论", f"可靠度 {ans['reliability']:.0%}，保形集合 {len(ans['set'])} 个根因；不跑孪生、不调大模型")
    else:
        r.emit("harness", "慢脑 · 双层递归", "转慢脑原因：" + "、".join(ans["reasons"]) +
               (f"；近邻练习 {ttt['tasks']} 题，探测命中 {ttt['probe_before']}/{ttt['probe_n']} → {ttt['probe_after']}/{ttt['probe_n']}，"
                + ("采纳练习后的结论" if ttt["adopt"] else "回退到线上记忆") if ttt else ""))
    return await r.finish({"hypotheses": out["final"], "harness": {"answer": ans, "receipt": receipt}})


@DBOS.step()
async def s_knowledge(state: dict) -> dict:
    r = _Run("knowledge")
    return await r.finish(await steps.knowledge(state, r.emit))


@DBOS.step()
async def s_investigate(state: dict) -> dict:
    r = _Run("investigate")
    if not (r.llm.available and get_settings().investigator):
        return await r.finish({"investigation": None})
    with session_scope() as db:
        ds = db.get(FabDataset, state["case"]["id"])
        data = ds.data if ds else None
    if not data:
        return await r.finish({"investigation": None})
    r.emit("investigate", "代码模式调查", "大模型编写 Python，在 Monty 沙箱中切分批次 / FDC / 晶圆图数据核查假设")
    inv = await investigate(r.llm, state["case"], SYMPTOMS[state["symptom"]].name, data, state["hypotheses"])
    if not inv:
        r.emit("investigate", "调查未完成", "沙箱调查失败，沿用统计内核结论")
        return await r.finish({"investigation": None})
    hyps = fuse([dict(h, factors=list(h["factors"])) for h in state["hypotheses"]], inv)
    r.emit("investigate", "调查结论", "；".join(inv.findings[:2]) or "无新发现")
    return await r.finish({"investigation": inv.model_dump(), "hypotheses": hyps})


@DBOS.step()
async def s_explain(state: dict) -> dict:
    r = _Run("explain")
    return await r.finish(await steps.explain(state, r.llm, r.emit))


@DBOS.step()
async def s_capa(state: dict) -> dict:
    r = _Run("capa")
    return await r.finish(await steps.capa(state, r.llm, r.emit))


@DBOS.step()
async def s_review(state: dict) -> dict:
    r = _Run("review")
    return await r.finish(await steps.review(state, r.llm, r.emit, get_settings().critic_threshold))


@DBOS.step()
async def s_persist(state: dict) -> dict:
    """落库：分析、措施、案例状态、审计。作为一个步骤检查点，崩溃重放时不会重复写入。"""
    case_id = state["case"]["id"]
    trace = state.get("trace", [])
    usage = {k: sum(t.get(k, 0) for t in trace) for k in ("prompt_tokens", "completion_tokens", "llm_calls")}
    with session_scope() as db:
        version = len(db.exec(select(Analysis).where(Analysis.case_id == case_id)).all()) + 1
        an = Analysis(case_id=case_id, version=version, problem_statement=state.get("problem") or {},
                      evidence=state.get("evidence") or {}, hypotheses=state.get("hypotheses") or [],
                      counterfactual={**(state.get("counterfactual") or {}), "tests": state.get("tests") or [],
                                      "investigation": state.get("investigation"),
                                      "harness": (state.get("harness") or {}).get("answer")},
                      ishikawa=state.get("ishikawa") or {}, five_whys=state.get("five_whys") or [],
                      root_causes=state.get("root_causes") or [], containment=state.get("containment") or [],
                      critique=state.get("critique") or {}, iterations=state.get("iteration", 1),
                      references=[{"type": "case", **x} for x in state.get("similar") or []]
                                 + [{"type": "bullet", **b} for b in state.get("playbook") or []],
                      model=state.get("model", ""), trace=trace + [{"usage": usage, "workflow_id": DBOS.workflow_id}])
        db.add(an)
        db.flush()
        an.report_md = render_8d(state["case"], state, an.id)
        now = utcnow()
        for a in state.get("capa") or []:
            db.add(Action(case_id=case_id, analysis_id=an.id, kind=a["kind"], title=str(a["title"])[:500],
                          cause_code=a.get("cause_code", ""), owner_role=str(a.get("owner_role", "")),
                          due_days=a.get("due_days", 7), due_date=now + timedelta(days=a.get("due_days", 7)),
                          verify_metric=str(a.get("verify_metric", "")), baseline=str(a.get("baseline", "")),
                          target=str(a.get("target", ""))))
        hz = state.get("harness") or {}
        if hz.get("receipt"):
            db.add(HarnessReceipt(**hz["receipt"], analysis_id=an.id))
        case = db.get(Case, case_id)
        top = (state.get("hypotheses") or [{}])[0]
        case.symptom = state.get("symptom", case.symptom)
        case.stage = SYMPTOMS[case.symptom].stage
        case.evidence = state.get("evidence") or case.evidence
        case.severity = state.get("severity") or case.severity
        case.root_cause, case.root_cause_category = top.get("name", ""), top.get("six_m", "")
        case.status = "capa"
        db.add(case)
        db.add(AuditLog(actor=state.get("actor", "api"), action="agent.analyze", target=case.code,
                        detail={"analysis_id": an.id, "score": an.critique.get("score"), "model": an.model,
                                "top": top.get("cause_code"), "workflow_id": DBOS.workflow_id,
                                "learning_changed_top": an.counterfactual.get("changed"),
                                **({"route": hz["answer"]["route"], "reliability": hz["answer"]["reliability"]} if hz.get("answer") else {})}))
        db.commit()
        return {"case_id": case_id, "code": case.code, "analysis_id": an.id, "score": an.critique.get("score"),
                "top": top.get("cause_code"), "learning_changed_top": an.counterfactual.get("changed"),
                "report_md": an.report_md}


@DBOS.workflow()
async def rca_workflow(case_id: int, actor: str = "api") -> dict:
    s = get_settings()
    state: dict = {"case": await load_case(case_id), "trace": [], "actor": actor}

    def merge(out: dict) -> None:
        state["trace"].append(out.pop("_trace"))
        state.update(out)

    try:
        for step in (s_intake, s_evidence, s_infer, s_harness, s_knowledge):
            merge(await step(state))
        requested = DBOS.workflow_id in _LLM_OVERRIDES  # 调用方明确要求用某个大模型分析（如脚本化模型、人工点名深查）
        if requested or (state.get("harness") or {}).get("answer", {}).get("route") != "fast":  # 快脑结论不调用大模型调查员
            merge(await s_investigate(state))
        while True:
            for step in (s_explain, s_capa, s_review):
                merge(await step(state))
            if state["critique"]["passed"] or state["iteration"] >= s.max_iterations:
                break
        state["model"] = _llm().model_name
        await DBOS.write_stream_async(STREAM, {"type": "step", "node": "report", "label": "生成 8D 报告并落库", "detail": ""})
        result = await s_persist(state)
    except Exception as e:
        await DBOS.write_stream_async(STREAM, {"type": "error", "message": str(e)[:300]})
        await DBOS.close_stream_async(STREAM)
        raise
    await DBOS.close_stream_async(STREAM)
    return result


async def run_rca(case_id: int, actor: str = "api", llm: LLM | None = None) -> AsyncIterator[dict]:
    """启动持久化工作流并实时转发进度。"""
    wf_id = f"rca-KZ{case_id:04d}-{uuid.uuid4().hex[:8]}"
    if llm is not None:
        _LLM_OVERRIDES[wf_id] = llm
    try:
        with SetWorkflowID(wf_id):
            handle = await DBOS.start_workflow_async(rca_workflow, case_id, actor)
        yield {"type": "workflow", "workflow_id": wf_id}
        async for ev in DBOS.read_stream_async(wf_id, STREAM, polling_interval_sec=0.1, timeout_seconds=900):
            yield ev
        result = await handle.get_result()
        yield {"type": "done", "workflow_id": wf_id, **result}
    finally:
        _LLM_OVERRIDES.pop(wf_id, None)


async def workflow_steps(wf_id: str) -> list[dict]:
    rows = await DBOS.list_workflow_steps_async(wf_id, load_output=False)
    return [{"id": r["function_id"], "name": r["function_name"], "error": str(r["error"])[:200] if r.get("error") else None,
             "ms": (r["completed_at_epoch_ms"] - r["started_at_epoch_ms"]) if r.get("completed_at_epoch_ms") else None}
            for r in rows]


# ── 定时：持续学习夜间任务（DBOS cron）───────────────────────
@DBOS.workflow()
async def nightly_learning(scheduled_at, context) -> dict:
    return await _nightly_step()


@DBOS.step()
async def _nightly_step() -> dict:
    from ..improvement import run_cycle

    from ..db import session_scope
    from ..rsi.memory import load_memory, refit_and_store

    from ..harness.service import nightly as harness_nightly

    cycle = await run_cycle(trigger="schedule")
    with session_scope() as db:  # 夜间：用全部已验证样本（产线 + 孪生）重拟合证据权重
        w = refit_and_store(db, load_memory(db), source="nightly")
        db.commit()
        hz = await harness_nightly(db)  # harness：观察期检查 → 可靠度训练窗口（有触发才开）→ 每周一次 RRSI 策略进化
    return {"cycle_id": cycle.id, "weights_accepted": w["accepted"], "harness": hz}


async def ensure_schedule() -> None:
    cron = get_settings().nightly_cron
    if not cron:
        return
    try:
        if not await DBOS.get_schedule_async("nightly-learning"):
            await DBOS.create_schedule_async(schedule_name="nightly-learning", workflow_fn=nightly_learning,
                                             schedule=cron, cron_timezone="Asia/Shanghai")
    except Exception:  # noqa: BLE001  已存在或不支持时忽略，不影响主流程
        pass
