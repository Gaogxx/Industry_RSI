"""AG-UI 1.0 端点：Kaizen 控制台（对话为主界面）的“智能体 ↔ 界面”协议。

POST /agui  请求体为 AG-UI RunAgentInput；forwardedProps.mode 选择：
  rca    根因分析对话（分析 KZ-xxxx / 描述新问题 / 确认根因并学习 / 追问）
  coach  持续改进教练
  rsi    启动一次自我进化（RSI）会话，并以 AG-UI 事件实时直播
事件映射：
  工作流步骤 → REASONING_*（思考过程）           回答 → TEXT_MESSAGE_*
  面板卡片 → ACTIVITY_SNAPSHOT / ACTIVITY_DELTA（activityType = kaizen.*；卡片留在消息里，用户点开后才在主区域完整显示）
  关注的案例 → STATE_SNAPSHOT（workspace.focusCaseId）
  关案（改写产线记忆）→ RUN_FINISHED(outcome=interrupt)，界面批准后带 resume 发起下一次运行才执行（人在回路）
  RSI 三角色 → SUBAGENT_STARTED/FINISHED     学习曲线 → STATE_SNAPSHOT / STATE_DELTA（JSON Patch）
  BRS / CAL / DRS 阶段 → STEP_STARTED/FINISHED  单题与权重重拟合 → CUSTOM
"""

from __future__ import annotations

import re
import uuid
from collections.abc import AsyncIterator

from ag_ui.core import (ActivityDeltaEvent, ActivitySnapshotEvent, CustomEvent, EventType, Interrupt, ReasoningEndEvent,
                        ReasoningMessageContentEvent, ReasoningMessageEndEvent, ReasoningMessageStartEvent,
                        ReasoningStartEvent, RunErrorEvent, RunFinishedEvent, RunStartedEvent, StateDeltaEvent,
                        StateSnapshotEvent, StepFinishedEvent, StepStartedEvent, SubagentFinishedEvent,
                        SubagentStartedEvent, TextMessageContentEvent, TextMessageEndEvent, TextMessageStartEvent)
from ag_ui.core import RunFinishedInterruptOutcome, RunFinishedSuccessOutcome
from ag_ui.encoder import EventEncoder
from dbos import DBOS
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from .auth import actor_of, require_key
from .fab.flow import SYMPTOMS

router = APIRouter(dependencies=[Depends(require_key)])

ROLE = {"curriculum": "课程 agent", "executor": "执行 agent", "verifier": "验证 agent"}
PHASE = {"BRS": "广度递归自探索 BRS", "CAL": "证据校准 CAL", "DRS": "深度递归自探索 DRS"}
CLOSE_PREFIX = "int-close-"


def _text(messages: list[dict]) -> str:
    for m in reversed(messages or []):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                return "\n".join(p.get("text", "") for p in c if isinstance(p, dict))
            return str(c or "")
    return ""


def _activity(payload: dict) -> ActivitySnapshotEvent:
    kind = payload.get("type", "kaizen.panel")
    return ActivitySnapshotEvent(type=EventType.ACTIVITY_SNAPSHOT, message_id=f"act-{uuid.uuid4().hex[:8]}",
                                 activity_type=kind, content={k: v for k, v in payload.items() if k != "type"})


class Messages:
    """把 (kind, chunk) 流转成 AG-UI 的思考过程 / 回答 / 面板事件。"""

    def __init__(self) -> None:
        self.rid, self.mid = f"r-{uuid.uuid4().hex[:8]}", f"m-{uuid.uuid4().hex[:8]}"
        self.reasoning_open = self.text_open = False

    def reasoning(self, chunk: str):
        if not self.reasoning_open:
            yield ReasoningStartEvent(type=EventType.REASONING_START, message_id=self.rid)
            yield ReasoningMessageStartEvent(type=EventType.REASONING_MESSAGE_START, message_id=self.rid, role="reasoning")
            self.reasoning_open = True
        yield ReasoningMessageContentEvent(type=EventType.REASONING_MESSAGE_CONTENT, message_id=self.rid, delta=chunk)

    def end_reasoning(self):
        if self.reasoning_open:
            yield ReasoningMessageEndEvent(type=EventType.REASONING_MESSAGE_END, message_id=self.rid)
            yield ReasoningEndEvent(type=EventType.REASONING_END, message_id=self.rid)
            self.reasoning_open = False

    def text(self, chunk: str):
        yield from self.end_reasoning()
        if not self.text_open:
            yield TextMessageStartEvent(type=EventType.TEXT_MESSAGE_START, message_id=self.mid, role="assistant")
            self.text_open = True
        yield TextMessageContentEvent(type=EventType.TEXT_MESSAGE_CONTENT, message_id=self.mid, delta=chunk)

    def close(self):
        yield from self.end_reasoning()
        if self.text_open:
            yield TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id=self.mid)
            self.text_open = False

    async def pump(self, gen: AsyncIterator) -> AsyncIterator:
        async for kind, chunk in gen:
            if kind == "reasoning":
                for e in self.reasoning(chunk):
                    yield e
            elif kind == "activity":
                for e in self.end_reasoning():
                    yield e
                yield _activity(chunk)
                m = re.match(r"/cases/(\d+)", chunk.get("route", ""))
                if m:
                    yield StateSnapshotEvent(type=EventType.STATE_SNAPSHOT, snapshot={"workspace": {"focusCaseId": int(m.group(1))}})
            else:
                for e in self.text(chunk):
                    yield e
        for e in self.close():
            yield e


async def _chat_events(mode: str, messages: list[dict], actor: str, resume: list[dict], out: dict) -> AsyncIterator:
    from .chat import CONFIRM_RE, coach, confirm, plan_confirm, rca

    msg = Messages()
    pending = [r for r in resume if str(r.get("interruptId") or r.get("interrupt_id") or "").startswith(CLOSE_PREFIX)]
    if pending:  # 上一次运行停在“待批准的关案”，这次运行带 resume 回来
        r = pending[0]
        rid = str(r.get("interruptId") or r.get("interrupt_id"))
        case_id, cause_code = rid[len(CLOSE_PREFIX):].split("-", 1)
        payload = r.get("payload") or {}
        if r.get("status") == "resolved" and payload.get("approved"):
            note = f"（批准人：{actor}" + (f"，备注：{payload['reason']}" if payload.get("reason") else "") + "）"
            for e in msg.reasoning(f"✅ 关案已批准{note}\n"):
                yield e
            async for e in msg.pump(confirm(int(case_id), "", actor, cause_code=cause_code)):
                yield e
        else:
            async for e in msg.pump(_aiter([("content", "已取消：没有关案，记忆未改动。")])):
                yield e
        return

    text = _text(messages)
    m = CONFIRM_RE.search(text.strip()) if mode == "rca" else None
    if m:  # 关案会改写产线记忆（权重 1.0）并重拟合证据权重：先请求人工批准
        plan = plan_confirm(int(m.group(1)), text)
        if "error" in plan:
            async for e in msg.pump(_aiter([("content", plan["error"])])):
                yield e
            return
        rank = f"分析时排第 {plan['rank']} 位" if plan["rank"] else "分析时未列出，会作为新知识加入"
        async for e in msg.pump(_aiter([
                ("content", f"准备关闭 **{plan['code']}**，确认根因「{plan['cause_name']}」（{rank}）。\n\n"
                            "关案会把它作为**产线确认的标签**写入记忆（权重 1.0），并重拟合证据权重。请在下方批准。"),
                ("activity", {"type": "kaizen.case", "route": f"/cases/{plan['case_id']}", "open": True,
                              "title": f"{plan['code']} 待批准关案"})])):
            yield e
        out["interrupt"] = Interrupt(id=f"{CLOSE_PREFIX}{plan['case_id']}-{plan['cause_code']}", reason="approval",
                                     message=f"批准关闭 {plan['code']}，根因 = {plan['cause_name']}？",
                                     response_schema={"type": "object", "required": ["approved"],
                                                      "properties": {"approved": {"type": "boolean"}, "reason": {"type": "string"}}},
                                     metadata={"case_id": plan["case_id"], "cause_code": plan["cause_code"], "action": "case.close"})
        return
    gen = coach(messages, text, actor) if mode == "coach" else rca(messages, text, actor)
    async for e in msg.pump(gen):
        yield e


async def _aiter(items):
    for x in items:
        yield x


async def _rsi_events(actor: str) -> AsyncIterator:
    from .rsi.workflow import STREAM, start_session

    sid, wf = await start_session(trigger=actor)
    act = f"act-rsi-{sid}"
    yield StateSnapshotEvent(type=EventType.STATE_SNAPSHOT, snapshot={"session_id": sid, "workflow_id": wf, "curve": [], "tasks": []})
    yield ActivitySnapshotEvent(type=EventType.ACTIVITY_SNAPSHOT, message_id=act, activity_type="kaizen.rsi",
                                content={"route": f"/evolution?s={sid}", "open": True, "title": f"自我进化会话 #{sid}",
                                         "phase": "启动", "done": 0, "passes": 0, "hit1": None, "hit1_before": None})
    step_open, done, passes, first = None, 0, 0, None
    async for ev in DBOS.read_stream_async(wf, STREAM, polling_interval_sec=0.1, timeout_seconds=1800):
        t = ev.get("type")
        if t == "phase":
            if step_open:
                yield StepFinishedEvent(type=EventType.STEP_FINISHED, step_name=step_open)
                step_open = None
            if ev["phase"] in PHASE:
                step_open = PHASE[ev["phase"]]
                yield StepStartedEvent(type=EventType.STEP_STARTED, step_name=step_open)
                yield ActivityDeltaEvent(type=EventType.ACTIVITY_DELTA, message_id=act, activity_type="kaizen.rsi",
                                         patch=[{"op": "replace", "path": "/phase", "value": step_open}])
            yield CustomEvent(type=EventType.CUSTOM, name="rsi.phase", value=ev)
        elif t == "plan":
            yield CustomEvent(type=EventType.CUSTOM, name="rsi.plan", value=ev)
        elif t == "task":
            done, passes = done + 1, passes + (ev["verdict"] == "PASS")
            summary = {"curriculum": f"出题：{ev['variant']}", "executor": f"作答 {ev['attempts']} 次",
                       "verifier": f"判定 {ev['verdict']}"}
            for role in ("curriculum", "executor", "verifier"):  # 三个子智能体依次完成一题
                srid = f"{role}-{ev['id']}"
                yield SubagentStartedEvent(type=EventType.SUBAGENT_STARTED, subagent_run_id=srid, name=ROLE[role],
                                           description=f"RSI 练习题 #{ev['id']}")
                yield SubagentFinishedEvent(type=EventType.SUBAGENT_FINISHED, subagent_run_id=srid, result=summary[role])
            yield CustomEvent(type=EventType.CUSTOM, name="rsi.task",
                              value={**ev, "symptom_name": SYMPTOMS[ev["symptom"]].name if ev["symptom"] in SYMPTOMS else ev["symptom"]})
            yield StateDeltaEvent(type=EventType.STATE_DELTA, delta=[{"op": "add", "path": "/tasks/-", "value": ev}])
        elif t == "cal":  # 校准题按波次汇总
            done, passes = done + ev["n"], passes + ev["passes"]
            yield CustomEvent(type=EventType.CUSTOM, name="rsi.cal", value=ev)
        elif t == "weights":
            yield CustomEvent(type=EventType.CUSTOM, name="rsi.weights", value=ev)
        elif t == "eval":
            point = {k: ev[k] for k in ("step", "phase", "hit1", "partial", "memory")}
            first = point["hit1"] if first is None else first
            yield StateDeltaEvent(type=EventType.STATE_DELTA, delta=[{"op": "add", "path": "/curve/-", "value": point}])
            yield ActivityDeltaEvent(type=EventType.ACTIVITY_DELTA, message_id=act, activity_type="kaizen.rsi",
                                     patch=[{"op": "replace", "path": "/hit1", "value": point["hit1"]},
                                            {"op": "replace", "path": "/hit1_before", "value": first},
                                            {"op": "replace", "path": "/done", "value": done},
                                            {"op": "replace", "path": "/passes", "value": passes}])
        elif t == "done":
            if step_open:
                yield StepFinishedEvent(type=EventType.STEP_FINISHED, step_name=step_open)
                step_open = None
            yield ActivityDeltaEvent(type=EventType.ACTIVITY_DELTA, message_id=act, activity_type="kaizen.rsi",
                                     patch=[{"op": "replace", "path": "/phase", "value": "完成"},
                                            {"op": "replace", "path": "/hit1", "value": ev["hit1_after"]},
                                            {"op": "replace", "path": "/done", "value": ev["tasks"]},
                                            {"op": "replace", "path": "/passes", "value": ev["passes"]}])
            mid = f"m-{uuid.uuid4().hex[:8]}"
            w = ev.get("weights") or {}
            msg = (f"自我进化完成：共 {ev['tasks']} 道练习（广度 {ev['brs']} · 证据校准 {ev.get('cal', 0)} · 深度 {ev['drs']}），"
                   f"验证通过 {ev['passes']} 道；证据权重重拟合 {ev.get('refits_accepted', 0)} 次被验收门采纳、"
                   f"{ev.get('refits_rejected', 0)} 次被拒绝。\n\n留出测试集首位命中率 **{ev['hit1_before']} → {ev['hit1_after']}**，"
                   f"真因后验 {ev['partial_before']} → {ev['partial_after']}；生效记忆 {ev['memory_items']} 条。"
                   + ("\n\n当前证据权重：" + "，".join(f"{k} ×{v:.2f}" for k, v in w.items()) if w else ""))
            yield TextMessageStartEvent(type=EventType.TEXT_MESSAGE_START, message_id=mid, role="assistant")
            yield TextMessageContentEvent(type=EventType.TEXT_MESSAGE_CONTENT, message_id=mid, delta=msg)
            yield TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id=mid)
        elif t == "error":
            raise RuntimeError(ev.get("message", "RSI 会话失败"))


@router.post("/agui")
async def agui(request: Request):
    body = await request.json()
    thread_id = body.get("threadId") or body.get("thread_id") or f"t-{uuid.uuid4().hex[:8]}"
    run_id = body.get("runId") or body.get("run_id") or f"run-{uuid.uuid4().hex[:8]}"
    props = body.get("forwardedProps") or body.get("forwarded_props") or {}
    mode = props.get("mode", "rca")
    resume = body.get("resume") or []
    actor = actor_of(request, "console")
    encoder = EventEncoder(accept=request.headers.get("accept"))

    async def stream():
        yield encoder.encode(RunStartedEvent(type=EventType.RUN_STARTED, thread_id=thread_id, run_id=run_id))
        out: dict = {}
        try:
            events = _rsi_events(actor) if mode == "rsi" else _chat_events(mode, body.get("messages") or [], actor, resume, out)
            async for ev in events:
                yield encoder.encode(ev)
            outcome = (RunFinishedInterruptOutcome(type="interrupt", interrupts=[out["interrupt"]]) if out.get("interrupt")
                       else RunFinishedSuccessOutcome(type="success"))
            yield encoder.encode(RunFinishedEvent(type=EventType.RUN_FINISHED, thread_id=thread_id, run_id=run_id, outcome=outcome))
        except Exception as e:  # 错误作为 AG-UI 事件返回，界面可见
            yield encoder.encode(RunErrorEvent(type=EventType.RUN_ERROR, message=str(e)[:300]))

    return StreamingResponse(stream(), media_type=encoder.get_content_type(),
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
