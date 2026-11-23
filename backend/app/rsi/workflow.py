"""在线 RSI 探索会话：DBOS 持久化工作流，作用于数据库中的真实记忆（孪生经验权重 0.5）。

  s_eval（不做 RSI 的基线）→ s_breadth_plan（课程 agent 出广度题）→ s_wave × N（每波最多 4 题并行）
  → s_cal_plan → s_wave × N（证据校准：每波 24 道，只学证据权重，不改产线先验）
  → s_depth_plan / s_wave 循环（串行、按信息价值选题，课程 agent 决定何时停）→ s_finish
每波结束可重拟合证据权重：候选权重必须在封存的验收门面板上不退步才会被采纳并保存为新版本。

每个步骤都会从数据库加载最新记忆、完成后写回，并把练习题、判定、记忆变化落库；任何一步崩溃后都从断点继续，
不会重复练习或重复写入记忆。进度经 DBOS Stream "kaizen" 推送给控制台（AG-UI 事件）。
"""

from __future__ import annotations

import asyncio
import random

from dbos import DBOS, SetWorkflowID
from sqlmodel import col, select

from ..config import get_settings
from ..db import session_scope
from ..llm import LLM
from ..models import AuditLog, LearningEvent, RSIBenchmark, RSISession, RSITask, utcnow

from .agents import Curriculum, Executor, Verifier, knowledge_gaps
from .engine import (BREADTH, CAL_ATTEMPTS, CAL_TASKS, CAL_WAVE, CONCURRENCY, MAX_ATTEMPTS, MAX_DEPTH, PATIENCE, REFIT_EVERY,
                     cold_start_benchmark, consolidate, evaluate, practice)
from .memory import load_memory, refit_and_store, save_memory
from .twin import FabTwin, TaskSpec

STREAM = "kaizen"
LIVE_SANDBOX_WEIGHT = 0.5


def _history(db, session_id: int) -> list[dict]:
    rows = db.exec(select(RSITask).where(RSITask.session_id == session_id).order_by(col(RSITask.id))).all()
    return [{"spec": r.spec, "verdict": r.verdict, "phase": r.phase, "weak": r.weak,
             "delta": max((abs(e.get("after", 0) - e.get("before", 0)) for e in r.memory_changes
                           if e.get("kind") in ("confirm", "false_lead", "refute")), default=0.0),
             "new_items": sum(1 for e in r.memory_changes if e.get("kind") == "memory_add")} for r in rows]


def _curriculum() -> Curriculum:
    s = get_settings()
    llm = LLM(model_name=s.rsi_curriculum_model or None) if not s.offline else None
    return Curriculum(llm=llm, max_depth=MAX_DEPTH, patience=PATIENCE)


async def _append_point(session_id: int, point: dict) -> None:
    with session_scope() as db:
        sess = db.get(RSISession, session_id)
        sess.curve = list(sess.curve) + [point]
        db.add(sess)
        db.commit()
    await DBOS.write_stream_async(STREAM, {"type": "eval", **point})


@DBOS.step()
async def s_eval(session_id: int, phase: str, step: int) -> dict:
    with session_scope() as db:
        mem = load_memory(db)
    point = {"step": step, "phase": phase, **evaluate(mem), "memory": mem.size()}
    await _append_point(session_id, point)
    return point


@DBOS.step()
async def s_breadth_plan(session_id: int) -> list[dict]:
    with session_scope() as db:
        mem, hist = load_memory(db), _history(db, session_id)
    specs = await _curriculum().breadth(mem, hist, BREADTH)
    await DBOS.write_stream_async(STREAM, {"type": "phase", "phase": "BRS",
                                           "detail": f"课程 agent 出 {len(specs)} 道广度题（每波最多 {CONCURRENCY} 道并行）",
                                           "specs": [s.as_dict() for s in specs]})
    return [s.as_dict() for s in specs]


@DBOS.step()
async def s_cal_plan(session_id: int) -> list[dict]:
    with session_scope() as db:
        mem, hist = load_memory(db), _history(db, session_id)
    specs = _curriculum().calibration(mem, hist, CAL_TASKS, random.Random(session_id * 7919))
    await DBOS.write_stream_async(STREAM, {"type": "phase", "phase": "CAL",
                                           "detail": f"证据校准：{len(specs)} 道低成本练习（每波 {CAL_WAVE} 道并行），"
                                                     "只学“每类证据该信多少”，不改动产线先验"})
    return [s.as_dict() for s in specs]


@DBOS.step()
async def s_depth_plan(session_id: int, step: int) -> dict | None:
    with session_scope() as db:
        mem, hist = load_memory(db), _history(db, session_id)
    cur = _curriculum()
    spec = await cur.depth(mem, hist, step)
    if spec is None:
        await DBOS.write_stream_async(STREAM, {"type": "phase", "phase": "STOP", "detail": cur.stop_reason or "探索结束"})
        return None
    await DBOS.write_stream_async(STREAM, {"type": "plan", "phase": "DRS", "spec": spec.as_dict()})
    return spec.as_dict()


def _refit(db, session_id: int, phase: str) -> dict:
    res = refit_and_store(db, load_memory(db), source="rsi")
    rec = {"phase": phase, "accepted": res["accepted"], "weights": res["weights"], "n": res["n"],
           "gate": res.get("gate"), "reason": res.get("reason", "")}
    sess = db.get(RSISession, session_id)
    sess.weights_log = list(sess.weights_log) + [rec]
    db.add(sess)
    return rec


@DBOS.step()
async def s_wave(session_id: int, phase: str, specs: list[dict], seq: int, refit: bool = False) -> dict:
    """一波练习：执行 agent 作答 → 孪生干预反馈 → 验证 agent 判定 → 执行 agent 在判定后合并记忆（→ 证据权重重拟合）。"""
    twin, executor, verifier = FabTwin(), Executor(), Verifier()
    with session_scope() as db:
        mem = load_memory(db)
        base = len(db.exec(select(RSITask).where(RSITask.session_id == session_id)).all())
    snapshot = mem.clone()
    seed_base = 2_000_000 + session_id * 1000  # 与测试集（90 万段）、验收门（96 万段）的种子不重叠
    attempts = CAL_ATTEMPTS if phase == "CAL" else MAX_ATTEMPTS
    results = await asyncio.gather(*[asyncio.to_thread(practice, twin, snapshot, TaskSpec(**sp), seed_base + seq + i, executor,
                                                       verifier, attempts) for i, sp in enumerate(specs)])
    summary, weights = [], None
    with session_scope() as db:
        for i, r in enumerate(results):
            task = RSITask(session_id=session_id, phase=phase, step=base + i + 1, spec=r["spec"], verdict=r["verdict"],
                           attempts=r["attempts"], hidden=r["hidden"], evidence=r["evidence"], symptom=r["symptom"],
                           weak=r["weak"])
            db.add(task)
            db.flush()
            events, delta, new_items = consolidate(mem, r, ref=task.id, weight=LIVE_SANDBOX_WEIGHT, examples_only=phase == "CAL")
            task.memory_changes = events
            db.add(task)
            for e in events:
                db.add(LearningEvent(rsi_task_id=task.id, **{k: v for k, v in e.items() if k in LearningEvent.model_fields}))
            summary.append({"id": task.id, "symptom": r["symptom"], "verdict": r["verdict"], "attempts": len(r["attempts"]),
                            "variant": r["spec"]["variant"], "delta": delta, "new_items": new_items})
        save_memory(db, mem)
        db.flush()
        if refit:
            weights = _refit(db, session_id, phase)
        db.commit()
        mem = load_memory(db)
    if phase == "CAL":  # 校准题量大，按波次汇总推送
        await DBOS.write_stream_async(STREAM, {"type": "cal", "phase": phase, "n": len(summary),
                                               "passes": sum(1 for x in summary if x["verdict"] == "PASS"),
                                               "done": base + len(summary)})
    else:
        for x in summary:
            await DBOS.write_stream_async(STREAM, {"type": "task", "phase": phase, **x})
    if weights:
        await DBOS.write_stream_async(STREAM, {"type": "weights", **weights})
    point = {"step": base + len(results), "phase": phase, **evaluate(mem), "memory": mem.size()}
    await _append_point(session_id, point)
    return {"tasks": summary, "point": point, "weights": weights}


@DBOS.step()
async def s_finish(session_id: int, status: str = "done") -> dict:
    with session_scope() as db:
        sess = db.get(RSISession, session_id)
        tasks = db.exec(select(RSITask).where(RSITask.session_id == session_id)).all()
        curve = sess.curve
        sess.status, sess.finished_at = status, utcnow()
        log = sess.weights_log
        sess.summary = {"tasks": len(tasks), "passes": sum(1 for t in tasks if t.verdict == "PASS"),
                        "brs": sum(1 for t in tasks if t.phase == "BRS"), "cal": sum(1 for t in tasks if t.phase == "CAL"),
                        "drs": sum(1 for t in tasks if t.phase == "DRS"),
                        "refits_accepted": sum(1 for w in log if w["accepted"]), "refits_rejected": sum(1 for w in log if not w["accepted"]),
                        "weights": next((w["weights"] for w in reversed(log) if w["accepted"]), None),
                        "hit1_before": curve[0]["hit1"] if curve else None, "hit1_after": curve[-1]["hit1"] if curve else None,
                        "partial_before": curve[0]["partial"] if curve else None,
                        "partial_after": curve[-1]["partial"] if curve else None,
                        "memory_items": curve[-1]["memory"] if curve else None}
        db.add(sess)
        db.add(AuditLog(actor=sess.trigger, action="kaizen.session", target=f"RSI-{session_id}", detail=sess.summary))
        db.commit()
        summary = sess.summary
    await DBOS.write_stream_async(STREAM, {"type": "done", "session_id": session_id, **summary})
    await DBOS.close_stream_async(STREAM)
    return summary


@DBOS.step()
async def s_final_refit(session_id: int, step: int) -> dict:
    with session_scope() as db:
        rec = _refit(db, session_id, "DRS")
        db.commit()
        mem = load_memory(db)
        n = len(db.exec(select(RSITask).where(RSITask.session_id == session_id)).all())
    await DBOS.write_stream_async(STREAM, {"type": "weights", **rec})
    if rec["accepted"]:
        await _append_point(session_id, {"step": n, "phase": "DRS", **evaluate(mem), "memory": mem.size()})
    return rec


@DBOS.workflow()
async def rsi_workflow(session_id: int) -> dict:
    try:
        await s_eval(session_id, "w/o RSI", 0)
        specs = await s_breadth_plan(session_id)
        seq = 0
        for w in range(0, len(specs), CONCURRENCY):  # 预算在波次之间检查
            wave = specs[w:w + CONCURRENCY]
            await s_wave(session_id, "BRS", wave, seq, refit=w + CONCURRENCY >= len(specs))
            seq += len(wave)
        cal = await s_cal_plan(session_id)
        for w in range(0, len(cal), CAL_WAVE):
            wave = cal[w:w + CAL_WAVE]
            await s_wave(session_id, "CAL", wave, seq, refit=True)
            seq += len(wave)
        await DBOS.write_stream_async(STREAM, {"type": "phase", "phase": "DRS",
                                               "detail": "深度探索：串行、按信息价值选题，每题验证后立即固化记忆"})
        step = 0
        while True:
            spec = await s_depth_plan(session_id, step)
            if spec is None:
                break
            await s_wave(session_id, "DRS", [spec], seq, refit=(step + 1) % REFIT_EVERY == 0)
            seq, step = seq + 1, step + 1
        await s_final_refit(session_id, step)
        return await s_finish(session_id)
    except Exception as e:
        await DBOS.write_stream_async(STREAM, {"type": "error", "message": str(e)[:300]})
        await s_finish(session_id, "error")
        raise


async def start_session(trigger: str = "manual", return_handle: bool = False):
    with session_scope() as db:
        sess = RSISession(trigger=trigger, config={"breadth": BREADTH, "concurrency": CONCURRENCY, "cal_tasks": CAL_TASKS,
                                                   "cal_wave": CAL_WAVE, "max_depth": MAX_DEPTH, "patience": PATIENCE,
                                                   "sandbox_weight": LIVE_SANDBOX_WEIGHT})
        db.add(sess)
        db.commit()
        db.refresh(sess)
        sid = sess.id
    wf_id = f"rsi-session-{sid}"
    with session_scope() as db:
        sess = db.get(RSISession, sid)
        sess.workflow_id = wf_id
        db.add(sess)
        db.commit()
    with SetWorkflowID(wf_id):
        handle = await DBOS.start_workflow_async(rsi_workflow, sid)
    return (sid, wf_id, handle) if return_handle else (sid, wf_id)


async def run_benchmark(trigger: str = "manual") -> int:
    handle = await DBOS.start_workflow_async(benchmark_workflow, trigger)
    return await handle.get_result()


@DBOS.step()
async def s_benchmark(trigger: str) -> int:
    res = await cold_start_benchmark()
    with session_scope() as db:
        b = RSIBenchmark(trigger=trigger, results=res)
        db.add(b)
        db.commit()
        db.refresh(b)
        return b.id


@DBOS.workflow()
async def benchmark_workflow(trigger: str = "manual") -> int:
    return await s_benchmark(trigger)


def gaps_view() -> list[dict]:
    with session_scope() as db:
        return knowledge_gaps(load_memory(db), [])
