"""RSI 引擎：广度递归自探索（BRS）→ 证据校准（CAL，v2）→ 深度递归自探索（DRS）→ 测试时冻结记忆复用。

v2 在论文方案上加了两点：
  1. 不只学“本厂常见什么根因”（先验），还学“每类证据该信多少”（证据权重，见 rsi/calibration.py）；
  2. 课程按信息价值（知识梯度 × 证据模糊率）选题，并增加一批低成本的校准练习。
可纯内存运行（冷启动基准、消融），也可由 DBOS 工作流驱动并把每一步提交到数据库（见 rsi/workflow.py）。
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable

from ..fab.analytics import analyze_dataset
from ..fab.kg import rank, share
from .agents import Curriculum, Executor, Verifier
from .memory import MemoryState, consolidate_fail, consolidate_pass
from .twin import FabTwin, TaskSpec, eval_set

MAX_ATTEMPTS = 3
CAL_ATTEMPTS = 5  # 校准题允许多试几次：只在前 3 名里找到真因的样本会让权重学偏（选择偏差）


def evaluate(mem: MemoryState) -> dict:
    """在留出测试集上评估冻结的记忆：binary accuracy（首位命中）与 partial score（真因获得的后验概率）。"""
    hits, partial = 0, 0.0
    hw = mem.hotspot_weights()
    data = eval_set()
    for e in data:
        hyps = rank(e["evidence"], e["symptom"], mem.edges, hw, mem.weights)
        hits += bool(hyps and hyps[0].cause_code == e["truth"])
        partial += next((h.posterior for h in hyps if h.cause_code == e["truth"]), 0.0)
    return {"hit1": round(100 * hits / len(data), 2), "partial": round(100 * partial / len(data), 2), "n": len(data)}


def is_weak(ev: dict, sym: str, mem: MemoryState) -> bool:
    """只看证据（不看先验）时首位假设不到五成把握 → 证据模糊，这类题靠先验和证据权重取胜。"""
    hyps = rank(ev, sym, mem.edges, mem.hotspot_weights(), {**(mem.weights or {}), "先验知识": 0.0})
    return not hyps or hyps[0].posterior < 0.5


def practice(twin: FabTwin, mem: MemoryState, spec: TaskSpec, seed: int, executor: Executor, verifier: Verifier,
             max_attempts: int = MAX_ATTEMPTS) -> dict:
    """一道练习题：执行 agent 作答 → 孪生干预反馈 → 验证 agent 判定；失败则换下一个假设（动作—验证循环）。"""
    task = twin.make_task(spec, seed)
    ev = analyze_dataset(task.observation)
    attempts, exclude = [], set()
    for _ in range(max_attempts):
        a = executor.attempt(task, mem, exclude, ev)
        fb = twin.apply_fix(task, a["cause_code"])
        v = verifier.judge(task.requirements, fb, {x["kind"] for x in a["capa"]})
        attempts.append({"cause_code": a["cause_code"], "cause": a["cause"], "posterior": a["posterior"], "ranking": a["ranking"],
                         "memory_used": a["memory_used"], "feedback": fb, "verdict": v})
        if v["passed"]:
            break
        exclude.add(a["cause_code"])
    sym = a["symptom"]
    return {"spec": spec.as_dict(), "seed": seed, "symptom": sym, "evidence": ev, "weak": is_weak(ev, sym, mem),
            "attempts": attempts, "verdict": "PASS" if attempts[-1]["verdict"]["passed"] else "FAIL", "hidden": task.hidden}


def consolidate(mem: MemoryState, result: dict, ref: int | None = None, source: str = "sandbox",
                weight: float | None = None, examples_only: bool = False) -> tuple[list[dict], float, int]:
    """执行 agent 在验证判定之后合并经验。返回（学习事件, 最大份额变化, 新增记忆条数）。

    examples_only：只把验证过的结论记为证据权重的训练样本，不改先验、热点与记忆条目。
    在线模式下的校准练习用它：孪生里“证据长什么样”可以迁移到产线，但“各根因多常见”应以产线历史为准。
    """
    sym, ev = result["symptom"], result["evidence"]
    if examples_only:
        if result["verdict"] == "PASS":
            mem.examples.append((ev, sym, result["attempts"][-1]["cause_code"]))
        return [], 0.0, 0
    before = {c: share(mem.edges[sym], c) for c in mem.edges.get(sym, {})}
    n_items = len(mem.items)
    events = []
    for a in result["attempts"]:
        if not a["verdict"]["passed"]:
            events += consolidate_fail(mem, sym, a["cause_code"], ev, source=source, ref=ref, weight=weight)
    if result["verdict"] == "PASS":
        last = result["attempts"][-1]
        tried = [a["cause_code"] for a in result["attempts"]]
        events += consolidate_pass(mem, sym, last["cause_code"], tried, ev, source=source, ref=ref, weight=weight)
        mem.examples.append((ev, sym, last["cause_code"]))  # 验证过的结论 = 证据权重的训练标签
    after = {c: share(mem.edges[sym], c) for c in mem.edges.get(sym, {})}
    delta = max((abs(after[c] - before.get(c, 0)) for c in after), default=0.0)
    return events, round(delta, 4), len(mem.items) - n_items


Commit = Callable[[dict, list[dict]], Awaitable[None]]
Notify = Callable[[dict], Awaitable[None]]


BREADTH, MAX_DEPTH, PATIENCE, CONCURRENCY = 12, 36, 8, 4
CAL_TASKS, CAL_WAVE, REFIT_EVERY = 480, 24, 4
MODES = {  # 模式 → (广度, 校准, 深度, 学证据权重)
    "none": (False, False, False, False), "brs": (True, False, False, False), "drs": (False, False, True, False),
    "full": (True, False, True, False),  # v0.3：论文方案（只学先验与记忆）
    "full_w": (True, False, True, True),  # + 证据权重学习
    "v2": (True, True, True, True),  # + 证据校准批（Kaizen v0.4 默认）
}


async def run_session(mem: MemoryState, *, mode: str = "v2", breadth: int = BREADTH, concurrency: int = CONCURRENCY,
                      max_depth: int = MAX_DEPTH, patience: int = PATIENCE, seed_base: int = 700_000,
                      cal_tasks: int = CAL_TASKS, sandbox_weight: float = 0.5, cal_updates_prior: bool | None = None,
                      curriculum: Curriculum | None = None,
                      verifier: Verifier | None = None, commit: Commit | None = None, notify: Notify | None = None,
                      twin: FabTwin | None = None) -> dict:
    """mode 见 MODES。sandbox_weight：孪生验证经验的权重。已有产线记忆时取 0.5（让位于真实确认）；新产线冷启动时取 1.0。

    cal_updates_prior：校准练习是否也更新先验。默认只在冷启动（没有产线历史、孪生是唯一知识来源）时更新。
    """
    do_brs, do_cal, do_drs, learn_w = MODES[mode]
    if cal_updates_prior is None:
        cal_updates_prior = sandbox_weight >= 1.0
    curriculum = curriculum or Curriculum(max_depth=max_depth, patience=patience)
    verifier, executor, twin = verifier or Verifier(), Executor(), twin or FabTwin()
    curve = [{"step": 0, "phase": "w/o RSI", **evaluate(mem), "memory": mem.size()}]
    history: list[dict] = []
    seq = 0
    gate = {"accepted": 0, "rejected": 0}

    async def emit(ev: dict) -> None:
        if notify:
            await notify(ev)

    async def record(result: dict, phase: str, step: int) -> dict:
        events, delta, new_items = consolidate(mem, result, weight=sandbox_weight,
                                               examples_only=phase == "CAL" and not cal_updates_prior)
        rec = {**result, "phase": phase, "step": step, "delta": delta, "new_items": new_items, "events": events}
        history.append(rec)
        if commit:
            await commit(rec, events)
        await emit({"type": "task", "phase": phase, "step": step, "symptom": result["symptom"], "verdict": result["verdict"],
                    "attempts": len(result["attempts"]), "variant": result["spec"]["variant"], "delta": delta, "new_items": new_items})
        return rec

    async def refit() -> None:
        if learn_w:
            res = await asyncio.to_thread(mem.refit)
            if res["n"] >= 20:
                gate["accepted" if res["accepted"] else "rejected"] += 1
                await emit({"type": "weights", "accepted": res["accepted"], "weights": res["weights"], "n": res["n"],
                            "gate": res.get("gate"), "reason": res.get("reason", "")})

    async def run_waves(specs: list[TaskSpec], phase: str, width: int, max_attempts: int) -> None:
        nonlocal seq
        for w in range(0, len(specs), width):  # 预算在波次之间检查，不打断进行中的波次
            wave = specs[w:w + width]
            snapshot = mem.clone()  # 同一波次的题都基于波次开始时的记忆作答
            results = await asyncio.gather(*[asyncio.to_thread(practice, twin, snapshot, s, seed_base + seq + i, executor,
                                                               verifier, max_attempts) for i, s in enumerate(wave)])
            seq += len(wave)
            for r in results:
                await record(r, phase, len(history) + 1)
            await refit()
            curve.append({"step": len(history), "phase": phase, **evaluate(mem), "memory": mem.size()})
            await emit({"type": "eval", **curve[-1]})

    if do_brs:
        specs = await curriculum.breadth(mem, history, breadth)
        await emit({"type": "phase", "phase": "BRS", "detail": f"广度探索：{len(specs)} 道题，最多 {concurrency} 道并行"})
        await run_waves(specs, "BRS", concurrency, MAX_ATTEMPTS)

    if do_cal:
        specs = curriculum.calibration(mem, history, cal_tasks, random.Random(seed_base + 31))
        await emit({"type": "phase", "phase": "CAL", "detail": f"证据校准：{len(specs)} 道低成本练习（每波 {CAL_WAVE} 道并行），学习每类证据该信多少"})
        await run_waves(specs, "CAL", CAL_WAVE, CAL_ATTEMPTS)

    if do_drs:
        await emit({"type": "phase", "phase": "DRS", "detail": "深度探索：串行、难度递增，每道题验证后立即固化记忆"})
        step = 0
        while True:
            spec = await curriculum.depth(mem, history, step)
            if spec is None:
                await emit({"type": "phase", "phase": "STOP", "detail": curriculum.stop_reason or "探索结束"})
                break
            r = await asyncio.to_thread(practice, twin, mem, spec, seed_base + seq, executor, verifier)
            seq += 1
            await record(r, "DRS", len(history) + 1)
            if (step + 1) % REFIT_EVERY == 0:
                await refit()
            curve.append({"step": len(history), "phase": "DRS", **evaluate(mem), "memory": mem.size()})
            await emit({"type": "eval", **curve[-1]})
            step += 1
        await refit()
        if learn_w and curve[-1]["phase"] == "DRS":
            curve[-1].update(evaluate(mem))

    passes = sum(1 for t in history if t["verdict"] == "PASS")
    return {"mode": mode, "curve": curve, "tasks": history,
            "summary": {"tasks": len(history), "passes": passes, "brs": sum(1 for t in history if t["phase"] == "BRS"),
                        "cal": sum(1 for t in history if t["phase"] == "CAL"),
                        "drs": sum(1 for t in history if t["phase"] == "DRS"), "weights": mem.weights, "gate": gate,
                        "hit1_before": curve[0]["hit1"], "hit1_after": curve[-1]["hit1"],
                        "partial_before": curve[0]["partial"], "partial_after": curve[-1]["partial"],
                        "memory_items": mem.size()}}


async def cold_start_benchmark(seeds: tuple[int, ...] = (500_000, 610_000, 720_000, 830_000),
                               modes: tuple[str, ...] = ("none", "brs", "drs", "full", "full_w", "v2")) -> dict:
    """新产线冷启动：从教科书先验出发，比较 不做 RSI / 仅广度 / 仅深度 / 完整 RSI 在留出测试集上的结果
    （对应论文图 4 的消融）。每种模式跑多个随机种子，报告均值与最差值，避免挑选幸运的一次。"""
    out: dict = {"seeds": list(seeds), "eval_size": len(eval_set()), "modes": {}}
    for m in modes:
        runs = [await run_session(MemoryState.textbook(), mode=m, seed_base=sd, sandbox_weight=1.0) for sd in seeds]
        hits = [r["summary"]["hit1_after"] for r in runs]
        parts = [r["summary"]["partial_after"] for r in runs]
        out["modes"][m] = {"hit1_mean": round(sum(hits) / len(hits), 2), "hit1_min": min(hits), "hit1_max": max(hits),
                           "partial_mean": round(sum(parts) / len(parts), 2), "tasks_mean": round(sum(r["summary"]["tasks"] for r in runs) / len(runs), 1),
                           "weights": runs[0]["summary"]["weights"], "curve": runs[0]["curve"]}
    out["baseline_hit1"] = out["modes"].get("none", {}).get("hit1_mean")
    out["oracle"] = oracle_ceiling()
    return out


def oracle_ceiling() -> dict:
    """参考天花板：知道孪生真实根因分布（先验）且证据权重用留出集本身拟合时能达到的命中率——任何学习都不可能稳定超过它。"""
    from ..fab.kg import EdgeState
    from .calibration import features, fit
    from .twin import SITE_TRUTH

    mem = MemoryState.textbook()
    for sym, dist in SITE_TRUTH.items():
        for c, p in dist.items():
            if c in mem.edges.get(sym, {}):
                e: EdgeState = mem.edges[sym][c]
                e.alpha = e.alpha0 + 400 * p
                e.beta = e.beta0 + 400 * (1 - p)
    hw = mem.hotspot_weights()
    res = fit([(features(e["evidence"], e["symptom"], mem.edges, hw), e["truth"]) for e in eval_set()], lam=0.1)
    prior_only = evaluate(mem)
    mem.weights = res["weights"]
    return {"prior_only": prior_only["hit1"], "hit1": evaluate(mem)["hit1"], "weights": res["weights"]}
