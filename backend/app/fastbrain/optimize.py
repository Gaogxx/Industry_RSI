"""RSI-Opt（arXiv:2610.32026）用在快脑小模型的训练上：两层。

内层（单次训练之内）：训练循环在每个 epoch 末停下来，由控制器决定继续 / 学习率减半 / 回滚到最好的一轮 / 提前停止
  （见 model.fit；控制器启用哪些规则由训练知识库决定）。
外层（跨 run）：进化“怎么训练”——训练策略（学习率、轮数、网络结构、正则、出题配比、产线样本权重）。
  · 存档：所有候选（包括被拒的）留档，跨 run 继承；父代采样 ∝ sigmoid(得分 − 中位数) / (1 + 子代数)。
  · 提议：每次改 1~2 个旋钮，往哪个方向改参考训练知识库里该方向的历史成败（tip:policy:*）；停用的方向几乎不再尝试。
  · 验收：每个候选从贝叶斯初始化重新训练（公平比较），在 search 池上训练、select 集上评估；
          RSI-Opt 原文没有验收门、留出任务第 1 代后就不再提升——这里补上：
          对数损失配对区间上界 < 0（显著更好）才接受；落在噪声带内时，只有更便宜（训练步数少 20% 或参数少）才接受；
          命中率配对区间下界不得低于 −2 个百分点。最后只用一次封存的 report 集决定是否换掉在用策略。
  · 何时跑：事件触发，不定时——还没有任何策略存档；或连续 2 个训练窗口没通过健康门 / 晋升门（训练做法可能过时）；
          连续 2 次外层进化都没有接受任何候选则退避，直到出现新的触发信号。
"""

from __future__ import annotations

import math
import random
import uuid

from sqlmodel import Session, col, select

from ..models import AuditLog, FastBrainModel, TrainingPolicy, utcnow
from ..rsi.memory import MemoryState
from ..rsi.twin import FabTwin
from ..stats import paired_ci
from . import playbook
from .data import frozen_split, pool_samples, split_val
from .features import bayes_init
from .model import FastModel, evaluate, fit

# 旋钮 → (可选值, 默认值, 说明)。只有这些键可以被进化（白名单）
SPACE: dict[str, tuple[list, object, str]] = {
    "lr": ([0.005, 0.01, 0.02, 0.04], 0.01, "学习率（Adam）"),
    "epochs": ([20, 30, 40, 60], 40, "最多训练轮数（控制器可能提前停止）"),
    "hidden": ([4, 8, 12], 8, "深部分隐层宽度"),
    "rank": ([4], 4, "DCN-v2 交叉层的秩（固定开启，不参与进化）"),
    "emb": ([0, 4], 4, "根因 ID 嵌入维度（0 = 不用嵌入）"),
    "l2": ([1e-4, 3e-4, 1e-3, 3e-3], 3e-4, "L2 正则（宽部分锚定在贝叶斯权重）"),
    "bias_l2": ([3e-4, 1e-3, 3e-3, 1e-2], 1e-3, "根因偏置层的 L2"),
    "anchor": ([0.0, 0.001, 0.003, 0.01], 0.003, "连续训练时锚定上一版参数的强度（防遗忘）"),
    "patience": ([2, 3, 4], 3, "控制器：几轮验证损失不创新低算平台期"),
    "hard_share": ([0.3, 0.4, 0.5, 0.6, 0.7], 0.5, "训练池里困难题的比例"),
    "decoy_share": ([0.1, 0.2, 0.33, 0.45], 0.33, "训练池里 PM 干扰题的比例"),
    "receipt_weight": ([1.0, 2.0, 3.0, 4.0], 2.0, "产线确认样本相对孪生样本的权重"),
}
DEFAULT = {k: v[1] for k, v in SPACE.items()}
ARCH = ("hidden", "rank", "emb")
SPLITS = {"search": (5_100_000, 120), "select": (5_200_000, 25), "report": (5_300_000, 25)}
NON_INFERIOR = -0.02
FAIL_STREAK = 2


def clamp(params: dict) -> dict:
    out = dict(DEFAULT)
    for k, v in (params or {}).items():
        if k in SPACE:
            opts = SPACE[k][0]
            out[k] = min(opts, key=lambda o: abs(o - v))
    out["batch"] = 32
    return out


def active_policy(db: Session) -> tuple[int, dict]:
    row = db.exec(select(TrainingPolicy).where(TrainingPolicy.status == "active").order_by(col(TrainingPolicy.id).desc())).first()
    return (row.id, clamp(row.params)) if row else (0, clamp({}))


def pending_triggers(db: Session) -> list[str]:
    """外层何时跑（事件触发）。"""
    out = []
    if db.exec(select(TrainingPolicy)).first() is None:
        out.append("bootstrap")
    last_run = db.exec(select(TrainingPolicy).order_by(col(TrainingPolicy.id).desc())).first()
    q = select(FastBrainModel).order_by(col(FastBrainModel.id).desc())
    if last_run is not None:
        q = q.where(col(FastBrainModel.created_at) > last_run.created_at)
    recent = db.exec(q.limit(FAIL_STREAK)).all()
    if len(recent) == FAIL_STREAK and all(r.status in ("failed", "rejected") for r in recent):
        out.append(f"windows_failing:{FAIL_STREAK}")
    if out and "bootstrap" not in out:  # 退避：最近连续 2 次外层进化都没接受任何候选，等新的窗口结果再说
        runs = db.exec(select(TrainingPolicy.run_id).distinct().order_by(col(TrainingPolicy.id).desc()).limit(FAIL_STREAK * 8)).all()
        last_two = list(dict.fromkeys(runs))[:FAIL_STREAK]
        if len(last_two) == FAIL_STREAK and all(
                not db.exec(select(TrainingPolicy).where(TrainingPolicy.run_id == r, col(TrainingPolicy.status).in_(["accepted", "active"]))).first()
                for r in last_two) and not recent:
            out = []
    return out


def _cost(pol: dict, log: dict, model: FastModel) -> dict:
    return {"steps": log["steps"], "params": model.n_params}


async def _train_eval(pol: dict, mem: MemoryState, pools: dict, evals: list[dict], seed: int, ctl: dict, twin) -> dict:
    key = tuple(sorted((k, v) for k, v in pol.items() if k in ("hard_share", "decoy_share")))
    if key not in pools:
        seed0, per = SPLITS["search"]
        pools[key] = await pool_samples(mem, seed0, per, {"hard_share": pol["hard_share"], "decoy_share": pol["decoy_share"]}, twin)
    train, val = split_val(pools[key], seed=seed)
    m = FastModel(pol["hidden"], init_weights=mem.weights, seed=seed, rank=pol["rank"], emb=pol["emb"])
    log = fit(m, train, val, {**pol, "anchor": 0.0}, a0=bayes_init(mem.weights), seed=seed, controller=ctl)
    return {"model": m, "log": log, "eval": [evaluate(m, e) for e in evals], "cost": _cost(pol, log, m)}


def _public(ev: dict) -> dict:
    return {k: v for k, v in ev.items() if not k.startswith("_")}


# anchor 只在连续训练里起作用（外层从贝叶斯初始化公平比较时不搜索它）；只有一个取值的旋钮（交叉层秩，固定开启）也不搜索
SEARCHED = [k for k in SPACE if k != "anchor" and len(SPACE[k][0]) > 1]


def _propose(db: Session, parent: dict, n_edits: int, rng: random.Random) -> tuple[dict, list[dict]]:
    names = SEARCHED
    cand, edits = dict(parent), []
    picked: list[str] = []
    while len(picked) < n_edits:
        n = rng.choice(names)
        if n not in picked:
            picked.append(n)
    for n in picked:
        opts = SPACE[n][0]
        i = opts.index(min(opts, key=lambda o: abs(o - parent[n])))
        dirs = [d for d in (-1, 1) if 0 <= i + d < len(opts)]
        w = [playbook.tip_weight(db, n, d) for d in dirs]
        d = rng.choices(dirs, weights=w)[0]
        cand[n] = opts[i + d]
        edits.append({"component": n, "from": parent[n], "to": cand[n], "direction": d})
    return clamp(cand), edits


def _sample_parent(archive: list[dict], rng: random.Random) -> dict:
    med = sorted(a["score"] for a in archive)[len(archive) // 2]
    w = [1 / (1 + math.exp(-(a["score"] - med) * 20)) / (1 + a["children"]) for a in archive]
    return rng.choices(archive, weights=w)[0]


async def evolve_training(db: Session, mem: MemoryState, *, generations: int = 2, children: int = 2, when=None, seed: int = 5,
                          twin: FabTwin | None = None, trigger: str = "manual") -> dict:
    when = when or utcnow()
    twin = twin or FabTwin()
    run_id = f"rsiopt-{when:%Y%m%d%H%M}-{uuid.uuid4().hex[:4]}"
    rng = random.Random(seed)
    ctl = playbook.guidance(db)["controller"]
    inc_id, inc = active_policy(db)
    sel = await frozen_split(mem, SPLITS["select"][0], SPLITS["select"][1], twin)
    pools: dict = {}
    base = await _train_eval(inc, mem, pools, [sel], seed, ctl, twin)
    if inc_id == 0:  # 第一次：默认策略作为基线存档并启用，之后的候选都以它为父代
        row = TrainingPolicy(run_id=run_id, params=inc, status="active", note="初始训练策略（基线）",
                             scores={"select": _public(base["eval"][0]), "cost": base["cost"]}, created_at=when)
        db.add(row)
        db.flush()
        inc_id = row.id
    best = {"params": inc, "id": inc_id, "res": base}
    past = db.exec(select(TrainingPolicy).where(col(TrainingPolicy.status).in_(["accepted", "active", "retired"]))).all()
    archive = [{"params": inc, "score": -base["eval"][0]["logloss"], "children": 0, "id": inc_id}]
    archive += [{"params": clamp(p.params), "score": -((p.scores.get("select") or {}).get("logloss") or 9), "children": 0, "id": p.id}
                for p in past if p.id != inc_id][-6:]
    log = []
    for gen in range(generations):
        n_edits = max(1, 2 - gen)  # 改动数随代数减少
        for _ in range(children):
            parent = _sample_parent(archive, rng)
            parent["children"] += 1
            cand, edits = _propose(db, parent["params"], n_edits, rng)
            if cand == parent["params"]:
                continue
            res = await _train_eval(cand, mem, pools, [sel], seed, ctl, twin)
            ev, bev = res["eval"][0], best["res"]["eval"][0]
            mean, lo, hi = paired_ci(ev["_ll"], bev["_ll"])
            hmean, hlo, hhi = paired_ci(ev["_correct"], bev["_correct"])
            cheaper = (res["cost"]["steps"] < 0.8 * best["res"]["cost"]["steps"]
                       or res["cost"]["params"] < best["res"]["cost"]["params"])
            status = "rejected"
            if hlo < NON_INFERIOR:
                note = f"select 集命中率可能退步（配对差下界 {hlo * 100:+.1f} 个百分点）"
            elif hi < 0:
                status, note = "accepted", f"select 集对数损失显著下降（配对差 95% 区间 [{lo:+.3f}, {hi:+.3f}]）"
            elif mean <= 0.005 and cheaper:
                status, note = "accepted", "对数损失在噪声带内，但训练更便宜（步数少 20% 以上或参数更少）：复杂度感知接受"
            else:
                note = f"对数损失改进不显著（[{lo:+.3f}, {hi:+.3f}]）" + ("" if cheaper else "，也没有更便宜")
            row = TrainingPolicy(run_id=run_id, parent_id=parent["id"] or None, params=cand, edits=edits, status=status, note=note,
                                 scores={"select": _public(ev), "cost": res["cost"], "generation": gen,
                                         "controller": [d["action"] for d in res["log"]["decisions"]],
                                         "diff_logloss": {"mean": mean, "ci95": [lo, hi]}, "diff_hit1": {"mean": hmean, "ci95": [hlo, hhi]}},
                                 created_at=when)
            db.add(row)
            db.flush()
            for e in edits:
                playbook.record_policy_edit(db, e["component"], e["direction"], status == "accepted", run_id, note, when)
            log.append({"id": row.id, "status": status, "edits": edits, "note": note})
            if status == "accepted":
                best = {"params": cand, "id": row.id, "res": res}
                archive.append({"params": cand, "score": -ev["logloss"], "children": 0, "id": row.id})
    promoted, report = False, None
    if best["id"] != inc_id:  # 只在最后用一次封存的 report 集
        rep = await frozen_split(mem, SPLITS["report"][0], SPLITS["report"][1], twin)
        r_inc = (await _train_eval(inc, mem, pools, [rep], seed, ctl, twin))["eval"][0]
        r_new = (await _train_eval(best["params"], mem, pools, [rep], seed, ctl, twin))["eval"][0]
        mean, lo, hi = paired_ci(r_new["_ll"], r_inc["_ll"])
        hmean, hlo, hhi = paired_ci(r_new["_correct"], r_inc["_correct"])
        promoted = mean <= 0.005 and hlo >= NON_INFERIOR
        report = {"incumbent": _public(r_inc), "candidate": _public(r_new), "diff_logloss": {"mean": mean, "ci95": [lo, hi]},
                  "diff_hit1": {"mean": hmean, "ci95": [hlo, hhi]}}
        row = db.get(TrainingPolicy, best["id"])
        row.scores = {**row.scores, "report": report}
        if promoted:
            for old in db.exec(select(TrainingPolicy).where(TrainingPolicy.status == "active")).all():
                old.status = "retired"
                db.add(old)
            row.status, row.note = "active", row.note + "；report 集不退步，换为在用训练策略"
        db.add(row)
    db.add(AuditLog(actor="system:rsi-opt", action="training.evolve", target=run_id,
                    detail={"trigger": trigger, "candidates": len(log), "promoted": promoted}, created_at=when))
    db.commit()
    return {"run_id": run_id, "trigger": trigger, "incumbent": {"id": inc_id, "params": inc, "select": _public(base["eval"][0])},
            "candidates": log, "best": {"id": best["id"], "params": best["params"]}, "report": report, "promoted": promoted}
