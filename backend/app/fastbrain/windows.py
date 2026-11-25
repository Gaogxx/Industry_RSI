"""快脑小模型的训练窗口（参照 A-RSI，arXiv:2610.38568：何时更新、最后是否更新）。

何时开窗（事件触发，任一满足）：
  bootstrap        还没有训练过任何窗口
  new_labels:N     上个窗口之后新增 ≥ 20 条工程师确认的收据
  drift            孪生同步了产线变更（同时强制换一批新的训练池）
  pool_saturated   上个窗口判定样本池已饱和（见下）
  slow_overrule:N  慢脑在上个窗口之后推翻快脑 ≥ 8 次（System 2 的结论回流到 System 1；被推翻多的症状配额加倍）
  line_drop        在用模型在新确认收据上的命中率比它的留出集命中率低 10 个百分点以上

窗口怎么训：固定步数（最多若干 epoch），从上一版被接受的参数继续（连续训练，锚定上一版防遗忘）；
  样本池由课程 agent 按固定配额出题，标签由孪生程序给出；再加上产线确认的收据（权重由训练策略决定）。
换池：样本池里“仍有信息量”（在用模型给真因的概率 < 0.7）的比例跌到 2 个窗口前的 85% 以下（同一池至少用过 3 个窗口），
  或在用模型在池上的命中率比该池第一次使用时高出 10 个百分点 → 下个窗口换一批新题（A-RSI 的换池规则）。
健康门（训练本身正不正常）：参数有限且没有爆炸（|θ| ≤ 50）、验证损失确实下降、参数确实变了、输出没有塌缩
  （平均首位概率在 0.2~0.995）、没有过拟合（验证损失 − 训练损失 ≤ 0.15）、校准温度在 0.5~2.0 之间、
  没有哪个症状在验证集上比贝叶斯差 0.25 以上（分组护栏）、样本够。
  不过 → 换一批种子重试一次；再不过 → 停（halt），保留上一版。
晋升门（新模型能不能上线；A-RSI 没有留出集晋升门，这在工业场景不够，这里补上）：
  留出集每晋升一次就换一块全新的（孪生可以无限出题，不让同一块留出集被反复使用）；在这块留出集上
  ① 对数损失：候选 − 在用版本的配对 bootstrap 95% 区间上界 ≤ 0；② 命中率：配对区间下界 ≥ −2 个百分点；
  ③ ECE 不变差（容差 0.01）；④ 对贝叶斯：命中率配对区间下界 ≥ −2 个百分点；
  ⑤ 分组护栏：任何一个症状的对数损失不比在用版本差 0.2 以上；⑥ 保形集合实际覆盖 ≥ 87%（承诺 90%，留 3 个点抽样余量）；
  ⑦ 产线时间留出（最近 20% 已确认的收据，不进训练）：满 10 条时，比在用版本、比贝叶斯净多答错都不到 3 例
     （样本少，按配对的净差例数判，不按百分比——11 条里差 1 例就是 9 个百分点）。
  全过才晋升为 active，拿到快脑的裁决权。
"""

from __future__ import annotations

import math

from sqlmodel import Session, col, func, select

from ..models import AuditLog, FastBrainModel, HarnessReceipt, utcnow
from ..rsi.memory import MemoryState
from ..rsi.twin import FabTwin, eval_set
from ..stats import paired_ci
from . import optimize, playbook
from .data import (CAL_PER_SYMPTOM, CAL_SEED, HOLDOUT_BASE, HOLDOUT_PER_SYMPTOM, HOLDOUT_ROUNDS, HOLDOUT_SPAN, POOL_SEED, frozen_split,
                   pool_samples, receipt_samples, split_val, to_sample)
from .features import bayes_init
from .model import FastModel, calibrate_temperature, evaluate, fit

NEW_LABELS, OVERRULES, LINE_DROP, LINE_NET = 20, 8, 10.0, 3
CONFORMAL_ALPHA = 0.10  # 保形预测：给出覆盖真因 90% 的根因集合（可靠性规格，不参与进化）
POOL_PER_SYMPTOM = 300
INFO_P, INFO_DROP, INFO_MIN_WINDOWS, INFO_LAG, SATURATION = 0.7, 0.85, 3, 2, 10.0
NON_INFERIOR = -0.02


def active_row(db: Session) -> FastBrainModel | None:
    return db.exec(select(FastBrainModel).where(FastBrainModel.status == "active").order_by(col(FastBrainModel.id).desc())).first()


def load_active(db: Session) -> tuple[FastBrainModel | None, FastModel | None]:
    row = active_row(db)
    return (row, FastModel.from_dict(row.params)) if row else (None, None)


def _since(db: Session) -> FastBrainModel | None:
    return db.exec(select(FastBrainModel).order_by(col(FastBrainModel.id).desc())).first()


def overrules(db: Session, since=None) -> dict[str, int]:
    """慢脑推翻快脑的次数（按症状）：慢脑在探测题上明显赢过快脑、最终首位与快脑不同。"""
    q = select(HarnessReceipt).where(HarnessReceipt.route == "slow")
    if since is not None:
        q = q.where(col(HarnessReceipt.created_at) > since)
    out: dict[str, int] = {}
    for r in db.exec(q).all():
        chosen = ((r.inner or {}).get("arbiter") or {}).get("chosen")
        if chosen and chosen != "fast" and r.top != r.top_fast:
            out[r.symptom] = out.get(r.symptom, 0) + 1
    return out


def pending_triggers(db: Session) -> list[str]:
    last = _since(db)
    since = last.created_at if last else None
    out = []
    if last is None:
        out.append("bootstrap")
    q = select(func.count(col(HarnessReceipt.id))).where(col(HarnessReceipt.verified_at).is_not(None))
    n_new = db.exec(q.where(col(HarnessReceipt.verified_at) > since) if since else q).one()
    if n_new >= NEW_LABELS:
        out.append(f"new_labels:{n_new}")
    aq = select(func.count(col(AuditLog.id))).where(AuditLog.action == "twin.update")
    if since and db.exec(aq.where(col(AuditLog.created_at) > since)).one():
        out.append("drift")
    if last and (last.window or {}).get("refresh_next"):
        out.append("pool_saturated")
    ov = sum(overrules(db, since).values()) if since else 0
    if ov >= OVERRULES:
        out.append(f"slow_overrule:{ov}")
    act = active_row(db)
    if act is not None and since is not None:
        rows = db.exec(select(HarnessReceipt).where(HarnessReceipt.fast_model == act.id, col(HarnessReceipt.verified_at) > since)).all()
        rows = [r for r in rows if "fast_match" in (r.outcome or {})]
        hold = ((act.metrics or {}).get("candidate") or {}).get("hit1")
        if len(rows) >= 10 and hold is not None:
            hit = 100 * sum(1 for r in rows if r.outcome["fast_match"]) / len(rows)
            if hit < hold - LINE_DROP:
                out.append(f"line_drop:{hit:.0f}<{hold:.0f}")
    return out


def calibrate_conformal(scores: list[float], alpha: float = CONFORMAL_ALPHA) -> float:
    """分割保形：分数 = 1 − 模型给真因的概率；阈值 q 取校准集分数的 ⌈(n+1)(1−α)⌉/n 分位。"""
    s = sorted(scores)
    n = len(s)
    if not n:
        return 1.0
    k = min(n - 1, max(0, math.ceil((n + 1) * (1 - alpha)) - 1))
    return round(s[k], 4)


def truth_scores(model: FastModel, data: list[dict]) -> list[float]:
    out = []
    for s in data:
        if s["y"] is None or not s["X"]:
            out.append(1.0)
            continue
        out.append(1 - model.probs(s["X"], s["codes"])[s["y"]])
    return out


def _sym_ll(model: FastModel, data: list[dict]) -> dict[str, float]:
    ev = evaluate(model, data)
    by: dict[str, list[float]] = {}
    for s, ll in zip([s for s in data if s["X"]], ev["_ll"]):
        by.setdefault(s["sym"], []).append(ll)
    return {k: sum(v) / len(v) for k, v in by.items()}


def _groups(model: FastModel, data: list[dict]) -> dict:
    out = {}
    for g, pick in (("decoy", lambda s: s["decoy"]), ("hard", lambda s: s["difficulty"] == "hard")):
        a = [s for s in data if pick(s)]
        b = [s for s in data if not pick(s)]
        if a and b:
            out[g] = {"hit1": evaluate(model, a)["hit1"], "rest": evaluate(model, b)["hit1"], "n": len(a)}
    return out


def _pub(ev: dict) -> dict:
    return {k: v for k, v in ev.items() if not k.startswith("_")}


async def train_window(db: Session, mem: MemoryState, trigger: str, when=None, twin: FabTwin | None = None) -> FastBrainModel:
    when = when or utcnow()
    twin = twin or FabTwin()
    pol_id, pol = optimize.active_policy(db)
    guide = playbook.guidance(db)
    prev = _since(db)
    inc_row, inc = load_active(db)
    pw = (prev.window if prev else {}) or {}
    refresh = (not prev) or bool(pw.get("refresh_next")) or "drift" in trigger
    pool = pw.get("pool", 0) + (1 if refresh else 0)
    pool_windows = 1 if refresh else pw.get("pool_windows", 0) + 1
    ov = overrules(db, prev.created_at if prev else None)
    focus = set(guide["focus"]) | {s for s, n in ov.items() if n >= 2}
    mix = {"hard_share": pol["hard_share"], "decoy_share": min(0.6, pol["decoy_share"] + guide["decoy_boost"])}
    h_round = len(db.exec(select(FastBrainModel.id).where(col(FastBrainModel.status).in_(["active", "retired"]))).all())
    h_seed = HOLDOUT_BASE + (h_round % HOLDOUT_ROUNDS) * HOLDOUT_SPAN
    holdout = await frozen_split(mem, h_seed, HOLDOUT_PER_SYMPTOM, twin)
    line_all = receipt_samples(db, mem, pol["receipt_weight"])  # 最新的在前
    k_line = max(10, len(line_all) // 5) if len(line_all) >= 30 else 0
    line_hold, line_train = line_all[:k_line], line_all[k_line:]
    cal = await frozen_split(mem, CAL_SEED, CAL_PER_SYMPTOM, twin)
    a0 = bayes_init(mem.weights)
    bayes = FastModel.bayes(mem.weights)
    incumbent = inc if inc is not None else bayes
    same_arch = inc is not None and inc.arch == {k: pol[k] for k in ("hidden", "rank", "emb")}
    attempt, health, log, cand, twin_data, data = 0, {}, {}, None, [], []
    while attempt < 2:  # 健康门不过：换一批种子重试一次；再不过就停，保留上一版
        seed0 = POOL_SEED + ((pool * 2 + attempt) % 24) * 100_000
        twin_data = await pool_samples(mem, seed0, POOL_PER_SYMPTOM, mix, twin, focus)
        data = twin_data + line_train
        train, val = split_val(data, seed=pool * 2 + attempt)
        cand = inc.clone() if same_arch else FastModel(pol["hidden"], init_weights=mem.weights, seed=pool * 2 + attempt,
                                                         rank=pol["rank"], emb=pol["emb"])
        cand.T = 1.0
        theta0 = list(cand.theta)
        log = fit(cand, train, val, {**pol, "anchor": pol["anchor"] if same_arch else 0.0}, a0=a0, seed=pool * 2 + attempt,
                  controller=guide["controller"])
        finite = all(math.isfinite(v) for v in cand.theta)
        conf = [max(cand.probs(s["X"], s["codes"])) for s in val[:300] if s["X"]]
        mean_conf = sum(conf) / max(1, len(conf))
        cand.T = calibrate_temperature(cand, cal) if finite else 1.0
        gap = log["val_best"] - (log["epochs"][-1]["train"] if log["epochs"] else log["val_best"])
        worst = max(((v - _sym_ll(bayes, val).get(k, v)), k) for k, v in _sym_ll(cand, val).items()) if finite and val else (0.0, "")
        health = {"samples": log["samples"], "finite": finite, "bounded": finite and max(abs(v) for v in cand.theta) <= 50,
                  "learned": log["val_best"] < log["val_start"] - 1e-3, "moved": max(abs(x - y) for x, y in zip(cand.theta, theta0)) > 1e-4,
                  "mean_conf": round(mean_conf, 3), "not_collapsed": 0.2 < mean_conf < 0.995,
                  "overfit_gap": round(gap, 4), "not_overfit": gap <= 0.15, "T": cand.T, "T_ok": 0.5 <= cand.T <= 2.0,
                  "worst_vs_bayes": {"symptom": worst[1], "delta": round(worst[0], 4)}, "groups_ok": worst[0] <= 0.25}
        health["ok"] = all(health[k] for k in ("finite", "bounded", "learned", "moved", "not_collapsed", "not_overfit", "T_ok", "groups_ok")) \
            and log["samples"] >= 600
        if health["ok"]:
            break
        attempt += 1
    # A-RSI 换池判定：在用模型在这一池上还有多少“没学会”的题
    inc_pool = evaluate(incumbent, twin_data)
    informative = round(sum(1 for s, p in zip(twin_data, truth_scores(incumbent, twin_data)) if s["y"] is not None and 1 - p < INFO_P)
                        / max(1, len(twin_data)), 4)
    history = ([] if refresh else list(pw.get("info_history", []))) + [informative]
    first_hit = inc_pool["hit1"] if refresh else pw.get("pool_first_hit", inc_pool["hit1"])
    saturated = (pool_windows >= INFO_MIN_WINDOWS and len(history) > INFO_LAG and informative <= INFO_DROP * history[-1 - INFO_LAG]) \
        or inc_pool["hit1"] - first_hit >= SATURATION
    q = calibrate_conformal(truth_scores(cand, cal))
    pool_top = {}
    for s in twin_data:
        pool_top.setdefault(s["sym"], {}).setdefault(s["truth"], 0)
        pool_top[s["sym"]][s["truth"]] += 1
    pool_top = {sym: {"top": max(d, key=d.get), "share": round(max(d.values()) / sum(d.values()), 3), "n": sum(d.values())}
                for sym, d in pool_top.items()}
    window = {"trigger": trigger, "pool": pool, "pool_windows": pool_windows, "attempts": min(attempt + 1, 2),
              "health": health, "informative": informative, "info_history": history[-6:], "pool_first_hit": first_hit,
              "refresh_next": bool(saturated), "receipts": len(line_train), "line_holdout": len(line_hold), "focus": sorted(focus),
              "overrules": ov, "mix": mix, "applied": guide["applied"], "controller": guide["controller"], "arch": cand.arch,
              "params": cand.n_params, "continued_from": inc_row.id if (inc_row and same_arch) else None}
    conformal = {"alpha": CONFORMAL_ALPHA, "q": q, "n_cal": len(cal)}
    if not health.get("ok"):
        row = FastBrainModel(status="failed", params=cand.to_dict(), policy_id=pol_id, window=window, log=log, metrics={},
                             conformal=conformal, created_at=when)
        db.add(row)
        db.flush()
        playbook.after_window(db, row.id, decisions=log.get("decisions", []), pool_top=pool_top, groups={}, inc_groups={},
                              applied=guide["applied"], promoted=False, gate_note="健康门未通过（已重试一次，停止）", when=when)
        db.add(AuditLog(actor="system:train", action="fastbrain.halt", target="fast-model", detail={"trigger": trigger, "health": health},
                        created_at=when))
        db.commit()
        return row
    ce, ie, be = evaluate(cand, holdout), evaluate(incumbent, holdout), evaluate(bayes, holdout)
    ll = paired_ci(ce["_ll"], ie["_ll"])
    hit = paired_ci(ce["_correct"], ie["_correct"])
    hb = paired_ci(ce["_correct"], be["_correct"])
    s_c, s_i = _sym_ll(cand, holdout), _sym_ll(incumbent, holdout)
    worst = max(((s_c[k] - s_i.get(k, s_c[k]), k) for k in s_c), default=(0.0, ""))
    q_cov = sum(1 for sc in truth_scores(cand, holdout) if sc <= q) / max(1, len(holdout))
    line = None
    if len(line_hold) >= 10:  # 样本少：按“净多答错几例”判（配对的 McNemar 式比较），不按百分比
        lc, li, lb = evaluate(cand, line_hold), evaluate(incumbent, line_hold), evaluate(bayes, line_hold)
        net = lambda a, b: int(sum(1 for x, y in zip(a["_correct"], b["_correct"]) if y > x) - sum(1 for x, y in zip(a["_correct"], b["_correct"]) if x > y))  # noqa: E731
        line = {"n": len(line_hold), "candidate": lc["hit1"], "incumbent": li["hit1"], "bayes": lb["hit1"],
                "net_loss_vs_incumbent": net(lc, li), "net_loss_vs_bayes": net(lc, lb)}
    checks = {"logloss": ll[2] <= 0, "hit1": hit[1] >= NON_INFERIOR, "ece": ce["ece"] <= ie["ece"] + 0.01, "vs_bayes": hb[1] >= NON_INFERIOR,
              "symptom": worst[0] <= 0.2, "coverage": q_cov >= 1 - CONFORMAL_ALPHA - 0.03,
              "line": line is None or (line["net_loss_vs_incumbent"] < LINE_NET and line["net_loss_vs_bayes"] < LINE_NET)}
    promote = all(checks.values())
    failed = [k for k, v in checks.items() if not v]
    gate_note = "全部通过" if promote else "未通过：" + "、".join(
        {"logloss": f"对数损失区间上界 {ll[2]:+.3f} > 0", "hit1": f"命中率区间下界 {hit[1] * 100:+.1f} 个百分点",
         "ece": f"ECE {ce['ece']:.3f} 比在用版本差", "vs_bayes": f"对贝叶斯命中率下界 {hb[1] * 100:+.1f} 个百分点",
         "symptom": f"「{worst[1]}」对数损失比在用版本差 {worst[0]:+.3f}", "coverage": f"保形覆盖 {q_cov:.0%} < 87%",
         "line": f"产线时间留出：比在用版本净多答错 {line and line['net_loss_vs_incumbent']} 例、比贝叶斯净多答错 {line and line['net_loss_vs_bayes']} 例"}[k]
        for k in failed)
    test = [to_sample({"symptom": e["symptom"], "evidence": e["evidence"], "truth": e["truth"]}, mem) for e in eval_set(twin.era)]
    groups, inc_groups = _groups(cand, holdout), _groups(incumbent, holdout)
    conformal["coverage_holdout"] = round(q_cov, 3)
    metrics = {"candidate": _pub(ce), "incumbent": _pub(ie), "bayes": _pub(be), "incumbent_kind": "model" if inc is not None else "bayes",
               "diff_logloss": {"mean": ll[0], "ci95": [ll[1], ll[2]]}, "diff_hit1": {"mean": hit[0], "ci95": [hit[1], hit[2]]},
               "diff_vs_bayes": {"mean": hb[0], "ci95": [hb[1], hb[2]]}, "checks": checks, "promoted": promote, "gate_note": gate_note,
               "groups": groups, "test": {"model": evaluate(cand, test)["hit1"], "bayes": evaluate(bayes, test)["hit1"], "n": len(test)},
               "holdout": {"seed": h_seed, "round": h_round, "n": len(holdout)}, "line_holdout": line,
               "worst_symptom": {"symptom": worst[1], "delta": round(worst[0], 4)}}
    row = FastBrainModel(status="active" if promote else "rejected", params=cand.to_dict(), policy_id=pol_id, window=window, log=log,
                         metrics=metrics, conformal=conformal, created_at=when)
    if promote and inc_row is not None:
        inc_row.status = "retired"
        db.add(inc_row)
    db.add(row)
    db.flush()
    window["notes"] = playbook.after_window(db, row.id, decisions=log["decisions"], pool_top=pool_top, groups=groups, inc_groups=inc_groups,
                                            applied=guide["applied"], promoted=promote, gate_note=gate_note, when=when)
    row.window = dict(window)
    db.add(row)
    db.add(AuditLog(actor="system:train", action="fastbrain.promote" if promote else "fastbrain.reject", target="fast-model",
                    detail={"trigger": trigger, "gate": gate_note, "hit1": ce["hit1"], "bayes": be["hit1"]}, created_at=when))
    db.commit()
    db.refresh(row)
    return row
