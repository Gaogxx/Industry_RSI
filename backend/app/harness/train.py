"""训练机制：可靠度模型（与免训练的 RSI 并存）+ 保形预测。

训练的是什么：一个 9 个参数的逻辑回归，输入快脑排序的特征（首位后验、领先幅度、熵、证据强度、证据信号数、
PM 后突变、事实支持、同条件快脑的历史成功率），输出“首位根因是真因”的概率。它不改根因排序（排序仍由统计证据 +
贝叶斯记忆给出），只决定可靠度、快慢路由和报告里的置信说明。

训练数据：数字孪生按程序出题、程序知道真因（参照 A-RSI，arXiv:2610.38568：标签由程序计算，不用模型投票）；
加上工程师关案确认过的收据（权重 2）。模型自己的高置信结论从不当作标签。

何时更新（训练窗口的触发条件，任一满足即开窗）：
  1. 新增 20 条现场确认的收据；
  2. 孪生同步了产线变更（漂移）；
  3. 样本池“仍有信息量”的比例跌到 2 个窗口前的 85% 以下（同一样本池至少用过 3 个窗口）→ 换一批新的孪生题。
     这条照搬 A-RSI 的换池规则；“有信息量” = 在用模型对该样本仍明显判错（|y − p| > 0.3）。

晋升与回滚：健康门（损失有限、样本够、参数确实变了）不过就换一批种子重试一次，再失败就停，保留上一版；
健康门通过后做影子评估：在冻结的留出集上，候选的对数损失配对 bootstrap 95% 区间上界 ≤ 0（不退步）且 ECE 不变差，
才晋升为 active。A-RSI 没有留出集晋升门，这在工业场景不够，这里补上。

答案可靠性：可靠度是校准过的概率（报告 ECE）；保形预测在孪生校准集上定阈值 q，给出“覆盖真因 90%”的根因集合，
集合里不止一个根因时明确标为“需现场鉴别”。这个覆盖率保证只对孪生的分布严格成立，产线上用已确认的收据持续监控。
"""

from __future__ import annotations

import math
import random

from sqlmodel import Session, col, func, select

from ..fab.kg import rank
from ..models import AuditLog, HarnessReceipt, ReliabilityModel, utcnow
from ..rsi.agents import Executor, Verifier
from ..rsi.calibration import _solve
from ..rsi.engine import practice_many
from ..rsi.memory import MemoryState
from ..rsi.twin import FabTwin, TaskSpec
from . import policy
from .brain import FEATURES, feature_vector
from .experience import signature

NEW_LABELS, POOL_PER_SYMPTOM, HOLDOUT_PER_SYMPTOM, CAL_PER_SYMPTOM = 20, 30, 10, 10
INFO_DROP, INFO_MIN_WINDOWS, INFO_LAG, INFO_GAP = 0.85, 3, 2, 0.3
POOL_SEED, HOLDOUT_SEED, CAL_SEED = 7_500_000, 8_800_000, 8_900_000
DIFFS = ("easy", "medium", "hard", "hard")


def active_model(db: Session) -> ReliabilityModel | None:
    return db.exec(select(ReliabilityModel).where(ReliabilityModel.status == "active").order_by(col(ReliabilityModel.id).desc())).first()


async def twin_samples(mem: MemoryState, seed0: int, per_symptom: int, twin: FabTwin | None = None) -> list[dict]:
    """孪生样本：每题按快脑排序作答一次（验证 agent 判定是否修好 = 首位是否正确），并记下真因的后验（用于保形校准）。"""
    twin = twin or FabTwin()
    specs, seeds = [], []
    for si, sym in enumerate(twin.truth):
        for j in range(per_symptom):
            specs.append(TaskSpec(sym, DIFFS[j % len(DIFFS)], decoy_pm=j % 3 == 0, variant="证据校准"))
            seeds.append(seed0 + si * 1000 + j)
    out = []
    for i in range(0, len(specs), 96):
        res = await practice_many(twin, mem, specs[i:i + 96], seeds[i:i + 96], Executor(), Verifier(), max_attempts=1)
        for r in res:
            hyps = [h.as_dict() for h in rank(r["evidence"], r["symptom"], mem.edges, mem.hotspot_weights(), mem.weights)]
            truth = r["hidden"]["cause"]
            out.append({"fv": feature_vector(hyps, signature(r["evidence"], hyps), {"facts": []}), "y": 1.0 if r["verdict"] == "PASS" else 0.0,
                        "truth_post": next((h["posterior"] for h in hyps if h["cause_code"] == truth), 0.0), "w": 1.0,
                        "symptom": r["symptom"]})
    return out


def receipt_samples(db: Session, limit: int = 400) -> list[dict]:
    rows = db.exec(select(HarnessReceipt).where(col(HarnessReceipt.verified_at).is_not(None))
                   .order_by(col(HarnessReceipt.verified_at).desc()).limit(limit)).all()
    out = []
    for r in rows:
        fv = (r.signature or {}).get("fv")
        if fv and "top_match" in (r.outcome or {}):
            out.append({"fv": fv, "y": 1.0 if r.outcome.get("fast_match") else 0.0, "w": 2.0, "receipt": r.id})
    return out


def fit_logistic(samples: list[dict], lam: float = 1.0, iters: int = 30) -> dict:
    k = len(FEATURES)
    w = [0.0] * k
    xs = [[s["fv"][f] for f in FEATURES] for s in samples]
    for _ in range(iters):
        g = [0.0 if i == 0 else -lam * w[i] for i in range(k)]
        h = [[(-lam if (i == j and i > 0) else 0.0) for j in range(k)] for i in range(k)]
        for x, s in zip(xs, samples):
            z = max(-30.0, min(30.0, sum(wi * xi for wi, xi in zip(w, x))))
            p = 1 / (1 + math.exp(-z))
            r, v = s["w"] * (s["y"] - p), s["w"] * p * (1 - p)
            for i in range(k):
                g[i] += r * x[i]
                for j in range(k):
                    h[i][j] -= v * x[i] * x[j]
        step = _solve([[-x for x in row] for row in h], g)
        w = [wi + si for wi, si in zip(w, step)]
        if max(abs(si) for si in step) < 1e-6:
            break
    return {f: round(v, 5) for f, v in zip(FEATURES, w)}


def _prob(coef: dict | None, fv: dict) -> float:
    if not coef:
        return min(max(fv["top_post"], 1e-4), 1 - 1e-4)
    z = max(-30.0, min(30.0, sum(coef.get(f, 0.0) * fv[f] for f in FEATURES)))
    return min(max(1 / (1 + math.exp(-z)), 1e-4), 1 - 1e-4)


def scores(coef: dict | None, samples: list[dict]) -> dict:
    ps = [_prob(coef, s["fv"]) for s in samples]
    ll = [-(s["y"] * math.log(p) + (1 - s["y"]) * math.log(1 - p)) for s, p in zip(samples, ps)]
    bins: dict[int, list] = {}
    for s, p in zip(samples, ps):
        bins.setdefault(min(9, int(p * 10)), []).append((p, s["y"]))
    ece = sum(len(b) * abs(sum(p for p, _ in b) / len(b) - sum(y for _, y in b) / len(b)) for b in bins.values()) / max(1, len(samples))
    return {"logloss": round(sum(ll) / max(1, len(ll)), 4), "ece": round(ece, 4), "n": len(samples),
            "brier": round(sum((p - s["y"]) ** 2 for s, p in zip(samples, ps)) / max(1, len(ps)), 4), "_ll": ll}


def paired_ci(a: list[float], b: list[float], reps: int = 1000, seed: int = 7) -> tuple[float, float, float]:
    """配对 bootstrap：mean(a − b) 的 95% 区间。"""
    d = [x - y for x, y in zip(a, b)]
    if not d:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    means = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(reps))
    return round(sum(d) / len(d), 4), round(means[int(0.025 * reps)], 4), round(means[int(0.975 * reps) - 1], 4)


def calibrate_conformal(samples: list[dict], alpha: float = policy.CONFORMAL_ALPHA) -> float:
    s = sorted(1 - x["truth_post"] for x in samples)
    n = len(s)
    k = min(n - 1, max(0, math.ceil((n + 1) * (1 - alpha)) - 1))
    return round(s[k], 4)


def coverage(samples: list[dict], q: float) -> float:
    return round(sum(1 - x["truth_post"] <= q for x in samples) / max(1, len(samples)), 4)


def pending_triggers(db: Session) -> list[str]:
    last = db.exec(select(ReliabilityModel).order_by(col(ReliabilityModel.id).desc())).first()
    since = last.created_at if last else None
    q = select(func.count(col(HarnessReceipt.id))).where(col(HarnessReceipt.verified_at).is_not(None))
    n_new = db.exec(q.where(col(HarnessReceipt.verified_at) > since) if since else q).one()
    out = []
    if last is None:
        out.append("bootstrap")
    if n_new >= NEW_LABELS:
        out.append(f"new_labels:{n_new}")
    aq = select(func.count(col(AuditLog.id))).where(AuditLog.action == "twin.update")
    if since and db.exec(aq.where(col(AuditLog.created_at) > since)).one():
        out.append("drift")
    if last and (last.window or {}).get("refresh_next"):
        out.append("pool_saturated")
    return out


async def train_window(db: Session, mem: MemoryState, trigger: str, when=None, twin: FabTwin | None = None) -> ReliabilityModel:
    when = when or utcnow()
    prev = db.exec(select(ReliabilityModel).order_by(col(ReliabilityModel.id).desc())).first()
    act = active_model(db)
    pw = (prev.window if prev else {}) or {}
    pool = pw.get("pool", 0) + (1 if (not prev or pw.get("refresh_next") or "drift" in trigger) else 0)
    pool_windows = (pw.get("pool_windows", 0) + 1) if pool == pw.get("pool") else 1
    holdout = await twin_samples(mem, HOLDOUT_SEED, HOLDOUT_PER_SYMPTOM, twin)
    cal = await twin_samples(mem, CAL_SEED, CAL_PER_SYMPTOM, twin)
    attempt, coef, health, data = 0, None, {}, []
    while attempt < 2:  # 健康门不过：换一批种子重试一次；再不过就停，保留上一版
        data = await twin_samples(mem, POOL_SEED + (pool + attempt) * 100_000, POOL_PER_SYMPTOM, twin) + receipt_samples(db)
        coef = fit_logistic(data)
        finite = all(math.isfinite(v) for v in coef.values())
        moved = act is None or max(abs(coef[f] - act.coef.get(f, 0.0)) for f in FEATURES) > 1e-4
        health = {"samples": len(data), "finite": finite, "moved": moved, "ok": finite and moved and len(data) >= 200}
        if health["ok"]:
            break
        attempt += 1
    p_act = [_prob(act.coef if act else None, s["fv"]) for s in data]
    informative = round(sum(abs(s["y"] - p) > INFO_GAP for s, p in zip(data, p_act)) / max(1, len(data)), 4)
    history = (pw.get("info_history", []) if pool == pw.get("pool") else []) + [informative]
    refresh = pool_windows >= INFO_MIN_WINDOWS and len(history) > INFO_LAG and informative <= INFO_DROP * history[-1 - INFO_LAG]
    q = calibrate_conformal(cal)
    window = {"trigger": trigger, "pool": pool, "pool_windows": pool_windows, "attempts": attempt + 1, "health": health,
              "informative": informative, "info_history": history[-6:], "refresh_next": bool(refresh),
              "receipts": sum(1 for s in data if s["w"] > 1)}
    conformal = {"alpha": policy.CONFORMAL_ALPHA, "q": q, "n_cal": len(cal), "coverage_holdout": coverage(holdout, q),
                 "set_size_holdout": None}
    if not health["ok"]:
        row = ReliabilityModel(status="failed", coef=coef or {}, window=window, metrics={}, conformal=conformal, created_at=when)
        db.add(row)
        db.add(AuditLog(actor="system:train", action="reliability.failed", target="reliability-model", detail=window, created_at=when))
        db.commit()
        return row
    cand = scores(coef, holdout)
    inc = scores(act.coef if act else None, holdout)
    mean, lo, hi = paired_ci(cand.pop("_ll"), inc.pop("_ll"))
    promote = hi <= 0.0 and cand["ece"] <= inc["ece"] + 0.01
    metrics = {"candidate": cand, "incumbent": inc, "incumbent_kind": "model" if act else "posterior",
               "logloss_diff": {"mean": mean, "ci95": [lo, hi]}, "promoted": promote}
    row = ReliabilityModel(status="active" if promote else "rejected", coef=coef, window=window, metrics=metrics,
                           conformal=conformal, created_at=when)
    if promote and act is not None:
        act.status = "retired"
        db.add(act)
    elif act is not None:  # 候选没晋升，但保形阈值按当前记忆重新校准
        act.conformal = conformal
        db.add(act)
    db.add(row)
    db.add(AuditLog(actor="system:train", action="reliability.promote" if promote else "reliability.reject", target="reliability-model",
                    detail={"trigger": trigger, "logloss_diff": metrics["logloss_diff"], "ece": cand["ece"]}, created_at=when))
    db.commit()
    return row
