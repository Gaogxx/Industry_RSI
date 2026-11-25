"""快脑小模型的训练数据。

- 孪生题：课程 agent 按固定配额出题（Curriculum.training_pool），孪生生成 MES/FDC 数据，标签由孪生程序直接给出
  （参照 A-RSI：标签可程序验证，不用模型投票、不用模型自评）。这只在孪生里成立；
- 产线题：工程师关案确认过根因的收据（权重更高）。模型自己的高置信结论从不当作标签。
- 留出集（晋升门用）每晋升一次就换一块全新的（防止同一块留出集被反复用来做决定后也被“调”进去）；
  校准集（温度、保形阈值用）固定；所有种子段互不重叠，也不与 RSI 的测试集、验收门面板重叠。
- 产线时间留出：最近 20% 已确认的收据不进训练，专门给晋升门检验（用过去训练、用未来检验）。
"""

from __future__ import annotations

import asyncio
import random
import zlib
from collections import OrderedDict

from sqlmodel import Session, col, select

from ..fab.analytics import analyze_dataset
from ..fab.analytics import raw_signals
from ..models import Case, FabDataset, HarnessReceipt
from ..rsi.agents import Curriculum
from ..rsi.memory import MemoryState
from ..rsi.twin import FabTwin, TaskSpec, observed_symptom
from .features import candidate_rows

POOL_SEED, CAL_SEED = 7_500_000, 8_900_000  # 训练池 750 万 ~ 990 万段（按池编号轮换 24 块）；校准集固定
HOLDOUT_BASE, HOLDOUT_ROUNDS, HOLDOUT_SPAN = 10_000_000, 40, 50_000  # 留出集每晋升一次换一块（1000 万 ~ 1200 万段）
HOLDOUT_PER_SYMPTOM, CAL_PER_SYMPTOM = 40, 20
KEEP = ("summary", "suspects", "change_point", "fdc", "raw_signals")
_OBS: OrderedDict = OrderedDict()
_OBS_MAX = 24_000


def twin_key(twin: FabTwin) -> int:
    return zlib.crc32(repr(sorted((k, sorted(v.items())) for k, v in twin.truth.items())).encode())


def observe(twin: FabTwin, spec: TaskSpec, seed: int) -> dict:
    """孪生出一道题：只返回 agent 能看到的证据摘要 + 孪生程序给出的标签（真因）。"""
    task = twin.make_task(spec, seed)
    ev = analyze_dataset(task.observation)
    return {"symptom": observed_symptom(spec.symptom, ev), "evidence": {k: ev.get(k) for k in KEEP},
            "truth": task.hidden["cause"], "difficulty": spec.difficulty, "decoy": spec.decoy_pm, "seed": seed}


async def observe_many(twin: FabTwin, specs: list[TaskSpec], seeds: list[int]) -> list[dict]:
    from ..rsi.engine import _pool

    tk = twin_key(twin)
    keys = [(tk, sp.symptom, sp.difficulty, sp.decoy_pm, sd) for sp, sd in zip(specs, seeds)]
    todo = [i for i, k in enumerate(keys) if k not in _OBS]
    pool = _pool() if len(todo) >= 200 else None
    if pool is not None:
        loop = asyncio.get_running_loop()
        try:
            res = await asyncio.gather(*[loop.run_in_executor(pool, observe, twin, specs[i], seeds[i]) for i in todo])
        except Exception:  # noqa: BLE001  进程池不可用时退回本进程
            res = [observe(twin, specs[i], seeds[i]) for i in todo]
    else:
        res = [observe(twin, specs[i], seeds[i]) for i in todo]
    for i, r in zip(todo, res):
        _OBS[keys[i]] = r
    out = []
    for k in keys:
        _OBS.move_to_end(k)
        out.append(_OBS[k])
    while len(_OBS) > _OBS_MAX:
        _OBS.popitem(last=False)
    return out


def to_sample(obs: dict, mem: MemoryState, w: float = 1.0, source: str = "twin") -> dict:
    hyps, X = candidate_rows(obs["evidence"], obs["symptom"], mem)
    codes = [h["cause_code"] for h in hyps]
    truth = obs.get("truth")
    return {"X": X, "codes": codes, "y": codes.index(truth) if truth in codes else None, "w": w, "sym": obs["symptom"],
            "truth": truth, "difficulty": obs.get("difficulty", ""), "decoy": bool(obs.get("decoy")), "source": source}


async def pool_samples(mem: MemoryState, seed0: int, per_symptom: int, mix: dict, twin: FabTwin | None = None,
                       focus: set[str] | None = None, symptoms: list[str] | None = None) -> list[dict]:
    twin = twin or FabTwin()
    rng = random.Random(seed0)
    specs = Curriculum().training_pool(per_symptom, mix, rng, focus, symptoms)
    obs = await observe_many(twin, specs, [seed0 + i for i in range(len(specs))])
    return [to_sample(o, mem) for o in obs]


async def frozen_split(mem: MemoryState, seed0: int, per_symptom: int, twin: FabTwin | None = None) -> list[dict]:
    """固定种子、固定配比（半数困难、三分之一加 PM 干扰）的留出 / 校准集。"""
    return await pool_samples(mem, seed0, per_symptom, {"hard_share": 0.5, "decoy_share": 0.33}, twin)


def receipt_samples(db: Session, mem: MemoryState, weight: float, limit: int = 400) -> list[dict]:
    """工程师关案确认过的收据 → 产线训练样本（证据取自案例本身，标签 = 现场确认的根因）。"""
    rows = db.exec(select(HarnessReceipt, Case).where(HarnessReceipt.case_id == Case.id, col(HarnessReceipt.verified_at).is_not(None))
                   .order_by(col(HarnessReceipt.verified_at).desc()).limit(limit)).all()
    out = []
    for rc, case in rows:
        truth = (rc.outcome or {}).get("cause_code")
        if not truth or not case.evidence or not rc.symptom:
            continue
        ev = case.evidence
        if "raw_signals" not in ev:  # 升级前分析的案例没有原始信号摘要：有原始数据就补算，没有就不用它训练
            ds = db.get(FabDataset, case.id)
            if ds is None or not ds.data:
                continue
            ev = {**ev, "raw_signals": raw_signals(ds.data)}
        s = to_sample({"symptom": rc.symptom, "evidence": ev, "truth": truth}, mem, weight, "line")
        s["receipt"] = rc.id
        out.append(s)
    return out


def split_val(samples: list[dict], frac: float = 1 / 6, seed: int = 0) -> tuple[list[dict], list[dict]]:
    idx = list(range(len(samples)))
    random.Random(seed).shuffle(idx)
    k = int(len(idx) * frac)
    val = {i for i in idx[:k]}
    return [s for i, s in enumerate(samples) if i not in val], [s for i, s in enumerate(samples) if i in val]
