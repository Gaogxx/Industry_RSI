"""晶圆厂 / 封测厂数据模拟器：生成与 MES、YMS、FDC、EAP 结构一致的批次履历、晶圆图、FDC 与 PM 事件。

真实部署时用 MES/YMS/FDC 的只读接口替换本模块即可，分析引擎只依赖这里输出的数据结构。
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

from .flow import CAUSES, ROUTES, STEPS, SYMPTOMS, Scenario

GRID = 25  # 晶圆图每行/列 die 数（演示用；真实 AI 芯片大 die 每片约 60 颗）


def _cause(symptom: str, code: str):
    return next(c for c in CAUSES[symptom] if c.code == code)


def _iso(t: datetime) -> str:
    return t.replace(microsecond=0).isoformat()


def wafer_map(pattern: str | None, rng: random.Random, severity: float = 1.0) -> list[str]:
    """生成晶圆图：'.' 晶圆外，'0' 良品 die，'1' 失效 die。"""
    c = (GRID - 1) / 2
    R = GRID / 2
    line = None
    if pattern == "scratch":
        ang = rng.uniform(0, math.pi)
        off = rng.uniform(-3, 3)
        line = (math.cos(ang), math.sin(ang), off)
    blobs = [(rng.uniform(4, GRID - 5), rng.uniform(4, GRID - 5)) for _ in range(rng.choice((1, 2)))]
    rows = []
    for i in range(GRID):
        row = []
        for j in range(GRID):
            dx, dy = j - c, i - c
            r = math.hypot(dx, dy) / R
            if r > 0.98:
                row.append(".")
                continue
            p = 0.025
            if pattern == "edge" and r > 0.80:
                p += 0.62 * severity
            elif pattern == "center" and r < 0.36:
                p += 0.58 * severity
            elif pattern == "scratch" and line:
                dist = abs(-line[1] * dx + line[0] * dy - line[2])
                if dist < 0.75 and r < 0.9:
                    p += 0.85 * severity
            elif pattern == "cluster":
                if any(math.hypot(j - bx, i - by) < 2.6 for bx, by in blobs):
                    p += 0.8 * severity
            elif pattern == "random":
                p += 0.10 * severity
            row.append("1" if rng.random() < p else "0")
        rows.append("".join(row))
    return rows


def simulate(sc: Scenario, now: datetime | None = None) -> dict:
    rng = random.Random(sc.seed)
    now = now or datetime.now(timezone.utc)
    sym = SYMPTOMS[sc.symptom]
    cause = _cause(sc.symptom, sc.cause)
    route = ROUTES[sym.stage]
    step = STEPS[cause.step]

    # 激励源：机台/腔体（tool_specific）或物料批次（材料类根因）
    # extras: tool=指定机台（数字孪生的机台热点）；decoy_pm=在无关机台加一次 PM 干扰；fixed=已按真因修复（干预验证）
    exc_tool = rng.choice(step.tools)
    if sc.extras.get("tool") in step.tools:
        exc_tool = sc.extras["tool"]
    fixed = bool(sc.extras.get("fixed"))
    mat_key = f"{cause.step}#物料批次"
    batches = [f"{cause.step[:3]}-B{rng.randint(100, 999)}" for _ in range(4)]
    exc_batch = batches[1]
    start = now - timedelta(days=rng.uniform(7, 12))

    events = []
    if cause.pm_related:
        events.append({"tool": exc_tool, "time": _iso(start - timedelta(hours=rng.uniform(1, 6))), "type": "PM"})
    for _ in range(3):  # 干扰项：其它机台的正常 PM
        s = STEPS[rng.choice(route)]
        events.append({"tool": rng.choice(s.tools), "time": _iso(now - timedelta(days=rng.uniform(2, 20))), "type": "PM"})
    decoy_t = start - timedelta(hours=rng.uniform(2, 8))
    if sc.extras.get("decoy_pm"):  # 压力测试：在变点附近给一台无关机台做 PM，诱导“PM 后突变”的误判
        other_step = STEPS[[x for x in route if x != cause.step and x in STEPS][0]]
        events.append({"tool": other_step.tools[0], "time": _iso(decoy_t), "type": "PM"})
    events.sort(key=lambda e: e["time"])

    lots = []
    for k in range(sc.n_lots):
        t = now - timedelta(days=24) + timedelta(days=24 * (k + rng.random()) / sc.n_lots)
        steps = {s: rng.choice(STEPS[s].tools) for s in route}
        if not cause.tool_specific:
            steps[mat_key] = batches[min(3, int(rng.random() * 4))]
        hit = not fixed and t >= start and (steps.get(cause.step) == exc_tool if cause.tool_specific else steps.get(mat_key) == exc_batch)
        lot = {"lot": f"L{t:%y%m}{k:04d}", "time": _iso(t), "steps": steps}
        # 噪声与激励幅度无条件抽取：修复前后同一随机流，干预对比只差“有没有激励”
        if sym.stage == "fe":
            base_y, drop = rng.gauss(92.0, 1.2), rng.gauss(6.5 * sc.strength + 0.8, 1.2)
            y = base_y - (drop if hit else 0)
            lot["yield"] = round(y, 2)
            lot["fail"] = y < 88.5
        else:
            base_r, add = abs(rng.gauss(0.3, 0.12)), abs(rng.gauss(2.2 * sc.strength, 0.5))
            rate = base_r + (add if hit else 0)
            lot["fail_rate"] = round(rate, 3)
            lot["fail"] = rate > 1.0
        lot["_hit"] = hit
        lots.append(lot)

    # FDC：可疑工序所有机台 + 一个干扰工序
    fdc_params = sorted({p for c in CAUSES[sc.symptom] if c.step == cause.step for p in c.fdc}) or ["腔体压力"]
    other = rng.choice([s for s in route if s != cause.step and s in STEPS])
    fdc: dict[str, list[dict]] = {}
    base = {p: rng.uniform(10, 60) for p in fdc_params + ["温度", "压力"]}
    for tool in step.tools + STEPS[other].tools:
        params = fdc_params if tool in step.tools else ["温度", "压力"]
        rows = []
        for lot in lots:
            if lot["steps"].get(cause.step if tool in step.tools else other) != tool:
                continue
            row = {"lot": lot["lot"], "time": lot["time"]}
            for p in params:
                mu, sd = base[p], base[p] * 0.03
                v = rng.gauss(mu, sd)
                if not fixed and tool == exc_tool and p in cause.fdc and lot["time"] >= _iso(start):
                    v += 3.0 * sd * max(sc.strength, 0.35)
                row[p] = round(v, 3)
            rows.append(row)
        fdc[tool] = rows

    wafers = []
    if sym.wafer_pattern:
        # 良率工程师调取最差的几片晶圆看空间特征（不论是否越过失效阈值）
        bad = ([l for l in lots if l["fail"] and l["_hit"]] or sorted(lots, key=lambda l: l["yield"]))[:4]
        good = [l for l in lots if not l["fail"]][-2:]
        for l in bad:
            wafers.append({"wafer": f"{l['lot']}-W{rng.randint(1, 25):02d}", "lot": l["lot"], "fail": True,
                           "map": wafer_map(sym.wafer_pattern, rng, 0.6 + 0.4 * sc.strength)})
        for l in good:
            wafers.append({"wafer": f"{l['lot']}-W{rng.randint(1, 25):02d}", "lot": l["lot"], "fail": False,
                           "map": wafer_map(None, rng)})
    for l in lots:
        l.pop("_hit")
    return {"route": route + ([mat_key] if not cause.tool_specific else []), "metric": sym.metric,
            "lots": lots, "events": events, "fdc": fdc, "wafers": wafers}
