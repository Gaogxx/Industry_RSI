"""演示数据：一家虚构的 AI 加速器芯片供应链（12 英寸先进逻辑晶圆厂 → 2.5D 先进封装 + HBM → 测试 → 液冷整机柜）。

- 过去约 18 个月 120 个已关闭案例：按时间顺序“先分析、后关案学习”回放，知识图谱与经验手册是真实学出来的；
- 本厂真实根因分布与教科书先验不同（例如边缘环形失效在本厂多由刻蚀聚焦环磨损引起，而教科书首推 CMP），
  且约六成案例证据模糊（共性分析不显著） —— 这正是持续学习发挥作用的地方；
- 5 个进行中的案例带完整原始数据（批次履历、晶圆图、FDC、PM 事件），用于现场演示。

全部数据为模拟生成，不代表任何真实企业。接入 MES/YMS/FDC 后关闭 KAIZEN_SEED_DEMO。
"""

from __future__ import annotations

import random
from datetime import timedelta

from sqlmodel import Session, select

from .fab.analytics import analyze_dataset
from .fab.flow import CAUSES, STEPS, SYMPTOMS, Scenario, cause_by_code
from .fab.learning import ensure_edges, learn_from_closure, rank_case, refute_cause
from .fab.simulate import simulate
from .rsi.twin import HOTSPOTS, SITE_TRUTH, hotspot_tool
from .models import Action, Analysis, Case, FabDataset, utcnow

# 本厂（虚构）的真实规律定义在数字孪生里（rsi/twin.py）：历史案例与孪生练习面对同一个环境
# 各症状在一年中出现的相对频次（前道晶圆图类异常最多）
SYMPTOM_WEIGHTS = {"WM_EDGE": 3, "WM_SCRATCH": 2, "WM_CENTER": 2, "WM_CLUSTER": 2, "TEST_VMIN": 2, "PKG_WARP": 1, "HBM_OPEN": 1}
HISTORY_CASES = 120
HISTORY_DAYS = 540

SITES = {"fe": ("Fab A · 12 英寸先进逻辑", "AX-900 加速器 die（5nm 级）"),
         "pkg": ("封测厂 B · 2.5D 先进封装", "AX-900 2.5D 封装（中介层 + 8×HBM3E）"),
         "test": ("测试厂 B · FT/BI/SLT", "AX-900 成品"),
         "sys": ("系统厂 C · 整机柜集成", "AX-RACK72 液冷整机柜")}

# 进行中的演示案例：(症状, 真因, 证据强度, 严重度, 标题, 描述, 关键数据, 损失万元)
OPEN_CASES = [
    ("WM_EDGE", "ETCH_FOCUS_RING", 0.25, "S1", "AX-900 die CP 良率下降，晶圆边缘环形失效",
     "Fab A 近 9 天 AX-900 die CP 良率由 92% 降至 89.5%，失效集中在晶圆外圈；YMS 报警 3 次，初步看各机台差异不明显。",
     {"CP良率": "89.5%", "基线良率": "92.0%", "低良率批次": 4}, 420.0),
    ("PKG_WARP", "SUBSTRATE_CTE", 0.8, "S1", "2.5D 封装翘曲超标，C4 凸点开路",
     "封测厂 B 2.5D 封装后翘曲 > 规格，FT 出现 C4 凸点开路，X-Ray 确认边角凸点非润湿。",
     {"翘曲(µm)": 118, "规格(µm)": 90, "开路DPPM": 3200}, 460.0),
    ("HBM_OPEN", "TCB_COPLANARITY", 0.7, "S2", "HBM3E 堆叠后链路训练失败率上升",
     "8 层 HBM3E 贴装后 FT 链路训练失败率升高，失效集中在部分通道。",
     {"链路失效DPPM": 2600, "基线DPPM": 400}, 380.0),
    ("TEST_VMIN", "SOCKET_CONTACT", 0.25, "S2", "FT 高频档良率下降，Vmin 右移",
     "AX-900 成品 FT 最高速度档比例下降 7 个百分点，Vmin 分布右移约 15mV，复测部分可通过。",
     {"高档良率下降": "7%", "Vmin右移(mV)": 15}, 300.0),
    ("WM_CLUSTER", "PM_SEASONING", 0.6, "S2", "刻蚀层缺陷密度 D0 异常，簇状颗粒",
     "KLA 扫描发现金属刻蚀层簇状颗粒缺陷，D0 由 0.08 升至 0.21 个/cm²。",
     {"D0(个/cm²)": 0.21, "基线D0": 0.08}, 260.0),
    ("OVL_DRIFT", "SCANNER_MATCHING", 0.75, "S2", "7nm 级 DUV 多重图形金属层套刻超规",
     "国产 12 英寸厂 SAQP 金属层套刻 3σ 由 2.6nm 升至 3.9nm，EPE 叠加导致 CP 短路增加，近期新导入一台 DUV 扫描机。",
     {"Overlay 3σ(nm)": 3.9, "规格(nm)": 3.0, "CP短路失效": "2.1%"}, 540.0,
     {"site": "Fab D · 国产 12 英寸先进逻辑", "product": "国产 AI 训练芯片（7nm 级，DUV 多重图形）"}),
]


def _title(sym: str, rng: random.Random) -> str:
    s = SYMPTOMS[sym]
    prod = SITES[s.stage][1].split("（")[0]
    return f"{prod} {s.name}" + rng.choice(["", "（夜班报警）", "（周末突发）", "（客户投诉触发）", ""])


def _close_history_case(db: Session, case: Case, cause_code: str, rng: random.Random) -> None:
    hyps, frozen = rank_case(db, case.evidence, case.symptom)
    ranked = [h.cause_code for h in hyps]
    an = Analysis(case_id=case.id, evidence=case.evidence, hypotheses=[h.as_dict() for h in hyps],
                  counterfactual={"top": frozen[0].cause_code, "top_name": frozen[0].name,
                                  "top_posterior": round(frozen[0].posterior, 3)},
                  model="offline-rules", created_at=case.created_at + timedelta(hours=2))
    db.add(an)
    db.flush()
    found = cause_by_code(cause_code)[1]
    first = hyps[0]
    when = case.closed_at
    if first.cause_code != cause_code and rng.random() < 0.5:  # 先按首位假设整改，验证无效 → 现场否定
        db.add(Action(case_id=case.id, analysis_id=an.id, kind="corrective", title=f"按首位假设整改：{first.name}",
                      cause_code=first.cause_code, status="ineffective", result="整改后指标未恢复",
                      due_date=case.created_at + timedelta(days=5), completed_at=case.created_at + timedelta(days=5)))
        refute_cause(db, case, first.cause_code, when=case.created_at + timedelta(days=5))
    for kind, text in zip(("containment", "corrective", "preventive"), found.capa):
        days = {"containment": 1, "corrective": 7, "preventive": 21}[kind]
        db.add(Action(case_id=case.id, analysis_id=an.id, kind=kind, title=text, cause_code=cause_code,
                      owner_role={"containment": "生产主管", "corrective": "工艺/设备工程师", "preventive": "模块经理"}[kind],
                      due_days=days, due_date=case.created_at + timedelta(days=days), status="verified",
                      verify_metric=found.metric, result="已验证有效",
                      completed_at=case.created_at + timedelta(days=days - (1 if rng.random() < 0.8 else -2))))
    case.cause_code, case.root_cause, case.root_cause_category = cause_code, found.name, found.six_m
    db.add(case)
    db.flush()
    learn_from_closure(db, case, cause_code, ranked, when=when)


def seed(db: Session) -> int:
    if db.exec(select(Case)).first():
        return 0
    ensure_edges(db)
    rng = random.Random(2025)
    assert set(SYMPTOM_WEIGHTS) <= set(SITE_TRUTH)
    now = utcnow()
    # ① 历史：约 18 个月 120 个已关闭案例（约四成证据清晰，六成证据模糊）
    for k in range(HISTORY_CASES):
        sym = rng.choices(list(SYMPTOM_WEIGHTS), weights=list(SYMPTOM_WEIGHTS.values()))[0]
        dist = SITE_TRUTH[sym]
        cause = rng.choices(list(dist), weights=list(dist.values()))[0]
        strength = rng.uniform(0.7, 1.0) if rng.random() < 0.4 else rng.uniform(0.0, 0.12)
        tool = hotspot_tool(cause, sym, rng)
        created = now - timedelta(days=HISTORY_DAYS - (HISTORY_DAYS - 20) * k / HISTORY_CASES)
        ds = simulate(Scenario(sym, cause, strength=strength, seed=1000 + k, n_lots=100, extras={"tool": tool}), created)
        ev = analyze_dataset(ds)
        s = SYMPTOMS[sym]
        site, product = SITES[s.stage]
        top = (ev.get("suspects") or [{}])[0]
        case = Case(title=_title(sym, rng), stage=s.stage, symptom=sym, product=product, site=site,
                    line=STEPS[top["step"]].name if top.get("step") in STEPS else "",
                    equipment=top.get("value", "") if "#" not in top.get("factor", "#") else "",
                    severity=rng.choice(["S1", "S2", "S2", "S3"]), status="closed",
                    description=f"{s.name}，{s.metric} 异常，共 {ev['summary']['fail_lots']} 个失效批次。",
                    data={"失效批次": ev["summary"]["fail_lots"], "批次总数": ev["summary"]["lots"]},
                    evidence=ev, cost_impact=round(rng.uniform(20, 400), 1), downtime_hours=round(rng.uniform(0, 30), 1),
                    reporter="演示数据", source="seed", created_at=created,
                    closed_at=created + timedelta(days=rng.uniform(5, 25)))
        db.add(case)
        db.flush()
        _close_history_case(db, case, cause, rng)
    db.commit()
    from .rsi.memory import load_memory, refit_and_store

    refit_and_store(db, load_memory(db), source="production")  # 产线历史同样训练证据权重（经验收门）
    db.commit()
    # ② 进行中：带完整原始数据
    for i, (sym, cause, strength, sev, title, desc, data, cost, *extra) in enumerate(OPEN_CASES):
        s = SYMPTOMS[sym]
        site, product = SITES[s.stage]
        if extra:
            site, product = extra[0].get("site", site), extra[0].get("product", product)
        c = cause_by_code(cause)[1]
        case = Case(title=title, stage=s.stage, symptom=sym, product=product, site=site, line=STEPS[c.step].name,
                    severity=sev, status="open", description=desc, data=data, cost_impact=cost,
                    reporter="演示数据", source="seed", created_at=now - timedelta(days=2 - i * 0.3))
        db.add(case)
        db.flush()
        hot = HOTSPOTS.get(cause)
        db.add(FabDataset(case_id=case.id, data=simulate(Scenario(sym, cause, strength=strength, seed=77 + i,
                                                                  extras={"tool": hot[0] if hot else None}), now)))
    db.commit()
    return HISTORY_CASES + len(OPEN_CASES)


def demo_truth(case: Case) -> str | None:
    """演示用：进行中案例的“真因”（模拟工程师现场确认的结果）。"""
    for sym, cause, *_ in OPEN_CASES:
        if case.symptom == sym and case.source == "seed":
            return cause
    return None


assert all(c in {x.code for x in CAUSES[s]} for s, d in SITE_TRUTH.items() for c in d)  # noqa: S101
