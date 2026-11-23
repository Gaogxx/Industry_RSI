"""演示数据：一家虚构的 AI 加速器芯片供应链（12 英寸先进逻辑晶圆厂 → 2.5D 先进封装 + HBM → 测试 → 液冷整机柜）。

按时间顺序回放过去约 18 个月（去年到现在），所有记录都带真实的历史时间：
- 132 个历史案例：建案时用当时的记忆分析并生成 CAPA，5–25 天后由 QA 确认根因关案、执行 agent 学习；
  其中最近的 12 个仍在进行中（分析中 / 措施执行 / 效果验证），带完整原始数据与 CAPA；
- 本厂真实根因分布与教科书先验不同，并且会随产线变更而变化（rsi/twin.py 的 CHANGES）；约六成案例证据模糊；
- 每月一次改进周期（夜间定时，同时重拟合证据权重）；约每月一次 RSI 自我进化会话（面对当月的数字孪生）；
- 每一步都写审计日志；最后加入 6 个刚上报、带完整原始数据的待分析案例，用于现场演示。

全部数据为模拟生成，不代表任何真实企业。接入 MES/YMS/FDC 后关闭 KAIZEN_SEED_DEMO。
"""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import timedelta

from sqlmodel import Session, col, select

from .fab.analytics import analyze_dataset
from .fab.flow import CAUSES, STEPS, SYMPTOMS, Scenario, cause_by_code
from .fab.learning import ensure_edges, learn_from_closure, rank_case, refute_cause
from .fab.simulate import simulate
from .improvement import build_cycle
from .models import Action, Analysis, AuditLog, Case, EvidenceModel, FabDataset, LearningEvent, RSISession, RSITask, utcnow
from .rsi.twin import CHANGES, HOTSPOTS, SITE_TRUTH, FabTwin, era_at, hotspot_tool, truth_for_era

log = logging.getLogger("kaizen.seed")
# 演示数据准备进度（/api/meta 返回给控制台显示）
STATE = {"state": "idle", "progress": 0.0, "detail": ""}


def _progress(p: float, detail: str, state: str = "seeding") -> None:
    import time

    now = time.monotonic()
    if STATE.get("_stage") != detail.split("：")[0][:6] and STATE.get("_t"):
        log.debug("演示数据阶段 %s 用时 %.1f s", STATE.get("_stage"), now - STATE["_t"])
    STATE.update(state=state, progress=round(min(max(p, 0.0), 1.0), 3), detail=detail)


# 各症状在一年中出现的相对频次（前道晶圆图类异常最多）
SYMPTOM_WEIGHTS = {"WM_EDGE": 3, "WM_SCRATCH": 2, "WM_CENTER": 2, "WM_CLUSTER": 2, "TEST_VMIN": 2, "PKG_WARP": 1, "HBM_OPEN": 1}
HISTORY_CASES = 120  # 已关闭
HISTORY_DAYS = 540
# 仍在进行中的近期案例：(几天前上报, 状态)
IN_PROGRESS = [(46, "verifying"), (41, "verifying"), (37, "verifying"), (33, "capa"), (29, "capa"), (25, "capa"),
               (21, "capa"), (17, "capa"), (13, "capa"), (9, "analyzing"), (6, "analyzing"), (4, "analyzing")]
CYCLE_DAYS = list(range(510, 0, -30))  # 每月一次改进周期
# RSI 会话：产线变更同步到孪生两天后跑一次，把变化学进来（最后一次变更由今天启动时的在线会话处理）
RSI_DAYS = [352] + [c.days_ago - 2 for c in CHANGES[:-1]]  # 首次会话 + 每次产线变更两天后

SITES = {"fe": ("Fab A · 12 英寸先进逻辑", "AX-900 加速器 die（5nm 级）"),
         "pkg": ("封测厂 B · 2.5D 先进封装", "AX-900 2.5D 封装（中介层 + 8×HBM3E）"),
         "test": ("测试厂 B · FT/BI/SLT", "AX-900 成品"),
         "sys": ("系统厂 C · 整机柜集成", "AX-RACK72 液冷整机柜")}
# 虚构的账号：上报人按工段，关案由 QA 批准
ENGINEERS = {"fe": ["lin.yield", "wang.etch", "chen.cmp", "xu.litho"], "pkg": ["zhao.pkg", "qian.bond"],
             "test": ["sun.test", "ma.ft"], "sys": ["liu.rack"]}
APPROVERS = ["zhou.qa", "he.qe", "gao.qa"]
OWNER = {"containment": "生产主管", "corrective": "工艺/设备工程师", "preventive": "模块经理"}
DUE = {"containment": 1, "corrective": 7, "preventive": 21}

# 进行中的演示案例：(症状, 真因, 证据强度, 严重度, 标题, 描述, 关键数据, 损失万元)
OPEN_CASES = [
    ("WM_EDGE", "ETCH_FOCUS_RING", 0.25, "S1", "AX-900 die CP 良率下降，晶圆边缘环形失效",
     "Fab A 近 9 天 AX-900 die CP 良率由 92% 降至 89.5%，失效集中在晶圆外圈；YMS 报警 3 次，初步看各机台差异不明显。",
     {"CP良率": "89.5%", "基线良率": "92.0%", "低良率批次": 4}, 420.0),
    ("PKG_WARP", "SUBSTRATE_CTE", 0.8, "S1", "2.5D 封装翘曲超标，C4 凸点开路",
     "封测厂 B 2.5D 封装后翘曲 > 规格，FT 出现 C4 凸点开路，X-Ray 确认边角凸点非润湿。",
     {"翘曲(µm)": 118, "规格(µm)": 90, "开路DPPM": 3200}, 460.0),
    ("HBM_OPEN", "TCB_COPLANARITY", 0.7, "S2", "HBM3E 堆叠后链路训练失败率上升",
     "12 层 HBM3E 贴装后 FT 链路训练失败率升高，失效集中在部分通道。",
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
    return f"{prod} {s.name}" + rng.choice(["", "（夜班报警）", "（周末突发）", "（客户投诉触发）", "", "（YMS 自动报警）"])


def _audit(db: Session, actor: str, action: str, target: str, when, **detail) -> None:
    db.add(AuditLog(actor=actor, action=action, target=target, detail=detail, created_at=when))


def _add_actions(db: Session, case: Case, analysis_id: int, cause_code: str, when, status: str = "todo") -> list[Action]:
    found = cause_by_code(cause_code)
    if not found:
        return []
    c = found[1]
    rows = []
    for kind, text in zip(("containment", "corrective", "preventive"), c.capa):
        a = Action(case_id=case.id, analysis_id=analysis_id, kind=kind, title=text, cause_code=cause_code, owner_role=OWNER[kind],
                   due_days=DUE[kind], due_date=when + timedelta(days=DUE[kind]), status=status, verify_metric=c.metric,
                   created_at=when)
        db.add(a)
        rows.append(a)
    return rows


class _Timeline:
    """按时间顺序回放：建案 → 分析 → CAPA → 关案学习；穿插每月改进周期与 RSI 会话。"""

    def __init__(self, db: Session, now, rng: random.Random):
        self.db, self.now, self.rng = db, now, rng
        self.cases: list[dict] = []

    def days_ago(self, when) -> float:
        return (self.now - when).total_seconds() / 86400

    # ── 案例 ──
    def plan_cases(self) -> None:
        rng = self.rng
        n_closed = HISTORY_CASES
        for k in range(n_closed):
            created = self.now - timedelta(days=HISTORY_DAYS - (HISTORY_DAYS - 52) * k / n_closed)
            self.cases.append({"k": k, "created": created, "closed": created + timedelta(days=rng.uniform(5, 25)), "status": "closed"})
        for j, (d, status) in enumerate(IN_PROGRESS):
            self.cases.append({"k": n_closed + j, "created": self.now - timedelta(days=d, hours=rng.uniform(0, 8)),
                               "closed": None, "status": status})

    def create(self, spec: dict) -> None:
        db, rng = self.db, self.rng
        when, k = spec["created"], spec["k"]
        era = era_at(self.days_ago(when))
        sym = rng.choices(list(SYMPTOM_WEIGHTS), weights=list(SYMPTOM_WEIGHTS.values()))[0]
        dist = truth_for_era(era)[sym]
        cause = rng.choices(list(dist), weights=list(dist.values()))[0]
        strength = rng.uniform(0.7, 1.0) if rng.random() < 0.4 else rng.uniform(0.0, 0.12)
        tool = hotspot_tool(cause, sym, rng)
        ds = simulate(Scenario(sym, cause, strength=strength, seed=1000 + k, n_lots=100, extras={"tool": tool}), when)
        ev = analyze_dataset(ds)
        s = SYMPTOMS[sym]
        site, product = SITES[s.stage]
        top = (ev.get("suspects") or [{}])[0]
        reporter = rng.choice(ENGINEERS[s.stage])
        case = Case(title=_title(sym, rng), stage=s.stage, symptom=sym, product=product, site=site,
                    line=STEPS[top["step"]].name if top.get("step") in STEPS else "",
                    equipment=top.get("value", "") if "#" not in top.get("factor", "#") else "",
                    severity=rng.choice(["S1", "S2", "S2", "S3"]), status="analyzing",
                    description=f"{s.name}，{s.metric} 异常，共 {ev['summary']['fail_lots']} 个失效批次。",
                    data={"失效批次": ev["summary"]["fail_lots"], "批次总数": ev["summary"]["lots"]},
                    evidence=ev, cost_impact=round(rng.uniform(20, 400), 1), downtime_hours=round(rng.uniform(0, 30), 1),
                    reporter=reporter, source=rng.choice(["yms", "yms", "mes", "manual"]), created_at=when)
        db.add(case)
        db.flush()
        _audit(db, reporter, "case.create", case.code, when, title=case.title, source=case.source)
        spec.update(case_id=case.id, cause=cause, sym=sym, reporter=reporter)
        if spec["status"] != "closed":  # 进行中的案例保留原始数据，稍后跑完整分析工作流
            db.add(FabDataset(case_id=case.id, data=ds))
            db.commit()
            return
        hyps, frozen = rank_case(db, ev, sym)
        an = Analysis(case_id=case.id, evidence=ev, hypotheses=[h.as_dict() for h in hyps],
                      counterfactual={"top": frozen[0].cause_code, "top_name": frozen[0].name,
                                      "top_posterior": round(frozen[0].posterior, 3), "changed": frozen[0].cause_code != hyps[0].cause_code},
                      model="offline-rules", created_at=when + timedelta(hours=2))
        db.add(an)
        db.flush()
        _audit(db, "agent", "case.analyze", case.code, when + timedelta(hours=2), top=hyps[0].cause_code,
               posterior=round(hyps[0].posterior, 3))
        _add_actions(db, case, an.id, hyps[0].cause_code, when + timedelta(hours=3), status="doing")
        case.status = "capa"
        db.add(case)
        spec.update(analysis_id=an.id, ranked=[h.cause_code for h in hyps])
        db.commit()

    def close(self, spec: dict) -> None:
        db, rng = self.db, self.rng
        case = db.get(Case, spec["case_id"])
        cause, when, created = spec["cause"], spec["closed"], spec["created"]
        found = cause_by_code(cause)[1]
        acts = db.exec(select(Action).where(Action.case_id == case.id)).all()
        first = spec["ranked"][0]
        owner = rng.choice(ENGINEERS[case.stage])
        if first == cause:
            for a in acts:
                a.status, a.result = "verified", "已验证有效"
                a.completed_at = created + timedelta(days=a.due_days - (1 if rng.random() < 0.8 else -2))
                db.add(a)
                _audit(db, owner, "action.update", f"A-{a.id}", a.completed_at, status="verified")
        else:
            for a in acts:
                if a.kind == "containment":
                    a.status, a.result, a.completed_at = "verified", "围堵有效", created + timedelta(days=1)
                    db.add(a)
                elif a.kind == "corrective" and rng.random() < 0.5:  # 先按首位假设整改，验证无效 → 现场否定
                    a.status, a.result, a.completed_at = "ineffective", "整改后指标未恢复", created + timedelta(days=5)
                    db.add(a)
                    _audit(db, owner, "action.update", f"A-{a.id}", a.completed_at, status="ineffective")
                    refute_cause(db, case, first, when=a.completed_at)
                else:
                    db.delete(a)
            db.flush()
            for kind, text in list(zip(("containment", "corrective", "preventive"), found.capa))[1:]:
                done = created + timedelta(days=DUE[kind] + (1 if rng.random() < 0.8 else -1) + 3)
                db.add(Action(case_id=case.id, analysis_id=spec["analysis_id"], kind=kind, title=text, cause_code=cause,
                              owner_role=OWNER[kind], due_days=DUE[kind] + 4, due_date=created + timedelta(days=DUE[kind] + 4),
                              status="verified", verify_metric=found.metric, result="已验证有效", created_at=created + timedelta(days=4),
                              completed_at=min(done, when)))
        case.cause_code, case.root_cause, case.root_cause_category = cause, found.name, found.six_m
        case.status, case.closed_at = "closed", when
        db.add(case)
        db.flush()
        learn_from_closure(db, case, cause, spec["ranked"], when=when)
        _audit(db, rng.choice(APPROVERS), "case.close", case.code, when, cause_code=cause,
               rank_in_analysis=spec["ranked"].index(cause) + 1 if cause in spec["ranked"] else None)
        db.commit()

    # ── 每月改进周期（夜间定时：同时用全部已验证样本重拟合证据权重）──
    def cycle(self, when) -> None:
        from .rsi.memory import load_memory, refit_and_store

        db = self.db
        mem = load_memory(db)
        mem.era = era_at(self.days_ago(when))
        res = refit_and_store(db, mem, source="nightly", when=when)
        if res.get("n", 0) >= 20:
            _audit(db, "system:nightly", "weights.refit", "evidence-model", when, accepted=res["accepted"], n=res["n"])
        db.commit()
        build_cycle(db, "schedule", now=when + timedelta(minutes=5), actor="system:nightly")

    # ── RSI 会话（面对当时的数字孪生，孪生经验权重 0.5）──
    async def rsi(self, when, prev) -> None:
        lo, hi = self.days_ago(when), (self.days_ago(prev) if prev else HISTORY_DAYS)
        await run_history_session(self.db, when, era_at(lo), [c for c in CHANGES if hi > c.days_ago >= lo])

    # ── 产线变更：同步到孪生，并对受影响症状做部分遗忘 ──
    def change(self, c, when) -> None:
        from .fab.learning import _store
        from .rsi.memory import forget, load_memory, save_memory

        db = self.db
        mem = load_memory(db)
        ev = forget(mem, c.symptom, c.label)
        save_memory(db, mem, when)
        if ev:
            _store(db, [ev], None, when)
        _audit(db, "system:twin", "twin.update", c.symptom, when, change=c.label)
        db.commit()


def _slim(attempts: list[dict]) -> list[dict]:
    return [{"cause_code": a["cause_code"], "cause": a["cause"], "posterior": a["posterior"],
             "verdict": {"passed": a["verdict"]["passed"]}} for a in attempts]


async def run_history_session(db: Session, when, era: int, changes: list, trigger: str = "schedule") -> int:
    """按历史时间回放一次 RSI 会话（与在线工作流同样的三阶段、验收门与记忆合并），全部记录带当时的时间。"""
    from .rsi.agents import Curriculum
    from .rsi.engine import (BREADTH, CAL_TASKS, CAL_WAVE, CONCURRENCY, MAX_DEPTH, PATIENCE, run_session)
    from .rsi.memory import load_memory, save_memory, training_examples

    sess = RSISession(trigger=trigger, status="running", created_at=when,
                      config={"breadth": BREADTH, "concurrency": CONCURRENCY, "cal_tasks": CAL_TASKS, "cal_wave": CAL_WAVE,
                              "max_depth": MAX_DEPTH, "patience": PATIENCE, "sandbox_weight": 0.5, "era": era,
                              "changes": [{"label": c.label, "symptom": c.symptom, "days_ago": c.days_ago} for c in changes]})
    db.add(sess)
    db.commit()
    db.refresh(sess)
    sid = sess.id
    mem = load_memory(db)
    mem.era, mem.examples = era, training_examples(db)
    state, log, done = {"phase": "BRS"}, [], []

    async def notify(ev: dict) -> None:
        if ev["type"] == "phase":
            state["phase"] = ev["phase"]
        elif ev["type"] == "weights":
            log.append({"phase": state["phase"], **{k: ev.get(k) for k in ("accepted", "weights", "n", "gate", "reason", "avg_loglik")}})

    async def commit(rec: dict, events: list[dict]) -> None:
        done.append((rec, events))

    res = await run_session(mem, mode="v2", seed_base=2_000_000 + sid * 1000, sandbox_weight=0.5,
                            curriculum=Curriculum(max_depth=MAX_DEPTH, patience=PATIENCE),
                            twin=FabTwin(now=when, era=era), notify=notify, commit=commit, focus={c.symptom for c in changes})
    span = timedelta(minutes=40)
    tasks = []
    for i, (rec, _events) in enumerate(done):
        t = when + span * (i + 1) / (len(done) + 1)
        task = RSITask(session_id=sid, phase=rec["phase"], step=rec["step"], spec=rec["spec"], verdict=rec["verdict"],
                       attempts=_slim(rec["attempts"]) if rec["phase"] == "CAL" else rec["attempts"], hidden=rec["hidden"],
                       evidence=rec["evidence"], symptom=rec["symptom"], weak=rec["weak"], memory_changes=_events, created_at=t)
        db.add(task)
        tasks.append((task, _events, t))
    db.flush()
    for task, events, t in tasks:
        for e in events:
            db.add(LearningEvent(rsi_task_id=task.id, created_at=t,
                                 **{k: v for k, v in e.items() if k in LearningEvent.model_fields and k != "created_at"}))
    for i, w in enumerate(log):
        if w["accepted"]:
            db.add(EvidenceModel(weights=w["weights"], n=w["n"], avg_loglik=w["avg_loglik"] or 0.0, source="rsi", gate=w["gate"] or {},
                                 created_at=when + span * (i + 1) / (len(log) + 1)))
    finish = when + span
    save_memory(db, mem, when=finish)
    s = res["summary"]
    sess.curve, sess.weights_log, sess.status, sess.finished_at = res["curve"], log, "done", finish
    sess.summary = {**{k: s[k] for k in ("tasks", "passes", "brs", "cal", "drs", "hit1_before", "hit1_after", "partial_before",
                                         "partial_after", "memory_items")},
                    "refits_accepted": sum(1 for w in log if w["accepted"]), "refits_rejected": sum(1 for w in log if not w["accepted"]),
                    "weights": next((w["weights"] for w in reversed(log) if w["accepted"]), None),
                    "plateau": abs(s["hit1_after"] - s["hit1_before"]) < 0.25,
                    "plateau_reason": "no_gap" if changes else "no_change"}
    db.add(sess)
    _audit(db, f"system:{trigger}", "kaizen.session", f"RSI-{sid}", finish, **{k: sess.summary[k] for k in ("tasks", "hit1_before", "hit1_after")})
    db.commit()
    return sid


async def seed(db: Session) -> int:
    if db.exec(select(Case)).first():
        return 0
    ensure_edges(db)
    rng = random.Random(2025)
    assert set(SYMPTOM_WEIGHTS) <= set(SITE_TRUTH)
    now = utcnow()
    tl = _Timeline(db, now, rng)
    tl.plan_cases()
    events = [(c["created"], 0, "create", c) for c in tl.cases]
    events += [(now - timedelta(days=c.days_ago), 1, "change", c) for c in CHANGES]
    events += [(c["closed"], 1, "close", c) for c in tl.cases if c["closed"]]
    events += [(now - timedelta(days=d, hours=-2), 2, "cycle", None) for d in CYCLE_DAYS]
    events += [(now - timedelta(days=d, hours=-3), 3, "rsi", None) for d in RSI_DAYS]
    prev_rsi = None
    cost = {"create": 1, "close": 1, "cycle": 3, "change": 1, "rsi": 30}
    total, done = sum(cost[e[2]] for e in events), 0
    for when, _, kind, spec in sorted(events, key=lambda e: (e[0], e[1])):
        done += cost[kind]
        _progress(0.8 * done / total, f"按时间回放到 {when:%Y-%m-%d}" + ("：RSI 自我进化会话" if kind == "rsi" else ""))
        await asyncio.sleep(0)  # 让出事件循环：生成期间控制台照常响应
        if kind == "create":
            tl.create(spec)
        elif kind == "close":
            tl.close(spec)
        elif kind == "cycle":
            tl.cycle(when)
        elif kind == "change":
            tl.change(spec, when)
        else:
            await tl.rsi(when, prev_rsi)
            prev_rsi = when
    from .rsi.memory import load_memory, refit_and_store

    refit_and_store(db, load_memory(db), source="production", when=now - timedelta(hours=12))
    db.commit()
    # 待分析：刚上报、带完整原始数据（用于现场演示）
    for i, (sym, cause, strength, sev, title, desc, data, cost, *extra) in enumerate(OPEN_CASES):
        s = SYMPTOMS[sym]
        site, product = SITES[s.stage]
        if extra:
            site, product = extra[0].get("site", site), extra[0].get("product", product)
        c = cause_by_code(cause)[1]
        when = now - timedelta(days=2 - i * 0.3)
        case = Case(title=title, stage=s.stage, symptom=sym, product=product, site=site, line=STEPS[c.step].name,
                    severity=sev, status="open", description=desc, data=data, cost_impact=cost,
                    reporter="演示数据", source="seed", created_at=when)
        db.add(case)
        db.flush()
        hot = HOTSPOTS.get(cause)
        db.add(FabDataset(case_id=case.id, data=simulate(Scenario(sym, cause, strength=strength, seed=77 + i,
                                                                  extras={"tool": hot[0] if hot else None}), now)))
        _audit(db, rng.choice(ENGINEERS[s.stage]), "case.create", case.code, when, title=title, source="yms")
    db.commit()
    return HISTORY_CASES + len(IN_PROGRESS) + len(OPEN_CASES)


async def finish_in_progress(db_factory) -> None:
    """进行中的近期案例：跑完整分析工作流（8D、5-Why、CAPA），再把时间与状态还原到当时。"""
    from .services import analyze_case

    with db_factory() as db:
        rows = db.exec(select(Case).where(col(Case.status).in_(["analyzing"]), Case.source != "seed")).all()
        targets = [(c.id, c.created_at) for c in rows]
    plan = dict((d, st) for d, st in IN_PROGRESS)
    rng = random.Random(77)
    for case_id, created in targets:
        async for _ in analyze_case(case_id, actor="agent"):
            pass
        with db_factory() as db:
            case = db.get(Case, case_id)
            now = utcnow()
            d = round((now - created).total_seconds() / 86400)
            status = plan.get(d) or plan[min(plan, key=lambda x: abs(x - d))]
            shift = created + timedelta(hours=2) - now
            for an in db.exec(select(Analysis).where(Analysis.case_id == case_id)).all():
                an.created_at = an.created_at + shift
                db.add(an)
            for a in db.exec(select(Action).where(Action.case_id == case_id)).all():
                a.created_at = a.created_at + shift
                a.due_date = a.created_at + timedelta(days=a.due_days)
                owner = rng.choice(ENGINEERS[case.stage])
                if status == "verifying" or a.kind == "containment" or (status == "capa" and a.kind == "corrective" and d >= 24):
                    a.status, a.result = "done", "已完成，待效果验证"
                    a.completed_at = min(a.due_date + timedelta(days=rng.choice([-1, 0, 0, 1, 2])), now - timedelta(hours=6))
                elif status == "capa":
                    a.status = "doing" if (a.kind == "corrective" or d >= 24) else "todo"
                else:  # 分析中：围堵已完成，纠正 / 预防措施待分析结论确认后排期
                    a.status, a.due_date = "todo", now + timedelta(days=a.due_days)
                db.add(a)
                if a.completed_at:
                    db.add(AuditLog(actor=owner, action="action.update", target=f"A-{a.id}", detail={"status": a.status},
                                    created_at=a.completed_at))
            for row in db.exec(select(AuditLog).where(AuditLog.target == case.code, AuditLog.action != "case.create",
                                                      AuditLog.created_at >= now - timedelta(minutes=10))).all():
                row.created_at = row.created_at + shift
                db.add(row)
            case.status = status
            db.add(case)
            db.commit()


def _reset_demo() -> None:
    """上次演示数据回放中途被打断：清空后重新生成（只在 KAIZEN_SEED_DEMO 打开时发生）。"""
    from sqlmodel import SQLModel

    from .db import get_engine

    SQLModel.metadata.drop_all(get_engine())
    SQLModel.metadata.create_all(get_engine())


async def prepare_demo(rsi_on_startup: bool = True) -> None:
    """准备演示数据：按时间回放去年至今 → 进行中案例的完整分析 → 今天的改进周期 → 冷启动基准 → 今天的 RSI 会话。

    可以在后台运行（控制台先打开，数据陆续出现）；中途重启会从断点继续：回放没做完就重来，后面的步骤缺什么补什么。
    """
    from .db import session_scope
    from .improvement import run_cycle
    from .models import ImprovementCycle, RSIBenchmark

    bench = None
    try:
        with session_scope() as db:
            has_cases = db.exec(select(Case)).first() is not None
            replayed = db.exec(select(ImprovementCycle).where(ImprovementCycle.trigger == "seed")).first() is not None
            has_bench = db.exec(select(RSIBenchmark)).first() is not None
        if rsi_on_startup and not has_bench:  # 冷启动基准不依赖数据库：放到独立进程里，和历史回放同时跑
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor

            from .rsi.engine import benchmark_in_process

            ex = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
            bench = asyncio.get_running_loop().run_in_executor(ex, benchmark_in_process)
            ex.shutdown(wait=False)
        if has_cases and not replayed:
            log.warning("上次演示数据回放没有完成，重新生成")
            _reset_demo()
            has_cases = False
        if not has_cases:
            _progress(0.0, "开始按时间回放去年至今的产线")
            with session_scope() as db:
                n = await seed(db)
            _progress(0.82, "为进行中的案例运行完整分析工作流")
            await finish_in_progress(session_scope)
            _progress(0.86, "今天的改进周期")
            await run_cycle(trigger="seed")
            log.info("已写入 %s 个演示案例（约 18 个月的按时间回放历史）并完成今天的改进周期", n)
        if rsi_on_startup:
            from .rsi.workflow import run_benchmark, start_session

            with session_scope() as db:
                has_live = db.exec(select(RSISession).where(RSISession.trigger == "seed")).first() is not None
            if not has_bench:
                _progress(0.88, "新产线冷启动基准（4 个种子 × 6 种模式）")
                try:
                    results = await bench
                except Exception:  # noqa: BLE001  子进程不可用时退回到本进程
                    log.warning("独立进程跑基准失败，改在本进程运行", exc_info=True)
                    results = None
                if results is None:
                    await run_benchmark(trigger="seed")
                else:
                    with session_scope() as db:
                        db.add(RSIBenchmark(trigger="seed", results=results))
                        db.commit()
            if not has_live:
                _progress(0.94, "今天的 RSI 自我进化会话")
                _, _, handle = await start_session(trigger="seed", return_handle=True)
                summary = await handle.get_result()
                log.info("RSI：冷启动基准已完成；今天的在线会话 %s 道题，留出集首位命中率 %s → %s",
                         summary["tasks"], summary["hit1_before"], summary["hit1_after"])
        _progress(1.0, "演示数据已就绪", state="ready")
    except Exception as e:  # noqa: BLE001
        log.exception("演示数据准备失败")
        _progress(STATE["progress"], f"演示数据准备失败：{e}", state="error")


def demo_truth(case: Case) -> str | None:
    """演示用：待分析案例的“真因”（模拟工程师现场确认的结果）。"""
    for sym, cause, *_ in OPEN_CASES:
        if case.symptom == sym and case.source == "seed":
            return cause
    return None


assert all(c in {x.code for x in CAUSES[s]} for s, d in SITE_TRUTH.items() for c in d)  # noqa: S101
