"""训练知识库：RSI-Opt 的跨 run 知识——把历次训练里观察到的东西带到下一次训练（控制器、出题配比、策略提议都读它）。

分两类条目（每条一个 Beta 账本，失败追责到具体条目，停用但留档）：

事实（关于数据与环境，必须可核实，多次确认才生效，被推翻立即停用）
  fact:top:{症状}:{根因}       训练池里该症状的首位根因（孪生当前时段的真实分布）
  fact:shift:{症状}            上一条被新窗口推翻了 → 产线变了，接下来两个窗口该症状配额加倍
  fact:weak:decoy / hard       模型在 PM 干扰题 / 困难题上明显偏弱（留出集分组命中率差 ≥ 5 个百分点）

做法（关于怎么训练，带成败记录）
  tip:ctrl:lr_decay            平台期学习率减半（之后两轮验证损失是否创新低）
  tip:ctrl:rollback            验证损失回升时回滚到最好的一轮并减半
  tip:pool:focus_shift         分布变了的症状配额加倍（该窗口是否通过晋升门）
  tip:pool:decoy_boost         干扰题偏弱时提高干扰题比例（干扰题分组命中率是否提高）
  tip:policy:{旋钮}:{up|down}   RSI-Opt 外层改某个训练旋钮（候选是否被接受）

追责：窗口没通过晋升门 → 本窗口用到的出题做法各记一次失败（写明哪个窗口、为什么）；
P<0.45 且失败≥4 次停用——控制器和策略提议之后不再用它，但记录保留可查。
（TTSE 的双轨经验用在了 RSIAgent 的三个 agent 上，见 rsi/experience.py；这里不是 TTSE。）
"""

from __future__ import annotations

from sqlmodel import Session, col, select

from ..fab.flow import SYMPTOMS, cause_by_code
from ..ledger import p_success, record, trials
from ..models import TrainingNote, utcnow

FACT_MIN_CONFIRM = 2  # 训练窗口不频繁：事实被两个窗口独立确认即生效
WEAK_GAP = 5.0
SHIFT_WINDOWS = 2
DECOY_BOOST = 0.15

TIPS = {
    "tip:ctrl:lr_decay": "验证损失进入平台期时把学习率减半",
    "tip:ctrl:rollback": "验证损失回升时回滚到最好的一轮并把学习率减半",
    "tip:pool:focus_shift": "产线变更后分布变了的症状，训练池配额加倍",
    "tip:pool:decoy_boost": "模型在 PM 干扰题上偏弱时，把干扰题比例提高 15 个百分点",
}


def _get(db: Session, key: str, **init) -> TrainingNote:
    x = db.exec(select(TrainingNote).where(TrainingNote.key == key)).first()
    if x is None:
        x = TrainingNote(key=key, **init)
        db.add(x)
        db.flush()
    return x


def _tip(db: Session, key: str, when=None) -> TrainingNote:
    return _get(db, key, track="tip", text=TIPS.get(key, key), created_at=when or utcnow(), updated_at=when or utcnow())


def _usable(db: Session, key: str) -> bool:
    x = db.exec(select(TrainingNote).where(TrainingNote.key == key)).first()
    return x is None or x.status != "retired"


def guidance(db: Session) -> dict:
    """读手册：本窗口控制器启用哪些规则、哪些症状配额加倍、干扰题比例加多少。返回用到的条目（用于追责）。"""
    applied, ctl = [], {}
    for name, key in (("decay_on_plateau", "tip:ctrl:lr_decay"), ("rollback_on_rise", "tip:ctrl:rollback")):
        ctl[name] = _usable(db, key)
    shifts = db.exec(select(TrainingNote).where(col(TrainingNote.key).startswith("fact:shift:"), TrainingNote.status == "active")).all()
    focus = {x.symptom for x in shifts if (x.action or {}).get("windows_left", 0) > 0}
    if focus and _usable(db, "tip:pool:focus_shift"):
        applied.append("tip:pool:focus_shift")
    else:
        focus = set()
    weak = db.exec(select(TrainingNote).where(TrainingNote.key == "fact:weak:decoy")).first()
    boost = 0.0
    if weak is not None and weak.status in ("active", "stable") and _usable(db, "tip:pool:decoy_boost"):
        boost = DECOY_BOOST
        applied.append("tip:pool:decoy_boost")
    return {"controller": ctl, "focus": sorted(focus), "decoy_boost": boost, "applied": applied}


def _fact(db: Session, key: str, symptom: str, text: str, action: dict, ok: bool, ev: dict, when, blame: str = "") -> TrainingNote:
    x = _get(db, key, track="fact", symptom=symptom, text=text, action=action, status="candidate",
             created_at=when, updated_at=when)
    record(x, ok, ev, when, blame)
    if x.status == "candidate" and ok and trials(x) >= FACT_MIN_CONFIRM:
        x.status = "active"
    db.add(x)
    return x


def after_window(db: Session, window_id: int, *, decisions: list[dict], pool_top: dict, groups: dict, inc_groups: dict,
                 applied: list[str], promoted: bool, gate_note: str, when=None) -> list[dict]:
    """窗口结束后写手册：事实轨核实/推翻，做法轨记成败与追责。返回本次变动（给界面看）。"""
    when = when or utcnow()
    touched: list[TrainingNote] = []
    ev = {"window": window_id}
    # ① 控制器做法：看干预之后验证损失是否改善
    for d in decisions:
        key = {"lr_decay": "tip:ctrl:lr_decay", "rollback": "tip:ctrl:rollback"}.get(d["action"])
        if key and d.get("helped") is not None:
            x = _tip(db, key, when)
            record(x, d["helped"], {**ev, "epoch": d["epoch"]}, when,
                   blame=f"窗口 #{window_id} 第 {d['epoch']} 轮{'减半' if d['action'] == 'lr_decay' else '回滚'}后两轮内验证损失没有创新低")
            touched.append(x)
    # ② 事实轨：训练池里各症状的首位根因
    for sym, st in pool_top.items():
        found = cause_by_code(st["top"])
        name = found[1].name if found else st["top"]
        prefix = f"fact:top:{sym}:"
        for old in db.exec(select(TrainingNote).where(col(TrainingNote.key).startswith(prefix), TrainingNote.status != "retired")).all():
            if old.key != prefix + st["top"]:
                record(old, False, ev, when, blame=f"窗口 #{window_id}：训练池里首位变为「{name}」（{st['share']:.0%}）")
                old.status, old.retired_reason = "retired", f"被窗口 #{window_id} 推翻：产线变更后首位根因变了"
                db.add(old)
                touched.append(old)
                sh = _get(db, f"fact:shift:{sym}", track="fact", symptom=sym, status="active", created_at=when, updated_at=when,
                          text=f"{SYMPTOMS[sym].name if sym in SYMPTOMS else sym}：根因分布已变化，接下来 {SHIFT_WINDOWS} 个窗口配额加倍")
                sh.status, sh.action, sh.updated_at = "active", {"windows_left": SHIFT_WINDOWS, "since_window": window_id}, when
                sh.evidence = (list(sh.evidence) + [{**ev, "new_top": st["top"]}])[-30:]
                db.add(sh)
                touched.append(sh)
        touched.append(_fact(db, prefix + st["top"], sym,
                             f"{SYMPTOMS[sym].name if sym in SYMPTOMS else sym} 当前首位根因是「{name}」（训练池 {st['share']:.0%}）",
                             {"cause": st["top"]}, True, {**ev, "share": st["share"], "n": st["n"]}, when))
    # 用过的“配额加倍”倒计时
    if "tip:pool:focus_shift" in applied:
        for sh in db.exec(select(TrainingNote).where(col(TrainingNote.key).startswith("fact:shift:"), TrainingNote.status == "active")).all():
            left = (sh.action or {}).get("windows_left", 0) - 1
            sh.action = {**(sh.action or {}), "windows_left": max(0, left)}
            if left <= 0:
                sh.status, sh.retired_reason = "retired", "加倍配额已用完（留档）"
            db.add(sh)
    # ③ 事实轨：模型的薄弱分组（干扰题 / 困难题）
    for g in ("decoy", "hard"):
        if g not in groups:
            continue
        gap = groups[g]["rest"] - groups[g]["hit1"]
        label = {"decoy": "PM 干扰题", "hard": "困难题"}[g]
        key = f"fact:weak:{g}"
        exists = db.exec(select(TrainingNote).where(TrainingNote.key == key)).first()
        if gap >= WEAK_GAP or exists is not None:
            touched.append(_fact(db, key, "", f"快脑模型在{label}上偏弱（留出集命中率低 {gap:.1f} 个百分点）", {"group": g},
                                 gap >= WEAK_GAP, {**ev, "gap": round(gap, 1)}, when,
                                 blame=f"窗口 #{window_id}：{label}差距缩小到 {gap:.1f} 个百分点"))
    # ④ 出题做法：干扰题加量是否让干扰题分组变好；配额加倍是否让窗口通过晋升门
    if "tip:pool:decoy_boost" in applied and "decoy" in groups and "decoy" in inc_groups:
        better = groups["decoy"]["hit1"] > inc_groups["decoy"]["hit1"]
        x = _tip(db, "tip:pool:decoy_boost", when)
        record(x, better, {**ev, "decoy_hit1": groups["decoy"]["hit1"], "before": inc_groups["decoy"]["hit1"]}, when,
               blame=f"窗口 #{window_id}：加量后干扰题命中率没有提高")
        touched.append(x)
    if "tip:pool:focus_shift" in applied:
        x = _tip(db, "tip:pool:focus_shift", when)
        record(x, promoted, ev, when, blame=f"窗口 #{window_id} 未通过晋升门：{gate_note}")
        touched.append(x)
    db.flush()
    seen, out = set(), []
    for x in touched:
        if x.key not in seen:
            seen.add(x.key)
            out.append(view(x))
    return out


def record_policy_edit(db: Session, component: str, direction: int, accepted: bool, run_id: str, note: str, when=None) -> None:
    key = f"tip:policy:{component}:{'up' if direction > 0 else 'down'}"
    x = _get(db, key, track="tip", text=f"训练策略：把 {component} 调{'高' if direction > 0 else '低'}",
             action={"component": component, "direction": direction}, created_at=when or utcnow(), updated_at=when or utcnow())
    record(x, accepted, {"run": run_id}, when, blame=f"{run_id}：{note}")
    db.add(x)


def tip_weight(db: Session, component: str, direction: int) -> float:
    """策略提议的权重：做过且有效的方向更常被尝试；停用的方向几乎不再尝试。"""
    x = db.exec(select(TrainingNote).where(TrainingNote.key == f"tip:policy:{component}:{'up' if direction > 0 else 'down'}")).first()
    if x is None:
        return 1.0
    if x.status == "retired":
        return 0.15
    return 0.5 + p_success(x)


def view(x: TrainingNote) -> dict:
    return {"id": x.id, "track": x.track, "key": x.key, "symptom": x.symptom, "text": x.text, "status": x.status,
            "p": round(p_success(x), 3), "n": int(trials(x)), "retired_reason": x.retired_reason,
            "blame": list(x.blame)[-3:], "updated_at": x.updated_at}


def overview(db: Session) -> dict:
    rows = db.exec(select(TrainingNote).order_by(col(TrainingNote.updated_at).desc())).all()
    counts = {f"{t}:{s}": sum(1 for x in rows if x.track == t and x.status == s)
              for t in ("fact", "tip") for s in ("candidate", "active", "stable", "retired")}
    return {"counts": counts, "items": [view(x) for x in rows[:60]]}
