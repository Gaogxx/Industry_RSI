"""harness 的组件配置：快慢路由、慢脑各组件的开关与旋钮、裁决方式、大模型调查员，以及条件规则。

线上用存档里 status=active 的最新一版；没有时用默认值。RRSI 进化的就是这份配置（见 evolve.py）。

条件规则（来自改进经验库的提议，见 evolve.rule_proposals）：
  {"if": {"symptom": "WM_EDGE", "evidence": "weak"}, "set": {"slow.ttt": 0}, "why": "...", "source": "..."}
  if 只能用可观察的条件字段（症状、证据强弱、富集类型、PM 后突变），set 只能改白名单里的开关——防止规则夹带
  真值、种子、案例编号这类泄露信息（防泄漏审查见 evolve.critic）。
"""

from __future__ import annotations

from sqlmodel import Session, col, select

from ..fab.flow import SYMPTOMS
from ..models import HarnessPolicy

# 组件 → (默认值, 步长, 下限, 上限, 说明)。只有这些键可以被进化（白名单）
COMPONENTS: dict[str, tuple] = {
    "router.fast_min_conf": (0.85, 0.03, 0.70, 0.97, "快脑模型的校准概率达到多少才允许直接给结论"),
    "router.audit_every": (5, 1, 2, 10, "同一症状连续多少次快脑之后，强制慢脑抽查一次"),
    "slow.ttt": (1, 1, 0, 1, "慢脑：近邻训练（临时记忆在近邻题上学，快脑模型读临时记忆重排）"),
    "slow.ttt_tasks": (36, 12, 12, 72, "近邻题量（其中三分之一留作探测题）"),
    "slow.ttt_keep": (0.1, 0.1, 0.0, 0.5, "近邻练习前，临时记忆里该症状旧份额的保留比例"),
    "arbiter.min_gain": (0.1, 0.05, 0.0, 0.3, "挑战者在探测题上至少比快脑多答对多少比例才改判"),
    "experience.use": (1, 1, 0, 1, "是否按改进经验库调整（跳过反复无效的组件、同条件快脑答错过就转慢脑）"),
    "investigator.mode": (1, 1, 0, 2, "大模型调查员（配了 Key 才生效）：0 不调用；1 慢脑都调用；2 只在 S1 或探测打平时调用"),
}
DEFAULT: dict = {k: v[0] for k, v in COMPONENTS.items()} | {"rules": []}
RULE_IF = {"symptom": set(SYMPTOMS), "evidence": {"strong", "weak"}, "factor": {"tool", "batch", "none"}, "after_pm": {True, False}}
RULE_SET = {"slow.ttt": {0, 1}, "router.force_slow": {1}}
MAX_RULES = 6


def valid_rule(rule: dict) -> str:
    """规则的格式审查：返回空串表示合格，否则返回原因。"""
    cond, sets = rule.get("if") or {}, rule.get("set") or {}
    if not cond or not sets:
        return "条件或动作为空"
    for k, v in cond.items():
        if k not in RULE_IF:
            return f"条件字段「{k}」不可观察或不在白名单"
        if v not in RULE_IF[k]:
            return f"条件值 {k}={v} 不合法"
    for k, v in sets.items():
        if k not in RULE_SET or v not in RULE_SET[k]:
            return f"动作「{k}={v}」不在白名单"
    return ""


def rule_id(rule: dict) -> str:
    c = ",".join(f"{k}={rule['if'][k]}" for k in sorted(rule["if"]))
    s = ",".join(f"{k}={rule['set'][k]}" for k in sorted(rule["set"]))
    return f"{c}→{s}"


def clamp(params: dict) -> dict:
    out = dict(DEFAULT)
    for k, v in (params or {}).items():
        if k in COMPONENTS:
            _, step, lo, hi, _ = COMPONENTS[k]
            v = min(max(v, lo), hi)
            out[k] = int(round(v)) if isinstance(step, int) else round(float(v), 3)
    rules, seen = [], set()
    for r in (params or {}).get("rules", []) or []:
        if not valid_rule(r) and rule_id(r) not in seen:
            seen.add(rule_id(r))
            rules.append({"if": dict(r["if"]), "set": dict(r["set"]), "why": r.get("why", ""), "source": r.get("source", "")})
    out["rules"] = rules[:MAX_RULES]
    return out


def matches(rule: dict, symptom: str, cond: dict) -> bool:
    ctx = {"symptom": symptom, **cond}
    return all(ctx.get(k) == v for k, v in rule["if"].items())


def effective(params: dict, symptom: str, cond: dict) -> tuple[dict, list[str]]:
    """把匹配的条件规则叠加到全局配置上（后出现的规则优先）。"""
    out, hit = dict(params), []
    for r in params.get("rules", []):
        if matches(r, symptom, cond):
            out.update(r["set"])
            hit.append(rule_id(r))
    return out, hit


def active(db: Session) -> tuple[int, dict]:
    row = db.exec(select(HarnessPolicy).where(HarnessPolicy.status == "active").order_by(col(HarnessPolicy.id).desc())).first()
    return (row.id, clamp(row.params)) if row else (0, clamp({}))


def complexity(params: dict) -> float:
    """配置的复杂度：启用的慢脑组件 + 规则条数（RRSI 的正则项之一；实测成本另算）。"""
    return params["slow.ttt"] + 0.5 * len(params.get("rules", []))
