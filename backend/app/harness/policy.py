"""harness 策略：快慢路由与测试时训练的旋钮。线上用存档里 status=active 的最新一版；没有时用默认值。"""

from __future__ import annotations

from sqlmodel import Session, col, select

from ..models import HarnessPolicy

# 组件 → (默认值, 步长, 下限, 上限, 说明)。只有这些键可以被进化（白名单，防止候选夹带别的东西）
COMPONENTS: dict[str, tuple] = {
    "fast_min_reliability": (0.85, 0.03, 0.70, 0.97, "可靠度达到多少才允许快脑直接给结论"),
    "audit_every": (5, 1, 2, 10, "同一症状连续多少次快脑之后，强制慢脑抽查一次"),
    "ttt_tasks": (36, 12, 12, 96, "慢脑近邻练习的题量（其中三分之一留作探测）"),
    "ttt_keep": (0.1, 0.1, 0.0, 0.5, "近邻练习前，临时记忆里该症状旧份额的保留比例（测试时适应）"),
    "ttt_prior_weight": (1.0, 0.25, 0.25, 1.0, "近邻练习对临时先验的权重"),
    "adopt_min_gain": (0.1, 0.05, 0.0, 0.3, "近邻题命中率至少提高多少才采纳练习后的结论（默认至少多答对 2 道探测题）"),
    "use_experience": (1, 1, 0, 1, "是否按经验库调整（跳过无效练习、提前转慢脑）"),
}
DEFAULT = {k: v[0] for k, v in COMPONENTS.items()}
CONFORMAL_ALPHA = 0.10  # 保形预测：给出覆盖真因 90% 的根因集合（不参与进化，属于可靠性规格）


def clamp(params: dict) -> dict:
    out = dict(DEFAULT)
    for k, v in (params or {}).items():
        if k in COMPONENTS:
            _, step, lo, hi, _ = COMPONENTS[k]
            v = min(max(v, lo), hi)
            out[k] = int(round(v)) if isinstance(step, int) else round(float(v), 3)
    return out


def active(db: Session) -> tuple[int, dict]:
    row = db.exec(select(HarnessPolicy).where(HarnessPolicy.status == "active").order_by(col(HarnessPolicy.id).desc())).first()
    return (row.id, clamp(row.params)) if row else (0, dict(DEFAULT))


def complexity(params: dict) -> float:
    """策略的运行成本（越低越好）：慢脑练习题量为主。用于 RRSI 的“噪声内改进只在降低复杂度时接受”。"""
    return params["ttt_tasks"] / COMPONENTS["ttt_tasks"][3] + 0.1 * (1 - params["fast_min_reliability"])
