"""经验条目的 Beta 账本（harness 改进经验库与快脑训练手册共用）。

参照 Bayesian-Agent（arXiv:2606.08348）：每条经验是一个假设，用 Beta 后验记成败；
P(成功) < 0.45 且失败 ≥ 4 次停用（不删，留档可查），P ≥ 0.72 且 ≥ 3 次标为稳定。
"""

from __future__ import annotations

from .models import utcnow

RETIRE_P, RETIRE_FAILS = 0.45, 4
STABLE_P, STABLE_N = 0.72, 3


def p_success(x) -> float:
    return x.alpha / (x.alpha + x.beta)


def trials(x) -> float:
    return x.alpha + x.beta - 2


def settle(x) -> None:
    p, fails = p_success(x), x.beta - 1
    if x.status != "retired" and p < RETIRE_P and fails >= RETIRE_FAILS:
        x.status, x.retired_reason = "retired", f"成功率 {p:.0%}，失败 {fails:.0f} 次（低于 {RETIRE_P:.0%} 且失败 ≥ {RETIRE_FAILS} 次）"
    elif x.status == "active" and p >= STABLE_P and trials(x) >= STABLE_N:
        x.status = "stable"
    elif x.status == "stable" and p < STABLE_P:
        x.status = "active"


def record(x, ok: bool, evidence: dict, when=None, blame: str = "") -> None:
    """记一次成败；失败且写明原因时进入追责记录（最多留 20 条）。"""
    if ok:
        x.alpha += 1
    else:
        x.beta += 1
        if blame:
            x.blame = (list(x.blame) + [{**evidence, "why": blame}])[-20:]
    x.evidence = (list(x.evidence) + [evidence])[-30:]
    x.updated_at = when or utcnow()
    settle(x)
