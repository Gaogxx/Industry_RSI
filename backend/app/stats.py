"""共用的小统计工具：配对 bootstrap 区间、Wilson 区间。"""

from __future__ import annotations

import math
import random


def paired_ci(a: list[float], b: list[float], reps: int = 1000, seed: int = 7, alpha: float = 0.05) -> tuple[float, float, float]:
    """配对 bootstrap：mean(a − b) 及其 (1 − alpha) 区间（默认 95%；多重比较时传入校正后的 alpha）。"""
    d = [x - y for x, y in zip(a, b)]
    if not d:
        return 0.0, 0.0, 0.0
    reps = max(reps, int(20 / max(alpha, 1e-4)))  # alpha 很小时多抽一些，保证尾部分位数可用
    rng = random.Random(seed)
    n = len(d)
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    lo = means[min(reps - 1, int(alpha / 2 * reps))]
    hi = means[max(0, int((1 - alpha / 2) * reps) - 1)]
    return round(sum(d) / n, 4), round(lo, 4), round(hi, 4)


def wilson(k: int, n: int) -> list[float] | None:
    if not n:
        return None
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0, c - r), 3), round(min(1, c + r), 3)]
