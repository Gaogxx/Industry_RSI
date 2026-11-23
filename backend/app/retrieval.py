"""混合检索：BM25（中文二元切分）+ 可选向量检索，RRF 融合。

案例库在千到十万级时纯 Python BM25 足够快；更大规模时把 `search_cases`
换成 pgvector / Milvus 原生混合检索即可，接口不变。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from sqlmodel import Session, select

from .llm import embed
from .fab.flow import SYMPTOMS
from .models import Case

_CJK = re.compile(r"[一-鿿]+")
_WORD = re.compile(r"[a-zA-Z][a-zA-Z0-9_\-\.]*|\d+(?:\.\d+)?")


def tokenize(text: str) -> list[str]:
    text = text.lower()
    toks = _WORD.findall(text)
    for seg in _CJK.findall(text):
        if len(seg) == 1:
            toks.append(seg)
        toks.extend(seg[i:i + 2] for i in range(len(seg) - 1))
    return toks


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.avgdl = sum(map(len, docs)) / max(len(docs), 1)
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in docs]

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, d in zip(self.tf, self.docs):
            s = 0.0
            for q in query:
                if q in tf:
                    f = tf[q]
                    s += self.idf[q] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
            out.append(s)
        return out


def rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return fused


def _cos(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def case_text(c: Case) -> str:
    sym = SYMPTOMS[c.symptom].name if c.symptom in SYMPTOMS else ""
    return " ".join([c.title, c.description, sym, c.root_cause, c.equipment, c.line, c.product])


@dataclass
class Hit:
    case: Case
    score: float


async def search_cases(session: Session, query: str, *, stage: str | None = None, exclude_id: int | None = None,
                       limit: int = 5, closed_only: bool = False) -> list[Hit]:
    stmt = select(Case)
    if closed_only:
        stmt = stmt.where(Case.status == "closed")
    cases = [c for c in session.exec(stmt).all() if c.id != exclude_id]
    if not cases:
        return []
    texts = [case_text(c) for c in cases]
    bm = BM25([tokenize(t) for t in texts]).scores(tokenize(query))
    if stage:  # 同工段加权放在融合前，避免跨工段的泛化词（如“失效”“超标”）抢排名
        bm = [s * (1.6 if c.stage == stage else 1.0) for s, c in zip(bm, cases)]
    cutoff = 0.2 * max(bm) if bm else 0  # 相对阈值：过滤只沾到一两个泛化词的弱相关案例
    rankings = [sorted((i for i in range(len(cases)) if bm[i] > 0 and bm[i] >= cutoff), key=lambda i: -bm[i])]

    vecs = await embed([query] + texts)
    if vecs:
        sims = [_cos(vecs[0], v) for v in vecs[1:]]
        rankings.append(sorted(range(len(cases)), key=lambda i: -sims[i])[: limit * 4])

    fused = rrf(rankings)
    top = sorted(fused.items(), key=lambda kv: -kv[1])[:limit]
    return [Hit(cases[i], round(s * 1000, 2)) for i, s in top]
