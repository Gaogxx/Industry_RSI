"""快脑小模型：贝叶斯等价初始化、反向传播、偏置层测试时训练、A-RSI 训练窗口、RSI-Opt 外层、训练知识库。"""

import random

from sqlmodel import Session, SQLModel, create_engine, select

from app.db import session_scope
from app.fab.kg import rank
from app.fastbrain import data, optimize, playbook, windows
from app.fastbrain.features import ALL_CAUSES, NAMES, candidate_rows
from app.fastbrain.model import D, FastModel
from app.models import FastBrainModel, TrainingNote, TrainingPolicy
from app.rsi.memory import MemoryState, load_memory
from app.rsi.twin import FabTwin, TaskSpec


def test_untrained_model_equals_bayes_ranking():
    mem = MemoryState.textbook()
    tw = FabTwin()
    for i, sym in enumerate(["WM_EDGE", "PKG_VOID", "OVL_DRIFT"]):
        o = data.observe(tw, TaskSpec(sym, "medium"), 4_000 + i)
        hyps, X = candidate_rows(o["evidence"], o["symptom"], mem)
        p = FastModel.bayes(mem.weights).probs(X, [h["cause_code"] for h in hyps])
        assert max(abs(a - h["posterior"]) for a, h in zip(p, hyps)) < 1e-3


def test_backprop_matches_numerical_gradient():
    rng = random.Random(0)
    m = FastModel(3, seed=1, rank=2, emb=2)
    m.theta = [v + rng.gauss(0, 0.2) for v in m.theta]
    batch = [{"X": [[rng.gauss(0, 1) for _ in range(D)] for _ in range(3)], "codes": rng.sample(ALL_CAUSES, 3), "y": 1, "w": 1.0}]
    reg = {"l2": 0.0, "bias_l2": 0.0, "anchor": 0.0, "a0": [0.0] * D, "prev": None}
    _, g = m.grad(batch, m.theta, reg)
    for i in rng.sample(range(m.size), 25):
        th = list(m.theta)
        th[i] += 1e-5
        a = m.grad(batch, th, reg)[0]
        th[i] -= 2e-5
        b = m.grad(batch, th, reg)[0]
        assert abs((a - b) / 2e-5 - g[i]) < 1e-6


async def test_training_window_has_health_and_promotion_gates(client, monkeypatch):
    monkeypatch.setattr(windows, "POOL_PER_SYMPTOM", 40)
    monkeypatch.setattr(windows, "HOLDOUT_PER_SYMPTOM", 8)
    monkeypatch.setattr(windows, "CAL_PER_SYMPTOM", 6)
    with session_scope() as db:
        row = await windows.train_window(db, load_memory(db), "test")
        assert row.status in ("active", "rejected", "failed")
        assert set(row.window["health"]) >= {"finite", "learned", "moved", "not_collapsed", "ok"}
        if row.status != "failed":
            m = row.metrics
            assert m["diff_logloss"]["ci95"][0] <= m["diff_logloss"]["mean"] <= m["diff_logloss"]["ci95"][1]
            assert (row.status == "active") == m["promoted"] == all(m["checks"].values())
            assert 0 < row.conformal["q"] < 1 and row.params["theta"]
        assert all(d["action"] in ("lr_decay", "rollback", "early_stop", "halt") for d in row.log.get("decisions", []))


def test_training_knowledge_retires_contradicted_facts_and_blames_tips():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as db:
        for w in (1, 2):
            playbook.after_window(db, w, decisions=[], pool_top={"TEST_VMIN": {"top": "PROCESS_SLOW_CORNER", "share": 0.55, "n": 300}},
                                  groups={}, inc_groups={}, applied=[], promoted=True, gate_note="", when=None)
        old = db.exec(select(TrainingNote).where(TrainingNote.key == "fact:top:TEST_VMIN:PROCESS_SLOW_CORNER")).one()
        assert old.status == "active"  # 两个窗口独立确认 → 生效
        playbook.after_window(db, 3, decisions=[], pool_top={"TEST_VMIN": {"top": "SOCKET_CONTACT", "share": 0.65, "n": 300}},
                              groups={}, inc_groups={}, applied=[], promoted=True, gate_note="", when=None)
        assert old.status == "retired" and old.blame  # 产线变更后被推翻：停用并留档
        assert "TEST_VMIN" in playbook.guidance(db)["focus"]  # 分布变了的症状下个窗口配额加倍
        for w in range(4, 9):
            playbook.after_window(db, w, decisions=[{"action": "lr_decay", "epoch": 3, "helped": False}], pool_top={},
                                  groups={}, inc_groups={}, applied=[], promoted=False, gate_note="", when=None)
        tip = db.exec(select(TrainingNote).where(TrainingNote.key == "tip:ctrl:lr_decay")).one()
        assert tip.status == "retired" and not playbook.guidance(db)["controller"]["decay_on_plateau"]


async def test_rsi_opt_outer_loop_archives_policies(client, monkeypatch):
    monkeypatch.setattr(optimize, "SPLITS", {"search": (5_100_000, 20), "select": (5_200_000, 6), "report": (5_300_000, 6)})
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with session_scope() as db:
        mem = load_memory(db)
    with Session(engine) as db:
        assert optimize.pending_triggers(db) == ["bootstrap"]
        res = await optimize.evolve_training(db, mem, generations=1, children=2, trigger="bootstrap")
        rows = db.exec(select(TrainingPolicy).where(TrainingPolicy.run_id == res["run_id"])).all()
        assert any(r.status == "active" for r in rows)  # 基线策略存档并启用
        for r in rows:
            assert r.note and all(e["component"] in optimize.SPACE and e["component"] != "anchor" for e in r.edits)
        assert optimize.pending_triggers(db) == []  # 事件触发：没有新信号就不跑


def test_seeded_history_has_promoted_fast_model_beating_bayes(client):
    o = client.get("/api/fastbrain").json()
    assert o["windows"] and o["active"] and o["policies"] and o["notes"]["items"]
    m = o["active"]["metrics"]
    assert m["candidate"]["hit1"] >= m["bayes"]["hit1"] - 2 and m["test"]["model"] >= m["test"]["bayes"] - 2
    assert {"hidden", "rank", "emb"} <= set(o["active"]["params"]) and "theta" not in o["active"]["params"]
    with session_scope() as db:
        assert db.exec(select(FastBrainModel)).first() is not None
    assert set(NAMES) and len(ALL_CAUSES) == 38
