"""Harness 层：快慢双脑路由、近邻练习（测试时训练）、双轨经验库、RRSI 策略进化、可靠度模型与保形预测。"""

import random

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from app.db import session_scope
from app.harness import brain, evolve, experience, policy, train
from app.harness.ttt import test_time_training as run_ttt
from app.models import Case, HarnessExperience, HarnessPolicy, HarnessReceipt
from app.rsi.memory import load_memory
from app.rsi.twin import FabTwin


def test_seeded_history_has_dual_brain_receipts_experiences_and_versions(client):
    o = client.get("/api/harness").json()
    r = o["receipts"]
    assert r["total"] >= 120 and r["verified"] >= 100 and 0 < r["fast"] < r["total"]
    assert r["fast_verified"] and r["fast_correct"] / r["fast_verified"] >= 0.9  # 快脑只在可靠时接手
    assert r["ttt"] > 0 and set(r["reasons"]) <= set(brain.REASON_TEXT)
    assert sum(o["experiences"]["counts"].values()) > 0
    assert o["archive"] and o["models"] and o["benchmark"]
    b = o["benchmark"]["results"]
    assert b["dual"]["fast_wrong"] <= b["fast_only"]["fast_wrong"]
    assert b["stress"]["dual"]["hit1"] > b["stress"]["fast_only"]["hit1"]  # 产线刚变更时近邻练习能追回


def test_route_rules_follow_agnes_style_handoff():
    params = dict(policy.DEFAULT)
    fv = {"top_post": 0.97}
    base = dict(data_ok=True, severity="S2", verified=3, streak=0, exp={})
    assert brain.route(fv, 0.97, ["A"], params, **base) == []
    assert "missing_data" in brain.route(fv, 0.97, ["A"], params, **{**base, "data_ok": False})
    assert "critical" in brain.route(fv, 0.97, ["A"], params, **{**base, "severity": "S1"})
    assert "first_seen" in brain.route(fv, 0.97, ["A"], params, **{**base, "verified": 0})
    assert "audit" in brain.route(fv, 0.97, ["A"], params, **{**base, "streak": params["audit_every"] - 1})
    assert "low_reliability" in brain.route(fv, 0.5, ["A"], params, **base)
    assert "ambiguous" in brain.route(fv, 0.97, ["A", "B"], params, **base)
    retired = HarnessExperience(key="k", status="retired")
    assert "prior_error" in brain.route(fv, 0.97, ["A"], params, **{**base, "exp": {"fast": retired}})


def test_experience_beta_thresholds_blame_and_revoke():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as db:
        x = HarnessExperience(key="tip:fast:s:WM_EDGE:weak", track="tip")
        db.add(x)
        for i in range(4):
            experience._record(x, False, {"receipt": i}, blame="快脑答错")
        assert x.status == "retired" and len(x.blame) == 4  # P<0.45 且失败≥4 次：停用但留档
        y = HarnessExperience(key="tip:fast:s:WM_EDGE:strong", track="tip")
        for i in range(3):
            experience._record(y, True, {"receipt": 10 + i})
        assert y.status == "stable" and experience.p_success(y) >= 0.72


async def test_test_time_training_recovers_after_a_change_and_falls_back_otherwise(client):
    with session_scope() as db:
        mem = load_memory(db)
    stress = evolve.stressed_truth(["WM_EDGE"])
    twin = FabTwin(truth=stress)
    tasks = await evolve.make_split(mem, "report", twin, symptoms=["WM_EDGE"], per_symptom=8)
    shifted = [t for t in tasks if t["truth"] != t["hyps"][0]["cause_code"]][:3]
    assert shifted
    adopted = 0
    for t in shifted:
        r = await run_ttt(t["evidence"], t["symptom"], mem, dict(policy.DEFAULT), seed=t["seed"], twin=twin)
        adopted += r["adopt"] and r["top_after"] == t["truth"]
        assert r["probe_n"] == 12 and "ranking" in r
    assert adopted >= 1
    same = (await evolve.make_split(mem, "report", None, symptoms=["WM_CLUSTER"], per_symptom=2))[0]
    r = await run_ttt(same["evidence"], same["symptom"], mem, dict(policy.DEFAULT), seed=same["seed"])
    assert r["adopt"] is False or r["probe_after"] - r["probe_before"] >= 2  # 没有可靠提升就回退


async def test_analysis_writes_receipt_and_closure_updates_experience(client):
    from app.models import FabDataset
    from app.rsi.twin import TaskSpec

    case = client.post("/api/cases", json={"title": "Harness 测试：边缘环形失效", "symptom": "WM_EDGE", "severity": "S2",
                                           "site": "Harness 测试厂", "product": "工艺样品"}).json()
    data = FabTwin().make_task(TaskSpec("WM_EDGE", "easy"), seed=-6_200_001).observation
    with session_scope() as db:  # 自己建案并挂上孪生数据，不占用演示用的待分析案例
        db.add(FabDataset(case_id=case["id"], data=data))
        db.commit()
    res = client.post(f"/api/cases/{case['id']}/analyze", params={"stream": "false"})
    assert res.is_success, res.text
    detail = client.get(f"/api/cases/{case['id']}").json()
    hz = detail["analyses"][0]["counterfactual"]["harness"]
    assert hz["route"] in ("fast", "slow") and 0 <= hz["reliability"] <= 1 and hz["set"]
    assert "可靠度与快慢双脑" in detail["analyses"][0]["report_md"]
    with session_scope() as db:
        rc = db.exec(select(HarnessReceipt).where(HarnessReceipt.case_id == case["id"])).first()
        assert rc is not None and rc.verified_at is None and rc.signature.get("fv")
    cause = detail["candidates"][0]["code"]
    closed = client.post(f"/api/cases/{case['id']}/close", json={"cause_code": cause}).json()
    assert "top_match" in closed["harness"]
    with session_scope() as db:
        rc = db.get(HarnessReceipt, rc.id)
        assert rc.verified_at is not None and rc.outcome["cause_code"] == cause


def test_conformal_threshold_covers_calibration_set():
    rng = random.Random(3)
    samples = [{"truth_post": rng.random() ** 0.5} for _ in range(400)]
    q = train.calibrate_conformal(samples, alpha=0.1)
    assert train.coverage(samples, q) >= 0.9


async def test_reliability_window_has_health_and_promotion_gates(client, monkeypatch):
    monkeypatch.setattr(train, "POOL_PER_SYMPTOM", 18)
    monkeypatch.setattr(train, "HOLDOUT_PER_SYMPTOM", 6)
    monkeypatch.setattr(train, "CAL_PER_SYMPTOM", 6)
    with session_scope() as db:
        row = await train.train_window(db, load_memory(db), "test")
        assert row.status in ("active", "rejected", "failed")
        if row.status != "failed":
            m = row.metrics
            assert m["logloss_diff"]["ci95"][0] <= m["logloss_diff"]["mean"] <= m["logloss_diff"]["ci95"][1]
            assert (row.status == "active") == m["promoted"]
            assert set(row.coef) == set(brain.FEATURES) and 0 < row.conformal["q"] < 1


async def test_rrsi_run_archives_every_candidate_with_regularized_verdicts(client, monkeypatch):
    monkeypatch.setattr(evolve, "SPLITS", {"search": (6_100_000, 1), "select": (6_200_000, 2), "report": (6_300_000, 2)})
    with session_scope() as db:
        before = len(db.exec(select(HarnessPolicy)).all())
        res = await evolve.evolve_run(db, load_memory(db), generations=2, children=2, seed=3)
        rows = db.exec(select(HarnessPolicy).where(HarnessPolicy.run_id == res["run_id"])).all()
    assert len(rows) >= 1 and len(rows) + before > before
    for row in rows:
        assert row.status in ("accepted", "rejected", "active") and row.note
        assert all(e["component"] in policy.COMPONENTS for e in row.edits)
        gen = row.scores.get("generation")
        if gen is not None:
            assert len(row.edits) <= max(1, 3 - gen)  # 改动预算退火


def test_probation_rolls_back_when_fast_errors_rise():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as db:
        parent = HarnessPolicy(params=dict(policy.DEFAULT), status="retired")
        db.add(parent)
        db.flush()
        act = HarnessPolicy(params={**policy.DEFAULT, "fast_min_reliability": 0.7}, status="active", parent_id=parent.id)
        db.add(act)
        db.flush()
        case = Case(title="回放", symptom="WM_EDGE")
        db.add(case)
        db.flush()
        for i in range(10):
            db.add(HarnessReceipt(case_id=case.id, policy_version=act.id, route="fast", verified_at=case.created_at,
                                  outcome={"fast_match": i >= 4}))
        db.commit()
        out = evolve.probation_check(db)
        assert out["status"] == "rolled_back" and out["wrong"] == 4
        new_id, params = policy.active(db)
        assert new_id != act.id and params["fast_min_reliability"] == policy.DEFAULT["fast_min_reliability"]


@pytest.mark.parametrize("k", list(policy.COMPONENTS))
def test_policy_whitelist_and_bounds(k):
    _, step, lo, hi, _ = policy.COMPONENTS[k]
    assert policy.clamp({k: hi + 10 * step})[k] == hi and policy.clamp({k: lo - 10 * step})[k] == lo
    assert "evil" not in policy.clamp({"evil": 1})
