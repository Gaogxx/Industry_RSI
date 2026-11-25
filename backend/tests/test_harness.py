"""Harness 层：快慢双脑路由、慢脑探测题裁决、改进经验库、RRSI 进化 harness 组件与条件规则（事件触发）、观察期回滚。"""

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from app.db import session_scope
from app.fastbrain.model import FastModel
from app.harness import brain, evolve, experience, policy
from app.harness.slow import arbitrate, components
from app.models import Case, HarnessExperience, HarnessPolicy, HarnessReceipt
from app.rsi.memory import load_memory
from app.rsi.twin import FabTwin


def test_seeded_history_has_dual_brain_receipts_library_and_archive(client):
    o = client.get("/api/harness").json()
    r = o["receipts"]
    assert r["total"] >= 120 and r["verified"] >= 100 and 0 < r["fast"] < r["total"]
    assert r["with_model"] > 0  # 快脑模型晋升之后，收据记着用的是哪一版模型
    assert r["fast_verified"] and r["fast_correct"] / r["fast_verified"] >= 0.85  # 快脑只在把握足够时接手
    assert set(r["reasons"]) <= set(brain.REASON_TEXT) and r["chosen"]
    assert o["archive"] and o["benchmark"]
    b = o["benchmark"]["results"]
    assert b["model_only"]["hit1"] >= b["bayes_only"]["hit1"] - 2  # 训练出来的快脑不比贝叶斯差
    assert b["stress"]["dual"]["hit1"] >= b["stress"]["model_only"]["hit1"]  # 产线刚变更时慢脑能追回


def test_route_rules_follow_agnes_style_handoff():
    params = policy.clamp({})
    base = dict(data_ok=True, severity="S2", verified=3, streak=0, exp={})
    assert brain.route(0.97, ["A"], params, **base) == []
    assert "missing_data" in brain.route(0.97, ["A"], params, **{**base, "data_ok": False})
    assert "critical" in brain.route(0.97, ["A"], params, **{**base, "severity": "S1"})
    assert "first_seen" in brain.route(0.97, ["A"], params, **{**base, "verified": 0})
    assert "audit" in brain.route(0.97, ["A"], params, **{**base, "streak": params["router.audit_every"] - 1})
    assert "low_conf" in brain.route(0.5, ["A"], params, **base)
    assert "ambiguous" in brain.route(0.97, ["A", "B"], params, **base)
    assert "rule" in brain.route(0.97, ["A"], {**params, "router.force_slow": 1}, **base)
    retired = HarnessExperience(key="k", status="retired")
    assert "prior_error" in brain.route(0.97, ["A"], params, **{**base, "exp": {"fast": retired}})


def test_arbiter_needs_a_clear_probe_margin_and_prefers_fast_on_ties():
    assert arbitrate({"fast": 6, "bayes": 6, "ttt": 6}, 12, 0.1)["chosen"] == "fast"
    assert arbitrate({"fast": 6, "bayes": 7}, 12, 0.1)["chosen"] == "fast"  # 只多 1 道，需 ≥ 2 道
    out = arbitrate({"fast": 5, "bayes": 6, "ttt": 9}, 12, 0.1)
    assert out["chosen"] == "ttt" and "改判" in out["why"]
    assert arbitrate({"fast": 5, "bayes": 7}, 12, 0.1)["chosen"] == "bayes"  # 贝叶斯意见也能改判


async def test_slow_brain_components_recover_after_a_change(client):
    with session_scope() as db:
        mem = load_memory(db)
    twin = FabTwin(truth=evolve.stressed_truth(["WM_EDGE"]))
    tasks = await evolve.make_split(mem, "report", twin, symptoms=["WM_EDGE"], per_symptom=6)
    model = FastModel.bayes(mem.weights)
    prm = policy.clamp({})
    wins = 0
    for t in tasks[:4]:
        r = await components(t["evidence"], t["symptom"], mem, model, prm, t["seed"], twin, want_rankings=False)
        assert r["probe_n"] == 12 and set(r["hits"]) == {"fast", "bayes", "ttt"} and r["cost"] > 12
        wins += r["hits"]["ttt"] > r["hits"]["fast"]
    assert wins >= 1  # 记忆和模型都还停在变更前时，近邻训练在探测题上能赢过快脑


async def test_analysis_writes_receipt_and_closure_updates_library(client):
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
    assert hz["route"] in ("fast", "slow") and 0 <= hz["reliability"] <= 1 and hz["set"] and hz["fast_kind"] in ("model", "bayes")
    assert "快慢双脑与可靠度" in detail["analyses"][0]["report_md"]
    with session_scope() as db:
        rc = db.exec(select(HarnessReceipt).where(HarnessReceipt.case_id == case["id"])).first()
        assert rc is not None and rc.verified_at is None and "conf" in rc.signature
    cause = detail["candidates"][0]["code"]
    closed = client.post(f"/api/cases/{case['id']}/close", json={"cause_code": cause}).json()
    assert "top_match" in closed["harness"]
    with session_scope() as db:
        rc = db.get(HarnessReceipt, rc.id)
        assert rc.verified_at is not None and rc.outcome["cause_code"] == cause


def test_inner_reflection_only_records_observable_differences():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    cond = {"evidence": "weak", "after_pm": False, "factor": "none"}
    with Session(engine) as db:
        same = {"hits": {"fast": 6, "bayes": 6, "ttt": 6}, "tops": {"fast": "A", "ttt": "A"}, "probe_n": 12}
        assert experience.reflect_inner(db, "s", "WM_EDGE", cond, same) == []  # 没有行为差异：不新增条目
        for i in range(4):
            out = experience.reflect_inner(db, "s", "WM_EDGE", cond, {"hits": {"fast": 8, "ttt": 6}, "tops": {}, "probe_n": 12})
        x = out[0]
        assert x.status == "retired" and len(x.blame) == 4  # 赢不过快脑 4 次：停用但留档
        lib = experience.library(db)
        assert lib[0]["component"] == "ttt" and lib[0]["n"] == 4 and lib[0]["p"] < 0.45


def test_rule_whitelist_effective_overrides_and_leakage_critic():
    ok = {"if": {"symptom": "WM_EDGE", "evidence": "weak"}, "set": {"slow.ttt": 0}, "why": "x", "source": "library:ttt:WM_EDGE:weak"}
    assert policy.valid_rule(ok) == ""
    assert "不可观察" in policy.valid_rule({"if": {"truth": "ETCH_FOCUS_RING"}, "set": {"slow.ttt": 0}})
    assert "白名单" in policy.valid_rule({"if": {"symptom": "WM_EDGE"}, "set": {"router.fast_min_conf": 0.1}})
    p = policy.clamp({"rules": [ok, {"if": {"seed": 3}, "set": {"slow.ttt": 0}}]})
    assert len(p["rules"]) == 1  # 不合法的规则被丢掉
    eff, hit = policy.effective(p, "WM_EDGE", {"evidence": "weak", "after_pm": False, "factor": "none"})
    assert eff["slow.ttt"] == 0 and hit
    assert policy.effective(p, "WM_EDGE", {"evidence": "strong"})[0]["slow.ttt"] == 1
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as db:
        few = [{"symptom": "WM_EDGE", "cond": {"evidence": "weak"}}] * 2
        assert "支持题不足" in evolve.critic(db, ok, few, few)
        assert "来源" in evolve.critic(db, {**ok, "source": "hand"}, few * 2, few * 2)
        assert evolve.critic(db, ok, few * 2, few * 2) == ""


async def test_rrsi_run_archives_candidates_with_regularized_verdicts(client, monkeypatch):
    monkeypatch.setattr(evolve, "SPLITS", {"search": (6_100_000, 1), "select": (6_200_000, 1), "report": (6_300_000, 1)})
    with session_scope() as db:
        res = await evolve.evolve_run(db, load_memory(db), generations=2, children=2, seed=3, trigger="test")
        rows = db.exec(select(HarnessPolicy).where(HarnessPolicy.run_id == res["run_id"])).all()
    assert rows and all(r.status in ("accepted", "rejected", "active") and r.note for r in rows)
    for row in rows:
        for e in row.edits:
            assert e["component"] in policy.COMPONENTS or e["component"] == "rule"
        gen = row.scores.get("generation")
        if gen is not None:
            assert len(row.edits) <= max(1, 3 - gen)  # 改动预算退火
    assert all(r["verdict"] in ("通过", "拒绝") for r in res["reviews"])


def test_rrsi_is_event_triggered_with_backoff():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as db:
        assert evolve.pending_triggers(db) == ["bootstrap"]
        for run in ("r1", "r2"):
            db.add(HarnessPolicy(run_id=run, params=policy.clamp({}), status="rejected", note="x"))
        db.commit()
        assert evolve.pending_triggers(db) == []  # 没有新信号就不跑（不定时）


def test_probation_rolls_back_when_fast_errors_rise():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as db:
        parent = HarnessPolicy(params=policy.clamp({}), status="retired")
        db.add(parent)
        db.flush()
        act = HarnessPolicy(params={**policy.clamp({}), "router.audit_every": 8}, status="active", parent_id=parent.id)
        db.add(act)
        db.flush()
        case = Case(title="回放", symptom="WM_EDGE")
        db.add(case)
        db.flush()
        for i in range(12):
            db.add(HarnessReceipt(case_id=case.id, policy_version=act.id, route="fast", verified_at=case.created_at,
                                  outcome={"fast_match": i >= 3}))
        db.commit()
        out = evolve.probation_check(db)
        assert out["status"] == "ok" and out["wrong"] == 3  # 3/12：错误率下界没有显著高于基准，不回滚
        for i in range(8):
            db.add(HarnessReceipt(case_id=case.id, policy_version=act.id, route="fast", verified_at=case.created_at,
                                  outcome={"fast_match": i >= 6}))
        db.commit()
        out = evolve.probation_check(db)
        assert out["status"] == "rolled_back" and out["wrong"] == 9 and out["lower"] > out["baseline"]  # 统计检验：显著比承诺的差
        new_id, params = policy.active(db)
        assert new_id != act.id and params["router.audit_every"] == policy.DEFAULT["router.audit_every"]


@pytest.mark.parametrize("k", list(policy.COMPONENTS))
def test_policy_whitelist_and_bounds(k):
    _, step, lo, hi, _ = policy.COMPONENTS[k]
    assert policy.clamp({k: hi + 10 * step})[k] == hi and policy.clamp({k: lo - 10 * step})[k] == lo
    assert "evil" not in policy.clamp({"evil": 1})
