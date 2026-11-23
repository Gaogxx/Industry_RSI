"""统计内核与贝叶斯学习的单元测试（纯函数，不依赖服务）。"""

import random

from app.agent.steps import rubric
from app.fab import kg
from app.fab.analytics import analyze_dataset, change_point, commonality, fisher_greater, wafer_signature
from app.fab.flow import CAUSES, SYMPTOMS, Scenario
from app.fab.simulate import simulate, wafer_map
from app.retrieval import BM25, rrf, tokenize
from app.services import case_from_text, extract_metrics


def test_fisher_exact():
    assert fisher_greater(9, 1, 1, 9) < 0.01  # 强富集
    assert fisher_greater(5, 5, 5, 5) > 0.3  # 无差异


def test_commonality_finds_injected_tool_and_pm():
    ds = simulate(Scenario("WM_CLUSTER", "PM_SEASONING", strength=1.0, seed=3))
    top = commonality(ds["lots"])[0]
    assert top["step"] == "ETCH" and top["p_adj"] < 0.01 and top["lift"] > 3
    cp = change_point(ds["lots"], top["factor"], top["value"], ds["events"])
    assert cp and cp["after_pm"] and cp["rate_after"] > cp["rate_before"]


def test_material_cause_shows_up_as_batch_commonality():
    ev = analyze_dataset(simulate(Scenario("PKG_WARP", "SUBSTRATE_CTE", strength=1.0, seed=5)))
    assert "#物料批次" in ev["suspects"][0]["factor"]


def test_wafer_signatures():
    for pattern in ("edge", "center", "scratch", "cluster"):
        hits = sum(wafer_signature(wafer_map(pattern, random.Random(s), 0.9))["pattern"] == pattern for s in range(20))
        assert hits >= 16, pattern
    assert wafer_signature(wafer_map(None, random.Random(1)))["pattern"] == "good"


def test_bayesian_rank_uses_evidence_and_prior():
    edges = kg.prior_edges()
    neutral = kg.rank({}, "WM_EDGE", edges)
    assert neutral[0].cause_code == "CMP_RETAINING_RING"  # 无证据时 = 教科书先验
    ev = analyze_dataset(simulate(Scenario("WM_EDGE", "ETCH_FOCUS_RING", strength=1.0, seed=8)))
    assert kg.rank(ev, "WM_EDGE", edges)[0].cause_code == "ETCH_FOCUS_RING"  # 强证据压过先验
    assert abs(sum(h.posterior for h in neutral) - 1) < 1e-9


def test_learning_updates_dirichlet_shares():
    edges = kg.prior_edges()
    events = kg.learn(edges, "WM_EDGE", "ETCH_FOCUS_RING", ["CMP_RETAINING_RING", "ETCH_FOCUS_RING"])
    confirm = next(e for e in events if e["kind"] == "confirm")
    lead = next(e for e in events if e["kind"] == "false_lead")
    assert confirm["after"] > confirm["before"] and lead["after"] < lead["before"]
    assert kg.rank({}, "WM_EDGE", edges)[0].cause_code == "ETCH_FOCUS_RING"  # 一次确认即翻转（教科书差距很小）
    ref = kg.refute(edges, "WM_EDGE", "ETCH_FOCUS_RING")
    assert ref["after"] < ref["before"]
    new = kg.learn(edges, "WM_EDGE", "NEW_CAUSE_X", [], confirmed_name="新根因")
    assert new[0]["kind"] == "edge_added" and edges["WM_EDGE"]["NEW_CAUSE_X"].learned


def test_prequential_learning_beats_frozen_when_site_differs_from_textbook():
    rng = random.Random(0)
    stream = []
    for k in range(40):  # 证据模糊、本厂真因与教科书首位不同
        cause = rng.choices(["ETCH_FOCUS_RING", "CMP_RETAINING_RING"], weights=[0.8, 0.2])[0]
        stream.append({"code": k, "time": str(k), "symptom": "WM_EDGE", "cause_code": cause, "evidence": {}})
    s = kg.prequential(stream)["summary"]
    assert s["hit1_learning"] > s["hit1_frozen"] + 0.3
    assert s["hit1_learning_recent"] >= 0.7


def test_every_cause_is_well_formed():
    for sym, causes in CAUSES.items():
        assert sym in SYMPTOMS and abs(sum(c.prior for c in causes) - 1) < 1e-6
        for c in causes:
            assert all(c.capa) and c.metric


def test_text_parsing():
    c = case_from_text("Fab A 刻蚀层 CP 良率下降，失效集中在晶圆外圈，良率=86%，机台 ETC-02B 刚做完 PM")
    assert c["site"] == "Fab A" and c["equipment"] == "ETC-02B" and c["symptom"] == "WM_EDGE"
    assert extract_metrics("翘曲(µm)=118，开路DPPM 为 3200")["翘曲(µm)"] == "118"


def test_retrieval_primitives():
    assert "晶圆" in tokenize("晶圆边缘失效 ETC-02B")
    docs = [tokenize(t) for t in ["HBM 堆叠 开路", "晶圆 边缘 环形 失效", "液冷 泄漏"]]
    sc = BM25(docs).scores(tokenize("边缘失效"))
    assert max(range(3), key=lambda i: sc[i]) == 1
    assert rrf([[0, 1], [1, 0]])[0] == rrf([[0, 1], [1, 0]])[1]


def test_rubric_flags_ambiguity_and_missing_capa():
    state = {"evidence": {"suspects": []},
             "hypotheses": [{"posterior": 0.4}, {"posterior": 0.35}],
             "five_whys": ["1", "2", "3", "4", "5"], "capa": [{"kind": "corrective", "verify_metric": "m"}], "playbook": []}
    score, issues = rubric(state)
    assert score < 75 and any("鉴别验证" in i for i in issues) and any("围堵" in i for i in issues)
