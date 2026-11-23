"""端到端：学习历史 → DBOS 工作流分析（学习改变结论）→ 关案学习 → 负向学习 → 新知识 → 改进周期。"""

import json

from conftest import open_case


def test_seeded_history_has_measurable_learning(client):
    k = client.get("/api/kpis").json()
    assert k["kpis"]["cases_total"] == 138 and k["kpis"]["cases_closed"] == 120
    assert {"分析中", "措施执行", "效果验证", "待分析"} <= {s for s, n in k["kpis"]["by_status"].items() if n}
    bt = k["learning"]["backtest"]
    assert bt["cases"] == 120 and bt["hit1_learning"] > bt["hit1_frozen"]
    curve = client.get("/api/learning/curve").json()["latest"]
    assert len(curve["points"]) == 120 and curve["points"][-1]["cum_learning"] == bt["hit1_learning"]
    edges = client.get("/api/learning/edges").json()
    fr = next(e for e in edges if e["cause_code"] == "ETCH_FOCUS_RING")
    assert fr["posterior"] > fr["prior"] + 0.2 and fr["confirmed"] >= 10
    mem = client.get("/api/memory").json()
    assert any(m["source"] in ("real", "both") for m in mem) and all("condition" in m for m in mem)
    assert {e["kind"] for e in client.get("/api/learning/events?limit=3000").json()} >= {"confirm", "false_lead", "memory_add", "drift"}


def test_demo_history_spans_last_year(client):
    """演示历史按时间回放：改进周期、RSI 会话、审计记录都从去年开始，而不是全部写在今天。"""
    cycles = client.get("/api/improvement/cycles").json()
    sessions = client.get("/api/rsi/sessions").json()
    audit = client.get("/api/audit", params={"limit": 3000}).json()
    assert len(cycles) >= 15 and min(c["created_at"] for c in cycles)[:4] == "2025"
    assert len(sessions) >= 12 and min(s["created_at"] for s in sessions)[:4] == "2025"
    assert min(a["created_at"] for a in audit)[:4] == "2025" and {"case.close", "twin.update", "kaizen.session"} <= {a["action"] for a in audit}
    history = [s for s in sessions if s["trigger"] == "schedule"]
    assert all(s["config"]["changes"] for s in history[:-1])  # 除第一次外，每次会话前都有产线变更同步到孪生
    gains = [s["summary"]["hit1_after"] - s["summary"]["hit1_before"] for s in history]
    assert sum(g > 0 for g in gains) >= len(gains) - 1 and min(gains) >= 0


def test_durable_workflow_analysis_where_learning_changes_the_answer(client):
    case = open_case(client, "WM_EDGE")
    with client.stream("POST", f"/api/cases/{case['id']}/analyze") as r:
        events = [json.loads(line[5:]) for line in r.iter_lines() if line.startswith("data:")]
    assert events[0]["type"] == "workflow" and events[-1]["type"] == "done"
    labels = [e.get("label") for e in events if e["type"] == "step"]
    assert "晶圆图特征" in labels and "贝叶斯根因排序" in labels and "持续学习的影响" in labels
    done = events[-1]
    assert done["top"] == "ETCH_FOCUS_RING" and done["learning_changed_top"] is True
    assert "若只用教科书先验" in done["report_md"]
    steps = [s["name"] for s in client.get(f"/api/workflows/{events[0]['workflow_id']}/steps").json()]
    assert steps[:3] == ["load_case", "s_intake", "s_evidence"] and "s_persist" in steps

    detail = client.get(f"/api/cases/{case['id']}").json()
    an = detail["analyses"][0]
    assert an["counterfactual"]["changed"] and an["counterfactual"]["top"] == "CMP_RETAINING_RING"
    assert detail["dataset"]["wafers"] and detail["case"]["status"] == "capa"
    assert {a["kind"] for a in detail["actions"]} >= {"containment", "corrective", "preventive"}
    assert all(a["cause_code"] == "ETCH_FOCUS_RING" for a in detail["actions"])


def test_close_triggers_learning(client):
    case = next(c for c in client.get("/api/cases", params={"status": "capa"}).json() if c["symptom"] == "WM_EDGE")
    before = next(e for e in client.get("/api/learning/edges").json() if e["cause_code"] == "ETCH_FOCUS_RING")
    r = client.post(f"/api/cases/{case['id']}/close", json={"cause_code": "ETCH_FOCUS_RING"}).json()
    assert r["rank_in_analysis"] == 1 and r["learning_summary"]["cases"] == 121
    confirm = next(e for e in r["learning_events"] if e["kind"] == "confirm")
    assert confirm["after"] > confirm["before"]
    after = next(e for e in client.get("/api/learning/edges").json() if e["cause_code"] == "ETCH_FOCUS_RING")
    assert after["confirmed"] == before["confirmed"] + 1
    assert client.get(f"/api/cases/{case['id']}").json()["learning_events"]
    assert client.post(f"/api/cases/{case['id']}/close", json={"cause_code": "ETCH_FOCUS_RING"}).status_code == 422


def test_ineffective_capa_is_negative_learning(client):
    case = open_case(client, "OVL_DRIFT")
    done = client.post(f"/api/cases/{case['id']}/analyze", params={"stream": "false"}).json()
    # 产线上从未确认过“新导入扫描机匹配误差”，但 RSI 已在数字孪生中练习过 → 首位就是它
    assert done["top"] == "SCANNER_MATCHING"
    acts = client.get(f"/api/cases/{case['id']}").json()["actions"]
    corrective = next(a for a in acts if a["kind"] == "corrective")
    assert corrective["cause_code"] == done["top"]
    client.patch(f"/api/actions/{corrective['id']}", json={"status": "ineffective", "result": "套刻未改善"})
    events = client.get(f"/api/cases/{case['id']}").json()["learning_events"]
    refute = next(e for e in events if e["kind"] == "refute")
    assert refute["cause_code"] == done["top"] and refute["after"] < refute["before"] and refute["source"] == "real"
    assert any(e["kind"] == "memory_fail" for e in events)
    # 工程师最终确认是镜头热效应：排在它前面的假设被记为“误导”
    r = client.post(f"/api/cases/{case['id']}/close", json={"cause_code": "LENS_HEATING"}).json()
    kinds = [e["kind"] for e in r["learning_events"]]
    assert "confirm" in kinds and "false_lead" in kinds


def test_new_root_cause_grows_the_graph(client):
    case = open_case(client, "HBM_OPEN")
    client.post(f"/api/cases/{case['id']}/analyze", params={"stream": "false"})
    assert client.post(f"/api/cases/{case['id']}/close", json={"cause_code": "HBM_UNDERFILL_FLOW"}).status_code == 422
    r = client.post(f"/api/cases/{case['id']}/close", json={"cause_code": "HBM_UNDERFILL_FLOW",
                                                           "root_cause": "MR-MUF 模塑底填流动性批次异常"}).json()
    assert r["learning_events"][0]["kind"] == "edge_added"
    edge = next(e for e in client.get("/api/learning/edges").json() if e["cause_code"] == "HBM_UNDERFILL_FLOW")
    assert edge["learned"] and edge["confirmed"] == 1


def test_improvement_cycle_reports_learning(client):
    cycle = client.post("/api/improvement/run").json()
    assert any("持续学习" in f for f in cycle["findings"]) and cycle["learning"]["backtest"]["cases"] >= 123
    audit = {a["action"] for a in client.get("/api/audit?limit=300").json()}
    assert {"agent.analyze", "case.close", "action.update", "improvement.run"} <= audit


def test_feedback_search_memory_toggle(client):
    an_id = client.get(f"/api/cases/{open_case(client, 'PKG_WARP')['id']}").json()["analyses"]
    case = open_case(client, "PKG_WARP")
    done = client.post(f"/api/cases/{case['id']}/analyze", params={"stream": "false"}).json()
    assert client.post("/api/feedback", json={"analysis_id": done["analysis_id"], "rating": 5}).status_code == 201
    assert client.post("/api/feedback", json={"analysis_id": done["analysis_id"], "rating": 9}).status_code == 422
    assert client.get("/api/search", params={"q": "翘曲 凸点 开路"}).json()[0]["symptom"] == "PKG_WARP"
    b = client.get("/api/memory").json()[0]
    assert client.post(f"/api/memory/{b['id']}/status", json={"active": False}).json()["status"] == "retired"
    assert an_id == []
