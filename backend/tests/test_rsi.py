"""RSI（递归自我改进）：数字孪生、三角色职责分离、可进化记忆、BRS/CAL/DRS、证据权重与验收门、冷启动消融、在线会话、AG-UI。"""

import asyncio
import inspect
import json
import random
import time

from app.fab.analytics import analyze_dataset
from app.fab.kg import prior_edges
from app.rsi.agents import Curriculum, Executor, Verifier, knowledge_gradient
from app.rsi.calibration import features, fit
from app.rsi.engine import CAL_TASKS, cold_start_benchmark, practice
from app.rsi.memory import MemoryState, consolidate_fail, consolidate_pass
from app.rsi.twin import EVAL_PER_SYMPTOM, SITE_TRUTH, FabTwin, TaskSpec, eval_set


def test_twin_hides_truth_and_gives_intervention_feedback():
    twin = FabTwin()
    task = twin.make_task(TaskSpec("WM_EDGE", "medium"), seed=123)
    assert task.hidden["cause"] in SITE_TRUTH["WM_EDGE"]
    assert "cause" not in json.dumps(task.observation, ensure_ascii=False)[:2000]  # 观测数据里没有答案
    right = twin.apply_fix(task, task.hidden["cause"])
    wrong = twin.apply_fix(task, next(c for c in SITE_TRUTH["WM_EDGE"] if c != task.hidden["cause"]))
    assert right["after"] > right["before"] and abs(wrong["after"] - wrong["before"]) < 0.2
    assert set(right) == {"metric", "before", "after", "baseline"}  # 反馈只有指标，没有答案


def test_verifier_is_isolated_and_conservative():
    params = list(inspect.signature(Verifier.judge).parameters)
    assert params == ["self", "requirements", "feedback", "capa_kinds"]  # 看不到执行 agent 的排序/理由/记忆
    req = {"higher_is_better": True, "min_recovery": 0.6, "capa_kinds": ["containment", "corrective", "preventive"]}
    kinds = {"containment", "corrective", "preventive"}
    assert Verifier().judge(req, {"before": 90.0, "after": 92.0, "baseline": 92.1}, kinds)["passed"]
    assert not Verifier().judge(req, {"before": 90.0, "after": 90.2, "baseline": 92.1}, kinds)["passed"]
    tiny = Verifier().judge(req, {"before": 92.0, "after": 92.05, "baseline": 92.05}, kinds)
    assert not tiny["passed"] and not tiny["conclusive"]  # 低于量测噪声 → 不予通过，避免固化未验证经验
    assert not Verifier().judge(req, {"before": 90.0, "after": 92.0, "baseline": 92.1}, {"corrective"})["passed"]


def test_practice_act_verify_loop_retries_next_hypothesis():
    twin, mem = FabTwin(), MemoryState.textbook()
    results = [practice(twin, mem, TaskSpec("WM_EDGE", "hard"), seed, Executor(), Verifier()) for seed in range(40, 60)]
    multi = [r for r in results if len(r["attempts"]) > 1]
    assert multi, "应当出现首个假设被环境否定、换下一个假设的情况"
    for r in multi:
        tried = [a["cause_code"] for a in r["attempts"]]
        assert len(tried) == len(set(tried)) and not r["attempts"][0]["verdict"]["passed"]


def test_memory_keeps_conditions_uncertainty_and_source_weights():
    ev = {"suspects": [], "change_point": None}
    real, sb = MemoryState.textbook(), MemoryState.textbook()
    consolidate_pass(real, "RACK_LEAK", "FITTING_TORQUE", [], ev, source="real")
    consolidate_pass(sb, "RACK_LEAK", "FITTING_TORQUE", [], ev, source="sandbox")
    a0 = MemoryState.textbook().edges["RACK_LEAK"]["FITTING_TORQUE"].alpha
    assert real.edges["RACK_LEAK"]["FITTING_TORQUE"].alpha - a0 == 1.0
    assert sb.edges["RACK_LEAK"]["FITTING_TORQUE"].alpha - a0 == 0.5  # 孪生经验让位于产线确认
    item = next(i for i in sb.items.values() if i["kind"] == "procedure")
    assert item["condition"] == {"evidence": "weak", "after_pm": False, "factor": "none"} and item["passes"] == 1
    for _ in range(3):
        consolidate_fail(sb, "RACK_LEAK", "FITTING_TORQUE", ev, source="sandbox")
    assert item["status"] == "retired" and item["fails"] == 3  # 被反复否定的记忆自动停用
    assert any(i["kind"] == "lesson" for i in sb.items.values())


def test_curriculum_breadth_covers_stages_and_depth_stops():
    mem = MemoryState.textbook()
    specs = asyncio.run(Curriculum().breadth(mem, [], 12))
    assert len({s.symptom for s in specs}) == 12 and all(s.variant == "前置技能" for s in specs)
    hist = [{"spec": {"symptom": "WM_EDGE", "difficulty": "hard"}, "verdict": "PASS", "phase": "DRS", "delta": 0.0, "new_items": 0}] * 3
    assert asyncio.run(Curriculum(patience=3).depth(mem, hist, 3)) is None  # 连续无新知识 → 停止
    fail = [{"spec": {"symptom": "PKG_VOID", "difficulty": "medium"}, "verdict": "FAIL", "phase": "DRS", "delta": 0.1, "new_items": 1}]
    nxt = asyncio.run(Curriculum().depth(mem, fail, 1))
    assert nxt.symptom == "PKG_VOID" and nxt.variant == "失败驱动"


def test_cold_start_ablation_v2():
    """如实的消融：论文方案（只学先验与记忆）提升在噪声范围内；学证据权重 + 校准批才带来大幅、稳定的提升。"""
    assert len(eval_set()) == EVAL_PER_SYMPTOM * len(SITE_TRUTH)
    b = asyncio.run(cold_start_benchmark())
    m = b["modes"]
    weak = ("none", "brs", "drs", "full")
    assert m["full_w"]["hit1_mean"] > m["full"]["hit1_mean"]  # 同样 48 道题，多学证据权重就更好
    assert m["v2"]["hit1_mean"] > m["none"]["hit1_mean"] + 12
    assert m["v2"]["hit1_min"] > max(m[k]["hit1_max"] for k in weak)  # v2 最差的种子也好过 v0.3 最好的种子
    assert m["v2"]["partial_mean"] > m["none"]["partial_mean"] + 10
    assert b["oracle"]["hit1"] - m["v2"]["hit1_mean"] < 3  # 离参考天花板不到 3 个百分点
    assert m["v2"]["weights"]["共性分析"] > 1.5  # 学到：共性分析证据比专家设定更可信


def _labelled(n: int, seed0: int):
    twin, rng, out = FabTwin(), random.Random(seed0), []
    for i in range(n):
        sym = rng.choice(list(SITE_TRUTH))
        t = twin.make_task(TaskSpec(sym, rng.choice(["easy", "medium", "hard"])), seed=seed0 + i)
        ev = analyze_dataset(t.observation)
        out.append((ev, sym, t.hidden["cause"]))
    return out


def test_evidence_weights_learn_and_gate_rejects_noisy_labels():
    data = _labelled(160, 4_000_000)
    edges = prior_edges()
    res = fit([(features(ev, sym, edges, {}), c) for ev, sym, c in data])
    assert res["n"] >= 100 and res["weights"]["共性分析"] > 1.5  # 从已验证样本中学到“共性分析更可信”

    mem = MemoryState.textbook()
    mem.examples = data
    good = mem.refit()
    assert good["accepted"] and mem.weights == good["weights"]

    noisy, rng = MemoryState.textbook(), random.Random(1)
    noisy.weights = dict(good["weights"])
    noisy.examples = [(ev, sym, rng.choice(list(SITE_TRUTH[sym]))) for ev, sym, _ in data]  # 标签被随机打乱
    bad = noisy.refit()
    assert not bad["accepted"] and "验收门" in bad["reason"] and noisy.weights == good["weights"]  # 退步的更新被拦下


def test_knowledge_gradient_sees_beyond_one_sample():
    mem = MemoryState.textbook()
    table = mem.edges["WM_EDGE"]
    kg0 = knowledge_gradient(table)
    assert kg0 > 0  # KG(*)：一个样本翻不动最优选择时，看多个样本的平均增益
    for e in table.values():
        e.alpha += 40 * SITE_TRUTH["WM_EDGE"].get(e.cause_code, 0)
    assert knowledge_gradient(table) < kg0  # 学得越多，再练的价值越低


def test_startup_session_and_benchmark_exist(client):
    sessions = client.get("/api/rsi/sessions").json()
    first = sessions[-1]
    assert first["status"] == "done" and first["summary"]["brs"] == 12 and first["summary"]["tasks"] >= 13
    d = client.get(f"/api/rsi/sessions/{first['id']}").json()
    assert d["session"]["curve"][0]["phase"] == "w/o RSI" and len(d["tasks"]) == first["summary"]["tasks"]
    t = d["tasks"][0]
    assert t["spec"]["rationale"] and t["attempts"][0]["verdict"]["evidence"] and t["hidden"]["cause"]
    assert client.get("/api/rsi/benchmark").json()["results"]["modes"]["full"]
    mem = client.get("/api/memory").json()
    assert any(m["source"] in ("sandbox", "both") for m in mem)
    events = client.get("/api/learning/events?limit=3000").json()
    assert any(e["source"] == "sandbox" and e["rsi_task_id"] for e in events)


def test_live_session_via_api(client):
    r = client.post("/api/rsi/sessions").json()
    for _ in range(900):
        s = client.get(f"/api/rsi/sessions/{r['session_id']}").json()["session"]
        if s["status"] != "running":
            break
        time.sleep(0.1)
    assert s["status"] == "done" and s["workflow_id"] == r["workflow_id"]
    assert s["summary"]["cal"] == CAL_TASKS and s["summary"]["refits_accepted"] + s["summary"]["refits_rejected"] >= 3
    assert s["weights_log"] and all("gate" in w for w in s["weights_log"] if w["n"] >= 20)
    steps = [x["name"] for x in client.get(f"/api/workflows/{r['workflow_id']}/steps").json()]
    assert steps[:2] == ["s_eval", "s_breadth_plan"] and "s_wave" in steps and steps[-2:] == ["s_finish", "DBOS.closeStream"] or "s_finish" in steps
    assert client.get("/api/rsi/gaps").json()[0]["gap"] >= 0


def _agui(client, body):
    r = client.post("/agui", json=body, headers={"Accept": "text/event-stream"})
    return [json.loads(line[5:]) for line in r.text.splitlines() if line.startswith("data:")]


def test_agui_chat_events(client):
    evs = _agui(client, {"threadId": "t", "runId": "r", "messages": [{"id": "1", "role": "user", "content": "分析 KZ-0137"}],
                         "forwardedProps": {"mode": "rca"}})
    types = [e["type"] for e in evs]
    assert types[0] == "RUN_STARTED" and types[-1] == "RUN_FINISHED"
    assert "REASONING_MESSAGE_CONTENT" in types and "TEXT_MESSAGE_CONTENT" in types
    assert types.index("REASONING_END") < types.index("TEXT_MESSAGE_START")
    text = "".join(e.get("delta", "") for e in evs if e["type"] == "TEXT_MESSAGE_CONTENT")
    assert "8D 分析报告" in text


def test_agui_rsi_stream_uses_subagents_and_state_deltas(client):
    evs = _agui(client, {"threadId": "rsi", "runId": "r2", "messages": [], "forwardedProps": {"mode": "rsi"}})
    types = {e["type"] for e in evs}
    assert {"RUN_STARTED", "STATE_SNAPSHOT", "STEP_STARTED", "SUBAGENT_STARTED", "SUBAGENT_FINISHED", "STATE_DELTA",
            "RUN_FINISHED"} <= types
    names = {e["name"] for e in evs if e["type"] == "SUBAGENT_STARTED"}
    assert names == {"课程 agent", "执行 agent", "验证 agent"}


def test_agui_analysis_opens_case_panel(client):
    evs = _agui(client, {"threadId": "t3", "runId": "r3", "messages": [{"id": "1", "role": "user", "content": "分析 KZ-0137"}],
                         "forwardedProps": {"mode": "rca"}})
    acts = [e for e in evs if e["type"] == "ACTIVITY_SNAPSHOT"]
    assert acts and acts[0]["activityType"] == "kaizen.case" and acts[0]["content"]["open"]
    assert acts[0]["content"]["route"].startswith("/cases/") and acts[0]["content"]["top"]
    assert any(e["type"] == "STATE_SNAPSHOT" and e["snapshot"]["workspace"]["focusCaseId"] for e in evs)
    assert evs[-1]["type"] == "RUN_FINISHED" and evs[-1]["outcome"]["type"] == "success"


def test_agui_close_case_needs_approval_interrupt(client):
    """关案会改写产线记忆：先以 outcome=interrupt 结束运行，带 resume 批准后才执行；取消则什么都不改。"""
    evs = _agui(client, {"threadId": "t4", "runId": "r4", "forwardedProps": {"mode": "rca"},
                         "messages": [{"id": "1", "role": "user", "content": "Fab A 刻蚀层 CP 良率下降，失效集中在晶圆外圈，良率=88%"}]})
    route = next(e["content"]["route"] for e in evs if e["type"] == "ACTIVITY_SNAPSHOT")
    cid = int(route.rsplit("/", 1)[-1])
    ask = {"threadId": "t4", "runId": "r5", "forwardedProps": {"mode": "rca"},
           "messages": [{"id": "2", "role": "user", "content": f"确认 KZ-{cid:04d} 根因是刻蚀腔聚焦环磨损"}]}
    fin = _agui(client, ask)[-1]
    assert fin["type"] == "RUN_FINISHED" and fin["outcome"]["type"] == "interrupt"
    it = fin["outcome"]["interrupts"][0]
    assert it["id"].startswith(f"int-close-{cid}-") and it["responseSchema"]["required"] == ["approved"]
    assert client.get(f"/api/cases/{cid}").json()["case"]["status"] != "closed"  # 批准前没有任何写入

    cancel = _agui(client, {**ask, "runId": "r6", "resume": [{"interruptId": it["id"], "status": "cancelled"}]})
    assert "已取消" in "".join(e.get("delta", "") for e in cancel if e["type"] == "TEXT_MESSAGE_CONTENT")
    assert client.get(f"/api/cases/{cid}").json()["case"]["status"] != "closed"

    fin = _agui(client, {**ask, "runId": "r7"})[-1]
    ok = _agui(client, {**ask, "runId": "r8", "resume": [{"interruptId": fin["outcome"]["interrupts"][0]["id"], "status": "resolved",
                                                          "payload": {"approved": True, "reason": "QE 现场确认"}}]})
    text = "".join(e.get("delta", "") for e in ok if e["type"] == "TEXT_MESSAGE_CONTENT")
    assert "已关闭" in text and "证据权重" in text
    assert any(e["type"] == "ACTIVITY_SNAPSHOT" and e["activityType"] == "kaizen.learning" for e in ok)
    assert client.get(f"/api/cases/{cid}").json()["case"]["status"] == "closed"


def test_agui_rsi_stream_reports_calibration_and_gate(client):
    evs = _agui(client, {"threadId": "rsi2", "runId": "r9", "messages": [], "forwardedProps": {"mode": "rsi"}})
    names = {e["name"] for e in evs if e["type"] == "CUSTOM"}
    assert {"rsi.cal", "rsi.weights"} <= names
    steps = [e["stepName"] for e in evs if e["type"] == "STEP_STARTED"]
    assert steps == ["广度递归自探索 BRS", "证据校准 CAL", "深度递归自探索 DRS"]
    assert any(e["type"] == "ACTIVITY_DELTA" and e["activityType"] == "kaizen.rsi" for e in evs)
    w = client.get("/api/learning/weights").json()
    prod = w["by_source"].get("production", 0) + w["by_source"].get("nightly", 0)
    assert w["current"] and prod >= 1 and w["by_source"].get("rsi", 0) >= 1  # 两条来源写同一套权重
