"""集成面：AG-UI 对话技能（分析 / 建案 / 追问 / 教练）、MCP 工具与 MCP Apps、鉴权。"""

import json

from conftest import open_case


def _chat(client, messages, mode="rca") -> dict:
    """走 AG-UI 端点，把事件流还原成 {reasoning, content, activities, outcome}。"""
    msgs = [{"id": str(i), **m} for i, m in enumerate(messages)]
    r = client.post("/agui", json={"threadId": "t", "runId": "r", "messages": msgs, "forwardedProps": {"mode": mode}},
                    headers={"Accept": "text/event-stream"})
    evs = [json.loads(line[5:]) for line in r.text.splitlines() if line.startswith("data:")]
    return {"reasoning": "".join(e.get("delta", "") for e in evs if e["type"] == "REASONING_MESSAGE_CONTENT"),
            "content": "".join(e.get("delta", "") for e in evs if e["type"] == "TEXT_MESSAGE_CONTENT"),
            "activities": [e for e in evs if e["type"] == "ACTIVITY_SNAPSHOT"],
            "outcome": evs[-1].get("outcome", {}) if evs else {}}


def test_chat_analyze_existing_case_streams_durable_steps(client):
    case = open_case(client, "TEST_VMIN")
    out = _chat(client, [{"role": "user", "content": f"分析 {case['code']}"}])
    assert "DBOS 持久化工作流" in out["reasoning"] and "共性分析" in out["reasoning"] and "持续学习的影响" in out["reasoning"]
    assert "8D 分析报告" in out["content"] and "贝叶斯根因排序" in out["content"] and f"确认 {case['code']} 根因是" in out["content"]
    assert f"(#/cases/{case['id']})" in out["content"]  # 链接在控制台内打开案例页面，不再指向外部看板


def test_chat_confirm_unknown_cause_is_rejected_without_interrupt(client):
    bad = _chat(client, [{"role": "user", "content": "确认 KZ-0137 根因是月亮太圆"}])
    assert "没能识别根因" in bad["content"] and bad["outcome"]["type"] == "success"


def test_chat_new_case_without_data_and_followup(client):
    before = client.get("/api/kpis").json()["kpis"]["cases_total"]
    out = _chat(client, [{"role": "user", "content": "封测厂 B 2.5D 封装翘曲超标，C4 凸点开路，翘曲=125µm，开路DPPM=2800"}])
    assert "8D 分析报告" in out["content"] and "未关联 MES/YMS/FDC 数据" in out["content"]
    assert client.get("/api/kpis").json()["kpis"]["cases_total"] == before + 1
    newest = client.get("/api/cases").json()[0]
    assert newest["source"] == "chat"
    follow = _chat(client, [{"role": "user", "content": "…"}, {"role": "assistant", "content": f"# 8D 分析报告 · {newest['code']}"},
                            {"role": "user", "content": "如果不学习，结论会不一样吗？"}])
    assert "持续学习 vs 冻结先验" in follow["content"]
    assert client.get("/api/kpis").json()["kpis"]["cases_total"] == before + 1


def test_coach(client):
    out = _chat(client, [{"role": "user", "content": "运行改进周期"}], mode="coach")
    assert "持续改进周期复盘" in out["content"] and out["reasoning"]
    plain = _chat(client, [{"role": "user", "content": "学习效果怎么样？"}], mode="coach")
    assert "学习曲线" in plain["content"] and "冻结先验" in plain["content"]
    assert plain["activities"][0]["content"]["route"] == "/evolution"


def test_open_webui_compat_removed(client):
    assert client.get("/v1/models").status_code in (404, 405)
    assert "openwebui_url" not in client.get("/api/meta").json()


async def test_mcp_tools_and_app(client):
    from app.mcp_server import BOARD_URI, mcp

    tools = {t.name: t for t in await mcp.list_tools()}
    assert {"search_cases", "get_case", "analyze_case", "confirm_root_cause", "get_learning_status",
            "show_kaizen_board", "run_improvement_cycle"} <= set(tools)
    meta = tools["show_kaizen_board"].meta
    assert meta["ui"]["resourceUri"] == BOARD_URI
    assert "ui/initialize" in str(await mcp.read_resource(BOARD_URI))


def test_mcp_board_and_learning_status_payloads():
    from app.mcp_server import get_learning_status, show_kaizen_board

    board = show_kaizen_board()
    assert board["learning"]["curve"] and board["learning"]["summary"]["cases"] >= 120
    status = get_learning_status()
    assert status["backtest"] and status["edges"]


def test_mcp_http_initialize(client):
    r = client.post("/mcp", headers={"Accept": "application/json, text/event-stream", "Content-Type": "application/json"},
                    json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                        "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "pytest", "version": "1"}}})
    assert r.status_code == 200 and "kaizen" in r.text


def test_auth(client, settings):
    settings.api_key = "test-key"
    body = {"threadId": "t", "runId": "r", "messages": [], "forwardedProps": {"mode": "rca"}}
    assert client.post("/agui", json=body).status_code == 401
    assert client.post("/api/cases", json={"title": "xx案例"}).status_code == 401
    assert client.post("/mcp", json={}).status_code == 401
    assert client.get("/api/kpis").status_code == 200
    settings.dashboard_public = False
    assert client.get("/api/learning/curve").status_code == 401
