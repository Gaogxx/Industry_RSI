"""大模型路径：用 Pydantic AI 的 FunctionModel 脚本化模型，覆盖类型化输出、评审回环、代码模式调查员与降级。"""

import json

from pydantic_ai import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.llm import LLM
from conftest import open_case


class Script:
    def __init__(self):
        self.critic_calls = 0
        self.code_runs: list[str] = []
        self.explain_prompts: list[str] = []

    def __call__(self, messages, info: AgentInfo) -> ModelResponse:
        out = info.output_tools[0]
        props = out.parameters_json_schema.get("properties", {})
        tools = [t.name for t in info.function_tools]
        if "run_code" in tools and not self.code_runs:  # 调查员：先写代码查数据
            code = "h = hypotheses()\nr = fail_rate_by(factor='ETCH')\n{'top': h[0]['cause_code'], 'worst': r[0]}"
            self.code_runs.append(code)
            return ModelResponse(parts=[ToolCallPart("run_code", {"code": code})])
        if "findings" in props:
            returned = str(messages[-1].parts[-1].content) if hasattr(messages[-1].parts[-1], "content") else ""
            self.code_runs.append(returned)
            args = {"findings": ["ETCH 工序 ETC-02B 失效率最高（沙箱核查）"], "supports": ["PM_SEASONING"],
                    "contradicts": ["RETICLE_HAZE"], "next_checks": ["跑颗粒监控片"]}
        elif "problem_statement" in props:
            args = {"problem_statement": "刻蚀层 D0 异常升高", "what": "簇状颗粒", "where": "Fab A 刻蚀区", "when": "近 9 天",
                    "who": "良率工程", "how_many": "D0 0.21", "how_detected": "KLA 扫描", "impact": "良率损失",
                    "symptom": "WM_CLUSTER", "severity": "S2"}
        elif "five_whys" in props:
            self.explain_prompts.append(str(messages[0].parts[-1].content))
            args = {"five_whys": ["D0 升高", "颗粒来自腔体", "PM 后开始", "seasoning 片数不足", "复机标准缺颗粒监控片判据"],
                    "ishikawa_extra": {"method": ["复机 SOP 缺项"]}, "narrative": "LLM 结论：PM 后 seasoning 不足。", "risks": ["光罩雾化"]}
        elif "actions" in props:
            args = {"actions": [
                {"kind": "containment", "title": "暂停 ETC-02B 投片", "owner_role": "设备", "due_days": 1, "verify_metric": "hold 率"},
                {"kind": "corrective", "title": "补做 seasoning 并跑颗粒监控片", "owner_role": "刻蚀工程师", "due_days": 3,
                 "verify_metric": "D0", "target": "≤0.1"},
                {"kind": "preventive", "title": "复机 SOP 增加颗粒监控片判据", "owner_role": "模块经理", "due_days": 30,
                 "verify_metric": "复发次数"}]}
        elif "score" in props:
            self.critic_calls += 1
            args = {"score": 20 if self.critic_calls == 1 else 96, "issues": ["5-Why 未引用 FDC 数据"] if self.critic_calls == 1 else []}
        else:
            args = {}
        return ModelResponse(parts=[ToolCallPart(out.name, args)])


async def test_llm_path_typed_outputs_reflection_and_code_mode(client):
    from app.services import analyze_case

    script = Script()
    llm = LLM(model=FunctionModel(script), name="scripted-llm")
    case = open_case(client, "WM_CLUSTER")
    events = [ev async for ev in analyze_case(case["id"], llm=llm)]
    done = events[-1]
    assert done["type"] == "done", events[-3:]
    labels = [e.get("label") for e in events if e["type"] == "step"]
    assert "代码模式调查" in labels and "调查结论" in labels
    assert script.code_runs and "run_code" not in script.code_runs[0]
    assert "ETC" in script.code_runs[1]  # 沙箱真的执行了代码并把结果交回模型
    assert script.critic_calls == 2 and "5-Why 未引用 FDC 数据" in script.explain_prompts[1]

    an = client.get(f"/api/cases/{case['id']}").json()["analyses"][0]
    assert an["model"] == "scripted-llm" and an["iterations"] == 2
    assert an["five_whys"][-1] == "复机标准缺颗粒监控片判据" and "LLM 结论" in an["report_md"]
    pm = next(h for h in an["hypotheses"] if h["cause_code"] == "PM_SEASONING")
    assert any(f["name"] == "调查 Agent" and f["x"] == 1.3 for f in pm["factors"])
    assert an["counterfactual"]["investigation"]["supports"] == ["PM_SEASONING"]
    usage = an["trace"][-1]["usage"]
    assert usage["llm_calls"] >= 6 and usage["prompt_tokens"] > 0
    acts = client.get(f"/api/cases/{case['id']}").json()["actions"]
    assert any(a["title"] == "补做 seasoning 并跑颗粒监控片" for a in acts)


async def test_llm_failure_degrades_to_statistical_core(client):
    from app.services import analyze_case

    def broken(messages, info):
        raise RuntimeError("upstream 503")

    case = next(c for c in client.get("/api/cases").json() if c["source"] == "chat")
    done = [ev async for ev in analyze_case(case["id"], llm=LLM(model=FunctionModel(broken), name="broken"))][-1]
    assert done["type"] == "done" and "8D 分析报告" in done["report_md"]
    trace = client.get(f"/api/cases/{case['id']}").json()["analyses"][0]["trace"]
    assert any("503" in json.dumps(t, ensure_ascii=False) for t in trace)
