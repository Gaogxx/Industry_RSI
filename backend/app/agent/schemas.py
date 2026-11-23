"""LLM 输出的类型约束（Pydantic AI 自动校验，不合规会要求模型重试）。"""

from typing import Literal

from pydantic import BaseModel, Field


class IntakeOut(BaseModel):
    problem_statement: str
    what: str = ""
    where: str = ""
    when: str = ""
    who: str = ""
    how_many: str = ""
    how_detected: str = ""
    impact: str = ""
    symptom: str = Field(description="症状 code，必须来自给定列表")
    severity: Literal["S1", "S2", "S3", "S4"] = "S2"


class ExplainOut(BaseModel):
    five_whys: list[str] = Field(min_length=3, max_length=7)
    ishikawa_extra: dict[str, list[str]] = Field(default_factory=dict, description="键为 man/machine/material/method/environment/measurement")
    narrative: str
    risks: list[str] = Field(default_factory=list)


class CapaItem(BaseModel):
    kind: Literal["containment", "corrective", "preventive"]
    title: str
    owner_role: str = ""
    due_days: int = Field(default=7, ge=1, le=180)
    verify_metric: str
    baseline: str = ""
    target: str = ""


class CapaOut(BaseModel):
    actions: list[CapaItem] = Field(min_length=3)


class CriticOut(BaseModel):
    score: int = Field(ge=0, le=100)
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class InvestigationOut(BaseModel):
    findings: list[str] = Field(description="用数据验证得到的事实，每条带数字")
    supports: list[str] = Field(default_factory=list, description="数据支持的根因 code")
    contradicts: list[str] = Field(default_factory=list, description="数据反驳的根因 code")
    next_checks: list[str] = Field(default_factory=list, description="建议的下一步验证")
