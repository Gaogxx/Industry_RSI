"""数据模型（Kaizen · 芯片制造质量的根因分析与 RSI 递归自我改进）。

  真实环境（产线）：Case + FabDataset → Analysis → Action → 工程师确认根因 / CAPA 验证
  练习环境（数字孪生）：RSISession → RSITask（课程 agent 出题 · 执行 agent 作答 · 验证 agent 判定）
  可进化记忆（仅执行 agent 写入，且必须经验证 agent 判定后）：
      CausalEdge（症状→根因知识）· ToolHotspot（机台热点知识）· MemoryItem（流程与教训，保留条件与不确定性）
  度量：BacktestRun（真实案例预序回放）· RSIBenchmark（孪生留出测试集上的冷启动/消融）
"""

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(timezone.utc)  # SQLModel 0.0.50+ 要求带时区，读回统一为 UTC


CASE_STATUSES = {
    "open": "待分析",
    "analyzing": "分析中",
    "capa": "措施执行",
    "verifying": "效果验证",
    "closed": "已关闭",
}
SEVERITIES = {"S1": "重大", "S2": "严重", "S3": "一般", "S4": "轻微"}
SIX_M = {"man": "人", "machine": "机", "material": "料", "method": "法", "environment": "环", "measurement": "测"}
ACTION_KINDS = {"containment": "围堵措施", "corrective": "纠正措施", "preventive": "预防措施"}
ACTION_STATUSES = {"todo": "待办", "doing": "进行中", "done": "已完成", "verified": "已验证有效", "ineffective": "验证无效"}
BULLET_STATUSES = {"active": "生效", "retired": "已停用"}
MEMORY_KINDS = {"knowledge": "环境知识", "procedure": "可复用流程", "lesson": "失败教训"}
SOURCES = {"real": "产线验证", "sandbox": "孪生验证"}


def _json(default: Any = None):
    return Field(default_factory=lambda: default() if callable(default) else default, sa_column=Column(JSON))


class Case(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    stage: str = Field(default="fe", index=True)  # flow.STAGES
    symptom: str = Field(default="", index=True)  # flow.SYMPTOMS
    product: str = ""
    severity: str = "S2"
    status: str = Field(default="open", index=True)
    site: str = ""
    line: str = ""
    equipment: str = ""
    description: str = ""
    data: dict = _json(dict)  # 现场关键指标
    evidence: dict = _json(dict)  # 统计分析证据摘要（共性分析 / 晶圆图 / FDC / 变点）
    root_cause: str = ""
    root_cause_category: str = ""
    cause_code: str = ""  # 工程师确认的根因（关案时写入，是学习的“标签”）
    cost_impact: float = 0.0  # 万元
    downtime_hours: float = 0.0
    reporter: str = ""
    source: str = "manual"
    created_at: datetime = Field(default_factory=utcnow, index=True)
    closed_at: Optional[datetime] = None

    @property
    def code(self) -> str:
        return f"KZ-{self.id:04d}" if self.id else "KZ-NEW"


class FabDataset(SQLModel, table=True):
    """案例关联的原始数据（批次履历、晶圆图、FDC、PM 事件），真实部署时来自 MES/YMS/FDC 只读接口。"""

    case_id: int = Field(foreign_key="case.id", primary_key=True)
    data: dict = _json(dict)


class Analysis(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    case_id: int = Field(foreign_key="case.id", index=True)
    version: int = 1
    problem_statement: dict = _json(dict)
    evidence: dict = _json(dict)
    hypotheses: list = _json(list)  # 贝叶斯根因排序（含每个因子的贡献）
    counterfactual: dict = _json(dict)  # 若不学习（教科书先验）会怎样排序
    ishikawa: dict = _json(dict)
    five_whys: list = _json(list)
    root_causes: list = _json(list)
    containment: list = _json(list)
    critique: dict = _json(dict)
    references: list = _json(list)
    iterations: int = 1
    report_md: str = ""
    model: str = ""
    trace: list = _json(list)
    created_at: datetime = Field(default_factory=utcnow)


class Action(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    case_id: int = Field(foreign_key="case.id", index=True)
    analysis_id: Optional[int] = Field(default=None, foreign_key="analysis.id")
    kind: str = "corrective"
    title: str
    cause_code: str = ""  # 该措施针对的根因（验证无效时据此否定）
    owner_role: str = ""
    due_days: int = 7
    due_date: Optional[datetime] = None
    status: str = Field(default="todo", index=True)
    verify_metric: str = ""
    baseline: str = ""
    target: str = ""
    result: str = ""
    created_at: datetime = Field(default_factory=utcnow)
    completed_at: Optional[datetime] = None


class CausalEdge(SQLModel, table=True):
    """贝叶斯因果知识图谱的一条边：症状 → 根因，Beta(α, β)。"""

    id: Optional[int] = Field(default=None, primary_key=True)
    symptom: str = Field(index=True)
    cause_code: str = Field(index=True)
    name: str
    step: str = ""
    six_m: str = ""
    alpha: float
    beta: float
    alpha0: float
    beta0: float
    confirmed: int = 0
    refuted: int = 0
    confirmed_sandbox: int = 0
    learned: bool = False
    updated_at: datetime = Field(default_factory=utcnow)


class ToolHotspot(SQLModel, table=True):
    """环境知识：某根因在本厂多发于哪台机台/腔体（排序时作为 P(机台|根因) 似然）。"""

    id: Optional[int] = Field(default=None, primary_key=True)
    cause_code: str = Field(index=True)
    tool: str = Field(index=True)
    real: float = 0.0
    sandbox: float = 0.0
    updated_at: datetime = Field(default_factory=utcnow)


class LearningEvent(SQLModel, table=True):
    """每一次学习都留痕：哪条知识、因哪个案例、从多少变成多少。"""

    id: Optional[int] = Field(default=None, primary_key=True)
    case_id: Optional[int] = Field(default=None, foreign_key="case.id", index=True)
    kind: str  # confirm / false_lead / refute / edge_added / hotspot / memory_add / memory_pass / memory_fail / memory_retire
    source: str = "real"  # real（产线）/ sandbox（数字孪生）
    rsi_task_id: Optional[int] = Field(default=None, index=True)
    symptom: str = ""
    cause_code: str = ""
    before: float = 0.0
    after: float = 0.0
    detail: str = ""
    created_at: datetime = Field(default_factory=utcnow, index=True)


class MemoryItem(SQLModel, table=True):
    """可进化记忆条目（RSIAgent：执行 agent 独占写入，须经验证 agent 判定）。

    与只存结论的经验不同，每条记忆保留“在什么条件下”与“验证通过/失败多少次”，避免把未经充分验证的
    结论固化成规则（论文失败模式：记忆固化不可靠）。
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    kind: str = "procedure"  # knowledge / procedure / lesson
    key: str = Field(index=True)
    symptom: str = Field(index=True)
    cause_code: str = ""
    condition: dict = _json(dict)  # {"evidence": "strong|weak", "after_pm": bool, "factor": "tool|batch|none"}
    text: str
    passes: int = 0
    fails: int = 0
    source: str = "real"  # real / sandbox / both
    source_cases: list = _json(list)
    source_tasks: list = _json(list)
    status: str = Field(default="active", index=True)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class RSISession(SQLModel, table=True):
    """一次递归自我改进探索（广度 BRS → 证据校准 CAL → 深度 DRS），在数字孪生中进行。"""

    id: Optional[int] = Field(default=None, primary_key=True)
    workflow_id: str = ""
    trigger: str = "manual"
    status: str = "running"  # running / done / error
    config: dict = _json(dict)
    curve: list = _json(list)  # [{step, phase, hit1, partial, memory_items}]
    weights_log: list = _json(list)  # 每次证据权重重拟合：候选权重、样本数、验收门结果、是否采纳
    summary: dict = _json(dict)
    created_at: datetime = Field(default_factory=utcnow)
    finished_at: Optional[datetime] = None


class RSITask(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="rsisession.id", index=True)
    phase: str = "BRS"  # BRS / DRS
    step: int = 0
    spec: dict = _json(dict)  # 课程 agent 出题：症状、难度、变体类型、理由
    attempts: list = _json(list)  # 执行 agent 每次尝试 + 验证 agent 判定与证据
    verdict: str = ""  # PASS / FAIL
    memory_changes: list = _json(list)
    hidden: dict = _json(dict)  # 孪生内部真因（三个 agent 都看不到，仅供事后审计）
    evidence: dict = _json(dict)  # 统计证据（验证通过后作为证据权重学习的训练样本）
    symptom: str = ""
    weak: bool = False  # 只看证据时首位不到五成把握（课程 agent 估计“证据模糊率”用）
    created_at: datetime = Field(default_factory=utcnow)


class EvidenceModel(SQLModel, table=True):
    """学到的证据权重（各类证据的可信度指数），每次重拟合保存一版，分析真实案例时使用最新版。"""

    id: Optional[int] = Field(default=None, primary_key=True)
    weights: dict = _json(dict)
    n: int = 0
    avg_loglik: float = 0.0
    source: str = "rsi"  # rsi（孪生练习）/ production（产线确认）
    gate: dict = _json(dict)  # 验收门：采纳前后在封存面板上的首位命中率与对数损失
    created_at: datetime = Field(default_factory=utcnow)


class RSIBenchmark(SQLModel, table=True):
    """冷启动基准：从教科书先验出发，比较 完整 RSI / 仅广度 / 仅深度 / 不做 RSI 在留出测试集上的表现。"""

    id: Optional[int] = Field(default=None, primary_key=True)
    trigger: str = "manual"
    results: dict = _json(dict)
    created_at: datetime = Field(default_factory=utcnow)


class BacktestRun(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    trigger: str = "manual"
    points: list = _json(list)
    summary: dict = _json(dict)
    created_at: datetime = Field(default_factory=utcnow)


class Feedback(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    analysis_id: int = Field(foreign_key="analysis.id", index=True)
    rating: int
    comment: str = ""
    created_at: datetime = Field(default_factory=utcnow)


class AuditLog(SQLModel, table=True):
    """审计日志：满足《智能体规范应用与创新发展实施意见》(2026-05) 对可追溯、可问责的要求。"""

    id: Optional[int] = Field(default=None, primary_key=True)
    actor: str = "system"
    action: str
    target: str = ""
    detail: dict = _json(dict)
    created_at: datetime = Field(default_factory=utcnow, index=True)


class ImprovementCycle(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    trigger: str = "manual"
    kpis: dict = _json(dict)
    pareto: list = _json(list)
    learning: dict = _json(dict)
    findings: list = _json(list)
    summary_md: str = ""
    created_at: datetime = Field(default_factory=utcnow)
