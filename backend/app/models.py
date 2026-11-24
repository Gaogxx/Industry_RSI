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


# ── Harness 层（v0.6）：快慢双脑、任务经验、双层递归、RRSI 策略进化、可靠度模型 ─────────
class HarnessReceipt(SQLModel, table=True):
    """一次分析的收据（任务经验的原始记录）：走了快脑还是慢脑、为什么、内层递归做了什么、可靠度与保形集合。
    关案后写入 outcome，才参与经验库更新与可靠度模型训练（模型自评不当作标签）。"""

    id: Optional[int] = Field(default=None, primary_key=True)
    case_id: int = Field(foreign_key="case.id", index=True)
    analysis_id: Optional[int] = Field(default=None, index=True)
    scope: str = Field(default="", index=True)  # 产线范围：工厂 + 产品
    symptom: str = Field(default="", index=True)
    signature: dict = _json(dict)  # 证据条件：强/弱、富集类型、PM 后突变、首位、证据信号
    route: str = "fast"  # fast / slow
    reasons: list = _json(list)
    policy_version: int = 0
    reliability: float = 0.0  # 首位根因是真因的校准概率
    reliability_model: int = 0  # 0 = 尚无训练过的模型（用后验作近似）
    conformal: dict = _json(dict)  # {"alpha", "q", "set": [根因代码]}
    inner: dict = _json(dict)  # 内层递归：候选、探测、合成决定、近邻练习（TTT）
    experiences: list = _json(list)  # 用到的经验条目 id
    top_fast: str = ""  # 快脑给出的首位
    top: str = ""  # 最终首位
    outcome: dict = _json(dict)
    verified_at: Optional[datetime] = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=utcnow, index=True)


class HarnessExperience(SQLModel, table=True):
    """双轨任务经验（参照 TTSE 与 Harness² 的改进经验库）：fact = 可核实的产线事实；tip = 带条件的做法。
    每条用 Beta 后验记成败：P<0.45 且失败≥4 次停用（留档可审计），P≥0.72 且≥3 次标为稳定。拿不准时一律记为 tip。"""

    id: Optional[int] = Field(default=None, primary_key=True)
    track: str = Field(default="tip", index=True)  # fact / tip
    key: str = Field(index=True, unique=True)
    scope: str = Field(default="", index=True)  # "" = 全厂通用
    symptom: str = Field(default="", index=True)
    condition: dict = _json(dict)
    text: str = ""
    action: dict = _json(dict)  # tip：{"do": "ttt" | "slow" | "probe", ...}；fact：{"cause": ...}
    alpha: float = 1.0
    beta: float = 1.0
    status: str = Field(default="active", index=True)  # active / stable / retired
    evidence: list = _json(list)  # [{"receipt": id} | {"probe": {...}}]
    blame: list = _json(list)  # 被追责的记录：哪次、为什么
    retired_reason: str = ""
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class HarnessPolicy(SQLModel, table=True):
    """harness 策略存档（跨 run 继承）：每个候选都留档，包括被拒的；status=active 的那一版用于线上。"""

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(default="", index=True)
    parent_id: Optional[int] = None
    params: dict = _json(dict)
    edits: list = _json(list)  # [{"component", "from", "to", "hypothesis"}]
    scores: dict = _json(dict)  # {"search": {...}, "select": {...}, "report": {...}, "noise": ...}
    status: str = Field(default="candidate", index=True)  # candidate / accepted / rejected / active / retired / rolled_back
    note: str = ""
    created_at: datetime = Field(default_factory=utcnow, index=True)


class ReliabilityModel(SQLModel, table=True):
    """可靠度模型（训练机制，与免训练的 RSI 并存）：预测“首位根因是真因”的概率；附保形预测阈值。"""

    id: Optional[int] = Field(default=None, primary_key=True)
    status: str = Field(default="candidate", index=True)  # candidate / active / rejected / failed
    coef: dict = _json(dict)
    window: dict = _json(dict)  # 训练窗口：触发原因、样本池、样本数、信息量占比
    metrics: dict = _json(dict)  # 留出集：logloss / ECE / Brier / 保形覆盖率；对比在用版本
    conformal: dict = _json(dict)  # {"alpha", "q", "n_cal"}
    created_at: datetime = Field(default_factory=utcnow, index=True)


class HarnessBenchmark(SQLModel, table=True):
    """孪生对照实验：只用快脑 vs 快慢双脑（含近邻练习）；以及产线刚变更、记忆还没跟上的压力场景。"""

    id: Optional[int] = Field(default=None, primary_key=True)
    trigger: str = "manual"
    results: dict = _json(dict)
    created_at: datetime = Field(default_factory=utcnow)
