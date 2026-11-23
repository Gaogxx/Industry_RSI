"""AI 加速器（GPU 类）芯片全流程：前道晶圆制造 → CP → 2.5D 先进封装 + HBM → FT/老化/SLT → 板卡与液冷整机柜。

流程参考公开资料中英伟达数据中心 GPU 的供应链（代工厂先进制程 + CoWoS 类 2.5D 封装 + HBM + 系统级测试 +
ODM 整机柜集成）与国内晶圆厂 / 封测厂的典型产线；机台编号、参数均为虚构。

因果知识（症状 → 根因）带“教科书先验”，持续学习会根据本厂已验证的案例不断修正它们。
"""

from __future__ import annotations

from dataclasses import dataclass, field

STAGES = {
    "fe": "前道晶圆制造",
    "sort": "晶圆测试 CP",
    "pkg": "2.5D 先进封装",
    "test": "成品测试 FT/BI/SLT",
    "sys": "板卡与整机柜",
}


@dataclass(frozen=True)
class Step:
    code: str
    name: str
    stage: str
    tools: tuple[str, ...]


STEPS: dict[str, Step] = {s.code: s for s in [
    Step("LITHO", "光刻（EUV / DUV 多重图形）", "fe", ("SCN-01", "SCN-02", "SCN-03")),
    Step("ETCH", "干法刻蚀", "fe", ("ETC-01A", "ETC-01B", "ETC-02A", "ETC-02B")),
    Step("DEP", "薄膜沉积 CVD/ALD", "fe", ("DEP-01", "DEP-02", "DEP-03")),
    Step("PVD", "金属溅射 PVD", "fe", ("PVD-01", "PVD-02")),
    Step("CMP", "化学机械抛光 CMP", "fe", ("CMP-01", "CMP-02", "CMP-03")),
    Step("IMP", "离子注入", "fe", ("IMP-01", "IMP-02")),
    Step("WET", "湿法清洗与传送", "fe", ("WET-01", "WET-02")),
    Step("CP", "晶圆测试 CP", "sort", ("CPT-01", "CPT-02", "CPT-03")),
    Step("BUMP", "微凸点 / 倒装键合", "pkg", ("FCB-01", "FCB-02", "FCB-03")),
    Step("COWOS", "2.5D 中介层集成（CoWoS 类）", "pkg", ("CWS-01", "CWS-02")),
    Step("HBM", "HBM 堆叠贴装（TCB 热压键合）", "pkg", ("HBA-01", "HBA-02")),
    Step("UF", "底部填充与固化", "pkg", ("UF-01", "UF-02")),
    Step("FT", "成品测试 FT", "test", ("FTT-01", "FTT-02", "FTT-03")),
    Step("BI", "老化 Burn-in", "test", ("BI-01", "BI-02")),
    Step("SLT", "系统级测试 SLT", "test", ("SLT-01", "SLT-02", "SLT-03")),
    Step("SMT", "GPU 板卡 SMT", "sys", ("SMT-01", "SMT-02")),
    Step("RACK", "整机柜集成与液冷", "sys", ("RCK-01", "RCK-02", "RCK-03")),
]}

# 各类批次经过的工序（用于生成批次履历并做共性分析）
ROUTES = {
    "fe": ["LITHO", "ETCH", "DEP", "PVD", "CMP", "IMP", "WET", "CP"],
    "pkg": ["BUMP", "COWOS", "HBM", "UF", "FT"],
    "test": ["BUMP", "COWOS", "HBM", "UF", "FT", "BI", "SLT"],
    "sys": ["SMT", "RACK"],
}


@dataclass(frozen=True)
class Symptom:
    code: str
    name: str
    stage: str  # 路由类别：fe / pkg / test / sys
    metric: str
    keywords: tuple[str, ...]
    wafer_pattern: str | None = None  # 晶圆图空间特征


SYMPTOMS: dict[str, Symptom] = {s.code: s for s in [
    Symptom("WM_EDGE", "晶圆边缘环形失效", "fe", "CP 良率", ("边缘", "edge", "环形", "外圈"), "edge"),
    Symptom("WM_CENTER", "晶圆中心区域失效", "fe", "CP 良率", ("中心", "center", "中间"), "center"),
    Symptom("WM_SCRATCH", "晶圆划痕型失效", "fe", "CP 良率", ("划痕", "划伤", "scratch", "线状"), "scratch"),
    Symptom("WM_CLUSTER", "晶圆簇状颗粒失效", "fe", "CP 良率 / 缺陷密度 D0", ("簇状", "颗粒", "particle", "cluster", "D0"), "cluster"),
    Symptom("OVL_DRIFT", "套刻 / CD 漂移", "fe", "Overlay 3σ (nm)", ("套刻", "overlay", "CD", "多重图形", "对准")),
    Symptom("PKG_WARP", "封装翘曲导致凸点开路", "pkg", "翘曲 (µm) / 开路 DPPM", ("翘曲", "warpage", "开路", "C4", "中介层")),
    Symptom("PKG_VOID", "微凸点 / 底填空洞", "pkg", "空洞率 %", ("空洞", "void", "底填", "underfill", "微凸点")),
    Symptom("HBM_OPEN", "HBM 堆叠互连开路", "pkg", "HBM 链路失效 DPPM", ("HBM", "TSV", "堆叠", "TCB")),
    Symptom("TEST_VMIN", "Vmin 漂移 / 速度分档下降", "test", "Vmin (mV) / 高档良率", ("Vmin", "分档", "bin", "频率", "速度")),
    Symptom("BI_EARLY", "老化早期失效", "test", "BI 失效率 DPPM", ("老化", "burn-in", "早期失效", "早夭", "BI")),
    Symptom("SLT_HANG", "系统级测试失效（死机 / 静默数据错误 SDC）", "test", "SLT 失效率 %",
            ("SLT", "系统级", "死机", "训练失败", "hang", "SDC", "静默数据")),
    Symptom("RACK_LEAK", "液冷整机柜泄漏", "sys", "泄漏次数 / 柜", ("液冷", "泄漏", "漏液", "快接头", "整机柜", "NVL")),
]}


@dataclass(frozen=True)
class Cause:
    code: str
    name: str
    step: str
    six_m: str
    prior: float  # 教科书先验：该症状下此根因的概率
    fdc: tuple[str, ...] = ()
    pm_related: bool = False
    tool_specific: bool = True  # 是否会在某台机台/腔体上体现（决定共性分析能否抓到）
    capa: tuple[str, str, str] = ("", "", "")  # 围堵 / 纠正 / 预防
    metric: str = ""


# 症状 → 候选根因（先验来自通用工艺知识，代表“没有本厂数据时”的判断）
CAUSES: dict[str, list[Cause]] = {
    "WM_EDGE": [
        Cause("ETCH_FOCUS_RING", "刻蚀腔聚焦环磨损，边缘等离子体鞘层偏移", "ETCH", "machine", 0.30, ("RF反射功率", "边缘ESC温度"), True, True,
              ("暂停可疑腔体投片，已加工批次加扫边缘缺陷", "更换聚焦环并做边缘 CD 均匀性验证", "聚焦环按 RF 小时数预测性更换，纳入 FDC 联锁"), "边缘区良率"),
        Cause("CMP_RETAINING_RING", "CMP 抛光头保持环磨损导致边缘过抛", "CMP", "machine", 0.35, ("保持环压力",), False, True,
              ("隔离可疑批次并量测边缘膜厚", "更换保持环并重新标定边缘压力区", "保持环寿命与晶圆计数联动"), "边缘膜厚均匀性"),
        Cause("ALD_EDGE_TEMP", "ALD 加热盘边缘温度不均导致边缘膜厚偏薄", "DEP", "machine", 0.20, ("加热盘边缘温度",), False, True,
              ("加测边缘膜厚", "校准加热盘温区", "加热盘温度均匀性纳入 PM 验收"), "边缘膜厚"),
        Cause("LITHO_EBR", "光刻胶去边(EBR)宽度异常", "LITHO", "method", 0.15, (), False, False,
              ("复核 EBR 配方", "修正 EBR 参数并首件确认", "配方变更纳入 RMS 审批"), "边缘曝光有效区"),
    ],
    "WM_CENTER": [
        Cause("CVD_SHOWERHEAD", "CVD 喷淋头堵塞导致中心膜厚偏厚", "DEP", "machine", 0.40, ("喷淋头压差",), True, True,
              ("隔离批次并量测中心膜厚", "清洗/更换喷淋头", "喷淋头压差纳入 FDC 趋势预警"), "膜厚均匀性"),
        Cause("IMP_DOSE_CENTER", "离子注入扫描剂量中心偏移", "IMP", "machine", 0.30, ("束流均匀性",), False, True,
              ("加测中心方块电阻", "重新标定束流扫描", "剂量均匀性每班监控片"), "方块电阻均匀性"),
        Cause("WET_CENTER_RESIDUE", "单片湿法清洗中心残留（喷嘴流量不足）", "WET", "machine", 0.30, ("喷嘴流量",), False, True,
              ("加扫中心残留缺陷", "更换喷嘴并校准流量", "流量计纳入 EAP 实时联锁"), "中心缺陷数"),
    ],
    "WM_SCRATCH": [
        Cause("CMP_SLURRY_LPC", "CMP 研磨液大颗粒 (LPC) 超标", "CMP", "material", 0.45, (), False, False,
              ("更换研磨液批次与过滤器", "过滤器改为压差+时间双判据更换", "研磨液 LPC 在线监测并纳入来料检验"), "划痕缺陷数/片"),
        Cause("CMP_DISK", "CMP 修整盘掉钻", "CMP", "machine", 0.35, ("修整盘扭矩",), False, True,
              ("停机检查修整盘", "更换修整盘", "修整盘扭矩异常检测联锁"), "划痕缺陷数/片"),
        Cause("ROBOT_SCRATCH", "传送机械手 / 片盒定位偏差造成划伤", "WET", "machine", 0.20, ("机械手定位偏差",), False, True,
              ("检查机械手与片盒", "重新示教机械手", "机械手定位偏差纳入 EAP 监控"), "划伤片数"),
    ],
    "WM_CLUSTER": [
        Cause("PM_SEASONING", "腔体 PM 后 seasoning 不足，腔壁沉积物剥落", "ETCH", "method", 0.35, ("腔体压力波动",), True, True,
              ("暂停腔体，补做 seasoning 并跑颗粒监控片", "修订 PM 复机 SOP：颗粒监控片合格方可放行", "跨腔体横展复机标准"), "缺陷密度 D0"),
        Cause("RETICLE_HAZE", "光罩雾化 / 颗粒", "LITHO", "material", 0.35, (), False, True,
              ("下线光罩检查", "光罩清洗并复检", "光罩按曝光剂量累计做定期检查"), "重复性缺陷"),
        Cause("CVD_FLAKE", "CVD 腔体颗粒剥落", "DEP", "machine", 0.30, ("腔体颗粒计数",), True, True,
              ("停机清腔", "缩短清腔周期", "腔体颗粒计数趋势预警"), "缺陷密度 D0"),
    ],
    "OVL_DRIFT": [
        Cause("LENS_HEATING", "扫描机镜头热效应补偿失效", "LITHO", "machine", 0.40, ("镜头温度",), False, True,
              ("超规批次返工", "重新标定镜头热补偿", "镜头热补偿模型定期健康检查"), "Overlay 3σ"),
        Cause("R2R_MODEL", "R2R 反馈模型参数过时", "LITHO", "method", 0.35, (), False, False,
              ("人工补偿临时调整", "重新标定 R2R 模型", "R2R 模型漂移监控与定期重训"), "Overlay 3σ"),
        Cause("WAFER_WARP", "前层应力导致晶圆翘曲", "DEP", "material", 0.15, ("膜应力",), False, True,
              ("量测翘曲", "调整前层膜应力", "应力纳入 SPC"), "晶圆翘曲"),
        Cause("SCANNER_MATCHING", "新导入扫描机与既有机台套刻匹配误差（DUV 多重图形 EPE 叠加）", "LITHO", "machine", 0.10,
              ("匹配残差",), False, True,
              ("可疑层改派到基准机台", "做机台间匹配 (matching) 校正并更新 APC 补偿", "新机台导入增加多层匹配验收与 SAQP EPE 监控"),
              "机台间套刻匹配残差"),
    ],
    "PKG_WARP": [
        Cause("SUBSTRATE_CTE", "封装基板批次 CTE 偏差，与中介层失配", "COWOS", "material", 0.40, (), False, False,
              ("隔离同批次基板", "基板供应商 8D，增加来料 CTE 抽检", "基板来料 CTE 纳入 SQE 准入"), "翘曲 µm"),
        Cause("BOND_TEMP_UNIFORM", "2.5D 键合台温度不均", "COWOS", "machine", 0.35, ("键合台温差",), False, True,
              ("加测翘曲", "校准键合台温区", "键合台温差纳入 FDC 联锁"), "翘曲 µm"),
        Cause("CURE_PROFILE", "塑封/底填固化曲线偏离", "UF", "method", 0.25, ("固化炉温度",), False, True,
              ("复核固化曲线", "修正固化曲线并验证", "固化曲线变更走变更管理"), "翘曲 µm"),
    ],
    "PKG_VOID": [
        Cause("REFLOW_PROFILE", "回流曲线恒温区不足，助焊剂挥发物被包裹", "BUMP", "method", 0.40, ("回流峰值温度",), False, True,
              ("X-Ray 全检可疑批次", "重新优化回流曲线", "换线强制测温板验证"), "空洞率 %"),
        Cause("FLUX_RESIDUE", "助焊剂批次残留偏多", "BUMP", "material", 0.30, (), False, False,
              ("更换助焊剂批次", "清洗工艺验证", "助焊剂来料检验"), "空洞率 %"),
        Cause("UF_DISPENSE", "底填点胶温度/速度偏离", "UF", "machine", 0.30, ("点胶温度",), False, True,
              ("超声扫描 (C-SAM) 全检", "校准点胶阀与温控", "点胶参数纳入 FDC"), "底填空洞率"),
    ],
    "HBM_OPEN": [
        Cause("TCB_COPLANARITY", "TCB 键合头共面度偏差 / 压力不均", "HBM", "machine", 0.40, ("键合压力",), True, True,
              ("隔离可疑键合头产品", "校准键合头共面度", "键合压力曲线实时监控"), "HBM 链路失效 DPPM"),
        Cause("NCF_LOT", "NCF 非导电膜批次异常", "HBM", "material", 0.35, (), False, False,
              ("更换 NCF 批次", "供应商 8D", "NCF 来料流变测试"), "HBM 链路失效 DPPM"),
        Cause("HBM_INCOMING", "HBM 来料 TSV 缺陷（存储厂批次）", "HBM", "material", 0.25, (), False, False,
              ("按 HBM 来料批次追溯", "与存储厂联合 FA", "来料 KGSD 测试覆盖率提升"), "HBM 来料不良率"),
    ],
    "TEST_VMIN": [
        Cause("PROCESS_SLOW_CORNER", "前道工艺偏慢角（注入/栅极 CD 漂移）", "IMP", "material", 0.35, (), False, False,
              ("按前道批次追溯分档", "前道工艺窗口回中", "WAT 参数与分档联动预警"), "高档良率"),
        Cause("SOCKET_CONTACT", "测试座 / 探针接触电阻升高", "FT", "machine", 0.35, ("接触电阻",), False, True,
              ("更换测试座并复测", "测试座寿命按插拔次数管理", "接触电阻实时监控"), "复测通过率"),
        Cause("TEST_PROGRAM", "测试程序电压/温度设定变更", "FT", "method", 0.30, (), False, False,
              ("回退测试程序版本", "测试程序变更走审批", "测试程序版本纳入 RMS"), "Vmin 分布"),
    ],
    "BI_EARLY": [
        Cause("GATE_OXIDE", "栅氧完整性缺陷（前道）", "DEP", "material", 0.35, (), False, False,
              ("按前道批次隔离", "前道栅氧工艺复查", "增加 TDDB 抽样"), "BI 失效率"),
        Cause("BI_SOCKET_HEAT", "老化座局部过热", "BI", "machine", 0.35, ("老化板温度",), False, True,
              ("更换老化板", "老化板温度均匀性校准", "老化板温度监控"), "BI 失效率"),
        Cause("ESD_HANDLING", "搬运 ESD 损伤", "FT", "man", 0.30, (), False, False,
              ("ESD 稽核", "防静电培训与接地检查", "ESD 在线监控"), "ESD 失效数"),
    ],
    "SLT_HANG": [
        Cause("HBM_TRAINING", "HBM 链路训练裕量不足（PHY 参数）", "SLT", "method", 0.35, (), False, False,
              ("调整训练参数复测", "固件 PHY 参数优化", "训练裕量纳入 SLT 指标"), "SLT 失效率"),
        Cause("POWER_SEQ", "板级电源时序异常", "SLT", "machine", 0.35, ("上电时序偏差",), False, True,
              ("检查 SLT 母板电源", "更换母板 VRM", "母板电源健康检查"), "SLT 失效率"),
        Cause("TIM_COVERAGE", "热界面材料覆盖不足导致过温", "UF", "method", 0.30, (), False, True,
              ("加测结温", "TIM 点胶图形优化", "TIM 覆盖率 AOI 检测"), "结温"),
    ],
    "RACK_LEAK": [
        Cause("QD_ORING", "快换接头 O 型圈材料与冷却液不兼容", "RACK", "material", 0.40, (), False, False,
              ("更换同批次 O 型圈", "供应商材料兼容性验证", "来料材料认证"), "泄漏次数/柜"),
        Cause("FITTING_TORQUE", "快接头 / 分支管路接头装配扭矩不足", "RACK", "man", 0.35, ("拧紧扭矩",), False, True,
              ("复紧全部接头", "电动扭矩扳手 + 数据追溯", "装配防错"), "泄漏次数/柜"),
        Cause("MANIFOLD_WELD", "分歧管焊缝缺陷（供应商批次）", "RACK", "material", 0.25, (), False, False,
              ("隔离同批次分歧管", "供应商 8D 与焊缝 X-Ray", "焊缝来料抽检"), "泄漏次数/柜"),
    ],
}


def cause_by_code(code: str) -> tuple[str, Cause] | None:
    for sym, causes in CAUSES.items():
        for c in causes:
            if c.code == code:
                return sym, c
    return None


def guess_symptom(text: str) -> str:
    t = text.lower()
    best, score = "WM_EDGE", 0
    for s in SYMPTOMS.values():
        sc = sum(len(k) for k in s.keywords if k.lower() in t)
        if sc > score:
            best, score = s.code, sc
    return best


def domain_context(symptom: str) -> str:
    s = SYMPTOMS.get(symptom)
    if not s:
        return ""
    causes = "\n".join(f"- {c.code}：{c.name}（工序 {STEPS[c.step].name}）" for c in CAUSES.get(symptom, []))
    return f"工段：{STAGES[s.stage]}；症状：{s.name}；指标：{s.metric}\n候选根因：\n{causes}"


@dataclass
class Scenario:
    """数据模拟用的案例场景。"""

    symptom: str
    cause: str
    strength: float = 1.0  # 证据强度：1=信号明显；<0.5 共性分析难以区分
    seed: int = 0
    n_lots: int = 160
    extras: dict = field(default_factory=dict)
