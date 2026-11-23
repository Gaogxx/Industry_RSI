"""运行配置。全部来自环境变量 / .env，源码中不出现任何密钥。"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="KAIZEN_", extra="ignore")

    # ── 服务 ─────────────────────────────────────────────
    # 访问 /agui、/mcp 与写接口时需要的 Bearer Token；留空表示本地开发不鉴权
    api_key: str = ""
    # 看板只读接口是否允许匿名访问（内网部署可开，公网建议关闭）
    dashboard_public: bool = True
    database_url: str = "sqlite:///./data/kaizen.db"
    # DBOS 持久化工作流的系统库（SQLite 即可；多实例部署改为 PostgreSQL）
    dbos_database_url: str = "sqlite:///./data/kaizen-dbos.sqlite"
    seed_demo: bool = True
    # 可选：/mcp 工具服务器（Claude / VS Code / Cursor 等 MCP 客户端调用案例库与分析工作流）
    enable_mcp: bool = True

    # ── 大模型（任意 OpenAI 兼容接口：DeepSeek / 通义千问 DashScope / vLLM / Ollama）──
    # DeepSeek：deepseek-v4-pro（推理质量优先）或 deepseek-flash（成本优先，始终指向最新 Flash）
    #   注意 deepseek-chat / deepseek-reasoner 已于 2026-07-24 下线。
    # 通义千问：base_url=https://dashscope.aliyuncs.com/compatible-mode/v1  model=qwen4-max
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-v4-pro"
    llm_temperature: float = 0.2
    llm_timeout: float = 120.0
    # 没有配置 llm_api_key 时自动进入离线模式（规则引擎），便于演示与单测
    llm_offline: bool = False

    # 向量检索（可选）；不配置时只用 BM25 关键词检索
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_model: str = ""

    # ── 分析工作流 ───────────────────────────────────────
    critic_threshold: int = 75  # 低于此分数触发带意见重写
    max_iterations: int = 2
    investigator: bool = True  # 配置了大模型时启用代码模式调查员

    # ── RSI 三角色模型（留空则与执行 agent 相同）。论文：GLM-5.3 执行，Kimi-K3 课程 + 验证；
    #    国内可配 deepseek-v4-pro 执行、qwen4-max 课程（需同一 OpenAI 兼容网关或自行扩展）
    rsi_curriculum_model: str = ""
    rsi_on_startup: bool = True  # 首次写入演示数据后自动跑一次冷启动基准 + 一次在线 RSI 会话

    # ── 持续学习 / 改进 ──────────────────────────────────
    nightly_cron: str = "0 2 * * *"  # DBOS 定时：改进周期 + 学习曲线回放；留空关闭

    @property
    def offline(self) -> bool:
        return self.llm_offline or not self.llm_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()
