"""大模型接入（Pydantic AI 2.x）：类型化输出 + 自动校验重试，任意 OpenAI 兼容端点（DeepSeek / Qwen / vLLM）。

- `structured(Model, ...)`：返回经 Pydantic 校验的对象，模型输出不合规时框架自动要求重试；
- 未配置 Key 时 `available=False`，调用方走确定性规则（统计 + 贝叶斯本身不依赖 LLM）；
- 测试可注入 `FunctionModel` / `TestModel`，无需真实 Key 即可覆盖 LLM 路径。
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TypeVar

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

from pydantic import BaseModel  # noqa: E402
from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.models import Model  # noqa: E402

from .config import Settings, get_settings  # noqa: E402

log = logging.getLogger("kaizen.llm")
T = TypeVar("T", bound=BaseModel)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    errors: list[str] = field(default_factory=list)

    def add(self, u) -> None:
        self.calls += max(1, getattr(u, "requests", 1) or 1)
        self.prompt_tokens += int(getattr(u, "input_tokens", 0) or 0)
        self.completion_tokens += int(getattr(u, "output_tokens", 0) or 0)

    def as_dict(self) -> dict:
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "calls": self.calls, "errors": self.errors[:3]}


class LLM:
    def __init__(self, settings: Settings | None = None, model: Model | None = None, name: str | None = None,
                 model_name: str | None = None):
        self.s = settings or get_settings()
        self._override = model
        self._name = name
        self._model_name = model_name  # 角色专用模型（如课程/验证 agent 用不同模型家族）
        self.usage = Usage()

    @property
    def available(self) -> bool:
        return self._override is not None or not self.s.offline

    @property
    def model_name(self) -> str:
        if self._override is not None:
            return self._name or "injected-model"
        return (self._model_name or self.s.llm_model) if self.available else "offline-rules"

    def model(self) -> Model:
        if self._override is not None:
            return self._override
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        return OpenAIChatModel(self._model_name or self.s.llm_model,
                               provider=OpenAIProvider(base_url=self.s.llm_base_url, api_key=self.s.llm_api_key))

    async def structured(self, output_type: type[T], instructions: str, prompt: str, *,
                         temperature: float | None = None) -> T | None:
        if not self.available:
            return None
        try:
            agent = Agent(self.model(), output_type=output_type, instructions=instructions, retries=2,
                          model_settings={"temperature": self.s.llm_temperature if temperature is None else temperature,
                                          "timeout": self.s.llm_timeout})
            result = await agent.run(prompt)
            self.usage.add(result.usage)
            return result.output
        except Exception as e:  # 网络/限流/鉴权/校验失败都降级，不中断工作流
            log.warning("LLM 调用失败，降级为规则：%s", e)
            self.usage.errors.append(str(e)[:200])
            return None

    async def stream(self, system: str, messages: list[dict]) -> AsyncIterator[str]:
        history = "\n".join(f"{'用户' if m['role'] == 'user' else '助手'}：{m['content']}" for m in messages[:-1])
        prompt = (f"对话记录：\n{history}\n\n" if history else "") + str(messages[-1]["content"])
        agent = Agent(self.model(), instructions=system)
        async with agent.run_stream(prompt) as run:
            async for delta in run.stream_text(delta=True):
                yield delta
            self.usage.add(run.usage)


async def embed(texts: list[str], settings: Settings | None = None) -> list[list[float]] | None:
    """可选向量检索（OpenAI 兼容 /embeddings）。"""
    import httpx

    s = settings or get_settings()
    if not s.embedding_model:
        return None
    base = (s.embedding_base_url or s.llm_base_url).rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(f"{base}/embeddings", headers={"Authorization": f"Bearer {s.embedding_api_key or s.llm_api_key}"},
                                  json={"model": s.embedding_model, "input": texts})
            r.raise_for_status()
            return [d["embedding"] for d in r.json()["data"]]
    except Exception as e:
        log.warning("Embedding 调用失败：%s", e)
        return None
