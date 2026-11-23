#!/usr/bin/env bash
# 本地开发：启动 Kaizen 后端（看板 http://localhost:8000）
# 未设置 KAIZEN_LLM_API_KEY 时使用离线规则引擎；设置后走 DeepSeek / Qwen 等 OpenAI 兼容接口。
set -euo pipefail
cd "$(dirname "$0")/../backend"
if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python -e ".[dev]"
fi
exec .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port "${PORT:-8000}" "$@"
