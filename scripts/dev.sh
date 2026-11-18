#!/usr/bin/env bash
# Development: backend with auto-reload on :3199, Vite dev server on :5199 (proxies /api).
set -euo pipefail
cd "$(dirname "$0")/.."
(cd backend && uv sync && exec .venv/bin/python -m srl --reload) &
BACKEND=$!
trap 'kill $BACKEND 2>/dev/null' EXIT
cd frontend && npm install --no-audit --no-fund && exec npx vite
