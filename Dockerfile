# Serverless RL InstantStart: API and built UI in one image, on one port (3199).
#   bash scripts/build.sh && bash scripts/start.sh
#
# The run templates (templates/) are not in the image: they are the code users edit, so
# start.sh mounts them read-only from the checkout and an edit needs no rebuild.

# ---- UI ----
FROM node:22-slim AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- Python environment ----
# Built in its own stage so that uv and its download cache stay out of the final image.
# Same base image and same /app/.venv path as the final stage, so the venv works as copied.
FROM python:3.11-slim AS venv
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/app/.venv UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY backend/pyproject.toml backend/uv.lock backend/
RUN --mount=type=cache,target=/root/.cache/uv cd backend && uv sync --frozen --no-dev --no-install-project

# ---- backend + UI ----
FROM python:3.11-slim
WORKDIR /app
# Bytecode is compiled at build time: the venv belongs to root, so the app user could not write it later.
COPY --from=venv /app/.venv /app/.venv
COPY backend/srl backend/srl
COPY --from=ui /ui/dist frontend/dist

# Same uid as the default EC2 user, so files in the mounted data/ stay editable on the host.
RUN useradd -m -u 1000 app && mkdir -p /app/data /home/app/.cache/huggingface && chown -R app /app/data /home/app/.cache
USER app
ENV PATH=/app/.venv/bin:$PATH PYTHONPATH=/app/backend SRL_DATA=/app/data
EXPOSE 3199
CMD ["python", "-m", "srl"]
