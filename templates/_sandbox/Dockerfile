# The environment sandbox image (docs/architecture.md 3.10). Build context is templates/:
#   docker buildx build --platform linux/arm64 -f templates/_sandbox/Dockerfile templates
# scripts/deploy_sandbox.py does this, pushes the image and points the runtime at it.
# AgentCore Runtime only runs linux/arm64 images.
FROM public.ecr.aws/docker/library/python:3.13-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY _sandbox/requirements.txt .
RUN pip install -r requirements.txt
COPY _sandbox/main.py _shared/sandbox_exec.py ./
# A missing dependency would otherwise only show as a runtime that never becomes healthy.
RUN python -c "import main, sandbox_exec; print('sandbox_exec', sandbox_exec.CODE_SHA)"
EXPOSE 8080
CMD ["python", "main.py"]
