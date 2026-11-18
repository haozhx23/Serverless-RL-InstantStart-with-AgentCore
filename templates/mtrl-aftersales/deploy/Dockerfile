# AgentCore Runtime 只支持 ARM64。在 x86 机器上构建需要先注册 qemu：
#   docker run --privileged --rm tonistiigi/binfmt --install arm64
# 然后 docker buildx build --platform linux/arm64 --push .
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 标准装法：完整依赖树。qemu 模拟下这一步是整个构建里最慢的部分。
COPY deploy/requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

# 构建期自检：容器里真的能 import 到运行所需的模块。
# 放在这里而不是留到部署后，是因为依赖缺失只会在 import 时暴露，
# 而 AgentCore 上的表现是"容器起不来"，排查链路长得多。
COPY sample/ ./sample/
COPY deploy/main.py .
RUN python -c "import sample.rollout_handler; from bedrock_agentcore.runtime import BedrockAgentCoreApp; print('import self-check OK')"

EXPOSE 8080
CMD ["python", "main.py"]
