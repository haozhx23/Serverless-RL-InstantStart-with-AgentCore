"""AgentCore Runtime 容器入口（实际部署用的就是这个文件）。

AgentCore Runtime 的容器契约（官方 runtime-http-protocol-contract）共五项：
host ``0.0.0.0``、port ``8080``、``POST /invocations``、``GET /ping``、**ARM64 架构**。
``BedrockAgentCoreApp`` 负责其中的前四项（监听地址、端口和两个路由）；
ARM64 不是应用框架能解决的，只能在构建镜像时保证（见 ``deploy/Dockerfile``）。

冷启动：为什么把重导入放到后台线程
----------------------------------
``import sagemaker.train.rft`` 会触发 ``sagemaker.core.__init__``，后者的传递导入
链很长（telemetry -> session_helper -> common_utils -> workflow -> pandas/mlflow）。
在本项目的 arm64 容器里实测：

    sagemaker.core                  24.90 s
    sagemaker.core.token_generator  24.92 s   （耗时全部来自 sagemaker.core）
    sagemaker.train.rft             26.04 s
    本样例自己的 sample.* 模块        0.14 s

（数据来自 x86 机器上的 qemu 模拟，真机 Graviton 会快若干倍，但量级不变。）

如果在模块顶层直接导入，uvicorn 要等二十多秒才开始监听，这段时间容器
**不响应 /ping 也不响应 /invocations**。实测后果：AgentCore 冷启动时打进来的
调用直接拿到 500，CloudWatch 里只有一行启动日志、看不到任何业务日志——
排查时极容易误判成 IAM 或报文问题。

放到 MTRL 的场景下这个问题会被放大：服务会并发下发大量 rollout
（``rollout_max_concurrency`` 默认 96），冷启动期间失败的 rollout 会被记成
失败轨迹，污染 ``analysis/batch_completion_ratio`` 这类指标。

``PingStatus`` 只有 ``Healthy`` 和 ``HealthyBusy`` 两种取值，没有"未就绪"，
所以无法靠 /ping 让 AgentCore 等待。可行的办法是反过来：**让 uvicorn 立刻
监听**，把重导入交给后台线程预热，并在预热完成前用 ``HealthyBusy`` 表明
自己还在加载。第一条请求如果比预热更早到达，会在 Python 的 import 锁上
等待预热线程完成，而不是重复导入一遍。
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from typing import Any

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("aftersales-agent")

from bedrock_agentcore.runtime import BedrockAgentCoreApp  # noqa: E402
from bedrock_agentcore.runtime.models import PingStatus  # noqa: E402

app = BedrockAgentCoreApp()

_warm_done = threading.Event()


def _preimport() -> None:
    """后台预热重依赖。失败只记日志，让首个请求把真实错误抛出来。"""
    t0 = time.time()
    try:
        import sample.rollout_handler  # noqa: F401

        logger.info("依赖预热完成，耗时 %.1fs", time.time() - t0)
    except Exception:
        logger.exception("依赖预热失败，首个请求会重新导入并抛出真实错误")
    finally:
        _warm_done.set()


threading.Thread(target=_preimport, name="preimport", daemon=True).start()


@app.ping
def ping_status() -> PingStatus:
    """预热未完成时报 HEALTHY_BUSY，表明本实例还在加载。

    枚举成员名是 UPPER_SNAKE（``HEALTHY`` / ``HEALTHY_BUSY``），而 ``.value``
    是 CamelCase（``Healthy`` / ``HealthyBusy``）。写成 ``PingStatus.HealthyBusy``
    会抛 AttributeError，``BedrockAgentCoreApp`` 会捕获它并回退到自动状态，
    因此这个错误不会让容器挂掉，只会在日志里出现
    "Custom ping handler failed, falling back to automatic"——很容易被忽略。

    注意 PingStatus 只有这两个取值，两者都算"健康"。因此 HEALTHY_BUSY 只是
    状态信号，**不能阻止 AgentCore 把请求路由过来**。真正缩短冷启动窗口的是
    让 uvicorn 尽早开始监听（见模块文档）。
    """
    return PingStatus.HEALTHY if _warm_done.is_set() else PingStatus.HEALTHY_BUSY


@app.entrypoint
def invoke(payload: dict[str, Any]) -> dict[str, Any]:
    """接收 MTRL 服务下发的 rollout 请求。

    payload 原样透传给 handle_rollout（``BedrockAgentCoreApp`` 的文档明确写
    "Invocation payloads are passed to the registered function unchanged"）。
    """
    meta = payload.get("metadata") or {}
    logger.info(
        "收到 rollout: keys=%s trajectory=%s 预热完成=%s",
        sorted(payload),
        meta.get("trajectory_id") or meta.get("rolloutId"),
        _warm_done.is_set(),
    )
    # 延迟导入：预热线程已经做过时这里是缓存命中；否则在 import 锁上等它完成。
    from sample.rollout_handler import handle_rollout

    try:
        return handle_rollout(payload)
    except Exception:
        # 打完整栈到 CloudWatch；异常仍然抛出，让 @sagemaker_rft_handler 报 error。
        logger.exception("rollout 处理失败")
        raise


if __name__ == "__main__":
    logger.info("启动 aftersales agent，监听 0.0.0.0:8080（依赖在后台预热）")
    app.run()
