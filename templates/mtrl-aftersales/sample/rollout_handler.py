"""MTRL rollout 处理函数：连接 SageMaker SDK 与业务 Agent 的唯一一层。

这一层刻意做薄，只负责四件事：

1. 从 rollout 报文里取出任务字符串（必要时解密）；
2. 构造指向 **Job Runtime Service** 的模型客户端，并注入 rollout 追踪头；
3. 调用与生产共用的 ``run_rollout``；
4. 返回 ``{"reward": ...}``，由 ``@sagemaker_rft_handler`` 负责回报。

关于装饰器的一个纠正：``@sagemaker_rft_handler`` **不会起 HTTP server**。
读 ``sagemaker/train/rft/decorators.py`` 可知它只做三件事——把 metadata 与
inferenceParams 写进 contextvar、在返回值里有 ``reward`` 时调用
``report_complete``、出异常时调用 ``report_error``。HTTP server 来自
AgentCore 的 ``BedrockAgentCoreApp``（见 ``entrypoint.py``）或你自己的 web 框架。

reward 可以是标量也可以是 list。返回 list 时装饰器走
``complete_rollout()`` + ``update_reward(list)``，即按轮 reward 通路，
对应超参 ``advantage_method="group_based_per_turn"``。
"""

from __future__ import annotations

import logging
import os
from typing import Any

from sagemaker.core.token_generator import generate_token
from sagemaker.train.rft import make_inference_headers, sagemaker_rft_handler

from sample.agent import (
    OpenAICompatClient,
    extract_task_string,
    parse_task,
    rollout_reward,
    run_rollout,
)

logger = logging.getLogger(__name__)

#: 训练时模型名由服务侧决定，官方示例用 "default"。
DEFAULT_MODEL_NAME = os.environ.get("ROLLOUT_MODEL", "default")
#: 设为 "1" 时回报按轮 reward 列表而不是标量。
PER_TURN_REWARD = os.environ.get("PER_TURN_REWARD") == "1"


def _decrypt_key() -> bytes | None:
    """从环境变量读取数据集解密密钥。

    生产中应改为从 Secrets Manager 取。密钥格式错误直接抛错，不静默按明文处理——
    把加密串当明文会得到一个"能跑但全错"的训练任务，而且训练指标上看不出来。
    """
    hex_key = os.environ.get("AFTERSALES_TASK_KEY_HEX")
    if not hex_key:
        return None
    key = bytes.fromhex(hex_key)
    if len(key) != 32:
        raise ValueError(f"AFTERSALES_TASK_KEY_HEX 应为 32 字节（64 hex 字符），实际 {len(key)} 字节")
    return key


def _resolve_endpoint(metadata: dict[str, Any]) -> str:
    endpoint = metadata.get("endpoint") or os.environ.get("RFT_RUNTIME_ENDPOINT")
    if not endpoint:
        raise RuntimeError(
            "rollout 报文的 metadata 里没有 endpoint，且未设置 RFT_RUNTIME_ENDPOINT。"
            " 无法确定 Job Runtime Service 地址。"
        )
    return endpoint.rstrip("/")


@sagemaker_rft_handler
def handle_rollout(payload: dict[str, Any]) -> dict[str, Any]:
    """处理一条 rollout 请求。

    Args:
        payload: 服务下发的 rollout 报文。

    Returns:
        含 ``reward`` 的字典；装饰器据此调用 CompleteRollout 与 UpdateReward。
    """
    metadata: dict[str, Any] = payload.get("metadata") or {}
    inference_params = payload.get("inferenceParams") or payload.get("inference_params")
    region = metadata.get("region") or os.environ.get("AWS_REGION") or "us-west-2"
    endpoint = _resolve_endpoint(metadata)

    # 追踪头不是可选项：服务靠它把每次推理关联到具体 job 与 trajectory。
    # 缺头不会报错，但轨迹记录不完整，MLflow 里也拼不出完整 trace。
    headers = make_inference_headers(metadata)
    if not headers:
        logger.warning("metadata 里没有 job_arn / trajectory_id，本条 rollout 的 trace 会不完整")

    token = generate_token(region=region)
    client = OpenAICompatClient(
        base_url=f"{endpoint}/v1",
        api_key=token,
        model=DEFAULT_MODEL_NAME,
        default_headers=headers,
    )

    task = parse_task(extract_task_string(payload), decrypt_key=_decrypt_key())
    result = run_rollout(task, client, inference_params=inference_params)

    reward = rollout_reward(result, per_turn=PER_TURN_REWARD)
    logger.info(
        "case=%s reward=%s decision=%s/%s coverage=%.2f",
        task["extra_info"].get("case_id"),
        reward,
        result.trace.decision.decision if result.trace.decision else None,
        result.ground_truth.decision,
        result.breakdown.metrics.get("required_tool_coverage", 0.0),
    )
    # metrics 只是给客户侧日志用；服务只读 reward。
    return {"reward": reward, "metrics": result.breakdown.metrics}
