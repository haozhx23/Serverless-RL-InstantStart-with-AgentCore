"""AgentCore Runtime 预检：起训之前先确认容器这一侧是通的。

为什么值得单独做一步：MTRL 训练任务的并发配额是 1，一次失败要等任务清理，
排查成本高。直接 invoke_agent_runtime 送一条合成 rollout 报文，可以在不花
训练费用的前提下验证六件事：

1. 容器能启动，``/invocations`` 与 ``/ping`` 响应正常；
2. AgentCore 把 payload 原样透传给了 entrypoint；
3. 任务字符串能从 ``instance.prompt`` 解析出来；
4. 环境快照能重建成 in-memory SQLite；
5. 运行角色能签出 bearer token（IAM 与信任策略正确）；
6. 到达"调用推理端点"这一步。

用法::

    python deploy/smoke_test_runtime.py --runtime-arn <ARN>            # 预期在推理调用处失败
    python deploy/smoke_test_runtime.py --runtime-arn <ARN> --endpoint https://real-endpoint

不带 --endpoint 时故意指向一个不可达地址：**能走到连接失败**就说明前 5 步都对了。
这一步"失败"是预期结果，不代表部署有问题；真正的失败信号是更早的报错
（payload 解析、任务解析、token 生成）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import boto3

from sample.dataset import generate_cases

UNREACHABLE = "https://job-runtime-smoke-test.invalid"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runtime-arn", required=True)
    p.add_argument("--region", default="us-west-2")
    p.add_argument("--endpoint", default=UNREACHABLE, help="Job Runtime Service 端点")
    p.add_argument("--case-seed", type=int, default=101)
    args = p.parse_args()

    case = generate_cases(1, seed=args.case_seed)[0]
    payload = {
        "instance": {"prompt": json.dumps(case, ensure_ascii=False, separators=(",", ":"))},
        "metadata": {
            "job_arn": f"arn:aws:sagemaker:{args.region}:000000000000:job/smoke-test",
            "trajectory_id": f"smoke-{uuid.uuid4().hex[:8]}",
            "endpoint": args.endpoint,
            "region": args.region,
        },
        "inferenceParams": {"temperature": 1.0, "topP": 1.0, "maxTokens": 512},
    }

    client = boto3.client("bedrock-agentcore", region_name=args.region)
    # 日志按这个时间点之后取，避免把上一次调用的日志当成本次结果——
    # 两者结论可能完全不同，实测中就出现过分类命中了 10 分钟前的旧日志。
    invoke_started_ms = int(time.time() * 1000) - 5_000
    print(f"invoke {args.runtime_arn}")
    print(f"  case={case['extra_info']['case_id']} "
          f"标准答案={case['reward_spec']['ground_truth']['decision']} "
          f"endpoint={args.endpoint}")
    try:
        resp = client.invoke_agent_runtime(
            agentRuntimeArn=args.runtime_arn,
            runtimeSessionId=uuid.uuid4().hex + "smoke",
            payload=json.dumps(payload).encode("utf-8"),
        )
    except Exception as e:
        text = str(e)
        print(f"\ninvoke_agent_runtime 报错：{type(e).__name__}: {text}")
        # AgentCore 把容器内的未捕获异常包装成 "Received error (500) from runtime"。
        # 这种情况下问题在容器**内部**，必须看 CloudWatch 才知道走到了哪一步——
        # 只看这条报错会误判成 IAM 或 runtime 状态问题。
        if "500" in text or "from runtime" in text:
            print("\n容器内抛了未捕获异常。取本次调用的 CloudWatch 日志：")
            logs_text = _wait_for_logs(args.runtime_arn, args.region, invoke_started_ms)
            for line in logs_text.splitlines()[-25:]:
                print(f"    {line.rstrip()}")
            return _classify(logs_text, args.runtime_arn, args.region, args.endpoint)
        print("失败发生在容器之外（IAM / runtime 状态 / 报文大小），与 agent 代码无关。")
        return 1

    body = resp.get("response")
    raw = body.read() if hasattr(body, "read") else body
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
    print(f"\nHTTP status: {resp.get('statusCode')}")
    print(f"响应:\n{text[:2000]}")

    lowered = text.lower()
    if "reward" in lowered:
        print("\n结论：agent 跑完了整条 rollout 并算出了 reward。")
        return 0
    if any(k in lowered for k in ("connection", "resolve", "getaddrinfo", "timeout", "invalid")):
        print(
            "\n结论：agent 走到了调用推理端点这一步才失败——这是 --endpoint 不可达时的"
            "预期结果，说明容器、报文透传、任务解析、环境重建、token 签发都正常。"
        )
        return 0
    print("\n结论：失败发生在推理调用之前，需要看 CloudWatch 日志定位。")
    return 2


def _log_group(runtime_arn: str) -> str:
    """AgentCore 的日志组命名规则：/aws/bedrock-agentcore/runtimes/<runtimeId>-DEFAULT。"""
    return f"/aws/bedrock-agentcore/runtimes/{runtime_arn.rsplit('/', 1)[-1]}-DEFAULT"


def _fetch_logs(runtime_arn: str, region: str, start_ms: int) -> str:
    """取 ``start_ms`` 之后的日志。取不到返回空串，不掩盖原始报错。"""
    logs = boto3.client("logs", region_name=region)
    group = _log_group(runtime_arn)
    try:
        streams = logs.describe_log_streams(
            logGroupName=group, orderBy="LastEventTime", descending=True, limit=3
        )["logStreams"]
    except Exception as e:
        print(f"  （读不到日志组 {group}: {type(e).__name__}）")
        return ""
    chunks: list[str] = []
    for st in streams:
        try:
            events = logs.get_log_events(
                logGroupName=group,
                logStreamName=st["logStreamName"],
                startTime=start_ms,
                startFromHead=True,
                limit=300,
            )["events"]
        except Exception:
            continue
        chunks.extend(ev.get("message", "") for ev in events)
    return "\n".join(chunks)


def _wait_for_logs(runtime_arn: str, region: str, start_ms: int, attempts: int = 10) -> str:
    """轮询直到日志里出现可判定的标志，或次数用尽。

    只等到"有任何日志"是不够的：容器启动行会先落地，traceback 晚几秒才到。
    见到启动行就返回，会让分类器判成"未识别的失败模式"。
    """
    text = ""
    for i in range(attempts):
        text = _fetch_logs(runtime_arn, region, start_ms)
        if any(marker in text.lower() for marker, _, _ in _STAGES):
            return text
        if i < attempts - 1:
            time.sleep(5)
    if not text.strip():
        print("  （本次调用的日志尚未投递到 CloudWatch）")
    else:
        print("  （等到超时仍未出现可判定的标志，下面是已拿到的日志）")
    return text


#: 校验链上的失败标志，顺序即先后：越靠前说明问题越基础。
#: advice 为 None 表示"这是预期的最远失败点"，即前序环节全部正常。
_STAGES: list[tuple[str, str | None, int]] = [
    ("modulenotfounderror",
     "容器缺依赖。开发机的 site-packages 会掩盖这个问题，"
     "必须在目标架构的容器里跑 import 冒烟测试。", 2),
    ("找不到任务内容",
     "rollout 报文形状与预期不符，检查 extract_task_string 的兼容分支。", 2),
    ("任务对象缺少必需字段",
     "prompt 列的任务对象结构不对，或加密数据集没传解密密钥。", 2),
    ("no aws credentials",
     "运行角色拿不到凭证，检查 bedrock-agentcore.amazonaws.com 信任策略。", 2),
    ("apiconnectionerror", None, 0),
    ("connection error", None, 0),
]


def _classify(logs_text: str, runtime_arn: str, region: str, endpoint: str) -> int:
    """按日志判断失败落在哪一步，给出可操作的结论。"""
    text = logs_text.lower()
    if not text.strip():
        print("\n拿不到本次调用的日志，无法判断失败位置。手动查看：")
        print(f"  aws logs tail {_log_group(runtime_arn)} --region {region} --since 15m")
        return 2

    for marker, advice, rc in _STAGES:
        if marker in text:
            if advice is None:
                print(
                    f"\n结论：agent 走到了调用推理端点这一步才失败。endpoint={endpoint} "
                    "不可达时这是**预期结果**——说明容器启动、报文透传、任务解析、"
                    "环境重建、bearer token 签发都正常。"
                )
                return 0
            print(f"\n结论：失败在推理调用之前。{advice}")
            return rc
    print("\n结论：日志里没有识别到已知失败模式，需要人工判读上面的输出。")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
