"""本地假 Job Runtime Service，用于在不动 AWS 资源的前提下跑通整条 rollout。

它实现真实服务的三个端点：

* ``POST /v1/chat/completions`` —— OpenAI 兼容推理。用 ``model`` 字段选择
  ``local/policies.py`` 里的脚本化策略，从而产出指定质量的轨迹。
* ``POST /complete-rollout`` —— 记录 SDK 报上来的 ``TrajectoryId`` 与 ``Status``。
* ``POST /update-reward`` —— 记录 SDK 报上来的 ``Rewards`` 数组。

后两个路径与请求体字段名取自 SDK 实现
（``sagemaker/train/rft/feedback.py``：``/complete-rollout``、``/update-reward``，
字段 ``JobArn`` / ``TrajectoryId`` / ``Status`` / ``Rewards``），
所以本地跑通意味着真 SDK 的回报通路被真正执行过，而不是被 mock 掉。

服务不校验 bearer token（真实服务会校验），其余报文结构保持一致。
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from local.policies import POLICIES


class Recorder:
    """线程安全地记录服务收到的回报。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.completions: list[dict[str, Any]] = []
        self.rewards: list[dict[str, Any]] = []
        self.inference_calls = 0
        self.seen_headers: list[dict[str, str]] = []

    def add_completion(self, body: dict[str, Any]) -> None:
        with self._lock:
            self.completions.append(body)

    def add_reward(self, body: dict[str, Any]) -> None:
        with self._lock:
            self.rewards.append(body)

    def add_inference(self, headers: dict[str, str]) -> None:
        with self._lock:
            self.inference_calls += 1
            self.seen_headers.append(headers)

    def reset(self) -> None:
        with self._lock:
            self.completions.clear()
            self.rewards.clear()
            self.seen_headers.clear()
            self.inference_calls = 0


class _Handler(BaseHTTPRequestHandler):
    recorder: Recorder  # 由 make_server 注入

    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # 静音访问日志
        pass

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw or b"{}")

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            # 真实服务要求客户 rollout server 提供 /health；这里对称提供一个。
            self._send(200, {"status": "healthy"})
        else:
            self._send(404, {"error": f"unknown path {self.path}"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            body = self._read_json()
        except json.JSONDecodeError as e:
            self._send(400, {"error": f"invalid json: {e}"})
            return

        if self.path.rstrip("/").endswith("/chat/completions"):
            self._handle_inference(body)
        elif self.path.rstrip("/") == "/complete-rollout":
            self.recorder.add_completion(body)
            self._send(200, {"ok": True})
        elif self.path.rstrip("/") == "/update-reward":
            self.recorder.add_reward(body)
            self._send(200, {"ok": True})
        else:
            self._send(404, {"error": f"unknown path {self.path}"})

    def _handle_inference(self, body: dict[str, Any]) -> None:
        self.recorder.add_inference(
            {
                k: v
                for k, v in self.headers.items()
                if k.lower().startswith("x-amzn-") or k.lower() == "authorization"
            }
        )
        # 真实训练里模型名由服务决定（官方示例用 "default"），本地用它选策略。
        # 因此 "default" 映射到 MOCK_DEFAULT_POLICY，默认 perfect。
        policy_name = body.get("model") or "perfect"
        if policy_name in ("default", "", None):
            policy_name = os.environ.get("MOCK_DEFAULT_POLICY", "perfect")
        policy = POLICIES.get(policy_name)
        if policy is None:
            self._send(400, {"error": f"unknown policy {policy_name!r}; 可选 {sorted(POLICIES)}"})
            return

        messages = body.get("messages") or []
        tools = body.get("tools") or []
        params = {k: body[k] for k in ("temperature", "max_tokens", "top_p") if k in body}
        try:
            assistant = policy.chat(messages, tools, params)
        except Exception as e:  # 让本地失败明确可见
            self._send(500, {"error": f"policy {policy_name} failed: {type(e).__name__}: {e}"})
            return

        message: dict[str, Any] = {"role": "assistant", "content": assistant.get("content") or ""}
        if assistant.get("tool_calls"):
            message["tool_calls"] = assistant["tool_calls"]

        self._send(
            200,
            {
                "id": "chatcmpl-local",
                "object": "chat.completion",
                "created": 0,
                "model": policy_name,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
        )


def make_server(port: int = 0, host: str = "127.0.0.1") -> tuple[ThreadingHTTPServer, Recorder]:
    """启动 mock 服务。``port=0`` 表示由系统分配空闲端口。

    ``host="0.0.0.0"`` 用于让容器内的 Agent 通过 docker bridge 网关访问，
    从而在本地就能用**实际要部署的容器镜像**跑完整 rollout。
    """
    recorder = Recorder()
    handler = type("BoundHandler", (_Handler,), {"recorder": recorder})
    server = ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, recorder


def endpoint_of(server: ThreadingHTTPServer) -> str:
    host, port = server.server_address[:2]
    return f"http://{host}:{port}"


if __name__ == "__main__":
    srv, rec = make_server(8765)
    print(f"mock Job Runtime Service 已启动：{endpoint_of(srv)}")
    print(f"可用策略：{sorted(POLICIES)}")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        srv.shutdown()
