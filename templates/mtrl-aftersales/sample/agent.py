"""售后裁决 Agent 的核心多轮循环。

这一层**不依赖 sagemaker SDK**，只依赖一个"能做 chat completion 的对象"。
这样同一份工具、状态管理和业务逻辑可以同时用于：

* 训练（模型客户端指向 Job Runtime Service，见 ``entrypoint.py``）；
* 生产（模型客户端指向生产推理端点）；
* 本地验证（模型客户端换成脚本化策略，见 ``local/policies.py``）。

单独维护一套训练用 Agent 是常见的坑：工具行为会慢慢和生产不一致，而这种
偏差往往要到部署后才暴露。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)

from sample.env import TERMINAL_TOOL, TOOL_SPECS, AfterSalesEnv, RolloutTrace
from sample.reward import RewardBreakdown, compute_per_turn_reward, compute_reward
from sample.rules import GroundTruth

DEFAULT_MAX_TURNS = 10


class ChatClient(Protocol):
    """模型客户端的最小接口。"""

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """返回一条 assistant 消息（OpenAI 格式，可能含 ``tool_calls``）。"""
        ...


@dataclass
class RolloutResult:
    trace: RolloutTrace
    breakdown: RewardBreakdown
    ground_truth: GroundTruth
    messages: list[dict[str, Any]]


# ---- payload / task 解析 -------------------------------------------------


def extract_task_string(payload: dict[str, Any]) -> str:
    """从 rollout 请求里取出任务字符串。

    这里同时兼容两种已观察到的报文形状，因为它们在官方材料里不一致：

    * SDK 契约（``sagemaker/train/rft/models.py`` 的 ``RolloutRequest``）：
      ``{"instance": {...}, "metadata": {...}, "inference_params": {...}}``，
      其中 ``instance`` 是数据文件里的一行，SDK docstring 用的是
      ``payload["instance"]["prompt"]``；
    * 公开文档（Preparing your Agent 页）：
      ``{"prompt": "string", "metadata": {...}, "inferenceParams": {...}}``。

    两者差异尚未确认是版本差异还是层级差异，因此这里做兼容而不是二选一。
    """
    inst = payload.get("instance")
    if isinstance(inst, dict):
        if "prompt" in inst:
            return _as_text(inst["prompt"])
        raise ValueError(f"instance 里没有 prompt 键，实际键为 {sorted(inst)}")
    if isinstance(inst, str):
        return inst
    if "prompt" in payload:
        return _as_text(payload["prompt"])
    raise ValueError(f"rollout 报文里找不到任务内容，顶层键为 {sorted(payload)}")


def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def parse_task(task_string: str, decrypt_key: bytes | None = None) -> dict[str, Any]:
    """把 prompt 列的字符串还原为任务对象。

    加密数据集必须显式传 ``decrypt_key``；解析失败直接抛错，不做静默兜底——
    静默把加密串当明文处理会得到一个"能跑但全错"的训练任务。
    """
    if decrypt_key is not None:
        from sample.dataset import decrypt_task

        task_string = decrypt_task(task_string, decrypt_key)
    task = json.loads(task_string)
    if not isinstance(task, dict):
        raise ValueError(f"任务对象应为 JSON object，实际为 {type(task).__name__}")
    for key in ("prompt", "reward_spec", "extra_info"):
        if key not in task:
            raise ValueError(f"任务对象缺少必需字段 {key!r}，实际字段 {sorted(task)}")
    return task


def ground_truth_of(task: dict[str, Any]) -> GroundTruth:
    gt = task["reward_spec"]["ground_truth"]
    return GroundTruth(
        decision=gt["decision"],
        amount=float(gt["amount"]),
        reason_code=gt["reason_code"],
        required_tools=tuple(gt["required_tools"]),
        rationale=task["extra_info"].get("rationale", ""),
    )


def normalize_inference_params(raw: dict[str, Any] | None) -> dict[str, Any]:
    """把服务下发的采样参数规范化。

    服务侧字段名有 camelCase 与 snake_case 两种写法，SDK 的 Strands adapter
    也是两种都试。**必须使用请求里下发的值**：Agent 沿用生产环境写死的采样
    参数不会报错，但实际采样分布会偏离训练配置，表现为收敛困难或无法复现。
    """
    if not raw:
        return {}
    out: dict[str, Any] = {}
    for snake, camel in (("temperature", "temperature"), ("max_tokens", "maxTokens"), ("top_p", "topP")):
        val = raw.get(snake)
        if val is None:
            val = raw.get(camel)
        if val is not None:
            out[snake] = val
    return out


# ---- 工具调用解析：原生 function calling + 文本协议回退 ------------------

#: 文本协议：模型把工具调用包在 <tool_call>...</tool_call> 里。
_TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

TOOL_NAMES = frozenset(
    {
        "get_order",
        "get_logistics",
        "get_return_policy",
        "get_customer_history",
        "check_inventory",
        "submit_decision",
    }
)


def _json_objects(text: str) -> list[str]:
    """按花括号配对扫描出所有顶层 JSON 对象候选。

    比正则可靠：工具参数本身是嵌套 JSON，正则很难正确处理任意深度的嵌套。
    """
    out: list[str] = []
    depth = 0
    start = -1
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    out.append(text[start : i + 1])
                    start = -1
    return out


def extract_text_tool_calls(content: str) -> list[dict[str, Any]]:
    """从自然语言输出里解析工具调用，作为原生 function calling 的回退。

    为什么需要这个：``Sample`` 接口兼容 OpenAI 格式，但**是否支持 ``tools``
    参数、以及模型是否真的产出 ``tool_calls``，取决于具体模型**。官方
    best practices 博客里 SOP-Bench 的 reward 解析的是 ``<final_output>``
    文本标签，说明那条链路走的是文本协议而不是原生 function calling。

    同时支持两种，可以避免"部署上去才发现模型不发 tool_calls"这类返工。
    解析顺序：``<tool_call>`` 块 → 内容里的裸 JSON 对象。都失败才算格式错误。
    """
    if not content:
        return []
    candidates = _TOOL_CALL_BLOCK.findall(content) or _json_objects(content)
    found: list[dict[str, Any]] = []
    for i, raw in enumerate(candidates):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name") or obj.get("tool")
        if name not in TOOL_NAMES:
            continue
        args = obj.get("arguments", obj.get("parameters", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        found.append(
            {
                "id": f"text_call_{i}",
                "type": "function",
                "function": {"name": name, "arguments": args if isinstance(args, dict) else {}},
            }
        )
    return found


# ---- 多轮循环 -----------------------------------------------------------


def run_rollout(
    task: dict[str, Any],
    client: ChatClient,
    inference_params: dict[str, Any] | None = None,
    max_turns: int | None = None,
    per_turn_reward: bool = False,
) -> RolloutResult:
    """执行一条完整 rollout：多轮工具调用直到提交裁决或达到轮数上限。"""
    extra = task["extra_info"]
    snapshot = extra["env_snapshot"]
    request_date = extra["request_date"]
    limit = max_turns or int(extra.get("max_turns", DEFAULT_MAX_TURNS))
    params = normalize_inference_params(inference_params)
    gt = ground_truth_of(task)

    messages: list[dict[str, Any]] = [dict(m) for m in task["prompt"]]

    with AfterSalesEnv(snapshot, request_date) as env:
        trace = env.trace
        for turn in range(limit):
            trace.turns = turn + 1
            assistant = client.chat(messages, TOOL_SPECS, params)
            messages.append(assistant)

            tool_calls = assistant.get("tool_calls") or []
            if not tool_calls:
                # 原生 function calling 没给结果时，尝试从文本里解析。
                tool_calls = extract_text_tool_calls(assistant.get("content") or "")
                if tool_calls:
                    trace.text_protocol_turns += 1
            if not tool_calls:
                # 两种协议都解析不出动作：视为格式错误，不再继续消耗轮数。
                trace.malformed_output = True
                trace.error = "assistant 消息里既没有 tool_calls，也无法从文本解析出工具调用"
                break

            terminal = False
            for tc in tool_calls:
                name, args, parse_err = _parse_tool_call(tc)
                if parse_err is not None:
                    trace.invalid_tool_calls += 1
                    messages.append(
                        _tool_message(tc, json.dumps({"error": parse_err}, ensure_ascii=False))
                    )
                    continue
                observation = env.call(name, args, turn)
                messages.append(_tool_message(tc, observation))
                if name == TERMINAL_TOOL and trace.decision is not None:
                    terminal = True
            if terminal:
                break

        breakdown = compute_reward(trace, gt)
        if per_turn_reward:
            breakdown.metrics["per_turn_reward"] = 1.0
        return RolloutResult(trace=trace, breakdown=breakdown, ground_truth=gt, messages=messages)


def rollout_reward(result: RolloutResult, per_turn: bool = False) -> float | list[float]:
    """取出要回报给服务的 reward。

    ``per_turn=True`` 时返回 list，由 SDK 走 ``update_reward(list)`` 通路。
    """
    if per_turn:
        return compute_per_turn_reward(result.trace, result.ground_truth)
    return result.breakdown.reward


def _parse_tool_call(tc: dict[str, Any]) -> tuple[str, dict[str, Any], str | None]:
    """解析一个 tool_call，返回 (name, args, error)。"""
    fn = tc.get("function") or {}
    name = fn.get("name")
    if not isinstance(name, str) or not name:
        return "", {}, "tool_call 缺少 function.name"
    raw_args = fn.get("arguments", "{}")
    if isinstance(raw_args, dict):
        return name, raw_args, None
    try:
        args = json.loads(raw_args or "{}")
    except json.JSONDecodeError as e:
        return name, {}, f"arguments 不是合法 JSON: {e}"
    if not isinstance(args, dict):
        return name, {}, f"arguments 应为 JSON object，实际为 {type(args).__name__}"
    return name, args, None


def _tool_message(tc: dict[str, Any], content: str) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": tc.get("id", "call_0"),
        "name": (tc.get("function") or {}).get("name", ""),
        "content": content,
    }


# ---- 生产 / 训练共用的 OpenAI 兼容客户端 -------------------------------


class OpenAICompatClient:
    """OpenAI 兼容的 chat 客户端。

    训练时把 ``base_url`` 指向 Job Runtime Service 并注入 rollout 追踪头；
    生产时指向生产推理端点。除这两处外逻辑完全一致。
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "default",
        default_headers: dict[str, str] | None = None,
        timeout: float = 120.0,
    ) -> None:
        import openai  # 延迟导入：本地验证路径不需要装 openai

        self._model = model
        #: 一旦发现服务不接受 tools，就在本进程内改走文本协议。
        self._tools_unsupported = False
        self._client = openai.OpenAI(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            default_headers=default_headers or {},
            timeout=timeout,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """发一次推理请求。

        ``tools`` 参数不一定被支持：``Sample`` 兼容 OpenAI 格式，但具体模型是否
        接受 function calling 并不保证。首次请求带 ``tools``；如果服务以 4xx 拒绝
        （报文里通常提到 tools / function / unsupported），就记下来并从此改用纯
        文本协议——system prompt 里已经写了 ``<tool_call>`` 的格式，Agent 仍能工作。

        这样做的理由很直接：一个训练任务的并发配额是 1，如果因为 ``tools``
        不被支持导致每条 rollout 都报错，整个任务的结果就废了，而且要等任务
        清理才能重试。宁可自动降级并在日志里留痕。
        """
        kwargs: dict[str, Any] = {"model": self._model, "messages": messages, **params}
        if tools and not self._tools_unsupported:
            kwargs["tools"] = tools

        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            if "tools" in kwargs and self._looks_like_tools_rejection(e):
                logger.warning(
                    "服务拒绝了 tools 参数，本进程后续改用文本协议：%s", str(e)[:300]
                )
                self._tools_unsupported = True
                kwargs.pop("tools")
                resp = self._client.chat.completions.create(**kwargs)
            else:
                raise

        msg = resp.choices[0].message
        out: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ]
        return out

    #: 报文里必须同时命中"提到了 tools/function"和"表达了不支持/非法"两类词，
    #: 才判定为服务不接受 tools。只用单边关键词会误判：实测中一条
    #: "unknown policy 'default'; 可选 [... 'tool_spam']" 的 400 就因为含子串
    #: "tool" 被误判成能力缺失，白重试了一次。
    _TOOLS_SUBJECT = ("tools", "tool_calls", "tool_choice", "functions", "function_call")
    _REJECTION_VERB = (
        "unsupported", "not supported", "unrecognized", "unexpected keyword",
        "unknown parameter", "unknown field", "invalid parameter", "does not support",
        "extra inputs are not permitted", "additional properties",
    )

    @classmethod
    def _looks_like_tools_rejection(cls, exc: Exception) -> bool:
        """判断异常是否是"服务不接受 tools 参数"。

        判定条件收得比较紧，宁可不降级也不误降级：

        * HTTP 状态必须是 4xx——5xx 与网络错误照常抛出，否则会把服务故障
          误判成能力缺失，掩盖真正的问题；
        * 报文里要同时提到 tools/function 这个主体，和"不支持/非法"这类判词。

        降级本身是有代价的：一旦误判，后续所有轮都不再发 tools，即使服务其实
        支持。所以宁可漏判（照常抛错、任务失败、人来看日志），也不要误判。
        """
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if isinstance(status, str) and status.isdigit():
            status = int(status)
        if isinstance(status, int) and not (400 <= status < 500):
            return False
        text = str(exc).lower()
        return any(s in text for s in cls._TOOLS_SUBJECT) and any(
            v in text for v in cls._REJECTION_VERB
        )
