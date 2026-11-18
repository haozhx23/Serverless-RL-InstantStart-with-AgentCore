"""脚本化策略：用来在本地验证 reward 函数是否真的区分不同质量的轨迹。

这些不是模型，而是行为固定的假策略。它们的意义在于：**reward 函数是客户
自己写的代码，必须在花钱训练之前证明它区分得开好轨迹和坏轨迹。**
每个策略对应一种真实会出现的失败模式。

============================  ==================================================
策略                          模拟的失败模式
============================  ==================================================
``perfect``                   完全按 SOP 执行（同时验证工具返回的信息足够解题）
``skip_logistics``            不查物流，用下单日期算窗口——最常见的错误
``blind_exchange``            质量问题一律给换货，不查库存
``always_reject``             把所有必查工具都调一遍，然后恒判 reject（刷流程分探测）
``tool_spam``                 反复调用同一工具后才提交（探测效率惩罚）
``double_submit``             正确裁决但提交两次
``malformed``                 只输出自然语言，不发工具调用
============================  ==================================================
"""

from __future__ import annotations

import json
import re
from typing import Any

from sample.env import days_between
from sample.rules import decide

_ORDER_RE = re.compile(r"订单号：(\S+)")
_DATE_RE = re.compile(r"申请日期：(\d{4}-\d{2}-\d{2})")


def _parse_user(messages: list[dict[str, Any]]) -> tuple[str, str, str]:
    """从 user 消息里取出 order_id、申请日期、claim_type。"""
    text = ""
    for m in messages:
        if m.get("role") == "user":
            text = m.get("content") or ""
            break
    om = _ORDER_RE.search(text)
    dm = _DATE_RE.search(text)
    if not om or not dm:
        raise ValueError("user 消息里缺少订单号或申请日期")
    claim = "quality" if "质量问题" in text else "no_reason"
    return om.group(1), dm.group(1), claim


def _observations(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """把已有的 tool 返回按工具名收集起来。"""
    obs: dict[str, Any] = {}
    for m in messages:
        if m.get("role") != "tool":
            continue
        name = m.get("name") or ""
        try:
            payload = json.loads(m.get("content") or "{}")
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "error" not in payload:
            obs[name] = payload
    return obs


def _call(name: str, args: dict[str, Any], idx: int = 0) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"call_{name}_{idx}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            }
        ],
    }


def _submit(decision: str, amount: float, reason: str, idx: int = 0) -> dict[str, Any]:
    return _call(
        "submit_decision",
        {"decision": decision, "amount": round(float(amount), 2), "reason_code": reason},
        idx,
    )


class BasePolicy:
    """按会话历史决定下一步动作。"""

    name = "base"

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        params: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError


class PerfectPolicy(BasePolicy):
    """严格按 SOP 逐步核实。裁决用 rules.decide 从**工具返回值**推出。

    这同时验证了一个容易被忽略的前提：工具暴露的信息足以解出任务。
    如果这个策略拿不到满分，说明工具设计有缺口，不是模型的问题。
    """

    name = "perfect"

    def chat(self, messages, tools, params):  # noqa: ANN001, D102
        order_id, request_date, claim = _parse_user(messages)
        obs = _observations(messages)

        if "get_order" not in obs:
            return _call("get_order", {"order_id": order_id})
        order = obs["get_order"]

        if "get_customer_history" not in obs:
            return _call("get_customer_history", {"customer_id": order["customer_id"]})
        history = obs["get_customer_history"]
        if history["abuse_flag"]:
            return _submit("escalate", 0.0, "ABUSE_REVIEW")

        if "get_logistics" not in obs:
            return _call("get_logistics", {"order_id": order_id})
        logistics = obs["get_logistics"]
        refund_amount = round(float(order["order_amount"]) - float(order["coupon_amount"]), 2)
        if logistics["delivery_status"] != "delivered":
            return _submit("escalate", 0.0, "NOT_DELIVERED")
        if logistics["damage_reported"]:
            return _submit("refund", refund_amount, "DAMAGED_IN_TRANSIT")

        if "get_return_policy" not in obs:
            return _call("get_return_policy", {"category": order["category"]})
        policy = obs["get_return_policy"]

        days = days_between(logistics["delivered_at"], request_date)
        need_inventory = (
            claim == "quality"
            and days <= int(policy["quality_window_days"])
            and bool(policy["exchange_supported"])
        )
        if need_inventory and "check_inventory" not in obs:
            return _call("check_inventory", {"sku": order["sku"]})
        inventory = obs.get("check_inventory", {"available_qty": 0})

        gt = decide(order, logistics, policy, history, inventory, claim, request_date)
        return _submit(gt.decision, gt.amount, gt.reason_code)


class SkipLogisticsPolicy(PerfectPolicy):
    """不查物流，改用下单日期算窗口——多轮任务里最常见的错误。

    因为 ordered_at 比 delivered_at 早 2~9 天，一部分样本会因此跨出窗口边界，
    得到与标准答案不同的裁决。同时 required_tool_coverage 会掉。
    """

    name = "skip_logistics"

    def chat(self, messages, tools, params):  # noqa: ANN001, D102
        order_id, request_date, claim = _parse_user(messages)
        obs = _observations(messages)

        if "get_order" not in obs:
            return _call("get_order", {"order_id": order_id})
        order = obs["get_order"]
        if "get_customer_history" not in obs:
            return _call("get_customer_history", {"customer_id": order["customer_id"]})
        history = obs["get_customer_history"]
        if history["abuse_flag"]:
            return _submit("escalate", 0.0, "ABUSE_REVIEW")
        if "get_return_policy" not in obs:
            return _call("get_return_policy", {"category": order["category"]})
        policy = obs["get_return_policy"]

        # 错误所在：用 ordered_at 而不是 delivered_at
        days = days_between(order["ordered_at"], request_date)
        refund_amount = round(float(order["order_amount"]) - float(order["coupon_amount"]), 2)
        window = int(policy["quality_window_days"] if claim == "quality" else policy["no_reason_window_days"])
        if days <= window:
            reason = "QUALITY_DEFECT" if claim == "quality" else "NO_REASON_RETURN"
            return _submit("refund", refund_amount, reason)
        return _submit("reject", 0.0, "WINDOW_EXPIRED")


class BlindExchangePolicy(PerfectPolicy):
    """质量问题一律给换货，不查库存。"""

    name = "blind_exchange"

    def chat(self, messages, tools, params):  # noqa: ANN001, D102
        order_id, request_date, claim = _parse_user(messages)
        obs = _observations(messages)
        if "get_order" not in obs:
            return _call("get_order", {"order_id": order_id})
        order = obs["get_order"]
        if "get_customer_history" not in obs:
            return _call("get_customer_history", {"customer_id": order["customer_id"]})
        if obs["get_customer_history"]["abuse_flag"]:
            return _submit("escalate", 0.0, "ABUSE_REVIEW")
        if "get_logistics" not in obs:
            return _call("get_logistics", {"order_id": order_id})
        if claim == "quality":
            return _submit("exchange", 0.0, "QUALITY_EXCHANGE")
        if "get_return_policy" not in obs:
            return _call("get_return_policy", {"category": order["category"]})
        refund_amount = round(float(order["order_amount"]) - float(order["coupon_amount"]), 2)
        return _submit("refund", refund_amount, "NO_REASON_RETURN")


class AlwaysRejectPolicy(BasePolicy):
    """把 5 个读工具全调一遍，然后恒判 reject。

    这是探针：如果它能拿到接近满分的 reward，说明 reward 设计有漏洞
    （流程分可以脱离结果正确性单独刷满，或数据集裁决分布不均衡）。
    """

    name = "always_reject"
    _ORDER = ["get_order", "get_customer_history", "get_logistics", "get_return_policy", "check_inventory"]

    def chat(self, messages, tools, params):  # noqa: ANN001, D102
        order_id, _, _ = _parse_user(messages)
        obs = _observations(messages)
        order = obs.get("get_order")
        for name in self._ORDER:
            if name in obs:
                continue
            if name == "get_order":
                return _call(name, {"order_id": order_id})
            if order is None:
                break
            if name == "get_customer_history":
                return _call(name, {"customer_id": order["customer_id"]})
            if name == "get_logistics":
                return _call(name, {"order_id": order_id})
            if name == "get_return_policy":
                return _call(name, {"category": order["category"]})
            if name == "check_inventory":
                return _call(name, {"sku": order["sku"]})
        return _submit("reject", 0.0, "WINDOW_EXPIRED")


class ToolSpamPolicy(PerfectPolicy):
    """先把 get_order 重复调 3 次，再正常走流程——探测效率惩罚是否生效。"""

    name = "tool_spam"

    def chat(self, messages, tools, params):  # noqa: ANN001, D102
        order_id, _, _ = _parse_user(messages)
        repeats = sum(
            1
            for m in messages
            if m.get("role") == "assistant"
            and any(
                (tc.get("function") or {}).get("name") == "get_order"
                for tc in (m.get("tool_calls") or [])
            )
        )
        if repeats < 3:
            return _call("get_order", {"order_id": order_id}, idx=repeats)
        return super().chat(messages, tools, params)


class DoubleSubmitPolicy(PerfectPolicy):
    """裁决正确，但在同一条消息里发两个 submit_decision——探测重复提交惩罚。

    环境在首次提交落库后就终止循环，所以重复提交只能通过一条 assistant 消息
    里的并行 tool_calls 出现。这也正是真实模型会犯的错。
    """

    name = "double_submit"

    def chat(self, messages, tools, params):  # noqa: ANN001, D102
        msg = super().chat(messages, tools, params)
        calls = msg.get("tool_calls") or []
        if calls and calls[0]["function"]["name"] == "submit_decision":
            first = calls[0]
            second = {
                "id": "call_submit_decision_dup",
                "type": "function",
                # 第二次给一个不同的答案，模拟"多押一注"
                "function": {
                    "name": "submit_decision",
                    "arguments": json.dumps(
                        {"decision": "escalate", "amount": 0, "reason_code": "ABUSE_REVIEW"},
                        ensure_ascii=False,
                    ),
                },
            }
            msg = dict(msg)
            msg["tool_calls"] = [first, second]
        return msg


class TextProtocolPolicy(PerfectPolicy):
    """裁决正确，但不发原生 tool_calls，只输出 <tool_call> 文本块。

    用来验证文本协议回退路径。真实模型是否发原生 tool_calls 取决于模型本身，
    这条回退是为了避免"部署上去才发现不通"。
    """

    name = "text_protocol"

    def chat(self, messages, tools, params):  # noqa: ANN001, D102
        msg = super().chat(messages, tools, params)
        calls = msg.get("tool_calls") or []
        if not calls:
            return msg
        fn = calls[0]["function"]
        args = fn["arguments"]
        if isinstance(args, str):
            args = json.loads(args)
        block = json.dumps({"name": fn["name"], "arguments": args}, ensure_ascii=False)
        return {
            "role": "assistant",
            "content": f"我先核实一下。\n<tool_call>{block}</tool_call>",
        }


class MalformedPolicy(BasePolicy):
    """只输出自然语言，不发工具调用。"""

    name = "malformed"

    def chat(self, messages, tools, params):  # noqa: ANN001, D102
        return {"role": "assistant", "content": "我认为应该给客户退款，建议按流程处理。"}


POLICIES: dict[str, BasePolicy] = {
    p.name: p
    for p in (
        PerfectPolicy(),
        SkipLogisticsPolicy(),
        BlindExchangePolicy(),
        AlwaysRejectPolicy(),
        ToolSpamPolicy(),
        DoubleSubmitPolicy(),
        TextProtocolPolicy(),
        MalformedPolicy(),
    )
}
