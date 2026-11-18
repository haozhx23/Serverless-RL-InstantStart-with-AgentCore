"""售后裁决 SOP 的可执行版本：标准答案与"必查工具"的唯一真源。

把 SOP 写成代码而不是写在文档里，有三个目的：

1. 数据生成器用它算 ground truth，reward 用它算合规项，两边不会漂移；
2. "必查工具"不是人工列的清单，而是从裁决路径推导出来的——只有在
   某个工具的返回值确实会改变结论时，它才是必查的；
3. 评估集与训练集共用同一份规则，避免出现官方 best practices 博客里
   "任务示例、标准答案字段和评估标签不一致"那类问题。

SOP 的判定顺序（顺序本身就是考点）：

    1. 客户在售后滥用名单        -> escalate / ABUSE_REVIEW
    2. 订单未签收                -> escalate / NOT_DELIVERED
    3. 运输破损已上报            -> refund   / DAMAGED_IN_TRANSIT
    4. 质量问题且在质量窗口内：
         支持换货且有库存        -> exchange / QUALITY_EXCHANGE
         否则                    -> refund   / QUALITY_DEFECT
    5. 无理由退货且在无理由窗口内 -> refund   / NO_REASON_RETURN
    6. 其余                      -> reject   / WINDOW_EXPIRED

退款金额一律是 ``order_amount - coupon_amount``（券不退），其余场景为 0。

关键陷阱：所有窗口都从 **签收日期**（``logistics.delivered_at``）起算，
不是下单日期。数据生成器会刻意让两者相差数天，使"按下单日期算"在部分
样本上得出不同结论——这样 reward 才能区分"查了物流"和"没查物流"。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sample.env import days_between


@dataclass(frozen=True)
class GroundTruth:
    decision: str
    amount: float
    reason_code: str
    required_tools: tuple[str, ...]
    #: 便于人工核对与排错，不参与打分。
    rationale: str


def _refund_amount(order: dict[str, Any]) -> float:
    return round(float(order["order_amount"]) - float(order["coupon_amount"]), 2)


def decide(
    order: dict[str, Any],
    logistics: dict[str, Any],
    policy: dict[str, Any],
    history: dict[str, Any],
    inventory: dict[str, Any],
    claim_type: str,
    request_date: str,
) -> GroundTruth:
    """按 SOP 计算标准答案，同时推导必查工具集合。

    Args:
        claim_type: ``"quality"``（质量问题）或 ``"no_reason"``（七天无理由）。
        request_date: 客户提交申请的日期（ISO 日期串）。
    """
    if claim_type not in ("quality", "no_reason"):
        raise ValueError(f"claim_type 必须是 quality 或 no_reason，收到 {claim_type!r}")

    # get_order 提供 category / customer_id / 金额，任何路径都绕不开；
    # get_customer_history 决定最高优先级的 abuse 分支，也必须调。
    required: list[str] = ["get_order", "get_customer_history"]

    if history["abuse_flag"]:
        return GroundTruth(
            "escalate", 0.0, "ABUSE_REVIEW", tuple(required),
            "客户在售后滥用名单，优先级最高，直接转人工",
        )

    # 走到这里必须确认签收状态与破损情况，因此 get_logistics 是必查的。
    required.append("get_logistics")

    if logistics["delivery_status"] != "delivered":
        return GroundTruth(
            "escalate", 0.0, "NOT_DELIVERED", tuple(required),
            f"物流状态为 {logistics['delivery_status']}，未签收不进入退换货流程",
        )

    if logistics["damage_reported"]:
        return GroundTruth(
            "refund", _refund_amount(order), "DAMAGED_IN_TRANSIT", tuple(required),
            "已上报运输破损，全额退款且不受窗口限制",
        )

    # 需要窗口判断，政策必查；窗口从签收日算起。
    required.append("get_return_policy")
    days = days_between(logistics["delivered_at"], request_date)

    if claim_type == "quality":
        if days <= int(policy["quality_window_days"]):
            if policy["exchange_supported"]:
                # 换货与退款的分叉由库存决定，此时 check_inventory 才是必查的。
                required.append("check_inventory")
                if int(inventory["available_qty"]) > 0:
                    return GroundTruth(
                        "exchange", 0.0, "QUALITY_EXCHANGE", tuple(required),
                        f"签收后 {days} 天，在质量窗口 {policy['quality_window_days']} 天内，"
                        f"支持换货且库存 {inventory['available_qty']} 件",
                    )
                return GroundTruth(
                    "refund", _refund_amount(order), "QUALITY_DEFECT", tuple(required),
                    f"签收后 {days} 天，在质量窗口内，支持换货但无库存，转退款",
                )
            return GroundTruth(
                "refund", _refund_amount(order), "QUALITY_DEFECT", tuple(required),
                f"签收后 {days} 天，在质量窗口内，该类目不支持换货，直接退款",
            )
        return GroundTruth(
            "reject", 0.0, "WINDOW_EXPIRED", tuple(required),
            f"签收后 {days} 天，已超出质量窗口 {policy['quality_window_days']} 天",
        )

    # claim_type == "no_reason"
    if days <= int(policy["no_reason_window_days"]):
        return GroundTruth(
            "refund", _refund_amount(order), "NO_REASON_RETURN", tuple(required),
            f"签收后 {days} 天，在无理由窗口 {policy['no_reason_window_days']} 天内",
        )
    return GroundTruth(
        "reject", 0.0, "WINDOW_EXPIRED", tuple(required),
        f"签收后 {days} 天，已超出无理由窗口 {policy['no_reason_window_days']} 天",
    )


def decide_from_snapshot(snapshot: dict[str, Any], claim_type: str, request_date: str) -> GroundTruth:
    """便捷入口：直接从环境快照算标准答案。"""
    order = snapshot["orders"][0]
    logistics = dict(snapshot["logistics"][0])
    logistics["damage_reported"] = bool(logistics["damage_reported"])
    policy = next(p for p in snapshot["policies"] if p["category"] == order["category"])
    policy = dict(policy)
    policy["exchange_supported"] = bool(policy["exchange_supported"])
    history = dict(snapshot["customer_history"][0])
    history["abuse_flag"] = bool(history["abuse_flag"])
    inventory = next(i for i in snapshot["inventory"] if i["sku"] == order["sku"])
    return decide(order, logistics, policy, history, inventory, claim_type, request_date)
