"""独立业务评估：与训练 reward 分开的验收口径。

为什么必须分开
--------------
训练 reward 是**为了产生梯度**而设计的：它给部分分、给流程分，好让训练早期
就有组内方差。业务方验收关心的是完全不同的问题——有多少工单能直接自动放行、
错判会赔多少钱、最高风险的错误有没有被拦住。用训练 reward 上升来汇报业务效果
会高估收益，因为流程分和部分分都会推高它。

本模块只算硬指标，没有部分分、没有流程分：

============================  ==================================================
指标                          含义
============================  ==================================================
``auto_pass_rate``            三字段全对的比例，即可直接自动放行的工单占比
``decision_accuracy``         裁决类别正确率
``compliance_violation_rate`` 必查工具没查全就下结论的比例（合规风险，与对错无关）
``abuse_leak_rate``           滥用客户被判成退款/换货的比例（最高风险错误）
``overpay_total``             逐件比较后多赔的金额之和（财务损失）
``underpay_total``            逐件比较后少赔的金额之和（客诉与监管风险）
``net_payout_delta``          多赔减少赔，用于看总体口径松紧
``window_violation_rate``     已过窗口却仍判退款/换货的比例
============================  ==================================================

``abuse_leak_rate`` 与 ``overpay_total`` 是"一票否决"型指标：即使 auto_pass_rate
提升，这两项变差也不应上线。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from sample.agent import ChatClient, run_rollout


@dataclass
class BusinessReport:
    n: int = 0
    auto_pass: int = 0
    decision_correct: int = 0
    compliance_violation: int = 0
    abuse_leak: int = 0
    window_violation: int = 0
    no_decision: int = 0
    #: 逐件累计，不做正负抵消——多赔和少赔的业务含义不同。
    overpay_total: float = 0.0
    underpay_total: float = 0.0
    confusion: Counter[tuple[str, str]] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, Any]:
        n = max(1, self.n)
        return {
            "n": self.n,
            "auto_pass_rate": self.auto_pass / n,
            "decision_accuracy": self.decision_correct / n,
            "compliance_violation_rate": self.compliance_violation / n,
            "abuse_leak_rate": self.abuse_leak / n,
            "window_violation_rate": self.window_violation / n,
            "no_decision_rate": self.no_decision / n,
            "overpay_total": round(self.overpay_total, 2),
            "underpay_total": round(self.underpay_total, 2),
            "net_payout_delta": round(self.overpay_total - self.underpay_total, 2),
            "overpay_per_case": round(self.overpay_total / n, 2),
            "underpay_per_case": round(self.underpay_total / n, 2),
        }

    def top_confusions(self, k: int = 5) -> list[tuple[str, str, int]]:
        """最常见的误判方向 (标准答案 -> 模型裁决, 次数)。"""
        return [(t, m, c) for (t, m), c in self.confusion.most_common(k) if t != m]


PAYOUT_DECISIONS = frozenset({"refund", "exchange"})


def evaluate(cases: list[dict[str, Any]], client: ChatClient) -> BusinessReport:
    """在评估集上跑一遍并汇总业务指标。"""
    rep = BusinessReport()
    for case in cases:
        result = run_rollout(case, client)
        gt = result.ground_truth
        d = result.trace.decision
        rep.n += 1

        if d is None:
            rep.no_decision += 1
            rep.confusion[(gt.decision, "<none>")] += 1
            # 没有裁决等于该赔的一分没赔。
            rep.underpay_total += gt.amount
            continue

        rep.confusion[(gt.decision, d.decision)] += 1
        if d.decision == gt.decision:
            rep.decision_correct += 1
            if abs(d.amount - gt.amount) <= 0.01 and d.reason_code == gt.reason_code:
                rep.auto_pass += 1

        # 合规：必查工具是否查全（与裁决对错无关，是流程风险）
        if not set(gt.required_tools) <= result.trace.read_tools_called():
            rep.compliance_violation += 1

        # 最高风险：滥用客户被放过
        if gt.reason_code == "ABUSE_REVIEW" and d.decision in PAYOUT_DECISIONS:
            rep.abuse_leak += 1

        # 已过窗口却仍赔付
        if gt.reason_code == "WINDOW_EXPIRED" and d.decision in PAYOUT_DECISIONS:
            rep.window_violation += 1

        # 逐件比较实际赔付与应赔付，多赔与少赔分别累计。
        paid = d.amount if d.decision == "refund" else 0.0
        owed = gt.amount if gt.decision == "refund" else 0.0
        if paid > owed:
            rep.overpay_total += paid - owed
        elif owed > paid:
            rep.underpay_total += owed - paid

    return rep
