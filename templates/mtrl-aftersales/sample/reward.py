"""分层稠密 reward，以及对应的防 reward hacking 设计。

为什么不用 0/1 reward
--------------------
四类裁决 + 金额 + reason_code 全对才给 1 分的话，训练早期几乎所有 rollout 都是 0，
一个 group 内 reward 全等 → 组内优势为 0 → 没有梯度。这正是 MTRL 指标里
``analysis/zero_adv_zero_reward``（整组全错）偏高的典型成因。分层给分让早期
就存在组内方差，是 RL 能起步的前提。

分数构成（满分 1.0）
-------------------
====================  ====  ==================================================
项                    权重  说明
====================  ====  ==================================================
decision              0.50  裁决类别与标准答案一致
amount                0.20  金额误差 <= 0.01，且 **decision 必须已正确**
reason_code           0.10  reason_code 一致，且 **decision 必须已正确**
流程合规              0.14  已调用的必查工具占比
调用效率              0.06  未调用无关工具、未重复调用
====================  ====  ==================================================

乘法惩罚项在最后统一施加。

防 reward hacking 的四个具体措施
--------------------------------
1. **amount 与 reason_code 由 decision 门控。** 否则模型只要恒填
   ``amount=0`` 就能在所有非退款样本上白拿 0.2 分（数据集里非退款样本占多数）。
2. **数据集四类裁决均衡**（见 ``dataset.py``）。恒选单一类别的期望得分被压到
   约 0.25 × 0.5 = 0.125，与满分差距足够大。
3. **流程分按"必查工具"占比给，而不是按调用次数给。** 按次数给分会直接鼓励
   刷工具调用，同时推高 prefill token 成本。
4. **重复提交裁决要扣分。** 否则模型可以连续提交多个不同答案来提高命中率；
   环境侧只认第一次，reward 侧再乘 0.8。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sample.env import READ_TOOLS, RolloutTrace
from sample.rules import GroundTruth

AMOUNT_TOLERANCE = 0.01

W_DECISION = 0.50
W_AMOUNT = 0.20
W_REASON = 0.10
W_COMPLIANCE = 0.14
W_EFFICIENCY = 0.06

#: 重复提交裁决的乘法惩罚。
PENALTY_MULTI_SUBMIT = 0.8
#: 每次非法工具调用的乘法惩罚，累乘后不低于 FLOOR_INVALID。
PENALTY_PER_INVALID = 0.95
FLOOR_INVALID = 0.7


@dataclass
class RewardBreakdown:
    """reward 的分项结果。``metrics`` 可直接打到 MLflow 便于逐项排查。"""

    reward: float
    metrics: dict[str, float]
    notes: list[str]


def compute_reward(trace: RolloutTrace, gt: GroundTruth) -> RewardBreakdown:
    """按分层规则给一条 rollout 打分。

    Args:
        trace: 环境记录的完整调用过程。
        gt: 该 prompt 的标准答案与必查工具。
    """
    notes: list[str] = []
    metrics: dict[str, float] = {
        "turns": float(trace.turns),
        "tool_calls": float(len(trace.calls)),
        "invalid_tool_calls": float(trace.invalid_tool_calls),
        "submit_attempts": float(trace.submit_attempts),
    }

    # ---- 硬性前置：没有可解析的裁决就是 0 分，不给部分分 ----------------
    if trace.malformed_output:
        notes.append("模型输出无法解析为合法工具调用")
        metrics.update(_zeros())
        return RewardBreakdown(0.0, metrics, notes)
    if trace.decision is None:
        notes.append("rollout 结束时没有提交裁决（超轮数或提前退出）")
        metrics.update(_zeros())
        return RewardBreakdown(0.0, metrics, notes)

    d = trace.decision

    # ---- 结果正确性 ---------------------------------------------------
    decision_ok = d.decision == gt.decision
    amount_ok = abs(d.amount - gt.amount) <= AMOUNT_TOLERANCE
    reason_ok = d.reason_code == gt.reason_code

    score = 0.0
    if decision_ok:
        score += W_DECISION
        # amount 与 reason 由 decision 门控，见模块文档措施 1。
        if amount_ok:
            score += W_AMOUNT
        else:
            notes.append(f"金额不符：模型 {d.amount}，标准答案 {gt.amount}")
        if reason_ok:
            score += W_REASON
        else:
            notes.append(f"reason_code 不符：模型 {d.reason_code}，标准答案 {gt.reason_code}")
    else:
        notes.append(f"裁决不符：模型 {d.decision}，标准答案 {gt.decision}")

    metrics["decision_correct"] = float(decision_ok)
    metrics["amount_correct"] = float(amount_ok)
    metrics["reason_correct"] = float(reason_ok)
    metrics["exact_match"] = float(decision_ok and amount_ok and reason_ok)

    # ---- 流程合规：必查工具占比 ----------------------------------------
    called_reads = trace.read_tools_called()
    required = set(gt.required_tools)
    hit = called_reads & required
    coverage = len(hit) / len(required) if required else 1.0
    score += W_COMPLIANCE * coverage
    if coverage < 1.0:
        notes.append(f"必查工具缺失：{sorted(required - called_reads)}")
    metrics["required_tool_coverage"] = coverage

    # ---- 调用效率：无关工具与重复调用 ----------------------------------
    extra_tools = called_reads - required
    dup_count = _duplicate_calls(trace)
    waste = len(extra_tools) + dup_count
    efficiency = max(0.0, 1.0 - waste / 3.0)
    score += W_EFFICIENCY * efficiency
    metrics["extra_tools"] = float(len(extra_tools))
    metrics["duplicate_calls"] = float(dup_count)
    metrics["efficiency"] = efficiency
    if extra_tools:
        notes.append(f"调用了与本路径无关的工具：{sorted(extra_tools)}")
    if dup_count:
        notes.append(f"重复调用（同名同参）{dup_count} 次")

    # ---- 乘法惩罚 ------------------------------------------------------
    penalty = 1.0
    if trace.submit_attempts > 1:
        penalty *= PENALTY_MULTI_SUBMIT
        notes.append(f"重复提交裁决 {trace.submit_attempts} 次，只认第一次并扣分")
    if trace.invalid_tool_calls:
        penalty *= max(FLOOR_INVALID, PENALTY_PER_INVALID ** trace.invalid_tool_calls)

    reward = round(max(0.0, min(1.0, score * penalty)), 6)
    metrics["penalty_multiplier"] = penalty
    metrics["reward"] = reward
    return RewardBreakdown(reward, metrics, notes)


def compute_per_turn_reward(trace: RolloutTrace, gt: GroundTruth) -> list[float]:
    """按轮 reward，用于 ``advantage_method="group_based_per_turn"``。

    SDK 的 ``RolloutFeedbackClient.update_reward`` 接受 ``float`` 或
    ``list[float]``（``sagemaker/train/rft/feedback.py``），装饰器在返回值
    ``reward`` 是 list 时会走 ``complete_rollout()`` + ``update_reward(list)``。
    官方文档目前只描述标量 reward，这条按轮通路只在 SDK 源码里可见。

    这里的分配方式：调用到必查工具的那一轮各记一份流程分，最后一轮记结果分。
    列表长度等于实际轮数。

    注意：``group_based_per_turn`` 的实际效果官方没有给对照数据，本函数是
    机制演示，不代表推荐配置。
    """
    total = compute_reward(trace, gt)
    turns = max(1, trace.turns)
    per_turn = [0.0] * turns
    required = set(gt.required_tools)

    process_part = W_COMPLIANCE + W_EFFICIENCY
    result_part = max(0.0, total.reward - process_part * total.metrics.get("required_tool_coverage", 0.0))

    hit_turns = sorted({c.turn for c in trace.calls if c.name in required and c.ok})
    if hit_turns:
        share = (process_part * total.metrics.get("required_tool_coverage", 0.0)) / len(hit_turns)
        for t in hit_turns:
            if 0 <= t < turns:
                per_turn[t] += share
    per_turn[-1] += result_part
    return [round(x, 6) for x in per_turn]


def _zeros() -> dict[str, float]:
    return {
        "decision_correct": 0.0,
        "amount_correct": 0.0,
        "reason_correct": 0.0,
        "exact_match": 0.0,
        "required_tool_coverage": 0.0,
        "extra_tools": 0.0,
        "duplicate_calls": 0.0,
        "efficiency": 0.0,
        "penalty_multiplier": 1.0,
        "reward": 0.0,
    }


def _duplicate_calls(trace: RolloutTrace) -> int:
    """统计同名同参的重复调用次数（首次不算重复）。"""
    seen: set[tuple[str, str]] = set()
    dups = 0
    for c in trace.calls:
        if c.name not in READ_TOOLS:
            continue
        key = (c.name, repr(sorted(c.args.items())))
        if key in seen:
            dups += 1
        else:
            seen.add(key)
    return dups
