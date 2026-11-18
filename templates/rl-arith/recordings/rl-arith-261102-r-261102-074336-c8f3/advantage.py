"""Advantage: one baseline for the whole batch. (reward - batch mean) / batch std.

The cookbook's Monte-Carlo estimator with ReB normalization (compute_monte_carlo_advantages,
use_rebn=True) reduces to this for single-turn episodes. Groups where all four completions
agree still get non-zero advantages, so more datums train: watch the train tokens and the
cost per step next to GRPO's.
"""

import statistics


def advantages(groups: list[list[float]]) -> list[list[float] | None]:
    flat = [r for rewards in groups for r in rewards]
    mean, std = statistics.mean(flat), statistics.pstdev(flat)
    if std == 0:
        return [None for _ in groups]
    return [[(r - mean) / std for r in rewards] for rewards in groups]
