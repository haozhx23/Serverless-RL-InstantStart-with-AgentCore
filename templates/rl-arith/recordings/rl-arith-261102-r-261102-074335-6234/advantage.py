"""Advantage: group mean baseline, not scaled. reward - group mean, per prompt.

Same groups train as with GRPO's normalization, but a group's advantages keep the size of
its reward differences instead of being divided by their spread.
"""

import statistics


def advantages(groups: list[list[float]]) -> list[list[float] | None]:
    out = []
    for rewards in groups:
        mean = statistics.mean(rewards)
        out.append(None if all(r == mean for r in rewards) else [r - mean for r in rewards])
    return out
