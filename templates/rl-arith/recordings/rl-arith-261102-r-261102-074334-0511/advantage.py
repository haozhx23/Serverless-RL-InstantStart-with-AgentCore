"""Advantage: group-normalized, as in GRPO. (reward - group mean) / group std, per prompt.

This file is one slot of the loop. loop.py passes the rewards of every group (the
completions sampled for one prompt) and turns each returned advantage into a datum's
`advantages` tensor; a group returned as None is left out of the batch. Same as the
cookbook's compute_grouped_advantages (rl/advantages.py), minus its epsilon.
"""

import statistics


def advantages(groups: list[list[float]]) -> list[list[float] | None]:
    out = []
    for rewards in groups:
        mean, std = statistics.mean(rewards), statistics.pstdev(rewards)
        # All right or all wrong: every advantage is 0, so the group adds no gradient.
        out.append(None if std == 0 else [(r - mean) / std for r in rewards])
    return out
