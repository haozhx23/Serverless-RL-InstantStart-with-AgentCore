"""Reward: format only. 1.0 when the completion is just a number, whatever the number is.

A reward that cannot tell right from wrong. The base model already answers with a bare
number, so the four completions of every group score the same: zero advantage, nothing to
train on, and the run spends sampling tokens to learn nothing. The reward check before the
session warns about it; run it anyway to see what "Groups with signal: 0%" looks like.
"""

import re


def reward(completion: str, answer: int) -> float:  # noqa: ARG001 - ignoring the answer is the point
    return 1.0 if re.fullmatch(r"\s*-?\d[\d,]*\s*", completion) else 0.0
