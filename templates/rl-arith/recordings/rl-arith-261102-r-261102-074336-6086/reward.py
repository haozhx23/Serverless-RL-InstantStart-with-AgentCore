"""Reward: exact match. 1.0 when the first number in the completion equals the product, else 0.0.

This file is one slot of the loop: loop.py calls reward(completion, answer) for every
sampled completion, in this process, and the service never sees it. Keep it deterministic,
so the four completions of a group are compared on equal terms.
"""

import re


def reward(completion: str, answer: int) -> float:
    m = re.search(r"-?\d[\d,]*", completion)
    return 1.0 if m and int(m.group().replace(",", "")) == answer else 0.0
