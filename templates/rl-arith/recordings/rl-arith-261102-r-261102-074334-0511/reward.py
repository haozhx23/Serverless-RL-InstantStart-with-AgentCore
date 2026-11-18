"""Reward: partial credit. 1.0 for the exact product; a wrong number earns up to 0.5 for digits in the right place.

A denser signal than exact match: in a group where all four answers are wrong, the ones
with more correct digits still score higher, so that group also produces a gradient.
Whether that helps a real model learn multiplication faster is not measured here.
"""

import re


def reward(completion: str, answer: int) -> float:
    m = re.search(r"-?\d[\d,]*", completion)
    if not m:
        return 0.0
    got, want = m.group().replace(",", ""), str(answer)
    if got == want:
        return 1.0
    width = max(len(got), len(want))
    same = sum(a == b for a, b in zip(got.rjust(width), want.rjust(width)))  # units digit lines up with units digit
    return 0.5 * same / width
