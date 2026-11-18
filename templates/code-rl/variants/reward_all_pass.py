"""Reward: all or nothing. 1.0 only when every test passed in time, else 0.0.

This file is one slot of the loop: loop.py calls reward(result) for every sampled body, in this
process, after the sandbox ran it. See reward.py for what `result` holds. A group whose bodies
all fail at least one test scores all zeros and gives no gradient.
"""


def reward(result: dict) -> float:
    ok = result["reported"] and not result["timed_out"] and result["passed"] == result["total"]
    return 1.0 if ok else 0.0
