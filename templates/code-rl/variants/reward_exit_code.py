"""Reward: the test program's exit code, as when grading by pytest's exit status. 1.0 for exit code 0.

This file is one slot of the loop: loop.py calls reward(result) for every sampled body, in this
process, after the sandbox ran it. See reward.py for what `result` holds.

It is here to be caught: a body that calls os._exit(0) ends the test program with exit code 0
before any result is printed, and scores full marks. pytest behaves the same way (measured on
2026-11-10 with pytest 9.2.0: exit 0, no test run). The reward check before the session runs
exactly such a body and warns.
"""


def reward(result: dict) -> float:
    return 1.0 if result["exit_code"] == 0 else 0.0
