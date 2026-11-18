"""Reward: the share of tests passed. 0 when the program left no result (it crashed, exited early or ran out of time).

This file is one slot of the loop: loop.py calls reward(result) for every sampled body, in this
process, after the sandbox ran it. `result` is what task.grade() saw there: passed and total
tests, timed_out, exit_code, and reported (whether runner.py printed its result line). Partial
credit lets a group whose bodies all fail some tests still tell better from worse.
"""


def reward(result: dict) -> float:
    if not result["reported"] or result["timed_out"]:
        return 0.0
    return result["passed"] / result["total"]
