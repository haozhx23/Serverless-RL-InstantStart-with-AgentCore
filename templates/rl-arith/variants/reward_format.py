"""Reward: format only. 1.0 when the completion is just a number, whatever the number is.

A reward that cannot tell right from wrong; the reward check before the session warns about
it. What it does then depends on the model.
Live, gpt-oss-20b did not always answer with a bare number (one run on 2026-11-02): 16 of its 32 first answers
were not bare numbers (worked arithmetic, "876 * 64 = 56064", a trailing period or quote), so
groups did carry a gradient, and RL taught it the format, raising the share of bare-number
answers from 50% to 91%. Held-out answers went from
9 to 13 correct in that run; how much of that is the format and how much arithmetic was not
measured.
"""

import re


def reward(completion: str, answer: int) -> float:  # noqa: ARG001 - ignoring the answer is the point
    return 1.0 if re.fullmatch(r"\s*-?\d[\d,]*\s*", completion) else 0.0
