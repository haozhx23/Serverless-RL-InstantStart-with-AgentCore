"""The task for the single-turn RL scenario: three-digit by two-digit multiplication.

Every problem has one right answer that code can check, so the reward needs no judge
model. To try your own task, change make_problem() and the reward in reward.py together.
"""

import re


def make_problem(rng):
    a, b = rng.randint(100, 999), rng.randint(11, 99)
    return f"Q: What is {a} * {b}?\nA:", a * b


def is_correct(completion: str, answer: int) -> bool:
    """How the held-out check decides right from wrong: the first number equals the product.

    It deliberately does not use reward.py, so a reward you swap or edit cannot change
    what counts as a correct answer in the before/after comparison.
    """
    m = re.search(r"-?\d[\d,]*", completion)
    return bool(m) and int(m.group().replace(",", "")) == answer
