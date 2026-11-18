"""Preference loss: hinge, as in SLiC (Zhao et al., 2023). loss = max(0, 1 - beta * margin).

A pair stops contributing once its scaled margin passes 1; until then every pair pushes
with the same force. Same form as TRL's "hinge" loss type.
"""


def loss_and_grad(margin: float, beta: float) -> tuple[float, float]:
    z = beta * margin
    return (1.0 - z, -beta) if z < 1.0 else (0.0, 0.0)
