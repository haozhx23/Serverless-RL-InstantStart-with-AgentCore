"""Preference loss: IPO (Azar et al., 2023). loss = (margin - 1 / (2 * beta))^2.

Pulls the margin toward a fixed target instead of pushing it ever wider, which is IPO's
answer to DPO overfitting on deterministic preferences. Same form as TRL's "ipo" loss type,
which averages the log-probs over tokens; this loop sums them, like the other variants.
"""


def loss_and_grad(margin: float, beta: float) -> tuple[float, float]:
    target = 1.0 / (2.0 * beta)
    return (margin - target) ** 2, 2.0 * (margin - target)
