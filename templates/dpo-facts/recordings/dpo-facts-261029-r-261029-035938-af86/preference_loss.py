"""Preference loss: DPO (Rafailov et al., 2023). loss = -log sigmoid(beta * margin).

This file is the loop's loss slot, and the service has no such loss: it ships cross_entropy,
importance_sampling, ppo and cispo. loop.py computes, for each pair,

    margin = (logp(chosen) - ref(chosen)) - (logp(rejected) - ref(rejected))

from sequence log-probs it got back from forward, calls loss_and_grad(margin, beta), and
turns dloss/dmargin into advantages for an importance_sampling forward_backward at ratio 1,
whose gradient is then exactly this loss's gradient (the cookbook's surrogate-loss trick).
Write any loss of the margin here and the rest of the loop stays the same.
"""

import math


def loss_and_grad(margin: float, beta: float) -> tuple[float, float]:
    z = beta * margin
    sig_neg = 1.0 / (1.0 + math.exp(z)) if z > -30 else 1.0   # sigmoid(-z), stable
    loss = math.log1p(math.exp(-z)) if z > -30 else -z         # -log sigmoid(z)
    return loss, -beta * sig_neg
