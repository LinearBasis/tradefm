"""Avellaneda-Stoikov market-making formula."""

import torch


def avellaneda_stoikov_quotes(
    mid: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    kappa: torch.Tensor,
    inventory: torch.Tensor,
    gamma: float,
    tau: float,
) -> dict[str, torch.Tensor]:
    """Avellaneda-Stoikov bid/ask quotes from heads' μ/σ/κ predictions.

    Inputs are broadcastable tensors (any shape, must broadcast together).
    `sigma` and `kappa` must be strictly positive; the heads in `DecisionModule`
    enforce this via softplus.

    Formula:
        reservation_price = mid + μ − γ · q · σ² · τ
        half_spread       = (1 / γ) · log(1 + γ / κ)
        bid               = reservation_price − half_spread
        ask               = reservation_price + half_spread
    """
    reservation = mid + mu - gamma * inventory * sigma.pow(2) * tau
    half_spread = (1.0 / gamma) * torch.log1p(gamma / kappa)
    return {
        "reservation": reservation,
        "half_spread": half_spread,
        "bid": reservation - half_spread,
        "ask": reservation + half_spread,
    }
