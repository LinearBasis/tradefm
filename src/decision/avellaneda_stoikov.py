"""Avellaneda-Stoikov market-making formula."""

import torch


def avellaneda_stoikov_quotes(
    mid: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    kappa: torch.Tensor,
    inventory: torch.Tensor,
    gamma: float,
) -> dict[str, torch.Tensor]:
    """Avellaneda-Stoikov bid/ask quotes from heads' μ/σ/κ predictions.

    Heads predict μ as total drift and σ as total realized volatility over the
    training window (HeadConfig.tau_sec seconds). The A-S formula's time
    horizon τ is therefore implicit in those predictions — equivalent to τ=1
    in the textbook formula.

    Inputs are broadcastable tensors. `sigma` and `kappa` must be strictly
    positive; the heads in `DecisionModule` enforce this via softplus.

    Formula:
        reservation_price = mid + μ − γ · q · σ²
        half_spread       = (1 / γ) · log(1 + γ / κ)
        bid               = reservation_price − half_spread
        ask               = reservation_price + half_spread
    """
    reservation = mid + mu - gamma * inventory * sigma.pow(2)
    half_spread = (1.0 / gamma) * torch.log1p(gamma / kappa)
    return {
        "reservation": reservation,
        "half_spread": half_spread,
        "bid": reservation - half_spread,
        "ask": reservation + half_spread,
    }
