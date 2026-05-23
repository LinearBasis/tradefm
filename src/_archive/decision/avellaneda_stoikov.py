"""Avellaneda-Stoikov market-making formula.

Transcribed from ESkripichnikov/market-making (strategies/baselines.py:241-252).
This is the *classical* A-S — pure inventory-skew, no drift term. The α-head's
output is intentionally NOT used here; if you want it in the loop, wrap this
function and add the drift externally.

All scalar inputs are in **price units** (rubles for MOEX, dollars for crypto).
The caller is responsible for converting head outputs from relative to price
units (e.g. `sigma_price = sigma_head * mid`).
"""

import torch


def avellaneda_stoikov_quotes(
    mid: torch.Tensor,
    sigma: torch.Tensor,
    kappa: torch.Tensor,
    inventory: torch.Tensor,
    gamma: float,
    delta_t: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Classical Avellaneda-Stoikov bid/ask quotes.

    Inputs:
        mid:        current mid-price (price units)
        sigma:      volatility (price std per √unit_time, e.g. per √sec)
        kappa:      order-arrival sensitivity (1/price units)
        inventory:  current asset position (lots)
        gamma:      risk aversion (1/price units), typically 0.1–1.0
        delta_t:    holding horizon in time units. With Skripichnikov's
                    σ-per-second scaling, set delta_t=1.0 sec.

    Formula:
        reservation = mid − q · γ · σ² · Δt
        spread      = γ · σ² · Δt + (2/γ) · log(1 + γ/κ)
        bid         = reservation − spread/2
        ask         = reservation + spread/2

    The first term of spread (γσ²Δt) is the inventory-risk premium; the
    second term [(2/γ)log(1+γ/κ)] is the order-flow profit margin.
    """
    sigma_sq_dt = sigma.pow(2) * delta_t
    reservation = mid - inventory * gamma * sigma_sq_dt
    spread = gamma * sigma_sq_dt + (2.0 / gamma) * torch.log1p(gamma / kappa)
    half_spread = spread / 2.0
    return {
        "reservation": reservation,
        "half_spread": half_spread,
        "bid": reservation - half_spread,
        "ask": reservation + half_spread,
    }
