"""State builders.

Three variants gated by EnvConfig.state_mode:

    "heads"  — α, σ, κ from heads + fast market features (3 + 11 = 14 dim)
    "hidden" — h_t ∈ R^{d_model} + fast market features (d_model + 11 dim)
    "both"   — α, σ, κ + h_t + fast features (3 + d_model + 11 dim)

All share the *fast tier* (market-microstructure features), which are not
routed through the encoder because their relevant timescale is ms, not s.

Fast tier (always present):
    inventory_norm       — position / max_inventory ∈ [-1, 1]
    tau_to_close         — (t_flatten - now) / session_length ∈ [0, 1]
    mid_delta            — (mid - last_mid) / tick (recent micro-drift)
    spread_ticks         — book_spread / tick
    log_bid_size         — log1p(best_bid_size)
    log_ask_size         — log1p(best_ask_size)
    imbalance            — (bid_size - ask_size) / (bid_size + ask_size)
    our_bid_offset_ticks — (mid - our_bid) / tick   (NaN-safe: -1 if no quote)
    our_ask_offset_ticks — (our_ask - mid) / tick   (NaN-safe: -1 if no quote)
    has_bid              — 1.0 if we have a live bid, else 0
    has_ask              — 1.0 if we have a live ask, else 0

Slow tier:
    state_mode = "heads"  → [alpha_rel, sigma_rel, kappa]                  (3 dim)
    state_mode = "hidden" → h_t.flatten()                                  (d_model dim)
    state_mode = "both"   → [alpha_rel, sigma_rel, kappa, h_t...]          (3 + d_model dim)

Total state dim:
    heads:  3 + 11 = 14
    hidden: d_model + 11
    both:   3 + d_model + 11
"""

import math

import numpy as np


FAST_TIER_DIM = 11


def state_dim(state_mode: str, d_model: int | None = None) -> int:
    if state_mode == "heads":
        return 3 + FAST_TIER_DIM
    if state_mode == "hidden":
        if d_model is None:
            raise ValueError("d_model required for state_mode='hidden'")
        return d_model + FAST_TIER_DIM
    if state_mode == "both":
        if d_model is None:
            raise ValueError("d_model required for state_mode='both'")
        return 3 + d_model + FAST_TIER_DIM
    raise ValueError(f"unknown state_mode: {state_mode}")


def build_fast_tier(
    *,
    position: float,
    max_inventory: int,
    now_sec: float,
    t_flatten_sec: float,
    session_length_sec: float,
    mid: float,
    last_mid: float | None,
    tick_size: float,
    book_spread: float,
    best_bid_size: float,
    best_ask_size: float,
    our_bid_price: float | None,
    our_ask_price: float | None,
) -> np.ndarray:
    inventory_norm = float(np.clip(position / max_inventory, -1.0, 1.0))
    tau = max(0.0, (t_flatten_sec - now_sec) / session_length_sec)

    if last_mid is None:
        mid_delta = 0.0
    else:
        mid_delta = (mid - last_mid) / tick_size

    spread_ticks = book_spread / tick_size
    log_bid_size = math.log1p(max(best_bid_size, 0.0))
    log_ask_size = math.log1p(max(best_ask_size, 0.0))

    denom = best_bid_size + best_ask_size
    imbalance = (best_bid_size - best_ask_size) / denom if denom > 0 else 0.0

    if our_bid_price is None or not math.isfinite(our_bid_price):
        our_bid_off = -1.0
    else:
        our_bid_off = (mid - our_bid_price) / tick_size
    if our_ask_price is None or not math.isfinite(our_ask_price):
        our_ask_off = -1.0
    else:
        our_ask_off = (our_ask_price - mid) / tick_size

    has_bid = 1.0 if (our_bid_price is not None and math.isfinite(our_bid_price)) else 0.0
    has_ask = 1.0 if (our_ask_price is not None and math.isfinite(our_ask_price)) else 0.0

    return np.array([
        inventory_norm, tau, mid_delta,
        spread_ticks, log_bid_size, log_ask_size, imbalance,
        our_bid_off, our_ask_off, has_bid, has_ask,
    ], dtype=np.float32)


def build_state_heads(
    *,
    alpha_rel: float,
    sigma_rel: float,
    kappa: float,
    fast_tier: np.ndarray,
) -> np.ndarray:
    slow = np.array([alpha_rel, sigma_rel, kappa], dtype=np.float32)
    return np.concatenate([slow, fast_tier])


def build_state_hidden(
    *,
    h_t: np.ndarray,  # shape (d_model,)
    fast_tier: np.ndarray,
) -> np.ndarray:
    return np.concatenate([h_t.astype(np.float32), fast_tier])


def build_state_both(
    *,
    alpha_rel: float,
    sigma_rel: float,
    kappa: float,
    h_t: np.ndarray,  # shape (d_model,)
    fast_tier: np.ndarray,
) -> np.ndarray:
    slow_heads = np.array([alpha_rel, sigma_rel, kappa], dtype=np.float32)
    return np.concatenate([slow_heads, h_t.astype(np.float32), fast_tier])
