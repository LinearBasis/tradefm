"""Action-mode mapping: (mode, action) → (bid_price, ask_price).

Three continuous modes (A/B/C) and three discrete grids (A_disc/B_disc/C_disc).
Each `compute_quotes_*` function takes the raw action plus context (A-S baseline
outputs, σ̂, best bid/ask, mid, etc.) and returns final quote prices in price units.

Mode summary:
    A — residual over A-S:    a=0 → exact A-S quotes
    B — A-S parameter tuning: a=0 → exact A-S quotes  (matches Alpha-AS spirit)
    C — direct β·spread:      a=0 → mid (= instant fill; NO warm start)

A_disc — 9-action grid (Δhalf_spread × Δskew)
B_disc — 20-action grid (γ × skew_mult)   ← exact Alpha-AS schedule
C_disc — 10-action grid (no-op + ask_level × bid_level on {1,3,5})  ← Smirnov table 2.1
"""

from dataclasses import dataclass

import numpy as np

from src._archive.decision.avellaneda_stoikov import avellaneda_stoikov_quotes


# --- Discrete grids ------------------------------------------------------------

# A_disc: 9 = 3 × 3
A_DISC_SPREAD_LEVELS = (-0.5, 0.0, 0.5)   # Δhalf_spread multiplier in exp space
A_DISC_SKEW_LEVELS = (-1.0, 0.0, 1.0)     # Δskew in units of σ·Δt

# B_disc: 20 = 4 × 5  (exactly Alpha-AS, Gaspar 2022)
B_DISC_GAMMA_LEVELS = (0.01, 0.1, 0.2, 0.9)
B_DISC_SKEW_MULT_LEVELS = (-0.10, -0.05, 0.00, 0.05, 0.10)

# C_disc: 10 actions (Smirnov table 2.1). (ask_level, bid_level) where 0 = no quote.
# Row order matches his table — first column is no-op.
C_DISC_LEVELS = (
    (0, 0),  # no quote
    (1, 1), (1, 3), (1, 5),
    (3, 1), (3, 3), (3, 5),
    (5, 1), (5, 3), (5, 5),
)


def action_space_size(mode: str) -> int:
    """Return action dim (continuous) or n_actions (discrete)."""
    return {
        "a": 2, "b": 2, "c": 2,
        "a_disc": len(A_DISC_SPREAD_LEVELS) * len(A_DISC_SKEW_LEVELS),
        "b_disc": len(B_DISC_GAMMA_LEVELS) * len(B_DISC_SKEW_MULT_LEVELS),
        "c_disc": len(C_DISC_LEVELS),
    }[mode]


def is_continuous(mode: str) -> bool:
    return mode in ("a", "b", "c")


# --- Context bundle passed to all compute_quotes_* -----------------------------

@dataclass
class QuoteContext:
    mid: float
    best_bid: float
    best_ask: float
    book_spread: float           # = best_ask - best_bid (in price units)
    best_bid_qty: float          # aggregate volume at top bid (lots)
    best_ask_qty: float          # aggregate volume at top ask (lots)
    inventory: float             # raw position (lots)
    sigma_price: float           # σ̂ from risk-head, in price units (= σ̂_rel * mid), already floored
    kappa: float                 # κ̂ from intensity-head
    dt_sec: float                # Δt for A-S (= quote_refresh_ms / 1000)
    tick_size: float
    max_half_spread_ticks: float
    gamma_as: float              # A-S risk aversion (driving Modes A/B baseline)


# --- Continuous modes ----------------------------------------------------------

def _as_baseline(ctx: QuoteContext) -> dict[str, float]:
    """Run A-S formula at current state, returning scalars (not tensors)."""
    import torch
    out = avellaneda_stoikov_quotes(
        mid=torch.tensor(ctx.mid),
        sigma=torch.tensor(ctx.sigma_price),
        kappa=torch.tensor(ctx.kappa),
        inventory=torch.tensor(ctx.inventory),
        gamma=ctx.gamma_as,
        delta_t=ctx.dt_sec,
    )
    return {k: float(v.item()) for k, v in out.items()}


def _cap_half_spread(half_spread: float, ctx: QuoteContext) -> float:
    """Clip half_spread to ≤ max_half_spread_ticks * tick, and ≥ 1 tick."""
    cap = ctx.max_half_spread_ticks * ctx.tick_size
    return float(np.clip(half_spread, ctx.tick_size, cap))


def compute_quotes_a(action: np.ndarray, ctx: QuoteContext, c_spread: float, c_skew: float) -> tuple[float, float]:
    """Mode A: residual over A-S.
        half_spread_final = half_spread_AS * exp(c_spread * a_spread)
        skew_final        = (reservation_AS - mid) + c_skew * a_skew * σ * Δt
    Action a = (a_spread, a_skew) ∈ [-1, 1]². a = 0 → exact A-S.
    """
    a_spread, a_skew = float(action[0]), float(action[1])
    asb = _as_baseline(ctx)

    half_spread = asb["half_spread"] * float(np.exp(c_spread * a_spread))
    half_spread = _cap_half_spread(half_spread, ctx)

    skew_as = asb["reservation"] - ctx.mid  # negative if long inventory
    skew_total = skew_as + c_skew * a_skew * ctx.sigma_price * ctx.dt_sec
    reservation = ctx.mid + skew_total

    return reservation - half_spread, reservation + half_spread


def compute_quotes_b(action: np.ndarray, ctx: QuoteContext, c_gamma: float, c_skew_b: float) -> tuple[float, float]:
    """Mode B: A-S parameter tuning.
        γ_final = γ_AS * exp(c_gamma * a_gamma)
        prices ← A-S(γ_final) * (1 + c_skew_b * a_skew)
    a = 0 → exact A-S. Multiplicative skew on the final prices matches Alpha-AS.
    """
    a_gamma, a_skew = float(action[0]), float(action[1])
    gamma_eff = ctx.gamma_as * float(np.exp(c_gamma * a_gamma))

    import torch
    out = avellaneda_stoikov_quotes(
        mid=torch.tensor(ctx.mid),
        sigma=torch.tensor(ctx.sigma_price),
        kappa=torch.tensor(ctx.kappa),
        inventory=torch.tensor(ctx.inventory),
        gamma=gamma_eff,
        delta_t=ctx.dt_sec,
    )
    half_spread = _cap_half_spread(float(out["half_spread"].item()), ctx)
    reservation = float(out["reservation"].item())

    skew_mult = 1.0 + c_skew_b * a_skew
    bid = (reservation - half_spread) * skew_mult
    ask = (reservation + half_spread) * skew_mult
    return bid, ask


def compute_quotes_c(action: np.ndarray, ctx: QuoteContext) -> tuple[float, float]:
    """Mode C: direct β·spread from mid (Smirnov SAC parameterization).
        bid = mid - β_bid * book_spread
        ask = mid + β_ask * book_spread
    a = 0 → bid=ask=mid (catastrophic), NOT a warm start.
    The max_half_spread cap is still enforced.
    """
    beta_bid, beta_ask = float(action[0]), float(action[1])

    # Project absolute distances from mid through cap
    raw_bid_off = beta_bid * ctx.book_spread
    raw_ask_off = beta_ask * ctx.book_spread

    cap = ctx.max_half_spread_ticks * ctx.tick_size
    bid_off = float(np.clip(raw_bid_off, ctx.tick_size, cap))
    ask_off = float(np.clip(raw_ask_off, ctx.tick_size, cap))

    return ctx.mid - bid_off, ctx.mid + ask_off


# --- Discrete modes ------------------------------------------------------------

def compute_quotes_a_disc(action: int, ctx: QuoteContext, c_spread: float, c_skew: float) -> tuple[float, float]:
    """A_disc: 3×3 grid of (Δhalf_spread mult, Δskew levels). Action index 0..8."""
    n_skew = len(A_DISC_SKEW_LEVELS)
    i_spread, i_skew = divmod(int(action), n_skew)
    a = np.array([A_DISC_SPREAD_LEVELS[i_spread], A_DISC_SKEW_LEVELS[i_skew]])
    return compute_quotes_a(a, ctx, c_spread, c_skew)


def compute_quotes_b_disc(action: int, ctx: QuoteContext) -> tuple[float, float]:
    """B_disc: 4×5 Alpha-AS grid (γ × skew_mult). Action index 0..19."""
    n_skew = len(B_DISC_SKEW_MULT_LEVELS)
    i_gamma, i_skew = divmod(int(action), n_skew)
    gamma_eff = B_DISC_GAMMA_LEVELS[i_gamma]
    skew_mult = 1.0 + B_DISC_SKEW_MULT_LEVELS[i_skew]

    import torch
    out = avellaneda_stoikov_quotes(
        mid=torch.tensor(ctx.mid),
        sigma=torch.tensor(ctx.sigma_price),
        kappa=torch.tensor(ctx.kappa),
        inventory=torch.tensor(ctx.inventory),
        gamma=gamma_eff,
        delta_t=ctx.dt_sec,
    )
    half_spread = _cap_half_spread(float(out["half_spread"].item()), ctx)
    reservation = float(out["reservation"].item())
    return (reservation - half_spread) * skew_mult, (reservation + half_spread) * skew_mult


def compute_quotes_c_disc(action: int, ctx: QuoteContext) -> tuple[float, float] | None:
    """C_disc: 10 actions per Smirnov. Returns None for no-op (no quotes placed).

    The mapping is (ask_level, bid_level) — both in ticks away from BBO. Level 0
    on either side means "do not place that side"; (0,0) means "no quote at all".
    """
    ask_level, bid_level = C_DISC_LEVELS[int(action)]
    if ask_level == 0 and bid_level == 0:
        return None
    # Quote price = best bid/ask shifted into the book by `level` ticks
    bid_px = ctx.best_bid - bid_level * ctx.tick_size if bid_level > 0 else float("nan")
    ask_px = ctx.best_ask + ask_level * ctx.tick_size if ask_level > 0 else float("nan")
    return bid_px, ask_px


# --- Dispatcher ----------------------------------------------------------------

def compute_quotes(
    mode: str,
    action,
    ctx: QuoteContext,
    c_spread: float,
    c_skew: float,
    c_gamma: float,
    c_skew_b: float,
) -> tuple[float, float] | None:
    """Single entry point. Returns (bid_px, ask_px) or None for no-quote (only C_disc)."""
    if mode == "a":
        return compute_quotes_a(action, ctx, c_spread, c_skew)
    if mode == "b":
        return compute_quotes_b(action, ctx, c_gamma, c_skew_b)
    if mode == "c":
        return compute_quotes_c(action, ctx)
    if mode == "a_disc":
        return compute_quotes_a_disc(action, ctx, c_spread, c_skew)
    if mode == "b_disc":
        return compute_quotes_b_disc(action, ctx)
    if mode == "c_disc":
        return compute_quotes_c_disc(action, ctx)
    raise ValueError(f"unknown action mode: {mode}")
