"""A-S running-utility reward: quadratic inventory penalty with σ̂².

    r_t = scale * [ ΔPnL_MTM_t − (γ_R / 2) · q_norm_t² · σ̂_t² · Δt ]

Where:
    ΔPnL_MTM_t = (balance_t + position_t · mid_t) − (balance_{t-1} + position_{t-1} · mid_{t-1})
    q_norm_t   = position_t / max_inventory ∈ [-1, 1]
    σ̂_t       = relative volatility from risk-head (already floored at sigma_floor)
    Δt        = quote_refresh_ms / 1000

Inventory penalty is **quadratic with σ²**, following A-S/CARA derivation:
Cartea-Jaimungal lineage, ISAC, Relaver. Linear |q| (Smirnov) is the simpler
alternative — we don't use it; ablation deferred unless DDPG/SAC tie within noise.

Fees are already in `balance` (hftbacktest's flat_per_trade_fee_model), so do
NOT subtract them separately.
"""

from dataclasses import dataclass


@dataclass
class PnLSnapshot:
    """Components of mark-to-market PnL at one timestep."""
    balance: float
    position: float
    mid: float

    def mtm(self) -> float:
        return self.balance + self.position * self.mid


def compute_step_reward(
    prev: PnLSnapshot,
    cur: PnLSnapshot,
    sigma_rel_hat: float,
    max_inventory: int,
    gamma_R: float,
    dt_sec: float,
    scale: float,
) -> tuple[float, dict]:
    """Return (reward, breakdown dict for logging)."""
    delta_pnl = cur.mtm() - prev.mtm()
    q_norm = cur.position / max_inventory
    inv_penalty = 0.5 * gamma_R * (q_norm ** 2) * (sigma_rel_hat ** 2) * dt_sec
    reward = scale * (delta_pnl - inv_penalty)
    return reward, {
        "delta_pnl": delta_pnl,
        "inv_penalty": inv_penalty,
        "q_norm": q_norm,
        "sigma_rel": sigma_rel_hat,
    }
