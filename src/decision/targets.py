"""Compute supervised targets for decision heads.

All functions operate on per-instrument-per-day data.
Targets are forward-looking: the last tau events of each sequence get NaN.
"""

import numpy as np
import polars as pl


def compute_alpha_targets(mid_price: np.ndarray, tau: int) -> np.ndarray:
    """Forward normalized return: (mid[t+τ] - mid[t]) / mid[t]."""
    n = len(mid_price)
    targets = np.full(n, np.nan, dtype=np.float64)
    if n > tau:
        targets[: n - tau] = (mid_price[tau:] - mid_price[: n - tau]) / mid_price[: n - tau]
    return targets


def compute_risk_targets(mid_price: np.ndarray, tau: int) -> np.ndarray:
    """Realized volatility: sqrt(sum of squared returns) over [t, t+τ)."""
    n = len(mid_price)
    targets = np.full(n, np.nan, dtype=np.float64)
    if n <= tau:
        return targets

    returns = np.diff(mid_price) / mid_price[:-1]
    # NaN returns (from NaN mid_price) → 0 contribution to volatility
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    sq_ret = returns**2
    cumsum = np.concatenate([[0.0], np.cumsum(sq_ret)])

    # RV[i] = sqrt(cumsum[i+tau] - cumsum[i]), valid for i <= n-tau-1
    end = n - tau
    rv = cumsum[tau : tau + end] - cumsum[:end]
    targets[:end] = np.sqrt(np.maximum(rv, 0.0))
    return targets


def compute_intensity_targets(
    order_times: np.ndarray,
    trade_times: np.ndarray,
    trade_volumes: np.ndarray,
    daily_volume: float,
    tau: int,
    session_length: float = 30840.0,
) -> np.ndarray:
    """Normalized trade volume/sec in forward window relative to daily average.

    For order event at position i:
        dt = order_times[i+tau] - order_times[i]
        vol = sum of trade volumes with time in [order_times[i], order_times[i+tau])
        κ = (vol / dt) / (daily_volume / session_length)

    κ > 1 means market is more active than average, κ < 1 means quieter.

    Args:
        order_times:   time_sec for each order event (sorted)
        trade_times:   time_sec for trade events (ACTION=2) in this instrument-day
        trade_volumes: VOLUME for each trade event
        daily_volume:  total traded volume for this instrument-day
        tau:           forward horizon in order events
        session_length: trading session duration in seconds
    """
    n = len(order_times)
    targets = np.full(n, np.nan, dtype=np.float64)

    if n <= tau or len(trade_times) == 0 or daily_volume <= 0:
        return targets

    # Cumulative trade volume for searchsorted-based range queries
    trade_cumvol = np.concatenate([[0.0], np.cumsum(trade_volumes.astype(np.float64))])

    t_starts = order_times[: n - tau]
    t_ends = order_times[tau:]  # time of the tau-th future order event
    dts = np.maximum(t_ends - t_starts, 1e-6)

    # Volume of trades in [t_start, t_end) via searchsorted
    j_starts = np.searchsorted(trade_times, t_starts, side="left")
    j_ends = np.searchsorted(trade_times, t_ends, side="left")
    vols = trade_cumvol[j_ends] - trade_cumvol[j_starts]

    expected_rate = daily_volume / session_length  # vol/sec baseline
    targets[: n - tau] = (vols / dts) / expected_rate

    return targets


def add_targets_to_orders(
    orders: pl.DataFrame,
    full_df: pl.DataFrame,
    tau: int = 512,
    session_length: float = 30840.0,
) -> pl.DataFrame:
    """Compute and attach all three target columns to orders DataFrame.

    Args:
        orders:   order events (ACTION in {0,1}) for one instrument-day, sorted by NO
        full_df:  all events for the same instrument-day (for extracting trades)
        tau:      forward horizon in order events
        session_length: session duration in seconds

    Returns:
        orders with added columns: target_alpha, target_risk, target_intensity
    """
    mid_price = orders["mid_price"].to_numpy()
    order_times = orders["time_sec"].to_numpy()

    # Alpha and risk: computed from order mid-prices
    alpha = compute_alpha_targets(mid_price, tau)
    risk = compute_risk_targets(mid_price, tau)

    # Intensity: needs trade data from full_df
    trades = full_df.filter(pl.col("ACTION") == 2).sort("NO")
    trade_times = trades["time_sec"].to_numpy() if trades.height > 0 else np.array([])
    trade_vols = trades["VOLUME"].to_numpy() if trades.height > 0 else np.array([])
    daily_volume = float(orders["daily_volume"].first())

    intensity = compute_intensity_targets(
        order_times, trade_times, trade_vols, daily_volume, tau, session_length,
    )

    return orders.with_columns(
        target_alpha=pl.Series(alpha),
        target_risk=pl.Series(risk),
        target_intensity=pl.Series(intensity),
    )
