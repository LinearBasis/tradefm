"""Compute supervised targets for decision heads.

All functions operate on per-instrument-per-day data.
Targets are forward-looking over a window of `tau_sec` seconds; events whose
window extends past the last order event of the day get NaN.
"""

import numpy as np
import polars as pl


def compute_alpha_targets(
    mid_price: np.ndarray, order_times: np.ndarray, tau_sec: float
) -> np.ndarray:
    """Forward normalized return: (mid[j] - mid[i]) / mid[i],
    where j = smallest index with order_times[j] ≥ order_times[i] + tau_sec."""
    n = len(mid_price)
    targets = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return targets
    j = np.searchsorted(order_times, order_times + tau_sec, side="left")
    valid = j < n
    idx = np.where(valid)[0]
    targets[idx] = (mid_price[j[idx]] - mid_price[idx]) / mid_price[idx]
    return targets


def compute_risk_targets(
    mid_price: np.ndarray, order_times: np.ndarray, tau_sec: float
) -> np.ndarray:
    """Realized volatility: sqrt(sum of squared returns) over [t, t + tau_sec)."""
    n = len(mid_price)
    targets = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return targets

    returns = np.diff(mid_price) / mid_price[:-1]
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    sq_ret = returns**2
    cumsum = np.concatenate([[0.0], np.cumsum(sq_ret)])  # length n

    j = np.searchsorted(order_times, order_times + tau_sec, side="left")
    valid = j < n
    idx = np.where(valid)[0]
    rv = cumsum[j[idx]] - cumsum[idx]
    targets[idx] = np.sqrt(np.maximum(rv, 0.0))
    return targets


def compute_intensity_targets(
    order_times: np.ndarray,
    trade_times: np.ndarray,
    trade_volumes: np.ndarray,
    daily_volume: float,
    tau_sec: float,
    session_length: float = 30840.0,
) -> np.ndarray:
    """Normalized trade volume/sec in the forward window relative to daily average.

    For order event at position i:
        vol = sum of trade volumes with time in [order_times[i], order_times[i] + tau_sec)
        κ = (vol / tau_sec) / (daily_volume / session_length)

    κ > 1 means market is more active than average, κ < 1 means quieter.
    Mask matches alpha/risk: only events whose window stays within the day are valid.
    """
    n = len(order_times)
    targets = np.full(n, np.nan, dtype=np.float64)

    if n == 0 or len(trade_times) == 0 or daily_volume <= 0:
        return targets

    trade_cumvol = np.concatenate([[0.0], np.cumsum(trade_volumes.astype(np.float64))])

    t_starts = order_times
    t_ends = order_times + tau_sec
    j_orders = np.searchsorted(order_times, t_ends, side="left")
    valid = j_orders < n

    j_starts = np.searchsorted(trade_times, t_starts, side="left")
    j_ends = np.searchsorted(trade_times, t_ends, side="left")
    vols = trade_cumvol[j_ends] - trade_cumvol[j_starts]

    expected_rate = daily_volume / session_length  # vol/sec baseline
    targets[valid] = (vols[valid] / tau_sec) / expected_rate

    return targets


def add_targets_to_orders(
    orders: pl.DataFrame,
    full_df: pl.DataFrame,
    tau_sec: float = 1.0,
    session_length: float = 30840.0,
) -> pl.DataFrame:
    """Compute and attach all three target columns to orders DataFrame.

    Args:
        orders:   order events (ACTION in {0,1}) for one instrument-day, sorted by NO
        full_df:  all events for the same instrument-day (for extracting trades)
        tau_sec:  forward horizon in seconds
        session_length: session duration in seconds

    Returns:
        orders with added columns: target_alpha, target_risk, target_intensity
    """
    mid_price = orders["mid_price"].to_numpy()
    order_times = orders["time_sec"].to_numpy()

    alpha = compute_alpha_targets(mid_price, order_times, tau_sec)
    risk = compute_risk_targets(mid_price, order_times, tau_sec)

    trades = full_df.filter(pl.col("ACTION") == 2).sort("NO")
    trade_times = trades["time_sec"].to_numpy() if trades.height > 0 else np.array([])
    trade_vols = trades["VOLUME"].to_numpy() if trades.height > 0 else np.array([])
    daily_volume = float(orders["daily_volume"].first())

    intensity = compute_intensity_targets(
        order_times, trade_times, trade_vols, daily_volume, tau_sec, session_length,
    )

    return orders.with_columns(
        target_alpha=pl.Series(alpha),
        target_risk=pl.Series(risk),
        target_intensity=pl.Series(intensity),
    )
