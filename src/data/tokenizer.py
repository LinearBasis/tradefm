"""Binning and composite token creation following TradeFM."""

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from src.config import PipelineConfig


@dataclass
class BinEdges:
    """Calibrated bin edges for each feature."""

    interarrival: np.ndarray = field(default_factory=lambda: np.array([]))
    price_depth: np.ndarray = field(default_factory=lambda: np.array([]))
    volume: np.ndarray = field(default_factory=lambda: np.array([]))
    price_level: np.ndarray = field(default_factory=lambda: np.array([]))
    liquidity: np.ndarray = field(default_factory=lambda: np.array([]))


def calibrate_bins(
    df: pl.DataFrame,
    cfg: PipelineConfig,
    dates: list[str] | None = None,
) -> BinEdges:
    """Calibrate bin edges from data.

    - Price-related (price_depth, price_level): quantile (equal-frequency) bins
    - Log-transformed (volume, interarrival): equal-width bins in log-space
    - Outliers beyond p1/p99 clipped before calibration

    If `dates` is provided, only rows matching those dates are used for calibration.
    """
    if dates is not None:
        df = df.filter(pl.col("date").is_in(dates))

    edges = BinEdges()

    lo = cfg.outlier_lower_pct
    hi = cfg.outlier_upper_pct

    # --- Interarrival: equal-width in log-space ---
    iat = df["interarrival"].drop_nulls().to_numpy()
    iat = iat[iat > 0]
    iat_log = np.log1p(iat)
    iat_log_clipped = _clip_percentile(iat_log, 0, hi)
    edges.interarrival = np.linspace(
        iat_log_clipped.min(), iat_log_clipped.max(), cfg.n_bins_interarrival + 1
    )

    # --- Volume: equal-width in log-space ---
    vol = df["log_volume"].drop_nulls().to_numpy()
    vol_clipped = _clip_percentile(vol, 0, hi)
    edges.volume = np.linspace(
        vol_clipped.min(), vol_clipped.max(), cfg.n_bins_volume + 1
    )

    # --- Price depth: quantile bins (symmetric, use both tails) ---
    pd_vals = df["price_depth"].drop_nulls().to_numpy()
    pd_clipped = _clip_percentile(pd_vals, lo, hi)
    edges.price_depth = np.quantile(
        pd_clipped, np.linspace(0, 1, cfg.n_bins_price_depth + 1)
    )

    # --- Price level: quantile bins (symmetric) ---
    pl_vals = df["price_level"].drop_nulls().to_numpy()
    pl_clipped = _clip_percentile(pl_vals, lo, hi)
    edges.price_level = np.quantile(
        pl_clipped, np.linspace(0, 1, cfg.n_bins_price_level + 1)
    )

    # --- Liquidity: quantile bins on daily_volume ---
    liq = df["daily_volume"].unique().drop_nulls().to_numpy().astype(np.float64)
    edges.liquidity = np.quantile(
        liq, np.linspace(0, 1, cfg.n_liquidity_bins + 1)
    )

    return edges


def digitize_features(df: pl.DataFrame, edges: BinEdges, cfg: PipelineConfig) -> pl.DataFrame:
    """Assign bin indices for all features.

    Outliers beyond calibrated range get special edge bins (0 or n_bins-1).
    """
    iat = df["interarrival"].to_numpy()
    iat_log = np.log1p(np.maximum(iat, 0))
    i_iat = _safe_digitize(iat_log, edges.interarrival, cfg.n_bins_interarrival)

    i_depth = _safe_digitize(
        df["price_depth"].to_numpy(), edges.price_depth, cfg.n_bins_price_depth
    )

    i_vol = _safe_digitize(
        df["log_volume"].to_numpy(), edges.volume, cfg.n_bins_volume
    )

    i_plevel = _safe_digitize(
        df["price_level"].to_numpy(), edges.price_level, cfg.n_bins_price_level
    )

    i_liq = _safe_digitize(
        df["daily_volume"].cast(pl.Float64).to_numpy(),
        edges.liquidity,
        cfg.n_liquidity_bins,
    )

    df = df.with_columns(
        bin_interarrival=pl.Series(i_iat, dtype=pl.Int16),
        bin_price_depth=pl.Series(i_depth, dtype=pl.Int16),
        bin_volume=pl.Series(i_vol, dtype=pl.Int16),
        bin_price_level=pl.Series(i_plevel, dtype=pl.Int16),
        bin_liquidity=pl.Series(i_liq, dtype=pl.Int16),
    )

    return df


def compose_trade_token(df: pl.DataFrame, cfg: PipelineConfig) -> pl.DataFrame:
    """Compose single integer trade token from bin indices via mixed-base encoding.

    itrade = (i_action * n_side * n_depth * n_vol * n_iat)
           + (i_side * n_depth * n_vol * n_iat)
           + (i_depth * n_vol * n_iat)
           + (i_vol * n_iat)
           + i_iat
    """
    nd = cfg.n_bins_price_depth
    nv = cfg.n_bins_volume
    nt = cfg.n_bins_interarrival

    df = df.with_columns(
        trade_token=(
            pl.col("order_action").cast(pl.Int32) * (2 * nd * nv * nt)
            + pl.col("side").cast(pl.Int32) * (nd * nv * nt)
            + pl.col("bin_price_depth").cast(pl.Int32) * (nv * nt)
            + pl.col("bin_volume").cast(pl.Int32) * nt
            + pl.col("bin_interarrival").cast(pl.Int32)
        )
    )

    return df


def decode_trade_token(token: int, cfg: PipelineConfig) -> dict:
    """Decode composite token back to feature bin indices."""
    nd = cfg.n_bins_price_depth
    nv = cfg.n_bins_volume
    nt = cfg.n_bins_interarrival

    i_action, rem = divmod(token, 2 * nd * nv * nt)
    i_side, rem = divmod(rem, nd * nv * nt)
    i_depth, rem = divmod(rem, nv * nt)
    i_vol, i_iat = divmod(rem, nt)

    return {
        "action": i_action,
        "side": i_side,
        "price_depth": i_depth,
        "volume": i_vol,
        "interarrival": i_iat,
    }


def _clip_percentile(arr: np.ndarray, lo_pct: float, hi_pct: float) -> np.ndarray:
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return arr
    lo_val = np.percentile(arr, lo_pct) if lo_pct > 0 else arr.min()
    hi_val = np.percentile(arr, hi_pct) if hi_pct < 100 else arr.max()
    result = arr[(arr >= lo_val) & (arr <= hi_val)]
    if len(result) == 0:
        return arr  # fallback: return unclipped
    return result


def _safe_digitize(values: np.ndarray, edges: np.ndarray, n_bins: int) -> np.ndarray:
    """Digitize values into [0, n_bins-1], clamping outliers to edge bins."""
    indices = np.digitize(values, edges[1:-1])  # n_bins-1 internal edges -> [0, n_bins-1]
    return np.clip(indices, 0, n_bins - 1)
