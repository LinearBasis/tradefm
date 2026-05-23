"""Scale-invariant feature engineering following TradeFM."""

import polars as pl
import numpy as np

from src.config import PipelineConfig

try:
    import numba
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def compute_ewvwap(df: pl.DataFrame, cfg: PipelineConfig) -> pl.DataFrame:
    """Compute EW-VWAP mid-price estimate per instrument per day from trade executions.

    Uses exponentially-weighted volume-weighted average price with time-based halflife.
    EMA state resets at each day boundary.
    Result is joined back to all rows as 'mid_price'.
    """
    result_frames = []

    groups = (
        df.select(["SECCODE", "date"])
        .unique()
        .sort(["SECCODE", "date"])
        .iter_rows()
    )

    for seccode, date in groups:
        sec_df = df.filter(
            (pl.col("SECCODE") == seccode) & (pl.col("date") == date)
        ).sort("NO")

        time_sec = sec_df["time_sec"].to_numpy()
        action = sec_df["ACTION"].to_numpy()
        trade_price = sec_df["TRADEPRICE"].to_numpy()
        volume = sec_df["VOLUME"].to_numpy()
        nos = sec_df["NO"].to_numpy()

        # Cast volume to float64 once at call-site so the JIT specializes
        # on a single numeric type and doesn't recompile across instruments.
        mid_prices = _ewvwap_jit(
            time_sec.astype(np.float64, copy=False),
            action,
            trade_price.astype(np.float64, copy=False),
            volume.astype(np.float64, copy=False),
            float(cfg.ewvwap_halflife_sec),
        )

        result_frames.append(
            pl.DataFrame({
                "NO": nos,
                "date": [date] * len(nos),
                "mid_price": mid_prices,
            })
        )

    mid_df = pl.concat(result_frames)
    return df.join(mid_df, on=["NO", "date"], how="left")


def _ewvwap_core(
    time_sec: np.ndarray,
    action: np.ndarray,
    trade_price: np.ndarray,
    volume: np.ndarray,
    halflife: float,
) -> np.ndarray:
    """EW-VWAP per row, updated only on trade events (ACTION=2).

    mid_price[i] = EMA(price * volume) / EMA(volume) at row i.

    This is a tight scalar loop over 1D numpy arrays — exactly the shape
    that numba @njit compiles to native machine code (see `_ewvwap_jit` below).
    Kept as a plain-Python function so the fallback (when numba is missing)
    and the JIT-compiled version share one source of truth.
    """
    n = len(time_sec)
    mid = np.full(n, np.nan, dtype=np.float64)

    ema_pv = 0.0
    ema_v = 0.0
    last_trade_time = -1.0
    initialized = False

    for i in range(n):
        if action[i] == 2 and not np.isnan(trade_price[i]):
            p = trade_price[i]
            v = volume[i]
            if not initialized:
                ema_pv = p * v
                ema_v = v
                initialized = True
            else:
                dt = max(time_sec[i] - last_trade_time, 1e-9)
                alpha = 1.0 - 0.5 ** (dt / halflife)
                ema_pv = alpha * p * v + (1.0 - alpha) * ema_pv
                ema_v = alpha * v + (1.0 - alpha) * ema_v
            last_trade_time = time_sec[i]
        if initialized and ema_v > 0:
            mid[i] = ema_pv / ema_v

    return mid


# JIT-compiled version. `cache=True` writes the compiled binary to
# __pycache__/_ewvwap_core.<sig>.nbi/.nbc so subsequent ProcessPool workers
# don't recompile. `nogil=True` releases the GIL during execution.
if _HAS_NUMBA:
    _ewvwap_jit = numba.njit(cache=True, nogil=True)(_ewvwap_core)
else:
    _ewvwap_jit = _ewvwap_core


def compute_open_price(df: pl.DataFrame) -> pl.DataFrame:
    """Opening price per instrument per day = first trade's TRADEPRICE."""
    opens = (
        df.filter(pl.col("ACTION") == 2)
        .sort("NO")
        .group_by(["SECCODE", "date"])
        .first()
        .select(["SECCODE", "date", pl.col("TRADEPRICE").alias("open_price")])
    )
    return df.join(opens, on=["SECCODE", "date"], how="left")


def compute_daily_volume(df: pl.DataFrame) -> pl.DataFrame:
    """Total traded volume per instrument per day (for liquidity binning)."""
    adv = (
        df.filter(pl.col("ACTION") == 2)
        .group_by(["SECCODE", "date"])
        .agg(pl.col("VOLUME").sum().alias("daily_volume"))
    )
    return df.join(adv, on=["SECCODE", "date"], how="left")


def build_order_features(df: pl.DataFrame, cfg: PipelineConfig) -> pl.DataFrame:
    """Build scale-invariant features for add/cancel events.

    Filters to ACTION in (0, 1) — only orders, not trades.
    Computes per-instrument per-day:
      - interarrival: seconds since previous order event
      - price_depth: (PRICE - mid_price) / mid_price
      - log_volume: log(1 + VOLUME)
      - price_level: (mid_price - open_price) / open_price
      - side: 0=buy, 1=sell
      - action: 0=add, 1=cancel
    """
    orders = (
        df.filter(pl.col("ACTION").is_in([0, 1]))
        .filter(pl.col("mid_price").is_not_null())
        .sort(["SECCODE", "date", "NO"])
    )

    orders = orders.with_columns(
        # Interarrival time (per instrument per day)
        interarrival=pl.col("time_sec")
        .diff()
        .over(["SECCODE", "date"])
        .clip(lower_bound=0.0),
        # Normalized price depth
        price_depth=(pl.col("PRICE") - pl.col("mid_price")) / pl.col("mid_price"),
        # Log volume
        log_volume=(pl.col("VOLUME").cast(pl.Float64) + 1.0).log(),
        # Price level vs open
        price_level=(pl.col("mid_price") - pl.col("open_price")) / pl.col("open_price"),
        # Side: 0=buy, 1=sell
        side=(pl.col("BUYSELL") == "S").cast(pl.Int8),
        # Action: 0=add (ACTION=1), 1=cancel (ACTION=0)
        order_action=pl.when(pl.col("ACTION") == 1).then(0).otherwise(1).cast(pl.Int8),
    )

    # Drop first row per instrument per day (interarrival=null)
    orders = orders.filter(pl.col("interarrival").is_not_null())

    return orders
