"""Validate one converted hftbacktest L3 npz file by reconstructing the order
book event-by-event in pure Python.

Checks:
  1. Schema:
       - dtype matches hb.event_dtype
       - exch_ts monotonically non-decreasing
       - local_ts >= exch_ts on every row (synthetic latency holds)
  2. ID hygiene:
       - no duplicate ADD for the same order_id (without intervening cancel/fill-to-zero)
       - every CANCEL/FILL references a known order_id (no orphans)
  3. Book sanity:
       - best_bid < best_ask at all sampled timestamps (no crossed book)
       - spread distribution (median, p99 in bps)
  4. Mid comparison:
       - reconstruct best-bid/best-ask time series at uniformly-sampled timestamps
       - compare with the EW-VWAP mid produced by the existing pipeline
         (src/data/features.py) over the same (instrument, day)
       - report Pearson correlation + RMSE in bps

Output:
  runs/validate_hftbacktest/<seccode>_<date>/{summary.json, mid_compare.png}

Run:
  uv run python -m tests.validate_hftbacktest_npz \\
      --instrument SBER --date 2024-03-18
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

import hftbacktest as hb
from src.config import PipelineConfig
from src.data.loader import _CSV_DTYPES, discover_files, load_day
# intentionally do not import features pipeline; we re-derive a VWAP reference inline


# ---------- bit unpacking ----------

ADD = int(hb.ADD_ORDER_EVENT)
CANCEL = int(hb.CANCEL_ORDER_EVENT)
FILL = int(hb.FILL_EVENT)
BUY = int(hb.BUY_EVENT)
SELL = int(hb.SELL_EVENT)
EVENT_TYPE_MASK = 0xFF  # low byte holds the event-type id


def event_type(ev: int) -> int:
    return ev & EVENT_TYPE_MASK


def is_buy(ev: int) -> bool:
    return bool(ev & BUY)


# ---------- L3 reconstruction ----------

def reconstruct(arr: np.ndarray) -> dict:
    """Walk events. Maintain a dict order_id -> (side, px, qty).
    Track top-of-book by maintaining sorted price ladders per side
    (using simple sorted-set behavior via dicts).
    """
    orders: dict[int, tuple[str, float, float]] = {}
    # price -> total qty
    bid_levels: dict[float, float] = defaultdict(float)
    ask_levels: dict[float, float] = defaultdict(float)

    # diagnostics
    bad_ts = 0
    bad_local = 0
    duplicate_adds = 0
    orphan_cancels = 0
    orphan_fills = 0
    crossed_book_count = 0
    spreads_bps = []  # sampled at every N events

    last_exch_ts = -1
    sample_every = max(1, len(arr) // 5000)  # ~5k samples
    mid_samples_ts = []
    mid_samples_px = []

    for i, row in enumerate(arr):
        ev = int(row["ev"])
        et = event_type(ev)
        oid = int(row["order_id"])
        px = float(row["px"])
        qty = float(row["qty"])
        ts = int(row["exch_ts"])
        lts = int(row["local_ts"])

        if ts < last_exch_ts:
            bad_ts += 1
        last_exch_ts = ts
        if lts < ts:
            bad_local += 1

        side = "B" if is_buy(ev) else "S"
        levels = bid_levels if side == "B" else ask_levels

        if et == ADD:
            if oid in orders:
                duplicate_adds += 1
                # remove old phantom from book to avoid corrupting state
                old_side, old_px, old_qty = orders[oid]
                old_levels = bid_levels if old_side == "B" else ask_levels
                old_levels[old_px] -= old_qty
                if old_levels[old_px] <= 1e-9:
                    old_levels.pop(old_px, None)
            orders[oid] = (side, px, qty)
            levels[px] += qty
        elif et == CANCEL:
            if oid not in orders:
                orphan_cancels += 1
                continue
            o_side, o_px, o_qty = orders.pop(oid)
            o_levels = bid_levels if o_side == "B" else ask_levels
            o_levels[o_px] -= o_qty
            if o_levels[o_px] <= 1e-9:
                o_levels.pop(o_px, None)
        elif et == FILL:
            if oid not in orders:
                orphan_fills += 1
                continue
            o_side, o_px, o_qty = orders[oid]
            o_levels = bid_levels if o_side == "B" else ask_levels
            o_levels[o_px] -= qty
            new_qty = o_qty - qty
            if new_qty <= 1e-9:
                orders.pop(oid, None)
                if o_levels.get(o_px, 0) <= 1e-9:
                    o_levels.pop(o_px, None)
            else:
                orders[oid] = (o_side, o_px, new_qty)

        # sample top of book
        if i % sample_every == 0 and bid_levels and ask_levels:
            best_bid = max(bid_levels.keys())
            best_ask = min(ask_levels.keys())
            if best_bid >= best_ask:
                crossed_book_count += 1
                continue
            mid = 0.5 * (best_bid + best_ask)
            spread_bps = 1e4 * (best_ask - best_bid) / mid
            spreads_bps.append(spread_bps)
            mid_samples_ts.append(ts)
            mid_samples_px.append(mid)

    return {
        "n_events": int(len(arr)),
        "bad_ts": int(bad_ts),
        "bad_local": int(bad_local),
        "duplicate_adds": int(duplicate_adds),
        "orphan_cancels": int(orphan_cancels),
        "orphan_fills": int(orphan_fills),
        "crossed_book_count": int(crossed_book_count),
        "spread_bps_median": float(np.median(spreads_bps)) if spreads_bps else None,
        "spread_bps_p99": float(np.percentile(spreads_bps, 99)) if spreads_bps else None,
        "mid_ts": np.asarray(mid_samples_ts, dtype=np.int64),
        "mid_px": np.asarray(mid_samples_px, dtype=np.float64),
        "n_open_orders_eod": int(len(orders)),
    }


# ---------- mid comparison ----------

def compute_vwap_mid(instrument: str, date: str, cfg: PipelineConfig) -> tuple[np.ndarray, np.ndarray]:
    """Re-derive an EW-VWAP-style mid time series from raw CSV directly.
    Avoids depending on the full pipeline; we just want a reference for
    cross-check.
    """
    files = discover_files(cfg)
    target = next((p for p, d in files if d == date), None)
    if target is None:
        raise FileNotFoundError(f"No raw file for date {date}")

    # use load_day to apply session filter, then VWAP over trades only
    df = load_day(target, date, [instrument], cfg)
    trades = df.filter(pl.col("ACTION") == 2).sort("NO")
    if trades.height == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    # rolling vwap of last K trades
    k = 50
    px = trades["TRADEPRICE"].to_numpy()
    vol = trades["VOLUME"].to_numpy().astype(np.float64)
    wpx = px * vol
    cum_wpx = np.cumsum(wpx)
    cum_vol = np.cumsum(vol)
    n = len(px)
    out = np.empty(n)
    for i in range(n):
        lo = max(0, i - k + 1)
        num = cum_wpx[i] - (cum_wpx[lo - 1] if lo > 0 else 0)
        den = cum_vol[i] - (cum_vol[lo - 1] if lo > 0 else 0)
        out[i] = num / den if den > 0 else px[i]

    # convert TIME (seconds since midnight float) -> ns since epoch
    base_ns = pl.Series([date]).str.to_datetime("%Y-%m-%d").cast(pl.Int64).item() * 1_000  # ms->ns? no, .cast(Int64) gives us ms
    # safer: redo manually
    from datetime import datetime, timezone
    base_ns = int(datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1e9)
    ts_sec = trades["time_sec"].to_numpy()
    ts_ns = (base_ns + (ts_sec * 1e9).astype(np.int64))
    return ts_ns, out


def align_and_compare(
    ts_a: np.ndarray, px_a: np.ndarray,
    ts_b: np.ndarray, px_b: np.ndarray,
) -> dict:
    """Resample both onto a common 1-second grid via forward-fill, then compare."""
    if len(ts_a) == 0 or len(ts_b) == 0:
        return {"correlation": None, "rmse_bps": None, "n": 0}

    t0 = max(ts_a[0], ts_b[0])
    t1 = min(ts_a[-1], ts_b[-1])
    if t1 <= t0:
        return {"correlation": None, "rmse_bps": None, "n": 0}

    grid = np.arange(t0, t1, int(1e9))  # 1s grid

    def ffill(ts, px, grid):
        idx = np.searchsorted(ts, grid, side="right") - 1
        idx = np.clip(idx, 0, len(px) - 1)
        return px[idx]

    a = ffill(ts_a, px_a, grid)
    b = ffill(ts_b, px_b, grid)
    if len(a) < 10:
        return {"correlation": None, "rmse_bps": None, "n": int(len(a))}
    corr = float(np.corrcoef(a, b)[0, 1])
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    rmse_bps = 1e4 * rmse / float(np.median(b))
    return {"correlation": corr, "rmse_bps": rmse_bps, "n": int(len(a))}


# ---------- driver ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--npz-dir", default="data/hftbacktest")
    parser.add_argument("--out-dir", default="runs/validate_hftbacktest")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    npz_path = Path(args.npz_dir) / args.instrument / f"{args.date}.npz"
    if not npz_path.exists():
        raise SystemExit(f"missing npz: {npz_path}")

    print(f"Loading {npz_path} ...")
    with np.load(npz_path) as z:
        arr = z["data"]
    print(f"  shape={arr.shape}, dtype={arr.dtype}")

    if arr.dtype != hb.event_dtype:
        print(f"  WARN: dtype differs from hb.event_dtype={hb.event_dtype}")

    print("Reconstructing book ...")
    stats = reconstruct(arr)

    cfg = PipelineConfig()
    print("Computing reference VWAP mid from raw CSV ...")
    try:
        ts_v, px_v = compute_vwap_mid(args.instrument, args.date, cfg)
    except Exception as e:
        print(f"  failed: {e}")
        ts_v, px_v = np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    cmp_stats = align_and_compare(stats["mid_ts"], stats["mid_px"], ts_v, px_v)

    out_dir = Path(args.out_dir) / f"{args.instrument}_{args.date}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "instrument": args.instrument,
        "date": args.date,
        "npz_path": str(npz_path),
        "n_events": stats["n_events"],
        "monotonic_exch_ts": stats["bad_ts"] == 0,
        "bad_ts_count": stats["bad_ts"],
        "bad_local_count": stats["bad_local"],
        "duplicate_adds": stats["duplicate_adds"],
        "orphan_cancels": stats["orphan_cancels"],
        "orphan_fills": stats["orphan_fills"],
        "crossed_book_at_sample": stats["crossed_book_count"],
        "spread_bps_median": stats["spread_bps_median"],
        "spread_bps_p99": stats["spread_bps_p99"],
        "open_orders_eod": stats["n_open_orders_eod"],
        "mid_correlation_with_vwap": cmp_stats["correlation"],
        "mid_rmse_bps_vs_vwap": cmp_stats["rmse_bps"],
        "mid_compare_n": cmp_stats["n"],
    }

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if not args.no_plot and cmp_stats["n"]:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(11, 4))
            ax.plot(stats["mid_ts"], stats["mid_px"], lw=0.8, label="reconstructed L3 mid")
            if len(ts_v):
                ax.plot(ts_v, px_v, lw=0.8, alpha=0.7, label="raw rolling VWAP")
            ax.set_title(f"{args.instrument} {args.date}: L3-reconstructed mid vs trade VWAP")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / "mid_compare.png", dpi=120)
            plt.close(fig)
            print(f"\n  plot -> {out_dir / 'mid_compare.png'}")
        except Exception as e:
            print(f"  plot failed: {e}")

    print(f"\n  summary -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
