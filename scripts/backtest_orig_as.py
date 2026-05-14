"""Avellaneda-Stoikov baseline backtest in hftbacktest L3.

Constants-only A-S (no model heads). Logs per-step state and per-fill events
to parquet for visualization in `notebooks/backtest_analysis.ipynb`.

Run:
    PYTHONPATH=. uv run python -m scripts.backtest_orig_as \\
        --instrument SBER --date 2024-03-22 --interval 60-120

`--interval` is `start-end` in minutes from the first event in the npz
(0 = session open). Use `--interval 60-` for "60 minutes after open until EoD".
"""

import argparse
import json
from pathlib import Path

import hftbacktest as hb
import numpy as np
import polars as pl

from src.decision.avellaneda_stoikov import avellaneda_stoikov_quotes
import torch


def parse_interval(s: str) -> tuple[float, float | None]:
    if "-" not in s:
        raise argparse.ArgumentTypeError(f"--interval must be 'start-end' (got {s!r})")
    a, b = s.split("-", 1)
    a = float(a) if a else 0.0
    b = float(b) if b else None
    return a, b


def round_to_tick(price: float, tick: float) -> int:
    return int(round(price / tick))


def main():
    p = argparse.ArgumentParser(description="A-S baseline backtest in hftbacktest L3")
    # What to backtest
    p.add_argument("--instrument", default="SBER")
    p.add_argument("--date", default="2024-03-22")
    p.add_argument("--data-dir", default="data/hftbacktest")
    p.add_argument("--interval", type=parse_interval, default="0-",
                   help="start-end minute offsets from first event (e.g. 60-120, 0-, 0-525)")
    # A-S formula constants
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--mu", type=float, default=0.0)
    p.add_argument("--sigma", type=float, default=1e-4)
    p.add_argument("--kappa", type=float, default=1.0)
    # Strategy mechanics
    p.add_argument("--quote-refresh-ms", type=float, default=1000.0)
    p.add_argument("--order-qty", type=float, default=1.0)
    p.add_argument("--max-inventory", type=int, default=50)
    p.add_argument("--flatten-min-before-end", type=float, default=5.0)
    # Market / venue
    p.add_argument("--tick-size", type=float, default=0.01)
    p.add_argument("--lot-size", type=float, default=1.0)
    p.add_argument("--latency-ms", type=float, default=20.0)
    p.add_argument("--maker-fee", type=float, default=-0.00005)
    p.add_argument("--taker-fee", type=float, default=0.00050)
    # Output
    p.add_argument("--output-dir", default="runs/backtest_orig_as")
    p.add_argument("--run-name", default=None)
    args = p.parse_args()

    interval_start_min, interval_end_min = args.interval
    quote_refresh_ns = int(args.quote_refresh_ms * 1e6)
    latency_ns = int(args.latency_ms * 1e6)

    npz_path = Path(args.data_dir) / args.instrument / f"{args.date}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    raw = np.load(str(npz_path), allow_pickle=False)
    arr = raw[raw.files[0]]
    t_first = int(arr["exch_ts"][0])
    t_last = int(arr["exch_ts"][-1])
    t_interval_start = t_first + int(interval_start_min * 60 * 1e9)
    t_interval_end = (
        t_last if interval_end_min is None
        else t_first + int(interval_end_min * 60 * 1e9)
    )
    t_flatten = t_interval_end - int(args.flatten_min_before_end * 60 * 1e9)
    print(f"Session: first {t_first} → last {t_last}  ({(t_last - t_first)/1e9/60:.1f} min)")
    print(f"Trading interval: +{interval_start_min:.0f} → "
          f"+{interval_end_min if interval_end_min is not None else 'eod'}m  "
          f"(flatten at +{(t_flatten - t_first)/1e9/60:.1f}m)")

    asset = (
        hb.BacktestAsset()
        .data([str(npz_path)])
        .linear_asset(1.0)
        .constant_order_latency(latency_ns, latency_ns)
        .l3_fifo_queue_model()
        .tick_size(args.tick_size)
        .lot_size(args.lot_size)
        .flat_per_trade_fee_model(args.maker_fee, args.taker_fee)
    )
    hbt = hb.HashMapMarketDepthBacktest([asset])

    if t_interval_start > t_first:
        skip_ns = t_interval_start - t_first
        if hbt.elapse(skip_ns) != 0:
            print("Backtest ended during skip-to-interval — no data for interval.")
            return

    # --- Logs ---
    ts_log: list[int] = []
    bb_log: list[float] = []
    ba_log: list[float] = []
    obid_log: list[float] = []
    oask_log: list[float] = []
    pos_log: list[float] = []
    bal_log: list[float] = []
    fee_log: list[float] = []

    fill_ts: list[int] = []
    fill_side: list[str] = []
    fill_qty: list[float] = []
    fill_px: list[float] = []
    fill_inv: list[float] = []

    # --- State ---
    bid_oid: int | None = None
    ask_oid: int | None = None
    next_oid_counter = 0
    BID_OID_BASE = 10**8
    ASK_OID_BASE = 2 * 10**8
    EOD_OID_BASE = 9 * 10**8

    mu = torch.tensor(args.mu)
    sigma = torch.tensor(args.sigma)
    kappa = torch.tensor(args.kappa)

    prev_position = 0.0
    prev_balance = 0.0

    n_quotes_placed = 0
    n_steps = 0

    while True:
        if hbt.elapse(quote_refresh_ns) != 0:
            break
        n_steps += 1
        now = hbt.current_timestamp
        if now > t_interval_end:
            break

        depth = hbt.depth(0)
        bbt = depth.best_bid_tick
        bat = depth.best_ask_tick
        if bbt <= 0 or bat == 2**31 - 1 or bat <= bbt:
            continue
        best_bid = bbt * args.tick_size
        best_ask = bat * args.tick_size
        mid = 0.5 * (best_bid + best_ask)

        position = float(hbt.position(0))
        sv = hbt.state_values(0)
        balance = float(sv.balance)
        fee = float(sv.fee)

        # Detect fills via position delta
        delta_pos = position - prev_position
        if delta_pos != 0:
            delta_bal = balance - prev_balance
            px = -delta_bal / delta_pos
            side = "buy" if delta_pos > 0 else "sell"
            fill_ts.append(now)
            fill_side.append(side)
            fill_qty.append(abs(delta_pos))
            fill_px.append(px)
            fill_inv.append(position)
        prev_position = position
        prev_balance = balance

        # End-of-interval unwind
        if now >= t_flatten:
            if bid_oid is not None:
                hbt.cancel(0, bid_oid, False); bid_oid = None
            if ask_oid is not None:
                hbt.cancel(0, ask_oid, False); ask_oid = None
            if abs(position) > 0:
                next_oid_counter += 1
                oid = EOD_OID_BASE + next_oid_counter
                if position > 0:
                    hbt.submit_sell_order(0, oid, best_bid, abs(position),
                                          hb.GTC, hb.LIMIT, False)
                else:
                    hbt.submit_buy_order(0, oid, best_ask, abs(position),
                                         hb.GTC, hb.LIMIT, False)
            ts_log.append(now); bb_log.append(best_bid); ba_log.append(best_ask)
            obid_log.append(float("nan")); oask_log.append(float("nan"))
            pos_log.append(position); bal_log.append(balance); fee_log.append(fee)
            continue

        # A-S quotes
        q = torch.tensor(position)
        out = avellaneda_stoikov_quotes(
            mid=torch.tensor(mid), mu=mu, sigma=sigma, kappa=kappa,
            inventory=q, gamma=args.gamma,
        )
        bid_px = out["bid"].item()
        ask_px = out["ask"].item()

        place_bid = position < args.max_inventory
        place_ask = position > -args.max_inventory

        bid_tick = min(round_to_tick(bid_px, args.tick_size), bbt)
        ask_tick = max(round_to_tick(ask_px, args.tick_size), bat)

        if bid_oid is not None:
            hbt.cancel(0, bid_oid, False); bid_oid = None
        if ask_oid is not None:
            hbt.cancel(0, ask_oid, False); ask_oid = None

        our_bid_px = float("nan")
        our_ask_px = float("nan")
        if place_bid:
            next_oid_counter += 1
            bid_oid = BID_OID_BASE + next_oid_counter
            px = bid_tick * args.tick_size
            hbt.submit_buy_order(0, bid_oid, px, args.order_qty,
                                 hb.GTC, hb.LIMIT, False)
            our_bid_px = px
            n_quotes_placed += 1
        if place_ask:
            next_oid_counter += 1
            ask_oid = ASK_OID_BASE + next_oid_counter
            px = ask_tick * args.tick_size
            hbt.submit_sell_order(0, ask_oid, px, args.order_qty,
                                  hb.GTC, hb.LIMIT, False)
            our_ask_px = px
            n_quotes_placed += 1

        ts_log.append(now); bb_log.append(best_bid); ba_log.append(best_ask)
        obid_log.append(our_bid_px); oask_log.append(our_ask_px)
        pos_log.append(position); bal_log.append(balance); fee_log.append(fee)

    # --- Final stats / output ---
    final_pos = float(hbt.position(0))
    sv = hbt.state_values(0)
    final_balance = float(sv.balance)
    final_fee = float(sv.fee)

    interval_label = (
        f"{int(interval_start_min):03d}-"
        f"{int(interval_end_min) if interval_end_min is not None else 'eod':>03}"
    )
    run_name = args.run_name or f"{args.instrument}_{args.date}_{interval_label}"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pl.DataFrame({
        "ts": ts_log,
        "best_bid": bb_log, "best_ask": ba_log,
        "our_bid": obid_log, "our_ask": oask_log,
        "position": pos_log,
        "balance": bal_log, "fee": fee_log,
    }).write_parquet(out_dir / "steps.parquet")

    pl.DataFrame({
        "ts": fill_ts, "side": fill_side, "qty": fill_qty,
        "price": fill_px, "inventory_after": fill_inv,
    }).write_parquet(out_dir / "fills.parquet")

    summary = {
        "instrument": args.instrument,
        "date": args.date,
        "interval_start_min": interval_start_min,
        "interval_end_min": interval_end_min,
        "n_steps": n_steps,
        "n_quotes_placed": n_quotes_placed,
        "n_fills": len(fill_ts),
        "final_position": final_pos,
        "final_balance": final_balance,
        "final_fee": final_fee,
        "pnl": final_balance,
        "mean_position": float(np.mean(pos_log)) if pos_log else 0.0,
        "max_abs_position": float(max(abs(p) for p in pos_log)) if pos_log else 0.0,
        "mean_book_spread": float(np.mean([a - b for a, b in zip(ba_log, bb_log)])) if bb_log else 0.0,
        "as_params": {"gamma": args.gamma,
                       "mu": args.mu, "sigma": args.sigma, "kappa": args.kappa},
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Steps:                 {n_steps}")
    print(f"Quotes placed:         {n_quotes_placed}")
    print(f"Fills:                 {len(fill_ts)}")
    print(f"Final position:        {final_pos}")
    print(f"PnL (balance):         {final_balance:.2f}  (fees {final_fee:.4f})")
    print(f"Output:                {out_dir}/")

    hbt.close()


if __name__ == "__main__":
    main()
