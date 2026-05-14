"""Avellaneda-Stoikov backtest with model-driven μ/σ/κ in hftbacktest L3.

Same simulator and logging as `backtest_orig_as.py`, but the A-S inputs come
from the trained transformer + decision heads instead of constants.

Time-alignment: hftbacktest works in ns-from-epoch on `exch_ts`; our tokens
parquet stores `time_sec` (seconds from midnight). At each decision we
compute `current_time_sec = 36000 + (now_ns - t_first_ns) / 1e9` and take the
trailing `context_length` tokens with `time_sec ≤ current_time_sec` — that's
the model's view at that moment.

Run:
    PYTHONPATH=. uv run python -m scripts.backtest_our_as \\
        --transformer-checkpoint /tmp/tradefm_smoke/transformer/best.pt \\
        --heads-checkpoint /tmp/tradefm_smoke/heads/best_heads.pt \\
        --interval 60-120
"""

import argparse
import json
from pathlib import Path

import hftbacktest as hb
import numpy as np
import polars as pl
import torch

from src.decision.avellaneda_stoikov import avellaneda_stoikov_quotes
from src.decision.heads import DecisionModule
from src.models.transformer import OrderFlowTransformer


SECONDS_FROM_MIDNIGHT_AT_T_FIRST = 36000.0  # 10:00:00 in seconds


def parse_interval(s: str) -> tuple[float, float | None]:
    if "-" not in s:
        raise argparse.ArgumentTypeError(f"--interval must be 'start-end' (got {s!r})")
    a, b = s.split("-", 1)
    return float(a) if a else 0.0, float(b) if b else None


def round_to_tick(price: float, tick: float) -> int:
    return int(round(price / tick))


def main():
    p = argparse.ArgumentParser(description="Model-driven A-S backtest in hftbacktest L3")
    # What
    p.add_argument("--instrument", default="SBER")
    p.add_argument("--date", default="2024-03-22")
    p.add_argument("--data-dir", default="data/hftbacktest")
    p.add_argument("--tokens-parquet", default="data/processed/tokens.parquet")
    p.add_argument("--interval", type=parse_interval, default="0-")
    # Model
    p.add_argument("--transformer-checkpoint", required=True)
    p.add_argument("--heads-checkpoint", required=True)
    p.add_argument("--device", default="cpu")
    # A-S risk aversion (the only A-S param we still pick by hand; μ/σ/κ from heads)
    p.add_argument("--gamma", type=float, default=0.1)
    # Strategy mechanics
    p.add_argument("--quote-refresh-ms", type=float, default=1000.0)
    p.add_argument("--order-qty", type=float, default=1.0)
    p.add_argument("--max-inventory", type=int, default=50)
    p.add_argument("--flatten-min-before-end", type=float, default=5.0)
    # Venue
    p.add_argument("--tick-size", type=float, default=0.01)
    p.add_argument("--lot-size", type=float, default=1.0)
    p.add_argument("--latency-ms", type=float, default=20.0)
    p.add_argument("--maker-fee", type=float, default=-0.00005)
    p.add_argument("--taker-fee", type=float, default=0.00050)
    # Output
    p.add_argument("--output-dir", default="runs/backtest_our_as")
    p.add_argument("--run-name", default=None)
    args = p.parse_args()

    interval_start_min, interval_end_min = args.interval
    quote_refresh_ns = int(args.quote_refresh_ms * 1e6)
    latency_ns = int(args.latency_ms * 1e6)
    device = torch.device(args.device)

    npz_path = Path(args.data_dir) / args.instrument / f"{args.date}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    # --- Load model ---------------------------------------------------------
    ck_xfmr = torch.load(args.transformer_checkpoint, map_location=device, weights_only=False)
    model_cfg = ck_xfmr["config"]
    transformer = OrderFlowTransformer(model_cfg).to(device)
    transformer.load_state_dict(ck_xfmr["model_state_dict"])
    transformer.eval()
    for pp in transformer.parameters():
        pp.requires_grad = False
    heads = DecisionModule(model_cfg.d_model).to(device)
    ck_heads = torch.load(args.heads_checkpoint, map_location=device, weights_only=False)
    heads.load_state_dict(ck_heads["heads_state_dict"])
    heads.eval()
    print(f"Transformer: d_model={model_cfg.d_model}, context_length={model_cfg.context_length}")

    # --- Load tokens stream for the instrument ------------------------------
    tokens_df = (
        pl.read_parquet(args.tokens_parquet)
        .filter(pl.col("SECCODE") == args.instrument)
        .sort("time_sec")
    )
    if len(tokens_df) == 0:
        raise RuntimeError(f"No tokens for {args.instrument} in {args.tokens_parquet}")
    inst_id = sorted(pl.read_parquet(args.tokens_parquet)["SECCODE"].unique().to_list()).index(args.instrument)
    tok_time = tokens_df["time_sec"].to_numpy()
    tok_trade = tokens_df["trade_token"].to_numpy().astype(np.int64)
    tok_pl = tokens_df["bin_price_level"].to_numpy().astype(np.int64)
    tok_liq = tokens_df["bin_liquidity"].to_numpy().astype(np.int64)
    print(f"Tokens for {args.instrument}: {len(tokens_df):,}, time_sec [{tok_time[0]:.0f}, {tok_time[-1]:.0f}], inst_id={inst_id}")

    # --- Build hftbacktest --------------------------------------------------
    raw = np.load(str(npz_path), allow_pickle=False)
    arr = raw[raw.files[0]]
    t_first = int(arr["exch_ts"][0])
    t_last = int(arr["exch_ts"][-1])
    t_interval_start = t_first + int(interval_start_min * 60 * 1e9)
    t_interval_end = t_last if interval_end_min is None else t_first + int(interval_end_min * 60 * 1e9)
    t_flatten = t_interval_end - int(args.flatten_min_before_end * 60 * 1e9)
    print(f"Session: {(t_last - t_first)/1e9/60:.1f} min total")
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
        if hbt.elapse(t_interval_start - t_first) != 0:
            print("Backtest ended during skip-to-interval.")
            return

    inst_id_t = torch.tensor([inst_id], dtype=torch.long, device=device)
    ctx = model_cfg.context_length

    # --- Logs / state -------------------------------------------------------
    ts_log, bb_log, ba_log, obid_log, oask_log = [], [], [], [], []
    pos_log, bal_log, fee_log = [], [], []
    mu_log, sigma_log, kappa_log = [], [], []
    fill_ts, fill_side, fill_qty, fill_px, fill_inv = [], [], [], [], []

    bid_oid = ask_oid = None
    next_oid = 0
    BID_OID_BASE, ASK_OID_BASE, EOD_OID_BASE = 10**8, 2 * 10**8, 9 * 10**8

    prev_position = prev_balance = 0.0
    n_quotes_placed = n_steps = n_forwards = 0

    while True:
        if hbt.elapse(quote_refresh_ns) != 0:
            break
        n_steps += 1
        now = hbt.current_timestamp
        if now > t_interval_end:
            break

        depth = hbt.depth(0)
        bbt, bat = depth.best_bid_tick, depth.best_ask_tick
        if bbt <= 0 or bat == 2**31 - 1 or bat <= bbt:
            continue
        best_bid = bbt * args.tick_size
        best_ask = bat * args.tick_size
        mid = 0.5 * (best_bid + best_ask)

        position = float(hbt.position(0))
        sv = hbt.state_values(0)
        balance, fee = float(sv.balance), float(sv.fee)

        # Detect fills via position delta
        delta_pos = position - prev_position
        if delta_pos != 0:
            delta_bal = balance - prev_balance
            px = -delta_bal / delta_pos
            side = "buy" if delta_pos > 0 else "sell"
            fill_ts.append(now); fill_side.append(side); fill_qty.append(abs(delta_pos))
            fill_px.append(px); fill_inv.append(position)
        prev_position, prev_balance = position, balance

        # EOD unwind
        if now >= t_flatten:
            if bid_oid is not None: hbt.cancel(0, bid_oid, False); bid_oid = None
            if ask_oid is not None: hbt.cancel(0, ask_oid, False); ask_oid = None
            if abs(position) > 0:
                next_oid += 1
                oid = EOD_OID_BASE + next_oid
                if position > 0:
                    hbt.submit_sell_order(0, oid, best_bid, abs(position), hb.GTC, hb.LIMIT, False)
                else:
                    hbt.submit_buy_order(0, oid, best_ask, abs(position), hb.GTC, hb.LIMIT, False)
            ts_log.append(now); bb_log.append(best_bid); ba_log.append(best_ask)
            obid_log.append(float("nan")); oask_log.append(float("nan"))
            pos_log.append(position); bal_log.append(balance); fee_log.append(fee)
            mu_log.append(float("nan")); sigma_log.append(float("nan")); kappa_log.append(float("nan"))
            continue

        # --- Run model on current context ---
        cur_time_sec = SECONDS_FROM_MIDNIGHT_AT_T_FIRST + (now - t_first) / 1e9
        # Largest index i where tok_time[i] <= cur_time_sec
        i_end = int(np.searchsorted(tok_time, cur_time_sec, side="right"))
        i_start = max(0, i_end - ctx)
        if i_end - i_start < 2:
            # Not enough context — skip this decision
            ts_log.append(now); bb_log.append(best_bid); ba_log.append(best_ask)
            obid_log.append(float("nan")); oask_log.append(float("nan"))
            pos_log.append(position); bal_log.append(balance); fee_log.append(fee)
            mu_log.append(float("nan")); sigma_log.append(float("nan")); kappa_log.append(float("nan"))
            continue

        tok_t = torch.from_numpy(tok_trade[i_start:i_end]).unsqueeze(0).to(device)
        plv_t = torch.from_numpy(tok_pl[i_start:i_end]).unsqueeze(0).to(device)
        liq_t = torch.from_numpy(tok_liq[i_start:i_end]).unsqueeze(0).to(device)
        with torch.no_grad():
            hidden = transformer.extract_hidden_states(tok_t, plv_t, liq_t, inst_id_t)
            preds = heads(hidden)  # dict (1, T)
        mu_v = preds["alpha"][0, -1].item()
        sigma_v = preds["risk"][0, -1].item()
        kappa_v = preds["intensity"][0, -1].item()
        n_forwards += 1

        # A-S quotes
        out = avellaneda_stoikov_quotes(
            mid=torch.tensor(mid),
            mu=torch.tensor(mu_v),
            sigma=torch.tensor(sigma_v),
            kappa=torch.tensor(kappa_v),
            inventory=torch.tensor(position),
            gamma=args.gamma,
        )
        bid_px, ask_px = out["bid"].item(), out["ask"].item()

        place_bid = position < args.max_inventory
        place_ask = position > -args.max_inventory
        bid_tick = min(round_to_tick(bid_px, args.tick_size), bbt)
        ask_tick = max(round_to_tick(ask_px, args.tick_size), bat)

        if bid_oid is not None: hbt.cancel(0, bid_oid, False); bid_oid = None
        if ask_oid is not None: hbt.cancel(0, ask_oid, False); ask_oid = None

        our_bid_px = our_ask_px = float("nan")
        if place_bid:
            next_oid += 1
            bid_oid = BID_OID_BASE + next_oid
            px = bid_tick * args.tick_size
            hbt.submit_buy_order(0, bid_oid, px, args.order_qty, hb.GTC, hb.LIMIT, False)
            our_bid_px = px
            n_quotes_placed += 1
        if place_ask:
            next_oid += 1
            ask_oid = ASK_OID_BASE + next_oid
            px = ask_tick * args.tick_size
            hbt.submit_sell_order(0, ask_oid, px, args.order_qty, hb.GTC, hb.LIMIT, False)
            our_ask_px = px
            n_quotes_placed += 1

        ts_log.append(now); bb_log.append(best_bid); ba_log.append(best_ask)
        obid_log.append(our_bid_px); oask_log.append(our_ask_px)
        pos_log.append(position); bal_log.append(balance); fee_log.append(fee)
        mu_log.append(mu_v); sigma_log.append(sigma_v); kappa_log.append(kappa_v)

    # --- Output -------------------------------------------------------------
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
        "mu": mu_log, "sigma": sigma_log, "kappa": kappa_log,
    }).write_parquet(out_dir / "steps.parquet")

    pl.DataFrame({
        "ts": fill_ts, "side": fill_side, "qty": fill_qty,
        "price": fill_px, "inventory_after": fill_inv,
    }).write_parquet(out_dir / "fills.parquet")

    summary = {
        "instrument": args.instrument, "date": args.date,
        "interval_start_min": interval_start_min,
        "interval_end_min": interval_end_min,
        "n_steps": n_steps, "n_forwards": n_forwards,
        "n_quotes_placed": n_quotes_placed, "n_fills": len(fill_ts),
        "final_position": final_pos, "final_balance": final_balance,
        "final_fee": final_fee, "pnl": final_balance,
        "mean_position": float(np.mean(pos_log)) if pos_log else 0.0,
        "max_abs_position": float(max(abs(p) for p in pos_log)) if pos_log else 0.0,
        "mean_book_spread": float(np.mean([a - b for a, b in zip(ba_log, bb_log)])) if bb_log else 0.0,
        "model": {
            "transformer_checkpoint": args.transformer_checkpoint,
            "heads_checkpoint": args.heads_checkpoint,
            "context_length": model_cfg.context_length,
            "d_model": model_cfg.d_model,
        },
        "as_params": {"gamma": args.gamma,
                      "mu": "from heads", "sigma": "from heads", "kappa": "from heads"},
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Steps: {n_steps}  Forwards: {n_forwards}  Quotes: {n_quotes_placed}  Fills: {len(fill_ts)}")
    print(f"Final position: {final_pos}")
    print(f"PnL (balance): {final_balance:.2f}  (fees {final_fee:.4f})")
    print(f"Output: {out_dir}/")
    hbt.close()


if __name__ == "__main__":
    main()
