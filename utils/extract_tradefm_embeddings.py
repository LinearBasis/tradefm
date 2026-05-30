"""CLI: подготовка эмбеддингов TradeFM по MD-потоку (одно значение на секунду)."""
from __future__ import annotations

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import argparse
import json
import pickle
import time

import numpy as np
import polars as pl
import torch

from src.config import ModelConfig, PipelineConfig  # noqa: E402
from src.data.features import (  # noqa: E402
    build_order_features, compute_daily_volume, compute_ewvwap, compute_open_price,
)
from src.data.loader import _base_scan  # noqa: E402
from src.data.tokenizer import Tokenizer  # noqa: E402
from src.models.transformer import OrderFlowTransformer  # noqa: E402


MSK_OFFSET_SEC = 3 * 3600


def load_tokenize_day(
    orderlog_path: Path, instruments: list[str], date: str, tokenizer: Tokenizer,
    cfg: PipelineConfig,
) -> pl.DataFrame:
    """Load + featurize + tokenize one day. Returns DataFrame with columns
    [time_sec, trade_token, bin_price_level, bin_liquidity, SECCODE]
    sorted by time_sec.
    """
    print(f"[tokenize] loading {orderlog_path}...", flush=True)
    t0 = time.time()
    df = (
        _base_scan(orderlog_path, date, cfg)
        .filter(pl.col("SECCODE").is_in(instruments))
        .collect()
    )
    print(f"  loaded {df.height:,} rows in {time.time() - t0:.1f}s", flush=True)
    t0 = time.time()
    df = compute_ewvwap(df, cfg)
    df = compute_open_price(df)
    df = compute_daily_volume(df)
    orders = build_order_features(df, cfg)
    print(f"  features built ({orders.height:,} orders) in {time.time() - t0:.1f}s", flush=True)
    t0 = time.time()
    tokens = tokenizer.transform(orders)
    print(f"  tokenized in {time.time() - t0:.1f}s", flush=True)
    cols = ["time_sec", "trade_token", "bin_price_level", "bin_liquidity", "SECCODE"]
    out = tokens.select(cols).sort("time_sec")
    return out


def md_pkl_unique_seconds(md_path: Path) -> np.ndarray:
    """Load md pkl, return unique exchange_ts ns truncated to integer second (UTC)."""
    print(f"[md] loading {md_path}...", flush=True)
    t0 = time.time()
    md = pickle.load(open(md_path, "rb"))
    print(f"  {len(md):,} updates in {time.time() - t0:.1f}s", flush=True)
    ts_ns = np.array([u.exchange_ts for u in md], dtype=np.int64)
    sec_ns = (ts_ns // int(1e9)) * int(1e9)  # truncate to second
    uniq = np.unique(sec_ns)
    print(f"  {len(uniq):,} unique seconds", flush=True)
    return uniq


def ns_to_sec_of_day_msk(ts_ns: np.ndarray) -> np.ndarray:
    """ns since epoch UTC → seconds since midnight MSK."""
    sec = ts_ns.astype(np.float64) / 1e9
    return (sec + MSK_OFFSET_SEC) % 86400.0


def build_windows(
    tokens_inst: pl.DataFrame, ts_sec_unique_ns: np.ndarray, context_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """For each target ts (unix-ns, second-aligned), find rightmost event-index
    in tokens whose time_sec_msk <= ts_sec_msk. Return (event_idx[N], mask[N]).

    Skips ts entries with no preceding event (mask=False).
    """
    time_sec_msk = tokens_inst["time_sec"].to_numpy()  # already sec-of-day MSK
    ts_sec_msk = ns_to_sec_of_day_msk(ts_sec_unique_ns)
    # Right-bisect: rightmost token with time_sec <= ts. For tie use idx-1 of >.
    # np.searchsorted with side='right' gives insertion point after equal → idx-1 = last <= ts.
    ins = np.searchsorted(time_sec_msk, ts_sec_msk, side="right")
    event_idx = ins - 1
    valid = event_idx >= 0
    return event_idx, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orderlog", type=Path, required=True)
    ap.add_argument("--md", type=Path, nargs="+", required=True,
                    help="One or more md_stream pkl paths")
    ap.add_argument("--instrument", type=str, default="YNDX")
    ap.add_argument("--date", type=str, required=True, help="YYYY-MM-DD")
    ap.add_argument("--ckpt", type=Path,
                    default=Path("checkpoints/best_20m.pt"))
    ap.add_argument("--tokenizer", type=Path,
                    default=Path("checkpoints/tokenizer.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("embeddings"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-threads", type=int, default=8)
    ap.add_argument("--layer", type=int, default=-1, help="hidden layer to extract (-1=last)")
    ap.add_argument("--bf16", action="store_true", help="use bf16 inference if supported")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.num_threads)
    print(f"torch threads: {torch.get_num_threads()}", flush=True)

    # --- Load model ---
    print(f"[model] loading {args.ckpt}", flush=True)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg: ModelConfig = ck["config"]
    cfg.use_compile = False
    model = OrderFlowTransformer(cfg)
    model.load_state_dict(ck["model_state_dict"], strict=False)
    model.eval()
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    if args.bf16:
        model = model.to(dtype)
    print(f"  d_model={cfg.d_model}  ctx={cfg.context_length}  vocab={cfg.vocab_size}  "
          f"val_loss={ck.get('val_loss')}  params={sum(p.numel() for p in model.parameters()):,}",
          flush=True)
    CTX = cfg.context_length

    # --- Load tokenizer ---
    print(f"[tokenizer] loading {args.tokenizer}", flush=True)
    pcfg = PipelineConfig()
    tokenizer = Tokenizer.load(args.tokenizer, cfg=pcfg)
    if args.instrument not in tokenizer.instruments:
        raise RuntimeError(f"{args.instrument} not in tokenizer instruments: "
                           f"{tokenizer.instruments}")
    inst_id = sorted(tokenizer.instruments).index(args.instrument)
    print(f"  {args.instrument} → instrument_id={inst_id}", flush=True)

    # --- Tokenize day ---
    tokens_day = load_tokenize_day(
        args.orderlog, [args.instrument], args.date, tokenizer, pcfg,
    )
    # Keep only the target instrument (in case tokenize covers multiple)
    tokens_inst = tokens_day.filter(pl.col("SECCODE") == args.instrument).sort("time_sec")
    print(f"[tokens] {tokens_inst.height:,} events for {args.instrument} "
          f"({tokens_inst['time_sec'][0]:.1f}s .. {tokens_inst['time_sec'][-1]:.1f}s)",
          flush=True)
    trade_tokens = tokens_inst["trade_token"].to_numpy().astype(np.int64)
    price_levels = tokens_inst["bin_price_level"].to_numpy().astype(np.int64)
    liquidities = tokens_inst["bin_liquidity"].to_numpy().astype(np.int64)
    # Sanity: token values within vocab
    assert trade_tokens.max() < cfg.vocab_size, \
        f"trade_token max {trade_tokens.max()} >= vocab {cfg.vocab_size}"
    assert price_levels.max() < cfg.n_price_level_bins
    assert liquidities.max() < cfg.n_liquidity_bins

    # --- For each md pkl, build per-second embedding table ---
    inst_id_t = torch.zeros(1, dtype=torch.long)  # filled with inst_id below

    for md_path in args.md:
        key = md_path.stem
        out_path = args.out_dir / f"{key}__emb.npz"
        print(f"\n=== {key} ===", flush=True)

        ts_unique_ns = md_pkl_unique_seconds(md_path)
        event_idx, valid = build_windows(tokens_inst, ts_unique_ns, CTX)
        n_total = len(ts_unique_ns)
        n_valid = int(valid.sum())
        print(f"[align] {n_valid}/{n_total} seconds have ≥1 preceding event "
              f"(skipping {n_total - n_valid} early)", flush=True)
        if n_valid == 0:
            print("  no valid windows, skipping", flush=True)
            continue

        ts_ns_v = ts_unique_ns[valid]
        ev_idx_v = event_idx[valid]
        emb = np.zeros((n_valid, cfg.d_model), dtype=np.float32)

        # Build batches
        B = args.batch_size
        t_start = time.time()
        last_print = t_start

        with torch.inference_mode():
            for bstart in range(0, n_valid, B):
                bend = min(bstart + B, n_valid)
                idxs = ev_idx_v[bstart:bend]
                # Each window: tokens from [max(0, idx-CTX+1), idx+1]; left-pad with zeros
                trade_batch = np.zeros((bend - bstart, CTX), dtype=np.int64)
                plev_batch = np.zeros((bend - bstart, CTX), dtype=np.int64)
                liq_batch = np.zeros((bend - bstart, CTX), dtype=np.int64)
                for k, idx in enumerate(idxs):
                    end = idx + 1
                    start = max(0, end - CTX)
                    L = end - start
                    trade_batch[k, -L:] = trade_tokens[start:end]
                    plev_batch[k, -L:] = price_levels[start:end]
                    liq_batch[k, -L:] = liquidities[start:end]
                trade_t = torch.from_numpy(trade_batch)
                plev_t = torch.from_numpy(plev_batch)
                liq_t = torch.from_numpy(liq_batch)
                inst_t = torch.full((bend - bstart,), inst_id, dtype=torch.long)
                h = model.extract_latent(trade_t, plev_t, liq_t, inst_t, layer=args.layer)
                emb[bstart:bend] = h.float().cpu().numpy()

                # ETA / progress
                now = time.time()
                if now - last_print > 10.0 or bend == n_valid:
                    elapsed = now - t_start
                    rate = bend / max(elapsed, 1e-6)
                    eta_sec = (n_valid - bend) / max(rate, 1e-6)
                    print(f"  [{bend:>6d}/{n_valid}] {100*bend/n_valid:5.1f}%  "
                          f"{rate:6.1f} emb/s  elapsed={elapsed/60:.1f}min  "
                          f"ETA={eta_sec/60:.1f}min", flush=True)
                    last_print = now

        # Save
        np.savez(
            out_path,
            ts_ns=ts_ns_v.astype(np.int64),
            sec_of_day_msk=ns_to_sec_of_day_msk(ts_ns_v),
            event_idx=ev_idx_v.astype(np.int64),
            emb=emb,
            meta=np.array(json.dumps({
                "instrument": args.instrument,
                "date": args.date,
                "ckpt": str(args.ckpt),
                "tokenizer": str(args.tokenizer),
                "context_length": CTX,
                "d_model": cfg.d_model,
                "n_layers": cfg.n_layers,
                "layer_extracted": args.layer,
                "val_loss": ck.get("val_loss"),
                "n_md_seconds_total": int(n_total),
                "n_md_seconds_valid": int(n_valid),
                "n_orderlog_events": int(tokens_inst.height),
            }), dtype=object),
        )
        print(f"  wrote {out_path}  shape={emb.shape}  size={out_path.stat().st_size/1024:.1f}KB",
              flush=True)


if __name__ == "__main__":
    main()
