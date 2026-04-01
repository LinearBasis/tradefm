"""Main data pipeline: raw CSV -> tokenized sequences."""

import os
import time

import polars as pl

from src.config import PipelineConfig
from src.data.loader import load_raw
from src.data.features import (
    compute_ewvwap,
    compute_open_price,
    compute_daily_volume,
    build_order_features,
)
from src.data.tokenizer import (
    BinEdges,
    calibrate_bins,
    digitize_features,
    compose_trade_token,
)


def run_pipeline(cfg: PipelineConfig | None = None) -> tuple[pl.DataFrame, BinEdges]:
    """Execute full pipeline. Returns tokenized DataFrame and calibrated bin edges."""
    if cfg is None:
        cfg = PipelineConfig()

    t0 = time.time()

    # 1. Load & filter
    print("--- Loading data ---")
    df = load_raw(cfg)

    # 2. Mid-price estimation
    print("--- Computing EW-VWAP ---")
    df = compute_ewvwap(df, cfg)
    df = compute_open_price(df)
    df = compute_daily_volume(df)

    # 3. Feature engineering (filters to orders only)
    print("--- Building features ---")
    orders = build_order_features(df, cfg)

    # 4. Calibrate bins
    print("--- Calibrating tokenizer ---")
    edges = calibrate_bins(orders, cfg)

    # 5. Digitize & compose tokens
    print("--- Tokenizing ---")
    orders = digitize_features(orders, edges, cfg)
    orders = compose_trade_token(orders, cfg)

    # 6. Build sequences per instrument
    sequences = build_sequences(orders, cfg)

    elapsed = time.time() - t0
    print(f"--- Done in {elapsed:.1f}s ---")
    print(f"Tokens: {orders.height:,}, Vocab: {cfg.vocab_size:,}")
    print(f"Instruments: {orders['SECCODE'].n_unique()}")
    print(f"Sequences: {len(sequences)}")

    # 7. Save
    os.makedirs(cfg.output_dir, exist_ok=True)

    orders.select([
        "NO", "SECCODE", "time_sec",
        "trade_token", "bin_price_level", "bin_liquidity",
        # raw features for debugging / analysis
        "interarrival", "price_depth", "log_volume",
        "order_action", "side", "mid_price",
    ]).write_parquet(os.path.join(cfg.output_dir, "tokens.parquet"))

    save_sequences(sequences, cfg)

    return orders, edges


def build_sequences(df: pl.DataFrame, cfg: PipelineConfig) -> dict[str, pl.DataFrame]:
    """Split tokenized data into per-instrument sequences, sorted by time."""
    sequences = {}
    for seccode in df["SECCODE"].unique().sort().to_list():
        seq = df.filter(pl.col("SECCODE") == seccode).sort("NO")
        sequences[seccode] = seq
    return sequences


def save_sequences(sequences: dict[str, pl.DataFrame], cfg: PipelineConfig) -> None:
    """Save per-instrument sequences as parquet files."""
    seq_dir = os.path.join(cfg.output_dir, "sequences")
    os.makedirs(seq_dir, exist_ok=True)

    cols = ["trade_token", "bin_price_level", "bin_liquidity"]

    for seccode, seq in sequences.items():
        seq.select(cols).write_parquet(os.path.join(seq_dir, f"{seccode}.parquet"))


if __name__ == "__main__":
    run_pipeline()
