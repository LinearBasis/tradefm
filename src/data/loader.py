"""Load raw MOEX OrderLog CSV and parse fields."""

import polars as pl

from src.config import PipelineConfig


def parse_time_to_seconds(time_col: pl.Expr) -> pl.Expr:
    """HHMMSSXXXXXX -> float seconds from midnight."""
    t = time_col.cast(pl.Utf8).str.zfill(12)
    hh = t.str.slice(0, 2).cast(pl.Int32)
    mm = t.str.slice(2, 2).cast(pl.Int32)
    ss = t.str.slice(4, 2).cast(pl.Int32)
    us = t.str.slice(6, 6).cast(pl.Int64)
    return hh * 3600 + mm * 60 + ss + us / 1_000_000


def load_raw(cfg: PipelineConfig) -> pl.DataFrame:
    """Load CSV, parse time, filter instruments."""
    df = pl.read_csv(
        cfg.raw_path,
        dtypes={
            "NO": pl.Int64,
            "SECCODE": pl.Utf8,
            "BUYSELL": pl.Utf8,
            "TIME": pl.Int64,
            "ORDERNO": pl.Int64,
            "ACTION": pl.Int8,
            "PRICE": pl.Float64,
            "VOLUME": pl.Int64,
            "TRADENO": pl.Float64,
            "TRADEPRICE": pl.Float64,
        },
    )

    df = df.with_columns(time_sec=parse_time_to_seconds(pl.col("TIME")))

    # Filter instruments
    event_counts = df.group_by("SECCODE").len().rename({"len": "n_events"})

    if cfg.min_events_per_day > 0:
        event_counts = event_counts.filter(pl.col("n_events") >= cfg.min_events_per_day)

    if cfg.top_n_instruments is not None:
        event_counts = event_counts.sort("n_events", descending=True).head(
            cfg.top_n_instruments
        )

    selected = event_counts.select("SECCODE")
    df = df.join(selected, on="SECCODE", how="inner")

    print(
        f"Loaded {df.height:,} rows, "
        f"{df['SECCODE'].n_unique()} instruments"
    )
    return df
