"""Load raw MOEX OrderLog CSV(s) and parse fields."""

import re
from pathlib import Path

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


_CSV_DTYPES = {
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
}


def extract_date_from_filename(path: str | Path) -> str:
    """Extract date from filename containing 8 consecutive digits (YYYYMMDD)."""
    name = Path(path).stem
    match = re.search(r"(\d{8})", name)
    if not match:
        raise ValueError(f"Cannot extract date from filename: {name}")
    d = match.group(1)
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}"


def _base_scan(path: Path, date: str, cfg: PipelineConfig) -> pl.LazyFrame:
    """Lazy-scan one CSV/TXT, add date, parse time, filter auctions."""
    return (
        pl.scan_csv(str(path), schema_overrides=_CSV_DTYPES)
        .with_columns(pl.lit(date).alias("date"))
        .with_columns(time_sec=parse_time_to_seconds(pl.col("TIME")))
        .filter(
            pl.col("time_sec").is_between(
                cfg.continuous_start_sec, cfg.continuous_end_sec
            )
        )
    )


def discover_files(cfg: PipelineConfig) -> list[tuple[Path, str]]:
    """Find data files and extract dates. Returns sorted list of (path, date)."""
    raw_dir = Path(cfg.raw_dir)
    data_files = sorted(
        list(raw_dir.glob("*.csv")) + list(raw_dir.glob("*.txt"))
    ) if raw_dir.is_dir() else []

    if data_files:
        return [(f, extract_date_from_filename(f)) for f in data_files]
    elif cfg.raw_path and Path(cfg.raw_path).exists():
        return [(Path(cfg.raw_path), "0000-00-00")]
    else:
        raise FileNotFoundError(
            f"No data files in {cfg.raw_dir} and raw_path={cfg.raw_path!r} not found"
        )


def select_instruments(
    files: list[tuple[Path, str]], cfg: PipelineConfig
) -> list[str]:
    """Determine top-N instruments by event count across all files (lazy, memory-efficient)."""
    count_frames = []
    for path, date in files:
        counts = (
            _base_scan(path, date, cfg)
            .group_by(["SECCODE", "date"])
            .len()
            .rename({"len": "n_events"})
        )
        count_frames.append(counts)

    all_counts = pl.concat(count_frames).collect()

    # Require min_events_per_day on every day
    if cfg.min_events_per_day > 0:
        instruments_below = (
            all_counts
            .filter(pl.col("n_events") < cfg.min_events_per_day)
            .select("SECCODE")
            .unique()
        )
        all_counts = all_counts.join(instruments_below, on="SECCODE", how="anti")

    # Rank by total
    totals = (
        all_counts
        .group_by("SECCODE")
        .agg(pl.col("n_events").sum().alias("total_events"))
        .sort("total_events", descending=True)
    )

    if cfg.top_n_instruments is not None:
        totals = totals.head(cfg.top_n_instruments)

    selected = totals["SECCODE"].to_list()
    print(f"  Selected {len(selected)} instruments")
    return selected


def load_day(
    path: Path, date: str, instruments: list[str], cfg: PipelineConfig
) -> pl.DataFrame:
    """Load a single day, filtering to selected instruments. Returns DataFrame with date column."""
    df = (
        _base_scan(path, date, cfg)
        .filter(pl.col("SECCODE").is_in(instruments))
        .collect()
    )
    return df
