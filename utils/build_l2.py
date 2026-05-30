"""CLI: parquet OrderLog → npz L2-снапшотов глубины height."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import re
import time as _wall
from datetime import datetime, timezone, timedelta

import numpy as np
import polars as pl
from tqdm import tqdm

from utils.book import L2Book

MSK = timezone(timedelta(hours=3))
_DATE_RE = re.compile(r"OrderLog(\d{8})\.parquet$")


def parse_date_from_filename(path: Path) -> datetime:
    m = _DATE_RE.search(path.name)
    if not m:
        raise ValueError(f"cannot extract YYYYMMDD from {path.name}")
    return datetime.strptime(m.group(1), "%Y%m%d").replace(tzinfo=MSK)


def make_dtype(height: int) -> np.dtype:
    return np.dtype([
        ("ts", "i8"),
        ("A",  "f8", (height,)),
        ("vA", "f8", (height,)),
        ("B",  "f8", (height,)),
        ("vB", "f8", (height,)),
    ])


def time_to_ns_offset(t: int) -> int:
    H = t // 10_000_000_000
    M = (t // 100_000_000) % 100
    S = (t // 1_000_000) % 100
    us_part = t % 1_000_000
    return (((H * 60 + M) * 60 + S) * 1_000_000 + us_part) * 1_000


def build(
    parquet_path: Path,
    seccode: str,
    height: int,
    output_path: Path,
    progress: bool = True,
) -> dict:
    date = parse_date_from_filename(parquet_path)
    epoch_ns = int(date.astimezone(timezone.utc).timestamp() * 1_000_000_000)

    df = (
        pl.scan_parquet(parquet_path)
        .filter(pl.col("SECCODE") == seccode)
        .select(["TIME", "ACTION", "BUYSELL", "ORDERNO", "PRICE", "VOLUME"])
        .collect()
    )
    n_rows = df.height
    if n_rows == 0:
        raise RuntimeError(f"no rows for SECCODE={seccode} in {parquet_path}")

    times    = df["TIME"].to_numpy()
    actions  = df["ACTION"].to_numpy()
    sides    = df["BUYSELL"].to_numpy()
    orderno  = df["ORDERNO"].to_numpy()
    prices   = df["PRICE"].to_numpy()
    volumes  = df["VOLUME"].to_numpy()
    del df

    out = np.empty(n_rows, dtype=make_dtype(height))
    n_out = 0

    book = L2Book()
    prev_bbo: tuple | None = None
    current_time = -1
    pending_change = False
    snapshots_skipped_thin = 0

    iter_range = tqdm(range(n_rows), disable=not progress, unit="ev", smoothing=0.05)
    for i in iter_range:
        t = int(times[i])
        if t != current_time and current_time != -1 and pending_change:
            tup = book.top(height)
            if tup is not None:
                ap, asz, bp, bsz = tup
                ts_ns = epoch_ns + time_to_ns_offset(current_time)
                rec = out[n_out]
                rec["ts"] = ts_ns
                rec["A"]  = ap
                rec["vA"] = asz
                rec["B"]  = bp
                rec["vB"] = bsz
                n_out += 1
            else:
                snapshots_skipped_thin += 1
            pending_change = False

        book.apply(
            int(actions[i]),
            str(sides[i]),
            int(orderno[i]),
            float(prices[i]),
            int(volumes[i]),
        )
        current_time = t
        bbo = book.bbo()
        if bbo != prev_bbo:
            pending_change = True
            prev_bbo = bbo

    if pending_change:
        tup = book.top(height)
        if tup is not None:
            ap, asz, bp, bsz = tup
            ts_ns = epoch_ns + time_to_ns_offset(current_time)
            rec = out[n_out]
            rec["ts"] = ts_ns
            rec["A"]  = ap
            rec["vA"] = asz
            rec["B"]  = bp
            rec["vB"] = bsz
            n_out += 1
        else:
            snapshots_skipped_thin += 1

    out = out[:n_out].copy()

    meta = {
        "seccode": seccode,
        "date": date.date().isoformat(),
        "source": str(parquet_path),
        "height": height,
        "emit_policy": "on_top_change",
        "rows_in":  int(n_rows),
        "snapshots_out": int(n_out),
        "snapshots_skipped_thin": int(snapshots_skipped_thin),
        "stats": vars(book.stats),
        "ts_first": int(out["ts"][0]) if n_out > 0 else None,
        "ts_last":  int(out["ts"][-1]) if n_out > 0 else None,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, data=out, meta=np.array(json.dumps(meta)))
    return meta


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="build_l2")
    p.add_argument("--orderlog", required=True, type=Path)
    p.add_argument("--seccode", required=True)
    p.add_argument("--height", type=int, default=10)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--no-progress", action="store_true")
    args = p.parse_args(argv)

    t0 = _wall.perf_counter()
    meta = build(args.orderlog, args.seccode, args.height, args.output,
                 progress=not args.no_progress)
    elapsed = _wall.perf_counter() - t0

    print(json.dumps({**meta, "elapsed_sec": round(elapsed, 2)}, indent=2),
          file=sys.stderr)
    print(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
