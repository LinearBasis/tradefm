"""Convert MOEX OrderLog CSV files to hftbacktest L3 npz format.

Output layout:
    <output_dir>/<SECCODE>/<YYYY-MM-DD>.npz   # one numpy structured array per (instr, day)

The npz key is "data" (matches hftbacktest convention for ``add_file``).

Mapping rules (see notes/moex_orderlog_docs.pdf and project memory
"Market orders decision"):
    ACTION=1, BUYSELL=B, PRICE>0  -> ADD_ORDER_EVENT | BUY_EVENT
    ACTION=1, BUYSELL=S, PRICE>0  -> ADD_ORDER_EVENT | SELL_EVENT
    ACTION=0                      -> CANCEL_ORDER_EVENT (side from BUYSELL)
    ACTION=2                      -> FILL_EVENT (side from BUYSELL, px = TRADEPRICE)

    All ev's get | EXCH_EVENT | LOCAL_EVENT.
    local_ts = exch_ts + LATENCY_NS (synthetic, MOEX has no receive_ts).

Drop list (ROWS WHOSE ORDERNO is a market order, i.e. had ACTION=1 with PRICE=0):
    - the ACTION=1 with PRICE=0 itself
    - all ACTION=2 rows for this ORDERNO (own-side fills, also have PRICE=0)
    - the ACTION=0 row for this ORDERNO (remainder cancel) if any

Resting-side ACTION=2 rows for trades against market orders are KEPT (different
ORDERNO, real PRICE) — they alone carry trade information into the simulator.

Sorting: stable by (TIME, NO) — NO is the original row number in the OrderLog.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

import hftbacktest as hb

# Synthetic local-side latency (MOEX OrderLog has no receive_ts)
LATENCY_NS = 10_000_000  # 10 ms


# ---------- field-level helpers ----------

def _parse_time_to_ns_offset(time_col: pl.Expr) -> pl.Expr:
    """HHMMSSZZZXXX (12-digit int: 2H 2M 2S 3ms 3us, MOEX since Mar 2016)
    -> nanoseconds offset within the day.
    """
    t = time_col.cast(pl.Utf8).str.zfill(12)
    hh = t.str.slice(0, 2).cast(pl.Int64)
    mm = t.str.slice(2, 2).cast(pl.Int64)
    ss = t.str.slice(4, 2).cast(pl.Int64)
    ms = t.str.slice(6, 3).cast(pl.Int64)
    us = t.str.slice(9, 3).cast(pl.Int64)
    return ((hh * 3600 + mm * 60 + ss) * 1_000_000_000
            + ms * 1_000_000
            + us * 1_000)


def _date_to_epoch_ns(date_str: str) -> int:
    """YYYY-MM-DD -> ns since Unix epoch (UTC midnight).

    NOTE: MOEX timestamps are exchange-local (Moscow). For our purposes the
    simulator only needs monotonically increasing ns; we treat the day boundary
    as UTC midnight which is consistent within a per-day npz.
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000_000)


def _date_from_filename(path: Path) -> str:
    """OrderLog20240318.txt -> '2024-03-18'."""
    m = re.search(r"(\d{8})", path.stem)
    if not m:
        raise ValueError(f"cannot extract date from {path.name}")
    d = m.group(1)
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}"


# ---------- core conversion ----------

CSV_DTYPES = {
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


def _compute_ev_numpy(action: np.ndarray, buysell: np.ndarray) -> np.ndarray:
    """Bitfield ev = type | side | EXCH | LOCAL, computed in uint64.

    Doing this in numpy avoids Polars Int32 overflow on EXCH_EVENT=0x80000000
    (silent null promotion that produced ev=0 for every row).
    """
    n = len(action)
    out = np.full(n, np.uint64(int(hb.EXCH_EVENT) | int(hb.LOCAL_EVENT)), dtype=np.uint64)
    # type
    type_codes = np.where(
        action == 1, np.uint64(int(hb.ADD_ORDER_EVENT)),
        np.where(action == 0, np.uint64(int(hb.CANCEL_ORDER_EVENT)),
                 np.uint64(int(hb.FILL_EVENT))),
    ).astype(np.uint64)
    # side
    side_codes = np.where(
        buysell == "B",
        np.uint64(int(hb.BUY_EVENT)),
        np.uint64(int(hb.SELL_EVENT)),
    ).astype(np.uint64)
    return out | type_codes | side_codes


def convert_day(
    path: Path,
    instruments: list[str],
    output_dir: Path,
    latency_ns: int = LATENCY_NS,
    overwrite: bool = False,
) -> dict:
    """Convert one OrderLog file. Writes one npz per (instrument, day).

    Returns a stats dict.
    """
    date = _date_from_filename(path)
    base_ns = _date_to_epoch_ns(date)
    print(f"[{date}] {path.name}: scanning...")

    lf = (
        pl.scan_csv(str(path), schema_overrides=CSV_DTYPES)
        .filter(pl.col("SECCODE").is_in(instruments))
    )

    # Pass 1: collect ORDERNOs that are market orders (ACTION=1 & PRICE=0)
    market_ids_lf = (
        lf.filter((pl.col("ACTION") == 1) & (pl.col("PRICE") == 0))
        .select("ORDERNO")
        .unique()
    )

    # Pass 2: filter rows, build ev / px / qty / order_id / ts, group by SECCODE
    # We compute everything in a single collect per file (polars optimizes the
    # graph: market_ids gets materialized once, then anti-joined).
    df = (
        lf.join(market_ids_lf, on="ORDERNO", how="anti")
        .with_columns(
            time_ns=_parse_time_to_ns_offset(pl.col("TIME")) + base_ns,
        )
        .with_columns(
            local_ns=pl.col("time_ns") + latency_ns,
            # FILL: use TRADEPRICE; otherwise PRICE
            px=pl.when(pl.col("ACTION") == 2)
            .then(pl.col("TRADEPRICE"))
            .otherwise(pl.col("PRICE")),
            qty=pl.col("VOLUME").cast(pl.Float64),
            order_id=pl.col("ORDERNO").cast(pl.UInt64),
            tradeno_for_audit=pl.col("TRADENO").fill_null(0).cast(pl.Int64),
        )
        .select(
            "SECCODE",
            "NO",
            "ACTION",
            "BUYSELL",
            pl.col("time_ns").alias("exch_ts"),
            pl.col("local_ns").alias("local_ts"),
            "px",
            "qty",
            "order_id",
            "tradeno_for_audit",
        )
        # stable order: by (exch_ts, NO). NO is monotonic in the source file
        # within the same TIME bucket — preserves exchange order for ties.
        .sort(["SECCODE", "exch_ts", "NO"])
        .collect()
    )

    print(f"[{date}] kept {len(df):,} rows after market-order filter")

    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"date": date, "by_instrument": {}}

    for inst in instruments:
        inst_dir = output_dir / inst
        inst_dir.mkdir(parents=True, exist_ok=True)
        out_path = inst_dir / f"{date}.npz"
        if out_path.exists() and not overwrite:
            print(f"  skip {inst} (exists)")
            continue

        sub = df.filter(pl.col("SECCODE") == inst)
        n = len(sub)
        if n == 0:
            print(f"  skip {inst} (empty)")
            continue

        arr = np.zeros(n, dtype=hb.event_dtype)
        arr["ev"] = _compute_ev_numpy(
            sub["ACTION"].to_numpy(),
            sub["BUYSELL"].to_numpy(),
        )
        arr["exch_ts"] = sub["exch_ts"].to_numpy().astype(np.int64)
        arr["local_ts"] = sub["local_ts"].to_numpy().astype(np.int64)
        arr["px"] = sub["px"].to_numpy().astype(np.float64)
        arr["qty"] = sub["qty"].to_numpy().astype(np.float64)
        arr["order_id"] = sub["order_id"].to_numpy().astype(np.uint64)
        arr["ival"] = sub["tradeno_for_audit"].to_numpy().astype(np.int64)
        # fval: leave 0

        np.savez(out_path, data=arr)
        stats["by_instrument"][inst] = n
        print(f"  wrote {inst}: {n:,} events -> {out_path}")

    return stats


def main():
    import argparse

    from src.config import PipelineConfig
    from src.data.loader import discover_files, select_instruments

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data/hftbacktest")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = PipelineConfig()
    files = discover_files(cfg)
    print(f"Discovered {len(files)} files")
    instruments = select_instruments(files, cfg)
    print(f"Selected {len(instruments)} instruments: {instruments}\n")

    output_dir = Path(args.output_dir)
    all_stats = []
    for path, _date in files:
        all_stats.append(
            convert_day(path, instruments, output_dir, overwrite=args.overwrite)
        )

    print("\n=== Summary ===")
    for s in all_stats:
        total = sum(s["by_instrument"].values())
        print(f"  {s['date']}: {total:>12,} events across {len(s['by_instrument'])} instruments")


if __name__ == "__main__":
    main()
