"""CLI: npz L2-снапшоты + parquet trades → pickle потока MdUpdate для симулятора."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import List

import numpy as np
import polars as pl

from simulator.simulator import MdUpdate, OrderbookSnapshotUpdate, AnonTrade  # noqa: E402


def load_snapshots(npz_path: Path, md_latency_ns: int = 10_000_000,
                   offset: int = 0, length: int | None = None) -> list[MdUpdate]:
    z = np.load(npz_path, allow_pickle=False)
    data = z["data"]
    end = len(data) if length is None else min(offset + length, len(data))
    data = data[offset:end]
    out: list[MdUpdate] = []
    for rec in data:
        ts = int(rec["ts"])
        asks = list(zip(rec["A"].tolist(), rec["vA"].tolist()))
        bids = list(zip(rec["B"].tolist(), rec["vB"].tolist()))
        ob = OrderbookSnapshotUpdate(
            exchange_ts=ts, receive_ts=ts + md_latency_ns, asks=asks, bids=bids
        )
        out.append(MdUpdate(exchange_ts=ts, receive_ts=ts + md_latency_ns, orderbook=ob, trade=None))
    return out


MSK_OFFSET_NS = 3 * 3600 * 1_000_000_000


def _time_to_ns_offset(t: int) -> int:
    H = t // 10_000_000_000
    M = (t // 100_000_000) % 100
    S = (t // 1_000_000) % 100
    us = t % 1_000_000
    return (((H * 60 + M) * 60 + S) * 1_000_000 + us) * 1_000


def load_trades(
    parquet_path: Path,
    seccode: str,
    date_iso: str,
    md_latency_ns: int = 10_000_000,
) -> list[MdUpdate]:
    epoch_ns = int(np.datetime64(date_iso, "ns").astype(np.int64)) - MSK_OFFSET_NS
    df = (
        pl.scan_parquet(parquet_path)
        .filter((pl.col("SECCODE") == seccode) & (pl.col("ACTION") == 2))
        .select(["TIME", "BUYSELL", "TRADEPRICE", "VOLUME", "TRADENO"])
        .unique(subset=["TRADENO"], keep="first")
        .collect()
    )
    if df.height == 0:
        return []
    times = df["TIME"].to_numpy()
    sides_resting = df["BUYSELL"].to_numpy()
    prices = df["TRADEPRICE"].to_numpy()
    vols = df["VOLUME"].to_numpy()

    out: list[MdUpdate] = []
    for i in range(df.height):
        ts = epoch_ns + _time_to_ns_offset(int(times[i]))
        agg = "ASK" if sides_resting[i] == "B" else "BID"
        tr = AnonTrade(exchange_ts=ts, receive_ts=ts + md_latency_ns,
                       side=agg, size=float(vols[i]), price=float(prices[i]))
        out.append(MdUpdate(exchange_ts=ts, receive_ts=ts + md_latency_ns,
                            orderbook=None, trade=tr))
    return out


def build_md_stream(
    npz_path: Path,
    parquet_path: Path,
    seccode: str,
    date_iso: str,
    md_latency_ns: int = 10_000_000,
    limit: int | None = None,
    snap_offset: int = 0,
    snap_length: int | None = None,
) -> list[MdUpdate]:
    books = load_snapshots(npz_path, md_latency_ns=md_latency_ns,
                            offset=snap_offset, length=snap_length)
    trades = load_trades(parquet_path, seccode, date_iso, md_latency_ns=md_latency_ns)
    if books and trades:
        lo, hi = books[0].exchange_ts, books[-1].exchange_ts
        trades = [t for t in trades if lo <= t.exchange_ts <= hi]
    md = books + trades
    md.sort(key=lambda x: (x.exchange_ts, x.receive_ts, 0 if x.orderbook is not None else 1))
    if limit is not None:
        md = md[:limit]
    return md


if __name__ == "__main__":
    import argparse, json, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, type=Path)
    ap.add_argument("--parquet", required=True, type=Path)
    ap.add_argument("--seccode", required=True)
    ap.add_argument("--date", required=True, help="YYYY-MM-DD (MSK)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--snap-offset", type=int, default=0)
    ap.add_argument("--snap-length", type=int, default=None)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    t0 = time.perf_counter()
    md = build_md_stream(args.npz, args.parquet, args.seccode, args.date, limit=args.limit,
                         snap_offset=args.snap_offset, snap_length=args.snap_length)
    print(json.dumps({"n_md": len(md), "elapsed_sec": round(time.perf_counter()-t0, 1)}))
    import pickle
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(md, f)
