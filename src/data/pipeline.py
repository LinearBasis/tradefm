"""Main data pipeline: raw CSV(s) -> tokenized per-instrument-per-day sequences.

Processes one day at a time. Days are independent after calibration, so the
two heavy stages (calibration features, full per-day processing) parallelise
across days via a ProcessPoolExecutor when `cfg.n_jobs > 1`.
"""

import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import polars as pl
from tqdm import tqdm

from src.config import PipelineConfig
from src.data.features import (
    build_order_features,
    compute_daily_volume,
    compute_ewvwap,
    compute_open_price,
)
from src.data.loader import discover_files, load_day, select_instruments
from src.data.tokenizer import (
    BinEdges,
    calibrate_bins,
    compose_trade_token,
    digitize_features,
)
from src.decision.targets import add_targets_to_orders


# --------------------------------------------------------------------------- #
# Worker initialisation                                                        #
# --------------------------------------------------------------------------- #

def _worker_init(polars_threads: int) -> None:
    """Run once per worker process before any task.

    Polars reads POLARS_MAX_THREADS at import time, so this only takes effect
    when workers are launched via the *spawn* start method (fresh interpreter).
    """
    os.environ["POLARS_MAX_THREADS"] = str(max(1, polars_threads))


# --------------------------------------------------------------------------- #
# Per-day functions (top-level so they pickle cleanly into worker processes)  #
# --------------------------------------------------------------------------- #

def _calibrate_day(
    path: str, date: str, instruments: list[str], cfg: PipelineConfig
) -> pl.DataFrame:
    """Load + featurise one calibration day. Returns the order-level frame
    used by `calibrate_bins` (concatenated across days in the parent)."""
    df = load_day(path, date, instruments, cfg)
    df = compute_ewvwap(df, cfg)
    df = compute_open_price(df)
    df = compute_daily_volume(df)
    return build_order_features(df, cfg)


def _process_day(
    path: str,
    date: str,
    instruments: list[str],
    edges: BinEdges,
    cfg: PipelineConfig,
    train_set: set[str],
    val_set: set[str],
    seq_dir: str,
) -> tuple[str, dict[str, dict], int, int]:
    """Full pipeline for a single day: features → tokenize → write per-
    instrument parquets. Returns (date, manifest_entries, n_day_tokens, n_rows).
    """
    df = load_day(path, date, instruments, cfg)
    n_rows = df.height

    df = compute_ewvwap(df, cfg)
    df = compute_open_price(df)
    df = compute_daily_volume(df)
    orders = build_order_features(df, cfg)

    orders = digitize_features(orders, edges, cfg)
    orders = compose_trade_token(orders, cfg)

    token_cols = ["trade_token", "bin_price_level", "bin_liquidity"]
    target_cols = ["target_alpha", "target_risk", "target_intensity"]
    session_len = cfg.continuous_end_sec - cfg.continuous_start_sec

    entries: dict[str, dict] = {}
    n_day_tokens = 0

    for seccode in sorted(instruments):
        sec_orders = orders.filter(pl.col("SECCODE") == seccode).sort("NO")
        if sec_orders.height == 0:
            continue

        sec_full = df.filter(pl.col("SECCODE") == seccode).sort("NO")
        seq = add_targets_to_orders(
            sec_orders, sec_full, tau=512, session_length=session_len,
        )

        key = f"{seccode}_{date}"
        seq.select(token_cols + target_cols).write_parquet(
            os.path.join(seq_dir, f"{key}.parquet")
        )

        split = "train" if date in train_set else ("val" if date in val_set else "unknown")
        n_valid = seq.filter(pl.col("target_alpha").is_not_null()).height
        entries[key] = {
            "split": split,
            "n_tokens": seq.height,
            "n_valid_targets": n_valid,
        }
        n_day_tokens += seq.height

    return date, entries, n_day_tokens, n_rows


# --------------------------------------------------------------------------- #
# Parallel runners                                                             #
# --------------------------------------------------------------------------- #

def _run_parallel(fn, tasks, n_jobs, polars_threads, desc):
    """Dispatch `fn(*task)` over a process pool with a tqdm bar.

    Falls back to a serial loop when n_jobs == 1, which avoids spawn overhead
    and surfaces tracebacks directly (handy on the cluster).
    """
    results = []
    if n_jobs <= 1:
        for task in tqdm(tasks, desc=desc, unit="day"):
            results.append(fn(*task))
        return results

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=n_jobs,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(polars_threads,),
    ) as pool:
        futures = {pool.submit(fn, *task): task for task in tasks}
        with tqdm(total=len(tasks), desc=desc, unit="day") as bar:
            for fut in as_completed(futures):
                results.append(fut.result())
                bar.update(1)
    return results


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

def run_pipeline(cfg: PipelineConfig | None = None) -> BinEdges:
    """Execute full pipeline. Returns calibrated bin edges."""
    if cfg is None:
        cfg = PipelineConfig()

    t0 = time.time()
    n_jobs = max(1, int(cfg.n_jobs))
    pthreads = max(1, int(cfg.polars_threads_per_worker))

    # 1. Discover files and select instruments ------------------------------
    print("--- Discovering data ---")
    files = discover_files(cfg)
    sorted_dates = [date for _, date in files]
    print(f"  Found {len(files)} day(s): {sorted_dates[0]} .. {sorted_dates[-1]}")
    print(f"  Parallelism: n_jobs={n_jobs}, POLARS_MAX_THREADS/worker={pthreads}")

    instruments = select_instruments(files, cfg)

    # 2. Date split ---------------------------------------------------------
    n_dates = len(sorted_dates)
    val_days = cfg.val_days
    if n_dates <= val_days:
        print(f"  Warning: only {n_dates} day(s) but val_days={val_days}. Using 1 day for val.")
        val_days = min(1, n_dates - 1)

    train_dates = sorted_dates[: n_dates - val_days] if val_days > 0 else sorted_dates
    val_dates = sorted_dates[n_dates - val_days:] if val_days > 0 else []
    print(f"  Train: {len(train_dates)} day(s), Val: {len(val_dates)} day(s)")

    train_set, val_set = set(train_dates), set(val_dates)

    # 3. Calibrate bins (training days only, parallel) ----------------------
    print("--- Calibrating tokenizer ---")
    cal_days = cfg.calibration_days
    cal_dates = train_dates[:cal_days] if cal_days is not None else train_dates
    cal_files = [(p, d) for p, d in files if d in set(cal_dates)]

    cal_tasks = [(p, d, instruments, cfg) for p, d in cal_files]
    cal_orders = _run_parallel(
        _calibrate_day, cal_tasks, n_jobs, pthreads, desc="Calibrate",
    )
    edges = calibrate_bins(pl.concat(cal_orders), cfg)
    del cal_orders  # free memory before processing stage

    # 4. Per-day processing (parallel) --------------------------------------
    print("--- Processing days ---")
    seq_dir = os.path.join(cfg.output_dir, "sequences")
    os.makedirs(seq_dir, exist_ok=True)

    proc_tasks = [
        (p, d, instruments, edges, cfg, train_set, val_set, seq_dir)
        for p, d in files
    ]
    results = _run_parallel(
        _process_day, proc_tasks, n_jobs, pthreads, desc="Tokenize",
    )

    # 5. Aggregate manifest -------------------------------------------------
    manifest = {
        "train_dates": train_dates,
        "val_dates": val_dates,
        "instruments": sorted(instruments),
        "sequences": {},
    }
    total_tokens = 0
    total_rows = 0
    for date, entries, n_tokens, n_rows in results:
        manifest["sequences"].update(entries)
        total_tokens += n_tokens
        total_rows += n_rows

    with open(os.path.join(seq_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"--- Done in {elapsed:.1f}s ---")
    print(f"Total rows: {total_rows:,}  →  Total tokens: {total_tokens:,}")
    print(f"Vocab: {cfg.vocab_size:,} | Instruments: {len(instruments)} | Days: {n_dates}")
    print(f"Sequences: {len(manifest['sequences'])}  → {seq_dir}/manifest.json")

    return edges


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run tradefm data pipeline")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="Number of worker processes for per-day stages")
    parser.add_argument("--polars-threads", type=int, default=None,
                        help="POLARS_MAX_THREADS in each worker")
    parser.add_argument("--raw-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = PipelineConfig()
    if args.n_jobs is not None:
        cfg.n_jobs = args.n_jobs
    if args.polars_threads is not None:
        cfg.polars_threads_per_worker = args.polars_threads
    if args.raw_dir is not None:
        cfg.raw_dir = args.raw_dir
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

    run_pipeline(cfg)
