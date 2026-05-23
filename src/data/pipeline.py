"""Main data pipeline: raw CSV(s) -> tokenized per-instrument-per-day sequences.

Processes one day at a time. Days are independent after calibration, so the
two heavy stages (calibration features, full per-day processing) parallelise
across days via a ProcessPoolExecutor when `cfg.n_jobs > 1`.

Tokenizer state (edges + ADV) is persisted to `data/processed/tokenizer.json`
and can be reused across runs via `--reuse-tokenizer`.
"""

import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import polars as pl
from tqdm import tqdm

from src.config import PipelineConfig
from src.data.features import (
    build_order_features,
    compute_daily_volume,
    compute_ewvwap,
    compute_open_price,
)
from src.data.loader import _base_scan, discover_files, load_day, select_instruments
from src.data.tokenizer import Tokenizer


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

def _featurise_day(
    path: str, date: str, instruments: list[str], cfg: PipelineConfig
) -> pl.DataFrame:
    """Load + EW-VWAP + open + ADV-source + order features. Returns the order frame."""
    df = load_day(path, date, instruments, cfg)
    df = compute_ewvwap(df, cfg)
    df = compute_open_price(df)
    df = compute_daily_volume(df)
    return build_order_features(df, cfg)


def _calibration_payload(
    path: str, date: str, instruments: list[str], cfg: PipelineConfig
) -> tuple[pl.DataFrame, dict[str, float]]:
    """Per-day calibration payload: feature columns + per-instrument daily volume."""
    orders = _featurise_day(path, date, instruments, cfg)
    features = orders.select([
        "interarrival", "log_volume", "price_depth", "price_level",
    ])
    daily_vols = {
        row["SECCODE"]: float(row["dv"])
        for row in (
            orders.group_by("SECCODE")
            .agg(pl.col("daily_volume").first().alias("dv"))
            .iter_rows(named=True)
        )
    }
    return features, daily_vols


def _process_day(
    path: str,
    date: str,
    instruments: list[str],
    tokenizer: Tokenizer,
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
    orders = _featurise_day_from_loaded(df, cfg)
    orders = tokenizer.transform(orders)

    token_cols = ["trade_token", "bin_price_level", "bin_liquidity", "time_sec", "SECCODE"]
    entries: dict[str, dict] = {}
    n_day_tokens = 0

    for seccode in sorted(instruments):
        sec_orders = orders.filter(pl.col("SECCODE") == seccode).sort("NO")
        if sec_orders.height == 0:
            continue

        key = f"{seccode}_{date}"
        sec_orders.select(token_cols).write_parquet(
            os.path.join(seq_dir, f"{key}.parquet")
        )

        split = "train" if date in train_set else ("val" if date in val_set else "unknown")
        entries[key] = {
            "split": split,
            "n_tokens": sec_orders.height,
        }
        n_day_tokens += sec_orders.height

    return date, entries, n_day_tokens, n_rows


def _fit_tokenizer(
    files: list,
    train_dates: list[str],
    instruments: list[str],
    cfg: PipelineConfig,
    n_jobs: int,
    pthreads: int,
    save_path: Path,
) -> Tokenizer:
    """Run calibration stage in parallel, fit Tokenizer, persist to disk."""
    print("--- Calibrating tokenizer ---")
    cal_days = cfg.calibration_days
    cal_dates = train_dates[:cal_days] if cal_days is not None else train_dates
    cal_files = [(p, d) for p, d in files if d in set(cal_dates)]

    cal_tasks = [(p, d, instruments, cfg) for p, d in cal_files]
    cal_results = _run_parallel(
        _calibration_payload, cal_tasks, n_jobs, pthreads, desc="Calibrate",
    )
    feature_frames = [r[0] for r in cal_results]
    adv_per_day: dict[str, list[float]] = {}
    for _, day_adv in cal_results:
        for sec, dv in day_adv.items():
            adv_per_day.setdefault(sec, []).append(dv)

    tokenizer = Tokenizer.fit_from_features(
        pl.concat(feature_frames), adv_per_day, cfg,
    )
    tokenizer.save(save_path)
    print(f"  Saved tokenizer to {save_path}")
    return tokenizer


def _seccodes_in_files(files: list, cfg: PipelineConfig) -> set[str]:
    """Lightweight scan: union of SECCODEs present across all files (lazy)."""
    present: set[str] = set()
    for path, date in files:
        codes = (
            _base_scan(path, date, cfg)
            .select("SECCODE").unique().collect()["SECCODE"].to_list()
        )
        present.update(codes)
    return present


def _validate_instruments_present(
    files: list, instruments: list[str], cfg: PipelineConfig, *, strict: bool = False,
) -> None:
    """Verify all `instruments` appear in the raw data. Warn (or raise) on misses."""
    present = _seccodes_in_files(files, cfg)
    missing = [s for s in instruments if s not in present]
    if not missing:
        return
    head = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
    msg = (
        f"{len(missing)}/{len(instruments)} instruments missing from data: [{head}]. "
        f"They will be silently skipped during tokenisation."
    )
    if strict:
        raise RuntimeError(msg)
    print(f"  WARNING: {msg}")


def _featurise_day_from_loaded(df: pl.DataFrame, cfg: PipelineConfig) -> pl.DataFrame:
    """Same as _featurise_day but skips the load step (df already in memory)."""
    df = compute_ewvwap(df, cfg)
    df = compute_open_price(df)
    df = compute_daily_volume(df)
    return build_order_features(df, cfg)


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

def run_pipeline(
    cfg: PipelineConfig | None = None,
    reuse_tokenizer: bool = False,
    instruments_whitelist: list[str] | None = None,
) -> Tokenizer:
    """Execute full pipeline. Returns the fitted (or loaded) Tokenizer.

    Instrument selection priority:
      1. If `reuse_tokenizer` and `tokenizer.json` exists → use tokenizer's instruments.
      2. Else if `instruments_whitelist` provided → use it.
      3. Else → `select_instruments(files, cfg)` (top-N by event count).
    """
    if cfg is None:
        cfg = PipelineConfig()

    t0 = time.time()
    n_jobs = max(1, int(cfg.n_jobs))
    pthreads = max(1, int(cfg.polars_threads_per_worker))

    # 1. Discover files ------------------------------------------------------
    print("--- Discovering data ---")
    files = discover_files(cfg)
    sorted_dates = [date for _, date in files]
    print(f"  Found {len(files)} day(s): {sorted_dates[0]} .. {sorted_dates[-1]}")
    print(f"  Parallelism: n_jobs={n_jobs}, POLARS_MAX_THREADS/worker={pthreads}")

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

    seq_dir = os.path.join(cfg.output_dir, "sequences")
    os.makedirs(seq_dir, exist_ok=True)
    tokenizer_path = Path(cfg.output_dir) / "tokenizer.json"

    # 3. Resolve instruments + tokenizer (3 paths, see docstring) -----------
    if reuse_tokenizer and tokenizer_path.exists():
        print(f"--- Loading tokenizer from {tokenizer_path} ---")
        tokenizer = Tokenizer.load(tokenizer_path, cfg=cfg)
        instruments = tokenizer.instruments
        print(f"  Reusing {len(instruments)} instruments from tokenizer")
        if instruments_whitelist is not None:
            print(f"  NOTE: --instruments ignored (tokenizer is the source of truth)")
        _validate_instruments_present(files, instruments, cfg, strict=False)
    elif instruments_whitelist is not None:
        instruments = list(instruments_whitelist)
        print(f"--- Using {len(instruments)} whitelist instruments ---")
        _validate_instruments_present(files, instruments, cfg, strict=True)
        tokenizer = _fit_tokenizer(
            files, train_dates, instruments, cfg, n_jobs, pthreads, tokenizer_path,
        )
    else:
        instruments = select_instruments(files, cfg)
        tokenizer = _fit_tokenizer(
            files, train_dates, instruments, cfg, n_jobs, pthreads, tokenizer_path,
        )

    # 4. Per-day processing (parallel) --------------------------------------
    print("--- Processing days ---")
    proc_tasks = [
        (p, d, instruments, tokenizer, cfg, train_set, val_set, seq_dir)
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
        "session_length_sec": cfg.continuous_end_sec - cfg.continuous_start_sec,
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

    return tokenizer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run tradefm data pipeline")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="Number of worker processes for per-day stages")
    parser.add_argument("--polars-threads", type=int, default=None,
                        help="POLARS_MAX_THREADS in each worker")
    parser.add_argument("--raw-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--top-n-instruments", type=int, default=None,
                        help="Keep only top-N most active instruments (smoke: 2)")
    parser.add_argument("--reuse-tokenizer", action="store_true",
                        help="Skip calibration if data/processed/tokenizer.json exists. "
                             "Also reuses tokenizer's instrument list (overrides top-N selection).")
    parser.add_argument("--instruments", type=str, default=None,
                        help="Comma-separated SECCODE whitelist. Overrides top-N selection. "
                             "Ignored when --reuse-tokenizer is set.")
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
    if args.top_n_instruments is not None:
        cfg.top_n_instruments = args.top_n_instruments

    whitelist = [s.strip() for s in args.instruments.split(",")] if args.instruments else None
    run_pipeline(cfg, reuse_tokenizer=args.reuse_tokenizer, instruments_whitelist=whitelist)
