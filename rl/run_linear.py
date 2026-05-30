"""Запуск конфигураций A (DQN-Linear) и B (DQN-Linear-Oracle)."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import multiprocessing as mp
import os
import time
from dataclasses import dataclass

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np


TRAIN_PKL = "md_streams/YNDX_intraday_20240422_train.pkl"
TEST_PKL = "md_streams/YNDX_intraday_20240422_test.pkl"
DATE = "2024-04-22"
SEEDS = (1, 2, 3, 4, 5, 6, 7, 8)
INITIAL_INV = 1000.0
INVENTORY_MAX = 2000.0

TRAIN_MD = None
TEST_MD = None


@dataclass
class WorkerSpec:
    name: str
    hidden: tuple
    use_oracle: bool
    steps: int
    seed: int


def worker(spec: WorkerSpec) -> dict:
    import torch
    torch.set_num_threads(1)
    from rl.mm_env import MMEnv, EnvConfig
    from rl.mm_env_oracle import MMEnvOracle, OracleEnvConfig
    from rl.dqn import DQNConfig, DQNAgent, train_dqn, evaluate_dqn

    t0 = time.perf_counter()
    global TRAIN_MD, TEST_MD
    if TRAIN_MD is None:
        import pickle
        TRAIN_MD = pickle.load(open(TRAIN_PKL, "rb"))
        TEST_MD = pickle.load(open(TEST_PKL, "rb"))
    tr_md = TRAIN_MD
    te_md = TEST_MD

    env_cls = MMEnvOracle if spec.use_oracle else MMEnv
    cfg_cls = OracleEnvConfig if spec.use_oracle else EnvConfig
    env_cfg = cfg_cls(inventory_max=INVENTORY_MAX, trade_size=1.0,
                      step_delay_ns=int(1e9),
                      risk_alpha=0.05, initial_inventory=INITIAL_INV)
    env_tr = env_cls(tr_md, cfg=env_cfg)
    env_te = env_cls(te_md, cfg=env_cfg)

    dcfg = DQNConfig(
        hidden=spec.hidden, lr=1e-3, gamma=0.99,
        eps_start=0.9, eps_end=0.05, eps_decay_steps=spec.steps // 2,
        buffer_size=20_000, batch_size=64,
        target_update_every=500, warmup_steps=500, seed=spec.seed,
    )
    a = DQNAgent(state_dim=env_tr.state_dim, cfg=dcfg)
    t_tr = time.perf_counter()
    train_dqn(env_tr, a, max_steps=spec.steps)
    train_sec = time.perf_counter() - t_tr

    res = evaluate_dqn(env_te, a)

    def mid(u):
        ob = u.orderbook
        if ob is None or not ob.asks or not ob.bids:
            return None
        return (ob.asks[0][0] + ob.bids[0][0]) / 2.0
    m0 = next((mid(u) for u in te_md if mid(u) is not None), None)
    m1 = next((mid(u) for u in reversed(te_md) if mid(u) is not None), None)
    neutral = res["final_pnl"] - res["final_inv"] * (m1 - m0)

    return {
        "setup": spec.name, "date": DATE, "seed": spec.seed,
        "hidden": list(spec.hidden), "use_oracle": spec.use_oracle, "max_steps": spec.steps,
        "state_dim": env_tr.state_dim,
        "net_pnl": res["final_pnl"], "final_inv": res["final_inv"],
        "mid_start": m0, "mid_end": m1, "pnl_neutral": neutral,
        "n_trades": res["n_trades"],
        "train_sec": round(train_sec, 1),
        "total_sec": round(time.perf_counter() - t0, 1),
    }


def run_parallel(specs: list[WorkerSpec], n_workers: int, label: str,
                 out_path: Path, prior_rows: list[dict]) -> list[dict]:
    print(f"\n--- {label}: {len(specs)} tasks × {n_workers} workers ---", flush=True)
    rows = list(prior_rows)
    ctx = mp.get_context("fork")
    t0 = time.perf_counter()
    with ctx.Pool(n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(worker, specs), 1):
            rows.append(r)
            print(f"  [{i}/{len(specs)}] {r['setup']:<18s} s{r['seed']}  "
                  f"PnL={r['net_pnl']:+9.2f}  neutral={r['pnl_neutral']:+8.2f}  "
                  f"trades={r['n_trades']:>5}  inv={r['final_inv']:+6.1f}  "
                  f"train={r['train_sec']:.0f}s", flush=True)
            with open(out_path, "w") as f:
                json.dump({"results": rows}, f, indent=2)
                f.flush(); os.fsync(f.fileno())
    print(f"  wall: {(time.perf_counter() - t0)/60:.1f}min", flush=True)
    return rows


def aggregate(rows: list[dict], setup: str) -> dict:
    vals = [r["pnl_neutral"] for r in rows if r["setup"] == setup]
    trades = [r["n_trades"] for r in rows if r["setup"] == setup]
    arr = np.array(vals); tr = np.array(trades)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    cv = abs(std / arr.mean()) if abs(arr.mean()) > 1e-9 else float("inf")
    return {"n": len(arr), "mean": float(arr.mean()), "std": std, "cv": cv,
            "raw": vals, "trades_mean": float(tr.mean())}


def main():
    out_path = Path("results/iter/linear.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_workers = 2
    print(f"=== DQN-Linear на {DATE} intraday-сплите, {len(SEEDS)} сидов, "
          f"{n_workers} воркера ===", flush=True)

    import pickle
    global TRAIN_MD, TEST_MD
    t0 = time.perf_counter()
    print(f"  загружаю {TRAIN_PKL} + {TEST_PKL}...", flush=True)
    TRAIN_MD = pickle.load(open(TRAIN_PKL, "rb"))
    TEST_MD = pickle.load(open(TEST_PKL, "rb"))
    print(f"    train n={len(TRAIN_MD):,}  test n={len(TEST_MD):,}  за "
          f"{time.perf_counter() - t0:.1f}s", flush=True)

    p1_specs = [WorkerSpec(name="baseline_linear", hidden=(128, 64),
                           use_oracle=False, steps=200000, seed=s) for s in SEEDS]
    rows = run_parallel(p1_specs, n_workers, "Фаза 1: baseline_linear",
                        out_path, [])
    p1 = aggregate(rows, "baseline_linear")
    print(f"\n  baseline_linear: {p1['mean']:+.0f} ± {p1['std']:.0f}  "
          f"cv={p1['cv']:.2f}  trades_avg={p1['trades_mean']:.0f}", flush=True)

    p2_specs = [WorkerSpec(name="oracle_linear", hidden=(128, 64),
                           use_oracle=True, steps=200000, seed=s) for s in SEEDS]
    rows = run_parallel(p2_specs, n_workers, "Фаза 2: oracle_linear",
                        out_path, rows)
    p2 = aggregate(rows, "oracle_linear")
    print(f"\n  oracle_linear: {p2['mean']:+.0f} ± {p2['std']:.0f}  "
          f"cv={p2['cv']:.2f}  trades_avg={p2['trades_mean']:.0f}", flush=True)

    print("\n=== Сводка ===\n", flush=True)
    print(f"  {'setup':<22s}  {'mean':>10s}  {'std':>8s}  {'cv':>5s}  {'trades':>8s}  n")
    print(f"  {'baseline_linear':<22s}  {p1['mean']:>+10.0f}  {p1['std']:>+8.0f}  "
          f"{p1['cv']:>5.2f}  {p1['trades_mean']:>8.0f}  {p1['n']}")
    print(f"  {'oracle_linear':<22s}  {p2['mean']:>+10.0f}  {p2['std']:>+8.0f}  "
          f"{p2['cv']:>5.2f}  {p2['trades_mean']:>8.0f}  {p2['n']}")

    uplift = p2["mean"] - p1["mean"]
    print(f"\n  Oracle uplift: {uplift:+.0f}", flush=True)

    from math import sqrt
    if p1['n'] > 1 and p2['n'] > 1:
        se = sqrt(p1['std']**2 / p1['n'] + p2['std']**2 / p2['n'])
        t_stat = uplift / se if se > 0 else float('inf')
        print(f"  Welch's t-statistic: {t_stat:.2f}", flush=True)

    summary = {
        "baseline_linear": p1,
        "oracle_linear": p2,
        "oracle_uplift": float(uplift),
        "n_seeds": len(SEEDS),
    }
    with open(out_path, "w") as f:
        json.dump({"results": rows, "summary": summary}, f, indent=2)
        f.flush(); os.fsync(f.fileno())
    print(f"\nзаписано {out_path}")


if __name__ == "__main__":
    main()
