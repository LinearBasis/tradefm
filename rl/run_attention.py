"""Запуск конфигураций C (DQN-Attn) и D (DQN-Attn-Oracle)."""
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
FEATURE_DEPTH = 10

TRAIN_MD = None
TEST_MD = None


@dataclass
class WorkerSpec:
    name: str
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
    env_cfg = cfg_cls(
        inventory_max=INVENTORY_MAX, trade_size=1.0,
        step_delay_ns=int(1e9),
        risk_alpha=0.05, initial_inventory=INITIAL_INV,
        feature_depth=FEATURE_DEPTH,
    )
    env_tr = env_cls(tr_md, cfg=env_cfg)
    env_te = env_cls(te_md, cfg=env_cfg)

    dcfg = DQNConfig(
        hidden=(128, 64), lr=1e-3, gamma=0.99,
        eps_start=0.9, eps_end=0.05, eps_decay_steps=spec.steps // 2,
        buffer_size=20_000, batch_size=64,
        target_update_every=500, warmup_steps=500, seed=spec.seed,
        qnet_kind="attention",
        attn_feature_depth=FEATURE_DEPTH, attn_d_model=16,
        attn_n_heads=4, attn_n_layers=1,
    )
    a = DQNAgent(state_dim=env_tr.state_dim, cfg=dcfg)
    n_params = sum(p.numel() for p in a.online.parameters())
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
        "qnet": "attention", "use_oracle": spec.use_oracle,
        "max_steps": spec.steps, "state_dim": env_tr.state_dim,
        "n_params": n_params,
        "net_pnl": res["final_pnl"], "final_inv": res["final_inv"],
        "mid_start": m0, "mid_end": m1, "pnl_neutral": neutral,
        "n_trades": res["n_trades"],
        "train_sec": round(train_sec, 1),
        "total_sec": round(time.perf_counter() - t0, 1),
    }


def run_parallel(specs, n_workers, label, out_path, prior_rows):
    print(f"\n--- {label}: {len(specs)} tasks × {n_workers} workers ---", flush=True)
    rows = list(prior_rows)
    ctx = mp.get_context("fork")
    t0 = time.perf_counter()
    with ctx.Pool(n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(worker, specs), 1):
            rows.append(r)
            print(f"  [{i}/{len(specs)}] {r['setup']:<24s} s{r['seed']}  "
                  f"neutral={r['pnl_neutral']:+9.2f}  trades={r['n_trades']:>5}  "
                  f"inv={r['final_inv']:+6.1f}  train={r['train_sec']:.0f}s", flush=True)
            with open(out_path, "w") as f:
                json.dump({"results": rows}, f, indent=2)
                f.flush(); os.fsync(f.fileno())
    print(f"  wall: {(time.perf_counter() - t0)/60:.1f}min", flush=True)
    return rows


def aggregate(rows, setup):
    vals = [r["pnl_neutral"] for r in rows if r["setup"] == setup]
    trades = [r["n_trades"] for r in rows if r["setup"] == setup]
    arr = np.array(vals); tr = np.array(trades)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    cv = abs(std / arr.mean()) if abs(arr.mean()) > 1e-9 else float("inf")
    return {"n": len(arr), "mean": float(arr.mean()), "std": std, "cv": cv,
            "trades_mean": float(tr.mean())}


def main():
    out_path = Path("results/iter/attention.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_workers = 2
    print(f"=== DQN-Attn на {DATE} intraday-сплите, {len(SEEDS)} сидов, "
          f"{n_workers} воркера ===", flush=True)

    import pickle
    global TRAIN_MD, TEST_MD
    t0 = time.perf_counter()
    print(f"  загружаю {TRAIN_PKL} + {TEST_PKL}...", flush=True)
    TRAIN_MD = pickle.load(open(TRAIN_PKL, "rb"))
    TEST_MD = pickle.load(open(TEST_PKL, "rb"))
    print(f"    train n={len(TRAIN_MD):,}  test n={len(TEST_MD):,}  за "
          f"{time.perf_counter() - t0:.1f}s", flush=True)

    p1_specs = [WorkerSpec(name="baseline_attn", use_oracle=False, steps=200000, seed=s)
                for s in SEEDS]
    rows = run_parallel(p1_specs, n_workers, "Фаза 1: baseline_attn", out_path, [])
    p1 = aggregate(rows, "baseline_attn")
    print(f"\n  baseline_attn: {p1['mean']:+.0f} ± {p1['std']:.0f}  cv={p1['cv']:.2f}  "
          f"trades_avg={p1['trades_mean']:.0f}", flush=True)

    p2_specs = [WorkerSpec(name="oracle_attn", use_oracle=True, steps=200000, seed=s)
                for s in SEEDS]
    rows = run_parallel(p2_specs, n_workers, "Фаза 2: oracle_attn", out_path, rows)
    p2 = aggregate(rows, "oracle_attn")
    print(f"\n  oracle_attn: {p2['mean']:+.0f} ± {p2['std']:.0f}  cv={p2['cv']:.2f}  "
          f"trades_avg={p2['trades_mean']:.0f}", flush=True)

    print("\n=== Сравнение attention vs linear ===\n", flush=True)
    no_attn = {"baseline_linear": {"mean": 23137, "std": 21633, "cv": 0.94},
               "oracle_linear":   {"mean": 30611, "std": 16033, "cv": 0.52}}
    linear_path = Path("results/iter/linear.json")
    if linear_path.exists():
        try:
            d = json.load(open(linear_path))
            s = d.get("summary", {})
            for k in ("baseline_linear", "oracle_linear"):
                if k in s:
                    no_attn[k] = {"mean": s[k]["mean"], "std": s[k]["std"], "cv": s[k]["cv"]}
        except Exception:
            pass

    print(f"  {'setup':<24s}  {'mean':>10s}  {'std':>8s}  {'cv':>5s}")
    print(f"  {'baseline_linear':<24s}  {no_attn['baseline_linear']['mean']:>+10.0f}  "
          f"{no_attn['baseline_linear']['std']:>+8.0f}  {no_attn['baseline_linear']['cv']:>5.2f}")
    print(f"  {'baseline_attn':<24s}  {p1['mean']:>+10.0f}  {p1['std']:>+8.0f}  {p1['cv']:>5.2f}")
    print(f"  {'oracle_linear':<24s}  {no_attn['oracle_linear']['mean']:>+10.0f}  "
          f"{no_attn['oracle_linear']['std']:>+8.0f}  {no_attn['oracle_linear']['cv']:>5.2f}")
    print(f"  {'oracle_attn':<24s}  {p2['mean']:>+10.0f}  {p2['std']:>+8.0f}  {p2['cv']:>5.2f}")

    summary = {
        "baseline_attn": p1, "oracle_attn": p2,
        "linear_reference": no_attn, "n_seeds": len(SEEDS),
    }
    with open(out_path, "w") as f:
        json.dump({"results": rows, "summary": summary}, f, indent=2)
        f.flush(); os.fsync(f.fileno())
    print(f"\nзаписано {out_path}")


if __name__ == "__main__":
    main()
