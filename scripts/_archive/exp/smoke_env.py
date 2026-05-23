"""Smoke test the RL environment end-to-end.

Loads a real checkpointed transformer + heads, builds RLMarketMakingEnv,
takes 5 random-action steps, and prints (obs shape, reward, info).

Run:
    PYTHONPATH=. uv run python -m scripts.exp.smoke_env \\
        --transformer-checkpoint /path/to/transformer.pt \\
        --heads-checkpoint /path/to/heads.pt \\
        [--action-mode a] [--state-mode heads]
"""

import argparse

import numpy as np

from src._archive.decision.rl.action import action_space_size, is_continuous
from src._archive.decision.rl.config import EnvConfig
from src._archive.decision.rl.env import RLMarketMakingEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--instrument", default="SBER")
    p.add_argument("--date", default="2024-03-22")
    p.add_argument("--data-dir", default="data/hftbacktest")
    p.add_argument("--sequences-dir", default="data/processed/sequences",
                   help="dir holding manifest.json + per-(inst,date) parquets")
    p.add_argument("--tokens-parquet", default=None,
                   help="override sequences/<INST>_<DATE>.parquet with an "
                        "explicit file (matches backtest_orig_as.py style)")
    p.add_argument("--transformer-checkpoint", required=True)
    p.add_argument("--heads-checkpoint", required=True)
    p.add_argument("--action-mode", default="a",
                   choices=["a", "b", "c", "a_disc", "b_disc", "c_disc"])
    p.add_argument("--state-mode", default="heads", choices=["heads", "hidden", "both"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    cfg = EnvConfig(
        instrument=args.instrument,
        date=args.date,
        data_dir=args.data_dir,
        sequences_dir=args.sequences_dir,
        tokens_parquet_override=args.tokens_parquet,
        transformer_checkpoint=args.transformer_checkpoint,
        heads_checkpoint=args.heads_checkpoint,
        action_mode=args.action_mode,
        state_mode=args.state_mode,
        device=args.device,
    )

    env = RLMarketMakingEnv(cfg)
    print(f"action_size={env.action_size}, continuous={env.is_continuous}, obs_dim={env.observation_dim}")

    obs = env.reset(seed=args.seed)
    print(f"reset OK; obs shape={obs.shape}, dtype={obs.dtype}")
    print(f"obs[:5]={obs[:5]}")

    for t in range(args.n_steps):
        if env.is_continuous:
            action = rng.uniform(-1.0, 1.0, size=env.action_size).astype(np.float32)
        else:
            action = int(rng.integers(0, env.action_size))

        obs, reward, done, info = env.step(action)
        print(
            f"[step {t}] action={action} | reward={reward:.4f} | "
            f"position={info.position:.1f} | balance={info.balance:.2f} | "
            f"mid={info.mid:.3f} | fills={info.fills} | done={done}"
        )
        if done:
            print("episode terminated")
            break

    env.close()
    print("smoke OK")


if __name__ == "__main__":
    main()
