"""Launcher: train one RL agent.

Reads three JSON config files (env, agent, train) and dispatches to `train()`.
Any field can be overridden via --env.<field>, --agent.<field>, --train.<field>.

Run:
    PYTHONPATH=. uv run python -m scripts.exp.train_rl \\
        --env configs/exp/rl_smoke.json \\
        --agent configs/exp/rl_agent_sac.json \\
        --train configs/exp/rl_train.json
"""

import argparse
import json
from dataclasses import fields
from pathlib import Path

from src.config import load_from_json
from src._archive.decision.rl.config import AgentConfig, EnvConfig, TrainConfig
from src._archive.decision.rl.train import train


def _apply_overrides(cfg, overrides: dict, name: str):
    """Apply `field=value` overrides to a dataclass instance. Casts to field type."""
    field_types = {f.name: f.type for f in fields(cfg)}
    for k, v in overrides.items():
        if k not in field_types:
            raise ValueError(f"unknown {name}.{k}")
        # Best-effort cast (JSON is already typed, but CLI strings need coercion)
        cur = getattr(cfg, k)
        if isinstance(cur, bool):
            v = str(v).lower() in ("1", "true", "yes")
        elif isinstance(cur, int) and not isinstance(cur, bool):
            v = int(v)
        elif isinstance(cur, float):
            v = float(v)
        setattr(cfg, k, v)


def _collect_overrides(args, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in vars(args).items()
            if k.startswith(prefix) and v is not None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, help="path to env JSON config")
    p.add_argument("--agent", required=True, help="path to agent JSON config")
    p.add_argument("--train", required=True, help="path to train JSON config")
    # A few common overrides as named args
    p.add_argument("--seed", type=int, default=None, help="override train.seed")
    p.add_argument("--n-rollouts", type=int, default=None, help="override train.n_rollouts")
    p.add_argument("--run-name", default=None, help="override train.run_name")
    p.add_argument("--device", default=None, help="override env.device")
    p.add_argument("--action-mode", default=None,
                   choices=["a", "b", "c", "a_disc", "b_disc", "c_disc"],
                   help="override env.action_mode")
    p.add_argument("--state-mode", default=None, choices=["heads", "hidden", "both"],
                   help="override env.state_mode")
    p.add_argument("--max-steps", type=int, default=None,
                   help="override env.max_steps_per_rollout (for fast smoke)")
    p.add_argument("--learn-starts", type=int, default=None,
                   help="override agent.learn_starts (warmup before SGD)")
    p.add_argument("--sequences-dir", default=None,
                   help="override env.sequences_dir")
    p.add_argument("--tokens-parquet", default=None,
                   help="override env.tokens_parquet_override "
                        "(matches backtest_orig_as.py style)")
    p.add_argument("--transformer-checkpoint", default=None,
                   help="override env.transformer_checkpoint")
    p.add_argument("--heads-checkpoint", default=None,
                   help="override env.heads_checkpoint")
    p.add_argument("--instrument", default=None, help="override env.instrument")
    p.add_argument("--date", default=None, help="override env.date")
    p.add_argument("--log-dir", default=None, help="override train.log_dir")
    args = p.parse_args()

    env_cfg = load_from_json(EnvConfig, args.env)
    agent_cfg = load_from_json(AgentConfig, args.agent)
    train_cfg = load_from_json(TrainConfig, args.train)

    if args.seed is not None: train_cfg.seed = args.seed
    if args.n_rollouts is not None: train_cfg.n_rollouts = args.n_rollouts
    if args.run_name is not None: train_cfg.run_name = args.run_name
    if args.device is not None: env_cfg.device = args.device
    if args.action_mode is not None: env_cfg.action_mode = args.action_mode
    if args.state_mode is not None: env_cfg.state_mode = args.state_mode
    if args.max_steps is not None: env_cfg.max_steps_per_rollout = args.max_steps
    if args.learn_starts is not None: agent_cfg.learn_starts = args.learn_starts
    if args.sequences_dir is not None: env_cfg.sequences_dir = args.sequences_dir
    if args.tokens_parquet is not None: env_cfg.tokens_parquet_override = args.tokens_parquet
    if args.transformer_checkpoint is not None:
        env_cfg.transformer_checkpoint = args.transformer_checkpoint
    if args.heads_checkpoint is not None:
        env_cfg.heads_checkpoint = args.heads_checkpoint
    if args.instrument is not None: env_cfg.instrument = args.instrument
    if args.date is not None: env_cfg.date = args.date
    if args.log_dir is not None: train_cfg.log_dir = args.log_dir

    run_dir = train(env_cfg, agent_cfg, train_cfg)
    print(f"DONE: {run_dir}")


if __name__ == "__main__":
    main()
