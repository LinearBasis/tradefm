"""Generic RL training loop.

Dispatches to the correct agent class based on AgentConfig.algorithm.
Off-policy algorithms (SAC, DDPG, TD3, D3QN) all use the same loop —
rollout one full session, push transitions into replay, run N SGD updates
per env step once `learn_starts` is reached.

TensorBoard logging:
    - per-step (every log_every_steps within an episode): scalar losses/metrics
    - per-rollout: episode return, PnL, fills, position stats, eval if scheduled
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src._archive.decision.rl.action import action_space_size, is_continuous
from src._archive.decision.rl.buffer import ReplayBuffer
from src._archive.decision.rl.config import AgentConfig, EnvConfig, TrainConfig
from src._archive.decision.rl.env import RLMarketMakingEnv


def _build_agent(algorithm: str, obs_dim: int, action_dim: int, cfg: AgentConfig, device: torch.device):
    if algorithm == "sac":
        from src._archive.decision.rl.agents.sac import SAC
        return SAC(obs_dim, action_dim, cfg, device)
    if algorithm == "ddpg":
        from src._archive.decision.rl.agents.ddpg import DDPG
        return DDPG(obs_dim, action_dim, cfg, device)
    if algorithm == "td3":
        from src._archive.decision.rl.agents.td3 import TD3
        return TD3(obs_dim, action_dim, cfg, device)
    if algorithm == "d3qn":
        from src._archive.decision.rl.agents.d3qn import D3QN
        return D3QN(obs_dim, action_dim, cfg, device)
    raise ValueError(f"unknown algorithm: {algorithm}")


def _set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _autogen_run_name(env_cfg: EnvConfig, agent_cfg: AgentConfig, seed: int) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"{agent_cfg.algorithm}_{env_cfg.action_mode}_{env_cfg.state_mode}_seed{seed}_{ts}"


def train(env_cfg: EnvConfig, agent_cfg: AgentConfig, train_cfg: TrainConfig) -> str:
    """Run one training experiment. Returns the run directory path."""
    _set_seed(train_cfg.seed)

    # Validate mode/algo compatibility
    mode_continuous = is_continuous(env_cfg.action_mode)
    algo_continuous = agent_cfg.algorithm in ("sac", "ddpg", "td3")
    if mode_continuous != algo_continuous:
        raise ValueError(
            f"algorithm={agent_cfg.algorithm} (continuous={algo_continuous}) "
            f"incompatible with action_mode={env_cfg.action_mode} (continuous={mode_continuous})"
        )

    device = torch.device(env_cfg.device)
    env = RLMarketMakingEnv(env_cfg)

    obs_dim = env.observation_dim
    action_size = env.action_size
    action_dtype = np.float32 if mode_continuous else np.int64

    agent = _build_agent(agent_cfg.algorithm, obs_dim, action_size, agent_cfg, device)
    buffer = ReplayBuffer(
        capacity=agent_cfg.buffer_size,
        obs_dim=obs_dim,
        action_dim=action_size if mode_continuous else 1,
        action_dtype=action_dtype,
        seed=train_cfg.seed,
    )

    run_name = train_cfg.run_name or _autogen_run_name(env_cfg, agent_cfg, train_cfg.seed)
    run_dir = Path(train_cfg.log_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))

    # Persist configs for reproducibility
    with open(run_dir / "config.json", "w") as f:
        json.dump({
            "env": asdict(env_cfg),
            "agent": asdict(agent_cfg),
            "train": asdict(train_cfg),
        }, f, indent=2)

    global_step = 0
    rollout_returns: list[float] = []

    print(f"[train] obs_dim={obs_dim}, action_size={action_size}, continuous={mode_continuous}")
    print(f"[train] run_dir={run_dir}")

    for rollout in range(train_cfg.n_rollouts):
        obs = env.reset(seed=train_cfg.seed + rollout)
        done = False
        ep_return = 0.0
        ep_fills = 0
        ep_steps = 0
        ep_start = time.time()

        positions: list[float] = []
        rewards_in_ep: list[float] = []

        while not done:
            # --- Pick action ---
            if global_step < agent_cfg.learn_starts:
                # Pure exploration: uniform random within action space
                if mode_continuous:
                    action = np.random.uniform(-1.0, 1.0, size=action_size).astype(np.float32)
                else:
                    action = int(np.random.randint(0, action_size))
            else:
                action = agent.select_action(obs, deterministic=False)

            next_obs, reward, done, info = env.step(action)
            buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs

            ep_return += reward
            ep_fills += info.fills
            positions.append(info.position)
            rewards_in_ep.append(reward)
            ep_steps += 1
            global_step += 1

            # --- SGD updates ---
            if len(buffer) >= max(agent_cfg.batch_size, agent_cfg.learn_starts):
                for _ in range(train_cfg.updates_per_step):
                    batch = buffer.sample(agent_cfg.batch_size)
                    metrics = agent.update(batch)
                if global_step % train_cfg.log_every_steps == 0:
                    for k, v in metrics.items():
                        writer.add_scalar(k, v, global_step)

        # --- End-of-rollout logging ---
        rollout_returns.append(ep_return)
        writer.add_scalar("rollout/return", ep_return, rollout)
        writer.add_scalar("rollout/final_balance", info.balance, rollout)
        writer.add_scalar("rollout/final_position", info.position, rollout)
        writer.add_scalar("rollout/fills", ep_fills, rollout)
        writer.add_scalar("rollout/steps", ep_steps, rollout)
        writer.add_scalar("rollout/mean_abs_position", float(np.mean(np.abs(positions))), rollout)
        writer.add_scalar("rollout/wall_time_sec", time.time() - ep_start, rollout)

        print(
            f"[rollout {rollout:3d}/{train_cfg.n_rollouts}] "
            f"return={ep_return:9.2f} | final_balance={info.balance:10.2f} | "
            f"final_pos={info.position:5.1f} | fills={ep_fills:4d} | "
            f"steps={ep_steps} | wall={time.time()-ep_start:.1f}s"
        )

        if (rollout + 1) % train_cfg.save_every_rollouts == 0:
            ckpt = run_dir / f"agent_rollout{rollout+1:04d}.pt"
            torch.save({
                "agent": agent.state_dict(),
                "rollout": rollout + 1,
                "global_step": global_step,
            }, ckpt)

    # Final checkpoint
    torch.save({"agent": agent.state_dict(), "rollout": train_cfg.n_rollouts,
                "global_step": global_step}, run_dir / "agent_final.pt")
    writer.close()
    env.close()

    print(f"[train] done. mean return over last 5 rollouts: {np.mean(rollout_returns[-5:]):.2f}")
    return str(run_dir)
