"""Deep Deterministic Policy Gradient (Lillicrap et al. 2015).

Vanilla DDPG: deterministic actor + single critic + Gaussian exploration noise.
Known issues: Q-overestimation, hyperparam sensitivity (cf. Smirnov dropping it).
TD3 fixes those — if DDPG collapses in smoke runs, switch algorithm: "td3".
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src._archive.decision.rl.agents.networks import (
    DeterministicActor,
    QNet,
    hard_update,
    soft_update,
)
from src._archive.decision.rl.config import AgentConfig


class DDPG:
    def __init__(self, obs_dim: int, action_dim: int, cfg: AgentConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.action_dim = action_dim

        self.actor = DeterministicActor(obs_dim, action_dim, cfg.hidden_dim, cfg.n_hidden_layers).to(device)
        self.actor_target = DeterministicActor(obs_dim, action_dim, cfg.hidden_dim, cfg.n_hidden_layers).to(device)
        hard_update(self.actor_target, self.actor)
        for p in self.actor_target.parameters(): p.requires_grad = False

        self.q = QNet(obs_dim, action_dim, cfg.hidden_dim, cfg.n_hidden_layers).to(device)
        self.q_target = QNet(obs_dim, action_dim, cfg.hidden_dim, cfg.n_hidden_layers).to(device)
        hard_update(self.q_target, self.q)
        for p in self.q_target.parameters(): p.requires_grad = False

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=cfg.critic_lr)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.actor(obs_t).squeeze(0).cpu().numpy()
        if not deterministic:
            a = a + np.random.normal(0.0, self.cfg.action_noise_std, size=self.action_dim).astype(np.float32)
            a = np.clip(a, -1.0, 1.0)
        return a.astype(np.float32)

    def update(self, batch: dict) -> dict:
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)

        # --- Critic update ---
        with torch.no_grad():
            next_a = self.actor_target(next_obs)
            target_q = self.q_target(next_obs, next_a)
            y = rewards + self.cfg.discount * (1.0 - dones) * target_q
        q_pred = self.q(obs, actions)
        q_loss = F.mse_loss(q_pred, y)
        self.q_opt.zero_grad(set_to_none=True); q_loss.backward(); self.q_opt.step()

        # --- Actor update (deterministic policy gradient) ---
        actor_loss = -self.q(obs, self.actor(obs)).mean()
        self.actor_opt.zero_grad(set_to_none=True); actor_loss.backward(); self.actor_opt.step()

        # --- Target soft updates ---
        soft_update(self.actor_target, self.actor, self.cfg.target_update_tau)
        soft_update(self.q_target, self.q, self.cfg.target_update_tau)

        return {
            "loss/q": float(q_loss.item()),
            "loss/actor": float(actor_loss.item()),
            "stats/q_mean": float(q_pred.mean().item()),
        }

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "q": self.q.state_dict(),
            "q_target": self.q_target.state_dict(),
        }

    def load_state_dict(self, sd: dict) -> None:
        self.actor.load_state_dict(sd["actor"]); self.actor_target.load_state_dict(sd["actor_target"])
        self.q.load_state_dict(sd["q"]); self.q_target.load_state_dict(sd["q_target"])
