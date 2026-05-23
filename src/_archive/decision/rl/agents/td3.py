"""Twin-Delayed DDPG (Fujimoto et al. 2018).

DDPG + three fixes:
    (a) Twin critics with min — anti-overestimation (like SAC's twins)
    (b) Delayed actor update — actor every `td3_policy_delay` critic updates
    (c) Target action smoothing — Gaussian noise on target action before Q-target
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


class TD3:
    def __init__(self, obs_dim: int, action_dim: int, cfg: AgentConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.action_dim = action_dim
        self._update_count = 0

        self.actor = DeterministicActor(obs_dim, action_dim, cfg.hidden_dim, cfg.n_hidden_layers).to(device)
        self.actor_target = DeterministicActor(obs_dim, action_dim, cfg.hidden_dim, cfg.n_hidden_layers).to(device)
        hard_update(self.actor_target, self.actor)
        for p in self.actor_target.parameters(): p.requires_grad = False

        self.q1 = QNet(obs_dim, action_dim, cfg.hidden_dim, cfg.n_hidden_layers).to(device)
        self.q2 = QNet(obs_dim, action_dim, cfg.hidden_dim, cfg.n_hidden_layers).to(device)
        self.q1_target = QNet(obs_dim, action_dim, cfg.hidden_dim, cfg.n_hidden_layers).to(device)
        self.q2_target = QNet(obs_dim, action_dim, cfg.hidden_dim, cfg.n_hidden_layers).to(device)
        hard_update(self.q1_target, self.q1); hard_update(self.q2_target, self.q2)
        for p in self.q1_target.parameters(): p.requires_grad = False
        for p in self.q2_target.parameters(): p.requires_grad = False

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=cfg.critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=cfg.critic_lr)

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

        # --- Twin-critic update with smoothed target action ---
        with torch.no_grad():
            next_a = self.actor_target(next_obs)
            noise = torch.randn_like(next_a) * self.cfg.td3_target_noise
            noise = noise.clamp(-self.cfg.td3_target_noise_clip, self.cfg.td3_target_noise_clip)
            next_a = (next_a + noise).clamp(-1.0, 1.0)
            tq1 = self.q1_target(next_obs, next_a)
            tq2 = self.q2_target(next_obs, next_a)
            target_q = torch.min(tq1, tq2)
            y = rewards + self.cfg.discount * (1.0 - dones) * target_q

        q1_pred = self.q1(obs, actions); q2_pred = self.q2(obs, actions)
        q1_loss = F.mse_loss(q1_pred, y); q2_loss = F.mse_loss(q2_pred, y)
        self.q1_opt.zero_grad(set_to_none=True); q1_loss.backward(); self.q1_opt.step()
        self.q2_opt.zero_grad(set_to_none=True); q2_loss.backward(); self.q2_opt.step()

        metrics = {
            "loss/q1": float(q1_loss.item()),
            "loss/q2": float(q2_loss.item()),
            "stats/q_mean": float(q1_pred.mean().item()),
        }

        # --- Delayed actor + target updates ---
        self._update_count += 1
        if self._update_count % self.cfg.td3_policy_delay == 0:
            actor_loss = -self.q1(obs, self.actor(obs)).mean()
            self.actor_opt.zero_grad(set_to_none=True); actor_loss.backward(); self.actor_opt.step()
            soft_update(self.actor_target, self.actor, self.cfg.target_update_tau)
            soft_update(self.q1_target, self.q1, self.cfg.target_update_tau)
            soft_update(self.q2_target, self.q2, self.cfg.target_update_tau)
            metrics["loss/actor"] = float(actor_loss.item())

        return metrics

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(), "q2_target": self.q2_target.state_dict(),
            "update_count": self._update_count,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.actor.load_state_dict(sd["actor"]); self.actor_target.load_state_dict(sd["actor_target"])
        self.q1.load_state_dict(sd["q1"]); self.q2.load_state_dict(sd["q2"])
        self.q1_target.load_state_dict(sd["q1_target"]); self.q2_target.load_state_dict(sd["q2_target"])
        self._update_count = int(sd.get("update_count", 0))
