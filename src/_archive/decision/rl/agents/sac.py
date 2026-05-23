"""Soft Actor-Critic (Haarnoja et al. 2018).

Canonical setup:
    - Stochastic Gaussian-tanh actor
    - Twin Q-critics with target networks (anti-overestimation)
    - Auto-tuned entropy temperature α with target_entropy = -action_dim
    - Soft target updates with τ
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src._archive.decision.rl.agents.networks import (
    GaussianTanhActor,
    QNet,
    hard_update,
    soft_update,
)
from src._archive.decision.rl.config import AgentConfig


class SAC:
    def __init__(self, obs_dim: int, action_dim: int, cfg: AgentConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.action_dim = action_dim

        self.actor = GaussianTanhActor(obs_dim, action_dim, cfg.hidden_dim, cfg.n_hidden_layers).to(device)
        self.q1 = QNet(obs_dim, action_dim, cfg.hidden_dim, cfg.n_hidden_layers).to(device)
        self.q2 = QNet(obs_dim, action_dim, cfg.hidden_dim, cfg.n_hidden_layers).to(device)
        self.q1_target = QNet(obs_dim, action_dim, cfg.hidden_dim, cfg.n_hidden_layers).to(device)
        self.q2_target = QNet(obs_dim, action_dim, cfg.hidden_dim, cfg.n_hidden_layers).to(device)
        hard_update(self.q1_target, self.q1)
        hard_update(self.q2_target, self.q2)
        for p in self.q1_target.parameters(): p.requires_grad = False
        for p in self.q2_target.parameters(): p.requires_grad = False

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=cfg.critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=cfg.critic_lr)

        # Entropy temperature
        self.target_entropy = cfg.sac_target_entropy if cfg.sac_target_entropy is not None else -float(action_dim)
        self.log_alpha = torch.tensor(cfg.sac_init_log_alpha, device=device, requires_grad=cfg.sac_auto_alpha)
        if cfg.sac_auto_alpha:
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.actor_lr)
        else:
            self.alpha_opt = None

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.actor.act(obs_t, deterministic=deterministic).squeeze(0).cpu().numpy()
        return a.astype(np.float32)

    def update(self, batch: dict) -> dict:
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)

        # --- Critic targets ---
        with torch.no_grad():
            next_a, next_logp = self.actor(next_obs)
            tq1 = self.q1_target(next_obs, next_a)
            tq2 = self.q2_target(next_obs, next_a)
            target_q = torch.min(tq1, tq2) - self.alpha.detach() * next_logp
            y = rewards + self.cfg.discount * (1.0 - dones) * target_q

        # --- Critic updates ---
        q1_pred = self.q1(obs, actions)
        q2_pred = self.q2(obs, actions)
        q1_loss = F.mse_loss(q1_pred, y)
        q2_loss = F.mse_loss(q2_pred, y)
        self.q1_opt.zero_grad(set_to_none=True); q1_loss.backward(); self.q1_opt.step()
        self.q2_opt.zero_grad(set_to_none=True); q2_loss.backward(); self.q2_opt.step()

        # --- Actor update ---
        new_a, logp = self.actor(obs)
        q1_new = self.q1(obs, new_a)
        q2_new = self.q2(obs, new_a)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha.detach() * logp - q_new).mean()
        self.actor_opt.zero_grad(set_to_none=True); actor_loss.backward(); self.actor_opt.step()

        # --- Alpha update ---
        if self.alpha_opt is not None:
            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad(set_to_none=True); alpha_loss.backward(); self.alpha_opt.step()
        else:
            alpha_loss = torch.tensor(0.0)

        # --- Target soft update ---
        soft_update(self.q1_target, self.q1, self.cfg.target_update_tau)
        soft_update(self.q2_target, self.q2, self.cfg.target_update_tau)

        return {
            "loss/q1": float(q1_loss.item()),
            "loss/q2": float(q2_loss.item()),
            "loss/actor": float(actor_loss.item()),
            "loss/alpha": float(alpha_loss.item()),
            "stats/alpha": float(self.alpha.item()),
            "stats/entropy": float(-logp.mean().item()),
            "stats/q_mean": float(q_new.mean().item()),
        }

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }

    def load_state_dict(self, sd: dict) -> None:
        self.actor.load_state_dict(sd["actor"])
        self.q1.load_state_dict(sd["q1"]); self.q2.load_state_dict(sd["q2"])
        self.q1_target.load_state_dict(sd["q1_target"]); self.q2_target.load_state_dict(sd["q2_target"])
        self.log_alpha.data.copy_(sd["log_alpha"].to(self.device))
