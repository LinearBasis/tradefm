"""Dueling Double Deep Q-Network.

Combines:
    - Dueling architecture: Q(s,a) = V(s) + (A(s,a) − mean A(s,·))
    - Double-DQN target: online net picks action, target net evaluates
    - ε-greedy exploration with linear decay

For discrete-action modes only (a_disc, b_disc, c_disc).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src._archive.decision.rl.agents.networks import DuelingQNet, hard_update
from src._archive.decision.rl.config import AgentConfig


class D3QN:
    def __init__(self, obs_dim: int, n_actions: int, cfg: AgentConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.n_actions = n_actions
        self._update_count = 0
        self._select_count = 0

        self.q = DuelingQNet(obs_dim, n_actions, cfg.hidden_dim, cfg.n_hidden_layers).to(device)
        self.q_target = DuelingQNet(obs_dim, n_actions, cfg.hidden_dim, cfg.n_hidden_layers).to(device)
        hard_update(self.q_target, self.q)
        for p in self.q_target.parameters(): p.requires_grad = False

        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=cfg.d3qn_lr)

    def _current_epsilon(self) -> float:
        frac = min(self._select_count / max(self.cfg.epsilon_decay_steps, 1), 1.0)
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> int:
        self._select_count += 1
        eps = 0.0 if deterministic else self._current_epsilon()
        if np.random.rand() < eps:
            return int(np.random.randint(0, self.n_actions))
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_vals = self.q(obs_t).squeeze(0)
        return int(torch.argmax(q_vals).item())

    def update(self, batch: dict) -> dict:
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)

        # --- Double-DQN target ---
        with torch.no_grad():
            next_actions = torch.argmax(self.q(next_obs), dim=-1)
            next_q_target = self.q_target(next_obs).gather(-1, next_actions.unsqueeze(-1)).squeeze(-1)
            y = rewards + self.cfg.discount * (1.0 - dones) * next_q_target

        q_pred = self.q(obs).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        loss = F.smooth_l1_loss(q_pred, y)
        self.q_opt.zero_grad(set_to_none=True); loss.backward(); self.q_opt.step()

        self._update_count += 1
        if self._update_count % self.cfg.d3qn_target_update_freq == 0:
            hard_update(self.q_target, self.q)

        return {
            "loss/q": float(loss.item()),
            "stats/q_mean": float(q_pred.mean().item()),
            "stats/epsilon": float(self._current_epsilon()),
        }

    def state_dict(self) -> dict:
        return {
            "q": self.q.state_dict(),
            "q_target": self.q_target.state_dict(),
            "update_count": self._update_count,
            "select_count": self._select_count,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.q.load_state_dict(sd["q"]); self.q_target.load_state_dict(sd["q_target"])
        self._update_count = int(sd.get("update_count", 0))
        self._select_count = int(sd.get("select_count", 0))
