"""Shared NN building blocks for RL agents."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(in_dim: int, out_dim: int, hidden_dim: int, n_hidden_layers: int) -> nn.Sequential:
    """ReLU MLP. Init: PyTorch defaults (Kaiming for Linear); final-layer
    weights downscaled ×0.01 to keep initial outputs near zero (helps both
    SAC and DDPG stability)."""
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(n_hidden_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    final = nn.Linear(hidden_dim, out_dim)
    with torch.no_grad():
        final.weight.mul_(0.01)
        final.bias.zero_()
    layers.append(final)
    return nn.Sequential(*layers)


class GaussianTanhActor(nn.Module):
    """SAC-style stochastic actor: outputs μ(s), log σ(s), samples tanh-squashed action.

    Action range is assumed [-1, 1] (post-tanh). The agent caller scales/maps to
    real action space via `compute_quotes()` etc.

    Returns:
        action — (B, action_dim), in [-1, 1]
        log_prob — (B,) log π(a|s), corrected for tanh Jacobian
    """

    LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, n_hidden_layers: int):
        super().__init__()
        self.trunk = mlp(obs_dim, 2 * action_dim, hidden_dim, n_hidden_layers)
        self.action_dim = action_dim

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.trunk(obs)
        mu, log_std = out.chunk(2, dim=-1)
        log_std = log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        u = normal.rsample()  # reparameterized
        a = torch.tanh(u)
        # log π(a|s) = log N(u; μ, σ) − sum log(1 − a²)
        # Using the numerically stable form (Spinning Up tricks):
        log_prob = normal.log_prob(u).sum(-1) - (2 * (math.log(2) - u - F.softplus(-2 * u))).sum(-1)
        return a, log_prob

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        out = self.trunk(obs)
        mu, log_std = out.chunk(2, dim=-1)
        if deterministic:
            return torch.tanh(mu)
        std = log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX).exp()
        u = mu + std * torch.randn_like(std)
        return torch.tanh(u)


class DeterministicActor(nn.Module):
    """DDPG/TD3-style deterministic actor with tanh output (action ∈ [-1, 1])."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, n_hidden_layers: int):
        super().__init__()
        self.trunk = mlp(obs_dim, action_dim, hidden_dim, n_hidden_layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.trunk(obs))


class QNet(nn.Module):
    """Continuous-action Q-network: Q(s, a)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, n_hidden_layers: int):
        super().__init__()
        self.trunk = mlp(obs_dim + action_dim, 1, hidden_dim, n_hidden_layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.trunk(x).squeeze(-1)


class DuelingQNet(nn.Module):
    """Discrete-action dueling Q-network: Q(s, ·) = V(s) + (A(s, ·) - mean A(s, ·))."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int, n_hidden_layers: int):
        super().__init__()
        # Shared trunk
        layers: list[nn.Module] = [nn.Linear(obs_dim, hidden_dim), nn.ReLU()]
        for _ in range(max(n_hidden_layers - 1, 0)):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.shared = nn.Sequential(*layers)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.adv_head = nn.Linear(hidden_dim, n_actions)
        with torch.no_grad():
            self.value_head.weight.mul_(0.01); self.value_head.bias.zero_()
            self.adv_head.weight.mul_(0.01); self.adv_head.bias.zero_()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.shared(obs)
        v = self.value_head(h)             # (B, 1)
        a = self.adv_head(h)               # (B, n_actions)
        return v + (a - a.mean(dim=-1, keepdim=True))


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.mul_(1.0 - tau).add_(sp, alpha=tau)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())
