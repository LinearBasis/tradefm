"""DQN-агент."""
from __future__ import annotations

import random
from collections import deque, namedtuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.mm_env import N_ACTIONS

Transition = namedtuple("Transition", ["s", "a", "r", "s2", "done"])


@dataclass
class DQNConfig:
    hidden: tuple = (128, 64)
    lr: float = 1e-3
    gamma: float = 0.99
    eps_start: float = 0.9
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000
    buffer_size: int = 50_000
    batch_size: int = 64
    target_update_every: int = 500
    warmup_steps: int = 1000
    device: str = "cpu"
    seed: int = 0
    qnet_kind: str = "linear"
    attn_feature_depth: int = 10
    attn_d_model: int = 16
    attn_n_heads: int = 4
    attn_n_layers: int = 1
    tradefm_emb_dim: int = 0
    tradefm_proj_dim: int = 0


class QNet(nn.Module):
    def __init__(self, in_dim: int, n_actions: int, hidden,
                 tradefm_emb_dim: int = 0, tradefm_proj_dim: int = 0):
        super().__init__()
        self.tradefm_emb_dim = tradefm_emb_dim
        self.tradefm_proj_dim = tradefm_proj_dim
        if tradefm_emb_dim > 0 and tradefm_proj_dim > 0:
            self.tradefm_proj = nn.Linear(tradefm_emb_dim, tradefm_proj_dim)
            mlp_in = (in_dim - tradefm_emb_dim) + tradefm_proj_dim
        else:
            self.tradefm_proj = None
            mlp_in = in_dim
        layers = []
        prev = mlp_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if self.tradefm_proj is not None:
            d = self.tradefm_emb_dim
            base = x[..., :-d]
            emb = x[..., -d:]
            proj = torch.relu(self.tradefm_proj(emb))
            x = torch.cat([base, proj], dim=-1)
        return self.net(x)


class DQNAgent:
    def __init__(self, state_dim: int, cfg: DQNConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.n_actions = N_ACTIONS
        if cfg.qnet_kind == "attention":
            from rl.dqn_attn import AttentionQNet, AttnQNetConfig
            acfg = AttnQNetConfig(
                feature_depth=cfg.attn_feature_depth,
                d_model=cfg.attn_d_model,
                n_heads=cfg.attn_n_heads,
                n_attn_layers=cfg.attn_n_layers,
                mlp_hidden=cfg.hidden,
                tradefm_emb_dim=cfg.tradefm_emb_dim,
                tradefm_proj_dim=cfg.tradefm_proj_dim,
            )
            self.online = AttentionQNet(state_dim, self.n_actions, acfg).to(cfg.device)
            self.target = AttentionQNet(state_dim, self.n_actions, acfg).to(cfg.device)
        else:
            self.online = QNet(state_dim, self.n_actions, cfg.hidden,
                               tradefm_emb_dim=cfg.tradefm_emb_dim,
                               tradefm_proj_dim=cfg.tradefm_proj_dim).to(cfg.device)
            self.target = QNet(state_dim, self.n_actions, cfg.hidden,
                               tradefm_emb_dim=cfg.tradefm_emb_dim,
                               tradefm_proj_dim=cfg.tradefm_proj_dim).to(cfg.device)
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.buffer: deque = deque(maxlen=cfg.buffer_size)
        self.step_count = 0

    def epsilon(self) -> float:
        t = min(self.step_count / self.cfg.eps_decay_steps, 1.0)
        return self.cfg.eps_start + (self.cfg.eps_end - self.cfg.eps_start) * t

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy and random.random() < self.epsilon():
            return random.randrange(self.n_actions)
        with torch.no_grad():
            x = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.cfg.device)
            q = self.online(x)
            return int(q.argmax(dim=1).item())

    def push(self, s, a, r, s2, done):
        self.buffer.append(Transition(s.astype(np.float32), int(a), float(r),
                                       s2.astype(np.float32), bool(done)))

    def train_step(self):
        if len(self.buffer) < max(self.cfg.batch_size, self.cfg.warmup_steps):
            return None
        batch = random.sample(self.buffer, self.cfg.batch_size)
        s = torch.from_numpy(np.stack([t.s for t in batch])).to(self.cfg.device)
        a = torch.tensor([t.a for t in batch], dtype=torch.long, device=self.cfg.device)
        r = torch.tensor([t.r for t in batch], dtype=torch.float32, device=self.cfg.device)
        s2 = torch.from_numpy(np.stack([t.s2 for t in batch])).to(self.cfg.device)
        done = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.cfg.device)
        q_sa = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target(s2).max(dim=1).values
            target = r + self.cfg.gamma * q_next * (1 - done)
        loss = F.smooth_l1_loss(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()
        if self.step_count % self.cfg.target_update_every == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())


def train_dqn(env, agent: DQNAgent, max_steps: int) -> list[float]:
    from tqdm import tqdm
    s = env.reset()
    losses: list[float] = []
    pbar = tqdm(range(max_steps), desc="DQN", unit="step", mininterval=0.3)
    for i in pbar:
        a = agent.act(s)
        s2, r, done, _ = env.step(a)
        agent.push(s, a, r, s2, done)
        loss = agent.train_step()
        if loss is not None:
            losses.append(loss)
        agent.step_count += 1
        s = s2
        if done:
            s = env.reset()
        if (i + 1) % 50 == 0 and losses:
            pbar.set_postfix(eps=f"{agent.epsilon():.2f}",
                             loss=f"{float(np.mean(losses[-50:])):.3f}")
    return losses


def evaluate_dqn(env, agent: DQNAgent) -> dict:
    from tqdm import tqdm
    s = env.reset()
    done = False
    pbar = tqdm(desc="DQN-eval", unit="step", mininterval=0.3)
    while not done:
        a = agent.act(s, greedy=True)
        s, _, done, _ = env.step(a)
        pbar.update(1)
    pbar.close()
    return env.snapshot_stats()
