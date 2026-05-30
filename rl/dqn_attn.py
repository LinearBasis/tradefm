"""QNet с self-attention над уровнями стакана."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class AttnQNetConfig:
    feature_depth: int = 10
    d_model: int = 16
    n_heads: int = 4
    n_attn_layers: int = 1
    mlp_hidden: tuple = (128, 64)
    tradefm_emb_dim: int = 0
    tradefm_proj_dim: int = 0


class AttentionQNet(nn.Module):
    def __init__(self, in_dim: int, n_actions: int,
                 attn_cfg: AttnQNetConfig | None = None):
        super().__init__()
        self.cfg = attn_cfg or AttnQNetConfig()
        self.D = self.cfg.feature_depth
        self.book_dim = 4 * self.D
        self.scalar_dim = in_dim - self.book_dim
        assert self.scalar_dim >= 0, f"in_dim={in_dim} < book_dim={self.book_dim}"

        self.book_proj = nn.Linear(4, self.cfg.d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.d_model,
            nhead=self.cfg.n_heads,
            dim_feedforward=2 * self.cfg.d_model,
            dropout=0.0,
            batch_first=True,
            activation="relu",
        )
        self.attn = nn.TransformerEncoder(enc_layer, num_layers=self.cfg.n_attn_layers)

        self.tradefm_emb_dim = self.cfg.tradefm_emb_dim
        self.tradefm_proj_dim = self.cfg.tradefm_proj_dim
        if self.tradefm_emb_dim > 0 and self.tradefm_proj_dim > 0:
            assert self.scalar_dim >= self.tradefm_emb_dim
            self.tradefm_proj = nn.Linear(self.tradefm_emb_dim, self.tradefm_proj_dim)
            effective_scalar_dim = self.scalar_dim - self.tradefm_emb_dim + self.tradefm_proj_dim
        else:
            self.tradefm_proj = None
            effective_scalar_dim = self.scalar_dim

        mlp_in = self.cfg.d_model + effective_scalar_dim
        layers = []
        prev = mlp_in
        for h in self.cfg.mlp_hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        book_flat = x[:, : self.book_dim]
        scalar = x[:, self.book_dim:]
        book = book_flat.view(-1, 4, self.D).transpose(1, 2).contiguous()
        tokens = self.book_proj(book)
        out = self.attn(tokens)
        pooled = out.mean(dim=1)
        if self.tradefm_proj is not None:
            d = self.tradefm_emb_dim
            scalar_base = scalar[..., :-d]
            emb = scalar[..., -d:]
            proj = torch.relu(self.tradefm_proj(emb))
            scalar = torch.cat([scalar_base, proj], dim=-1)
        merged = torch.cat([pooled, scalar], dim=-1)
        return self.mlp(merged)
