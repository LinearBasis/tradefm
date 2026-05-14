"""Decoder-only Transformer for autoregressive order flow modeling."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelConfig


class RMSNorm(nn.Module):
    """Standard RMSNorm. Used for optional QK-norm in attention (Gemma 2 / DeepSeek-V4 style)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embedding to q or k.

    x:   (B, n_heads, T, d_head)        d_head must be even
    cos, sin: (T, d_head // 2)
    """
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    """Grouped-Query Causal Self-Attention with RoPE.

    n_kv_heads < n_heads → K/V are smaller; groups of (n_heads // n_kv_heads)
    Q-heads share each K/V head. n_kv_heads == n_heads recovers standard MHA.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        assert cfg.n_heads % cfg.n_kv_heads == 0, "n_kv_heads must divide n_heads"
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.d_head = cfg.d_model // cfg.n_heads
        assert self.d_head % 2 == 0, "d_head must be even for RoPE"

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.d_head, bias=False)
        self.kv_proj = nn.Linear(cfg.d_model, 2 * cfg.n_kv_heads * self.d_head, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.attn_drop = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)
        # Optional QK-norm. When `qk_norm=False` (default), these are no-ops and
        # contribute no parameters to state_dict — old checkpoints stay compatible.
        if getattr(cfg, "qk_norm", False):
            self.q_norm: nn.Module = RMSNorm(self.d_head)
            self.k_norm: nn.Module = RMSNorm(self.d_head)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        kv = self.kv_proj(x).reshape(B, T, 2, self.n_kv_heads, self.d_head)
        k, v = kv.unbind(dim=2)
        k = k.transpose(1, 2)  # (B, n_kv_heads, T, d_head)
        v = v.transpose(1, 2)

        # QK-norm (no-op when cfg.qk_norm is False — nn.Identity).
        q = self.q_norm(q)
        k = self.k_norm(k)

        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.attn_drop if self.training else 0.0,
        )  # (B, n_heads, T, d_head)
        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.d_head)
        return self.resid_drop(self.out_proj(out))


class SwiGLU(nn.Module):
    """SwiGLU FFN: down(SiLU(gate(x)) * up(x)). Three matrices, no bias (Llama-style)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class PreNormBlock(nn.Module):
    """Pre-LayerNorm Transformer decoder block (GQA + SwiGLU + RoPE)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.ffn(self.ln2(x))
        return x


class OrderFlowTransformer(nn.Module):
    """GPT-style decoder-only transformer for order flow next-token prediction.

    Input per timestep:
        - trade_token: int [0, vocab_size)  — composite token
        - price_level: int [0, n_price_level_bins) — context
        - liquidity:   int [0, n_liquidity_bins)   — context
        - instrument_id: int [0, n_instruments)     — context

    Output: logits over vocab_size for next token prediction.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Token embedding
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # Context embeddings (additive fusion)
        self.price_level_emb = nn.Embedding(cfg.n_price_level_bins, cfg.d_context)
        self.liquidity_emb = nn.Embedding(cfg.n_liquidity_bins, cfg.d_context)
        self.instrument_emb = nn.Embedding(cfg.n_instruments, cfg.d_context)
        self.context_proj = nn.Linear(3 * cfg.d_context, cfg.d_model)

        # Rotary positional embedding: precompute inverse frequencies (no parameters).
        # cos/sin are derived per-forward from current sequence length, so the model
        # is not capped at any predefined context length at inference time.
        d_head = cfg.d_model // cfg.n_heads
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head))
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

        self.drop = nn.Dropout(cfg.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([PreNormBlock(cfg) for _ in range(cfg.n_layers)])

        # Final layer norm
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Output head (weight-tied with token embedding)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.token_emb.weight  # weight tying

        self._init_weights()

    def _rope_cos_sin(self, T: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (cos, sin) of shape (T, d_head // 2) for current sequence length."""
        t = torch.arange(T, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.rope_inv_freq.to(device))
        return freqs.cos(), freqs.sin()

    def _init_weights(self):
        """Initialize weights following GPT-2 conventions."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        trade_tokens: torch.Tensor,    # (B, T)
        price_levels: torch.Tensor,     # (B, T)
        liquidities: torch.Tensor,      # (B, T)
        instrument_ids: torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        """Forward pass. Returns logits (B, T, vocab_size)."""
        _, T = trade_tokens.shape

        # Embeddings
        tok = self.token_emb(trade_tokens)  # (B, T, d_model)

        ctx = torch.cat([
            self.price_level_emb(price_levels),
            self.liquidity_emb(liquidities),
            self.instrument_emb(instrument_ids.unsqueeze(1).expand(-1, T)),
        ], dim=-1)  # (B, T, 3*d_context)
        ctx = self.context_proj(ctx)  # (B, T, d_model)

        x = self.drop(tok + ctx)

        cos, sin = self._rope_cos_sin(T, trade_tokens.device)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, cos, sin)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)
        return logits

    @torch.no_grad()
    def extract_latent(
        self,
        trade_tokens: torch.Tensor,
        price_levels: torch.Tensor,
        liquidities: torch.Tensor,
        instrument_ids: torch.Tensor,
        layer: int = -1,
    ) -> torch.Tensor:
        """Extract hidden state from specified layer at last position.

        Returns: (B, d_model) latent representation.
        """
        _, T = trade_tokens.shape

        tok = self.token_emb(trade_tokens)
        ctx = torch.cat([
            self.price_level_emb(price_levels),
            self.liquidity_emb(liquidities),
            self.instrument_emb(instrument_ids.unsqueeze(1).expand(-1, T)),
        ], dim=-1)
        ctx = self.context_proj(ctx)

        x = self.drop(tok + ctx)

        cos, sin = self._rope_cos_sin(T, trade_tokens.device)

        # Resolve negative layer index
        target_layer = layer if layer >= 0 else self.cfg.n_layers + layer

        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin)
            if i == target_layer:
                return x[:, -1, :]  # (B, d_model)

        # If layer == -1 or n_layers-1, we get here after all blocks
        return x[:, -1, :]

    @torch.no_grad()
    def extract_hidden_states(
        self,
        trade_tokens: torch.Tensor,
        price_levels: torch.Tensor,
        liquidities: torch.Tensor,
        instrument_ids: torch.Tensor,
        layer: int = -1,
    ) -> torch.Tensor:
        """Extract hidden states from specified layer at all positions.

        Returns: (B, T, d_model) hidden states.
        """
        _, T = trade_tokens.shape

        tok = self.token_emb(trade_tokens)
        ctx = torch.cat([
            self.price_level_emb(price_levels),
            self.liquidity_emb(liquidities),
            self.instrument_emb(instrument_ids.unsqueeze(1).expand(-1, T)),
        ], dim=-1)
        ctx = self.context_proj(ctx)

        x = self.drop(tok + ctx)

        cos, sin = self._rope_cos_sin(T, trade_tokens.device)

        target_layer = layer if layer >= 0 else self.cfg.n_layers + layer

        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin)
            if i == target_layer:
                return x  # (B, T, d_model)

        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
