---
name: Transformer uses RoPE, not learned absolute positional embeddings
description: 2026-05-09 switched src/models/transformer.py from learned absolute pos_emb to RoPE. Old checkpoints incompatible.
type: project
originSessionId: b2f57b68-c034-46f4-af86-74f3c3a74f04
---
`OrderFlowTransformer` uses **RoPE** (rotary positional embedding) applied to q/k inside `CausalSelfAttention`, with no learned positional parameters. The old `self.pos_emb = nn.Embedding(context_length, d_model)` was removed.

**Why:**
Polars audit on 5 days × 20 instruments showed cumulative event-token counts at the start of the continuous session (10:05) grow much faster than `context_length=512`: median ~4700 tokens by 10:07, ~14K by 10:10, ~156K by 11:00, low millions by EoD on YNDX/MGNT. Any realistic `context_length` is exhausted long before session end on liquid stocks → **sliding window is mandatory in prod**. Learned absolute pos_emb is OOD when the window slides (K/V in the cache were computed with positional info baked in; shifting positions logically without recomputing K/V is mathematically incorrect). RoPE attention depends only on the *relative* angle `q_pos − k_pos`, so sliding the window is exact and KV-cache stays valid without recomputation.

**How to apply:**
- Do not reintroduce `nn.Embedding(...)` for positions in `transformer.py`. Any new attention block must accept `cos, sin` from `OrderFlowTransformer._rope_cos_sin(T, device)` and call `apply_rotary(q, cos, sin)` and `apply_rotary(k, cos, sin)` before SDPA.
- `cfg.context_length` is now a **training-budget hyperparameter** (length of dataset windows in `dataset.py`), not a hard architectural cap. The model can run inference at any T (verified: T=1024 forward works with cfg.context_length=512). The `assert T <= context_length` in `forward` was removed.
- Old transformer checkpoints (pre-2026-05-09) are incompatible — they reference `pos_emb.weight`. Heads (μ/σ/κ) trained on the old latent are also invalid. Full retrain required: transformer first, then heads.
- For prod KV-cache `step()` (still TODO), assign monotonically increasing positions to new tokens; sliding window = drop oldest K/V rows and continue incrementing. No position-shift recomputation needed thanks to RoPE.
- RoPE base (theta) is hardcoded to 10000.0 in `OrderFlowTransformer.__init__`. If extreme extrapolation past training length is wanted, consider NTK-aware scaling — but for ≤4× extrapolation 10000.0 is fine.
