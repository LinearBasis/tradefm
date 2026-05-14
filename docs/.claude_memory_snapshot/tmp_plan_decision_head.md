---
name: Decision head plan
description: Two-stage plan for decision module — supervised multi-head (alpha/risk/intensity) then RL fine-tuning
type: project
---

## Stage 1: Supervised multi-head over frozen latents

Three heads on top of `extract_latent()` output (d_model=128):

1. **Alpha (μ)** — Linear → scalar. Target: future mid-price return over horizon τ
2. **Risk (σ)** — Linear → scalar (positive). Target: realized volatility over horizon τ
3. **Intensity (κ)** — Linear → scalar (positive). Target: empirical fill rate proxy

Combination via A-S/GLFT formula:
- `reservation_price = mid + μ − γ * q * σ²τ`
- `optimal_spread = (2/γ) * ln(1 + γ/κ)`

Evaluate in hftbacktest → PnL, Sharpe, drawdown.

## Stage 2: RL fine-tuning (if Stage 1 works)

State: `[latent, μ, σ, κ, inventory, position_age, ...]`
Action: `(δ_bid, δ_ask, size_bid, size_ask)` — continuous or discretized
Reward: `realized PnL per step − γ * inventory²`

Policy: small MLP, trained with PPO or SAC in hftbacktest simulator.

**Why staged:** RL without working supervised heads = endless reward shaping debug. Stage 1 proves latents are informative and gives interpretable baseline for the thesis.

**Key risks:**
- Fill rate (κ) hardest to estimate — no queue position in data, start with empirical trade frequency proxy
- RL sim-to-real gap with hftbacktest
- Inventory penalty γ needs tuning
