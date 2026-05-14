---
name: DiffQuant article reference
description: Differentiable simulator approach for Stage 2 — alternative to RL, end-to-end Sharpe optimization
type: reference
---

Habr article: https://habr.com/ru/articles/1022254/
GitHub: github.com/YuriyKolesnikov/diffquant

**What it is:** end-to-end differentiable trading pipeline (model → position → simulator → PnL → Sharpe), trained via backprop, no RL needed.

**Key ideas for Stage 2:**
- Differentiable simulator as alternative to PPO/SAC — more stable, no reward shaping
- Direction × Gate policy: `position = tanh(d/τ) × σ(g/τ)`, gate starts biased toward flat
- Hybrid loss: `-Sharpe + turnover penalty + drawdown penalty + inventory bias penalty + flat fraction control`
- Without these regularizations, model collapses (hyperactivity → flat collapse → directional bias)
- smooth_abs(x) = sqrt(x² + ε) to keep gradients flowing through zero

**Why:** could replace RL in our Stage 2 with a simpler, gradient-based approach to optimize the A-S parameter combination or learn a direct policy.

**How to apply:** after Stage 1 (supervised heads) proves latents are informative, build a differentiable MM simulator wrapping hftbacktest logic and train policy end-to-end on Sharpe.
