---
name: Strategy direction — market making
description: Market making strategy (not directional). hftbacktest L3 mode as evaluation simulator. ESkripichnikov rejected. DiffQuant diff-sim is a separate training track.
type: project
---

Strategy is **market making** (двусторонние котировки bid/ask), not directional (buy/sell signal).

**Baseline:** Avellaneda-Stoikov / GLFT — uses only volatility and trading intensity.
**Our model:** TradeFM representation → decision head outputs μ/σ/κ → A-S/GLFT formula.

## Backtesting infrastructure

**Primary simulator:** hftbacktest in **L3 mode** (market-by-order).
- Repo: github.com/nkaz001/hftbacktest
- L3 mode chosen because MOEX OrderLog ACTION 1/0/2 maps 1:1 to ADD_ORDER_EVENT / CANCEL_ORDER_EVENT / FILL_EVENT — no information loss.
- Modeling realism: queue position, partial fills, latency, fees — all built-in.
- Data format: npz with structured array `(ev, exch_ts, local_ts, px, qty, order_id, ival, faval)`.
- Conversion details in `reference_hftbacktest.md`.

**Differentiable simulator (Exp 3):** separate track for E2E ∇Sharpe training.
- hftbacktest is **not differentiable** (Numba/Rust, discrete queue ops).
- Diff-sim trains policy via gradient flow; final evaluation is in hftbacktest L3.
- Pattern: "train on diff-sim, evaluate on hftbacktest" (sim-to-real analog).
- Best fill model variant: **Variant D** — surrogate NN trained on hftbacktest data → differentiable but realistic.

**Rejected:** ESkripichnikov simulator (github.com/ESkripichnikov/market-making).
- Used by Smirnov in his thesis, recommended by научник.
- Has ready-made A-S baselines but **no queue position** → fill rate overestimated 2–5×.
- Snapshot-based, single-asset, slow on 10M+ ticks. Bad fit for honest MM evaluation.

## Why
Market making is natural fit for event-level microstructure modeling. Fair price + spread/skew estimation is where learned representation can add value over classical models.

## How to apply
- Decision head outputs spread/skew parameters (or μ/σ/κ for A-S formula), not buy/sell signal.
- Evaluation: PnL, Sharpe, max drawdown, avg inventory, fill rate, avg spread, turnover.
- Comparison vs A-S/GLFT baseline in **same simulator** (hftbacktest L3) — closes the gap that Smirnov left (he only compared vs buy-and-hold).
- First infra task: MOEX OrderLog → npz converter (`src/data/moex_to_hftbacktest.py`).
