# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Diploma project: learning latent market representations from trade/order flow event data (inspired by the TradeFM paper in `notes/`) and using them for trading decision-making. **Not** direct price prediction.

## Pipeline (planned)

Raw trades (CSV) → scale-invariant features → sequence construction → autoregressive sequence model → latent market state → decision module (buy/sell/hold) → backtesting (PnL, Sharpe, drawdown).

## Current State

Early stage — no `src/` code yet. Key assets:
- `data.csv` (~400MB, ~7.6M rows) — raw trade/order flow data with TIME column (HHMMSSXXXXXX format, 10:00–18:45 trading session)
- `eda_time.ipynb` — exploratory analysis of TIME distribution and order rate
- `docs/README.md` — full project description and planned structure
- `notes/2602.23784v1.pdf` — TradeFM reference paper

## Data Notes

- TIME is encoded as HHMMSS + 6-digit microseconds (e.g., `184459475316` → 18:44:59.475316)
- ~39,667 rows have default TIME=100000000000 (10:00:00.000000) — likely pre-market or batch orders
- Median order rate ~170/sec, extreme spikes up to ~51K/sec (18:40:01 close auction)
- Trading session: 10:00–18:45

## Planned Structure (from README)

```
src/data/       — data loading, feature engineering
src/models/     — sequence / generative models
src/decision/   — decision-making heads
src/backtest/   — backtesting engine
src/utils/      — shared utilities
configs/        — experiment configs
notebooks/      — analysis notebooks
```

## Language & Tools

Project uses Python (pandas, numpy, matplotlib). Model framework not yet chosen.
Notebooks use Russian-language comments and labels.
