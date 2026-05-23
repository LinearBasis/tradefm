"""RL infrastructure for market making.

Layout:
- config.py   — EnvConfig / AgentConfig / TrainConfig dataclasses
- action.py   — Modes A/B/C + their discrete variants → bid/ask quotes
- reward.py   — quadratic A-S utility reward
- state.py    — state builders (heads-only and h_t variants)
- env.py      — RLMarketMakingEnv wrapping hftbacktest
"""
