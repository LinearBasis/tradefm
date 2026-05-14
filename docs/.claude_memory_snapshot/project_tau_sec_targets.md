---
name: Head targets switched to seconds-based horizon
description: 2026-05-11 — α/σ/κ targets now span tau_sec seconds, not tau events. A-S formula no longer takes tau. Old sequences parquets invalid.
type: project
originSessionId: 28eafb54-f262-4642-b4e4-695fffb66865
---
Decision-head targets (`src/decision/targets.py`) now use a **time-based forward window**, not an event-count one. Three places carry the canonical value:
- `PipelineConfig.tau_sec: float = 1.0` — used by `pipeline.py` when generating `data/processed/sequences/*.parquet`
- `HeadConfig.tau_sec: float = 1.0` — must match the above; documented for inference/backtest consistency
- `configs/base_heads.json` carries the same value

**Why:**
The A-S formula `reservation = mid + μ − γ·q·σ²·τ` lives in time units. Old targets measured σ²/μ over 512 *events*, which on liquid stocks is ~2 sec, on thin ones ~8 min — mismatched with τ in seconds used by the formula. Same numeric σ now meant different physical horizons across instruments.

**How to apply:**
- `compute_alpha_targets`, `compute_risk_targets`, `compute_intensity_targets` take `(mid_price, order_times, tau_sec)`. They use `np.searchsorted(order_times, order_times + tau_sec, side="left")` to find the future endpoint per event. Last events whose window extends past the day's last order event get NaN.
- `avellaneda_stoikov_quotes` **no longer accepts `tau`**. The formula is now `reservation = mid + μ − γ·q·σ²` (τ=1 implicit, since the heads already predict total μ/σ over the training window of `tau_sec` seconds).
- Old `data/processed/sequences/*.parquet` files (generated before this switch) are **incompatible** — the three target columns are events-based. Token columns (`trade_token`, `bin_price_level`, `bin_liquidity`) are byte-identical with new pipeline runs, but pipeline writes all 6 columns in one block, so the simplest fix is `rm -rf data/processed/sequences && python -m src.data.pipeline --tau-sec 1.0 ...`.
- `tau_sec` reasonable range: 0.5 / 1.0 / 5.0 / 30.0. Smaller → more valid targets near session end, sharper signal but more noise. The choice mostly affects the α-head (σ-head is robust either way).

Tokenizer (`src/data/tokenizer.py`, `features.py`), transformer architecture, and event-level granularity of model I/O are **unchanged** — only the *meaning of head targets* shifted.
