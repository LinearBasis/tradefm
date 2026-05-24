"""Rollout-based stylized fact metrics (paper §9.1-9.2).

Closed-loop generation: model predicts next i_trade token, simulator processes
it, simulator state feeds back as next-step context. Per-event time series
(mid, spread, OBI, iat, depth, vol, side, action) are collected for analysis.

Stylized facts on simulated log returns:
  - ACF(returns): should decay quickly to 0 (efficient markets)
  - ACF(|returns|): should decay slowly (volatility clustering)
  - kurtosis: high at short Δt, converges to 3 at long Δt (heavy tails)

Distributional fidelity: K-S and W1 distance per microstructure quantity.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from scipy import stats

from src.data.tokenizer import Tokenizer, subtoken_factors
from src.eval.simulator import MinimalLOB
from src.models.transformer import OrderFlowTransformer


# --------------------------------------------------------------------------- #
# Token decoding + sampling                                                    #
# --------------------------------------------------------------------------- #

def _decode_composite(token: int, factors: tuple[int, int, int, int, int]):
    _, n_s, n_d, n_v, n_t = factors
    block_a = n_s * n_d * n_v * n_t
    block_s = n_d * n_v * n_t
    block_d = n_v * n_t
    i_a, rem = divmod(token, block_a)
    i_s, rem = divmod(rem, block_s)
    i_d, rem = divmod(rem, block_d)
    i_v, i_t = divmod(rem, n_t)
    return int(i_a), int(i_s), int(i_d), int(i_v), int(i_t)


def _sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    repetition_penalty: float = 1.0,
    recent_tokens: list[int] | None = None,
) -> int:
    """Multinomial sampling with optional repetition penalty (paper §8.2 uses 1.2)."""
    logits = logits.clone()
    if recent_tokens and repetition_penalty != 1.0:
        for tok in set(recent_tokens):
            if logits[tok] > 0:
                logits[tok] /= repetition_penalty
            else:
                logits[tok] *= repetition_penalty
    if temperature != 1.0:
        logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


# --------------------------------------------------------------------------- #
# Rollout                                                                      #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def run_rollout(
    model: OrderFlowTransformer,
    tokenizer: Tokenizer,
    seed_tokens: list[int],
    seed_plevels: list[int],
    seed_liquidity: int,
    instrument_id: int,
    init_mid: float,
    n_events: int = 1024,
    temperature: float = 1.0,
    repetition_penalty: float = 1.2,
    context_length: int | None = None,
    device: torch.device | None = None,
    seed_book_levels: int = 20,
    seed_book_qty: int = 100,
    seed_book_tick_rel: float = 1e-4,
) -> dict[str, np.ndarray]:
    """Generate `n_events` via closed-loop model + simulator.

    Seed context (`seed_tokens`, `seed_plevels`) is supplied from real data so
    the model has a non-empty history before sampling. The simulator is seeded
    with a symmetric synthetic book around `init_mid`.
    """
    device = device or next(model.parameters()).device
    cfg = model.cfg
    ctx_len = context_length or cfg.context_length
    factors = subtoken_factors(cfg.vocab_size)

    sim = MinimalLOB(init_mid=init_mid)
    sim.seed_book(
        n_levels=seed_book_levels,
        qty_per_level=seed_book_qty,
        tick=init_mid * seed_book_tick_rel,
    )

    # Mutable history (rolling window).
    hist_tokens = list(seed_tokens)
    hist_plevels = list(seed_plevels)
    hist_liq = [seed_liquidity] * len(hist_tokens)

    out = {k: [] for k in (
        "ts", "mid", "spread", "obi", "bid_vol", "ask_vol",
        "iat", "depth", "vol", "action", "side", "token",
    )}

    model.eval()
    inst_tensor = torch.tensor([instrument_id], dtype=torch.long, device=device)

    for _ in range(n_events):
        window_t = hist_tokens[-ctx_len:]
        window_p = hist_plevels[-ctx_len:]
        window_l = hist_liq[-ctx_len:]
        tt = torch.tensor(window_t, dtype=torch.long, device=device).unsqueeze(0)
        pl_t = torch.tensor(window_p, dtype=torch.long, device=device).unsqueeze(0)
        liq_t = torch.tensor(window_l, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(tt, pl_t, liq_t, inst_tensor)[0, -1, :].float()

        recent = window_t[-32:] if repetition_penalty != 1.0 else None
        next_tok = _sample_token(logits, temperature, repetition_penalty, recent)

        i_a, i_s, i_d, i_v, i_t = _decode_composite(next_tok, factors)
        depth = tokenizer.bin_centroid("price_depth", i_d)
        vol = max(int(round(tokenizer.bin_centroid("volume", i_v))), 1)
        iat = max(tokenizer.bin_centroid("interarrival", i_t), 1e-4)

        snap = sim.step(action=i_a, side=i_s, price_depth=depth, volume=vol, iat_sec=iat)

        rel = (sim.mid - sim.opening_price) / max(sim.opening_price, 1e-9)
        new_plevel = int(np.digitize(np.array([rel]), tokenizer.edges.price_level)[0])
        new_plevel = max(0, min(new_plevel, cfg.n_price_level_bins - 1))

        hist_tokens.append(next_tok)
        hist_plevels.append(new_plevel)
        hist_liq.append(seed_liquidity)

        out["ts"].append(snap["ts"])
        out["mid"].append(snap["mid"])
        out["spread"].append(snap["spread"])
        out["obi"].append(snap["obi"])
        out["bid_vol"].append(snap["bid_vol"])
        out["ask_vol"].append(snap["ask_vol"])
        out["iat"].append(iat)
        out["depth"].append(depth)
        out["vol"].append(vol)
        out["action"].append(i_a)
        out["side"].append(i_s)
        out["token"].append(next_tok)

    return {k: np.asarray(v) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# Stylized facts                                                               #
# --------------------------------------------------------------------------- #

def _resample_log_returns(mid: np.ndarray, ts: np.ndarray, delta_t: float) -> np.ndarray:
    """Step-interpolate mid(t) onto a uniform Δt-second grid, return log returns.

    Grid: ts[0], ts[0]+Δt, ..., ts[-1]. At each grid point, takes the last
    observed mid with ts <= grid_point (right-continuous step function).
    """
    if len(mid) < 2 or delta_t <= 0:
        return np.array([])
    t0, t1 = float(ts[0]), float(ts[-1])
    if t1 - t0 < delta_t:
        return np.array([])
    n_grid = int((t1 - t0) / delta_t) + 1
    grid = t0 + np.arange(n_grid) * delta_t
    idx = np.searchsorted(ts, grid, side="right") - 1
    idx = np.clip(idx, 0, len(mid) - 1)
    mid_on_grid = mid[idx]
    finite = (mid_on_grid > 0) & np.isfinite(mid_on_grid)
    if finite.sum() < 2:
        return np.array([])
    log_ret = np.diff(np.log(mid_on_grid))
    return log_ret[np.isfinite(log_ret)]


def compute_stylized_facts(
    mid: np.ndarray,
    ts: np.ndarray,
    delta_t_seconds: tuple[float, ...] = (10.0, 30.0, 60.0, 120.0),
    acf_delta_t: float = 1.0,
    max_lag: int = 50,
) -> dict:
    """Paper-§9.1 stylized facts on simulated log returns.

    - ACF(returns) and ACF(|returns|): on log returns resampled at `acf_delta_t`
      seconds (default 1s). Lags are in units of `acf_delta_t`.
    - kurtosis at multiple Δt_r: wall-clock time aggregation per paper Fig. 4
      (right panel) — log returns resampled at each Δt and kurtosis computed.

    Both use step-interpolation of mid(t) onto a uniform grid, which is what
    you want for a non-uniform event stream: kurtosis_10s means "kurtosis of
    10-second log returns" regardless of underlying event rate.
    """
    mid = np.asarray(mid, dtype=np.float64)
    ts = np.asarray(ts, dtype=np.float64)
    finite = np.isfinite(mid) & (mid > 0) & np.isfinite(ts)
    mid = mid[finite]
    ts = ts[finite]
    if len(mid) < 3:
        empty = np.array([1.0])
        return {
            "acf_returns": empty,
            "acf_abs_returns": empty,
            "kurtosis_event": float("nan"),
            **{f"kurtosis_{int(d)}s": float("nan") for d in delta_t_seconds},
        }

    # Per-event log returns — kept as informational baseline kurtosis.
    event_log_ret = np.diff(np.log(mid))
    event_log_ret = event_log_ret[np.isfinite(event_log_ret)]

    # ACF on Δt-resampled log returns (paper Fig. 4 left + middle, x in seconds).
    acf_ret = _resample_log_returns(mid, ts, acf_delta_t)
    n_acf = len(acf_ret)
    lags = min(max_lag, max(n_acf - 2, 1))

    def acf(x, n_lags):
        out = np.zeros(n_lags + 1)
        if len(x) < 2:
            out[0] = 1.0
            return out
        x_c = x - x.mean()
        var = (x_c ** 2).sum()
        if var == 0:
            out[0] = 1.0
            return out
        n = len(x)
        for k in range(n_lags + 1):
            out[k] = (x_c[: n - k] * x_c[k:]).sum() / var
        return out

    acf_r = acf(acf_ret, lags)
    acf_abs = acf(np.abs(acf_ret), lags)

    # Wall-clock kurtosis at multiple Δt (paper Fig. 4 right).
    kurts = {}
    for d in delta_t_seconds:
        ret_d = _resample_log_returns(mid, ts, d)
        kurts[f"kurtosis_{int(d)}s"] = (
            float(stats.kurtosis(ret_d)) if len(ret_d) >= 4 else float("nan")
        )

    return {
        "acf_returns": acf_r,
        "acf_abs_returns": acf_abs,
        "acf_delta_t": acf_delta_t,
        "kurtosis_event": float(stats.kurtosis(event_log_ret)) if len(event_log_ret) >= 4 else float("nan"),
        **kurts,
    }


# --------------------------------------------------------------------------- #
# Distributional fidelity                                                      #
# --------------------------------------------------------------------------- #

def compute_distributional_fidelity(
    real: dict[str, np.ndarray],
    gen: dict[str, np.ndarray],
    quantities: Iterable[str] = ("spread", "iat", "depth", "vol", "obi", "bid_vol", "ask_vol"),
) -> dict[str, dict[str, float]]:
    """Per-quantity K-S statistic and Wasserstein-1 distance between real and gen.

    K-S is scale-invariant (operates on CDFs) → reported on raw values.
    W₁ is in units of the underlying quantity → paper §9.2 mean-variance
    normalises both distributions by the **real** distribution's stats before
    W₁ to make the metric comparable across quantities (bp vs shs etc.).
    """
    out: dict[str, dict[str, float]] = {}
    for q in quantities:
        if q not in real or q not in gen:
            continue
        r = real[q][np.isfinite(real[q])]
        g = gen[q][np.isfinite(gen[q])]
        if len(r) < 2 or len(g) < 2:
            out[q] = {"ks": float("nan"), "w1": float("nan")}
            continue
        ks_stat, _ = stats.ks_2samp(r, g)
        mu, sd = float(r.mean()), float(r.std())
        if sd > 0:
            w1 = stats.wasserstein_distance((r - mu) / sd, (g - mu) / sd)
        else:
            w1 = stats.wasserstein_distance(r, g)
        out[q] = {"ks": float(ks_stat), "w1": float(w1)}
    return out
