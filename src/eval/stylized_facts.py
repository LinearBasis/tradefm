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
        "mid", "spread", "obi", "bid_vol", "ask_vol",
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

def compute_stylized_facts(returns: np.ndarray, max_lag: int = 50) -> dict:
    """ACF of returns + ACF of |returns| + kurtosis at increasing aggregation."""
    returns = returns[np.isfinite(returns)]
    n = len(returns)
    if n < 3:
        return {
            "acf_returns": np.array([1.0]),
            "acf_abs_returns": np.array([1.0]),
            "kurtosis": float("nan"),
            "kurtosis_agg_2": float("nan"),
            "kurtosis_agg_5": float("nan"),
            "kurtosis_agg_10": float("nan"),
        }
    max_lag = min(max_lag, n - 2)
    abs_returns = np.abs(returns)

    def acf(x, lags):
        out = np.zeros(lags + 1)
        x_centered = x - x.mean()
        var = (x_centered ** 2).sum()
        if var == 0:
            out[0] = 1.0
            return out
        for k in range(lags + 1):
            out[k] = (x_centered[: n - k] * x_centered[k:]).sum() / var
        return out

    def agg_kurt(step):
        if n < 2 * step + 2:
            return float("nan")
        # Sum every `step` consecutive returns (proxy for time aggregation)
        trimmed = returns[: (n // step) * step].reshape(-1, step).sum(axis=1)
        if len(trimmed) < 4:
            return float("nan")
        return float(stats.kurtosis(trimmed))

    return {
        "acf_returns": acf(returns, max_lag),
        "acf_abs_returns": acf(abs_returns, max_lag),
        "kurtosis": float(stats.kurtosis(returns)),
        "kurtosis_agg_2": agg_kurt(2),
        "kurtosis_agg_5": agg_kurt(5),
        "kurtosis_agg_10": agg_kurt(10),
    }


# --------------------------------------------------------------------------- #
# Distributional fidelity                                                      #
# --------------------------------------------------------------------------- #

def compute_distributional_fidelity(
    real: dict[str, np.ndarray],
    gen: dict[str, np.ndarray],
    quantities: Iterable[str] = ("spread", "iat", "depth", "vol", "obi", "bid_vol", "ask_vol"),
) -> dict[str, dict[str, float]]:
    """Per-quantity K-S statistic and Wasserstein-1 distance between real and gen distributions."""
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
        w1 = stats.wasserstein_distance(r, g)
        out[q] = {"ks": float(ks_stat), "w1": float(w1)}
    return out
