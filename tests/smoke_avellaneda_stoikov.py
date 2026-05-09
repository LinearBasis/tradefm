"""Smoke test: full chain transformer → heads → A-S quotes.

Loads the smoke transformer + heads checkpoints, runs one batch of the val
dataset through the model, computes μ/σ/κ via the heads, then applies the
Avellaneda-Stoikov formula for several inventory levels and prints summary
statistics. Verifies that the pipeline produces finite, sensibly-shaped
quotes; not a quality check.

Run:
    PYTHONPATH=. uv run python -m tests.smoke_avellaneda_stoikov
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import HeadConfig, ModelConfig, load_from_json
from src.data.dataset import OrderFlowDataset
from src.decision.avellaneda_stoikov import avellaneda_stoikov_quotes
from src.decision.heads import DecisionModule
from src.models.transformer import OrderFlowTransformer

TRANSFORMER_CKPT = Path("/tmp/tradefm_smoke/transformer/best.pt")
HEADS_CKPT = Path("/tmp/tradefm_smoke/heads/best_heads.pt")
HEADS_CONFIG = "configs/smoke_heads.json"

# A-S parameters for the smoke.
GAMMA = 0.1
INVENTORY_LEVELS = [-10.0, -1.0, 0.0, 1.0, 10.0]
# Synthetic mid (we don't have real prices in the tokenized parquet — only bins).
# 100.0 is just a sentinel; the point of the smoke is shape and finiteness.
MOCK_MID = 100.0


def main():
    device = torch.device("cpu")

    # --- Load transformer ---------------------------------------------------
    ck_xfmr = torch.load(TRANSFORMER_CKPT, map_location=device, weights_only=False)
    model_cfg: ModelConfig = ck_xfmr["config"]
    transformer = OrderFlowTransformer(model_cfg).to(device)
    transformer.load_state_dict(ck_xfmr["model_state_dict"])
    transformer.eval()
    for p in transformer.parameters():
        p.requires_grad = False
    print(f"Transformer: {sum(p.numel() for p in transformer.parameters()):,} params, "
          f"d_model={model_cfg.d_model}, context_length={model_cfg.context_length}")

    # --- Load heads ---------------------------------------------------------
    head_cfg: HeadConfig = load_from_json(HeadConfig, HEADS_CONFIG)
    heads = DecisionModule(model_cfg.d_model).to(device)
    ck_heads = torch.load(HEADS_CKPT, map_location=device, weights_only=False)
    heads.load_state_dict(ck_heads["heads_state_dict"])
    heads.eval()
    print(f"Heads: {sum(p.numel() for p in heads.parameters()):,} params, "
          f"τ={head_cfg.tau}")

    # --- One batch from val -------------------------------------------------
    ds = OrderFlowDataset(model_cfg, split="val")
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    print(f"Batch: tokens {tuple(batch['trade_tokens'].shape)}, "
          f"instruments {batch['instrument_id'].tolist()}")

    # --- Forward through transformer + heads -------------------------------
    inputs = {
        "trade_tokens": batch["trade_tokens"][:, :-1].to(device),
        "price_levels": batch["price_levels"][:, :-1].to(device),
        "liquidities": batch["liquidities"][:, :-1].to(device),
        "instrument_ids": batch["instrument_id"].to(device),
    }
    with torch.no_grad():
        hidden = transformer.extract_hidden_states(**inputs)        # (B, T, d_model)
        preds = heads(hidden)                                       # dict of (B, T)
    mu, sigma, kappa = preds["alpha"], preds["risk"], preds["intensity"]
    print(f"\nHead outputs: shape {tuple(mu.shape)}")
    print(f"  μ (alpha):     mean={mu.mean().item():+.3e}  std={mu.std().item():.3e}  "
          f"min={mu.min().item():+.3e}  max={mu.max().item():+.3e}")
    print(f"  σ (risk):      mean={sigma.mean().item():.3e}  std={sigma.std().item():.3e}  "
          f"min={sigma.min().item():.3e}  max={sigma.max().item():.3e}")
    print(f"  κ (intensity): mean={kappa.mean().item():.3e}  std={kappa.std().item():.3e}  "
          f"min={kappa.min().item():.3e}  max={kappa.max().item():.3e}")
    assert (sigma > 0).all(), "sigma must be positive (softplus enforced)"
    assert (kappa > 0).all(), "kappa must be positive (softplus enforced)"
    assert torch.isfinite(mu).all() and torch.isfinite(sigma).all() and torch.isfinite(kappa).all()

    # --- A-S quoting at the last position of each sequence -----------------
    # Take the last-position prediction per sample: this is the "current" state.
    mu_t = mu[:, -1]      # (B,)
    sigma_t = sigma[:, -1]
    kappa_t = kappa[:, -1]
    mid = torch.full_like(mu_t, MOCK_MID)

    print(f"\nA-S quotes (mock mid={MOCK_MID}, γ={GAMMA}, τ={head_cfg.tau}):")
    print(f"{'q':>6}  {'reservation':>12}  {'half_spread':>12}  {'bid':>10}  {'ask':>10}  {'skew':>10}")
    for q_val in INVENTORY_LEVELS:
        q = torch.full_like(mu_t, q_val)
        out = avellaneda_stoikov_quotes(
            mid=mid, mu=mu_t, sigma=sigma_t, kappa=kappa_t,
            inventory=q, gamma=GAMMA, tau=float(head_cfg.tau),
        )
        # Take batch-mean for printout
        res = out["reservation"].mean().item()
        hs = out["half_spread"].mean().item()
        bid = out["bid"].mean().item()
        ask = out["ask"].mean().item()
        skew = res - MOCK_MID
        print(f"{q_val:>+6.1f}  {res:>12.6f}  {hs:>12.6f}  {bid:>10.6f}  {ask:>10.6f}  {skew:>+10.6f}")
        for name, t in out.items():
            assert torch.isfinite(t).all(), f"{name} has non-finite values at q={q_val}"

    print("\nAll assertions passed: μ/σ/κ finite, σ/κ positive, A-S quotes finite.")


if __name__ == "__main__":
    main()
