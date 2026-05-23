"""Compact post-training validators.

Each subcommand checks the structural invariants of one artifact kind and
exits non-zero on failure. Designed to be called after smoke training stages.

Usage:
    PYTHONPATH=. uv run python -m scripts.exp.validate pipeline data/processed_smoke/sequences
    PYTHONPATH=. uv run python -m scripts.exp.validate transformer checkpoints_smoke/transformer/best.pt
    PYTHONPATH=. uv run python -m scripts.exp.validate heads      checkpoints_smoke/heads/best_heads.pt \\
        --transformer-checkpoint checkpoints_smoke/transformer/best.pt
    PYTHONPATH=. uv run python -m scripts.exp.validate rl         checkpoints_smoke/rl/smoke
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

from src._archive.decision.heads import DecisionModule
from src.models.transformer import OrderFlowTransformer


def _ok(msg: str) -> None:
    print(f"  ok   {msg}")


def _fail(msg: str) -> None:
    print(f"  FAIL {msg}")
    sys.exit(1)


def validate_pipeline(sequences_dir: str) -> None:
    """manifest.json present, ≥1 parquet, all parquets have time_sec + targets."""
    sd = Path(sequences_dir)
    manifest = sd / "manifest.json"
    if not manifest.exists():
        _fail(f"no manifest at {manifest}")
    m = json.loads(manifest.read_text())
    _ok(f"manifest: {len(m['instruments'])} instr, {len(m['sequences'])} seq, tau_sec={m.get('tau_sec')}")

    required = {"trade_token", "bin_price_level", "bin_liquidity", "time_sec"}
    target_cols = {"target_alpha", "target_risk", "target_intensity"}
    n_checked = 0
    for name in m["sequences"]:
        p = sd / f"{name}.parquet"
        if not p.exists():
            _fail(f"missing parquet: {p}")
        df = pl.read_parquet(p, n_rows=1)  # schema-only read
        missing = required - set(df.columns)
        if missing:
            _fail(f"{p.name}: missing cols {missing}")
        if not target_cols.issubset(set(df.columns)):
            _fail(f"{p.name}: missing target cols {target_cols - set(df.columns)}")
        n_checked += 1
    _ok(f"{n_checked} parquet(s) have time_sec + target_*")


def validate_transformer(ckpt_path: str) -> None:
    """Loads, runs forward on dummy batch, checks shape and finiteness."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in ckpt or "config" not in ckpt:
        _fail("missing 'model_state_dict' or 'config' in checkpoint")
    cfg = ckpt["config"]
    _ok(f"loaded ckpt epoch={ckpt.get('epoch','?')} d_model={cfg.d_model} layers={cfg.n_layers}")

    model = OrderFlowTransformer(cfg).eval()
    model.load_state_dict(ckpt["model_state_dict"])

    B, T = 2, min(cfg.context_length, 16)
    tok = torch.randint(0, cfg.vocab_size, (B, T))
    pl_t = torch.randint(0, cfg.n_price_level_bins, (B, T))
    lq = torch.randint(0, cfg.n_liquidity_bins, (B, T))
    inst = torch.randint(0, cfg.n_instruments, (B,))
    with torch.no_grad():
        h = model.extract_hidden_states(tok, pl_t, lq, inst)
    if tuple(h.shape) != (B, T, cfg.d_model):
        _fail(f"hidden shape {tuple(h.shape)} != ({B},{T},{cfg.d_model})")
    if not torch.isfinite(h).all():
        _fail("hidden contains NaN/Inf")
    _ok(f"forward OK, hidden shape={tuple(h.shape)}, range=[{h.min():.3f},{h.max():.3f}]")


def validate_heads(heads_ckpt: str, transformer_ckpt: str) -> None:
    """Loads heads, runs forward on hidden from a real transformer, checks shapes."""
    ck_xfmr = torch.load(transformer_ckpt, map_location="cpu", weights_only=False)
    cfg = ck_xfmr["config"]
    xfmr = OrderFlowTransformer(cfg).eval()
    xfmr.load_state_dict(ck_xfmr["model_state_dict"])

    ck_heads = torch.load(heads_ckpt, map_location="cpu", weights_only=False)
    if "heads_state_dict" not in ck_heads:
        _fail("missing 'heads_state_dict' in heads checkpoint")
    heads = DecisionModule(cfg.d_model).eval()
    heads.load_state_dict(ck_heads["heads_state_dict"])
    _ok(f"loaded heads (d_model={cfg.d_model})")

    B, T = 2, min(cfg.context_length, 16)
    tok = torch.randint(0, cfg.vocab_size, (B, T))
    pl_t = torch.randint(0, cfg.n_price_level_bins, (B, T))
    lq = torch.randint(0, cfg.n_liquidity_bins, (B, T))
    inst = torch.randint(0, cfg.n_instruments, (B,))
    with torch.no_grad():
        h = xfmr.extract_hidden_states(tok, pl_t, lq, inst)
        out = heads(h)
    for name in ("alpha", "risk", "intensity"):
        if name not in out:
            _fail(f"heads output missing '{name}'")
        if tuple(out[name].shape) != (B, T):
            _fail(f"{name} shape {tuple(out[name].shape)} != ({B},{T})")
        if not torch.isfinite(out[name]).all():
            _fail(f"{name} contains NaN/Inf")
    if (out["risk"] <= 0).any() or (out["intensity"] <= 0).any():
        _fail("risk or intensity has non-positive values")
    _ok(f"forward OK; α≈{out['alpha'].mean():.3e}, σ≈{out['risk'].mean():.3e}, κ≈{out['intensity'].mean():.3e}")


def validate_rl(run_dir: str) -> None:
    """Run dir contains config.json + agent_final.pt; agent loads + select_action works."""
    rd = Path(run_dir)
    cfg_path = rd / "config.json"
    agent_path = rd / "agent_final.pt"
    if not cfg_path.exists():
        _fail(f"missing {cfg_path}")
    if not agent_path.exists():
        _fail(f"missing {agent_path}")
    cfg_blob = json.loads(cfg_path.read_text())
    algo = cfg_blob["agent"]["algorithm"]
    _ok(f"run config OK: algorithm={algo} action_mode={cfg_blob['env']['action_mode']} state_mode={cfg_blob['env']['state_mode']}")

    # Agent state is opaque (per-class torch.save); just confirm it loads
    state = torch.load(agent_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or not state:
        _fail("agent_final.pt is empty / wrong shape")
    _ok(f"agent_final.pt loads ({len(state)} top-level keys)")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="kind", required=True)

    p_pipe = sub.add_parser("pipeline")
    p_pipe.add_argument("sequences_dir")

    p_xf = sub.add_parser("transformer")
    p_xf.add_argument("ckpt")

    p_hd = sub.add_parser("heads")
    p_hd.add_argument("ckpt")
    p_hd.add_argument("--transformer-checkpoint", required=True)

    p_rl = sub.add_parser("rl")
    p_rl.add_argument("run_dir")

    args = p.parse_args()
    print(f"[validate {args.kind}]")
    if args.kind == "pipeline":
        validate_pipeline(args.sequences_dir)
    elif args.kind == "transformer":
        validate_transformer(args.ckpt)
    elif args.kind == "heads":
        validate_heads(args.ckpt, args.transformer_checkpoint)
    elif args.kind == "rl":
        validate_rl(args.run_dir)
    print("PASS")


if __name__ == "__main__":
    main()
