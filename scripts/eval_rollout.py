"""Standalone CLI: load a checkpoint, run closed-loop rollouts, compute stylized facts.

Usage:
    python -m scripts.eval_rollout \
        --checkpoint checkpoints/best.pt \
        --tokenizer data/processed/tokenizer.json \
        --val-sequences data/processed/sequences/ \
        --n-rollouts 10 --n-events 1024 \
        --output runs/eval/best/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.tensorboard import SummaryWriter

from src.config import ModelConfig
from src.data.tokenizer import Tokenizer, subtoken_factors
from src.eval.stylized_facts import (
    compute_distributional_fidelity,
    compute_stylized_facts,
    run_rollout,
)
from src.models.transformer import OrderFlowTransformer


def _load_model(ckpt_path: Path, device: torch.device) -> tuple[OrderFlowTransformer, ModelConfig]:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg: ModelConfig = ck["config"]
    model = OrderFlowTransformer(cfg).to(device)
    model.load_state_dict(ck["model_state_dict"], strict=False)
    model.eval()
    return model, cfg


def _decode_real_distributions(
    val_parquets: list[Path], tokenizer: Tokenizer, vocab_size: int,
) -> dict[str, np.ndarray]:
    """Reconstruct (iat, depth, vol) distributions from tokenized val parquets
    by decoding composite tokens through bin centroids. Spread/OBI/volumes are
    not stored in tokenized parquets and are returned as empty arrays."""
    n_a, n_s, n_d, n_v, n_t = subtoken_factors(vocab_size)
    block_a = n_s * n_d * n_v * n_t
    block_s = n_d * n_v * n_t
    block_d = n_v * n_t

    iats, depths, vols = [], [], []
    for pf in val_parquets:
        toks = pl.read_parquet(pf)["trade_token"].to_numpy()
        rem_after_a = toks % block_a
        i_s = rem_after_a // block_s
        rem_after_s = rem_after_a % block_s
        i_d = rem_after_s // block_d
        rem_after_d = rem_after_s % block_d
        i_v = rem_after_d // n_t
        i_t = rem_after_d % n_t
        # Bin centroids → continuous values (vectorized via per-bin LUT).
        iat_lut = np.array([tokenizer.bin_centroid("interarrival", b) for b in range(n_t)])
        vol_lut = np.array([tokenizer.bin_centroid("volume", b) for b in range(n_v)])
        depth_lut = np.array([tokenizer.bin_centroid("price_depth", b) for b in range(n_d)])
        iats.append(iat_lut[i_t])
        vols.append(vol_lut[i_v])
        depths.append(depth_lut[i_d])

    return {
        "iat": np.concatenate(iats) if iats else np.array([]),
        "vol": np.concatenate(vols) if vols else np.array([]),
        "depth": np.concatenate(depths) if depths else np.array([]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--val-sequences", type=str, required=True)
    parser.add_argument("--instrument", type=str, default=None,
                        help="Instrument to roll out for (default: first in manifest)")
    parser.add_argument("--n-rollouts", type=int, default=5)
    parser.add_argument("--n-events", type=int, default=512)
    parser.add_argument("--init-mid", type=float, default=100.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    seq_dir = Path(args.val_sequences)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = _load_model(Path(args.checkpoint), device)
    tokenizer = Tokenizer.load(Path(args.tokenizer))
    manifest = json.load(open(seq_dir / "manifest.json"))
    instruments = manifest["instruments"]
    # Reproducibility check: model, tokenizer, and val data must agree on the
    # instrument set. Silent mismatch → wrong instrument_id → garbage rollouts.
    if sorted(tokenizer.instruments) != sorted(instruments):
        raise RuntimeError(
            f"Instrument mismatch: tokenizer.instruments ({len(tokenizer.instruments)}) "
            f"!= manifest.instruments ({len(instruments)}). Tokenizer-only: "
            f"{sorted(set(tokenizer.instruments) - set(instruments))[:5]}, "
            f"manifest-only: {sorted(set(instruments) - set(tokenizer.instruments))[:5]}"
        )
    if cfg.n_instruments != len(instruments):
        raise RuntimeError(
            f"Model trained on {cfg.n_instruments} instruments, val data has "
            f"{len(instruments)}. Probably wrong checkpoint or wrong --val-sequences."
        )
    instrument = args.instrument or instruments[0]
    inst_id = instruments.index(instrument)

    # Seed window from a val parquet for this instrument.
    val_keys = [k for k, v in manifest["sequences"].items()
                if v["split"] == "val" and k.startswith(instrument + "_")]
    if not val_keys:
        raise RuntimeError(f"No val sequences for instrument {instrument!r}")
    seed_df = pl.read_parquet(seq_dir / f"{val_keys[0]}.parquet")
    ctx = min(cfg.context_length, seed_df.height)
    seed_tokens = seed_df["trade_token"].to_numpy()[:ctx].astype(np.int64).tolist()
    seed_plevels = seed_df["bin_price_level"].to_numpy()[:ctx].astype(np.int64).tolist()
    seed_liq = int(seed_df["bin_liquidity"][0])

    # Real distributions from val parquets (centroid-approximated for iat/depth/vol).
    real_val_paths = [seq_dir / f"{k}.parquet" for k in val_keys]
    real_dist = _decode_real_distributions(real_val_paths, tokenizer, cfg.vocab_size)

    # Run rollouts.
    print(f"Instrument: {instrument} (id={inst_id}) | rollouts={args.n_rollouts} × {args.n_events}")
    all_gens = []
    for i in range(args.n_rollouts):
        torch.manual_seed(args.seed + i)
        gen = run_rollout(
            model=model, tokenizer=tokenizer,
            seed_tokens=seed_tokens, seed_plevels=seed_plevels, seed_liquidity=seed_liq,
            instrument_id=inst_id, init_mid=args.init_mid, n_events=args.n_events,
            temperature=args.temperature, repetition_penalty=args.repetition_penalty,
            device=device,
        )
        print(f"  rollout {i+1}/{args.n_rollouts}: mid range [{gen['mid'].min():.4f}, {gen['mid'].max():.4f}]")
        all_gens.append(gen)

    gen_pool = {k: np.concatenate([g[k] for g in all_gens]) for k in all_gens[0].keys()}
    # Per-rollout stylized facts, then average. Don't concatenate (m1, m2)
    # across rollouts — the join boundary fakes a non-physical log return.
    sf_list = [compute_stylized_facts(g["mid"], g["ts"]) for g in all_gens]

    def _mean(key):
        vals = [s[key] for s in sf_list if np.isfinite(s[key])]
        return float(np.mean(vals)) if vals else float("nan")

    def _mean_lag(key, idx):
        vals = [float(s[key][idx]) for s in sf_list
                if len(s[key]) > idx and np.isfinite(s[key][idx])]
        return float(np.mean(vals)) if vals else float("nan")

    fidelity = compute_distributional_fidelity(
        real_dist, gen_pool, quantities=("iat", "depth", "vol"),
    )

    metrics = {
        "instrument": instrument,
        "n_rollouts": args.n_rollouts,
        "n_events": args.n_events,
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        "stylized_facts": {
            "kurtosis_event": _mean("kurtosis_event"),
            "kurtosis_10s": _mean("kurtosis_10s"),
            "kurtosis_30s": _mean("kurtosis_30s"),
            "kurtosis_60s": _mean("kurtosis_60s"),
            "kurtosis_120s": _mean("kurtosis_120s"),
            "acf_returns_lag1": _mean_lag("acf_returns", 1),
            "acf_returns_lag5": _mean_lag("acf_returns", 5),
            "acf_abs_returns_lag1": _mean_lag("acf_abs_returns", 1),
            "acf_abs_returns_lag5": _mean_lag("acf_abs_returns", 5),
        },
        "rollout_aggregates": {
            "mean_iat": float(gen_pool["iat"].mean()),
            "mean_vol": float(gen_pool["vol"].mean()),
            "frac_buy": float((gen_pool["side"] == 0).mean()),
            "frac_add": float((gen_pool["action"] == 0).mean()),
            "mean_spread": float(np.nanmean(gen_pool["spread"])),
            "mean_obi": float(np.nanmean(gen_pool["obi"])),
        },
        "distributional_fidelity": fidelity,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    writer = SummaryWriter(str(out_dir))
    for cat, vals in metrics.items():
        if isinstance(vals, dict):
            for k, v in vals.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(f"{cat}/{k}", v, 0)
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, (int, float)):
                            writer.add_scalar(f"{cat}/{k}/{kk}", vv, 0)
    writer.close()

    print(f"\nWrote {out_dir / 'metrics.json'}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
