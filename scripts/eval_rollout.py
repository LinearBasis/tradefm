"""Standalone CLI: load a checkpoint, run closed-loop rollouts, compute stylized facts.

Single-instrument:
    python -m scripts.eval_rollout \
        --checkpoint checkpoints/best.pt \
        --tokenizer data/processed/tokenizer.json \
        --val-sequences data/processed/sequences/ \
        --instrument SBER \
        --n-rollouts 10 --n-events 1024 \
        --output runs/eval/best/

Sweep across every instrument in the manifest (paper §9.1 style):
    python -m scripts.eval_rollout \
        --checkpoint checkpoints/best.pt \
        --tokenizer data/processed/tokenizer.json \
        --val-sequences data/processed/sequences/ \
        --all-instruments \
        --n-rollouts 10 --n-events 1024 \
        --output runs/eval/best/

In sweep mode `metrics.json` has two sections:
  - `per_instrument`: full metrics dict per instrument
  - `aggregated`: mean across instruments + per-liquidity-tier means
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

    iat_lut = np.array([tokenizer.bin_centroid("interarrival", b) for b in range(n_t)])
    vol_lut = np.array([tokenizer.bin_centroid("volume", b) for b in range(n_v)])
    depth_lut = np.array([tokenizer.bin_centroid("price_depth", b) for b in range(n_d)])

    iats, depths, vols = [], [], []
    for pf in val_parquets:
        toks = pl.read_parquet(pf)["trade_token"].to_numpy()
        rem_a = toks % block_a
        rem_s = rem_a % block_s
        rem_d = rem_s % block_d
        i_v = rem_d // n_t
        i_t = rem_d % n_t
        i_d = rem_s // block_d
        iats.append(iat_lut[i_t])
        vols.append(vol_lut[i_v])
        depths.append(depth_lut[i_d])

    return {
        "iat": np.concatenate(iats) if iats else np.array([]),
        "vol": np.concatenate(vols) if vols else np.array([]),
        "depth": np.concatenate(depths) if depths else np.array([]),
    }


def _rollout_for_instrument(
    instrument: str,
    model: OrderFlowTransformer,
    cfg: ModelConfig,
    tokenizer: Tokenizer,
    manifest: dict,
    seq_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict | None:
    """Run N rollouts for one instrument, compute stylized facts + fidelity.

    Returns None if the instrument has no val sequences in the manifest.
    """
    inst_id = manifest["instruments"].index(instrument)
    val_keys = [k for k, v in manifest["sequences"].items()
                if v["split"] == "val" and k.startswith(instrument + "_")]
    if not val_keys:
        return None

    seed_df = pl.read_parquet(seq_dir / f"{val_keys[0]}.parquet")
    ctx = min(cfg.context_length, seed_df.height)
    seed_tokens = seed_df["trade_token"].to_numpy()[:ctx].astype(np.int64).tolist()
    seed_plevels = seed_df["bin_price_level"].to_numpy()[:ctx].astype(np.int64).tolist()
    seed_liq = int(seed_df["bin_liquidity"][0])

    real_dist = _decode_real_distributions(
        [seq_dir / f"{k}.parquet" for k in val_keys], tokenizer, cfg.vocab_size,
    )

    print(f"Instrument: {instrument} (id={inst_id}, liq_bin={seed_liq}) | "
          f"rollouts={args.n_rollouts} × {args.n_events}")
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
        print(f"  rollout {i+1}/{args.n_rollouts}: "
              f"mid range [{gen['mid'].min():.4f}, {gen['mid'].max():.4f}]")
        all_gens.append(gen)

    gen_pool = {k: np.concatenate([g[k] for g in all_gens]) for k in all_gens[0].keys()}
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

    return {
        "instrument": instrument,
        "liquidity_bin": seed_liq,
        "n_rollouts": args.n_rollouts,
        "n_events": args.n_events,
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


def _aggregate_across_instruments(per_inst: dict[str, dict]) -> dict:
    """Mean across instruments per leaf metric, plus per-liquidity-tier means.

    Walks the nested metric dicts; numeric leaves are averaged. Non-numeric
    fields (instrument name, etc.) are dropped from the aggregate.
    """
    if not per_inst:
        return {}

    def collect_numeric(d, prefix=""):
        out = {}
        for k, v in d.items():
            key = f"{prefix}/{k}" if prefix else k
            if isinstance(v, (int, float)):
                out[key] = float(v)
            elif isinstance(v, dict):
                out.update(collect_numeric(v, key))
        return out

    flat_per_inst = {inst: collect_numeric(m) for inst, m in per_inst.items()}

    def _mean_safe(values):
        vals = [v for v in values if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    all_keys = sorted({k for m in flat_per_inst.values() for k in m})
    overall = {k: _mean_safe([m.get(k, float("nan")) for m in flat_per_inst.values()])
               for k in all_keys}

    # Per-liquidity-tier means (paper §9.1 reports per-tier).
    by_tier: dict[int, list[dict]] = {}
    for inst, m in per_inst.items():
        by_tier.setdefault(int(m["liquidity_bin"]), []).append(flat_per_inst[inst])
    per_tier = {}
    for tier, flats in sorted(by_tier.items()):
        per_tier[f"liq_bin_{tier}"] = {
            k: _mean_safe([f.get(k, float("nan")) for f in flats]) for k in all_keys
        }

    return {
        "overall": overall,
        "per_liquidity_tier": per_tier,
        "n_instruments": len(per_inst),
    }


def _write_tb_scalars(writer: SummaryWriter, prefix: str, metrics: dict, step: int = 0) -> None:
    """Recursively write numeric leaves of `metrics` as TB scalars under `prefix`."""
    for k, v in metrics.items():
        tag = f"{prefix}/{k}" if prefix else k
        if isinstance(v, (int, float)):
            writer.add_scalar(tag, v, step)
        elif isinstance(v, dict):
            _write_tb_scalars(writer, tag, v, step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--val-sequences", type=str, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--instrument", type=str, default=None,
                       help="Instrument to roll out for (default: first in manifest)")
    group.add_argument("--all-instruments", action="store_true",
                       help="Iterate over every instrument in the manifest, "
                            "aggregate metrics across them.")
    parser.add_argument("--n-rollouts", type=int, default=5)
    parser.add_argument("--n-events", type=int, default=512)
    parser.add_argument("--init-mid", type=float, default=100.0,
                        help="Used for ALL instruments in --all-instruments mode "
                             "(centroid LUTs are scale-invariant; absolute mid only "
                             "affects spread/depth in absolute price units).")
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

    # Reproducibility consistency checks (apply in both modes).
    if sorted(tokenizer.instruments) != sorted(instruments):
        raise RuntimeError(
            f"Instrument mismatch: tokenizer.instruments ({len(tokenizer.instruments)}) "
            f"!= manifest.instruments ({len(instruments)})."
        )
    if cfg.n_instruments != len(instruments):
        raise RuntimeError(
            f"Model trained on {cfg.n_instruments} instruments, val data has "
            f"{len(instruments)}."
        )

    if args.all_instruments:
        targets = instruments
    else:
        targets = [args.instrument or instruments[0]]

    per_inst: dict[str, dict] = {}
    for inst in targets:
        m = _rollout_for_instrument(inst, model, cfg, tokenizer, manifest, seq_dir, args, device)
        if m is None:
            print(f"  [skip] {inst}: no val sequences in manifest")
            continue
        per_inst[inst] = m

    if not per_inst:
        raise RuntimeError("No instruments produced metrics (no val sequences for any target)")

    # Single-instrument mode: keep flat schema for backward-compat with old eval consumers.
    if not args.all_instruments and len(per_inst) == 1:
        out_metrics = next(iter(per_inst.values()))
    else:
        out_metrics = {
            "per_instrument": per_inst,
            "aggregated": _aggregate_across_instruments(per_inst),
        }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(out_metrics, f, indent=2)

    writer = SummaryWriter(str(out_dir))
    _write_tb_scalars(writer, "", out_metrics)
    writer.close()

    print(f"\nWrote {out_dir / 'metrics.json'}  ({len(per_inst)} instruments)")
    if args.all_instruments:
        agg = out_metrics["aggregated"]["overall"]
        print(f"Aggregated overall (mean across {len(per_inst)} instruments):")
        for k, v in agg.items():
            if "kurtosis" in k or "acf" in k or "fidelity" in k:
                print(f"  {k}: {v:.4f}")
    else:
        print(json.dumps(out_metrics, indent=2))


if __name__ == "__main__":
    main()
