"""Forward-pass reproducibility check across code versions.

Loads a transformer checkpoint, runs a deterministic forward pass on a fixed
synthetic batch, and saves logits + hidden states to a .pt file. Run the same
script in two codebases (tradefm4 vs tradefm5) against the same checkpoint,
then compare the two outputs — they must match (bitwise on CPU, or within fp
tolerance) iff the model is architecturally identical.

Usage:
    # In tradefm4 (old code):
    PYTHONPATH=. uv run python -m scripts.smoke_checkpoint_reproducibility \\
        --checkpoint /path/to/best.pt --out /tmp/out_old.pt

    # In tradefm5 (new code):
    PYTHONPATH=. uv run python -m scripts.smoke_checkpoint_reproducibility \\
        --checkpoint /path/to/best.pt --out /tmp/out_new.pt

    # Compare:
    PYTHONPATH=. uv run python -m scripts.smoke_checkpoint_reproducibility \\
        --compare /tmp/out_old.pt /tmp/out_new.pt

The compare step prints max absolute / relative differences and verdicts at
several tolerance levels (bitwise, atol=1e-6, atol=1e-4). For a correctly
backward-compatible refactor with `qk_norm=False` and `optimizer="adamw"`,
the difference on CPU should be exactly zero.
"""

import argparse
from pathlib import Path

import torch

from src.config import ModelConfig
from src.models.transformer import OrderFlowTransformer


SEED = 1234
BATCH = 2
SEQ_LEN = 64


def deterministic_inputs(cfg: ModelConfig):
    """Build a fixed synthetic input that depends only on cfg dimensions and SEED."""
    g = torch.Generator(device="cpu").manual_seed(SEED)
    tt = torch.randint(0, cfg.vocab_size, (BATCH, SEQ_LEN), generator=g)
    pl = torch.randint(0, cfg.n_price_level_bins, (BATCH, SEQ_LEN), generator=g)
    liq = torch.randint(0, cfg.n_liquidity_bins, (BATCH, SEQ_LEN), generator=g)
    inst = torch.randint(0, cfg.n_instruments, (BATCH,), generator=g)
    return tt, pl, liq, inst


def run_forward(checkpoint: Path, out: Path) -> None:
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg: ModelConfig = ck["config"]

    model = OrderFlowTransformer(cfg).eval()
    missing, unexpected = model.load_state_dict(ck["model_state_dict"], strict=False)
    if missing:
        print(f"  WARN: missing keys in checkpoint (fresh-init in this codebase): {missing}")
    if unexpected:
        print(f"  WARN: unexpected keys in checkpoint: {unexpected}")

    tt, pl, liq, inst = deterministic_inputs(cfg)

    torch.manual_seed(SEED)
    with torch.no_grad():
        logits = model(tt, pl, liq, inst)
        hidden = model.extract_hidden_states(tt, pl, liq, inst)

    output = {
        "config_summary": {
            "d_model": cfg.d_model,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "n_kv_heads": getattr(cfg, "n_kv_heads", None),
            "context_length": cfg.context_length,
            "d_ff": cfg.d_ff,
            "qk_norm": getattr(cfg, "qk_norm", False),
            "vocab_size": cfg.vocab_size,
        },
        "logits": logits,
        "hidden": hidden,
        "inputs": {"tt": tt, "pl": pl, "liq": liq, "inst": inst},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, out)

    print(f"  Saved: {out}")
    print(f"  Config: {output['config_summary']}")
    print(f"  logits  shape={tuple(logits.shape)}  mean={logits.mean().item():+.6e}  "
          f"std={logits.std().item():.6e}  first={logits.flatten()[0].item():+.6e}")
    print(f"  hidden  shape={tuple(hidden.shape)}  mean={hidden.mean().item():+.6e}  "
          f"std={hidden.std().item():.6e}  first={hidden.flatten()[0].item():+.6e}")


def compare(old: Path, new: Path) -> int:
    """Return 0 on pass, 1 on failure."""
    o = torch.load(old, map_location="cpu", weights_only=False)
    n = torch.load(new, map_location="cpu", weights_only=False)

    print(f"OLD config: {o['config_summary']}")
    print(f"NEW config: {n['config_summary']}")

    # Inputs must match — same SEED + same cfg dims → identical tensors
    for k in ("tt", "pl", "liq", "inst"):
        if not torch.equal(o["inputs"][k], n["inputs"][k]):
            print(f"  FAIL: input '{k}' differs (different SEED or cfg?)")
            return 1
    print("Inputs identical ✓")

    failed = False
    for name in ("logits", "hidden"):
        a = o[name].float()
        b = n[name].float()
        if a.shape != b.shape:
            print(f"  FAIL: {name} shape differs: {tuple(a.shape)} vs {tuple(b.shape)}")
            failed = True
            continue

        diff = (a - b).abs()
        max_diff = diff.max().item()
        max_abs = a.abs().max().item()
        rel = max_diff / (max_abs + 1e-12)

        bitwise = torch.equal(a, b)
        within_1e9 = torch.allclose(a, b, atol=1e-9, rtol=0)
        within_1e6 = torch.allclose(a, b, atol=1e-6, rtol=1e-5)
        within_1e4 = torch.allclose(a, b, atol=1e-4, rtol=1e-3)

        print(f"\n{name}:")
        print(f"  max |diff|        = {max_diff:.3e}")
        print(f"  max |old|         = {max_abs:.3e}")
        print(f"  relative          = {rel:.3e}")
        print(f"  bitwise equal     : {bitwise}")
        print(f"  ≤ 1e-9            : {within_1e9}")
        print(f"  ≤ 1e-6            : {within_1e6}")
        print(f"  ≤ 1e-4            : {within_1e4}")

        if not within_1e4:
            print(f"  FAIL: {name} drift exceeds 1e-4 — architecture / weights NOT equivalent")
            failed = True

    print()
    if failed:
        print("VERDICT: FAIL — old and new codebases produce different outputs.")
        return 1
    print("VERDICT: PASS — outputs match within float tolerance.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, help="transformer best.pt to load")
    p.add_argument("--out", type=Path, help="where to save this codebase's forward outputs")
    p.add_argument("--compare", nargs=2, metavar=("OLD_PT", "NEW_PT"),
                   help="compare two .pt outputs produced by --out runs")
    args = p.parse_args()

    if args.compare:
        return compare(Path(args.compare[0]), Path(args.compare[1]))
    if args.checkpoint and args.out:
        run_forward(args.checkpoint, args.out)
        return 0
    p.error("Provide either (--checkpoint AND --out) or --compare OLD NEW")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
