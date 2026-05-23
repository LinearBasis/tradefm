"""Training script for decision heads over a frozen transformer backbone
(single- or multi-GPU via DDP).

All experiment hyperparameters (τ, loss weights, optimizer, transformer-checkpoint
path, output dirs) live in a JSON config in `configs/`. CLI carries only runtime
concerns: device, AMP, workers, smoke `--max-steps`, resume, TensorBoard run name.

Usage examples:
    # Smoke on laptop
    python -m scripts.train_heads --config configs/smoke_heads.json \\
        --device cpu --allow-cpu --amp none --max-steps 20

    # 4×H100 cluster
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \\
        -m scripts.train_heads --config configs/cluster_heads.json \\
        --amp bf16 --num-workers 8

GPU SELECTION
-------------
`CUDA_VISIBLE_DEVICES` chooses physical GPUs; `--nproc-per-node=N` must equal
that GPU count.
"""

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import HeadConfig, ModelConfig
from src.data.dataset import OrderFlowDataset
from src._archive.decision.heads import DecisionModule
from src.models.transformer import OrderFlowTransformer


# --------------------------------------------------------------------------- #
# Distributed helpers (mirror scripts/train_transformer.py)                    #
# --------------------------------------------------------------------------- #

_BANNER = (
    "\n"
    "================================================================\n"
    "  WARNING: CUDA IS NOT AVAILABLE — falling back to {device}.\n"
    "  This is fine for a smoke test on a laptop, but on the cluster\n"
    "  it almost certainly means a misconfiguration:\n"
    "    - check `nvidia-smi` (driver / GPU visibility)\n"
    "    - check CUDA_VISIBLE_DEVICES is set correctly\n"
    "    - check torch was installed with CUDA support\n"
    "      (python -c 'import torch; print(torch.version.cuda)')\n"
    "  Pass --allow-cpu to silence this warning.\n"
    "================================================================\n"
)


def _ddp_active() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _ddp_setup() -> tuple[int, int, int, torch.device]:
    if _ddp_active():
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        world_size = int(os.environ["WORLD_SIZE"])
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size, torch.device(f"cuda:{local_rank}")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return 0, 0, 1, device


def _ddp_cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def _is_main(rank: int) -> bool:
    return rank == 0


def _all_reduce_mean(value: float, world_size: int, device: torch.device) -> float:
    if world_size == 1:
        return value
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / world_size)


def _gather_arrays(arr: np.ndarray, world_size: int, device: torch.device) -> np.ndarray:
    """Gather variable-length 1D float arrays from all ranks onto rank 0.
    Returns concatenated array on rank 0, empty on other ranks.
    """
    if world_size == 1:
        return arr

    t = torch.from_numpy(arr.astype(np.float32)).to(device)
    sizes = [torch.zeros(1, dtype=torch.int64, device=device) for _ in range(world_size)]
    dist.all_gather(sizes, torch.tensor([t.numel()], dtype=torch.int64, device=device))
    max_n = int(max(s.item() for s in sizes))

    padded = torch.zeros(max_n, dtype=torch.float32, device=device)
    padded[: t.numel()] = t
    bufs = [torch.zeros(max_n, dtype=torch.float32, device=device) for _ in range(world_size)]
    dist.all_gather(bufs, padded)

    if not _is_main(0):  # all ranks compute, rank 0 returns
        pass
    parts = []
    for i, b in enumerate(bufs):
        n = int(sizes[i].item())
        parts.append(b[:n].cpu().numpy())
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Training utilities                                                           #
# --------------------------------------------------------------------------- #

def get_lr(step: int, warmup_steps: int, total_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def _amp_dtype(name: str | None) -> torch.dtype | None:
    if name is None or name == "none":
        return None
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"Unknown --amp value: {name!r}")


def _rankdata(arr: np.ndarray) -> np.ndarray:
    order = arr.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(arr) + 1, dtype=np.float64)
    return ranks


def _spearman_corr(pred: np.ndarray, target: np.ndarray) -> float:
    if len(pred) < 10:
        return 0.0
    return float(np.corrcoef(_rankdata(pred), _rankdata(target))[0, 1])


def _pearson_corr(pred: np.ndarray, target: np.ndarray) -> float:
    if len(pred) < 10:
        return 0.0
    if pred.std() < 1e-12 or target.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(pred, target)[0, 1])


def load_transformer(
    cfg: HeadConfig, model_cfg: ModelConfig, device: torch.device, rank: int,
) -> OrderFlowTransformer:
    ckpt_path = Path(cfg.transformer_checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Transformer checkpoint not found: {ckpt_path}")

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_cfg = ck.get("config", model_cfg)
    model = OrderFlowTransformer(saved_cfg).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    if _is_main(rank):
        n = sum(p.numel() for p in model.parameters())
        print(f"Transformer: {n:,} params (frozen), loaded from epoch {ck.get('epoch', '?')}")
    return model, saved_cfg


def train_epoch(
    transformer: nn.Module,
    heads: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: HeadConfig,
    device: torch.device,
    step: int,
    total_steps: int,
    warmup_steps: int,
    epoch: int,
    amp_dtype: torch.dtype | None,
    scaler: "torch.amp.GradScaler | None",
    rank: int,
    world_size: int,
    writer: SummaryWriter | None,
    max_steps: int | None = None,
) -> tuple[dict[str, float], int]:
    heads.train()
    sums = {"alpha": 0.0, "risk": 0.0, "intensity": 0.0, "total": 0.0}
    n_batches = 0

    use_amp = amp_dtype is not None and device.type == "cuda"

    iterable = loader
    if _is_main(rank):
        iterable = tqdm(
            loader,
            desc=f"Epoch {epoch}/{cfg.max_epochs} [train]",
            leave=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )

    # Underlying head module (unwrap DDP for compute_loss access)
    heads_core = heads.module if isinstance(heads, DDP) else heads

    for batch in iterable:
        tokens = batch["trade_tokens"][:, :-1].to(device, non_blocking=True)
        plevels = batch["price_levels"][:, :-1].to(device, non_blocking=True)
        liqs = batch["liquidities"][:, :-1].to(device, non_blocking=True)
        inst_ids = batch["instrument_id"].to(device, non_blocking=True)
        mask = batch["target_mask"][:, :-1].to(device, non_blocking=True)

        targets = {
            "alpha": batch["target_alpha"][:, :-1].to(device, non_blocking=True),
            "risk": batch["target_risk"][:, :-1].to(device, non_blocking=True),
            "intensity": batch["target_intensity"][:, :-1].to(device, non_blocking=True),
        }

        lr = get_lr(step, warmup_steps, total_steps, cfg.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        # Frozen transformer → hidden states (no_grad even under autocast)
        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    hidden = transformer.extract_hidden_states(
                        tokens, plevels, liqs, inst_ids, layer=cfg.latent_layer,
                    )
            else:
                hidden = transformer.extract_hidden_states(
                    tokens, plevels, liqs, inst_ids, layer=cfg.latent_layer,
                )

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                preds = heads(hidden)
                losses = heads_core.compute_loss(
                    preds, targets, mask,
                    huber_delta=cfg.huber_delta,
                    w_alpha=cfg.w_alpha, w_risk=cfg.w_risk, w_intensity=cfg.w_intensity,
                )
            if scaler is not None:
                scaler.scale(losses["total"]).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                losses["total"].backward()
                optimizer.step()
        else:
            preds = heads(hidden)
            losses = heads_core.compute_loss(
                preds, targets, mask,
                huber_delta=cfg.huber_delta,
                w_alpha=cfg.w_alpha, w_risk=cfg.w_risk, w_intensity=cfg.w_intensity,
            )
            losses["total"].backward()
            optimizer.step()

        for k in sums:
            sums[k] += losses[k].item()
        n_batches += 1
        step += 1

        if _is_main(rank) and hasattr(iterable, "set_postfix"):
            iterable.set_postfix(
                loss=f"{losses['total'].item():.4f}",
                avg=f"{sums['total']/n_batches:.4f}",
                a=f"{sums['alpha']/n_batches:.2e}",
                s=f"{sums['risk']/n_batches:.4f}",
                k=f"{sums['intensity']/n_batches:.4f}",
                lr=f"{lr:.1e}",
            )
            if writer is not None:
                for kk in ("total", "alpha", "risk", "intensity"):
                    writer.add_scalar(f"train/loss_{kk}", losses[kk].item(), step)
                writer.add_scalar("train/lr", lr, step)

        if max_steps is not None and step >= max_steps:
            break

    avgs = {k: _all_reduce_mean(v / max(n_batches, 1), world_size, device) for k, v in sums.items()}
    return avgs, step


@torch.no_grad()
def evaluate(
    transformer: nn.Module,
    heads: nn.Module,
    loader: DataLoader,
    cfg: HeadConfig,
    device: torch.device,
    epoch: int,
    amp_dtype: torch.dtype | None,
    rank: int,
    world_size: int,
) -> tuple[dict[str, float], dict[str, float]]:
    heads.eval()
    sums = {"alpha": 0.0, "risk": 0.0, "intensity": 0.0, "total": 0.0}
    n_batches = 0

    use_amp = amp_dtype is not None and device.type == "cuda"

    all_preds = {"alpha": [], "risk": [], "intensity": []}
    all_targets = {"alpha": [], "risk": [], "intensity": []}
    max_samples = 500_000
    n_collected = 0

    heads_core = heads.module if isinstance(heads, DDP) else heads

    iterable = loader
    if _is_main(rank):
        iterable = tqdm(
            loader,
            desc=f"Epoch {epoch}/{cfg.max_epochs} [val]  ",
            leave=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
        )

    for batch in iterable:
        tokens = batch["trade_tokens"][:, :-1].to(device, non_blocking=True)
        plevels = batch["price_levels"][:, :-1].to(device, non_blocking=True)
        liqs = batch["liquidities"][:, :-1].to(device, non_blocking=True)
        inst_ids = batch["instrument_id"].to(device, non_blocking=True)
        mask = batch["target_mask"][:, :-1].to(device, non_blocking=True)

        targets = {
            "alpha": batch["target_alpha"][:, :-1].to(device, non_blocking=True),
            "risk": batch["target_risk"][:, :-1].to(device, non_blocking=True),
            "intensity": batch["target_intensity"][:, :-1].to(device, non_blocking=True),
        }

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                hidden = transformer.extract_hidden_states(
                    tokens, plevels, liqs, inst_ids, layer=cfg.latent_layer,
                )
                preds = heads(hidden)
                losses = heads_core.compute_loss(
                    preds, targets, mask,
                    huber_delta=cfg.huber_delta,
                    w_alpha=cfg.w_alpha, w_risk=cfg.w_risk, w_intensity=cfg.w_intensity,
                )
        else:
            hidden = transformer.extract_hidden_states(
                tokens, plevels, liqs, inst_ids, layer=cfg.latent_layer,
            )
            preds = heads(hidden)
            losses = heads_core.compute_loss(
                preds, targets, mask,
                huber_delta=cfg.huber_delta,
                w_alpha=cfg.w_alpha, w_risk=cfg.w_risk, w_intensity=cfg.w_intensity,
            )

        for k in sums:
            sums[k] += losses[k].item()
        n_batches += 1

        if n_collected < max_samples:
            m = mask.cpu()
            for key in all_preds:
                p = preds[key].float().cpu()[m].numpy()
                t = targets[key].cpu()[m].numpy()
                all_preds[key].append(p)
                all_targets[key].append(t)
            n_collected += int(m.sum().item())

        if _is_main(rank) and hasattr(iterable, "set_postfix"):
            iterable.set_postfix(loss=f"{sums['total']/n_batches:.4f}")

    avg_losses = {k: _all_reduce_mean(v / max(n_batches, 1), world_size, device) for k, v in sums.items()}

    # Concat per-rank predictions, gather to rank-0 for correlation metrics
    cat = lambda buf: np.concatenate(buf) if buf else np.zeros(0, dtype=np.float32)
    ap = _gather_arrays(cat(all_preds["alpha"]), world_size, device)
    at = _gather_arrays(cat(all_targets["alpha"]), world_size, device)
    rp = _gather_arrays(cat(all_preds["risk"]), world_size, device)
    rt = _gather_arrays(cat(all_targets["risk"]), world_size, device)
    ip = _gather_arrays(cat(all_preds["intensity"]), world_size, device)
    it = _gather_arrays(cat(all_targets["intensity"]), world_size, device)

    metrics = {}
    if _is_main(rank):
        eps = 1e-8
        metrics = {
            "ic_alpha": _pearson_corr(ap, at),
            "spearman_risk": _spearman_corr(np.log(rp + eps), np.log(rt + eps)),
            "spearman_intensity": _spearman_corr(np.log(ip + eps), np.log(it + eps)),
            "alpha_pred_std": float(ap.std()) if ap.size else 0.0,
            "alpha_pred_mean": float(ap.mean()) if ap.size else 0.0,
        }

    return avg_losses, metrics


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Train decision heads (DDP-aware)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to JSON config (overrides HeadConfig defaults)")
    # Runtime-only flags (not part of the experiment config)
    parser.add_argument("--device", type=str, default=None,
                        help="Override device for non-DDP mode (cpu/cuda/mps)")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", type=str, default="bf16",
                        choices=["bf16", "fp16", "none"])
    parser.add_argument("--allow-cpu", action="store_true",
                        help="Suppress the CUDA-not-available warning")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Stop after this many optimizer steps (smoke-test utility)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a heads checkpoint (.pt)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="TensorBoard run name (default: timestamp)")
    # Smoke / experiment overrides for cfg fields
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Override cfg.max_epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override cfg.batch_size")
    parser.add_argument("--sequences-dir", type=str, default=None,
                        help="Override cfg.sequences_dir")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Override cfg.checkpoint_dir")
    parser.add_argument("--transformer-checkpoint", type=str, default=None,
                        help="Override cfg.transformer_checkpoint")
    args = parser.parse_args()

    rank, local_rank, world_size, device = _ddp_setup()
    if not _ddp_active() and args.device:
        device = torch.device(args.device)

    if device.type != "cuda" and rank == 0 and not args.allow_cpu:
        import sys
        print(_BANNER.format(device=device.type), file=sys.stderr, flush=True)

    from src.config import load_from_json
    cfg = load_from_json(HeadConfig, args.config)
    # Apply CLI overrides
    if args.max_epochs is not None: cfg.max_epochs = args.max_epochs
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.sequences_dir is not None: cfg.sequences_dir = args.sequences_dir
    if args.checkpoint_dir is not None: cfg.checkpoint_dir = args.checkpoint_dir
    if args.transformer_checkpoint is not None:
        cfg.transformer_checkpoint = args.transformer_checkpoint
    model_cfg = ModelConfig()
    model_cfg.sequences_dir = cfg.sequences_dir
    if _is_main(rank):
        print(f"Loaded config: {args.config}")

    amp_dtype = _amp_dtype(args.amp)
    if device.type != "cuda":
        amp_dtype = None

    if _is_main(rank):
        print(f"World size: {world_size} | Device: {device} | AMP: {args.amp if amp_dtype else 'off'}")

    # Frozen transformer (loaded on every rank — needed for inference)
    transformer, saved_cfg = load_transformer(cfg, model_cfg, device, rank)
    # Heads need d_model matching the saved transformer
    cfg.d_model = saved_cfg.d_model
    # Also propagate context_length to dataset window length
    model_cfg.context_length = saved_cfg.context_length
    model_cfg.stride = saved_cfg.stride
    model_cfg.batch_size = cfg.batch_size

    if _is_main(rank):
        print("Loading datasets...")
    train_ds = OrderFlowDataset(model_cfg, split="train", include_targets=True)
    val_ds = OrderFlowDataset(model_cfg, split="val", include_targets=True)
    if train_ds.manifest_tau_sec is not None and train_ds.manifest_tau_sec != cfg.tau_sec:
        raise ValueError(
            f"tau_sec mismatch: heads config wants {cfg.tau_sec}, "
            f"manifest was generated with {train_ds.manifest_tau_sec}. "
            f"Regenerate sequences or align cfg.tau_sec."
        )
    if _is_main(rank):
        print(f"Train windows: {len(train_ds):,}, Val windows: {len(val_ds):,}")

    use_ddp = world_size > 1
    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True) if use_ddp else None
    val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False) if use_ddp else None

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=pin,
        persistent_workers=args.num_workers > 0,
        drop_last=use_ddp,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers, pin_memory=pin,
        persistent_workers=args.num_workers > 0,
    )

    heads = DecisionModule(cfg.d_model).to(device)
    if _is_main(rank):
        n = sum(p.numel() for p in heads.parameters() if p.requires_grad)
        print(f"Decision heads: {n:,} trainable params")
    if use_ddp:
        heads = DDP(heads, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    optimizer = torch.optim.AdamW(
        heads.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda") if (amp_dtype == torch.float16) else None

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.max_epochs
    warmup_steps = int(total_steps * cfg.warmup_fraction)
    if _is_main(rank):
        print(f"Steps/epoch: {steps_per_epoch}, Total: {total_steps}, Warmup: {warmup_steps}")

    ckpt_dir = Path(cfg.checkpoint_dir)
    writer = None
    if _is_main(rank):
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        run_name = args.run_name or f"heads_{time.strftime('%Y%m%d-%H%M%S')}"
        log_dir = Path("runs") / run_name
        writer = SummaryWriter(log_dir=str(log_dir))
        print(f"TensorBoard: {log_dir}\n")

    best_val_loss = float("inf")
    patience_counter = 0
    step = 0
    start_epoch = 1

    if args.resume:
        rp = Path(args.resume)
        if not rp.exists():
            raise FileNotFoundError(f"--resume: {rp} not found")
        ck = torch.load(rp, map_location=device, weights_only=False)
        target = heads.module if use_ddp else heads
        target.load_state_dict(ck["heads_state_dict"])
        if "optimizer_state_dict" in ck:
            optimizer.load_state_dict(ck["optimizer_state_dict"])
        start_epoch = int(ck.get("epoch", 0)) + 1
        best_val_loss = float(ck.get("val_losses", {}).get("total", best_val_loss))
        step = int(ck.get("step", 0))
        if _is_main(rank):
            print(f"Resumed from {rp} (epoch {start_epoch}, best val {best_val_loss:.4f})")

    for epoch in range(start_epoch, cfg.max_epochs + 1):
        if use_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        t0 = time.time()
        train_losses, step = train_epoch(
            transformer, heads, train_loader, optimizer, cfg, device,
            step, total_steps, warmup_steps, epoch,
            amp_dtype, scaler, rank, world_size, writer,
            max_steps=args.max_steps,
        )
        val_losses, val_metrics = evaluate(
            transformer, heads, val_loader, cfg, device, epoch,
            amp_dtype, rank, world_size,
        )
        elapsed = time.time() - t0

        if _is_main(rank):
            marker = "*" if val_losses["total"] < best_val_loss else " "
            print(
                f"  Epoch {epoch:2d}/{cfg.max_epochs} | "
                f"train {train_losses['total']:.4f} "
                f"(α={train_losses['alpha']:.2e} σ={train_losses['risk']:.4f} κ={train_losses['intensity']:.4f}) | "
                f"val {val_losses['total']:.4f} "
                f"(α={val_losses['alpha']:.2e} σ={val_losses['risk']:.4f} κ={val_losses['intensity']:.4f}) | "
                f"{elapsed:.0f}s {marker}"
            )
            print(
                f"           "
                f"IC(α)={val_metrics['ic_alpha']:+.4f}  "
                f"Spearman(σ)={val_metrics['spearman_risk']:+.4f}  "
                f"Spearman(κ)={val_metrics['spearman_intensity']:+.4f}  "
                f"α_std={val_metrics['alpha_pred_std']:.2e}"
            )

            if writer is not None:
                for split, losses in [("train", train_losses), ("val", val_losses)]:
                    for k, v in losses.items():
                        writer.add_scalar(f"{split}/loss_{k}_epoch", v, epoch)
                for k, v in val_metrics.items():
                    writer.add_scalar(f"val/metrics/{k}", v, epoch)

            sd = (heads.module if use_ddp else heads).state_dict()
            torch.save({
                "epoch": epoch,
                "step": step,
                "heads_state_dict": sd,
                "optimizer_state_dict": optimizer.state_dict(),
                "val_losses": val_losses,
                "val_metrics": val_metrics,
                "config": cfg,
            }, ckpt_dir / "last.pt")

            if val_losses["total"] < best_val_loss:
                best_val_loss = val_losses["total"]
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "step": step,
                    "heads_state_dict": sd,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_losses": val_losses,
                    "val_metrics": val_metrics,
                    "config": cfg,
                }, ckpt_dir / "best_heads.pt")
            else:
                patience_counter += 1

        if args.max_steps is not None and step >= args.max_steps:
            if _is_main(rank):
                print(f"\nReached --max-steps={args.max_steps}; stopping smoke run.")
            break

        if use_ddp:
            stop_signal = torch.tensor(
                [1 if patience_counter >= cfg.patience else 0],
                dtype=torch.int64, device=device,
            )
            dist.broadcast(stop_signal, src=0)
            if stop_signal.item() == 1:
                if _is_main(rank):
                    print(f"\nEarly stopping at epoch {epoch} (patience={cfg.patience})")
                break
        else:
            if patience_counter >= cfg.patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={cfg.patience})")
                break

    if _is_main(rank):
        if writer is not None:
            writer.close()
        print(f"\nBest val loss: {best_val_loss:.4f}")
        print(f"Checkpoint: {ckpt_dir / 'best_heads.pt'}")

    _ddp_cleanup()


if __name__ == "__main__":
    main()
