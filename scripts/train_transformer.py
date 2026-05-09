"""Training script for OrderFlowTransformer (single- or multi-GPU via DDP).

All experiment hyperparameters (architecture, optimizer, dataset stride/length,
checkpoint dir) are read from a JSON config in `configs/`. The CLI carries only
runtime concerns: device, AMP, DataLoader workers, smoke `--max-steps`, resume,
TensorBoard run name.

Usage examples:
    # Smoke on laptop
    python -m scripts.train_transformer --config configs/smoke.json \\
        --device cpu --allow-cpu --amp none --max-steps 30

    # 4×H100 on the cluster
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \\
        -m scripts.train_transformer --config configs/cluster.json \\
        --amp bf16 --num-workers 8

    # Resume
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \\
        -m scripts.train_transformer --config configs/cluster.json \\
        --resume checkpoints/last.pt

GPU SELECTION
-------------
`CUDA_VISIBLE_DEVICES` chooses physical GPUs; torchrun sees only those, renumbered
to 0..N-1 inside the process. `--nproc-per-node=N` MUST equal that GPU count.
"""

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import ModelConfig
from src.data.dataset import OrderFlowDataset
from src.models.transformer import OrderFlowTransformer


# --------------------------------------------------------------------------- #
# Distributed helpers                                                          #
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
    """Initialise NCCL process group when launched via torchrun.

    Returns (rank, local_rank, world_size, device).
    Always safe to call: in single-process mode it returns (0, 0, 1, cuda/cpu/mps).
    """
    if _ddp_active():
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        world_size = int(os.environ["WORLD_SIZE"])
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return rank, local_rank, world_size, device
    # Non-DDP fallback
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
    """Reduce a scalar across ranks, returning mean."""
    if world_size == 1:
        return value
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / world_size)


# --------------------------------------------------------------------------- #
# Training utilities                                                           #
# --------------------------------------------------------------------------- #

def get_lr(step: int, warmup_steps: int, total_steps: int, base_lr: float) -> float:
    """Cosine annealing with linear warmup."""
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
    raise ValueError(f"Unknown --amp value: {name!r} (expected bf16/fp16/none)")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: ModelConfig,
    device: torch.device,
    step: int,
    total_steps: int,
    warmup_steps: int,
    epoch: int,
    max_epochs: int,
    amp_dtype: torch.dtype | None,
    scaler: "torch.amp.GradScaler | None",
    rank: int,
    world_size: int,
    writer: SummaryWriter | None,
    max_steps: int | None = None,
) -> tuple[float, int]:
    """Train for one epoch. Returns (avg_loss_across_ranks, updated_step)."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    use_amp = amp_dtype is not None and device.type == "cuda"

    iterable = loader
    if _is_main(rank):
        iterable = tqdm(
            loader,
            desc=f"Epoch {epoch}/{max_epochs} [train]",
            leave=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )

    for batch in iterable:
        tokens = batch["trade_tokens"].to(device, non_blocking=True)
        plevels = batch["price_levels"].to(device, non_blocking=True)
        liqs = batch["liquidities"].to(device, non_blocking=True)
        inst_ids = batch["instrument_id"].to(device, non_blocking=True)

        input_tokens = tokens[:, :-1]
        target_tokens = tokens[:, 1:]
        input_plevels = plevels[:, :-1]
        input_liqs = liqs[:, :-1]

        lr = get_lr(step, warmup_steps, total_steps, cfg.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(input_tokens, input_plevels, input_liqs, inst_ids)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, cfg.vocab_size), target_tokens.reshape(-1),
                )
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()
        else:
            logits = model(input_tokens, input_plevels, input_liqs, inst_ids)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, cfg.vocab_size), target_tokens.reshape(-1),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        step += 1

        if _is_main(rank) and hasattr(iterable, "set_postfix"):
            avg_loss = total_loss / n_batches
            iterable.set_postfix(
                loss=f"{loss.item():.4f}",
                avg=f"{avg_loss:.4f}",
                ppl=f"{math.exp(min(avg_loss, 20)):.0f}",
                lr=f"{lr:.1e}",
            )
            if writer is not None:
                writer.add_scalar("train/loss_step", loss.item(), step)
                writer.add_scalar("train/lr", lr, step)

        if max_steps is not None and step >= max_steps:
            break

    local_avg = total_loss / max(n_batches, 1)
    return _all_reduce_mean(local_avg, world_size, device), step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    cfg: ModelConfig,
    device: torch.device,
    epoch: int,
    max_epochs: int,
    amp_dtype: torch.dtype | None,
    rank: int,
    world_size: int,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    use_amp = amp_dtype is not None and device.type == "cuda"

    iterable = loader
    if _is_main(rank):
        iterable = tqdm(
            loader,
            desc=f"Epoch {epoch}/{max_epochs} [val]  ",
            leave=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
        )

    for batch in iterable:
        tokens = batch["trade_tokens"].to(device, non_blocking=True)
        plevels = batch["price_levels"].to(device, non_blocking=True)
        liqs = batch["liquidities"].to(device, non_blocking=True)
        inst_ids = batch["instrument_id"].to(device, non_blocking=True)

        input_tokens = tokens[:, :-1]
        target_tokens = tokens[:, 1:]
        input_plevels = plevels[:, :-1]
        input_liqs = liqs[:, :-1]

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(input_tokens, input_plevels, input_liqs, inst_ids)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, cfg.vocab_size), target_tokens.reshape(-1),
                )
        else:
            logits = model(input_tokens, input_plevels, input_liqs, inst_ids)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, cfg.vocab_size), target_tokens.reshape(-1),
            )
        total_loss += loss.item()
        n_batches += 1

        if _is_main(rank) and hasattr(iterable, "set_postfix"):
            avg = total_loss / n_batches
            iterable.set_postfix(loss=f"{avg:.4f}", ppl=f"{math.exp(min(avg, 20)):.0f}")

    local_avg = total_loss / max(n_batches, 1)
    return _all_reduce_mean(local_avg, world_size, device)


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Train OrderFlowTransformer (DDP-aware)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to JSON config (overrides ModelConfig defaults)")
    # Runtime-only flags (not part of the experiment config)
    parser.add_argument("--device", type=str, default=None,
                        help="Override device for non-DDP mode (cpu/cuda/mps)")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", type=str, default="bf16",
                        choices=["bf16", "fp16", "none"],
                        help="Mixed precision on CUDA (default bf16; ignored on cpu/mps)")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="Suppress the CUDA-not-available warning")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Stop after this many optimizer steps (smoke-test utility)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a checkpoint (.pt)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="TensorBoard run name (default: timestamp)")
    args = parser.parse_args()

    # --- DDP / device -------------------------------------------------------
    rank, local_rank, world_size, device = _ddp_setup()

    # Allow non-DDP user to force device choice
    if not _ddp_active() and args.device:
        device = torch.device(args.device)

    # Loud warning if we're not on CUDA — a footgun on the cluster.
    if device.type != "cuda" and _is_main(rank) and not args.allow_cpu:
        import sys
        print(_BANNER.format(device=device.type), file=sys.stderr, flush=True)

    from src.config import load_from_json
    cfg = load_from_json(ModelConfig, args.config)
    if _is_main(rank):
        print(f"Loaded config: {args.config}")

    amp_dtype = _amp_dtype(args.amp)
    if device.type != "cuda":
        amp_dtype = None

    if _is_main(rank):
        print(f"World size: {world_size} | Device: {device} | AMP: {args.amp if amp_dtype else 'off'}")

    # --- Data ---------------------------------------------------------------
    if _is_main(rank):
        print("Loading datasets...")
    train_ds = OrderFlowDataset(cfg, split="train")
    val_ds = OrderFlowDataset(cfg, split="val")
    if _is_main(rank):
        print(f"Train windows: {len(train_ds):,}, Val windows: {len(val_ds):,}")
        print(f"Instruments: {len(train_ds.instrument_names)} {train_ds.instrument_names}")

    use_ddp = world_size > 1
    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True) if use_ddp else None
    val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False) if use_ddp else None

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=pin,
        persistent_workers=args.num_workers > 0,
        drop_last=use_ddp,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers, pin_memory=pin,
        persistent_workers=args.num_workers > 0,
    )

    # --- Model --------------------------------------------------------------
    cfg.n_instruments = len(train_ds.instrument_names)
    model = OrderFlowTransformer(cfg).to(device)
    if _is_main(rank):
        print(f"Parameters: {model.count_parameters():,} total, "
              f"{model.count_trainable_parameters():,} trainable")

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr,
        weight_decay=cfg.weight_decay, betas=cfg.betas,
    )

    scaler = torch.amp.GradScaler("cuda") if (amp_dtype == torch.float16) else None

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.max_epochs
    warmup_steps = int(total_steps * cfg.warmup_fraction)
    if _is_main(rank):
        print(f"Steps/epoch: {steps_per_epoch}, Total: {total_steps}, Warmup: {warmup_steps}")

    # --- Logging / checkpoint dirs (rank 0 only) ----------------------------
    ckpt_dir = Path(cfg.checkpoint_dir)
    writer = None
    if _is_main(rank):
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
        log_dir = Path("runs") / run_name
        writer = SummaryWriter(log_dir=str(log_dir))
        print(f"TensorBoard: {log_dir}")

    best_val_loss = float("inf")
    patience_counter = 0
    step = 0
    start_epoch = 1

    # --- Resume -------------------------------------------------------------
    if args.resume:
        rp = Path(args.resume)
        if not rp.exists():
            raise FileNotFoundError(f"--resume: {rp} not found")
        ck = torch.load(rp, map_location=device, weights_only=False)
        # DDP: state_dict keys may have "module." prefix
        sd = ck["model_state_dict"]
        target = model.module if use_ddp else model
        target.load_state_dict(sd)
        if "optimizer_state_dict" in ck:
            optimizer.load_state_dict(ck["optimizer_state_dict"])
        start_epoch = int(ck.get("epoch", 0)) + 1
        best_val_loss = float(ck.get("val_loss", best_val_loss))
        step = int(ck.get("step", 0))
        if _is_main(rank):
            print(f"Resumed from {rp} (epoch {start_epoch}, best val {best_val_loss:.4f})")

    # --- Training loop ------------------------------------------------------
    if _is_main(rank):
        print()
    for epoch in range(start_epoch, cfg.max_epochs + 1):
        if use_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        t0 = time.time()
        train_loss, step = train_epoch(
            model, train_loader, optimizer, cfg, device,
            step, total_steps, warmup_steps, epoch, cfg.max_epochs,
            amp_dtype, scaler, rank, world_size, writer,
            max_steps=args.max_steps,
        )
        val_loss = evaluate(
            model, val_loader, cfg, device, epoch, cfg.max_epochs,
            amp_dtype, rank, world_size,
        )
        elapsed = time.time() - t0

        if _is_main(rank):
            val_ppl = math.exp(min(val_loss, 20))
            train_ppl = math.exp(min(train_loss, 20))
            marker = "*" if val_loss < best_val_loss else " "
            print(
                f"  Epoch {epoch:2d}/{cfg.max_epochs} | "
                f"train {train_loss:.4f} (ppl {train_ppl:.0f}) | "
                f"val {val_loss:.4f} (ppl {val_ppl:.0f}) | "
                f"{elapsed:.0f}s {marker}"
            )
            if writer is not None:
                writer.add_scalars("loss/epoch", {"train": train_loss, "val": val_loss}, epoch)
                writer.add_scalars("perplexity/epoch", {"train": train_ppl, "val": val_ppl}, epoch)
                writer.add_scalar("lr/epoch", get_lr(step, warmup_steps, total_steps, cfg.lr), epoch)

            # Always save "last", and "best" when improved
            sd = (model.module if use_ddp else model).state_dict()
            torch.save({
                "epoch": epoch,
                "step": step,
                "model_state_dict": sd,
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": cfg,
            }, ckpt_dir / "last.pt")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "step": step,
                    "model_state_dict": sd,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": cfg,
                }, ckpt_dir / "best.pt")
            else:
                patience_counter += 1

        # Sync patience across ranks (so all ranks decide to stop together)
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
        print(f"\nBest val loss: {best_val_loss:.4f} (PPL: {math.exp(min(best_val_loss, 20)):.0f})")
        print(f"Checkpoint: {ckpt_dir / 'best.pt'}")

    _ddp_cleanup()


if __name__ == "__main__":
    main()
