"""Hyperparameter search for decision heads (alpha / risk / intensity) using Optuna.

Single-GPU per trial. The frozen transformer backbone is loaded ONCE before the
study starts and shared across all trials — only the head MLPs and the
optimizer are rebuilt per trial. Each trial trains for a few epochs and reports
its objective after every epoch — bad trials are pruned early by MedianPruner.
The study is persisted to SQLite, so the search resumes if you re-run.

Search space (defaults — edit `_suggest`):
    lr               log-uniform [1e-4, 1e-2]
    weight_decay     log-uniform [1e-4, 1e-1]
    warmup_fraction  uniform     [0.01, 0.15]
    huber_delta      log-uniform [1e-5, 1e-3]   # alpha-loss scale (returns ≈ 1e-4)
    w_alpha          uniform     [0.1, 3.0]
    w_risk           uniform     [0.1, 3.0]
    w_intensity      uniform     [0.1, 3.0]
    latent_layer     categorical [-1, -2]       # which transformer layer feeds the heads

Objectives (--objective):
    loss        (default) minimise val total loss
    composite   minimise -(IC(α) + Spearman(σ) + Spearman(κ))  — directly targets
                the metrics the project cares about (paper-style)

Usage:
    # Single GPU
    uv run python -m scripts.sweep_heads --n-trials 20 --epochs-per-trial 10 --num-workers 8

    # Pin to a specific GPU
    CUDA_VISIBLE_DEVICES=0 uv run python -m scripts.sweep_heads --n-trials 20 --epochs-per-trial 10 --num-workers 8

    # Parallel sweep across 4 GPUs — all workers share the same SQLite study,
    # so TPE sees every completed trial across all workers. Total trials = N * --n-trials.
    mkdir -p sweeps logs
    CUDA_VISIBLE_DEVICES=0 uv run python -m scripts.sweep_heads --n-trials 20 --epochs-per-trial 10 --num-workers 8 > logs/heads_sweep0.log 2>&1 &
    CUDA_VISIBLE_DEVICES=1 uv run python -m scripts.sweep_heads --n-trials 20 --epochs-per-trial 10 --num-workers 8 > logs/heads_sweep1.log 2>&1 &
    CUDA_VISIBLE_DEVICES=2 uv run python -m scripts.sweep_heads --n-trials 20 --epochs-per-trial 10 --num-workers 8 > logs/heads_sweep2.log 2>&1 &
    CUDA_VISIBLE_DEVICES=3 uv run python -m scripts.sweep_heads --n-trials 20 --epochs-per-trial 10 --num-workers 8 > logs/heads_sweep3.log 2>&1 &
    wait

    # Aggregated report
    uv run python -m scripts.sweep_heads --report

    # TensorBoard (per-trial subfolder under runs/sweep/heads/)
    uv run tensorboard --logdir runs/sweep/heads

    # Composite objective targeting IC + Spearman directly
    uv run python -m scripts.sweep_heads --n-trials 20 --epochs-per-trial 10 --objective composite

    # Different transformer checkpoint
    uv run python -m scripts.sweep_heads --transformer-checkpoint checkpoints/big/best.pt
"""

import argparse
import math
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from src.config import HeadConfig, ModelConfig
from src.data.dataset import OrderFlowDataset
from src._archive.decision.heads import DecisionModule
from src.models.transformer import OrderFlowTransformer


# --------------------------------------------------------------------------- #
# Logging                                                                      #
# --------------------------------------------------------------------------- #

def _log_dir(storage: str, study_name: str, trial_num: int) -> Path:
    """runs/sweep/heads/{db_name}/trial_{N}.

    `db_name` is the SQLite filename stem when storage is SQLite; otherwise we
    fall back to the study name (e.g. for Postgres backends).
    """
    if storage.startswith("sqlite"):
        db_name = Path(storage.split("///")[-1]).stem
    else:
        db_name = study_name
    return Path("runs/sweep/heads") / db_name / f"trial_{trial_num}"


# --------------------------------------------------------------------------- #
# Search space                                                                 #
# --------------------------------------------------------------------------- #

def _suggest(trial: optuna.Trial, base: HeadConfig) -> HeadConfig:
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    warmup_fraction = trial.suggest_float("warmup_fraction", 0.01, 0.15)
    huber_delta = trial.suggest_float("huber_delta", 1e-5, 1e-3, log=True)
    w_alpha = trial.suggest_float("w_alpha", 0.1, 3.0)
    w_risk = trial.suggest_float("w_risk", 0.1, 3.0)
    w_intensity = trial.suggest_float("w_intensity", 0.1, 3.0)
    latent_layer = trial.suggest_categorical("latent_layer", [-1, -2])

    return replace(
        base,
        lr=lr,
        weight_decay=weight_decay,
        warmup_fraction=warmup_fraction,
        huber_delta=huber_delta,
        w_alpha=w_alpha,
        w_risk=w_risk,
        w_intensity=w_intensity,
        latent_layer=latent_layer,
    )


# --------------------------------------------------------------------------- #
# Correlation metrics (rank-0, single-process — no DDP here)                   #
# --------------------------------------------------------------------------- #

def _rankdata(arr: np.ndarray) -> np.ndarray:
    order = arr.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(arr) + 1, dtype=np.float64)
    return ranks


def _spearman_corr(pred: np.ndarray, target: np.ndarray) -> float:
    if len(pred) < 10:
        return 0.0
    if pred.std() < 1e-12 or target.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(_rankdata(pred), _rankdata(target))[0, 1])


def _pearson_corr(pred: np.ndarray, target: np.ndarray) -> float:
    if len(pred) < 10:
        return 0.0
    if pred.std() < 1e-12 or target.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(pred, target)[0, 1])


# --------------------------------------------------------------------------- #
# Training helpers (kept inline so this file is self-contained)               #
# --------------------------------------------------------------------------- #

def _get_lr(step: int, warmup_steps: int, total_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def _train_one_epoch(
    transformer, heads, loader, optimizer, cfg, device, step,
    total_steps, warmup_steps, amp_dtype, scaler, max_steps=None,
    writer: SummaryWriter | None = None, log_every: int = 50,
):
    heads.train()
    use_amp = amp_dtype is not None and device.type == "cuda"
    sums = {"alpha": 0.0, "risk": 0.0, "intensity": 0.0, "total": 0.0}
    n = 0
    for batch in loader:
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

        lr = _get_lr(step, warmup_steps, total_steps, cfg.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        # Frozen transformer forward — under no_grad even with autocast
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
                losses = heads.compute_loss(
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
            losses = heads.compute_loss(
                preds, targets, mask,
                huber_delta=cfg.huber_delta,
                w_alpha=cfg.w_alpha, w_risk=cfg.w_risk, w_intensity=cfg.w_intensity,
            )
            losses["total"].backward()
            optimizer.step()

        for k in sums:
            sums[k] += losses[k].item()
        n += 1
        step += 1

        if writer is not None and step % log_every == 0:
            writer.add_scalar("train/lr", lr, step)
            for k in sums:
                writer.add_scalar(f"train/loss_{k}_step", losses[k].item(), step)

        if max_steps is not None and step >= max_steps:
            break

    avgs = {k: v / max(n, 1) for k, v in sums.items()}
    return avgs, step


@torch.no_grad()
def _evaluate(
    transformer, heads, loader, cfg, device, amp_dtype,
    max_batches=None, max_samples=500_000,
):
    heads.eval()
    use_amp = amp_dtype is not None and device.type == "cuda"
    sums = {"alpha": 0.0, "risk": 0.0, "intensity": 0.0, "total": 0.0}
    n_batches = 0

    all_preds = {"alpha": [], "risk": [], "intensity": []}
    all_targets = {"alpha": [], "risk": [], "intensity": []}
    n_collected = 0

    for batch in loader:
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
                losses = heads.compute_loss(
                    preds, targets, mask,
                    huber_delta=cfg.huber_delta,
                    w_alpha=cfg.w_alpha, w_risk=cfg.w_risk, w_intensity=cfg.w_intensity,
                )
        else:
            hidden = transformer.extract_hidden_states(
                tokens, plevels, liqs, inst_ids, layer=cfg.latent_layer,
            )
            preds = heads(hidden)
            losses = heads.compute_loss(
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

        if max_batches is not None and n_batches >= max_batches:
            break

    avg_losses = {k: v / max(n_batches, 1) for k, v in sums.items()}

    cat = lambda buf: np.concatenate(buf) if buf else np.zeros(0, dtype=np.float32)
    ap, at = cat(all_preds["alpha"]), cat(all_targets["alpha"])
    rp, rt = cat(all_preds["risk"]),  cat(all_targets["risk"])
    ip, it = cat(all_preds["intensity"]), cat(all_targets["intensity"])

    eps = 1e-8
    metrics = {
        "ic_alpha": _pearson_corr(ap, at),
        "spearman_risk": _spearman_corr(np.log(rp + eps), np.log(rt + eps)),
        "spearman_intensity": _spearman_corr(np.log(ip + eps), np.log(it + eps)),
        "alpha_pred_std": float(ap.std()) if ap.size else 0.0,
    }
    return avg_losses, metrics


# --------------------------------------------------------------------------- #
# Optuna objective                                                             #
# --------------------------------------------------------------------------- #

def _build_objective(args, transformer, train_ds, val_ds, device, amp_dtype):
    def objective(trial: optuna.Trial) -> float:
        base = HeadConfig()
        # The transformer's d_model is fixed by the loaded checkpoint
        base.d_model = args.d_model
        base.transformer_checkpoint = args.transformer_checkpoint
        if args.batch_size is not None:
            base.batch_size = args.batch_size

        cfg = _suggest(trial, base)

        trial.set_user_attr("d_model", cfg.d_model)
        trial.set_user_attr("batch_size", cfg.batch_size)

        log_dir = _log_dir(args.storage, args.study_name, trial.number)
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))
        print(f"[trial {trial.number}] log_dir={log_dir}  params={trial.params}",
              flush=True)

        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )

        heads = DecisionModule(cfg.d_model).to(device)
        optimizer = torch.optim.AdamW(
            heads.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
        scaler = torch.amp.GradScaler("cuda") if (amp_dtype == torch.float16) else None

        steps_per_epoch = (
            args.max_steps_per_epoch
            if args.max_steps_per_epoch is not None
            else len(train_loader)
        )
        total_steps = steps_per_epoch * args.epochs_per_trial
        warmup_steps = int(total_steps * cfg.warmup_fraction)

        step = 0
        best_obj = float("inf")
        try:
            for epoch in range(1, args.epochs_per_trial + 1):
                train_losses, step = _train_one_epoch(
                    transformer, heads, train_loader, optimizer, cfg, device,
                    step, total_steps, warmup_steps,
                    amp_dtype, scaler,
                    max_steps=step + steps_per_epoch if args.max_steps_per_epoch else None,
                    writer=writer,
                )
                val_losses, val_metrics = _evaluate(
                    transformer, heads, val_loader, cfg, device, amp_dtype,
                    max_batches=args.max_val_batches,
                )

                # Per-epoch TB
                for split, losses in (("train", train_losses), ("val", val_losses)):
                    for k, v in losses.items():
                        writer.add_scalar(f"{split}/loss_{k}_epoch", v, epoch)
                for k, v in val_metrics.items():
                    writer.add_scalar(f"val/metrics/{k}", v, epoch)

                # Compose the trial's objective
                if args.objective == "loss":
                    obj = val_losses["total"]
                else:  # composite — minimize negative sum of correlations
                    obj = -(val_metrics["ic_alpha"]
                            + val_metrics["spearman_risk"]
                            + val_metrics["spearman_intensity"])

                best_obj = min(best_obj, obj)

                # Stash both per-trial so --report can show everything
                trial.set_user_attr(f"epoch_{epoch}_val_total", val_losses["total"])
                trial.set_user_attr(f"epoch_{epoch}_ic_alpha", val_metrics["ic_alpha"])
                trial.set_user_attr(f"epoch_{epoch}_sp_risk", val_metrics["spearman_risk"])
                trial.set_user_attr(f"epoch_{epoch}_sp_intensity", val_metrics["spearman_intensity"])

                print(
                    f"[trial {trial.number}] epoch {epoch}/{args.epochs_per_trial}  "
                    f"val_total={val_losses['total']:.4f}  "
                    f"IC(α)={val_metrics['ic_alpha']:+.4f}  "
                    f"Sp(σ)={val_metrics['spearman_risk']:+.4f}  "
                    f"Sp(κ)={val_metrics['spearman_intensity']:+.4f}  "
                    f"obj={obj:+.4f}",
                    flush=True,
                )

                trial.report(obj, step=epoch)
                if trial.should_prune():
                    print(f"[trial {trial.number}] pruned at epoch {epoch}", flush=True)
                    raise optuna.TrialPruned()

            # HParams tab
            hparams = {k: v for k, v in trial.params.items()}
            writer.add_hparams(
                hparams,
                {"hparam/objective": best_obj,
                 "hparam/val_total": val_losses["total"],
                 "hparam/ic_alpha": val_metrics["ic_alpha"],
                 "hparam/sp_risk": val_metrics["spearman_risk"],
                 "hparam/sp_intensity": val_metrics["spearman_intensity"]},
                run_name=".",
            )
        finally:
            writer.flush()
            writer.close()
            del heads, optimizer
            if device.type == "cuda":
                torch.cuda.empty_cache()

        return best_obj

    return objective


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

def _select_device(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _amp_dtype(name: str) -> torch.dtype | None:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[name]


def _load_frozen_transformer(ckpt_path: str, device: torch.device):
    p = Path(ckpt_path)
    if not p.exists():
        raise FileNotFoundError(f"Transformer checkpoint not found: {p}")
    ck = torch.load(p, map_location=device, weights_only=False)
    saved_cfg = ck["config"]  # ModelConfig saved during sweep/train
    model = OrderFlowTransformer(saved_cfg).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    for prm in model.parameters():
        prm.requires_grad = False
    n = sum(prm.numel() for prm in model.parameters())
    print(f"Loaded frozen transformer: {n:,} params  d_model={saved_cfg.d_model}  "
          f"context_length={saved_cfg.context_length}  "
          f"(epoch {ck.get('epoch', '?')}, val_loss {ck.get('val_loss', '?')})")
    return model, saved_cfg


def main():
    parser = argparse.ArgumentParser(description="Optuna sweep for decision heads")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--epochs-per-trial", type=int, default=10)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None,
                        help="Cap training steps per epoch (for very fast sweeps)")
    parser.add_argument("--max-val-batches", type=int, default=None,
                        help="Cap validation batches per epoch")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override per-GPU batch_size")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--amp", type=str, default="bf16",
                        choices=["bf16", "fp16", "none"])
    parser.add_argument("--objective", type=str, default="loss",
                        choices=["loss", "composite"],
                        help="`loss` = minimise val total loss; "
                             "`composite` = minimise -(IC(α)+Sp(σ)+Sp(κ))")
    parser.add_argument("--transformer-checkpoint", type=str,
                        default="checkpoints/best.pt",
                        help="Frozen backbone checkpoint")
    parser.add_argument("--storage", type=str,
                        default="sqlite:///sweeps/heads.db",
                        help="Optuna storage URL")
    parser.add_argument("--study-name", type=str, default="heads-sweep")
    parser.add_argument("--report", action="store_true",
                        help="Print best trial(s) and exit (no training)")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed; actual TPE seed = (seed + PID) so parallel "
                             "workers don't sample identical params.")
    args = parser.parse_args()

    Path("sweeps").mkdir(exist_ok=True)

    storage = args.storage
    sampler_seed = (args.seed + os.getpid()) & 0xFFFFFFFF
    sampler = TPESampler(multivariate=True, group=True, seed=sampler_seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        load_if_exists=True,
    )

    if args.report:
        _print_report(study, args.top_k)
        return

    device = _select_device(args.device)
    amp_dtype = _amp_dtype(args.amp)
    if device.type != "cuda":
        amp_dtype = None
    print(f"Device: {device} | AMP: {args.amp if amp_dtype else 'off'}")
    print(f"Storage: {storage} | Study: {args.study_name} | Objective: {args.objective}")

    # Load frozen backbone ONCE — shared across all trials
    transformer, saved_cfg = _load_frozen_transformer(args.transformer_checkpoint, device)

    # Build datasets ONCE — same context_length / stride as the trained backbone
    print("Loading datasets...")
    cfg_for_data = ModelConfig()
    cfg_for_data.context_length = saved_cfg.context_length
    cfg_for_data.stride = saved_cfg.stride
    train_ds = OrderFlowDataset(cfg_for_data, split="train", include_targets=True)
    val_ds = OrderFlowDataset(cfg_for_data, split="val", include_targets=True)
    print(f"Train windows: {len(train_ds):,}, Val windows: {len(val_ds):,}")

    # Pass d_model into args so _build_objective uses the right size
    args.d_model = saved_cfg.d_model

    objective = _build_objective(args, transformer, train_ds, val_ds, device, amp_dtype)

    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)
    print(f"\nFinished {len(study.trials)} trial(s) in {time.time() - t0:.0f}s")
    _print_report(study, args.top_k)


def _print_report(study: optuna.Study, top_k: int) -> None:
    print(f"\nStudy: {study.study_name}  ({len(study.trials)} trials)")
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"Completed: {len(completed)}  |  Pruned: {len(pruned)}  |  Failed: "
          f"{len(study.trials) - len(completed) - len(pruned)}")

    if not completed:
        print("(no completed trials yet)")
        return

    best = study.best_trial
    print(f"\nBest trial: #{best.number}  objective={study.best_value:+.4f}")
    for k, v in best.params.items():
        print(f"    {k:>16} = {v}")
    # Show the per-epoch metrics stored in user_attrs
    print("    -- val metrics by epoch --")
    epochs = sorted({int(k.split("_")[1]) for k in best.user_attrs
                     if k.startswith("epoch_") and k.endswith("_val_total")})
    for e in epochs:
        print(
            f"    epoch {e}: total={best.user_attrs.get(f'epoch_{e}_val_total', float('nan')):.4f}  "
            f"IC(α)={best.user_attrs.get(f'epoch_{e}_ic_alpha', float('nan')):+.4f}  "
            f"Sp(σ)={best.user_attrs.get(f'epoch_{e}_sp_risk', float('nan')):+.4f}  "
            f"Sp(κ)={best.user_attrs.get(f'epoch_{e}_sp_intensity', float('nan')):+.4f}"
        )

    print(f"\nTop-{top_k} trials by objective:")
    sorted_trials = sorted(completed, key=lambda t: t.value)[:top_k]
    print(f"  {'#':>4}  {'obj':>9}  params")
    for t in sorted_trials:
        params_str = " ".join(
            f"{k}={v:.2e}" if isinstance(v, float) else f"{k}={v}"
            for k, v in t.params.items()
        )
        print(f"  {t.number:>4}  {t.value:>+9.4f}  {params_str}")


if __name__ == "__main__":
    main()
