"""Hyperparameter search for OrderFlowTransformer using Optuna.

Single-GPU per trial. Each trial trains for a few epochs and reports the
validation loss after every epoch — bad trials are pruned early by
MedianPruner. The study is persisted to SQLite, so the search resumes
from where it stopped if you re-run the script.

Search space (defaults — edit `_suggest` to taste):
    lr              log-uniform [3e-5, 3e-3]
    dropout         uniform     [0.0, 0.25]
    weight_decay    log-uniform [1e-3, 1e-1]
    warmup_fraction uniform     [0.01, 0.15]
    grad_clip_norm  uniform     [0.5, 2.0]
    betas2          uniform     [0.9, 0.999]   (β1 fixed at 0.9)
    # optional architecture knobs (uncomment in _suggest):
    # d_model         categorical [256, 384, 512, 768]
    # n_layers        categorical [4, 6, 8, 12]

Usage:
    # 1. Single GPU, sequential trials:
    uv run python -m scripts.sweep --n-trials 20 --epochs-per-trial 10 --num-workers 8

    # 2. Pin to a specific GPU:
    CUDA_VISIBLE_DEVICES=0 uv run python -m scripts.sweep --n-trials 20 --epochs-per-trial 10 --num-workers 8

    # 3. Parallel sweep across 4 GPUs — all workers share the same SQLite study,
    #    so TPE sees every completed trial across all workers. Total trials = N * --n-trials.
    mkdir -p sweeps logs
    CUDA_VISIBLE_DEVICES=0 uv run python -m scripts.sweep --n-trials 20 --epochs-per-trial 10 --num-workers 8 > logs/sweep0.log 2>&1 &
    CUDA_VISIBLE_DEVICES=1 uv run python -m scripts.sweep --n-trials 20 --epochs-per-trial 10 --num-workers 8 > logs/sweep1.log 2>&1 &
    CUDA_VISIBLE_DEVICES=2 uv run python -m scripts.sweep --n-trials 20 --epochs-per-trial 10 --num-workers 8 > logs/sweep2.log 2>&1 &
    CUDA_VISIBLE_DEVICES=3 uv run python -m scripts.sweep --n-trials 20 --epochs-per-trial 10 --num-workers 8 > logs/sweep3.log 2>&1 &
    wait

    # 4. Inspect aggregated results across all workers:
    uv run python -m scripts.sweep --report

    # 5. Watch trials live in TensorBoard (per-trial subfolder under runs/sweep/):
    uv run tensorboard --logdir runs/sweep

    # Tip: if the study DB lives on NFS, point --storage to a local path to avoid SQLite locks:
    #   --storage sqlite:////local/scratch/$USER/transformer.db
"""

import argparse
import math
import os
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from tqdm.auto import tqdm

from src.config import ModelConfig
from src.data.dataset import OrderFlowDataset
from src.models.transformer import OrderFlowTransformer


# --------------------------------------------------------------------------- #
# Logging                                                                      #
# --------------------------------------------------------------------------- #

def _log_dir(storage: str, study_name: str, trial_num: int) -> Path:
    """runs/sweep/transformer/{db_name}/trial_{N}.

    `db_name` is the SQLite filename stem when storage is SQLite; otherwise we
    fall back to the study name (e.g. for Postgres backends).
    """
    if storage.startswith("sqlite"):
        db_name = Path(storage.split("///")[-1]).stem
    else:
        db_name = study_name
    return Path("runs/sweep/transformer") / db_name / f"trial_{trial_num}"


# --------------------------------------------------------------------------- #
# Search space                                                                 #
# --------------------------------------------------------------------------- #

def _suggest(trial: optuna.Trial, base: ModelConfig) -> ModelConfig:
    """Sample one set of hyperparameters and return a derived ModelConfig."""
    lr = trial.suggest_float("lr", 3e-5, 3e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.25)
    weight_decay = trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True)
    warmup_fraction = trial.suggest_float("warmup_fraction", 0.01, 0.15)
    grad_clip_norm = trial.suggest_float("grad_clip_norm", 0.5, 2.0)
    beta2 = trial.suggest_float("beta2", 0.9, 0.999)

    # --- Optional architecture knobs (uncomment to enable) ---
    # d_model = trial.suggest_categorical("d_model", [256, 384, 512, 768])
    # n_layers = trial.suggest_categorical("n_layers", [4, 6, 8, 12])
    # n_heads = trial.suggest_categorical("n_heads", [4, 8, 12])
    # d_ff = d_model * 4

    return replace(
        base,
        lr=lr,
        dropout=dropout,
        weight_decay=weight_decay,
        warmup_fraction=warmup_fraction,
        grad_clip_norm=grad_clip_norm,
        betas=(0.9, beta2),
        # d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff,
    )


# --------------------------------------------------------------------------- #
# Training helpers (kept inline so this file is self-contained)               #
# --------------------------------------------------------------------------- #

def _get_lr(step: int, warmup_steps: int, total_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def _train_one_epoch(
    model, loader, optimizer, cfg, device, step, total_steps, warmup_steps,
    amp_dtype, scaler, max_steps=None, writer: SummaryWriter | None = None,
    log_every: int = 50, tqdm_desc: str | None = None, tqdm_disable: bool = False,
):
    model.train()
    use_amp = amp_dtype is not None and device.type == "cuda"
    total = 0.0
    n = 0
    pbar = tqdm(
        loader,
        desc=tqdm_desc or "train",
        leave=False,
        disable=tqdm_disable,
        mininterval=1.0,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )
    for batch in pbar:
        tokens = batch["trade_tokens"].to(device, non_blocking=True)
        plevels = batch["price_levels"].to(device, non_blocking=True)
        liqs = batch["liquidities"].to(device, non_blocking=True)
        inst_ids = batch["instrument_id"].to(device, non_blocking=True)

        x = tokens[:, :-1]
        y = tokens[:, 1:]
        pl_in = plevels[:, :-1]
        liq_in = liqs[:, :-1]

        lr = _get_lr(step, warmup_steps, total_steps, cfg.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(x, pl_in, liq_in, inst_ids)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, cfg.vocab_size), y.reshape(-1),
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
            logits = model(x, pl_in, liq_in, inst_ids)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, cfg.vocab_size), y.reshape(-1),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

        total += loss.item()
        n += 1
        step += 1
        if writer is not None and step % log_every == 0:
            writer.add_scalar("train/loss_step", loss.item(), step)
            writer.add_scalar("train/lr", lr, step)
        avg = total / n
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            avg=f"{avg:.4f}",
            ppl=f"{math.exp(min(avg, 20)):.0f}",
            lr=f"{lr:.1e}",
        )
        if max_steps is not None and step >= max_steps:
            break
    pbar.close()
    return total / max(n, 1), step


@torch.no_grad()
def _evaluate(
    model, loader, cfg, device, amp_dtype, max_batches=None,
    tqdm_desc: str | None = None, tqdm_disable: bool = False,
):
    model.eval()
    use_amp = amp_dtype is not None and device.type == "cuda"
    total = 0.0
    n = 0
    pbar = tqdm(
        loader,
        desc=tqdm_desc or "val  ",
        leave=False,
        disable=tqdm_disable,
        mininterval=1.0,
        total=max_batches if max_batches is not None else len(loader),
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    )
    for batch in pbar:
        tokens = batch["trade_tokens"].to(device, non_blocking=True)
        plevels = batch["price_levels"].to(device, non_blocking=True)
        liqs = batch["liquidities"].to(device, non_blocking=True)
        inst_ids = batch["instrument_id"].to(device, non_blocking=True)

        x = tokens[:, :-1]
        y = tokens[:, 1:]
        pl_in = plevels[:, :-1]
        liq_in = liqs[:, :-1]

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(x, pl_in, liq_in, inst_ids)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, cfg.vocab_size), y.reshape(-1),
                )
        else:
            logits = model(x, pl_in, liq_in, inst_ids)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, cfg.vocab_size), y.reshape(-1),
            )

        total += loss.item()
        n += 1
        avg = total / n
        pbar.set_postfix(loss=f"{avg:.4f}", ppl=f"{math.exp(min(avg, 20)):.0f}")
        if max_batches is not None and n >= max_batches:
            break
    pbar.close()
    return total / max(n, 1)


# --------------------------------------------------------------------------- #
# Optuna objective                                                             #
# --------------------------------------------------------------------------- #

def _build_objective(args, train_ds, val_ds, n_instruments, device, amp_dtype):
    def objective(trial: optuna.Trial) -> float:
        base = ModelConfig()
        base.n_instruments = n_instruments
        if args.batch_size is not None:
            base.batch_size = args.batch_size
        if args.context_length is not None:
            base.context_length = args.context_length

        cfg = _suggest(trial, base)

        # Note hyperparameters in trial user_attrs for the report
        trial.set_user_attr("d_model", cfg.d_model)
        trial.set_user_attr("n_layers", cfg.n_layers)
        trial.set_user_attr("batch_size", cfg.batch_size)

        # TensorBoard writer: runs/sweep/transformer/{db_name}/trial_{N}
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

        model = OrderFlowTransformer(cfg).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr,
            weight_decay=cfg.weight_decay, betas=cfg.betas,
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
        best_val = float("inf")
        try:
            epoch_pbar = tqdm(
                range(1, args.epochs_per_trial + 1),
                desc=f"trial {trial.number}",
                leave=True,
                disable=args.no_tqdm,
                bar_format="{l_bar}{bar}| ep {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
            )
            for epoch in epoch_pbar:
                train_loss, step = _train_one_epoch(
                    model, train_loader, optimizer, cfg, device,
                    step, total_steps, warmup_steps,
                    amp_dtype, scaler,
                    max_steps=step + steps_per_epoch if args.max_steps_per_epoch else None,
                    writer=writer,
                    tqdm_desc=f"trial {trial.number} ep {epoch}/{args.epochs_per_trial} [train]",
                    tqdm_disable=args.no_tqdm,
                )
                val_loss = _evaluate(
                    model, val_loader, cfg, device, amp_dtype,
                    max_batches=args.max_val_batches,
                    tqdm_desc=f"trial {trial.number} ep {epoch}/{args.epochs_per_trial} [val]  ",
                    tqdm_disable=args.no_tqdm,
                )
                best_val = min(best_val, val_loss)
                ppl = math.exp(min(val_loss, 20))
                epoch_pbar.set_postfix(
                    train=f"{train_loss:.4f}",
                    val=f"{val_loss:.4f}",
                    ppl=f"{ppl:.0f}",
                    best=f"{best_val:.4f}",
                )

                writer.add_scalar("train/loss_epoch", train_loss, epoch)
                writer.add_scalar("val/loss", val_loss, epoch)
                writer.add_scalar("val/ppl", ppl, epoch)
                print(
                    f"[trial {trial.number}] epoch {epoch}/{args.epochs_per_trial}  "
                    f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  ppl={ppl:.2f}",
                    flush=True,
                )

                # Report intermediate result for pruning
                trial.report(val_loss, step=epoch)
                if trial.should_prune():
                    print(f"[trial {trial.number}] pruned at epoch {epoch}", flush=True)
                    raise optuna.TrialPruned()

            # HParams tab: associate this trial's params with its best val_loss
            hparams = {k: v for k, v in trial.params.items()}
            writer.add_hparams(
                hparams,
                {"hparam/val_loss": best_val,
                 "hparam/val_ppl": math.exp(min(best_val, 20))},
                run_name=".",
            )
        finally:
            writer.flush()
            writer.close()
            # Free GPU memory before next trial
            del model, optimizer
            if device.type == "cuda":
                torch.cuda.empty_cache()

        return best_val

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


def main():
    parser = argparse.ArgumentParser(description="Optuna sweep for OrderFlowTransformer")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--epochs-per-trial", type=int, default=3,
                        help="How many epochs each trial trains for")
    parser.add_argument("--max-steps-per-epoch", type=int, default=None,
                        help="Cap training steps per epoch (for very fast sweeps)")
    parser.add_argument("--max-val-batches", type=int, default=None,
                        help="Cap validation batches per epoch")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch_size (per-GPU)")
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--amp", type=str, default="bf16",
                        choices=["bf16", "fp16", "none"])
    parser.add_argument("--storage", type=str, default="sqlite:///sweeps/transformer.db",
                        help="Optuna storage URL (sqlite default; lets parallel workers share study)")
    parser.add_argument("--study-name", type=str, default="transformer-sweep")
    parser.add_argument("--report", action="store_true",
                        help="Print best trial(s) and exit (no training)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="How many top trials to print under --report")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed; actual TPE seed = (seed + PID) so parallel "
                             "workers don't sample identical params.")
    parser.add_argument("--no-tqdm", action="store_true",
                        help="Disable tqdm progress bars (recommended when "
                             "running multiple parallel workers redirected to log files).")
    args = parser.parse_args()

    Path("sweeps").mkdir(exist_ok=True)

    storage = args.storage
    # IMPORTANT: per-worker seed.
    # If 4 parallel workers share the same TPESampler seed, they all sample the
    # same random params during the startup phase (TPE's first ~10 trials are
    # random) AND they may collide later because each worker maintains its own
    # in-memory RNG state — only completed trial *results* are shared via
    # SQLite, not the sampler RNG. Mixing PID into the seed gives each worker
    # an independent stream while still being reproducible (per-process).
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
    print(f"Storage: {storage} | Study: {args.study_name}")

    print("Loading datasets...")
    cfg_for_data = ModelConfig()
    if args.context_length is not None:
        cfg_for_data.context_length = args.context_length
    train_ds = OrderFlowDataset(cfg_for_data, split="train")
    val_ds = OrderFlowDataset(cfg_for_data, split="val")
    print(f"Train windows: {len(train_ds):,}, Val windows: {len(val_ds):,}")
    n_instruments = len(train_ds.instrument_names)

    objective = _build_objective(args, train_ds, val_ds, n_instruments, device, amp_dtype)

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

    print(f"\nBest trial: #{study.best_trial.number}  val_loss={study.best_value:.4f}  "
          f"(PPL={math.exp(min(study.best_value, 20)):.2f})")
    for k, v in study.best_trial.params.items():
        print(f"    {k:>16} = {v}")

    print(f"\nTop-{top_k} trials by val_loss:")
    sorted_trials = sorted(completed, key=lambda t: t.value)[:top_k]
    print(f"  {'#':>4}  {'val_loss':>9}  {'PPL':>6}  params")
    for t in sorted_trials:
        ppl = math.exp(min(t.value, 20))
        params_str = " ".join(
            f"{k}={v:.2e}" if isinstance(v, float) else f"{k}={v}"
            for k, v in t.params.items()
        )
        print(f"  {t.number:>4}  {t.value:>9.4f}  {ppl:>6.2f}  {params_str}")


if __name__ == "__main__":
    main()
