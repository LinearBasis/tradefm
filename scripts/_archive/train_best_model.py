"""Train OrderFlowTransformer using best hyperparameters from an Optuna study.

After scripts/sweep finishes, this script loads the best completed trial and
launches the full training run (scripts.train.main) with those hyperparameters
baked into ModelConfig defaults. Supports single-GPU and DDP via torchrun —
any unrecognised flags are forwarded to scripts/train.

What gets patched from the trial:
    lr, dropout, weight_decay, warmup_fraction, grad_clip_norm, betas (β2)
    + architecture knobs (d_model, n_layers, n_heads, d_ff, d_context,
      context_length) if the sweep had them enabled.

Usage:
    # Inspect best trial without launching training
    uv run python -m scripts.train_best_model --print-only

    # Single GPU, full long run
    uv run python -m scripts.train_best_model --epochs 30 --batch-size 64 --num-workers 8

    # 4 GPUs via DDP — same flags as scripts/train, but ModelConfig defaults
    # are already patched from the sweep.
    CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --standalone --nproc-per-node=4 \
        -m scripts.train_best_model --epochs 30 --batch-size 32 --num-workers 8

    # Use a different study/db
    uv run python -m scripts.train_best_model \
        --study-name my-sweep --storage sqlite:///sweeps/my.db --epochs 30

NOTE on effective batch size: the sweep ran each trial on 1 GPU with a fixed
per-GPU batch_size. When you launch the final run on N GPUs, effective batch
becomes N * batch_size. The patched LR was tuned for the single-GPU effective
batch — consider scaling it by `--lr {sweep_lr * N}` for N-GPU DDP, or keep
the same effective batch by passing `--batch-size {sweep_bs / N}`.
"""

import argparse
import sys

import optuna

from src.config import ModelConfig


def _load_best_params(storage: str, study_name: str) -> tuple[int, float, dict]:
    study = optuna.load_study(study_name=study_name, storage=storage)
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(
            f"No completed trials in study {study_name!r} at {storage}. "
            f"Has the sweep finished any trial?"
        )
    best = study.best_trial
    return best.number, float(best.value), dict(best.params)


def _patch_model_config(params: dict) -> dict:
    """Inject best-trial params as ModelConfig defaults BEFORE scripts.train
    instantiates it. We can't simply mutate ``__dataclass_fields__[k].default``
    because the dataclass-generated ``__init__`` bakes defaults into its
    signature at class-creation time. Instead we wrap ``__init__`` with a thin
    shim that fills in any unspecified field with the patch value — train.py
    calls ``ModelConfig()`` without args, so all patches take effect. Any CLI
    override (``--lr``, ``--dropout``, ...) still wins because it is applied
    via ``setattr`` *after* ``__init__`` in train.main."""
    applied: dict = {}
    patches: dict = {}

    for key in ("lr", "dropout", "weight_decay", "warmup_fraction", "grad_clip_norm"):
        if key in params:
            patches[key] = params[key]
            applied[key] = params[key]
    if "beta2" in params:
        patches["betas"] = (0.9, float(params["beta2"]))
        applied["betas"] = (0.9, float(params["beta2"]))
    for key in ("d_model", "n_layers", "n_heads", "d_ff",
                "d_context", "context_length"):
        if key in params:
            patches[key] = params[key]
            applied[key] = params[key]

    if not patches:
        return applied

    _orig_init = ModelConfig.__init__

    def _patched_init(self, **kwargs):
        for k, v in patches.items():
            kwargs.setdefault(k, v)
        _orig_init(self, **kwargs)

    ModelConfig.__init__ = _patched_init  # type: ignore[method-assign]
    return applied


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Launch scripts.train.main with hyperparameters from the best "
            "Optuna trial of a previously-run sweep. Unknown flags are "
            "forwarded verbatim to scripts/train (e.g. --epochs, --batch-size, "
            "--num-workers, --amp, --resume, --max-steps, --run-name, ...)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--storage", type=str,
                        default="sqlite:///sweeps/transformer.db",
                        help="Optuna storage URL of the completed sweep "
                             "(default: sqlite:///sweeps/transformer.db)")
    parser.add_argument("--study-name", type=str, default="transformer-sweep",
                        help="Optuna study name to read from "
                             "(default: transformer-sweep)")
    parser.add_argument("--print-only", action="store_true",
                        help="Print best params and exit (do not launch training)")
    args, forwarded = parser.parse_known_args()

    n, val, params = _load_best_params(args.storage, args.study_name)
    print(f"\nStudy: {args.study_name} ({args.storage})")
    print(f"Best trial #{n} | val_loss={val:.4f}")
    for k, v in params.items():
        if isinstance(v, float):
            print(f"  {k:>18} = {v:.6g}")
        else:
            print(f"  {k:>18} = {v}")

    if args.print_only:
        return

    applied = _patch_model_config(params)
    if applied:
        print("\nPatched ModelConfig defaults:")
        for k, v in applied.items():
            if isinstance(v, float):
                print(f"  {k:>18} = {v:.6g}")
            else:
                print(f"  {k:>18} = {v}")

    print(f"\nForwarding to scripts.train.main: {' '.join(forwarded) or '(no extra flags)'}\n")

    # Hand off to train.main with the forwarded argv. We import AFTER patching
    # so any module-level use of ModelConfig (there shouldn't be any, but just
    # in case) sees the new defaults.
    sys.argv = ["scripts.train", *forwarded]
    from scripts.train import main as train_main
    train_main()


if __name__ == "__main__":
    main()
