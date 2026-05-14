---
name: Hyperparameters live in JSON configs, not CLI flags
description: For training scripts, all experiment hyperparams must be in configs/*.json driving src/config.py dataclasses. CLI is for runtime concerns only.
type: feedback
originSessionId: b2f57b68-c034-46f4-af86-74f3c3a74f04
---
Training scripts (`scripts/train_transformer.py`, `scripts/train_heads.py`) take a JSON config via `--config configs/<name>.json`. The config is loaded with `src.config.load_from_json(ModelConfig|HeadConfig, path)` and overrides the dataclass defaults.

CLI flags are reserved for **runtime concerns only**: `--config`, `--device`, `--allow-cpu`, `--amp`, `--num-workers`, `--max-steps` (smoke utility), `--resume`, `--run-name`. Architecture sizes, optimizer hyperparams, dataset stride/length, paths, loss weights — all in JSON.

**Why:** User pushback on a smoke run that passed all hyperparams via CLI: source-of-truth gets split between dataclass defaults and CLI invocation, command lines blow up to 10+ flags, and reproducibility suffers (the canonical config of an experiment lives in shell history rather than a versioned file). The dataclasses already exist and are the right home; CLI override-everything pattern was working around that, not embracing it.

**How to apply:**
- When adding a new hyperparameter, put it in the corresponding `@dataclass` in `src/config.py` and (if the experiment needs a non-default value) in `configs/*.json`. Do NOT add a CLI flag for it.
- New experiments → new `configs/<exp>.json`. Avoid editing existing configs in place; copy and adjust.
- For one-off overrides during dev (e.g. quick sanity check), still spin up a JSON in `configs/_scratch/` rather than passing the value on CLI.
- The runtime-flag set is short and stable; don't grow it without a clear runtime/system reason.
