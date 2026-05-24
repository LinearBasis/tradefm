# Запуск TradeFM на кластере

Ветка: `cleared_tradefm`. Цель: натренировать декодер-only TradeFM на месяце MOEX-данных, замерить метрики paper'а (stylized facts, K-S, W₁).

## 0. Окружение

```bash
git clone https://github.com/LinearBasis/tradefm.git && cd tradefm
git checkout cleared_tradefm
uv sync
uv run python -c "import torch, numba; print('cuda?', torch.cuda.is_available(), '| numba', numba.__version__)"
```

Зависимости подтянутся из `pyproject.toml`. На H100 — `torch>=2.5` (нужно для `enable_gqa` в SDPA, и `torch.compile`'у на bf16).

## 1. Токенизация (~3–6 мин на месяц)

Pipeline принимает `.csv`, `.txt`, `.parquet` в одном каталоге. EW-VWAP под `@njit` (×100 быстрее старого). Tokenizer состояние (edges + ADV + instruments list) сохраняется в `data/processed/tokenizer.json` для воспроизводимости.

```bash
uv run python -m src.data.pipeline \
    --raw-dir data/raw \
    --output-dir data/processed \
    --top-n-instruments 20 \
    --n-jobs 8 --polars-threads 4
```

Артефакты:
- `data/processed/sequences/<SECCODE>_<date>.parquet` — токенизированные последовательности (per-instrument per-day)
- `data/processed/sequences/manifest.json` — train/val split, instruments
- `data/processed/tokenizer.json` — self-contained tokenizer (edges + ADV + instruments + extents)

Опциональные флаги:
- `--reuse-tokenizer` — пропустить калибровку, грузить tokenizer из `data/processed/tokenizer.json` (для прогона того же набора инструментов на новых данных)
- `--instruments "SBER,GAZP,LKOH"` — ручной whitelist вместо top-N

## 2. Трансформер (8×H100)

Production: **base_50m** (52M params, Muon hybrid, Llama-faithful: SwiGLU + d_ff=8/3·d_model + RMSNorm). Маленький вариант **base_20m** (19M) — для быстрых ablation-прогонов.

Оба конфига включают `use_compile: true` (`torch.compile` через TorchDynamo+Inductor → fused Triton-ядра, +30–80% throughput на H100), `enable_gqa=True` в SDPA (избегает материализации K/V), и `AdamW(fused=True)` (один CUDA-kernel на optimizer.step). Все три — silent fallback на CPU/MPS.

**52M** (рекомендуемая точка):
```bash
mkdir -p logs checkpoints
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run torchrun --standalone --nproc-per-node=8 \
    -m scripts.train_transformer --config configs/base_50m.json \
    --num-workers 8 --amp bf16 --run-name xfmr_50m \
    2>&1 | tee logs/xfmr_50m.log
```
Артефакт: `checkpoints/transformer_50m/best.pt`.

**19M** (быстрая итерация):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run torchrun --standalone --nproc-per-node=8 \
    -m scripts.train_transformer --config configs/base_20m.json \
    --num-workers 8 --amp bf16 --run-name xfmr_20m \
    2>&1 | tee logs/xfmr_20m.log
```
Артефакт: `checkpoints/transformer_20m/best.pt`.

При крэше — `--resume <checkpoint_dir>/last.pt`. Resume падает с понятной ошибкой, если число инструментов в данных отличается от чекпоинта.

`torch.compile` можно временно выключить флагом `--no-compile` (полезно при дебаге — eager-режим даёт читаемые tracebacks). Первый forward с компиляцией ~30–90 сек, дальше кешируется в `__pycache__/`. Чекпоинты сохраняются с «голыми» state_dict-ключами (без `_orig_mod.`/`module.` префиксов), поэтому совместимы между compiled/eager/DDP-режимами.

**Что мониторить в TensorBoard:**
- `train/loss_step`, `loss/epoch` — должна идти к 4–5 (старт у ln(16384) ≈ 9.7)
- `eval/acc_{action,side,depth,vol,iat}` — per-subtoken accuracy. action/side стартуют от 0.5 (random на бинарных), depth/vol/iat от 0.0625 (random на 16-way). Все 5 должны оторваться к 5–10 эпохе.
- `rollout/{kurtosis_10s,acf_returns_lag1,acf_abs_returns_lag1}` — приходит каждые `rollout_every_n_epochs` эпох (если включён в конфиге). ACF(returns) lag1 должен идти к 0, ACF(|returns|) — оставаться ≥0.3.

```bash
tensorboard --logdir runs/ --port 6006 --bind_all
```

## 3. Eval — closed-loop rollout + stylized facts (~10–30 мин)

Standalone-тулза, можно гонять на любом чекпоинте. Поддерживает single-инструмент и `--all-instruments` для sweep.

**Sweep по всем инструментам:**
```bash
uv run python -m scripts.eval_rollout \
    --checkpoint checkpoints/transformer_50m/best.pt \
    --tokenizer data/processed/tokenizer.json \
    --val-sequences data/processed/sequences \
    --all-instruments \
    --n-rollouts 10 --n-events 1024 \
    --init-mid 250.0 \
    --output runs/eval/xfmr_50m_best \
    --device cuda
```

**Один инструмент** (для дебага):
```bash
uv run python -m scripts.eval_rollout \
    --checkpoint checkpoints/transformer_50m/best.pt \
    --tokenizer data/processed/tokenizer.json \
    --val-sequences data/processed/sequences \
    --instrument SBER \
    --n-rollouts 10 --n-events 1024 \
    --init-mid 250.0 \
    --output runs/eval/xfmr_50m_SBER \
    --device cuda
```

Артефакт: `runs/eval/<name>/metrics.json` + TensorBoard events.

**Ориентиры для сравнения с paper (Table 2, 3):**
- `aggregated/overall/distributional_fidelity/iat/ks`: paper TradeFM-500M = **0.281**, Hawkes-baseline = 0.515
- `aggregated/overall/distributional_fidelity/depth/ks`: paper = **0.169**, Hawkes = 0.281
- `aggregated/overall/stylized_facts/kurtosis_10s`: real ≈ 80, paper TradeFM ≈ 60, random модель ≈ 6
- `aggregated/overall/stylized_facts/acf_abs_returns_lag1`: real ≈ 0.4 (volatility clustering), random ≈ 0

## 4. Cross-dataset reuse (новый месяц → существующий чекпоинт)

Tokenizer self-contained → переносим и применяем без повторной калибровки.

```bash
# На машине A — токенизируем март, тренируем
uv run python -m src.data.pipeline --raw-dir march_data --output-dir march_proc
# ... train as in §2 ...

# Переносим артефакты
scp march_proc/tokenizer.json B:/data/april_proc/tokenizer.json
scp checkpoints/transformer_50m/best.pt B:/checkpoints/

# На машине B — апрель с reuse'ом tokenizer'а из марта
uv run python -m src.data.pipeline \
    --raw-dir april_data --output-dir /data/april_proc \
    --reuse-tokenizer
# → пропускает калибровку, использует инструменты + bin edges + ADV из марта
# → WARNING если каких-то инструментов нет в апреле

uv run python -m scripts.eval_rollout \
    --checkpoint /checkpoints/best.pt \
    --tokenizer /data/april_proc/tokenizer.json \
    --val-sequences /data/april_proc/sequences \
    --all-instruments --n-rollouts 10 --n-events 1024 \
    --output runs/eval/april/
# → проверит, что tokenizer.instruments == manifest.instruments == cfg.n_instruments
# → упадёт с понятной ошибкой при рассинхроне
```

## Подводные камни

- **`--num-workers ≥ 8`** — иначе DataLoader станет горлышком на H100.
- **bf16 бесплатный на H100** — оставлять. fp16 включать только при OOM (нужен GradScaler).
- **OOM** → уменьшить `batch_size` в конфиге (64 → 32) или включить gradient checkpointing (не реализовано как флаг, потребует ручной правки PreNormBlock через `torch.utils.checkpoint.checkpoint`).
- **LR vs число GPU**: `cfg.lr=3e-4` под per-rank batch=64. На 8 GPU effective_batch=512. При нестабильном loss в первые шаги — снизить до `1e-4`; если кривая плоская — поднять до `6e-4`. Muon-сторона `muon_lr=0.02` отдельно — её обычно не трогать.
- **Rollout-хук в обучении** включается через `cfg.rollout_every_n_epochs > 0` (default 0 = off). Запускается **только на rank 0**, остальные ранги ждут — на 50M модели с `n_events=512 × n_rollouts=3` это ~30–60 сек простоя 7 GPU раз в N эпох. Ставить N ≥ 5.
- **Numba первый раз компилит EW-VWAP** ~1 сек в первом воркере (`cache=True` пишет в `__pycache__/`, остальные воркеры читают). Не пугаться.
- **Чекпоинты несовместимы между разными `n_instruments`**: train_transformer и eval_rollout явно падают при mismatch. Чтобы перенести модель на новый набор инструментов — `--reuse-tokenizer` (см. §4) или train from scratch.
- **Если grad clip ловит inf/nan** — посмотреть `train/lr` (warmup не должен дать step LR на первой итерации), затем уменьшить `muon_update_rescale` с 0.2 до 0.1.
