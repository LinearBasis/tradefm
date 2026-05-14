# Текущий запуск на кластере

Цель: получить **сравнение baseline A-S (константы) vs наша модель A-S (μ/σ/κ от голов)** на SBER. Данные (`data/raw/*`, `data/hftbacktest/*.npz`) уже на кластере; здесь только код, конфиги и эта инструкция.

## 0. Окружение

```bash
tar xzf tradefm.tar.gz && cd tradefm
uv sync
uv run python -c "import torch; print('cuda?', torch.cuda.is_available())"
```

## 1. Регенерация sequences (~30–60 мин)

Старые таргеты были в **событиях**, новые — в **секундах** (`tau_sec=1.0`). Старый `data/processed/sequences/` несовместим и должен быть пересобран.

```bash
rm -rf data/processed/sequences
uv run python -m src.data.pipeline --n-jobs 8 --polars-threads 4
```

## 2. Трансформер (4×H100)

Архитектура — Llama-family (GQA + SwiGLU + RoPE), повторяет концепции TradeFM paper. Два варианта размера; обучаем оба и сравним.

**75M** (paper operating point ~72 tok/p на 4 эпохах, рекомендованный):
```bash
mkdir -p logs checkpoints
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --standalone --nproc-per-node=4 \
    -m scripts.train_transformer --config configs/base.json \
    --num-workers 8 --amp bf16 --run-name xfmr_75m \
    2>&1 | tee logs/xfmr_75m.log
```
Артефакт: `checkpoints/transformer_75m/best.pt`.

**125M** (Chinchilla-friendly, ~43 tok/p на 4 эпохах):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --standalone --nproc-per-node=4 \
    -m scripts.train_transformer --config configs/base_125m.json \
    --num-workers 8 --amp bf16 --run-name xfmr_125m \
    2>&1 | tee logs/xfmr_125m.log
```
Артефакт: `checkpoints/transformer_125m/best.pt`.

При крэше — `--resume <checkpoint_dir>/last.pt`.

## 3. Decision heads (~часы)

Учим μ/σ/κ поверх замороженного латента. `configs/base_heads.json` имеет `tau_sec=1.0`, должно совпадать с шагом 1. **Запустить отдельно для каждого трансформера** — в `base_heads.json` нужно прописать соответствующий `transformer_checkpoint` (или пробросить через CLI, если поддерживается).

Если решишь обучать только под один трансформер — выбери лучший по val-loss трансформер из шага 2 и подсунь его в `transformer_checkpoint`.

```bash
# Пример для 75M трансформера (правь transformer_checkpoint в base_heads.json или клонируй конфиг)
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --standalone --nproc-per-node=4 \
    -m scripts.train_heads --config configs/base_heads.json \
    --num-workers 8 --amp bf16 --run-name heads_75m \
    2>&1 | tee logs/heads_75m.log
```

Артефакт: `checkpoints/heads/best_heads.pt`. **Главные метрики в логах:** `IC(α)`, `Spearman(σ)`, `Spearman(κ)`, `α_pred_std`. Если `IC(α) ≈ 0` и `α_pred_std → 0` — α-голова коллапсировала, в бэктесте её можно игнорировать.

## 4. Бэктесты — оба варианта на одном окне

Сравниваем на **SBER, 2024-03-22, минуты 60–120**.

```bash
# Baseline: A-S с константами
PYTHONPATH=. uv run python -m scripts.backtest_orig_as \
    --instrument SBER --date 2024-03-22 --interval 60-120

# Наша модель
PYTHONPATH=. uv run python -m scripts.backtest_our_as \
    --transformer-checkpoint checkpoints/transformer/best.pt \
    --heads-checkpoint checkpoints/heads/best_heads.pt \
    --instrument SBER --date 2024-03-22 --interval 60-120
```

Результаты: `runs/backtest_orig_as/SBER_2024-03-22_060-120/summary.json` и `runs/backtest_our_as/SBER_2024-03-22_060-120/summary.json`. Ключевые поля для сравнения: `pnl`, `n_fills`, `mean_position`, `max_abs_position`.

## Подводные камни

- **LR vs число GPU**: `configs/base.json` имеет `lr=3e-4` под 1 GPU. На 4 GPU эффективный батч ×4. При нестабильном loss в первые шаги — снизить до `1e-4`; если кривая слишком плоская — поднять до `6e-4`.
- **OOM** → уменьшить `batch_size` в конфиге (64 → 32) или `context_length` (1024 → 512).
- **`num_workers` ≥ 8** — иначе DataLoader станет горлышком.
- **bf16** бесплатный на H100 — оставлять.
- **hftbacktest npz** должны лежать в `data/hftbacktest/{SECCODE}/{date}.npz`. Если их нет: `uv run python -m src.data.moex_to_hftbacktest --output-dir data/hftbacktest`.
