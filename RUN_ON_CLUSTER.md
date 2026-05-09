# Quickstart на кластере

Архив содержит **только код** (~190 КБ): `src/`, `scripts/`, `tests/`, `docs/`, `pyproject.toml`, `uv.lock`, `CLAUDE.md`.

Допущения по данным на кластере:
- `data/raw/OrderLog*.txt` — есть (для регенерации L3 npz и tokenized sequences)
- `data/processed/sequences/*.parquet` — есть (после `src.data.pipeline`)

Если что-то отсутствует — см. секцию «Регенерация данных» внизу.

## 0. Распаковка и окружение

```bash
tar xzf tradefm.tar.gz && cd tradefm
uv sync                               # создаст .venv, поставит torch, hftbacktest, polars
uv run python -c "import torch; print('cuda?', torch.cuda.is_available())"
```

## 1. Тренировка трансформера (главная команда — 4×H100)

Гиперпараметры лежат в `configs/cluster.json` (создать перед запуском, см. шаблон в `configs/smoke.json`). Архитектура и ёмкости — там же.

```bash
mkdir -p logs runs
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --standalone --nproc-per-node=4 \
    -m scripts.train_transformer --config configs/cluster.json \
    --num-workers 8 --amp bf16 --run-name xfmr_full \
    2>&1 | tee logs/xfmr_full.log
```

Резюм при крэше: добавить `--resume checkpoints/last.pt`.

LR-скейлинг: `lr` в конфиге подобран под 1 GPU. На 4 GPU эффективный батч ×4 — либо поднять `lr` в конфиге пропорционально, либо снизить `batch_size` в конфиге до 8.

TensorBoard: `uv run tensorboard --logdir runs/ --bind_all`.

Артефакт: `checkpoints/best.pt` (или путь из `checkpoint_dir` в конфиге).

Smoke (на ноуте/CPU, несколько шагов):
```bash
uv run python -m scripts.train_transformer --config configs/smoke.json \
    --device cpu --allow-cpu --amp none --max-steps 30
```

## 2. Регенерация hftbacktest npz (нужно для Exp 1)

Требует `data/raw/OrderLog*.txt`.
```bash
mkdir -p data/hftbacktest
uv run python -m src.data.moex_to_hftbacktest --output-dir data/hftbacktest
```
~10–15 мин на 5 дней × 20 инструментов = 349M событий.

Валидация:
```bash
uv run python -m tests.validate_hftbacktest_npz --instrument SBER --date 2024-03-18
```

## 3. Тренировка decision heads

Multi-horizon ablation (план Exp 0b). Один конфиг на τ — например `configs/heads_tau512.json`. Требует, чтобы pipeline уже разложил targets под это τ (иначе перезапустить пайплайн).

```bash
for TAU in 128 256 512 1024; do
  CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --standalone --nproc-per-node=4 \
      -m scripts.train_heads --config configs/heads_tau${TAU}.json \
      --num-workers 8 --amp bf16 --run-name heads_tau${TAU} \
      2>&1 | tee logs/heads_tau${TAU}.log
done
```

Критерий: IC(α) > 0 значимо хотя бы при одном τ.

## 4. Чек-лист готовности «лучшая модель»

| Этап | Артефакт | Статус-команда |
|------|----------|----------------|
| Full train | `checkpoints/best.pt` | `python -c "import torch; ck=torch.load('checkpoints/best.pt', weights_only=False); print(ck['epoch'], ck['val_loss'])"` |
| Heads × 4 τ | `checkpoints/heads_tau*/best_heads.pt` | grep IC в `logs/heads_tau*.log` |
| L3 npz | `data/hftbacktest/*/2024-03-*.npz` | `ls data/hftbacktest/SBER/` |

## Регенерация данных (если на кластере чего-то нет)

Полный конвейер из сырых OrderLog в tokenized sequences:
```bash
uv run python -m src.data.pipeline                # CSV → features → tokens → sequences
```
~30–60 мин в зависимости от количества дней. Артефакты:
- `data/processed/tokens.parquet`
- `data/processed/sequences/*.parquet` (по одному на (SECCODE, day))

## Подводные камни

1. `bf16` бесплатный на A100/H100, выбран по умолчанию.
2. `num-workers >= 8` важно — иначе DataLoader станет горлышком.
3. Если OOM — снизить `batch_size` или `context_length` в конфиге. Gradient accumulation в `train_transformer.py` пока не реализован.
4. На multi-node нужен `torchrun --rdzv-backend=c10d --rdzv-endpoint=$MASTER:29500 --nnodes=N --node-rank=$NODE_RANK ...` вместо `--standalone`.
5. Старые чекпоинты (до перехода на RoPE 2026-05-09) несовместимы — в них есть `pos_emb.weight`, которого больше нет в архитектуре. Тренировать с нуля.
