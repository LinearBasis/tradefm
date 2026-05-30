# TradeFM + RL Market-Making

В ветке `master` лежит foundation-модель TradeFM. В ветке `rl-market-making`
к ней добавлен пайплайн RL-маркет-мейкинга с интеграцией скрытого
представления TradeFM в state RL-агента (дипломная работа).

## Структура

```
src/, scripts/, configs/   — TradeFM (как на master)
simulator/                 — событийный симулятор биржевого стакана
rl/                        — RL-агент, среды, драйверы экспериментов
utils/                     — подготовка данных (L3 → L2 → MD-поток → эмбеддинги)
```

## TradeFM (master-часть)

Decoder-only transformer на MOEX OrderLog. Pipeline:

```bash
uv sync

# Токенизация
uv run python -m src.data.pipeline \
    --raw-dir data/raw --output-dir data/processed \
    --top-n-instruments 20 --n-jobs 8 --polars-threads 4

# Обучение (multi-GPU DDP)
uv run torchrun --standalone --nproc-per-node=8 \
    -m scripts.train_transformer --config configs/base_50m.json \
    --num-workers 8 --amp bf16

# Оценка
uv run python -m scripts.eval_rollout \
    --checkpoint checkpoints/transformer_50m/best.pt \
    --tokenizer data/processed/tokenizer.json \
    --val-sequences data/processed/sequences \
    --all-instruments --n-rollouts 10 --n-events 1024 \
    --output runs/eval/xfmr_50m
```

Конфиги: `configs/base_50m.json` (52M, production), `configs/base_20m.json`
(19M, fast iteration), `configs/smoke.json` (CPU smoke).

## RL Market-Making

Шесть конфигураций DQN-агента маркет-мейкера на YNDX-данных MOEX
(OrderLog, 22.04.2024):

| № | Конфигурация | QNet | Доп. признаки |
|---|---|---|---|
| A | DQN-Linear           | полносвязный | — |
| B | DQN-Linear-Oracle    | полносвязный | future-mid |
| C | DQN-Attn             | self-attention над уровнями | — |
| D | DQN-Attn-Oracle      | self-attention над уровнями | future-mid |
| E | DQN-Linear-TradeFM   | полносвязный + проекция 384→16 | TradeFM-латент |
| F | DQN-Attn-TradeFM     | self-attention + проекция 384→16 | TradeFM-латент |

### Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-rl.txt
```

### Подготовка данных

```bash
python -m utils.build_l2 \
  --orderlog /path/to/OrderLog20240422.parquet \
  --seccode YNDX --height 10 \
  --output snapshots/YNDX_20240422.npz

python -m utils.convert_to_sim --npz snapshots/YNDX_20240422.npz \
  --parquet /path/to/OrderLog20240422.parquet --seccode YNDX --date 2024-04-22 \
  --snap-offset 0 --snap-length 90000 \
  --out md_streams/YNDX_intraday_20240422_train.pkl

python -m utils.convert_to_sim --npz snapshots/YNDX_20240422.npz \
  --parquet /path/to/OrderLog20240422.parquet --seccode YNDX --date 2024-04-22 \
  --snap-offset 90000 --snap-length 30000 \
  --out md_streams/YNDX_intraday_20240422_test.pkl
```

Для E/F — эмбеддинги TradeFM. Положить чекпойнт и токенизатор в
`checkpoints/best_20m.pt`, `checkpoints/tokenizer.json`:

```bash
python -m utils.extract_tradefm_embeddings \
  --orderlog /path/to/OrderLog20240422.parquet \
  --md md_streams/YNDX_intraday_20240422_train.pkl \
       md_streams/YNDX_intraday_20240422_test.pkl \
  --instrument YNDX --date 2024-04-22 \
  --out-dir embeddings --batch-size 32 --bf16
```

### Запуск

```bash
python -m rl.run_linear       # A + B
python -m rl.run_attention    # C + D
python -m rl.run_tradefm      # E + F  (нужны embeddings/)
```

### Результаты (8 сидов, 22.04.2024 intraday)

| Конфигурация       | mean PnL  | σ      | средн. число сделок |
|---|---:|---:|---:|
| DQN-Linear         | +23 137   | 21 633 | 1 401 |
| DQN-Linear-TradeFM | +27 843   | 18 427 | 1 217 |
| DQN-Linear-Oracle  | +30 611   | 16 033 | 1 391 |
| DQN-Attn           | +30 482   | 16 438 | 1 152 |
| DQN-Attn-TradeFM   | +35 184   | 12 927 | 1 387 |
| DQN-Attn-Oracle    | +38 376   |  9 841 | 1 573 |

Метрика — `pnl_neutral = final_pnl − final_inv·Δmid`.
