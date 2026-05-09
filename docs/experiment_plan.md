# План экспериментов: Decision Making для Market Making

## Context

Pretrained autoregressive transformer на order flow (325M токенов, 20 инструментов MOEX).
Три supervised головы поверх frozen latents: alpha (μ), risk (σ), intensity (κ).
Цель — decision module для market making (заработок на спреде).

Прогрессия экспериментов от простейших до продвинутых, с использованием идей
из DiffQuant (дифференцируемый симулятор, hybrid loss).

**Симулятор:** hftbacktest в **L3 mode** (market-by-order). Маппинг MOEX OrderLog
ACTION 1/0/2 → ADD_ORDER_EVENT / CANCEL_ORDER_EVENT / FILL_EVENT — без потерь.
Альтернатива (ESkripichnikov) отвергнута: нет queue position → завышенный fill rate.

---

## Experiment -1: Инфраструктура (до Exp 1)

**Цель:** конвертер MOEX OrderLog → hftbacktest L3 npz.

- `src/data/moex_to_hftbacktest.py` — читает OrderLog CSV, выдаёт npz с полями
  `(ev, exch_ts, local_ts, px, qty, order_id, ival, faval)`
- `ev` = `(ADD|CANCEL|FILL)_ORDER_EVENT | (BUY|SELL)_EVENT` через bitwise OR
- `local_ts = exch_ts + 10ms` (синтетический latency, у MOEX нет receive_ts)
- TIME (HHMMSSXXXXXX) → ns epoch
- Skip open/close auction events (TIME=100000000000 и аналогичные)
- Для каждого FILL: эмитировать **два** события (buy-side и sell-side order_id)

**Validation:** загрузить npz в hftbacktest L3, реконструировать стакан, сравнить
mid_price с тем, что считали в EDA (`eda_time.ipynb`). Должно совпадать.

---

## Experiment 0: Head Quality Evaluation

**Цель:** убедиться, что головы предсказывают осмысленно, прежде чем строить стратегию.

### 0a: Базовые метрики качества

- Alpha: IC (information coefficient) = корреляция предсказания с реальным return
- Risk: корреляция log(σ_pred) с log(σ_real)
- Intensity: корреляция log(κ_pred) с log(κ_real)
- Baseline: наивное предсказание = среднее по train set

### 0b: Multi-horizon ablation

Сравнить качество голов при разных τ: 128, 256, 512, 1024 событий.
Alpha может работать лучше на коротком горизонте (больше SNR),
risk — на длинном (стабильнее). Выбрать оптимальный τ per head.

### 0c: Визуализация

- Scatter plots: predicted vs actual
- Time series: предсказания поверх реальных значений (один инструмент, один день)
- Распределение ошибок

**Файлы:** `notebooks/eval_heads.ipynb`

**Критерий перехода:** IC(alpha) > 0 статистически значимо, R² > 0 для risk и intensity.

**Fallback если головы не работают:**
1. Unfreeze transformer, fine-tune end-to-end с head losses
2. Если и это не помогает — перейти сразу к Exp 3 (end-to-end), минуя supervised головы

---

## Experiment 1: A-S Formula Baseline

**Цель:** классический Avellaneda-Stoikov с предсказаниями голов, оценка в hftbacktest (L3).

**Формула:**
```
reservation_price = mid + μ − γ * q * σ² * τ
half_spread = (1/γ) * ln(1 + γ/κ)
bid = reservation_price − half_spread
ask = reservation_price + half_spread
```

### 1a: Grid search

- γ (risk aversion): [0.01, 0.1, 1.0, 10.0]
- max_inventory: [10, 50, 100]

### 1b: Ablation — вклад каждой головы

Комбинации (для диплома — показать, что каждая голова добавляет value):

| Конфигурация | μ | σ | κ |
|---|---|---|---|
| Constant baseline | 0 | empirical avg | empirical avg |
| Only alpha | predicted | empirical avg | empirical avg |
| Only risk | 0 | predicted | empirical avg |
| Only intensity | 0 | empirical avg | predicted |
| Alpha + risk | predicted | predicted | empirical avg |
| All three | predicted | predicted | predicted |

### 1c: Baselines

- Symmetric MM: bid = mid − δ_fixed, ask = mid + δ_fixed (без формулы вообще)
- A-S с константами (формула без голов)

**Метрики:** PnL, Sharpe, max drawdown, avg inventory, fill rate, avg spread

**Файлы:**
- `src/decision/avellaneda_stoikov.py` — A-S стратегия (под hftbacktest API, не ESkripichnikov)
- `scripts/backtest_as.py` — запуск в hftbacktest L3
- `notebooks/backtest_analysis.ipynb` — анализ

**Критерий перехода:** A-S с головами > A-S с константами по Sharpe.

---

## Experiment 2: Reward-Weighted Policy Learning

**Цель:** заменить фиксированную A-S формулу на learnable MLP, обученный
на реализованном PnL из hftbacktest L3 эпизодов (не на выходах A-S).

**Почему не supervised на A-S:** MLP имитирующий A-S не может превзойти A-S.
Обучение на реальном PnL позволяет найти нелинейности, которые формула не ловит.

**Архитектура:**
```
[μ, σ, κ, q, q², time_in_session] → MLP(64, 32) → (δ_bid, δ_ask) × gate
```
- δ_bid, δ_ask > 0 (softplus)
- gate ∈ [0, 1] (sigmoid, bias=-1 как в DiffQuant) — "не котировать" когда не уверен

**Обучение (reward-weighted regression):**
1. Собрать эпизоды из Exp 1 (A-S с разными γ) + случайные вариации
2. Для каждого эпизода есть realized PnL
3. Loss = -Σ advantage_i × log π(a_i | s_i), где advantage = PnL - baseline

Это по сути policy gradient без симулятора в цикле обучения.

**Файлы:**
- `src/decision/policy_mlp.py` — MLP policy с gate
- `scripts/train_policy.py` — обучение

**Критерий перехода:** MLP policy > лучший A-S из Exp 1 по Sharpe.

---

## Experiment 3: Differentiable MM Simulator (DiffQuant-style)

**Цель:** end-to-end оптимизация Sharpe через дифференцируемый симулятор.

**Важно:** hftbacktest **не дифференцируем** (Numba/Rust бэкенд, дискретные queue
операции). Diff-sim — отдельная инфраструктура **для обучения**. Финальная оценка
обученной policy всё равно происходит **в hftbacktest L3** (паттерн "train on
diff-sim, evaluate on hftbacktest" — аналог sim-to-real в робототехнике).

### Дифференцируемый симулятор

```python
# На каждом шаге t:
bid_t = mid_t - softplus(δ_bid_t)
ask_t = mid_t + softplus(δ_ask_t)

# Fill model (дифференцируемая аппроксимация)
p_bid_t = fill_model(bid_t, mid_t, order_flow_t)
p_ask_t = fill_model(ask_t, mid_t, order_flow_t)

# PnL
inventory_t = inventory_{t-1} + fill_bid_t - fill_ask_t
pnl_t = fill_ask_t * (ask_t - mid_t) - fill_bid_t * (mid_t - bid_t)
       + inventory_t * Δmid_t  # mark-to-market
```

### Fill model — ключевой компонент

Sigmoid-аппроксимация слишком груба для MM. Варианты:

**Вариант A: Learned fill model**
Обучить отдельную нейросеть предсказывать P(fill | δ, order_flow features)
на исторических данных (из hftbacktest). Зафиксировать при обучении policy.

**Вариант B: Parametric fill model**
```
P(fill в dt) = κ · exp(-α · δ) · dt
```
где κ и α калибруются из данных. Дифференцируемо по δ.

**Вариант C: Empirical fill curve + soft interpolation**
Построить эмпирическую кривую fill rate vs distance, аппроксимировать
дифференцируемым сплайном.

**Вариант D: Surrogate fill model на данных из hftbacktest**
1. Прогнать рандомизированные стратегии (разные γ, размеры, частоты) в hftbacktest L3
2. Собрать датасет `(state, δ, was_filled, fill_size)` из реалистичной queue model
3. Обучить маленькую NN `P(fill | δ, features)` — **дифференцируемая по построению**
4. Использовать как fill model в diff-sim для обучения policy через ∇Sharpe

Это самый честный вариант: fill model "знает" про queue position через данные
из hftbacktest, но остаётся дифференцируемой. Закрывает sim-to-real gap.

Рекомендация: B как стартовый бейзлайн → D как основной (если B хуже hftbacktest baseline).

### Policy

```
[latent, μ, σ, κ, inventory, inventory²] → MLP → (δ_bid, δ_ask) × gate
```
Инициализация весами из Exp 2.

### Hybrid Loss (адаптация DiffQuant)

```
L = -λ₁·Sharpe
  + λ₂·turnover_penalty           # smooth_abs(Δδ) — штраф за дёргание котировок
  + λ₃·log_drawdown               # log(1 + max_drawdown) — контроль просадки
  + λ₄·inventory_bias_penalty     # |mean(inventory)| → 0
  + λ₅·spread_floor_penalty       # min(spread) > min_allowed
```

**Из DiffQuant:**
- `smooth_abs(x) = sqrt(x² + ε)` — градиенты через ноль
- Gate mechanism с bias=-1 (старт в flat)
- Warm-up: начинаем с большими λ₂..λ₅, постепенно увеличиваем λ₁
- Без регуляризаций — коллапс (гиперактивность → flat → bias, как в статье)

**Файлы:**
- `src/decision/fill_model.py` — модель исполнения (варианты A/B/C/D)
- `src/decision/diff_simulator.py` — дифференцируемый MM симулятор
- `src/decision/policy_e2e.py` — end-to-end policy
- `scripts/train_e2e.py` — обучение
- `scripts/eval_in_hftbacktest.py` — финальная оценка policy в hftbacktest L3

**Критерий перехода:** E2E policy > A-S baseline по Sharpe **в hftbacktest** (val set).

---

## Experiment 4: RL Fine-tuning в hftbacktest (SAC)

**Цель:** если дифференцируемый симулятор слишком упрощает механику рынка,
RL в более реалистичном симуляторе.

**Setup:**
- Environment: hftbacktest L3 → gym-like interface
- State: `[latent(128), μ, σ, κ, inventory, unrealized_pnl, time_remaining]`
- Action: `(δ_bid, δ_ask, bid_size, ask_size)` — continuous
- Reward: `realized_pnl_step − λ·inventory²`

**Инициализация:** весами из Exp 3 (не с нуля!).

**Алгоритм:** SAC — continuous action space, entropy regularization.

**Файлы:**
- `src/decision/rl_env.py` — gym wrapper
- `src/decision/rl_policy.py` — SAC policy
- `scripts/train_rl.py` — обучение

---

## Сводная таблица

| # | Experiment | Что оптимизируем | Fill model | Симулятор для eval | Сложность |
|---|-----------|-----------------|------------|--------------------|-----------|
| -1 | Конвертер OrderLog → npz | — (инфра) | — | — | Низкая |
| 0 | Head eval + ablation | — (диагностика) | — | — | Низкая |
| 1 | A-S + head ablation | γ (grid search) | hftbacktest L3 | hftbacktest L3 | Низкая |
| 2 | MLP policy | Weights (reward-weighted) | hftbacktest L3 | hftbacktest L3 | Средняя |
| 3 | Diff simulator | Policy (Sharpe E2E) | Variant B/D (свой diff-sim) | hftbacktest L3 | Высокая |
| 4 | RL fine-tuning | Policy (SAC) | hftbacktest L3 | hftbacktest L3 | Высокая |

---

## Зависимости и риски

**Данные:** сейчас 5 дней — достаточно для Exp 0-1, пограничноно для Exp 2,
мало для Exp 3-4. Нужно подключить больше данных до Exp 3.

**Каждый эксперимент строится на предыдущем.** Если на этапе N нет улучшения —
останавливаемся и анализируем. Исключение: если головы не работают (Exp 0 fail),
пробуем fallback (fine-tune transformer или skip to Exp 3).

**Fill model** (Exp 3) — главный research risk. Реалистичность дифференцируемого
fill model определяет, насколько результаты Exp 3 переносятся в реальность.

---

## Порядок реализации

0. **Exp -1**: конвертер MOEX OrderLog → hftbacktest L3 npz + валидация на одном дне
1. Обучить головы → Exp 0 (eval notebook, multi-horizon ablation)
2. Exp 1 (A-S + ablation в hftbacktest L3)
3. Подключить больше данных MOEX (>20 дней)
4. Exp 2 (MLP policy, reward-weighted в hftbacktest L3)
5. Exp 3 (diff simulator) — основная ставка, eval в hftbacktest L3
6. Exp 4 (RL) — только если необходимо
