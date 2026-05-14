---
name: Architecture decisions for data pipeline and tokenization
description: Key design decisions for TradeFM adaptation — action types, features, tokenization, instrument selection
type: project
---

## Decided (2026-04-01)

### Action types in model vocabulary
- **add (ACTION=1) и cancel (ACTION=0)** — предсказываемые действия, 2 значения
- **trade (ACTION=2)** — НЕ входит в предсказываемую последовательность. Используется только для оценки mid-price (EW-VWAP). Сделки — детерминированное следствие matching engine, а не самостоятельные действия участников

### Scale-invariant features (predicted, входят в trade token)
- **∆t** — interarrival time (секунды) между событиями add/cancel
- **δp** — (PRICE - p_mid) / p_mid — normalized price depth
- **v** — log(1 + VOLUME)
- **side** — B/S (2 значения)
- **action** — add/cancel (2 значения)

### Contextual features (не предсказываются, conditioning)
- **∆p** — (p_mid - p_open) / p_open — intraday price level
- **liquidity bin** — low/mid/high по дневному объёму

### Архитектура должна позволять легко добавлять новые фичи

### Tokenization
- 16 бинов на continuous фичу (∆t, δp, v)
- Price-related → quantile (equal-frequency) bins
- Log-transformed (volume, ∆t) → equal-width bins в лог-пространстве
- Outliers за p1/p99 → специальные бины
- Composite token через mixed-base: vocabulary = 2 × 2 × 16 × 16 × 16 = 16,384
- Калибровка бинов на имеющихся данных (пока 1 день)

### Mid-price estimation
- EW-VWAP по TRADEPRICE из ACTION=2
- Halflife — гиперпараметр (подбираем позже)
- Без реконструкции LOB на первом этапе

### Instrument selection
- Начинаем с топ-10-20 ликвидных инструментов (по числу событий в день)
- Параметризовано — можно убрать ограничение и взять все 296

### Sequence construction
- Per-asset последовательности
- Train/val split строго по времени (не random)

**Why:** Адаптация TradeFM под наши MOEX данные. Формат OrderLog напрямую ложится на их add/cancel схему.
**How to apply:** Использовать при реализации data pipeline, feature engineering и tokenizer.
