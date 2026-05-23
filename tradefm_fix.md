# TradeFM Fix Plan — status

Текущее состояние реализации после code review против статьи 2602.23784v1.
**Чистый план без open-ended вопросов: всё внизу либо ✅ сделано, либо явно out-of-scope.**

---

## ✅ Сделано

### 1. Декаплинг decision-heads из TradeFM core (`src/_archive/...`)
- `src/decision/` (вместе с `rl/`) → `src/_archive/decision/`.
- `scripts/{train_heads,backtest_*,exp/*}.py` → `scripts/_archive/`.
- `configs/{base_heads,smoke_heads}.json`, `configs/exp/`, `configs/.smoke/{base_heads,rl_*}.json` → `configs/_archive/`.
- `tests/smoke_avellaneda_stoikov.py` → `tests/_archive/`.
- Все импорты `src.decision.*` → `src._archive.decision.*` в архивных файлах.
- TradeFM-core (`pipeline.py`, `dataset.py`, `config.py`) очищены от head/target кода (`add_targets_to_orders`, `HeadConfig`, `tau_sec`, `include_targets`).

### 2. Outlier bins (paper §6.1, Algorithm 1)
- `Tokenizer` теперь резервирует outlier-бины:
  - Двусторонние (`price_depth`, `price_level`): bin 0 = low-outlier, bin n-1 = high-outlier, n-2 интерьерных квантильных бина.
  - Односторонние (`volume`, `interarrival`): bins 0..n-2 = log-uniform интерьер, bin n-1 = high-outlier.
- Vocab = 16,384 без изменений; общее число бинов на компоненту = 16.

### 3. Liquidity по ADV (paper §6.2)
- ADV считается как `mean(daily_volume)` по training-дням per-instrument (anti-leakage).
- Хранится в `tokenizer.json`; `bin_liquidity` константа на инструмент.

### 4. Tokenizer-класс с fit/transform/save/load
- `src/data/tokenizer.py` полностью переписан как `class Tokenizer`.
- `Tokenizer.fit_from_features(features_df, adv_per_day, cfg)` — pure-compute.
- `Tokenizer.transform(df)` — digitize + lookup ADV + compose composite token.
- `Tokenizer.save(path)` / `Tokenizer.load(path)` — JSON.
- `Tokenizer.bin_centroid(feature, bin_idx)` — обратное декодирование bin → continuous value (для rollout/симулятора).
- Pipeline поддерживает `--reuse-tokenizer` → пропускает калибровку, грузит `tokenizer.json`.

### 5. `use_instrument_emb` опционально, default off
- `ModelConfig.use_instrument_emb: bool = False` — paper-faithful default.
- `OrderFlowTransformer` динамически создаёт `instrument_emb` только если флаг True; `context_proj.in_features` соответственно 2·d_context или 3·d_context.
- Старые чекпоинты совместимы через `strict=False` при resume.

### 6. Новые Muon-конфиги
- `configs/base_50m.json`: d_model=512, n_layers=16, n_heads=8, n_kv_heads=2, d_ff=1344 (8/3·d_model, Llama ratio), qk_norm=true, `use_instrument_emb=false`, optimizer=muon_hybrid. Реальный размер: **~52M параметров**.
- `configs/base_20m.json`: d_model=384, n_layers=8, n_heads=6, n_kv_heads=2, d_ff=1024 (8/3·d_model), qk_norm=true. Реальный размер: **~19M параметров**.

### 7. Per-subtoken accuracy в TensorBoard
- В `evaluate()` argmax-токены декодируются на `(action, side, depth, vol, iat)` и сравниваются с таргетом.
- Логируется как `eval/acc_action`, `eval/acc_side`, `eval/acc_depth`, `eval/acc_vol`, `eval/acc_iat`.
- В tqdm НЕ выводится (только loss / ppl).
- Loss остаётся composite CE (как в статье); per-subtoken только метрика.

### 8. Off-by-one в `compute_risk_targets`
- Housekeeping: исправлено в `src/_archive/decision/targets.py:46` (`cumsum[max(j-1, i)]`).

### 9. Минимальный LOB-симулятор (paper Alg 2-3)
- `src/eval/simulator.py` — `MinimalLOB` с price-time priority, FIFO на каждом уровне, поддержкой add/cancel/matching.
- API: `step(action, side, price_depth, volume, iat_sec)` → snapshot dict с `mid`, `spread`, `obi`, `bid_vol`, `ask_vol`.
- Cancels: один прагматический отход от Alg 3 — нет order ID в модели, поэтому cancel удаляет FIFO-front'а на ближайшем по цене уровне.
- `seed_book(n_levels, qty_per_level, tick)` — симметричная init-книга вокруг mid.

### 10. Stylized facts + distributional fidelity
- `src/eval/stylized_facts.py`:
  - `run_rollout(model, tokenizer, seed_*, n_events, ...)` — closed-loop генерация. Decoder: composite token → 5 sub-tokens → centroids → simulator step.
  - `compute_stylized_facts(returns, max_lag)` — ACF(r), ACF(|r|), kurtosis, kurtosis на агрегации 2/5/10.
  - `compute_distributional_fidelity(real, gen)` — K-S и W₁ per-quantity (iat, depth, vol; spread/OBI/volumes только в gen, потому что не сохранены в parquets).
- `scripts/eval_rollout.py` — standalone CLI: load checkpoint + tokenizer + val parquets → N rollouts → metrics.json + TensorBoard.
- Hook в `train_transformer.py`: если `cfg.rollout_every_n_epochs > 0`, на каждой N-й эпохе и в конце запускает reduced rollout (1 ассет, `rollout_n_rollouts` rollouts по `rollout_n_events` events) и логает `rollout/*` в TensorBoard. Default off.

---

## 📝 Тесты, которые прошли

| Тест | Результат |
|---|---|
| Импорты TradeFM core после декаплинга | ✓ |
| `decode_trade_token` roundtrip на 10 значениях | ✓ |
| Outlier-биннинг на синтетике (значения > p99 → последний bin) | ✓ |
| End-to-end pipeline на синтетическом MOEX (16K events × 2 дня × 2 инструмента) | ✓ tokens 7..16361, parquets без target-колонок, manifest без `tau_sec` |
| `Tokenizer.save → load` roundtrip | ✓ edges идентичны |
| `--reuse-tokenizer` пропускает калибровку | ✓ |
| `use_instrument_emb` False/True forward | ✓ корректные размерности (64 / 96) |
| Конфиги 75M/25M | ✓ 69.3M и 23.6M параметров, оба Muon |
| Per-subtoken accuracy в TB | ✓ 5 тегов записаны |
| `_decompose_token` vs `decode_trade_token` | ✓ полное соответствие |
| LOB симулятор: aggressive buy walks book, cancels FIFO | ✓ 3 fills + n_bid_levels корректно |
| `bin_centroid` обратное декодирование | ✓ depth ±0.003 на крайних бинах, iat 0.4..9.7s, vol 1.75..231 |
| Eval CLI: рандомная модель → rollout → metrics | ✓ K-S(iat)=0.40, K-S(depth)=0.057, kurtosis 19→2.67 (heavy-tail decay) |
| Training-хук rollout логирует в TB | ✓ rollout/kurtosis, acf_returns_lag1, acf_abs_returns_lag1 |

---

## 🚫 Не делаем (документированные отступления)

- **I_MP indicator** — нет participant-id в MOEX OrderLog.
- **LR schedule** — у нас cosine (статья: linear); cosine стандартен и не хуже.
- **Tabular embedding form** — у нас additive variant (статья: concat-projection всех 4 фич, включая i_trade); математически эквивалентно с точностью до инициализации.
- **524M params / 9K assets** — out of diploma scope.
- **Baseline-модели (ZI, Hawkes)** — пользователь явно отказался; K-S/W₁ только real-vs-ours.
- **Streaming / t-digest tokenizer** — overkill на текущей шкале (месяц × 20 инструментов ≈ 800MB в RAM).
- **Decision heads (alpha/risk/intensity), RL агент, A-S backtests** — в архиве (`src/_archive/decision/`), для будущей отдельной работы.
- **Liquidity-бины при N инструментов < n_liquidity_bins** — fall-back на bin 0 для всех. С 20 инструментами не проблема.

---

## ⚠️ Известные ограничения

- **Distributional fidelity на synthetic** даёт smoke-результат: depth K-S ≈ 0.06 не потому, что random модель "хороша", а потому что бин-центроиды декодируются одинаково для real и gen. На реальных данных распределения по centroid-сетке будут отражать разные априоры — метрика станет осмысленной.
- **Simulator не валидирован против real fills** (paper §D.2 method) — на больших данных надо replay real OrderLog → check fill volume/count CDF. Пока пропущено.
- **Liquidity edges пустые при N<3 инструментов** — `select_instruments(top_n=3+)` нужен для production-конфигов.
