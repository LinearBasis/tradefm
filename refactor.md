# Refactor / Critique Notes

Замечания по текущему коду, отсортированы по серьёзности.

## Критичное (баги, влияющие на корректность)

### 1. `target_risk` может быть 0 → взрыв лосса
`src/decision/targets.py:37-46`: если в окне `[t, t+τ)` мид не меняется (типично для L1-стакана между событиями), `rv=0`. В `src/decision/heads.py:88-91` это идёт под `log(0+1e-8) ≈ -18.4`, а предсказание ≈ `log(softplus(0.5)) ≈ -0.03`. Один такой пример даёт MSE ≈ 340 — gradient outlier, и таких событий могут быть тысячи.

**Fix:** клиппинг снизу (`max(rv, eps_floor**2)` с `eps_floor` ~ tick/price), либо отдельная маска валидности «было хотя бы одно движение мида».

### 2. Магическая константа времени в backtest_our_as.py
`scripts/backtest_our_as.py:33,209`: `SECONDS_FROM_MIDNIGHT_AT_T_FIRST = 36000.0` (10:00:00) добавляется к `(now - t_first)/1e9`. Предполагает, что первое событие в npz было ровно в 10:00:00 от полуночи. На самом деле `t_first = exch_ts[0]` (ns от epoch), а первое событие после фильтрации в `moex_to_hftbacktest` может быть в любой момент сессии.

**Fix:** считать настоящий оффсет «seconds-from-midnight на t_first» из npz, не подставлять `36000`.

### 3. `inst_id` извлекается из runtime-parquet, а не из manifest
`scripts/backtest_our_as.py:108`:
```python
inst_id = sorted(pl.read_parquet(args.tokens_parquet)["SECCODE"].unique().to_list()).index(args.instrument)
```
Во время обучения mapping строится из `manifest.json["instruments"]` (`src/data/dataset.py:49`). Если набор тикеров в runtime-файле отличается — embedding `instrument_emb` ткнётся в другой ID молча.

**Fix:** читать `manifest.json` из директории чекпойнта или зашить мэппинг в `ck_xfmr`.

### 4. Fill price загрязнён комиссией
`scripts/backtest_orig_as.py:166-176` и `scripts/backtest_our_as.py:182-188`:
```python
px = -delta_bal / delta_pos
```
`delta_balance = -px*qty - fee`, поэтому `-delta_bal/delta_pos = px + fee/qty`.

**Fix:** считать `px = -(delta_bal + delta_fee) / delta_pos` (использовать `state_values.fee` отдельно).

### 5. Не проверяется `pipeline.tau_sec == heads.tau_sec`
Комментарии в `src/config.py:68,117` это требуют, но `manifest.json` не записывает `tau_sec` (`src/data/pipeline.py:214-228`). Запустишь heads-trainer с `tau_sec=5` на parquets с `tau_sec=1` — heads тихо выучатся на чужой горизонт.

**Fix:** записать `tau_sec` (и `session_length`) в манифест и валидировать на загрузке.

## Высокое (хрупкость / drift hazard)

### 6. Дублирующиеся поля между ModelConfig, HeadConfig, PipelineConfig
`src/config.py`:
- `d_model` в `ModelConfig:30` и `HeadConfig:74`
- `tau_sec` в `HeadConfig:70` и `PipelineConfig:118`
- `vocab_size` статический в `ModelConfig:40`, но вычислимый в `PipelineConfig.vocab_size`
- `n_liquidity_bins` в `ModelConfig:42` и `PipelineConfig:125`
- `val_days` в `ModelConfig:53` и `PipelineConfig:130` (комментарий честно говорит «mirrors»)
- `session_length_sec=30840.0` в `HeadConfig:71` дублирует `continuous_end_sec - continuous_start_sec` из `PipelineConfig:105-106`

**Fix:** производные поля брать из манифеста или derived-property, без второй копии.

### 7. Хардкод d_context=32 против переменного d_model
В `configs/base_125m.json` (d_model=768) и `configs/base.json` (d_model=512) `d_context=32` — контекстная projection `nn.Linear(3*32, d_model)` (`src/models/transformer.py:121`) пропускает 96 в 768. Узкое горлышко.

**Fix:** `d_context = d_model // 8` или поднять до `d_model // 4`.

### 8. Котировки A-S при текущих дефолтах ставятся далеко в книгу
`scripts/backtest_*_as.py`:
```python
bid_tick = min(round_to_tick(bid_px, tick), bbt)
ask_tick = max(round_to_tick(ask_px, tick), bat)
```
Гарантирует postOnly, но при `gamma=0.1, κ=1` теоретический half_spread = `(1/0.1)*log(1+0.1/1) ≈ 0.953` (≈ 95 тиков для tick=0.01) → котировки далеко в книгу, почти не торгуются. Дефолты подобраны для иллюстрации, не для рынка.

**Fix:** пересчитать дефолты под реальный спред SBER, логировать `(bid_tick - bbt, bat - ask_tick)`.

### 9. NaN-маскинг в датасете завязан только на target_alpha
`src/data/dataset.py:103`: `valid = ~torch.isnan(t[:, 0])`. Если alpha валиден, а intensity — нет (например, `daily_volume<=0`), маска пропустит NaN.

**Fix:** `valid = ~torch.isnan(t).any(dim=1)`.

## Средний приоритет

### 10. `i_end - i_start < 2` — слишком мягкий гейт
`scripts/backtest_our_as.py:213`. С контекстом 2 токенов transformer выдаёт шум. Требовать хотя бы `ctx//4` или фиксированный warmup.

### 11. Лимит инвентори односторонний
`scripts/backtest_*_as.py`: при `position == max_inventory` `place_bid=False`, но между `cancel` и снятием в бирже есть `latency_ms` — за это время фил возможен и пробивает max_inventory.

### 12. `_safe_digitize` молча клипает
`src/data/tokenizer.py:178-181`. Out-of-distribution детектится только по бин-индексу 0/n-1.

**Fix:** логировать долю клипов в манифесте для мониторинга drift.

### 13. `extract_hidden_states` имеет `@torch.no_grad()`
`src/models/transformer.py:233`. Закрывает fine-tune трансформера через heads (Stage 2 RL/DiffQuant). Когда дойдёшь до fine-tune — снять декоратор или сделать отдельный `forward_to_layer`.

### 14. `_process_day` повторно фильтрует df по SECCODE
`src/data/pipeline.py:94-99` делает `orders.filter(SECCODE==x)` и `df.filter(SECCODE==x)` — два полных скана по дню. На 7.6M строк заметно.

**Fix:** `df.partition_by("SECCODE")` или `group_by` один раз.

### 15. compute_intensity_targets использует одно tau_sec для нормализации
`src/decision/targets.py:84-85`. При `τ=0.5` шум доминирует. Имеет смысл считать интенсивность на отдельном (более длинном) окне.

## Мелкое

- `src/decision/avellaneda_stoikov.py:30-31` не валидирует `kappa>0` / `sigma>0` для внешних вызовов.
- `flatten_min_before_end > interval_duration` (`scripts/backtest_our_as.py:122`) — нет гарда, t_flatten уйдёт в прошлое.
- `scripts/backtest_our_as.py:108` повторно читает весь `tokens_parquet`, чтобы извлечь список тикеров — нужно один раз.
- `tradefm2.zip.txt` в untracked: похоже на крупный артефакт; добавить в .gitignore?
- `SwiGLU` (`src/models/transformer.py:79`) накладывает dropout ПОСЛЕ `down`-projection; в Llama — до. Минор, не баг.
