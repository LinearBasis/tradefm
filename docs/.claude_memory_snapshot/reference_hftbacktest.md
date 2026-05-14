---
name: hftbacktest L3 reference
description: hftbacktest L3 (market-by-order) event format, MOEX OrderLog mapping, edge cases for the converter
type: reference
---

Backtesting library: github.com/nkaz001/hftbacktest
Docs: hftbacktest.readthedocs.io
We use **L3 mode** (market-by-order), not L2.

## Event format (npz structured array, 8 fields)

```
ev        u64  — event flags (type | side via bitwise OR)
exch_ts   i64  — exchange timestamp (nanoseconds)
local_ts  i64  — local timestamp = exch_ts + latency
px        f64  — price
qty       f64  — quantity
order_id  u64  — order ID (required for L3 to link ADD → CANCEL/FILL)
ival      i64  — reserved
faval     f64  — reserved
```

## L3 event constants

- `ADD_ORDER_EVENT` — new order placed
- `CANCEL_ORDER_EVENT` — order cancelled
- `MODIFY_ORDER_EVENT` — order modified (NOT used for MOEX, see edge cases)
- `FILL_EVENT` — execution
- `DEPTH_EVENT` — L2 depth update (not used in L3 mode for our purposes)
- `DEPTH_CLEAR_EVENT` — clear orderbook
- `BUY_EVENT` / `SELL_EVENT` — side flags, OR'd into `ev`

Side check pattern: `if ev & BUY_EVENT == BUY_EVENT`.

## MOEX OrderLog → hftbacktest mapping

| MOEX `ACTION` | hftbacktest event |
|---|---|
| 1 (new) | `ADD_ORDER_EVENT` |
| 0 (cancel) | `CANCEL_ORDER_EVENT` |
| 2 (trade) | `FILL_EVENT` |

Side: `BUYSELL` field → `BUY_EVENT` / `SELL_EVENT` flag in `ev`.
Order ID: `ORDERNO` → `order_id` field directly.
Price/Volume: `PRICE` → `px`, `VOLUME` → `qty`.
Time: MOEX `TIME` is HHMMSSXXXXXX format → convert to ns epoch for `exch_ts`.

## Edge cases / converter requirements

1. **No MODIFY in MOEX** — every modification is cancel + new with new ORDERNO. Don't emit MODIFY_ORDER_EVENT, just emit CANCEL then ADD. This is fine.

2. **Open/close auction** — synthetic events at session edges (e.g. 39K rows with TIME=100000000000 = 10:00:00.000000 sharp). Skip or mark separately, do not include in train/test.

3. **No receive timestamps in MOEX** — `local_ts = exch_ts + 10ms` synthetic latency (industry standard, same as Smirnov used).

4. **Trade events emit two FILLs** — MOEX trade row references both buy and sell ORDERNO. Emit two FILL_EVENT rows with respective order_ids and side flags.

5. **Initial book state** — OrderLog starts from empty book and grows. hftbacktest L3 handles this, but skip first ~minutes of warmup before computing metrics.

6. **Per-instrument tick size** — set in hftbacktest config, not derived from data. Examples: ОФЗ ~0.001, акции 0.01–1. Specify per asset.

## Validation procedure for the converter

1. Convert one MOEX day → npz.
2. Load in hftbacktest L3 → reconstruct orderbook.
3. Compare reconstructed `mid_price` time series with `eda_time.ipynb` analysis. Should match within tick precision.

## Why L3 not L2 for us

MOEX gives us per-order events. Aggregating to L2 (DEPTH_EVENT with price-level sizes) would lose information that the policy could use. L3 keeps the full granularity that matches our training data.
