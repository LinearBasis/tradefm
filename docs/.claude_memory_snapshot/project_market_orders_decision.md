---
name: Market orders dropped in MOEX → hftbacktest converter
description: Decision to drop all rows tied to market orders (ACTION=1 with PRICE=0) when converting MOEX OrderLog to hftbacktest L3 npz, with measured impact
type: project
---

Market orders cannot map cleanly to hftbacktest L3 ADD_ORDER_EVENT (no price level). The converter drops:
- the ACTION=1 row with PRICE=0 (the market order's own ADD)
- all ACTION=2 rows for that same ORDERNO (its own-side fills, also have PRICE=0)
- the optional ACTION=0 row for that ORDERNO if remainder got cancelled

Paired ACTION=2 rows on the **resting** side (different ORDERNO, real PRICE) are kept as FILL_EVENT — they carry the actual price discovery and aggression direction.

**Why:** A market order with PRICE=0 inserted as an L3 ADD goes to a phantom price level (0) and corrupts mid/microprice in some hftbacktest depth implementations. There is no clean L3 representation for "consume top of book in one event". Resting-side FILLs preserve the full information needed for our MM strategy.

**Measured impact on our selected top-20 instruments (5 days, 2024-03-18 to 2024-03-22):**
- Aggregate: 0.55% of ACTION=1 are market orders
- Worst offender: ROSN (4.38%), then LKOH (2.41%), TCSG (2.35%)
- 0 instruments above 5% threshold
- LQDT (8.2%), RNFT (5.6%) — filtered out by min_events_per_day, not in top-20

**How to apply:**
- When writing/maintaining `src/data/moex_to_hftbacktest.py`, implement two-pass: pass 1 collects ORDERNO set with PRICE=0 ADDs, pass 2 emits all rows except those with collected ORDERNOs.
- If Exp 0/1 backtests show anomalies (low fill rate, weird OFI) specifically on ROSN — first hypothesis is residual market-order-removal effect; check by re-running on LKOH/SBER for comparison.
- TRUR is an outlier (almost no ACTION=0, near-zero cancels) — bond ETF with one MM. Watch in head eval.

Source script: `scripts/count_market_orders.py`.
