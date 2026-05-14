---
name: Raw data format
description: MOEX OrderLog CSV format — columns, ACTION semantics (official docs), TIME encoding, trade pairing
type: reference
---

Official MOEX OrderLog format (source: https://fs.moex.com/f/3198/specifikacija-formata-dannyh.pdf, https://www.moex.com/ru/orders)

Columns: NO, SECCODE, BUYSELL, TIME, ORDERNO, ACTION, PRICE, VOLUME, TRADENO, TRADEPRICE

**ACTION values (official):**
- **0 = Cancel** (снятие заявки) — order removed from book. VOLUME = remaining unfilled volume.
- **1 = New order** (постановка заявки) — new order placed. VOLUME = visible order size.
- **2 = Trade** (сделка) — execution. VOLUME = trade volume. TRADENO and TRADEPRICE are populated. Each TRADENO appears exactly 2 rows (buy + sell sides). One ORDERNO can have multiple ACTION=2 rows (partial fills).

**No modify action exists** — to change an order, participant cancels (0) and places new (1).

**Other fields:**
- BUYSELL: B=buy, S=sell
- TIME: HHMMSSXXXXXX (6-digit microseconds), e.g. 184459475316 = 18:44:59.475316
- ~40K rows have TIME=100000000000 — pre-market orders with default 10:00:00.000000 timestamp
- Trading session: 10:00–18:45

**Proportions in our data:** ~55% new (ACTION=1), ~35% cancel (ACTION=0), ~10% trade (ACTION=2)

**Why:** Correct interpretation of ACTION is critical for feature engineering and LOB reconstruction.
**How to apply:** ACTION maps directly to TradeFM's add/cancel. Trades (ACTION=2) provide execution prices for mid-price estimation (EW-VWAP). Cancel+new pairs may represent order modifications.
