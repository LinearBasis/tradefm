---
name: Data scope and plans
description: Current data is one MOEX trading day (7.6M rows, 296 instruments). More days and other markets coming later.
type: project
---

Currently have one trading day of MOEX order flow data (~7.6M rows, 296 instruments, ~400MB CSV).
More days and other markets will be added later.

**Why:** This is a diploma project building up incrementally — start with one day, scale later.
**How to apply:** Design data pipeline and features to be market/day-agnostic from the start. Don't hardcode assumptions about single-day or single-market. But implement and debug on current data only.
