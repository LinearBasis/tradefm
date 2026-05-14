---
name: hftbacktest L3 converter — built and validated
description: Status of the MOEX OrderLog → hftbacktest L3 npz converter and the bug that almost slipped through
type: project
---

`src/data/moex_to_hftbacktest.py` produces per-(SECCODE, day) npz files at `data/hftbacktest/{SECCODE}/{YYYY-MM-DD}.npz` with key `"data"` (hftbacktest convention).

Pipeline (5 days × 20 instruments = 349M events total):
- 2024-03-18: 47.8M, 2024-03-19: 78.4M, 2024-03-20: 94.4M, 2024-03-21: 60.6M, 2024-03-22: 67.9M

Validation (`scripts/validate_hftbacktest_npz.py`) passed on SBER + GAZP for 2024-03-18:
- exch_ts monotonic, 0 orphan cancels/fills, 0 duplicate ADDs
- median spread 0.33 bps (SBER), 1.24 bps (GAZP) — sane
- mid corr vs trade-VWAP: 0.9996 / 0.998
- RMSE vs VWAP: 0.5–0.9 bps (within bid-ask)
- Small caveat: ~50–300 momentary crossed-book samples per day per instrument (≤6% of samples). Likely batch auction / paired-fill ordering artifacts; not a converter bug, since 0 orphan IDs.

**Bug almost slipped through:** the original implementation built `ev` via Polars bitwise OR
of `pl.lit(int(EXCH_EVENT))`. Polars promoted to Int32, which overflows for `EXCH_EVENT=0x80000000`,
silently produces NULL, and `astype(np.uint64)` converts NULL → 0. Result: every `ev` was 0.
The only signal was a `RuntimeWarning: invalid value encountered in cast` — easy to ignore.

**Why:** safety check before assuming the converter works. Always inspect `ev` distribution
on a sample npz after any change to `_compute_ev_numpy`.

**How to apply:**
- ev computation now happens in numpy after `.collect()`, never in Polars expressions.
- If you ever modify the bitfield builder, re-run `validate_hftbacktest_npz` on at least
  one (instrument, day) before declaring success — validate that types & flags decode correctly.

**Validation checks the npz contains:**
- correct dtype matching `hb.event_dtype` (note field is `fval`, not `faval`)
- monotonic exch_ts, local_ts ≥ exch_ts
- ID hygiene: every CANCEL/FILL references a known ADD
- L3 reconstruction produces a non-crossed book at sampled times
- mid corr/RMSE against rolling-VWAP from raw CSV
