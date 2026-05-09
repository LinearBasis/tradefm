"""Pre-flight validation of MOEX OrderLog data integrity before conversion to
hftbacktest L3 npz.

Runs per-file checks:
  1. TRADENO pairing — every TRADENO must appear exactly twice (one B, one S)
  2. Volume invariant per ORDERNO — ADD VOLUME == sum(FILL VOLUMEs) + CANCEL VOLUME (if any)
     (also empirically resolves the iceberg question: if invariant always holds,
      MOEX does not reuse ORDERNOs for hidden volume → icebergs use separate ORDERNOs)
  3. FILL ORDERNO coverage — every ORDERNO in ACTION=2 must have a corresponding ACTION=1
  4. CANCEL ORDERNO coverage — every ORDERNO in ACTION=0 must have a corresponding ACTION=1
  5. Opening auction window — distribution of TIME in [10:00, 10:10) to understand
     how big the open-batch is and where continuous trading starts

Filters: only checks the top-N instruments selected by PipelineConfig (the ones
we will actually convert).

Run: uv run python -m tests.validate_orderlog
"""

from pathlib import Path

import polars as pl

from src.config import PipelineConfig
from src.data.loader import (
    _CSV_DTYPES,
    discover_files,
    extract_date_from_filename,
    select_instruments,
)


# ---------- helpers ----------

def _scan(path: Path) -> pl.LazyFrame:
    """Lazy-scan one OrderLog file with proper dtypes (no session filter)."""
    return pl.scan_csv(str(path), schema_overrides=_CSV_DTYPES)


# ---------- checks ----------

def check_tradeno_pairing(lf: pl.LazyFrame) -> dict:
    """Each TRADENO must appear exactly 2 times among ACTION=2 rows."""
    df = (
        lf.filter(pl.col("ACTION") == 2)
        .group_by("TRADENO")
        .agg(pl.len().alias("n"))
        .group_by("n")
        .agg(pl.len().alias("count_tradenos"))
        .sort("n")
        .collect()
    )
    return {"distribution": df.to_dicts()}


def check_volume_invariant(lf: pl.LazyFrame, sample_limit: int = 10) -> dict:
    """Σ FILL VOLUME (own side, ACTION=2) per ORDERNO + ACTION=0 VOLUME (if any)
    should equal ACTION=1 VOLUME for that ORDERNO.

    NOTE: each FILL is recorded twice in OrderLog (paired by TRADENO). The
    invariant is per-ORDERNO, where each ORDERNO sees only its own side's
    fills — so we sum FILLs grouped by ORDERNO directly.

    Reports number of violations and a sample.
    """
    # Build per-ORDERNO aggregates
    add = (
        lf.filter(pl.col("ACTION") == 1)
        .group_by("ORDERNO")
        .agg(
            pl.col("VOLUME").first().alias("add_volume"),
            pl.col("PRICE").first().alias("add_price"),
        )
    )
    fills = (
        lf.filter(pl.col("ACTION") == 2)
        .group_by("ORDERNO")
        .agg(pl.col("VOLUME").sum().alias("filled_volume"))
    )
    cancels = (
        lf.filter(pl.col("ACTION") == 0)
        .group_by("ORDERNO")
        .agg(pl.col("VOLUME").last().alias("cancel_remainder"))
    )

    joined = (
        add.join(fills, on="ORDERNO", how="left")
        .join(cancels, on="ORDERNO", how="left")
        .with_columns(
            filled_volume=pl.col("filled_volume").fill_null(0),
            cancel_remainder=pl.col("cancel_remainder").fill_null(0),
        )
    )

    # In MOEX semantics: ACTION=0 VOLUME is the *remainder* visible at cancel
    # time, not the cancelled amount. The cancelled amount = remainder.
    # Invariant: add_volume == filled_volume + cancel_remainder
    # (if no cancel ever, then add_volume should equal filled_volume; remaining
    # would mean order is still open by EOD — possible but rare for liquid names)
    violated = joined.filter(
        pl.col("add_volume") != pl.col("filled_volume") + pl.col("cancel_remainder")
    )

    summary = (
        violated
        .with_columns(
            diff=pl.col("filled_volume") + pl.col("cancel_remainder") - pl.col("add_volume"),
            is_market=pl.col("add_price") == 0,
        )
        .group_by("is_market")
        .agg(
            pl.len().alias("n_violations"),
            pl.col("diff").min().alias("min_diff"),
            pl.col("diff").max().alias("max_diff"),
            pl.col("diff").median().alias("median_diff"),
        )
        .collect()
    )

    total = joined.select(pl.len()).collect().item()
    sample = violated.head(sample_limit).collect().to_dicts()

    return {
        "total_ordernos": total,
        "violations_by_kind": summary.to_dicts(),
        "sample_violations": sample,
    }


def check_fill_orderno_coverage(lf: pl.LazyFrame) -> dict:
    """Every ORDERNO in ACTION=2 must have an ACTION=1 row."""
    fill_ids = lf.filter(pl.col("ACTION") == 2).select("ORDERNO").unique()
    add_ids = lf.filter(pl.col("ACTION") == 1).select("ORDERNO").unique()

    orphans = fill_ids.join(add_ids, on="ORDERNO", how="anti").collect()
    return {"n_orphan_fill_ordernos": len(orphans)}


def check_cancel_orderno_coverage(lf: pl.LazyFrame) -> dict:
    """Every ORDERNO in ACTION=0 must have an ACTION=1 row."""
    cancel_ids = lf.filter(pl.col("ACTION") == 0).select("ORDERNO").unique()
    add_ids = lf.filter(pl.col("ACTION") == 1).select("ORDERNO").unique()

    orphans = cancel_ids.join(add_ids, on="ORDERNO", how="anti").collect()
    return {"n_orphan_cancel_ordernos": len(orphans)}


def check_opening_window(lf: pl.LazyFrame) -> dict:
    """Distribution of TIME values around opening (10:00 - 10:10)."""
    # TIME is HHMMSSZZZXXX (12 digits). Hour bucket = TIME // 10^10.
    # We'll bucket by minute within hour 10.
    df = (
        lf.with_columns(
            t12=pl.col("TIME").cast(pl.Utf8).str.zfill(12),
        )
        .with_columns(
            hh=pl.col("t12").str.slice(0, 2),
            mm=pl.col("t12").str.slice(2, 2),
        )
        .filter(pl.col("hh") == "10")
        .filter(pl.col("mm").cast(pl.Int32) < 10)
        .group_by(["mm"])
        .agg(
            pl.len().alias("rows"),
            (pl.col("ACTION") == 1).sum().alias("adds"),
            (pl.col("ACTION") == 0).sum().alias("cancels"),
            (pl.col("ACTION") == 2).sum().alias("trades"),
        )
        .sort("mm")
        .collect()
    )
    return {"by_minute": df.to_dicts()}


# ---------- per-file driver ----------

def validate_file(path: Path, instruments: list[str]) -> dict:
    """Run all checks for one OrderLog file, restricted to selected instruments."""
    print(f"\n=== {path.name} ===")
    lf = _scan(path).filter(pl.col("SECCODE").is_in(instruments))

    print("  [1] TRADENO pairing...", flush=True)
    pairing = check_tradeno_pairing(lf)
    for entry in pairing["distribution"]:
        print(f"      n_rows_per_TRADENO={entry['n']}: {entry['count_tradenos']:>10,} TRADENOs")

    print("  [2] Volume invariant per ORDERNO...", flush=True)
    invariant = check_volume_invariant(lf)
    print(f"      total ORDERNOs (with ADD): {invariant['total_ordernos']:>10,}")
    for entry in invariant["violations_by_kind"]:
        kind = "MARKET ORDER" if entry["is_market"] else "limit order "
        print(f"      {kind} violations: {entry['n_violations']:>8,}  "
              f"diff range [{entry['min_diff']}, {entry['max_diff']}], median={entry['median_diff']}")
    if invariant["sample_violations"]:
        print(f"      sample (first 3):")
        for s in invariant["sample_violations"][:3]:
            print(f"        ORDERNO={s['ORDERNO']} add={s['add_volume']} "
                  f"filled={s['filled_volume']} cancel_rem={s['cancel_remainder']} "
                  f"price={s['add_price']}")

    print("  [3] FILL ORDERNO coverage...", flush=True)
    fill_cov = check_fill_orderno_coverage(lf)
    print(f"      orphan FILL ORDERNOs (no matching ADD): {fill_cov['n_orphan_fill_ordernos']:>8,}")

    print("  [4] CANCEL ORDERNO coverage...", flush=True)
    cancel_cov = check_cancel_orderno_coverage(lf)
    print(f"      orphan CANCEL ORDERNOs (no matching ADD): {cancel_cov['n_orphan_cancel_ordernos']:>8,}")

    print("  [5] Opening window 10:00–10:10 (rows by minute)...", flush=True)
    opening = check_opening_window(lf)
    for entry in opening["by_minute"]:
        print(f"      10:{entry['mm']}  rows={entry['rows']:>10,}  "
              f"adds={entry['adds']:>10,}  cancels={entry['cancels']:>10,}  "
              f"trades={entry['trades']:>8,}")

    return {
        "file": path.name,
        "pairing": pairing,
        "invariant": invariant,
        "fill_coverage": fill_cov,
        "cancel_coverage": cancel_cov,
        "opening": opening,
    }


def main():
    cfg = PipelineConfig()
    files = discover_files(cfg)
    print(f"Discovered {len(files)} files")
    print("Selecting top-{} instruments...".format(cfg.top_n_instruments))
    instruments = select_instruments(files, cfg)
    print(f"Selected: {instruments}\n")

    results = []
    for path, _date in files:
        try:
            results.append(validate_file(path, instruments))
        except Exception as e:
            print(f"  ERROR on {path.name}: {e}")

    # Cross-file aggregate verdict
    print("\n\n=== AGGREGATE VERDICT ===")
    total_orphan_fills = sum(r["fill_coverage"]["n_orphan_fill_ordernos"] for r in results)
    total_orphan_cancels = sum(r["cancel_coverage"]["n_orphan_cancel_ordernos"] for r in results)
    print(f"  Total orphan FILL ORDERNOs across all days:   {total_orphan_fills:>10,}")
    print(f"  Total orphan CANCEL ORDERNOs across all days: {total_orphan_cancels:>10,}")

    # Pairing: any (n != 2) bucket present?
    bad_pairing = []
    for r in results:
        for entry in r["pairing"]["distribution"]:
            if entry["n"] != 2 and entry["count_tradenos"] > 0:
                bad_pairing.append((r["file"], entry["n"], entry["count_tradenos"]))
    if bad_pairing:
        print(f"  BAD pairing entries (n≠2):")
        for fname, n, cnt in bad_pairing:
            print(f"    {fname}: n={n} → {cnt} TRADENOs")
    else:
        print(f"  TRADENO pairing: ALL OK (every TRADENO appears exactly 2x)")

    # Volume invariant: violations from market orders are expected; from limits — should be 0
    limit_violations = 0
    for r in results:
        for entry in r["invariant"]["violations_by_kind"]:
            if not entry["is_market"]:
                limit_violations += entry["n_violations"]
    print(f"  Volume invariant violations on LIMIT orders: {limit_violations:>10,}")
    if limit_violations == 0:
        print("    → No hidden volume in single-ORDERNO icebergs. Each iceberg slice"
              " uses a fresh ORDERNO. Converter needs no special iceberg handling.")
    else:
        print("    → INVESTIGATE: some limit orders have volume mismatch."
              " May indicate icebergs with hidden volume in one ORDERNO.")


if __name__ == "__main__":
    main()
