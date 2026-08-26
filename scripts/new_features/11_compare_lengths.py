"""Join a panel's length-1 and length-3 per-task AUROC and report the conditioning delta.

At length 3 the same target sits at position 2 behind a two-query prefix, and inference is
teacher-forced -- so the delta is what knowing the TRUE answers to two earlier questions is
worth to the model on that task.

Usage: 11_compare_lengths.py <metrics_dir> <tag1> <tag3> [manifest_csv]
"""

import sys
from pathlib import Path

import polars as pl


def main() -> int:
    m = Path(sys.argv[1])
    t1, t3 = sys.argv[2], sys.argv[3]
    manifest = Path(sys.argv[4]) if len(sys.argv) > 4 else None

    a = pl.read_csv(m / f"by_task_{t1}.csv").select(
        "spec", "category", "n", "n_pos", "prevalence",
        pl.col("auroc").alias("auroc_len1"), pl.col("ci_lo").alias("lo1"), pl.col("ci_hi").alias("hi1"),
    )
    b = pl.read_csv(m / f"by_task_{t3}.csv").select(
        "spec", pl.col("auroc").alias("auroc_len3"),
        pl.col("ci_lo").alias("lo3"), pl.col("ci_hi").alias("hi3"),
    )
    t = a.join(b, on="spec").with_columns(
        (pl.col("auroc_len3") - pl.col("auroc_len1")).alias("delta"),
        ((pl.col("lo1") > 0.5) | (pl.col("hi1") < 0.5)).alias("sig1"),
    )
    if manifest is not None and manifest.exists():
        t = t.join(pl.read_csv(manifest).select("spec", "description"), on="spec", how="left")

    t = t.sort("delta", descending=True)
    t.write_csv(m / f"compare_{t1}_vs_{t3}.csv")

    has_desc = "description" in t.columns
    print(f"{'spec':<38}{'prev':>7}{'len1':>8}{'len3':>8}{'delta':>8}{'sig':>5}")
    for r in t.iter_rows(named=True):
        a1 = "n/a" if r["auroc_len1"] is None else f"{r['auroc_len1']:.3f}"
        a3 = "n/a" if r["auroc_len3"] is None else f"{r['auroc_len3']:.3f}"
        d = "" if r["delta"] is None else f"{r['delta']:+.3f}"
        print(f"{r['spec']:<38}{r['prevalence']:>7.3f}{a1:>8}{a3:>8}{d:>8}{('*' if r['sig1'] else ''):>5}")

    d = t["delta"].drop_nulls()
    print(f"\nlen3 - len1 delta: mean={d.mean():+.4f} median={d.median():+.4f} "
          f"improved={int((d > 0).sum())}/{d.len()}")
    for cat in ("dur", "evt", "anc"):
        dc = t.filter(pl.col("category") == cat)["delta"].drop_nulls()
        if dc.len():
            print(f"  {cat}: mean={dc.mean():+.4f} improved={int((dc > 0).sum())}/{dc.len()}")
    if has_desc:
        print("\ntop 5 tasks by conditioning gain:")
        for r in t.head(5).iter_rows(named=True):
            print(f"  {r['delta']:+.3f}  {r['description']}")
    print(f"\nwrote {m / f'compare_{t1}_vs_{t3}.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
