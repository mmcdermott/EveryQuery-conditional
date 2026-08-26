"""Survey the cohort vocabulary's SHAPE, so ICU concepts can be resolved without dumping it.

Prints per top-level prefix: number of leaf codes, total occurrences, how many carry a
description.  Then, for a few named prefixes, the depth-2 segments.  Counts only.
"""

import os
import sys
from pathlib import Path

import polars as pl


def main() -> int:
    cohort = Path(os.environ["TENSORIZED_COHORT_DIR"])
    codes = pl.read_parquet(cohort / "metadata" / "codes.parquet").select(
        "code", "description", "code/n_occurrences", "code/n_subjects"
    )
    n = codes.height
    n_desc = codes.filter(pl.col("description").is_not_null() & (pl.col("description") != "")).height
    print(f"leaf codes={n}  with description={n_desc} ({n_desc / n:.1%})")

    top = (
        codes.with_columns(pl.col("code").str.split("//").list.get(0).alias("p0"))
        .group_by("p0")
        .agg(
            pl.len().alias("n_codes"),
            pl.col("code/n_occurrences").sum().alias("n_occ"),
            pl.col("code/n_subjects").max().alias("max_subj"),
        )
        .sort("n_occ", descending=True)
    )
    print(f"\n--- top-level prefixes ({top.height} distinct) ---")
    print(f"{'prefix':<34}{'n_codes':>9}{'n_occ':>14}{'max_subj':>10}")
    for r in top.head(30).iter_rows(named=True):
        print(f"{r['p0']:<34}{r['n_codes']:>9}{r['n_occ']:>14}{r['max_subj']:>10}")

    for prefix in ("ICU_ADMISSION", "HOSPITAL_DISCHARGE", "ICU_DISCHARGE", "INFUSION_START",
                   "MEDICATION", "PROCEDURE"):
        sub = codes.filter(pl.col("code").str.starts_with(prefix + "//"))
        if sub.height == 0:
            print(f"\n--- {prefix}// : none ---")
            continue
        seg = (
            sub.with_columns(pl.col("code").str.split("//").list.get(1).alias("p1"))
            .group_by("p1")
            .agg(pl.len().alias("n"), pl.col("code/n_occurrences").sum().alias("n_occ"))
            .sort("n_occ", descending=True)
        )
        print(f"\n--- {prefix}// : {sub.height} codes, {seg.height} distinct depth-2 segments ---")
        for r in seg.head(8).iter_rows(named=True):
            print(f"    {str(r['p1'])[:60]:<62}{r['n']:>6}{r['n_occ']:>12}")

    # do descriptions ever carry drug names?
    print("\n--- description coverage by prefix ---")
    cov = (
        codes.with_columns(pl.col("code").str.split("//").list.get(0).alias("p0"))
        .with_columns((pl.col("description").is_not_null() & (pl.col("description") != "")).alias("has_d"))
        .group_by("p0")
        .agg(pl.len().alias("n"), pl.col("has_d").sum().alias("n_desc"))
        .sort("n", descending=True)
    )
    for r in cov.head(12).iter_rows(named=True):
        print(f"    {r['p0']:<34}{r['n_desc']:>7}/{r['n']:<7}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
