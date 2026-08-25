"""Print ONLY the schema (column names + dtypes) and row counts of the cohort parquets.

Never prints patient rows.  Output is safe to read into an agent's context.
"""

import sys
from pathlib import Path

import polars as pl

ROOT = Path("/experiments/EQ_conditional_experiments/data")


def describe(p: Path) -> None:
    try:
        lf = pl.scan_parquet(p)
        schema = lf.collect_schema()
        n = lf.select(pl.len()).collect().item()
        print(f"  {p.relative_to(ROOT)}  rows={n}")
        for name, dtype in schema.items():
            print(f"      {name}: {dtype}")
    except Exception as e:  # noqa: BLE001
        print(f"  {p}  ERROR {type(e).__name__}: {e}")


def first_parquet(d: Path) -> Path | None:
    if not d.is_dir():
        return None
    files = sorted(d.glob("*.parquet"))
    return files[0] if files else None


def main() -> int:
    cohort = ROOT / "tensorized_cohort"
    print("=== tensorized_cohort top-level dirs ===")
    for d in sorted(cohort.iterdir()):
        if d.is_dir():
            n_pq = len(list(d.rglob("*.parquet")))
            print(f"  {d.name}/   ({n_pq} parquet files)")

    for sub in ["data", "tokenization/event_seqs", "tokenization/schemas"]:
        print(f"\n=== {sub}/held_out (first shard schema) ===")
        p = first_parquet(cohort / sub / "held_out")
        if p is None:
            print("  (no parquet)")
        else:
            describe(p)

    print("\n=== metadata/codes.parquet ===")
    describe(cohort / "metadata" / "codes.parquet")

    print("\n=== shard counts per split ===")
    for sub in ["data", "tokenization/event_seqs"]:
        for split in ["train", "tuning", "held_out"]:
            d = cohort / sub / split
            if d.is_dir():
                print(f"  {sub}/{split}: {len(list(d.glob('*.parquet')))} shards")

    print("\n=== query_sequences_big (existing training labels) ===")
    qs = ROOT / "query_sequences_big"
    for split in ["train", "tuning", "held_out"]:
        p = first_parquet(qs / split)
        print(f"  {split}: {len(list((qs / split).glob('*.parquet'))) if (qs / split).is_dir() else 0} shards")
        if p is not None:
            describe(p)
            break

    print("\n=== is there a raw/intermediate MEDS event dir anywhere? ===")
    for cand in [
        ROOT / "tensorized_cohort" / "data",
        ROOT.parent / "data",
    ]:
        print(f"  {cand}: exists={cand.exists()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
