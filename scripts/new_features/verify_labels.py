"""Verify the three new features are actually present in a labels parquet tree.

Prints ONLY aggregate statistics — never subject ids, never code strings, never rows.
Usage: verify_labels.py <labels_dir> <split>
"""

import os
import sys
from pathlib import Path

import polars as pl


def main() -> int:
    labels_dir = Path(sys.argv[1])
    split = sys.argv[2]
    onto_dir = Path(os.environ["NF_ONTOLOGY_DIR"])

    files = sorted((labels_dir / split).glob("*.parquet"))
    print(f"labels: {labels_dir}/{split}  shards={len(files)}")
    if not files:
        print("NO SHARDS")
        return 1

    lf = pl.scan_parquet(files)
    cols = lf.collect_schema()
    print(f"columns: {dict(cols)}")

    df = lf.collect()
    n = df.height
    print(f"\nrows (sequences) = {n}")

    # --- alignment of the list columns -------------------------------------------------
    list_cols = [c for c in ("queries", "durations", "answers", "bound_events") if c in df.columns]
    lens = df.select([pl.col(c).list.len().alias(c) for c in list_cols])
    base = lens[list_cols[0]]
    for c in list_cols[1:]:
        same = (lens[c] == base).all()
        print(f"  len({c}) == len({list_cols[0]}): {same}")
    print(f"  sequence length: min={base.min()} max={base.max()} mean={base.mean():.2f}")

    ex = df.select(list_cols).explode(list_cols)
    n_q = ex.height
    print(f"  total queries = {n_q}")

    # --- Feature 2: event-bounded durations ---------------------------------------------
    print("\n--- FEATURE 2: event-bounded durations ---")
    if "bound_events" not in df.columns:
        print("  bound_events column ABSENT -> feature is OFF")
    else:
        n_bound = ex.filter(pl.col("bound_events").is_not_null()).height
        print(f"  bound_events column PRESENT")
        print(f"  event-bounded queries: {n_bound} / {n_q} = {n_bound / n_q:.3f}")
        # sentinel agreement
        bad_a = ex.filter(pl.col("bound_events").is_not_null() & (pl.col("durations") != -1.0)).height
        bad_b = ex.filter(pl.col("bound_events").is_null() & (pl.col("durations") == -1.0)).height
        print(f"  bounded-but-duration!=-1 : {bad_a} (must be 0)")
        print(f"  unbounded-but-duration==-1: {bad_b} (must be 0)")
        pos_b = ex.filter(pl.col("bound_events").is_not_null())["answers"].mean()
        pos_u = ex.filter(pl.col("bound_events").is_null())["answers"].mean()
        print(f"  positive rate  bounded={pos_b:.4f}  unbounded={pos_u:.4f}")

    # --- Feature 3: DAG-aware queries ----------------------------------------------------
    print("\n--- FEATURE 3: DAG-aware (ancestor) queries ---")
    vocab = pl.read_parquet(onto_dir / "ontology_vocab.parquet")
    anc = set(vocab.filter(~pl.col("is_observed_code"))["node_name"].to_list())
    leaf = set(vocab.filter(pl.col("is_observed_code"))["node_name"].to_list())

    qs = ex["queries"]
    n_anc_q = sum(1 for q in qs if q in anc)
    n_leaf_q = sum(1 for q in qs if q in leaf)
    n_oov = n_q - n_anc_q - n_leaf_q
    print(f"  ancestor queries: {n_anc_q} / {n_q} = {n_anc_q / n_q:.3f}")
    print(f"  leaf queries    : {n_leaf_q} / {n_q} = {n_leaf_q / n_q:.3f}")
    print(f"  out-of-vocab    : {n_oov} (must be 0)")

    if n_anc_q:
        anc_mask = pl.Series([q in anc for q in qs])
        print(f"  positive rate  ancestor={ex.filter(anc_mask)['answers'].mean():.4f}  "
              f"leaf={ex.filter(~anc_mask)['answers'].mean():.4f}")

    if "bound_events" in df.columns:
        bes = [b for b in ex["bound_events"] if b is not None]
        n_anc_b = sum(1 for b in bes if b in anc)
        print(f"  ancestor BOUNDARIES: {n_anc_b} / {len(bes)} = {n_anc_b / max(len(bes), 1):.3f}")

    # --- overall label balance ------------------------------------------------------------
    print(f"\noverall positive rate = {ex['answers'].mean():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
