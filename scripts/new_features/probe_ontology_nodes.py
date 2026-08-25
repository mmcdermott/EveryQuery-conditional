"""Check that the ancestor nodes an eval set needs actually exist in V_ext.

Prints ONLY: existence booleans for a few user-named anchors, descendant counts, and
AGGREGATE shape statistics of the ancestor layer.  It never dumps the code vocabulary.
"""

import os
import sys
from pathlib import Path

import polars as pl

# Anchors the user named explicitly, plus the censor query.
ANCHORS = ["HOSPITAL_ADMISSION", "HOSPITAL_ADMISSION//ANY", "TIMELINE//END", "MEDS_DEATH"]


def main() -> int:
    onto = Path(os.environ["NF_ONTOLOGY_DIR"])
    vocab = pl.read_parquet(onto / "ontology_vocab.parquet")
    closure = pl.read_parquet(onto / "event_to_query_nodes.parquet")

    print(f"V_ext rows={vocab.height}  closure rows={closure.height}")
    print(f"closure cols={closure.columns}")

    names = set(vocab["node_name"].to_list())
    print("\n--- anchor existence ---")
    for a in ANCHORS:
        present = a in names
        row = vocab.filter(pl.col("node_name") == a)
        obs = row["is_observed_code"].item() if row.height else None
        tid = row["token_id"].item() if row.height else None
        n_desc = closure.filter(pl.col("query_node") == a).height
        print(f"  {a!r:34s} present={present} is_observed_code={obs} token_id={tid} n_events_mapping_to_it={n_desc}")

    print("\n--- ancestor layer shape (aggregate only) ---")
    anc = vocab.filter(~pl.col("is_observed_code"))
    print(f"  n_ancestors={anc.height}")
    counts = (
        closure.group_by("query_node")
        .len()
        .rename({"len": "n_desc"})
        .join(vocab.select("node_name", "is_observed_code"), left_on="query_node", right_on="node_name")
    )
    anc_counts = counts.filter(~pl.col("is_observed_code"))["n_desc"]
    leaf_counts = counts.filter(pl.col("is_observed_code"))["n_desc"]
    print(f"  ancestor descendant-count: min={anc_counts.min()} median={anc_counts.median()} max={anc_counts.max()}")
    print(f"  leaf descendant-count:     min={leaf_counts.min()} median={leaf_counts.median()} max={leaf_counts.max()}")
    print(f"  ancestors with >=2 descendants: {(anc_counts >= 2).sum()}")
    print(f"  ancestors with >=5 descendants: {(anc_counts >= 5).sum()}")

    # How many ancestor names are top-level (no '//')?
    n_top = anc.filter(~pl.col("node_name").str.contains("//")).height
    print(f"  top-level ancestors (no '//'): {n_top}")

    print("\n--- token_id layout ---")
    leaves = vocab.filter(pl.col("is_observed_code"))["token_id"]
    ancs = anc["token_id"]
    print(f"  leaf token_id range=[{leaves.min()}, {leaves.max()}]")
    print(f"  ancestor token_id range=[{ancs.min()}, {ancs.max()}]")
    print(f"  ancestors strictly above all leaves: {ancs.min() > leaves.max()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
