"""Confirm WHICH cohort each env var points at, by shape/identity only (never rows).

Prints: whether TENSORIZED_COHORT_DIR and TOKENIZED_EVENTS_DIR are the same tree as the
/experiments archive, their codes.parquet row counts and max vocab index, and the ontology's
V_ext.  No patient rows, no code strings.
"""

import os
import sys
from pathlib import Path

import polars as pl

ARCHIVE = Path("/experiments/EQ_conditional_experiments/data/tensorized_cohort")


def codes_summary(p: Path, label: str) -> None:
    f = p / "metadata" / "codes.parquet"
    if not f.exists():
        print(f"{label}: no metadata/codes.parquet")
        return
    lf = pl.scan_parquet(f)
    cols = lf.collect_schema().names()
    n = lf.select(pl.len()).collect().item()
    print(f"{label}: rows={n}  has_parent_codes={'parent_codes' in cols}")
    if "code/vocab_index" in cols:
        mx = lf.select(pl.col("code/vocab_index").max()).collect().item()
        mn = lf.select(pl.col("code/vocab_index").min()).collect().item()
        print(f"{label}: vocab_index range=[{mn}, {mx}]")
    print(f"{label}: realpath_matches_archive={p.resolve() == ARCHIVE.resolve()}")
    print(f"{label}: under_/data={str(p).startswith('/data')}")


def main() -> int:
    for var in ("TENSORIZED_COHORT_DIR", "TOKENIZED_EVENTS_DIR", "DATA_DIR"):
        val = os.environ.get(var)
        if not val:
            print(f"{var}: UNSET")
            continue
        codes_summary(Path(val), var)
        print()

    codes_summary(ARCHIVE, "ARCHIVE(/experiments)")
    print()

    onto = os.environ.get("NF_ONTOLOGY_DIR")
    if onto:
        v = pl.scan_parquet(Path(onto) / "ontology_vocab.parquet")
        schema = v.collect_schema().names()
        n = v.select(pl.len()).collect().item()
        mx = v.select(pl.col("token_id").max()).collect().item()
        n_leaf = v.filter(pl.col("is_observed_code")).select(pl.len()).collect().item()
        n_anc = v.filter(~pl.col("is_observed_code")).select(pl.len()).collect().item()
        print(f"ontology_vocab: rows={n} cols={schema} max_token_id={mx} leaves={n_leaf} ancestors={n_anc}")
        print(f"=> V_ext (max_token_id+1) = {mx + 1}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
