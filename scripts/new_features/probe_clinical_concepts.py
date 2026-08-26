"""Report how well each candidate ICU concept resolves against the cohort vocabulary.

Prints, per concept, only: the number of matching nodes, the best node's occurrence/subject
counts, whether it is an ancestor (and how many descendants), and the node NAME.  It does not
dump the vocabulary -- only the handful of nodes an ICU task panel would actually use.

Matching runs over the ontology's node names AND the cohort's `description` column, so a concept
can be found either by code structure or by its human-readable label.
"""

import os
import re
import sys
from pathlib import Path

import polars as pl

# (concept, regex over node name, regex over description) -- either may be None.
CONCEPTS: list[tuple[str, str | None, str | None]] = [
    ("death", r"^MEDS_DEATH$", None),
    ("record_end", r"^TIMELINE//END$", None),
    ("icu_admission", r"^ICU_ADMISSION", None),
    ("hosp_admission", r"^HOSPITAL_ADMISSION", None),
    ("hosp_discharge", r"^HOSPITAL_DISCHARGE", None),
    ("icu_discharge", r"^ICU_DISCHARGE", None),
    ("transfer", r"^TRANSFER_TO", None),
    ("norepinephrine", None, r"norepinephrine|levophed"),
    ("epinephrine", None, r"\bepinephrine\b"),
    ("vasopressin", None, r"vasopressin"),
    ("phenylephrine", None, r"phenylephrine"),
    ("dopamine", None, r"dopamine"),
    ("invasive_ventilation", None, r"ventilat|endotracheal|intubat|tidal volume|peep"),
    ("dialysis_crrt", None, r"dialysis|crrt|hemofiltration|ultrafiltrat"),
    ("rbc_transfusion", None, r"red blood cell|packed rbc|prbc|transfus"),
    ("lactate", None, r"^lactate$|lactic acid"),
    ("creatinine", None, r"creatinine"),
    ("blood_culture", None, r"blood culture"),
    ("troponin", None, r"troponin"),
    ("platelet", None, r"platelet"),
    ("bilirubin", None, r"bilirubin"),
    ("gcs", None, r"gcs|glasgow"),
    ("map_bp", None, r"arterial blood pressure mean|mean arterial"),
    ("heart_rate", None, r"^heart rate$"),
    ("spo2", None, r"o2 saturation|spo2"),
    ("antibiotic", None, r"vancomycin|piperacillin|cefepime|meropenem|ceftriaxone"),
    ("sedation", None, r"propofol|fentanyl|midazolam|dexmedetomidine"),
    ("insulin", None, r"insulin"),
]


def main() -> int:
    onto = Path(os.environ["NF_ONTOLOGY_DIR"])
    cohort = Path(os.environ["TENSORIZED_COHORT_DIR"])

    vocab = pl.read_parquet(onto / "ontology_vocab.parquet")
    closure = pl.read_parquet(onto / "event_to_query_nodes.parquet")
    codes = pl.read_parquet(cohort / "metadata" / "codes.parquet").select(
        "code", "description", "code/n_occurrences", "code/n_subjects"
    )

    # Per-node totals via the closure (a leaf maps to itself).
    stats = (
        closure.join(codes, left_on="event_code", right_on="code", how="left")
        .fill_null(0)
        .group_by("query_node")
        .agg(
            pl.col("code/n_occurrences").sum().alias("n_occ"),
            pl.col("code/n_subjects").sum().alias("n_subj_ub"),
            pl.len().alias("n_desc"),
        )
    )
    nodes = vocab.join(stats, left_on="node_name", right_on="query_node", how="left").fill_null(0)

    # description lookup: node -> concatenated descriptions of its descendants
    desc = (
        closure.join(codes, left_on="event_code", right_on="code", how="left")
        .group_by("query_node")
        .agg(pl.col("description").drop_nulls().str.join(" | ").alias("desc"))
    )
    nodes = nodes.join(desc, left_on="node_name", right_on="query_node", how="left").with_columns(
        pl.col("desc").fill_null("")
    )

    print(f"{'concept':<22}{'matches':>9}{'best n_occ':>12}{'n_desc':>8}{'anc':>5}  node")
    for name, name_re, desc_re in CONCEPTS:
        m = nodes
        if name_re:
            m = m.filter(pl.col("node_name").str.contains(f"(?i){name_re}"))
        if desc_re:
            m = m.filter(pl.col("desc").str.contains(f"(?i){desc_re}"))
        if m.height == 0:
            print(f"{name:<22}{0:>9}{'-':>12}{'-':>8}{'-':>5}  (NO MATCH)")
            continue
        best = m.sort("n_occ", descending=True).head(1)
        r = best.row(0, named=True)
        anc = "yes" if not r["is_observed_code"] else "no"
        print(f"{name:<22}{m.height:>9}{r['n_occ']:>12}{r['n_desc']:>8}{anc:>5}  {r['node_name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
