"""Resolve named ICU concepts to nodes in this cohort's query vocabulary.

Each concept names ONE node the model can be asked about.  Resolution is deliberate rather than
fuzzy: a concept is either an exact node name that must exist, or a leaf-level match (on the
leaf's OWN code string or description, never a roll-up of its descendants' text -- that is what
made an earlier probe match every regex against the top-level `LAB` node).

Where a concept is a code with value bins, prefer the dual-role `X//ANY` subtree node so the
query means "this drug at any rate" rather than "this drug, unvalued".

`resolve_concepts` returns {concept: ResolvedConcept} and raises if anything is unresolved --
a spec naming a node that does not exist would fail at eval-generation anyway, but much later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import polars as pl

# kind: "exact"  -> node_name must equal `pattern`
#       "code"   -> regex over the LEAF code string
#       "desc"   -> regex over the LEAF description
CONCEPTS: dict[str, tuple[str, str]] = {
    # --- structural / administrative (exact ontology nodes) -----------------------------
    "death": ("exact", "MEDS_DEATH"),
    "record_end": ("exact", "TIMELINE//END"),
    "icu_admit": ("exact", "ICU_ADMISSION"),          # ancestor over 16 ICU types
    "icu_discharge": ("exact", "ICU_DISCHARGE"),      # ancestor over 16 ICU types
    "hosp_admit": ("exact", "HOSPITAL_ADMISSION"),    # ancestor over 70 admission types
    "hosp_discharge": ("exact", "HOSPITAL_DISCHARGE"),  # ancestor over 14 dispositions
    "discharge_died": ("exact", "HOSPITAL_DISCHARGE//DIED"),
    "discharge_hospice": ("exact", "HOSPITAL_DISCHARGE//HOSPICE"),
    "transfer": ("exact", "TRANSFER_TO"),             # ancestor over 71 wards
    # --- vasoactive infusions (MIMIC-IV itemids; //ANY = any infusion rate) --------------
    "norepinephrine": ("code", r"^INFUSION_START//221906(//ANY)?$"),
    "phenylephrine": ("code", r"^INFUSION_START//221749(//ANY)?$"),
    "vasopressin": ("code", r"^INFUSION_START//222315(//ANY)?$"),
    "epinephrine": ("code", r"^INFUSION_START//221289(//ANY)?$"),
    "propofol": ("code", r"^INFUSION_START//222168(//ANY)?$"),
    "insulin_infusion": ("code", r"^INFUSION_START//223258(//ANY)?$"),
    # --- labs (leaf descriptions) --------------------------------------------------------
    # Anchored patterns matched nothing -- MIMIC lab descriptions are not bare analyte names.
    # These resolve to whichever node carries the concept most often; that may be a
    # SPECIMEN_COLLECTED node ("was a lactate drawn") rather than a RESULT node.  Both are
    # meaningful ICU signals; the resolution table records which one each concept got.
    "lactate": ("desc", r"lactate"),
    "creatinine": ("desc", r"creatinine"),
    "troponin": ("desc", r"troponin"),
    "platelets": ("desc", r"platelet"),
    "bilirubin": ("desc", r"bilirubin"),
    "wbc": ("desc", r"white blood cell"),
    # --- medications (MIMIC MEDICATION//<drug name>) --------------------------------------
    "vancomycin": ("code", r"^MEDICATION//Vancomycin"),
    "furosemide": ("code", r"^MEDICATION//Furosemide"),
    "heparin": ("code", r"^MEDICATION//Heparin"),
}


@dataclass(frozen=True)
class ResolvedConcept:
    concept: str
    node: str
    is_ancestor: bool
    n_desc: int
    n_occ: int


def _node_table(ontology_dir: Path, cohort_dir: Path) -> pl.DataFrame:
    vocab = pl.read_parquet(ontology_dir / "ontology_vocab.parquet")
    closure = pl.read_parquet(ontology_dir / "event_to_query_nodes.parquet")
    codes = pl.read_parquet(cohort_dir / "metadata" / "codes.parquet").select(
        "code", "description", "code/n_occurrences"
    )
    stats = (
        closure.join(codes, left_on="event_code", right_on="code", how="left")
        .fill_null(0)
        .group_by("query_node")
        .agg(pl.col("code/n_occurrences").sum().alias("n_occ"), pl.len().alias("n_desc"))
    )
    nodes = vocab.join(stats, left_on="node_name", right_on="query_node", how="left").fill_null(0)
    # A leaf's OWN description (ancestors get null -- deliberately, so "desc" never rolls up).
    return nodes.join(
        codes.select(pl.col("code").alias("node_name"), "description"), on="node_name", how="left"
    )


def resolve_concepts(
    ontology_dir: str | Path, cohort_dir: str | Path, wanted: list[str] | None = None
) -> dict[str, ResolvedConcept]:
    nodes = _node_table(Path(ontology_dir), Path(cohort_dir))
    names = set(nodes["node_name"].to_list())
    out: dict[str, ResolvedConcept] = {}
    missing: list[str] = []

    for concept in wanted or list(CONCEPTS):
        kind, pattern = CONCEPTS[concept]
        if kind == "exact":
            cand = nodes.filter(pl.col("node_name") == pattern)
        elif kind == "code":
            cand = nodes.filter(pl.col("node_name").str.contains(f"(?i){pattern}"))
        elif kind == "desc":
            cand = nodes.filter(
                pl.col("description").is_not_null()
                & pl.col("description").str.contains(f"(?i){pattern}")
            )
        else:  # pragma: no cover
            raise ValueError(kind)

        if cand.height == 0:
            missing.append(concept)
            continue

        # Prefer the `//ANY` subtree node when both it and the bare code matched: the bare name
        # means "exactly this code", `//ANY` means "this code or any of its value bins".
        any_rows = cand.filter(pl.col("node_name").str.ends_with("//ANY"))
        pick = (any_rows if any_rows.height else cand).sort("n_occ", descending=True).row(0, named=True)
        out[concept] = ResolvedConcept(
            concept=concept,
            node=pick["node_name"],
            is_ancestor=not pick["is_observed_code"],
            n_desc=int(pick["n_desc"]),
            n_occ=int(pick["n_occ"]),
        )

    if missing:
        raise LookupError(f"unresolved concepts: {missing}")
    assert all(r.node in names for r in out.values())
    return out
