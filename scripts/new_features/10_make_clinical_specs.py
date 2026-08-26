"""Build a clinically meaningful ICU evaluation panel in the conditional query language.

Unlike `03_make_eval_specs.py`, which draws tasks at random to ask "does the model do better
than chance on an arbitrary query", this panel asks "can the model answer questions an intensivist
would actually ask".  Each task is hand-designed; the concepts are resolved to real nodes by
`clinical_concepts.py`.

Every task exists at two lengths with the SAME target:
  * length 1 -- the question asked cold;
  * length 3 -- the same question at position 2, behind a two-query CLINICAL prefix chosen to be
    informative for it (e.g. lactate drawn -> norepinephrine started -> will they die).
Because inference is teacher-forced, the length-3 target sees the TRUE answers to its prefix, so
the len1 -> len3 delta measures how much a clinically relevant history is worth to the model.

Name prefixes encode the query form so the scorer can group them:
  dur_  duration-bounded   [code, days]
  evt_  event-bounded      [code, -1, bound]   "does X happen before the next Y?"
  anc_  DAG/ancestor       [ancestor, days]    fires on any descendant

Writes designed_clin_len1.yaml, designed_clin_len3.yaml, clinical_manifest.csv.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import polars as pl
import yaml
from clinical_concepts import resolve_concepts

EVENT_BOUND_SENTINEL = -1

# (name, description, [step, step, target]) where a step is
#   (concept, days)                  -> duration-bounded
#   (concept, None, bound_concept)   -> event-bounded
PANEL: list[tuple[str, str, list[tuple]]] = [
    # ---- A. Mortality ---------------------------------------------------------------------
    ("dur_mortality_1d", "Death within 1 day",
     [("lactate", 1), ("norepinephrine", 1), ("death", 1)]),
    ("dur_mortality_7d", "Death within 7 days",
     [("icu_admit", 2), ("norepinephrine", 2), ("death", 7)]),
    ("dur_mortality_30d", "Death within 30 days",
     [("record_end", 30), ("icu_admit", 7), ("death", 30)]),
    ("evt_death_before_discharge", "In-hospital mortality: death before the next hospital discharge",
     [("icu_admit", None, "hosp_discharge"), ("norepinephrine", None, "hosp_discharge"),
      ("death", None, "hosp_discharge")]),
    ("dur_discharge_died_30d", "Discharge disposition DIED within 30 days",
     [("icu_admit", 7), ("norepinephrine", 7), ("discharge_died", 30)]),
    ("dur_discharge_hospice_30d", "Hospice discharge within 30 days (goals-of-care transition)",
     [("icu_admit", 7), ("record_end", 30), ("discharge_hospice", 30)]),
    ("evt_death_before_next_admission", "Death before the next hospital admission",
     [("hosp_discharge", None, "hosp_admit"), ("record_end", None, "hosp_admit"),
      ("death", None, "hosp_admit")]),
    # ---- B. ICU utilisation / escalation ----------------------------------------------------
    ("anc_icu_admit_2d", "ICU admission to ANY unit within 2 days",
     [("lactate", 1), ("norepinephrine", 1), ("icu_admit", 2)]),
    ("anc_icu_admit_7d", "ICU admission to ANY unit within 7 days",
     [("hosp_admit", 2), ("transfer", 2), ("icu_admit", 7)]),
    ("evt_icu_before_discharge", "ICU escalation before the next hospital discharge",
     [("lactate", None, "hosp_discharge"), ("vancomycin", None, "hosp_discharge"),
      ("icu_admit", None, "hosp_discharge")]),
    ("anc_icu_discharge_7d", "ICU discharge from ANY unit within 7 days (ICU LOS proxy)",
     [("icu_admit", 1), ("propofol", 2), ("icu_discharge", 7)]),
    ("anc_transfer_2d", "Transfer to ANY ward within 2 days",
     [("hosp_admit", 1), ("icu_admit", 2), ("transfer", 2)]),
    ("anc_hosp_admit_30d", "Hospital admission of ANY type within 30 days (readmission proxy)",
     [("record_end", 30), ("hosp_discharge", 7), ("hosp_admit", 30)]),
    ("evt_transfer_before_discharge", "Ward transfer before the next hospital discharge",
     [("icu_admit", None, "hosp_discharge"), ("norepinephrine", None, "hosp_discharge"),
      ("transfer", None, "hosp_discharge")]),
    # ---- C. Organ support / haemodynamics ---------------------------------------------------
    ("anc_norepinephrine_1d", "Norepinephrine at ANY infusion rate within 1 day",
     [("lactate", 1), ("icu_admit", 1), ("norepinephrine", 1)]),
    ("anc_norepinephrine_2d", "Norepinephrine at ANY infusion rate within 2 days",
     [("hosp_admit", 1), ("icu_admit", 2), ("norepinephrine", 2)]),
    ("evt_norepi_before_icu_discharge", "Vasopressor need before leaving the ICU",
     [("lactate", None, "icu_discharge"), ("propofol", None, "icu_discharge"),
      ("norepinephrine", None, "icu_discharge")]),
    ("anc_vasopressin_2d", "Vasopressin within 2 days (second-line pressor: refractory shock)",
     [("norepinephrine", 1), ("lactate", 1), ("vasopressin", 2)]),
    ("anc_epinephrine_2d", "Epinephrine within 2 days (third-line pressor)",
     [("norepinephrine", 1), ("vasopressin", 1), ("epinephrine", 2)]),
    ("anc_propofol_1d", "Propofol within 1 day (sedation; proxy for intubation)",
     [("icu_admit", 1), ("norepinephrine", 1), ("propofol", 1)]),
    ("evt_propofol_before_icu_discharge", "Sedation before leaving the ICU",
     [("norepinephrine", None, "icu_discharge"), ("lactate", None, "icu_discharge"),
      ("propofol", None, "icu_discharge")]),
    # ---- D. Labs / workup --------------------------------------------------------------------
    ("dur_lactate_1d", "Lactate drawn within 1 day (sepsis workup)",
     [("icu_admit", 1), ("vancomycin", 1), ("lactate", 1)]),
    ("dur_creatinine_2d", "Creatinine within 2 days (renal monitoring)",
     [("furosemide", 1), ("icu_admit", 2), ("creatinine", 2)]),
    ("dur_troponin_1d", "Troponin within 1 day (ACS workup)",
     [("icu_admit", 1), ("heparin", 1), ("troponin", 1)]),
    ("dur_platelets_1d", "Platelet count within 1 day",
     [("heparin", 1), ("icu_admit", 1), ("platelets", 1)]),
    ("dur_bilirubin_2d", "Bilirubin within 2 days (hepatic dysfunction monitoring)",
     [("icu_admit", 1), ("norepinephrine", 2), ("bilirubin", 2)]),
    ("evt_creatinine_before_icu_discharge", "Creatinine drawn before leaving the ICU",
     [("lactate", None, "icu_discharge"), ("norepinephrine", None, "icu_discharge"),
      ("creatinine", None, "icu_discharge")]),
    # ---- E. Treatment -------------------------------------------------------------------------
    ("anc_vancomycin_1d", "Vancomycin within 1 day (empiric antibiotics)",
     [("lactate", 1), ("icu_admit", 1), ("vancomycin", 1)]),
    ("evt_vancomycin_before_discharge", "Antibiotics before the next hospital discharge",
     [("lactate", None, "hosp_discharge"), ("icu_admit", None, "hosp_discharge"),
      ("vancomycin", None, "hosp_discharge")]),
    ("anc_furosemide_2d", "Furosemide within 2 days (volume overload management)",
     [("icu_admit", 1), ("creatinine", 2), ("furosemide", 2)]),
    ("anc_heparin_1d", "Heparin within 1 day (VTE prophylaxis / ACS)",
     [("icu_admit", 1), ("platelets", 1), ("heparin", 1)]),
]


def descendant_sets(onto: Path) -> dict[str, frozenset[str]]:
    closure = pl.read_parquet(onto / "event_to_query_nodes.parquet")
    out: dict[str, set[str]] = {}
    for ev, node in zip(closure["event_code"], closure["query_node"], strict=True):
        out.setdefault(node, set()).add(ev)
    return {k: frozenset(v) for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    onto = Path(os.environ["NF_ONTOLOGY_DIR"])
    cohort = Path(os.environ["TENSORIZED_COHORT_DIR"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    used = sorted({c for _, _, steps in PANEL for s in steps for c in (s[0], s[2] if len(s) == 3 else None) if c})
    res = resolve_concepts(onto, cohort, used)
    desc = descendant_sets(onto)

    def step_to_entry(step: tuple) -> list:
        concept = step[0]
        node = res[concept].node
        if len(step) == 2:
            return [node, step[1]]
        bound = res[step[2]].node
        # A query bounded by itself or by one of its own ancestors is unconditionally False.
        dq, db = desc.get(node, frozenset()), desc.get(bound, frozenset())
        if dq <= db:
            raise ValueError(
                f"{concept} bounded by {step[2]}: the bound is an ancestor-or-self of the query, "
                "so the answer is always False"
            )
        return [node, EVENT_BOUND_SENTINEL, bound]

    len1: dict[str, list] = {}
    len3: dict[str, list] = {}
    rows = []
    for name, description, steps in PANEL:
        assert len(steps) == 3, f"{name}: expected 3 steps, got {len(steps)}"
        entries = [step_to_entry(s) for s in steps]
        len3[name] = entries
        len1[name] = [entries[-1]]
        tgt = steps[-1]
        r = res[tgt[0]]
        rows.append(
            {
                "spec": name,
                "category": name.split("_")[0],
                "description": description,
                "target_concept": tgt[0],
                "target_node": r.node,
                "is_ancestor": r.is_ancestor,
                "n_descendants": r.n_desc,
                "horizon_days": tgt[1] if len(tgt) == 2 else None,
                "bound_concept": tgt[2] if len(tgt) == 3 else None,
                "bound_node": res[tgt[2]].node if len(tgt) == 3 else None,
            }
        )

    (out_dir / "designed_clin_len1.yaml").write_text(yaml.safe_dump(len1, sort_keys=True))
    (out_dir / "designed_clin_len3.yaml").write_text(yaml.safe_dump(len3, sort_keys=True))
    man = pl.DataFrame(rows).sort("spec")
    man.write_csv(out_dir / "clinical_manifest.csv")

    by_cat = man.group_by("category").len().sort("category")
    print(f"wrote {len(len1)} clinical tasks at length 1 and length 3 -> {out_dir}")
    print(f"by query form: {dict(zip(by_cat['category'], by_cat['len'], strict=True))}")
    print(f"\n{'spec':<38}{'form':>6}{'anc':>5}{'ndesc':>7}  target concept / bound")
    for r in man.iter_rows(named=True):
        b = f" before {r['bound_concept']}" if r["bound_concept"] else f" @{r['horizon_days']}d"
        print(f"{r['spec']:<38}{r['category']:>6}{('Y' if r['is_ancestor'] else 'n'):>5}"
              f"{r['n_descendants']:>7}  {r['target_concept']}{b}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
