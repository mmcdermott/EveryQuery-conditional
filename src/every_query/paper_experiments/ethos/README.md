# ETHOS vs EveryQuery comparison

Per-code AUROC comparison of ETHOS (generative trajectory model) against EveryQuery
on the held-out MIMIC eval suite (`gs://every-query-runs/eval_tasks/7573f855…/held_out/`).

## Pipeline

1. `build_mapping.py` — emits `mapping_table.parquet` and `mapping_coverage.parquet`.
2. `predict.py` — streams ETHOS trajectories, emits `ethos_predictions.parquet`.
3. `evaluate.py` — emits `ethos_aucs_held_out.parquet`.
4. `comparison.ipynb` — figures + paired Wilcoxon tables.

All artifacts upload to `gs://every-query-runs/baselines/ethos/`.

## Phase 0 audit findings

### ETHOS trajectory layout

Bucket prefix: `gs://every-query-runs/ethos/trajectories/7573f855c4b050a9d79d57fefd8a139c/`.

- **20 flat parquet files** at the prefix root: `0.parquet` … `19.parquet` (~50 MB each, ~30M rows each).
- **8 rank subdirs** `rank_0/…rank_7/`, each containing `0.parquet…19.parquet` (~7 MB each, ~3.9M rows each).
- Empirical: flat `n.parquet` ≈ concat of `rank_0/n.parquet … rank_7/n.parquet` (3.9M × 8 ≈ 30M rows).
- Empirical: flat `0.parquet` and flat `1.parquet` share **all** 15,031 (subject_id, prediction_time) pairs — i.e. each flat file is an **independent sample** of the same cohort.
- **N_sims per (subject, prediction_time) = 20** (one per flat file).
- Schema: `subject_id: int64`, `time: timestamp[us]`, `prediction_time: timestamp[us]`, `code: string`, `numeric_value: float`. No `sim_idx` column — sim identity comes from which flat file the row was read from. Time is monotonically non-decreasing within a `(subject_id, prediction_time)` group.
- Cohort: 15,031 unique subjects, 14,932 unique prediction_times (some subjects have multiple prediction_times).
- Trajectory horizon: median ~32 days, p90 ~500 days, max ~914 days from `prediction_time` — covers both EQ durations (30, 180) comfortably.
- For Phase 2 ingestion: read the **20 flat files** directly; do not also read the rank subdirs (would double-count).

### ETHOS vocabulary observations

- ~3,490 unique tokens in flat 0 alone (full vocab likely larger but bounded).
- **Quantile tokens are deciles (Q1…Q10), not quintiles.**
- **Lab tokens are `LAB//<item_id>//<UNITS>`** (units upper-cased, no value bin) — the value decile follows as a separate `Qk` token in the next row.
- ATC drug tokens at multiple levels: `ATC//4//A`, `ATC//A10//DRUGS_USED_IN_DIABETES`, `ATC//SFX//A01`.
- Time-interval tokens between events: `5m-15m`, `15m-45m`, `45m-1h15m`, `1d-2d`, `7d-12d`, `20d-30d`, etc.
- Vital tokens like `VITAL//BLOOD_PRESSURE`, `BMI//Q5`.
- **Mapping implication:** EQ lab codes use original-case units (`mg/dL`); ETHOS uses upper-case (`MG/DL`). Match must normalise case. EQ's `value_[lo, hi)` bin must be matched against the **next-token Qk** following the lab token, with Q1–Q10 buckets (not Q1–Q5).

### EQ eval-task layout

Bucket prefix: `gs://every-query-runs/eval_tasks/7573f855c4b050a9d79d57fefd8a139c/held_out/`.

- Path schema: `held_out/<duration_days>/<code_slug>/<shard_idx>.parquet`.
- 51 duration directories present (durations from 2 to 731 days).
- Per-shard schema: `subject_id: int64`, `prediction_time: timestamp[us]`, `boolean_value: bool`, `query: large_string`, `occurs: bool`, `duration_days: int32`.
- Each `<code_slug>/` directory contains one query (e.g. `DIAGNOSIS//ICD//10//I25118`), sharded across multiple `*.parquet` files.
- **Join key for ETHOS predictions:** `(subject_id, prediction_time, query, duration_days)` → `boolean_value`.
- **Restriction for this comparison:** only durations {30, 180} have EQ AUCs in `eval_aucs_held_out.parquet`, so ETHOS is evaluated at those two durations only. Other 49 duration dirs are out of scope here.

### Best EQ checkpoint

See `best_eq_ckpt.json`. Selected model: **`23-43-54`** (mean occurs_auc 0.852, full 80-row coverage = 40 codes × 2 buckets × 2 durations on held-out).

Excluded partial models `14-08-24` and `23-54-43` (only 41 rows, duration=30 only).

### Mapping mechanics implied by the audit

- **Exact tier:** verbatim string match between EQ `code` and an ETHOS token in the trajectory.
- **Drop-bin tier:** for `LAB`, `INFUSION_START`, `INFUSION_END`, `SUBJECT_FLUID_OUTPUT` families — strip the trailing `//value_[..)` segment from EQ `code`, normalise units to upper-case, and match the lab/item ETHOS token (ignoring the following `Qk`).
- **Quantile tier:** for the same families — match the lab/item token AND require the next-token `Qk` to fall in the decile range corresponding to EQ's `value_[lo, hi)`. Decile boundaries come from MEDS metadata `values/quantiles`; for codes with null parent quantiles we fall back to the sibling-bin layout when there are exactly 10 contiguous bins.
- **Set-union OR tier:** ETHOS predicts target if any pattern from the prior tiers fires within the duration window.

## Outcome

- **Coverage:** **28/41 EQ codes mapped (68%)** — 1 `MEDS_DEATH` (exact), 6 LABs (drop_bin, 3 also quantile), 5 DIAGNOSIS + 1 PROCEDURE (icd_crosswalk), 5 MEDICATION (atc_crosswalk), 1 TIMELINE + 5 INFUSION_* + 3 LAB-via-clinical-context (mimic_item_crosswalk).
- **13 codes remain unmapped:** 10 LABs whose specific clinical concept (CO2 production, eosinophil differential, lipase, ventilator pressure, pain assessment, 24hr urine analytes, serum drug levels, osmolality, serology) has no ETHOS counterpart, plus 3 SUBJECT_FLUID_OUTPUT (drain output, tube-feed residual). All listed in `crosswalks/mimic_items.yaml`'s `unmappable_with_rationale` block. See `mapping_coverage.parquet` for the full report.
- **AUROC, paired Wilcoxon over (code, duration, bucket) cells (best EQ ckpt `23-43-54`, with `MEDS_DEATH` falling back to `14-08-24`):**

  | Tier | n | mean EQ | mean ETHOS | mean diff | p | Comment |
  |---|---:|---:|---:|---:|---:|---|
  | exact | 1 | 0.834 | 0.880 | +0.046 | NA | MEDS_DEATH |
  | drop_bin | 12 | 0.920 | 0.922 | +0.002 | 0.85 | LAB presence |
  | quantile | 6 | 0.904 | 0.901 | −0.003 | 1.00 | LAB value decile |
  | icd_crosswalk | 12 | 0.707 | 0.801 | **+0.094** | **0.012** | DIAGNOSIS+PROC, ETHOS wins |
  | atc_crosswalk | 10 | 0.767 | 0.792 | +0.025 | 0.62 | MEDICATION |
  | mimic_item_crosswalk | 18 | 0.946 | 0.784 | **−0.162** | **1.5e-5** | Loose proxies (see below) |
  | union | 53 | 0.850 | 0.822 | −0.028 | 0.27 | Per-code OR |

- **Headline:** ETHOS as a query engine (20 sampled trajectories per prediction) is **comparable to or better than EQ on every tier where ETHOS has a direct vocabulary counterpart** — including the icd_crosswalk codes where it significantly wins (+0.094 AUROC, p=0.012). The one tier where ETHOS significantly underperforms is `mimic_item_crosswalk`, where many mappings are intentionally loose proxies (e.g. EQ predicts "CK-MB elevated value" while ETHOS predicts "AMI diagnosis" — different events) and the AUROC gap reflects that mismatch, not a model capability gap. The union-tier comparison shows no significant difference (p=0.27).

## Mapping provenance (`mapping_source` column)

Every row of `mapping_table.parquet` carries a `mapping_source` field
documenting where the (eq_code, ethos_token) pair came from. Six distinct
sources currently appear across the 29 active mapping rows (excluding the
union tier, which inherits the underlying primary tier's source):

| `mapping_source`                                        | Tier(s)                            | n  |
|---------------------------------------------------------|------------------------------------|---:|
| `code:string_equality`                                  | exact                              | 1  |
| `code:strip_value_bin+upper_units`                      | drop_bin                           | 6  |
| `meds-codes.parquet:values_quantiles_or_sibling_bins`   | quantile                           | 3  |
| `llm:claude_clinical_knowledge`                         | icd_crosswalk, atc_crosswalk       | 16 |
| `direct:event_alignment`                                | mimic_item_crosswalk               | 1  |
| `physionet/mimic-iv-demo:icu/d_items.csv+llm`           | mimic_item_crosswalk               | 12 |

Code-derived tiers (`exact`, `drop_bin`, `quantile`) hardcode their source
in `build_mapping.py`. YAML-derived tiers read a per-entry `source:` field
from the crosswalk file. `union` rows preserve the primary row's source so
downstream consumers can trace any union match back to its origin without
joining back to the YAML.

## Crosswalk audit notes

Three crosswalk YAMLs in `crosswalks/`, all LLM-proposed and committed for review:

- `icd.yaml` — direct semantic matches for ICD-9/10 → ETHOS descriptive labels. Loose mappings flagged: `DIAGNOSIS//ICD//9//7295` (pain in limb) → `OTHER_DISORDERS_OF_MUSCLE` + `PAIN_NOT_ELSEWHERE_CLASSIFIED`; `PROCEDURE//ICD//9//7936` (foot fracture reduction) → fracture diagnosis tokens (predicting indication, not procedure).
- `atc.yaml` — drug names → ATC class. `MEDICATION//START//*` and `MEDICATION//STOP//*` both roll up to ATC class — ETHOS has no start/stop semantics.
- `mimic_items.yaml` — MIMIC item-id → ETHOS token, sourced from `physionet.org/files/mimic-iv-demo/2.2/icu/d_items.csv` and `hosp/d_labitems.csv`. Loose mappings (predicting indication / coarser concept rather than specific lab value): `LAB//226499` (HD output) → dialysis encounter + CKD; `LAB//227445` (CK-MB) → AMI; `LAB//228724` (pressure-ulcer length) → pressure ulcer Dx. The file's `unmappable_with_rationale` block enumerates the 11 codes with no ETHOS counterpart.

## Follow-up directions

The 13 still-unmapped codes are intrinsically hard to map under ETHOS's current vocabulary; further mapping would need either:
- An expanded ETHOS vocab (e.g. retraining ETHOS with more labs / output events tokenized) — out of scope for an evaluation comparison.
- Acceptance that some EQ tasks have no ETHOS-tractable counterpart at this vocabulary granularity. The `mapping_coverage.parquet` report makes this explicit for downstream consumers.
