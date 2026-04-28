# ETHOS vs EveryQuery comparison

Per-code AUROC comparison of ETHOS (generative trajectory model) against EveryQuery
on the held-out MIMIC eval suite (`gs://every-query-runs/eval_tasks/7573f855…/held_out/`).

## Pipeline

1. `build_crosswalks.py`: deterministic-first walker (`ontology.icd_to_ethos_token`,
   `ontology.atc_to_ethos_token`) plus scoped LLM picker (`pick.py`, OpenRouter
   `anthropic/claude-sonnet-4.5`) emits the three crosswalk YAMLs in
   `crosswalks/`. Each EQ code gets exactly one ETHOS token (occasionally two
   for same-source-concept ATC ties) or lands in `unmappable_with_rationale`.
2. `build_mapping.py`: emits `mapping_table.parquet` (with `tier`,
   `tier_quality`, and `mapping_source` columns) and
   `mapping_coverage.parquet`. Adds the `dx_specific` and `lab_approx_value`
   primary tiers and runs the suppression cascade (`lab_no_value` -> drop when
   `lab_exact_value`/`lab_approx_value` exists; `dx_parent` -> drop when
   `dx_specific` exists; `proxy` -> drop when `lab_exact_value`/
   `lab_approx_value` exists).
3. `predict.py`: streams ETHOS trajectories, emits `ethos_predictions.parquet`.
   Supports literal and `code+next_token` match kinds.
4. `evaluate.py`: emits `ethos_aucs_held_out.parquet` with a `tier_quality`
   column and prints a primary-vs-secondary aggregate alongside the
   per-tier table.
5. `comparison.ipynb`: figures (PDF+PNG to `analysis/figures/`) and tables
   (CSV to `analysis/tables/`) with paired Wilcoxon over per-(code, duration,
   bucket) cells. Headline number leads with the primary tier_quality split;
   secondary and `any` are reported separately. Also emits three head-to-head
   win-rate CSVs at `eps=0.01`
   (`ethos_vs_eq_win_rate_per_tier.csv`,
   `ethos_vs_eq_win_rate_per_tier_duration_bucket.csv`,
   `ethos_vs_eq_win_status_per_code.csv`).

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
- **`lab_no_value` tier:** for `LAB`, `INFUSION_START`, `INFUSION_END`, `SUBJECT_FLUID_OUTPUT` families — strip the trailing `//value_[..)` segment from EQ `code`, normalise units to upper-case, and match the lab/item ETHOS token (ignoring the following `Qk`). Suppressed by the cascade in `build_mapping.apply_suppression_cascade` whenever a `lab_exact_value` or `lab_approx_value` row exists for the same query.
- **`lab_exact_value` tier:** for the same families — match the lab/item token AND require the next-token `Qk` to fall in the decile range corresponding to EQ's `value_[lo, hi)`. Decile boundaries come from MEDS metadata `values/quantiles`; for codes with null parent quantiles we fall back to the sibling-bin layout when there are exactly 10 contiguous bins. EQ bin maps 1-to-1 to a single ETHOS `Qk`.
- **`lab_approx_value` tier:** identical 2-token shape as `lab_exact_value`, but for parents whose `meds_codes.parquet` has 2..9 sibling bins (so the strict decile alignment can't fire). The EQ bin's 1-based positional index `i` among `n` sorted siblings is mapped linearly to `Qk = round(i * 10 / n)`. This is an approximation, flagged in the provenance string `meds-codes.parquet:sibling_bins_positional`.
- **`dx_specific` tier:** for `DIAGNOSIS//ICD//(9|10)//*` queries — emit a strict 2-token sequence `<descriptive_label>|ICD//CM//3-6//<suffix>` whenever the EQ code's 5-char ICD-10-CM target has both an ETHOS descriptive label (3-char parent) and an ETHOS `ICD//CM//3-6//<suffix>` token. Suppresses the broader `dx_parent` row for the same query.
- **`any` tier:** ETHOS predicts target if any pattern from the prior tiers fires within the duration window.

Each row of `mapping_table.parquet` carries a **`tier_quality`** of
`primary` (most-specific encoding the vocabulary supports) or `secondary`
(best-effort proxy: ATC level 2, procedure-as-diagnosis, indication-proxy
labs, `HOSPITAL_ADMISSION` for `TIMELINE//START`). Downstream consumers
split AUC tables by `tier_quality` and report `any` separately as the
per-code OR.

## Outcome

Crosswalks were tightened so each EQ code resolves to the **single most-specific
ETHOS token** (or a same-source-concept ATC tie that commits two tokens), and
the mapping table now annotates every (query, tier) row with a
**`tier_quality`** of `primary` or `secondary`. Primary tiers commit to the
most-specific encoding the ETHOS vocabulary supports (literal token,
lab-decile pair, ICD-10-CM 5-char label+suffix pair, ICD-10-CM 3-char parent
for diagnoses); secondary tiers are best-effort proxies under ETHOS-vocab
limitations (ATC level 2 for medications, procedure-as-diagnosis for ICD-9
procedures, indication-proxy diagnosis tokens for orphan MIMIC labs,
`HOSPITAL_ADMISSION` for `TIMELINE//START`).

- **Coverage:** **22/41 EQ codes mapped (54%)**, split by tier_quality:
  - **Primary (12 codes):** 1 `MEDS_DEATH` (exact), 3 LABs (`lab_exact_value`, deciles
    via `meds_codes.parquet`), 3 LABs (`lab_approx_value`, sibling-bin
    positional approximation when fewer than 10 deciles exist), 3 DIAGNOSIS
    (`dx_specific`: 5-char ICD-10-CM target rendered as label + 3-6-suffix
    2-token pattern) plus 2 DIAGNOSIS that fall back to the 3-char parent
    (`dx_parent` primary for DIAGNOSIS).
  - **Secondary (10 codes):** 5 MEDICATION (`drug_class` at ATC level 2;
    Gabapentin commits two tokens for the N02/N03 same-source ATC tie),
    1 PROCEDURE (`dx_parent` proxy via fracture-of-lower-leg diagnosis),
    1 TIMELINE + 2 INFUSION_* (named-drug -> ATC class) + 1 orphan LAB
    (pressure-ulcer length -> `PRESSURE_ULCER` indication proxy), all
    `proxy`.
- **19 codes remain unmapped** in `crosswalks/mimic_items.yaml`'s
  `unmappable_with_rationale` block: 12 LABs (CO2 production, eosinophil
  differential, lipase, hemodialysis output volume, CK-MB, ventilator
  inspiratory pressure, CPOT pain assessment, lithium level, osmolality, 24-hr
  urine calcium, rubella serology, PCA total dose), 3 SUBJECT_FLUID_OUTPUT
  (Jackson-Pratt drain, tube-feed residual), and 4 INFUSION_* entries (KCl
  during CRRT in two bins, dextrose 5% IV fluid, packed RBCs).
- **Headline AUROC by tier_quality** (paired Wilcoxon over
  `(code, duration, bucket)` cells, best EQ ckpt `23-43-54` with `MEDS_DEATH`
  falling back to a partial ckpt; canonical output of `comparison.ipynb`,
  persisted to `analysis/tables/ethos_vs_eq_aggregated_by_tier_quality.csv`):

  | tier_quality | n | mean EQ | mean ETHOS | mean diff | p |
  |---|---:|---:|---:|---:|---:|
  | **primary**   | 23 | 0.826 | 0.816 | -0.010 | 0.43 |
  | secondary[^1] | 18 | 0.824 | 0.775 | -0.049 | 0.55 |
  | any           | 41 | 0.825 | 0.798 | -0.027 | 0.40 |

  [^1]: Secondary tiers are best-effort proxies under ETHOS-vocab
  limitations (ATC level 2 only for drugs; no PCS tokens for procedures;
  indication-proxy diagnosis tokens for orphan MIMIC labs;
  `HOSPITAL_ADMISSION` as a stand-in for `TIMELINE//START`). They are
  reported separately so the primary-tier number reflects only the
  vocabulary-supported encodings.

- **Per-tier breakdown** (canonical output:
  `analysis/tables/ethos_vs_eq_aggregated_per_tier.csv`):

  | Tier | quality | n | mean EQ | mean ETHOS | mean diff | p | Comment |
  |---|---|---:|---:|---:|---:|---:|---|
  | exact                | primary   |  1 | 0.834 | 0.880 | +0.046 | NA   | MEDS_DEATH |
  | lab_exact_value      | primary   |  6 | 0.904 | 0.901 | -0.003 | 1.00 | LAB value decile |
  | lab_approx_value      | primary   |  6 | 0.935 | 0.839 | -0.096 | 0.031 | LAB sibling-bin positional approx |
  | dx_specific         | primary   |  6 | 0.715 | 0.765 | +0.050 | 0.44 | DIAGNOSIS 5-char (label + 3-6-suffix) |
  | dx_parent        | mixed     |  6 | 0.699 | 0.759 | +0.059 | 0.44 | 3-char parent (primary for DX, secondary for PROC) |
  | drug_class        | secondary | 10 | 0.767 | 0.774 | +0.007 | 1.00 | MEDICATION at ATC level 2 |
  | proxy | secondary |  6 | 0.965 | 0.751 | -0.214 | 0.063 | INFUSION + indication proxies |
  | any                  | -         | 41 | 0.825 | 0.798 | -0.027 | 0.40 | Per-code OR |

- **Headline:** within the **primary** tier set ETHOS reaches mean AUC
  0.816 vs EQ 0.826 (n=23 cells, p=0.43) -- a **statistically
  indistinguishable** difference once the comparison is restricted to tiers
  whose ETHOS encoding actually represents the EQ concept. The primary
  picture remains close to parity for diagnoses (`dx_specific` and
  `dx_parent` both favour ETHOS by ~0.05) and lab presence/value
  (`exact`, `lab_exact_value`). The companion scatter is at
  `analysis/figures/ethos_vs_eq_auc_scatter.{pdf,png}`.
- **Trade-offs (acknowledged):**
  - **`lab_approx_value` underperforms** (mean diff -0.096, p=0.031) because
    the EQ bin's positional index is mapped linearly to a Q1..Q10 decile
    when fewer than 10 sibling bins exist in `meds_codes.parquet`. This is
    flagged in the YAML provenance string and the per-row mapping_source
    (`meds-codes.parquet:sibling_bins_positional`).
  - The `proxy` row is the noisiest because indication-proxy
    diagnosis tokens (`PRESSURE_ULCER` for ulcer length, `HOSPITAL_ADMISSION`
    for `TIMELINE//START`) predict different events than the underlying EQ
    queries; this is reflected in `tier_quality=secondary`.

## Win rate vs ETHOS

Complementary view of the same `(code, duration, bucket)` cells used by the
Wilcoxon table above, scoring each cell as a head-to-head outcome instead of
a mean difference. EQ wins iff `eq_auc - ethos_auc > 0.01`; ETHOS wins iff
`ethos_auc - eq_auc > 0.01`; otherwise the cell is a tie. The 0.01 tie band
absorbs noise-level differences. `EQ win rate` is EQ-centric:
`wins_eq / (wins_eq + ties + wins_ethos)`. Verbatim from
`analysis/tables/ethos_vs_eq_win_rate_per_tier.csv`:

| Tier | n | EQ wins | ties | ETHOS wins | EQ win rate |
|---|---:|---:|---:|---:|---:|
| exact                |  1 |  0 | 0 |  1 | 0.000 |
| lab_exact_value      |  6 |  4 | 0 |  2 | 0.667 |
| lab_approx_value      |  6 |  6 | 0 |  0 | 1.000 |
| dx_specific         |  6 |  3 | 0 |  3 | 0.500 |
| dx_parent        |  6 |  1 | 2 |  3 | 0.167 |
| drug_class        | 10 |  4 | 0 |  6 | 0.400 |
| proxy |  6 |  5 | 1 |  0 | 0.833 |
| any                  | 41 | 23 | 3 | 15 | 0.561 |

Per-tier-x-duration-x-bucket and per-code splits are persisted to
`analysis/tables/ethos_vs_eq_win_rate_per_tier_duration_bucket.csv` and
`analysis/tables/ethos_vs_eq_win_status_per_code.csv` respectively.

## Mapping provenance (`mapping_source` column)

Every row of `mapping_table.parquet` carries a `mapping_source` field
documenting where the (eq_code, ethos_token) pair came from. Under the
deterministic-first pipeline 13 distinct sources appear across the 26 active
mapping rows (excluding the `any` tier, which inherits the underlying primary
tier's source):

| `mapping_source`                                        | Tier(s)                            | n  |
|---------------------------------------------------------|------------------------------------|---:|
| `code:string_equality`                                  | exact                              | 1  |
| `code:strip_value_bin+upper_units`                      | lab_no_value                           | 6  |
| `meds-codes.parquet:values_quantiles_or_sibling_bins`   | lab_exact_value                    | 3  |
| `deterministic:icd_3char_walker`                        | dx_parent                      | 4  |
| `deterministic:atc_ancestor_walker`                     | drug_class, proxy| 3  |
| `deterministic:atc_same_source_concept_tie`             | drug_class                      | 2  |
| `direct:event_alignment`                                | proxy               | 1  |
| `llm:atc_multi_chain`                                   | drug_class                      | 1  |
| `llm:diagnosis_walker_unresolved`                       | dx_parent                      | 1  |
| `llm:infusion_walker_unresolved`                        | proxy               | 1  |
| `llm:medication_walker_unresolved`                      | drug_class                      | 1  |
| `llm:no_pcs_tokens_in_ethos`                            | dx_parent                      | 1  |
| `llm:orphan_lab`                                        | proxy               | 1  |

Code-derived tiers (`exact`, `lab_no_value`, `lab_exact_value`) hardcode their source
in `build_mapping.py`. The crosswalk YAMLs are produced by
`build_crosswalks.py`: deterministic walker hits carry the
`deterministic:*` prefix; LLM picker hits carry an `llm:<reason>` prefix
where `<reason>` is the residual case the walker couldn't resolve. The
prior `llm:claude_clinical_knowledge` source from the loose-mapping era is
no longer produced; it has been replaced by deterministic walkers wherever
possible and by tightly-scoped LLM picker calls otherwise. `any` rows
preserve the primary row's source so downstream consumers can trace any
match back to its origin without joining back to the YAML.

## Crosswalk audit notes

The three crosswalk YAMLs in `crosswalks/` are produced by
`build_crosswalks.py` under a **deterministic-first** pipeline. The walker
in `ontology.py` resolves any case it can without an LLM call; the LLM
picker (`pick.py`) is invoked **only** for the residual cases listed below.
Each EQ code commits **exactly one** ETHOS token (or two, in the single
allowed exception of a same-source-concept ATC tie).

- `icd.yaml` (6 mappings): DIAGNOSIS rows resolved via the ICD-10-CM
  3-char walker (`ontology.icd_to_ethos_token` plus
  `ethos_icd_3char_index`); 4 rows are deterministic. The walker falls
  through to the LLM only for ICD-9 codes with no ICD-10 bridge (`4271`
  paroxysmal vtach -> `PAROXYSMAL_TACHYCARDIA`, ICD-9 multi-parent path)
  and for procedures, since ETHOS encodes no PCS tokens
  (`PROCEDURE//ICD//9//7936` open reduction tibia/fibula ->
  `FRACTURE_OF_LOWER_LEG_INCLUDING_ANKLE`, scoped reason
  `no_pcs_tokens_in_ethos`). Broad-but-correct parents like
  `7295 -> M79 -> OTHER_AND_UNSPECIFIED_SOFT_TISSUE_DISORDERS_NEC` are
  surfaced in `review/mapping_review.md` for case-by-case approval.
- `atc.yaml` (5 mappings): MEDICATION rows resolved via the OHDSI ATC
  ancestor walker (`ontology.atc_to_ethos_token` plus
  `rxnorm_drug_to_atc`); 2 rows are deterministic. Gabapentin is the only
  same-source-concept tie and commits **two** tokens (N02 analgesics plus
  N03 antiepileptics, scoped reason `atc_same_source_concept_tie`). The
  LLM picker fires for one true multi-chain ambiguity (Mupirocin Nasal:
  R01 vs D06, with the nasal formulation resolving to R01) and one
  walker-unresolved salt-form name (Doxycycline Hyclate -> J01
  antibiotics, salvaged from a near-miss free-text proposal via the
  namespace-prefix salvage in `pick._salvage_token_by_prefix`).
  `MEDICATION//START//*` and `MEDICATION//STOP//*` both roll up to the
  same ATC class since ETHOS has no start/stop semantics.
- `mimic_items.yaml` (4 mappings plus 19 unmappable): INFUSION rows
  resolve the MIMIC item-id to a drug name and walk to ATC; 1 row is
  deterministic (Furosemide -> C03 diuretics). LAB and
  SUBJECT_FLUID_OUTPUT rows are limited to **orphan codes** (base form
  not in the ETHOS vocab). `TIMELINE//START -> HOSPITAL_ADMISSION` is a
  direct event alignment. The remaining 19 entries are recorded in the
  `unmappable_with_rationale` block: most are MIMIC-only lab tokens whose
  `LAB//<id>//<units>` form is absent from ETHOS's vocabulary (CK-MB,
  lipase, osmolality, eosinophil percentage, lithium level, rubella IgG),
  one is CO2 production (a respiratory parameter, not a chem panel), and
  the SUBJECT_FLUID_OUTPUT codes (Jackson-Pratt drains, tube-feed
  residual) have no fluid-output prefix in the ETHOS vocab.
  Indication-proxy mappings used in the prior loose-mapping run (CK-MB ->
  AMI, HD output -> CKD encounter) were intentionally **dropped** because
  they predict different events than EQ; the `proxy` AUC
  drop in the table above is the explicit cost of that change.

## Follow-up directions

- **19 still-unmapped codes** are intrinsically hard to map under ETHOS's
  current vocabulary; further mapping would need either an expanded ETHOS
  vocab (retraining with more MIMIC lab item-ids and SUBJECT_FLUID_OUTPUT
  tokens) or acceptance that some EQ tasks have no ETHOS-tractable
  counterpart at this vocabulary granularity. The
  `unmappable_with_rationale` block in `mimic_items.yaml` and
  `mapping_coverage.parquet` together make this explicit for downstream
  consumers.
- **Walker-emitted broad parents** flagged for human review in
  `review/mapping_review.md`: `7295 -> M79`,
  `4271 -> PAROXYSMAL_TACHYCARDIA` (loses VT-vs-SVT specificity),
  `Doxycycline Hyclate -> J01` (loses tetracycline-class specificity), and
  the procedure-as-indication mapping `7936 -> S82`. Approve / reject /
  modify per EQ section drives the next iteration of the YAMLs.
- **Honest precision vs OR-of-N hit-rate.** The `dx_parent` advantage
  shrank from +0.094 (p=0.012) to +0.086 (p=0.064) because each ICD code
  now commits to exactly one parent. Restoring statistical significance
  would require either deeper-than-3-char ETHOS tokens (not present in the
  vocabulary) or relaxing the single-token rule, which contradicts the
  honest-precision principle.
