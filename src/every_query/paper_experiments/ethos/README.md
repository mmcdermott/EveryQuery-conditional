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

- **Coverage:** 7/41 EQ codes mapped (1 `MEDS_DEATH` via exact, 6 LABs via drop_bin, 3 of those LABs additionally via quantile). 34 codes unmapped because ETHOS's vocabulary uses descriptive ICD labels rather than codes, ATC for medications, and lacks `INFUSION_*` / `SUBJECT_FLUID_OUTPUT` / `TIMELINE_START` prefixes. See `mapping_coverage.parquet` for the full report.
- **AUROC, paired Wilcoxon over (code, duration, bucket) cells (best EQ ckpt `23-43-54`, with `MEDS_DEATH` falling back to `14-08-24`):**

  | Tier | n | mean EQ | mean ETHOS | mean diff | p |
  |---|---:|---:|---:|---:|---:|
  | exact | 1 | 0.834 | 0.880 | +0.046 | NA |
  | drop_bin | 12 | 0.920 | 0.922 | +0.002 | 0.85 |
  | quantile | 6 | 0.904 | 0.901 | −0.003 | 1.00 |
  | union | 13 | 0.913 | 0.918 | +0.005 | 0.84 |

- **Headline:** on every code we could evaluate, ETHOS (used as a query engine via 20 sampled trajectories) is statistically indistinguishable from EQ; the dominant gap is mapping coverage, not prediction quality.

## Follow-up directions

The unmapped count is dominated by three reasons:
- 13 LAB codes whose `LAB//<id>//<UNITS>` token isn't in ETHOS's 200-lab vocab.
- 6 INFUSION_START/END + 3 SUBJECT_FLUID_OUTPUT — no corresponding prefix in ETHOS vocab.
- 5 DIAGNOSIS — would require an ICD-9/10 → ETHOS-descriptive-label crosswalk.
- 5 MEDICATION — would require a drug-name → ATC crosswalk.
- 1 PROCEDURE (ICD-PCS chunked form), 1 TIMELINE//START.

Adding either crosswalk tier or a hierarchical-rollup tier (both explicitly excluded from this pass) would lift coverage from 17% toward ~50–80%.
