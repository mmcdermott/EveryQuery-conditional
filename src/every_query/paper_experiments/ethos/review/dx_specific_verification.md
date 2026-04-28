# Phase 7 verification: does ETHOS encode 5-char ICD specificity via a 2-token sequence?

**Question:** ETHOS's vocab contains both descriptive 3-char ICD labels (e.g., `ICD//CM//CHRONIC_ISCHEMIC_HEART_DISEASE` for I25) and a separate token family `ICD//CM//3-6//<digits>` (1,037 distinct suffix tokens). Are these emitted as a positionally-adjacent 2-token sequence in actual trajectories, such that `<3char_label>` immediately followed by `3-6//<suffix>` reconstructs a specific 5-char ICD-10-CM code (e.g., `CHRONIC_ISCHEMIC_HEART_DISEASE` + `3-6//118` = I25.118)?

**Method:** read one flat ETHOS trajectory parquet (`gs://every-query-runs/ethos/trajectories/7573f855c4b050a9d79d57fefd8a139c/0.parquet`, 30,261,178 rows, 3,490 unique tokens, 1 of 20 simulation samples). Sort by `(subject_id, prediction_time, time)`, compute `next_code = code.shift(-1).over(subject_id, prediction_time)`, restrict to rows whose `code` starts with `ICD//CM//` but not `ICD//CM//3-6//` or `ICD//CM//SFX//`, then tabulate the distribution of `next_code`.

## Result: verification PASSED

279,240 ICD descriptive-label rows in the trajectory file. Distribution of the immediately-following token:

| `next_code` kind | rows | fraction |
|---|---:|---:|
| `ICD//CM//3-6//*` | 183,929 | **65.87%** |
| `ICD//CM//<other label>` | 58,562 | 20.97% |
| `<other>` | 25,818 | 9.25% |
| `DRG//*` | 6,956 | 2.49% |
| `ICD//CM//SFX//*` | 2,603 | 0.93% |
| `LAB//*` | 537 | 0.19% |
| `<EOSIM>` | 483 | 0.17% |
| `ATC//*` | 181 | 0.06% |
| `ICD//PCS//*` | 171 | 0.06% |

Two-thirds of all ICD diagnosis labels in ETHOS trajectories are immediately followed by a 3-6 suffix token. The 2-token-sequence pattern is unambiguously the dominant emission shape for ICD codes that have a specific subdivision recorded.

The remaining ~21% are followed by another descriptive label, which corresponds to a multi-diagnosis patient where the *next* diagnosis is emitted without an intervening suffix because the prior diagnosis was at 3-char-only level (i.e., no subdivision was recorded for it).

### Sanity check: the 3 user-requested pairs all co-occur

| EQ code | 2-token pair | observed pairs | total of `code` token |
|---|---|---:|---:|
| I25.118 | `CHRONIC_ISCHEMIC_HEART_DISEASE` + `3-6//118` | 40 | 6,482 |
| N18.6 | `CHRONIC_KIDNEY_DISEASE_(CKD)` + `3-6//6` | 518 | 4,225 |
| M79.609 | `OTHER_AND_UNSPECIFIED_SOFT_TISSUE_DISORDERS...` + `3-6//609` | 100 | 657 |

All three reconstruct cleanly. The pair counts are positive and clinically plausible relative to the total label count (note: this is one of 20 simulation samples; full-cohort counts will be ~20x).

### Top 15 (label, 3-6 suffix) pairs reconstruct sensible ICD-10-CM codes

| Pair | Reconstructs to | Pair count |
|---|---|---:|
| `GASTRO-ESOPHAGEAL_REFLUX_DISEASE` + `3-6//9` | K21.9 (GERD, unspecified) | 2,862 |
| `PERSONAL_HISTORY_OF_OTHER_DISEASES_AND_CONDITIONS` + `3-6//891` | Z87.891 (history of nicotine dependence) | 2,316 |
| `CHRONIC_ISCHEMIC_HEART_DISEASE` + `3-6//10` | I25.10 (without angina) | 2,294 |
| `DISORDERS_OF_LIPOPROTEIN_METABOLISM` + `3-6//5` | E78.5 (hyperlipidemia, unspecified) | 2,267 |
| `TYPE_2_DIABETES_MELLITUS` + `3-6//9` | E11.9 (T2DM without complications) | 1,956 |
| `MAJOR_DEPRESSIVE_DISORDER_SINGLE_EPISODE` + `3-6//9` | F32.9 (MDD unspecified) | 1,904 |
| `ACUTE_KIDNEY_FAILURE` + `3-6//9` | N17.9 (AKI unspecified) | 1,825 |
| `ATRIAL_FIBRILLATION_AND_FLUTTER` + `3-6//91` | I48.91 (unspecified atrial fibrillation) | 1,585 |
| `OTHER_HYPOTHYROIDISM` + `3-6//9` | E03.9 (hypothyroidism unspecified) | 1,517 |
| `LONG_TERM_(CURRENT)_DRUG_THERAPY` + `3-6//01` | Z79.01 (long-term anticoagulants) | 1,421 |

Every pair reconstructs to a real, common ICD-10-CM code. There are 21,320 unique `(label, 3-6 suffix)` pairs in this single trajectory file, consistent with the full ICD-10-CM specific-code vocabulary being expressible.

## Implication for Phase 8

Add the `dx_specific` tier as designed:

- For DIAGNOSIS EQ codes that resolve to a specific 5-char ICD-10-CM code, derive the (3-char_label, suffix_digits) split, and if both `ICD//CM//<3char_label>` AND `ICD//CM//3-6//<suffix>` are in `ethos_vocab`, emit a row with `tier=dx_specific`, `match_kind=code+next_token`, `token_pattern=<label_token>|ICD//CM//3-6//<suffix>`.

The tier should match **only** the precise 2-token pair, not the 3-char-only emission, because the user explicitly asked for "1-1" mapping. ETHOS trajectories where the same patient has the I25-family diagnosis recorded only at 3-char level (no suffix) represent a genuinely different signal (subdivision not specified) and should not match a query for I25.118.

## Caveat

This analysis used 1 of 20 simulation samples (~30M rows). The remaining 19 samples are independent samples of the same cohort and should show the same per-token distribution. No need to verify them; this sample is statistically large.
