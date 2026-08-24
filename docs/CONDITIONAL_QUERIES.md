# Conditional query-sequence extension to EveryQuery

_Branch: `conditional-queries`._

This branch reworks EveryQuery from **independent single-query prediction** into a
**conditional query-sequence model**: a bidirectional encoder over the patient record feeds a
**block-autoregressive decoder** that answers an *ordered list* of queries, where each answer is
conditioned on the patient state **and the teacher-forced answers of all earlier queries** in the
sequence. The model therefore learns `P(A_j | patient, Q_1..Q_j, A_1..A_{j-1})` rather than the
marginal `P(A | patient, Q)`.

The headline finding: trained on enough independent random query sequences, the model's per-task
discriminability **increases monotonically with sequence position** — later queries, which see
more prior (query, answer) context, are answered measurably better — confirming the model uses the
conditional structure (see [Results](#results)).

---

## 1. Model design

`src/every_query/model/conditional_model.py` — `ConditionalQueryModel`

- **Patient encoder.** A ModernBERT encoder (bidirectional) embeds the tokenized MEDS event
  sequence up to the prediction time `t`. No query token is mixed into the patient sequence; the
  patient state is encoded once and shared across all queries via cross-attention.
- **Query decoder.** A `nn.TransformerDecoder` runs over the query/answer stream. Each query is a
  block of three tokens — `code`, `duration`, and a teacher-forced `answer` token. A
  **block-causal mask** (`build_block_causal_mask`) enforces: a query's code/duration see each
  other and *all* tokens of strictly earlier blocks (including their answers), but never the
  query's own answer token and never any later block. The answer for block `j` is read from the
  decoder output at that block's duration token, so it is conditioned on the patient state, all
  prior queries and their (ground-truth, teacher-forced) answers, and `Q_j` itself — never `A_j`.
- **Position encoding.** The encoder uses ModernBERT's native positions. The decoder adds learned
  **block-position** embeddings (which query block) + **token-type** embeddings (code/duration/
  answer role); `nn.TransformerDecoder` adds none itself.
- **Binary answers.** Every query asks *"is `code` observed in `(t, t+d)`?"* and the answer is
  binary YES/NO. There is no separate "censored" answer class and nothing is masked from the loss
  except padding. Loss is one BCE over all real query positions.

### Censoring is expressed as a query, not a label

`TIMELINE//END` (the real MEDS end-of-timeline code, emitted once per subject at the record's last
event) is queried like any other code. `(TIMELINE//END, d)` answered YES means "the record ends
within `d`" (the `d`-window is not fully observed); NO means data continue past `t+d`. Conditioning
a later query on that answer **recovers and generalizes** the original EveryQuery's implicit
`P(occurs | data exist after d)`:

- `[END d]=NO,  [C d]` → `P(C | data continue past d)`  (= original EveryQuery; ≈0 for terminal codes)
- `[END d]=YES, [C d]` → `P(C | record ends within d)`  (the actionable form for death etc.)
- `[C d]` alone → the marginal `P(C observed)`; recoverable as a prevalence-weighted average.

This is **strictly more expressive** than the original EveryQuery, which could only ever express
the `END=NO` slice.

> **v1 leak post-mortem.** An earlier design put a *same-horizon* censor query first, teacher-forced
> its answer, and 3-valued-labeled subsequent queries with censored outcomes masked from the loss.
> For terminal events this is catastrophic: death ends the record, so "data after t+30d?" is the
> logical complement of "died by 30d?", and the masking left the surviving death labels perfectly
> determined by the censor answer. The model learned to copy it — 30-day mortality AUROC hit 0.991,
> of which **0.996 came from the censor answer alone**. The binary-occurrence + END-as-query design
> here removes that leak structurally.

---

## 2. Pipeline & CLIs

The conditional pipeline mirrors the single-query one. New console scripts (in `pyproject.toml`):

| CLI | Purpose |
|---|---|
| `EQ_generate_query_sequences` | Sample fully-random query sequences per patient context; binary-label every query (`QuerySeqSchema` parquets). |
| `EQ_train --config-name=conditional_config` | Train `ConditionalQueryModel` (`conditional_lightning.ConditionalQueryLightningModule`). |
| `EQ_predict_sequences` | Teacher-forced per-position inference → flat per-query-position parquet. |
| `EQ_evaluate_sequences` | Per-position and per-(query, horizon) metric tables. |

Key source modules:

- `src/every_query/data/seq_dataset.py` — `ConditionalQueryPytorchDataset`, `ConditionalQueryBatch`,
  `EOS_CODE = "TIMELINE//END"`. Binary answers, no sentinel.
- `src/every_query/data/schema.py` — `QuerySeqSchema` (`queries`, `durations`, `answers` list cols).
- `src/every_query/generate_tasks/sample_query_sequences.py` — the 5-stage sequence sampler,
  mirroring `sample_tasks`: Stage 0 `build_prediction_times` (reused), Stage 1'
  `QuerySequenceDistribution` (a `QueryDistribution` subclass adding `min/max_queries` plus the
  default-off `eos_first_fraction` / `duration_mode` knobs), Stage 2 `sample_patient_contexts`
  (reused), Stage 3' `build_sequence_index`, Stage 4' `label_binary_occurrence` fanned out per
  shard. Contexts are drawn **globally across the split**, not per shard, and durations are
  continuous floats — both differ from the pre-port sampler, so the old checkpoint cannot be
  compared against new data without retraining. Seeding uses three independent axes:
  `derive_seed(seed, "queries")` (identical to the training sampler's draw),
  `derive_seed(seed, "contexts")`, and `derive_seed(seed, "sequences")` for sequence structure.
  `build_sequence_index_df` remains as the in-memory variant for supplied cohorts and the eval grid.
- `src/every_query/model/conditional_lightning.py` — Lightning module; metrics = pooled
  `answer_auc` + per-position breakdowns (training-time diagnostics only).
- `src/every_query/train/configs/conditional_config.yaml` — the training config (8-layer encoder,
  hidden 384, 4-layer decoder, bf16; vocab/positions are sized from the data by `train.py`).

### Helper scripts (`scripts/`)

| Script | What it does |
|---|---|
| `generate_mimic_sequences.py` | **Superseded** — the sampler fans out across shards itself now. Kept as a signpost to the equivalent `EQ_generate_query_sequences` invocation. |
| `make_clinical_task_sequences.py` | Build designed clinical conditional tasks (mortality, ICU→death, readmission) anchored 24h post-admission. |
| `make_position_probe.py` | Matched-code position probe (a fixed code at positions 1..P). |
| `eval_v2.py` | Query-form review eval: marginal `[C,d]`, EOS-conditioned `[END d][C d]` (P(C\|record ends) vs P(C\|data continue)), nested horizons. Counterfactual conditioning by overriding teacher-forced prior answers. |
| `eval_macro_position.py` | **The definitive conditioning eval** (see [§4](#4-evaluation-methodology)). |
| `eval_clinical.py` | **Designed clinical tasks** over three held-out anchor families (post-admission, post-discharge at HOME-discharge, random-time); single-query within-task AUROC + teacher-forced conditioning demos. Readmission anchored at discharge; each task names its exact MEDS code. |
| `eval_occurs_uncensored.py` | **Original-EveryQuery-comparable** macro occurs-AUROC on the uncensored cohort: `[TIMELINE//END,D]=0 [C,D]` scored only where the record extends past `t+D`. |
| `eval_per_position.py` | Pooled per-position AUROC with bootstrap CIs (superseded by the macro eval; kept for reference). |
| `eval_position_effect.py` | Earlier controlled 0/2/4-prior probe on curated codes. |
| `build_report_final.py` | Assemble the final PDF report (`reports/EveryQuery_Conditional_Report_FINAL.pdf`) from the macro, clinical, and uncensored summary JSONs (saved under `reports/results/`). |
| `build_report_v2.py` | Earlier PDF report builder (kept for reference). |

---

## 3. Experiments

All on MIMIC-IV in MEDS form (v0.3.0 tensorized cohort, 11,958 codes; intermediate string-coded
event shards reused from the EveryQuery comparator). Single NVIDIA GB10.

| Run | Data | Notes |
|---|---|---|
| `runs/main_v2` | 598k seqs, ~5 epochs | random-baseline (variable-length 1–5), the first v2 run |
| `runs/eos_v2` | EOS-aware sweep | `eos_first_fraction=0.5`, `duration_mode=same` |
| `runs/big_v2` | **28.79M unique seqs, 1 epoch** | fully-random **fixed-length-5**, no knobs, ~300k steps |

The `big_v2` run is the methodologically clean one: every training step sees a **distinct** iid
query sequence (no repetition), 86.9% of the 28.79M sequences are unique, and all 227,602 MIMIC-IV
subjects are covered across all 292 shards. Fixed length 5 is used because, under the block-causal
mask, position `j` depends only on blocks `0..j`, so shorter sequences are redundant prefixes —
fixed length trains every position with uniform per-position sample counts.

---

## 4. Evaluation methodology

**Use macro (per-task) AUC, not pooled AUROC.** Pooled AUROC scores cross-task pairs (a positive
for one query vs a negative for another) and is dominated by base-rate differences between queries
— it overstates per-query skill (≈0.91 pooled vs the honest ≈0.77 per-task here). The right metric
is AUC computed **within each query**, then **macro-averaged over queries**.

Since AUC = `P(score_pos > score_neg)` (Mann–Whitney), for one positive/negative patient pair drawn
for the same query the indicator `1[score_pos > score_neg]` is an unbiased estimate of that query's
AUC, and averaging over many queries estimates macro-AUC. `eval_macro_position.py` implements this
**triple-paired** estimator: per task, score the *same* positive/negative pair at every position
(random fillers before the target, their true answers teacher-forced), so the position trend is
fully paired (query + patient pair held fixed; only the amount of prior conditioning varies), and
bootstrap over tasks gives a CI on the slope / Spearman of macro-AUC vs position.

**Two explicit, both-valid positive-sampling schemes** (`--sampling`):

- `patient` (default, patient-level): pick a patient uniformly among those that permit a positive,
  then one of their positive prediction times uniformly — one patient, one vote.
- `pair` (context-level): pick one positive `(patient, prediction_time)` context uniformly over all
  such pairs — patients weighted by how many positive contexts they have.

Both use **occurrence-driven** positive construction — go to a real occurrence of code `C` at time
`τ` and pick a prediction time `t ∈ [τ−T, τ)` — so the label is guaranteed positive and **all
codes are estimable, rare included** (removing the coverage bias of sampling from a random pool).

---

## 5. Results

**Big iid run (`big_v2`), patient-uniform macro per-task AUC, 23,959 tasks** (step-50k checkpoint;
re-run on the converged checkpoint pending):

| position (priors) | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| macro-AUC | 0.7694 | 0.7753 | 0.7777 | 0.7791 | 0.7822 |

- **slope = +0.00295 / position, 95% CI [+0.0023, +0.0036] (excludes 0).**
- **Spearman ρ(position, macro-AUC) = +1.000, 95% CI [+0.90, +1.00].**

→ Per-task discriminability rises **monotonically and significantly** with position. The effect is
small in absolute terms (+0.013 from pos 0→4), as expected since *random* priors are only weakly
informative about a *random* target, but it is statistically real — confirming the model exploits
the conditional structure. The honest macro-AUC *level* is ~0.77–0.78 (vs the misleading pooled
~0.91).

Designed clinical tasks and the EOS-conditioned query forms are reported by `eval_v2.py` /
`build_report_v2.py`; the v2 PDF report (`EveryQuery_Conditional_Report_v2.pdf`) covers the leak
post-mortem, the query-form critical review, training stability, and the baseline-vs-EOS-aware
comparison.

---

## 6. Reproduce

Environment: a layered venv at `.venv/` overlays the prebuilt `MIMIC_experiments/venvs/eq`
site-packages (torch 2.11 + cu130 on aarch64/GB10) with this fork installed editable. Tests:
`./.venv/bin/python -m pytest tests/test_conditional_queries.py tests/test_conditional_cli.py`
(the CLI test exercises the full gen → train → predict → evaluate chain on a fixture cohort).

Data generation, training, and evaluation commands are in the helper scripts above; cohort paths
(`/home/mmd/MIMIC_experiments/...`) are machine-specific. `.env` holds the machine paths
(`TASK_DIR`, `OUTPUT_DIR`, `FINAL_DATA_DIR`, …).
