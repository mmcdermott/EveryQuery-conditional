# EveryQuery-conditional: the three new features, trained and evaluated

**Question.** `dev` added RoPE time encoding, event-bounded durations, and DAG-aware embeddings
and queries, none of which had been trained. Do they work end to end, and does a model trained
with all three answer queries better than chance — including queries only these features can
express?

**Answer.** Yes, on all three query forms, at both sequence lengths, on both a randomly drawn
panel and a hand-designed ICU panel. A tiny model trained for one epoch (14 minutes) reaches
macro AUROC 0.77–0.83 on held-out data. Conditioning on a *clinically informative* two-query
prefix is worth about 2.7× conditioning on a random one.

Branch `worktree-nf-train-eval` (from `dev` @ `a59f951`); no `src/` changes — the features work
as shipped. Scripts: `scripts/new_features/`. W&B: `EQ-conditional-new-features-test`, run
`mpfrq7nn`.

---

## 1. Setup

| | |
|---|---|
| cohort | MIMIC-IV MEDS tensorized, 13,908 codes, real `parent_codes` |
| ontology | 21,071 nodes = 13,908 leaves + 7,163 ancestors (399 dual-role `//ANY`); **V_ext = 21,072** |
| training labels | 300,000 train + 20,000 tuning query sequences; length U{1..5}; log-uniform horizons over [1, 731] d; `eventbound_fraction=0.5`; ontology on |
| model | hidden 256, 4 encoder layers, 2 decoder layers, 4 heads, ffn 1024, batch 64, `max_seq_len` 256, bf16 |
| training | 1 epoch = 4,688 steps in 13m53s on a shared GB10 |
| evaluation | `held_out`, 3,979 contexts, teacher-forced inference |

Everything the run wrote lives under `$NF_ROOT` (`$EQ_EXP_ROOT/new_features_test`).

---

## 2. Experiment 1 — are the features actually on?

Config keys can be set and still not take effect, so `09_verify_run.sh` checks three independent
places: the saved config, the checkpoint hyperparameters, and a real batch through the loaded
model. **22/22 checks pass.**

| feature | what proves it |
|---|---|
| RoPE time | `strip_delta_tokens=true` + `use_rope_time=true` in config *and* checkpoint hparams; log shows `RoPE time: stripping 25 delta-token vocab ids`; a live batch carries `time_pos_ids` shaped like `code`, nondecreasing over real tokens, in elapsed hours (max 370,880 h in-batch) |
| event-bounded | 50.1% of the 900,512 training queries are event-bounded, `-1` duration sentinel aligned on every one (**0 violations in either direction**); a live batch has `q_bound_codes` 48% bounded; `bound_marker` in the state dict |
| DAG-aware | encoder sized to **V_ext 21,072**, not the 13,909 cohort vocab; query vocabulary extended to 21,071 nodes (7,163 ancestors); 34.0% of training queries and 33.9% of event boundaries are ancestor nodes; input embedding is an `OntologyEmbedding` with 21,072 rows |

The cohort really does contain 23 `TIMELINE//DELTA*` codes, so RoPE-time performs a genuine
strip rather than a silent no-op.

**Latent hazard.** The two `ontology_dir` keys — `datamodule.dataset_kwargs.ontology_dir` and
`lightning_module.model.ontology_dir` — are **never cross-checked by the code**. If they drift,
queries address the wrong embedding rows and the run completes normally with meaningless
ancestor semantics. `04_train.sh` sets both from one shell variable deliberately, and the
verifier asserts they agree.

---

## 3. Experiment 2 — randomly drawn tasks

20 tasks per query form, drawn uniformly from the model's own 21,068-node query universe under
an occurrence floor (a code nobody ever has gives an undefined AUROC, which measures nothing).
At length 3 the target sits at position 2 behind two *randomly* drawn filler queries.

Macro AUROC at the target position:

| query form | len 1 | len 3 | scored | > 0.5 | CIs excluding 0.5 | sign-test p |
|---|---|---|---|---|---|---|
| duration-bounded | **0.826** | **0.831** | 20/20 | 20/20 | 20/20 | 1.9 × 10⁻⁶ |
| event-bounded | **0.770** | **0.778** | 20/20 | 16–17/20 | 17/20 | 0.012 / 0.0026 |
| DAG / ancestor | **0.788** | **0.795** | 20/20 | 18/20 | 18/20 | 4.0 × 10⁻⁴ |

Pooled AUROC by position at length 3 is **monotonic** — 0.687 → 0.711 → 0.729 — reproducing the
conditional-structure effect the v2 report found for the full-size model (0.769 → 0.782 across
positions 0→4). Pooled numbers are base-rate inflated and are for dynamics only; the macro table
is the honest level.

`HOSPITAL_ADMISSION`, a pure ancestor rolling up 70 child codes, scores **0.573 / 0.578**
(CI [0.557, 0.600]) — above chance and firing on descendants as intended, but the weakest
ancestor in the panel. At 23.8% prevalence it is a broad, near-ambient roll-up; other ancestors
in the same panel reach 0.95.

Per-task: `by_task_len1.csv`, `by_task_len3.csv`.

---

## 4. Experiment 3 — a clinically meaningful ICU panel

31 hand-designed tasks an intensivist would recognise, resolved to real nodes by
`clinical_concepts.py`: 10 duration-bounded, 8 event-bounded, 13 DAG/ancestor. Full mapping in
`clinical_manifest.csv`.

| query form | len 1 | len 3 | scored | > 0.5 | CIs excluding 0.5 |
|---|---|---|---|---|---|
| duration-bounded | **0.816** | **0.832** | 10/10 | 10/10 | 10/10 |
| event-bounded | **0.730** | **0.760** | 8/8 | 8/8 | 7/8 |
| DAG / ancestor | **0.815** | **0.829** | 13/13 | 13/13 | 13/13 |

**All 31 tasks score above 0.5 at both lengths**; 30/31 have a 95% CI excluding 0.5.

Selected tasks (length 1 → length 3):

| task | what it asks | prevalence | len 1 | len 3 |
|---|---|---|---|---|
| `anc_vasopressin_2d` | vasopressin in 2 d — refractory shock | 0.011 | 0.960 | 0.977 |
| `anc_icu_discharge_7d` | ICU discharge from **any** unit in 7 d | 0.209 | 0.954 | 0.961 |
| `anc_norepinephrine_1d` | norepinephrine at **any** rate in 1 d | 0.027 | 0.953 | 0.967 |
| `anc_propofol_1d` | propofol in 1 d — proxy for intubation | 0.037 | 0.951 | 0.959 |
| `anc_epinephrine_2d` | epinephrine in 2 d — third-line pressor | 0.006 | 0.942 | 0.966 |
| `evt_propofol_before_icu_discharge` | sedation **before leaving the ICU** | 0.052 | 0.896 | 0.927 |
| `evt_norepi_before_icu_discharge` | pressors **before leaving the ICU** | 0.039 | 0.892 | 0.929 |
| `dur_discharge_died_30d` | discharge disposition DIED in 30 d | 0.042 | 0.883 | 0.897 |
| `dur_mortality_1d` | death in 1 d | 0.014 | 0.872 | 0.882 |
| `dur_mortality_7d` | death in 7 d | 0.039 | 0.868 | 0.882 |
| `dur_lactate_1d` | lactate drawn in 1 d — sepsis workup | 0.079 | 0.861 | 0.872 |
| `dur_mortality_30d` | death in 30 d | 0.066 | 0.837 | 0.893 |
| `evt_icu_before_discharge` | ICU escalation **before next discharge** | 0.034 | 0.709 | 0.786 |
| `anc_hosp_admit_30d` | admission of any type in 30 d | 0.113 | 0.578 | 0.581 |
| `evt_death_before_discharge` | **in-hospital mortality** | 0.013 | 0.520 | 0.512 |

Two things worth reading carefully:

**The event-bounded form earns its place.** "Death before the next hospital discharge" *is*
in-hospital mortality, and no fixed horizon states it correctly — a 30-day window mixes in deaths
after discharge, a 7-day window misses long stays. Same for "vasopressors before leaving the
ICU". These questions are only expressible because of the new feature.

**In-hospital mortality is the one weak task** (0.520, CI [0.438, 0.601] — the only CI that
includes 0.5). It has 50 positives out of 3,979 contexts, and 46.3% of the event bounds never
fire, in which case the window runs to the end of the record and the query degenerates to "does
this ever occur again". A rarer, degeneracy-prone target on an undertrained tiny model is exactly
where you would expect it to fail. Contrast `dur_discharge_died_30d` (0.883), which asks nearly
the same clinical question through a duration-bounded disposition code and works well.

Per-task: `by_task_clin_len1.csv`, `by_task_clin_len3.csv`.

---

## 5. Experiment 4 — what is conditioning worth?

Both panels hold the target fixed and vary only what precedes it, so the length-1 → length-3
delta isolates the value of conditioning. Inference is teacher-forced, so position 2 sees the
*true* answers to its prefix.

| panel | prefix | mean Δ AUROC | median Δ | tasks improved |
|---|---|---|---|---|
| random (Exp. 2) | two random queries | **+0.007** | +0.005 | 39/60 (65%) |
| clinical (Exp. 3) | two clinically related queries | **+0.019** | +0.013 | **29/31 (94%)** |

A clinically informative history is worth roughly **2.7× a random one**, and it helps nearly
every task rather than about two-thirds. Within the clinical panel the gain is largest for
event-bounded queries (+0.031, 7/8 improved) and smallest for ancestors (+0.014, but 13/13
improved).

Largest gains — all cases where the prefix is genuinely diagnostic of the target:

| Δ | task |
|---|---|
| +0.077 | ICU escalation before the next hospital discharge |
| +0.059 | creatinine drawn before leaving the ICU |
| +0.056 | death within 30 days |
| +0.044 | ICU admission to any unit within 2 days |
| +0.037 | vasopressor need before leaving the ICU |

The two that did not improve are `dur_troponin_1d` (−0.000) and `evt_death_before_discharge`
(−0.008), the latter being the weak task discussed above.

This is a stronger version of the monotonic-position result: the model is not merely using
*more* context, it is using *relevant* context.

Comparisons: `compare_len1_vs_len3.csv`, `compare_clin_len1_vs_clin_len3.csv`.

---

## 6. Methodology notes

**Row identity is asserted, not assumed.** The eval grid is context-major, so sequence *k* is
`(contexts[k // N], specs[k % N])`, and the predictions parquet carries no `seq_id`. The scorer
reconstructs each row's spec from row order and **checks it against the spec YAML** — one dropped
label row would shift every downstream assignment silently. Match rate **1.000000** on all four
prediction files.

**The shipped evaluator is not sufficient for this question.** `EQ_evaluate_sequences` has no
macro average, no minimum-support gate, and does not group `by_query` by position — so at length
3 it pools the task under test with the prefix queries. `07_score.py` does the grouping the
experiment needs and reports Hanley–McNeil CIs plus a sign test against the 0.5 null.

**`scripts/eval_v3.py` cannot score this checkpoint at all.** It hand-builds the dataset with no
`dataset_kwargs`, so ancestor queries `KeyError` and RoPE-time raises on the missing
`time_pos_ids`. `EQ_predict_sequences` is the only correct path — it replays every flag from
`resolved_config.yaml`. The same defect affects `eval_v2.py`, `eval_per_position.py` and
`run_full_evaluation.py`.

**Event-bound validity.** A query bounded by itself or by one of its own ancestors is
unconditionally False. Both spec generators reject such pairs using the ontology closure.

---

## 7. Incidents and gotchas

**A transient CUDA fault.** A first run of the identical config died at step 1,638/4,688 in the
**backward** pass with `merge_sort: failed to synchronize: cudaErrorIllegalAddress`. The rerun
passed that same step and completed with zero faults, so it is **not deterministic**. It is not
an out-of-range index either: `probe_index_ranges.py` scanned 500 real batches and everything was
in range (`q_codes` max 21,071 vs V_ext 21,072; `q_answers` ∈ {0,1}; padding is `ANSWER_NO`=0 and
`block_pos_embed` is clamped). The likeliest site is `OntologyEmbedding`, which holds the mix
matrix as a **21,072 × 21,072 sparse COO tensor on GPU** whose backward through `A @ W` sorts
indices — exactly where `merge_sort` runs. Two other users' jobs shared the GPU. Worth watching
on longer runs.

Scoring the crashed run's step-1,638 checkpoint gave 0.815 / 0.768 / 0.776 on the random panel —
the same picture ~0.01–0.02 lower, as an undertrained checkpoint should.

**`env.sh` would have silently lost the W&B run.** It exports `WANDB_MODE=offline` and an *empty*
`WANDB_ENTITY`. A `${WANDB_MODE:-online}` default does not fix the first (it is set, just not to
what you want) and `train.py` raises on the second. Both are overridden unconditionally in
`scripts/new_features/_env.sh`.

**Hydra `=` on a dict node merges, it does not replace.** Swapping only `trainer.logger._target_`
to `WandbLogger` leaves the CSVLogger's `flush_logs_every_n_steps`, which `WandbLogger` forwards
into `wandb.init()` and which raises `TypeError` at the first `.experiment` access. Hence the
explicit `'~trainer.logger.flush_logs_every_n_steps'` deletion.

**`MEDSTorchBatch` has no `.to(device)`**, and `dataclasses.replace` trips its `torch.LongTensor`
isinstance assert (a CUDA long tensor is not one). Anything moving a batch by hand must set the
tensor fields in place. Lightning's own `transfer_batch_to_device` handles this generically, so
training is unaffected — but offline scoring scripts are not.

---

## 8. Reproducing

```bash
bash scripts/new_features/01_build_ontology.sh
bash scripts/new_features/02_sample_training_sequences.sh tuning 20000
bash scripts/new_features/02_sample_training_sequences.sh train 300000
bash scripts/new_features/03_make_eval_specs.sh          # random panel
bash scripts/new_features/10_make_clinical_specs.sh      # clinical ICU panel
bash scripts/new_features/04_train.sh 00:00:16:00
RUN_DIR=$(bash scripts/new_features/find_run_dir.sh cq-tiny-allfeat)
bash scripts/new_features/09_verify_run.sh "$RUN_DIR"
for t in len1 len3 clin_len1 clin_len3; do
    bash scripts/new_features/05_make_eval_labels.sh "$t" 4000
    bash scripts/new_features/06_predict.sh "$RUN_DIR" "$t"
done
bash scripts/new_features/08_score.sh --tags len1,len3
bash scripts/new_features/08_score.sh --tags clin_len1,clin_len3
bash scripts/new_features/11_compare_lengths.sh clin_len1 clin_len3 \
    "$NF_ROOT/eval_specs/clinical_manifest.csv"
```

`run_all.sh` chains the random-panel path. **Freeze `$NF_ONTOLOGY_DIR` after step 01** — the
checkpoint stores the ontology *path*, not the matrix, and re-reads it at every load, so
rebuilding in place silently changes the embeddings used at inference.

---

## 9. What this does and does not establish

**Established.** All three features train and serve end to end on a real cohort. A 14-minute
tiny model is well above chance on every query form, including queries that only the new features
can express. Ancestor queries labelled through the closure behave correctly — an ancestor fires
on any descendant. Relevant conditioning beats random conditioning by ~2.7×.

**Not established.** Nothing here isolates each feature's *contribution*: there is no ablation,
one seed, and one model size. Upstream found the ontology *embedding* effect on leaf tasks did
not replicate while the DAG *structure* was worth ~+0.039 AUROC on ancestor queries — this run is
consistent with that but does not test it. The absolute numbers are a floor: a tiny model, one
epoch, 300k sequences, against a reference full model trained for ~300k steps on 28.8M sequences.

**Next.** Ablate one feature at a time against a shared eval grid; a second seed; scale the model
before reading anything into the level rather than the sign.
