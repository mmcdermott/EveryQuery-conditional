# Conditional-queries redesign — handoff / resume notes

_Last updated mid-session. Branch: `conditional-queries`._

## What changed (the v2 redesign — binary observed-occurrence labels)

Per user direction, the conditional model was reworked so that:

- **Every answer is binary Y/N** = "is this code observed in `(t, t+d]`?". No `null`/CENSORED
  class, no loss masking except padding. An event we can't observe (record ends first) → NO.
- **Censoring is a query, not a label.** `TIMELINE//END` (a real vocab code, index 11884,
  one per subject at record end) is queried like any other code: `(TIMELINE//END, d)` = "does
  the record end within d". Conditioning a later query on its teacher-forced answer recovers &
  generalizes EQ's old `P(occurs | data exist after d)`:
  - `[EOS d]=NO  [C d]` → P(C | data exist after d)  (old EQ; ~0 for death)
  - `[EOS d]=YES [C d]` → P(C | record ends within d) (actionable death form)
  - `[C d]` alone → marginal P(C observed); = weighted avg of the two.
- **Fully random sequences**: L∼U{1..5} queries, each (random code incl. EOS, random duration),
  no privileged censor-first position. Two **default-off** sweep knobs exist:
  `eos_first_fraction` (force pos-0 = EOS some fraction) and `duration_mode`
  (random|same|nondecreasing).

### Why (the leak that triggered this)
The v1 design put a same-horizon censor query first and teacher-forced its answer. For terminal
events that answer ≈ the label (`censor=False ⟺ death`), so mortality AUROC hit 0.991 by the
model copying the censor answer (verified: AUROC from censor answer alone = 0.996). v1 also
*masked* censored subsequent answers, so the only loss-bearing death labels were censor-determined
— leak present in training, not just eval. v2 fixes this structurally.

## Files DONE (committed, coherent, package imports + doctests pass)
- `src/every_query/model/conditional_model.py` — N_ANSWER_CLASSES=2, BCE over all non-pad
  positions, removed CENSORED / occurs_loss_weight / censor_query_index / ConditionalQueryOutput
  .censor_loss/.occurs_loss. Module + class docstrings updated.
- `src/every_query/model/conditional_lightning.py` — metrics = answer_auc + per-position only
  (dropped censor/occurs); predict_step unchanged.
- `src/every_query/data/seq_dataset.py` — EOS_CODE="TIMELINE//END", binary answers, no sentinel,
  `eos_query_index`. (`CENSOR_QUERY_CODE` removed.)
- `src/every_query/data/schema.py` — QuerySeqSchema.answers = non-null bool, docstring updated.
- `src/every_query/generate_tasks/sample_query_sequences.py` — `build_sequence_index_df` (random
  seqs + eos_first_fraction + duration_mode), `label_binary_occurrence` (new binary labeler),
  `run_worker`/`main` updated. (`label_sequence_index_df` removed.)
- `src/every_query/generate_tasks/configs/sample_query_sequences_config.yaml` — min_queries/
  max_queries + eos_first_fraction/duration_mode.
- `src/every_query/train/configs/conditional_config.yaml` + `_demo_train_conditional.yaml` —
  vocab_size 11959, removed occurs_loss_weight.
- `scripts/generate_mimic_sequences.py` — arg names min/max-queries + eos/duration-mode.

## Files STILL TO UPDATE (reference removed symbols — will break tests/eval until fixed)
1. **`conftest.py`** — `seq_task_labels_dir` fixture imports `CENSOR_QUERY_CODE` and builds
   answers with `None`. Rewrite to binary answers (no None), include a `TIMELINE//END` query,
   no censor-first.
2. **`tests/test_conditional_queries.py`** — remove ANSWER_CENSORED, censor-first/leakage-masking
   expectations; update `make_batch`/labeling tests to binary; `test_label_sequence_known_answers`
   → use `label_binary_occurrence` semantics; drop `censor_query_index`/CENSOR_QUERY_CODE refs.
3. **`tests/test_conditional_cli.py`** — censor-first invariant + ANSWER nullability assertions →
   binary; generator now needs min_queries/max_queries args.
4. **`scripts/make_clinical_task_sequences.py`** — currently emits `(CENSOR, d)` first with the
   reverted censor-horizon hack. Redo clinical tasks under v2: sequences like
   `[TIMELINE//END d][MEDS_DEATH d]`, plus the marginal `[MEDS_DEATH d]`. Anchor = admission+24h.
5. **`scripts/make_position_probe.py`** — uses CENSOR_QUERY_CODE + label_sequence_index_df. Either
   delete or repurpose for an **informative-prior** probe (nested same-code horizons:
   `[EOS d][C d_short][C d_long]` and `[EOS d][C d_long]`) to test conditioning when priors carry
   signal (the v1 flat-probe used uninformed random fillers — that's why it was flat).
6. **`scripts/run_full_evaluation.py`** — remove censor/occurs split + 3-valued assumptions; new
   query forms: marginal, EOS-conditioned (P(C|EOS=1) vs P(C|EOS=0)), conditional chains,
   monotonicity (nested horizons). Within-query (macro) AUROC stays the headline.
7. **`scripts/build_report.py`** — rewrite results/§5-7 narrative for v2; add the leak post-mortem
   and the **critical review of query forms** the user asked for.

## Remaining workflow (Phase 1 → Phase 2)
1. Finish files 1–7 above; run `pytest tests/test_conditional_queries.py tests/test_conditional_cli.py`.
2. **Regenerate data** (binary labels, fully random). Old data at `experiments/tasks` is v1 — wipe
   & regen:
   ```
   rm -rf experiments/tasks experiments/tasks_clinical experiments/probe
   .venv/bin/python scripts/generate_mimic_sequences.py --data-dir /home/mmd/MIMIC_experiments/models/eq/intermediate \
     --out-dir experiments/tasks --processed /home/mmd/MIMIC_experiments/models/v0.3.0/preprocessed \
     --split train --n-contexts 2048 --jobs 10   # repeat for tuning/held_out (1024)
   ```
3. **Phase-1 train** (random baseline, eos_first_fraction=0):
   ```
   PATH=$PWD/.venv/bin:$PATH WANDB_MODE=disabled nohup .venv/bin/python -m every_query.train.train \
     --config-name=conditional_config output_dir=experiments/runs/main_v2 do_overwrite=true \
     > experiments/train_v2.log 2>&1 &
   ```
   ~30k steps, ~50 min on the GB10. Best ckpt → `experiments/runs/main_v2/best_model.ckpt`.
   NOTE: at eos_first_fraction=0, EOS is ~1/12k of queries, so EOS-conditioning will be weak in
   run 1 *by design* — that's the baseline. The Phase-2 sweep upweights it.
4. **Evaluate** (run_full_evaluation v2) + **build report**. Critically review query forms.
5. **Phase-2 sweep**: re-gen data + retrain for a few (eos_first_fraction, duration_mode) settings
   (e.g. eos_first_fraction ∈ {0, 0.25, 0.5}; duration_mode ∈ {random, same}); re-evaluate; compare
   whether they improve censoring-conditioned queries. Don't over-index — a few points.
6. Final report: v2 design, query-form critical review, Phase-1 results, Phase-2 comparison, leak
   post-mortem.

## Environment / how to run
- Layered venv: `.venv/bin/python` (editable fork over the prebuilt `MIMIC_experiments/venvs/eq`).
- Cohort: `/home/mmd/MIMIC_experiments/models/v0.3.0/preprocessed` (tensorized, 11958 codes).
- Labeling events (string codes, has TIMELINE//END at max_time):
  `/home/mmd/MIMIC_experiments/models/eq/intermediate/data/{split}/*.parquet`.
- `.env` already points TASK_DIR/OUTPUT_DIR/etc. at `experiments/`.
- v1 artifacts (superseded): `experiments/runs/main` (v1 model), `experiments/eval`,
  `EveryQuery_Conditional_Report.pdf`. Keep for the leak post-mortem; don't reuse the model.

## Key facts
- EOS = `TIMELINE//END`, vocab index 11884, exactly 1/subject, always at max_time (verified).
- v1 training was healthy (converged ~step 16k, val loss 0.206, small train/val gap) — the weak
  conditioning was a *data/label* issue (masking + uninformative random priors), not under/overfit.
