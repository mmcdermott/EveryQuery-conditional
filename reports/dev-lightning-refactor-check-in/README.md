# Dev-branch (#30 Lightning refactor) evaluation check-in

Does the `origin/dev` refactor of the multitask evaluation code (PR #31 / issue #30: `EQ_predict_multitask`
through `Trainer.predict`, the new `ConditionalMultitaskDataModule`, split-specific LightningModule loops)
change the AUROCs the evaluation suite computes?  **No, apart from a deliberate precision default.**
Scored at fp32 (`precision=32-true`) the dev code reproduces every stored probability of the 2026-09-05
`multitask-canonical-eval` suite **bit for bit** (3 sizes x 2 grids, 3.8M rows); the CLI's new default
`bf16-mixed` moves probabilities by ~1e-4 on average (max 3e-2), no per-task AUROC by more than 1.4e-3, and
no macro AUROC by more than 2e-5.  A retrain of the small model on the dev code lands inside the run-to-run
noise of the old code.  A static review of the diff (4 lenses, adversarially verified) found no other
number-changing edit.

## What was compared

Nothing was trained for the main comparison.  The three checkpoints of `multitask-canonical-eval`
(small 128x3, base 256x6, large 384x12; seed 140799) were re-scored on its evaluation grids with three
code variants, and the outputs compared row for row with the stored predictions:

| variant     | code                                                             | how it is scored                                                                  |
| ----------- | ---------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `stored`    | de0f5ee (`exp/multitask-canonical-eval`), run 2026-09-05          | the parquets under `multitask-canonical-eval/preds{,_horizon}/`, torch 2.14 venv  |
| `old_rerun` | de0f5ee, re-run 2026-09-07 in a fresh venv (repo uv.lock, torch 2.11) | the pre-#30 manual `torch.no_grad` loop (fp32 weights, no autocast, TF32 matmuls) |
| `new_fp32`  | 0b90790 (`origin/dev` = PR #31 merged), same fresh venv          | `Trainer.predict`, `precision=32-true`                                            |
| `new_bf16`  | 0b90790                                                          | `Trainer.predict` at the CLI's new default `precision=bf16-mixed`                 |

Evaluation sets (all `split=tuning` for the comparison, `batch_size=192` as in the old suite):

- **canonical**: 40 random K=5 query sequences x 14,132 tuning contexts = 565,280 rows (`evalgrid/eval`)
- **horizon**: 50 designed discrete-horizon sequences on the same contexts = 706,600 rows (`evalgrid_horizon/eval`)

The two **held_out** grids (never scored by the old suite) were scored with `new_bf16` only, as the updated
numbers (section "Held-out results").

Two controls separate the code change from everything else: `old_rerun` vs `stored` isolates the
environment (torch 2.11 vs 2.14, run-to-run noise), and `new_fp32` vs `old_rerun` isolates the code alone
(same venv, same precision).  Every variant's parquet was checked to be the same grid in the same row order
(subject, prediction time, the five window lists, target code, label) before any probability was compared.

## Pipeline

```
scripts/env.sh                  paths; BIN_NEW / BIN_OLD are the two worktree venvs
scripts/10_predict.sh           <variant> <grid> <split> <size> [run_dir]  -> preds/<variant>/<grid>_<split>/<size>.parquet
scripts/20_analyze.py           per-task AUROC / macro averages (the old suite's 50_analyze.py, generalized) -> metrics/<variant>/...
scripts/30_compare.py           row-for-row and task-for-task comparison of every pair -> metrics/comparison.{md,json}, per_task.md
scripts/40_report.py            updated results per evaluation set + per-stratum deltas -> metrics/summary.md
scripts/50_train_small.sh       retrain the small model with the old or the dev code (identical labels / seed / budget)
scripts/60_train_check.py       retrained models vs each other and vs the stored small run -> metrics/train_check.{md,json}
scripts/run_stream_new.sh       stream A: dev code, both precisions on the tuning grids, then bf16 on the held_out grids
scripts/run_stream_old.sh       stream B: re-score the stored parquets, then the old code on the tuning grids
scripts/run_train_check.sh      old code x2 + dev code x1 small retrains, each scored on the canonical tuning grid
```

Streams A and B ran concurrently on the GB10 (2026-09-07 21:35 to 2026-09-08 ~02:00); the training check ran
after stream B.  Logs are under `logs/`, one per stage; only aggregates are ever printed.

## Results

### 1. Row-level: the refactor is bit-exact at fp32; only the precision default moves numbers

`metrics/comparison.md` (tuning split).  "rows equal" = fraction of rows whose probability is bitwise
identical; dAUROC is per task; d macro is the difference of the macro AUROC over tasks.

| grid      | size  | `new_fp32` vs `stored` | `old_rerun` vs `stored` | `new_bf16` vs `stored`: max abs dprob | mean abs dprob | p99 | rows > 1e-2 | max abs dAUROC (task) | tasks > 5e-3 | d macro AUROC | d macro AUPRC |
| --------- | ----- | ---------------------- | ----------------------- | ------------------------------------- | -------------- | ------- | ----------- | --------------------- | ------------ | ------------- | ------------- |
| canonical | small | 100% rows equal        | 100% rows equal         | 9.4e-3                                | 1.2e-4         | 1.6e-3  | 0.0%        | 1.4e-3                | 0/40         | -0.00000      | -0.00005      |
| canonical | base  | 100% rows equal        | 100% rows equal         | 1.7e-2                                | 1.1e-4         | 1.8e-3  | 0.0%        | 7.1e-4                | 0/40         | +0.00001      | +0.00004      |
| canonical | large | 100% rows equal        | 100% rows equal         | 1.1e-2                                | 7.8e-5         | 1.2e-3  | 0.0%        | 2.0e-4                | 0/40         | -0.00000      | +0.00001      |
| horizon   | small | 100% rows equal        | 100% rows equal         | 2.4e-2                                | 1.6e-4         | 2.3e-3  | 0.0%        | 2.4e-4                | 0/50         | -0.00001      | -0.00003      |
| horizon   | base  | 100% rows equal        | 100% rows equal         | 8.6e-3                                | 1.2e-4         | 1.8e-3  | 0.0%        | 1.5e-4                | 0/50         | +0.00002      | +0.00006      |
| horizon   | large | 100% rows equal        | 100% rows equal         | 3.2e-2                                | 1.1e-4         | 1.1e-3  | 0.2%        | 1.9e-4                | 0/50         | -0.00001      | -0.00003      |

`new_fp32` vs `old_rerun` (same venv) is likewise 100% bitwise equal for all six, and `new_bf16` vs
`new_fp32` equals `new_bf16` vs `stored` exactly, so the whole bf16 delta is the autocast and none of it is
plumbing.  Bitwise equality also holds across the torch versions (stored: 2.14; reruns: 2.11), so the
GB10's fp32/TF32 kernels are deterministic for these shapes.  Per-task numbers: `metrics/per_task.md`.

### 2. Updated results on the evaluation sets (dev code, new default `bf16-mixed`)

`metrics/summary.md` has every variant and the per-stratum tables; the stored column is the 2026-09-05
parquet re-scored by this experiment's analysis script, which reproduces the old suite's own numbers to
every printed digit.

| set       | size  | macro AUROC [95% CI]  | median | macro AUPRC | stored macro AUROC | stored median | stored AUPRC |
| --------- | ----- | --------------------- | ------ | ----------- | ------------------ | ------------- | ------------ |
| canonical | small | 0.8425 [0.814, 0.868] | 0.8486 | 0.1551      | 0.8425             | 0.8485        | 0.1552       |
| canonical | base  | 0.8615 [0.837, 0.885] | 0.8856 | 0.1757      | 0.8615             | 0.8856        | 0.1757       |
| canonical | large | 0.8591 [0.836, 0.882] | 0.8634 | 0.1739      | 0.8591             | 0.8634        | 0.1739       |
| horizon   | small | 0.8692 [0.848, 0.890] | 0.8938 | 0.2182      | 0.8692             | 0.8938        | 0.2183       |
| horizon   | base  | 0.8857 [0.867, 0.904] | 0.8989 | 0.2488      | 0.8857             | 0.8990        | 0.2487       |
| horizon   | large | 0.8893 [0.872, 0.906] | 0.9078 | 0.2531      | 0.8893             | 0.9078        | 0.2532       |

Every per-stratum macro AUROC (by window type, start form, end form, horizon) moves by at most 2e-4
(`metrics/summary.md`); the old README's conclusions (base > small on 37/40 tasks, large ~ base on the
canonical set, small consistent large gain on the horizon set, delayed starts hardest) are unchanged.

### 3. Held-out results (new; the old suite never scored these grids)

Same tasks as the tuning grids (`configs/canonical_tasks.yaml`, `configs/horizon_tasks.yaml` of the old
suite), labeled at one prediction time per subject of a 50% subsample of the **held_out** split
(14,211 contexts; canonical 568,440 rows, horizon 710,550 rows; every task has both label classes).  Dev
code, default `bf16-mixed`.  Per-stratum tables are in `metrics/summary.md`.

| set       | size  | macro AUROC [95% CI]  | median | macro AUPRC | tuning-split macro AUROC (same tasks) |
| --------- | ----- | --------------------- | ------ | ----------- | ------------------------------------- |
| canonical | small | 0.8455 [0.820, 0.871] | 0.8566 | 0.1623      | 0.8425                                |
| canonical | base  | 0.8611 [0.836, 0.884] | 0.8762 | 0.1855      | 0.8615                                |
| canonical | large | 0.8597 [0.835, 0.883] | 0.8679 | 0.1828      | 0.8591                                |
| horizon   | small | 0.8613 [0.841, 0.882] | 0.8834 | 0.2171      | 0.8692                                |
| horizon   | base  | 0.8791 [0.861, 0.897] | 0.8941 | 0.2489      | 0.8857                                |
| horizon   | large | 0.8823 [0.865, 0.899] | 0.8954 | 0.2553      | 0.8893                                |

The size ordering of the tuning split carries over (base > small everywhere; large ~ base on the canonical
set, large > base by a few thousandths on the horizon set); the horizon set scores ~0.007 lower on
held_out than on tuning for every size, the canonical set is within +-0.003.  Delayed starts stay the
hardest stratum (canonical: 0.820 / 0.838 / 0.837 for small / base / large).

### 4. Training side: a dev-code retrain sits inside the old code's run-to-run noise

`metrics/train_check.md`.  Small model, identical labels / seed / batch / one-epoch budget as
`multitask-canonical-eval/scripts/30_train.sh`; every checkpoint scored on the canonical tuning grid through
the dev CLI at its default precision, so only training differs.  GPU training is not bit-reproducible
(`deterministic: false`), hence the old code twice.

| run           | code                    | tuning/loss at the 4 validations   | final train/loss_epoch | macro AUROC | median | macro AUPRC |
| ------------- | ----------------------- | ---------------------------------- | ---------------------- | ----------- | ------ | ----------- |
| small_old_a   | de0f5ee, 2026-09-07     | 0.0418 -> 0.0397 -> 0.0387 -> 0.0385 | 0.0418               | 0.8425      | 0.8485 | 0.1552      |
| small_old_b   | de0f5ee, 2026-09-08     | 0.0418 -> 0.0397 -> 0.0387 -> 0.0385 | 0.0418               | 0.8424      | 0.8485 | 0.1552      |
| small_new_a   | 0b90790 (dev), 2026-09-08 | 0.0418 -> 0.0397 -> 0.0387 -> 0.0385 | 0.0418             | 0.8425      | 0.8485 | 0.1552      |
| small_stored  | de0f5ee, 2026-09-05     | 0.0418 -> 0.0397 -> 0.0387 -> 0.0385 | 0.0418               | 0.8425      | 0.8486 | 0.1551      |

| pair                        | meaning                              | d macro AUROC [95% CI over tasks] | max abs dAUROC (task) | mean abs dAUROC (task) |
| --------------------------- | ------------------------------------ | --------------------------------- | --------------------- | ---------------------- |
| small_old_a - small_old_b   | old vs old, same seed: noise floor   | +0.0001 [-0.0000, +0.0002]        | 0.0011                | 0.0001                 |
| small_new_a - small_old_a   | dev vs old, same seed                | +0.0000 [-0.0001, +0.0002]        | 0.0019                | 0.0002                 |
| small_new_a - small_old_b   | dev vs old (second run)              | +0.0001 [-0.0000, +0.0003]        | 0.0029                | 0.0002                 |
| small_new_a - small_stored  | dev vs the stored 2026-09-05 run     | +0.0001 [-0.0001, +0.0003]        | 0.0036                | 0.0002                 |
| small_old_a - small_stored  | old today vs the stored run          | +0.0000 [-0.0001, +0.0002]        | 0.0017                | 0.0002                 |

### 5. Static review of the diff (`metrics/static_review.md`)

Four independent reviewers (inference numerics, dataset / row order, training loop, output / metrics) read
`git diff de0f5ee..origin/dev`, each finding was attacked by three adversarial refuters (110 agents in
total; 35 findings, all confirmed, 29 of them "verified harmless").  Conclusions:

- The **only** edit that alters a probability is `DEFAULT_PREDICT_PRECISION = "bf16-mixed"`
  (`predict_multitask.py`, `predict_multitask.yaml`): Lightning wraps `predict_step` in a bf16 autocast, so
  the backbone (duration MLPs, every Linear, SDPA) runs in bf16 while RMSNorm, the fp32 residual, the
  `autocast(enabled=False)` scoring head in `score_final_query` and the sigmoid stay fp32.
  `precision=32-true` makes the autocast a no-op and reproduces the old numerics (confirmed bitwise above).
- Verified identical: dataset construction (`dataclasses.replace` on `MEDSTorchDataConfig` is
  field-for-field equal; same `split` / `strip_delta_tokens` / `expected_vocab_size` / `max_windows`),
  batching (`BatchSampler(SequentialSampler, 192, drop_last=False)` on both sides, so pad widths and model
  inputs are identical; `pin_memory` / `persistent_workers` / `worker_init_fn` are inert here), device
  transfer, `eval()` mode, `inference_mode` vs `no_grad`, matmul precision "high" (set by the unchanged
  `setup_model`), output frame assembly and row order (the new `scored_codes` check can only raise).
- Training: `ConditionalMultitaskDataModule` pins `MultitaskBoundaryPytorchDataset` and inherits the
  train / val loaders unchanged; the explicit `validation_step` issues the same `tuning/loss` log call, so
  checkpoint selection and early stopping are unchanged; old checkpoints load through the new CLI without
  any dataset setting changing.
- Side notes worth a follow-up (none affects this suite): (a) `do_resume=true` on a pre-#30 run directory
  under the new default config is refused by `train/resume_check.py` (`datamodule._target_`,
  `data_class`, `eval_tasks_dir`, `max_windows` are not on its allow-list); (b) `held_out/loss` from
  `trainer.test` now means final-query BCE on the grid instead of the dense all-vocabulary BCE; (c) the
  docstring of `test_predict_multitask_accepts_a_pre_issue_30_run_dir` reads as old-vs-new parity but
  compares two runs of the new CLI; (d) the "pre-#30 loop ran in fp32" comments could say "fp32 weights
  with TF32 matmuls", which `32-true` also uses.

## Outputs

```
preds/<variant>/<grid>_<split>/<size>.parquet      one row per grid row (prob / label / target_code + window lists)
metrics/<variant>/<grid>_<split>/<size>/           task_metrics.parquet, summary.json
metrics/comparison.{md,json}, per_task.md          the row / task comparisons
metrics/summary.md                                 updated results + per-stratum deltas
metrics/train_check.{md,json}                      the retrain check
metrics/static_review.{md,json}                    the diff review
runs/small_{old_a,old_b,new_a}/                    the retrained small checkpoints (offline W&B)
logs/                                              one log per stage
```

Code: branch `exp/dev-lightning-refactor-check-in` of `~/EveryQuery-conditional` (origin/dev + the report
under `reports/dev-lightning-refactor-check-in/`); worktrees `.claude/worktrees/dev-lightning-refactor-check-in`
(dev) and `.claude/worktrees/old-eval-baseline` (de0f5ee) hold the two venvs the streams used.

Wall clock (two streams sharing the GPU): predict small 2.5 min, base 8-15 min, large 22-40 min per grid
(faster alone); training small ~10 min each.  Old suite reference, alone: 2 / 7 / 20 min.
