# Handoff — continue issues #24 → #23 after Phase F

Updated 2026-09-03 in `/home/gkondas/eqc-multitask-stack` after the user requested a
token-budget handoff. This document supplements, and does not replace, the two source documents:

1. `/home/gkondas/.claude/plans/HANDOFF-multitask-24-23.md`
2. `/home/gkondas/.claude/plans/act-as-the-lead-groovy-parasol.md`

Read both source documents completely before continuing. The original handoff's **Phase E —
CONTRACT CHECKPOINT** and committed code are the source of truth; older proposal text must not
override the implemented contract.

## Current repository state

- Worktree: `/home/gkondas/eqc-multitask-stack`
- Branch: `feat/multitask-window-ar`
- HEAD: `10b25b6` (`10b25b673e7c690b6bb0d67ec9b931e778df42e9`)
- Branch is three commits ahead of `origin/feat/multitask-boundary-sampler`.
- Commit sequence:
  - `e152048` — `Add explicit duration/event window starts to the multitask sampler and dataset (#24)`
  - `3eee1e8` — `Fix window-position wraparound and start-time shape hole from the #24 review`
  - `10b25b6` — `Add decoder-only ConditionalMultitaskARModel with tied vocabulary head (#23)`
- Phase F's implementation tree was clean immediately after `10b25b6`.
- This handoff file is intentionally untracked, so `git status` is no longer empty. Remove it only
  after the next agent has read it, or retain/commit it only with explicit user direction. Do not
  confuse it with an implementation change.
- Never push, merge, open a PR, or modify GitHub issues without explicit permission.

Confirm before continuing:

```bash
git log --oneline -4
git status --short --branch
git diff --name-only 3eee1e8..10b25b6
```

## Completed work

Phases A–E remain complete exactly as recorded in the original handoff. Do not redesign or
reimplement #24.

### Phase F — complete and committed

The sole implementation agent created commit `10b25b6` with the exact requested message:

```text
Add decoder-only ConditionalMultitaskARModel with tied vocabulary head (#23)
```

The committed diff from `3eee1e8` contains exactly the eight allowed files:

- `src/every_query/model/conditional_multitask_ar_model.py`
- `src/every_query/model/conditional_multitask_lightning.py`
- `src/every_query/train/configs/conditional_multitask_ar_config.yaml`
- `src/every_query/train/configs/_demo_train_conditional_multitask_ar.yaml`
- `tests/test_conditional_multitask_ar_model.py`
- `tests/test_conditional_multitask_cli.py`
- `src/every_query/train/train.py`
- `src/every_query/model/__init__.py`

Protected scalar and #24/data files are unchanged from `3eee1e8`. The implementation agent's audit:

```bash
git diff --name-only 3eee1e8 HEAD -- \
  conditional_ar_model.py conditional_model.py conditional_lightning.py data generate_tasks
```

returned empty.

Implementation incident already resolved: an initial eager export from `model/__init__.py` created
a multiprocessing circular import (`data.multitask_dataset -> seq_dataset -> conditional_model ->
model/__init__ -> conditional_multitask_lightning -> data.multitask_dataset`). It was repaired before
commit using lazy `__getattr__` exports. The full frozen #24 baseline passed afterward.

Phase F exact verified results:

- `tests/test_conditional_multitask_ar_model.py`: **22 passed in 0.52s**. This includes the explicit
  bf16-autocast/fp32-logits check.
- `tests/test_conditional_multitask_cli.py`: **2 passed in 118.20s**. This includes train+tuning
  multitask label generation, demo `EQ_train`, `last.ckpt`, `resolved_config.yaml`, position-budget
  assertion, and `setup_model(..., module_cls=ConditionalMultitaskLightningModule)` reload.
- Frozen #24 baseline:
  `uv run pytest tests/multitask tests/test_multitask_dataset_integration.py -q -p no:cacheprovider`
  → **197 passed, 1 skipped in 317.55s**.
- Scalar decoder-only regression: `tests/test_conditional_ar_model.py` → **34 passed in 21.14s**.
- Doctests for the new modules plus `train.py`: **5 passed in 25.29s**.
- Pre-commit on exactly the eight Phase F files: **all checks passed**.

## Phase G — complete, read-only reviewer

A separate read-only reviewer ran as `/root/phase_g_review`. The reviewer was explicitly forbidden
to edit or commit and finalized with **NO BLOCKING FINDINGS**. No repair commit is warranted or
should be manufactured. The reviewer confirmed:

- It had read both plan documents completely.
- It confirmed `3eee1e8..10b25b6` contains exactly the eight allowed files and no scalar/data/#24
  changes.
- It statically verified:
  - exact `[patient, W0, C0, A0, ..., W(K-1)]` ordering and `3K-2` tokens;
  - `W_i` gathers at `n_patient + 3*i`;
  - explicit `q_mask` token masks and loss masks;
  - paired legacy-start handling, with exactly one missing field rejected;
  - `K == 1` and `K > max_windows` behavior;
  - tied `get_input_embeddings().weight` readout and gradient path;
  - autocast-disabled fp32 matmul, bias `-3.0`, and PAD loss exclusion;
  - distinct start/end duration projections and event markers;
  - constant query-stream clinical-time RoPE positions;
  - Lightning, checkpoint, and config requirements.
- Reviewer rerun of new model+CLI tests: **24 passed, 15 warnings in 95.66s**.
- Reviewer scalar AR+CLI regression:
  `uv run pytest tests/test_conditional_ar_model.py tests/test_conditional_cli.py -q -p no:cacheprovider`
  → **55 passed, 1 failed, 17 warnings in 286.56s**.
- The failure was
  `tests/test_conditional_cli.py::test_eval_v3_scores_supplied_sequences`, with
  `KeyError: summary['num_evaluation_sequences']`. Phase F touches neither that test nor eval-v3 or
  scalar code. The reviewer judged it not attributable to `10b25b6`. It is distinct from the
  handoff's already-known scalar sampler-suite failure, so Phase H should rerun and record it
  separately; do not repair it as part of #23/#24 without evidence that this branch caused it.
- Review was strictly read-only; no files were edited or committed.
- The only non-blocking concern was the contractually expected O(B*K*V) memory for logits, elementwise
  BCE, and materialized `valid_mask`. No correctness concern was found.

## Phase H — not started

After Phase G is closed, run the full relevant suites from the source handoff:

```bash
uv run pytest tests/multitask tests/test_multitask_dataset_integration.py \
  tests/test_conditional_multitask_ar_model.py tests/test_conditional_multitask_cli.py \
  tests/test_conditional_ar_model.py tests/test_conditional_cli.py -q -p no:cacheprovider

uv run pytest tests/sampler tests/test_sampler_dataset_integration.py \
  tests/test_event_bounded.py tests/test_event_bounds_oracle.py -q -p no:cacheprovider

uv run pytest src -q -p no:cacheprovider
```

The known scalar-suite failure is pre-existing and out of scope:

```text
tests/sampler/test_eval_grid_per_shard.py::
test_config_cohort_keys_mirror_the_flat_sampler_and_sampling_keys_mirror_training
```

Do not fix it as part of #23/#24. Report it explicitly as pre-existing if reproduced.

Also execute one scripted end-to-end path:

```text
sample -> metadata/packed targets -> dataset -> collate -> model forward -> BCE -> backward ->
checkpoint save/reload
```

It must explicitly confirm:

- `targets.shape == logits.shape == (B, K, V)`;
- every condition answer equals its window target bit;
- each `W_i` uses its matching start/end specification;
- changing `A_i` cannot affect `W_i`, but can affect later windows;
- future conditions cannot affect earlier windows;
- all six start/end combinations reach the model;
- paired absent starts use legacy prediction-time behavior;
- scalar AR behavior remains unchanged;
- final implementation working tree is clean after commits.

The new CLI test already exercises the real label-generation -> dataset -> collate -> train/backward ->
checkpoint/reload path, while the new model tests exercise leakage and token-spec invariants. Phase H
should still run and record an explicit scripted path as required by the user, not merely cite those
tests separately.

## Final required report

Finish with a merge-readiness report containing:

- commits created (Phase F plus a Phase G repair commit only if needed);
- every test command and exact result;
- reviewer findings and repairs;
- remaining correctness and performance concerns;
- deferred work;
- an explicit ready-for-human-review verdict.

Known deferred/accepted items remain:

- no `EQ_predict_multitask` CLI; `predict_step` returns full-vocabulary probabilities;
- non-null ontology support is not implemented and must raise `NotImplementedError`;
- no full-vocabulary macro metrics in the Lightning module;
- no real-shard benchmark/pilot run;
- #24 flattened-window labeling has the already-accepted page-cache/transient-memory concerns recorded
  in the original handoff.

Standing rules: use exactly one implementation agent at a time; reviewers are read-only; preserve the
frozen #24 contract; preserve unrelated work; use no destructive git commands; never push/merge/PR or
modify issues without permission.
