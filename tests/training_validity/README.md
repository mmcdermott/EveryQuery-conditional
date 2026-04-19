# `tests/training_validity/`

End-to-end test that trains an `EveryQueryModel` on a designed-signal synthetic dataset
and asserts the trained model's predictions reflect the signal across both prediction
heads. The EQ analog of [`MEDS_EIC_AR`'s `test_pattern_generation.py`][meicar-gen]:
instead of asserting "the pipeline runs," it asserts "the model *learned* from training."

## What this test covers that the rest of the E2E suite does not

The other integration tests — `tests/test_process_data.py`, `tests/test_generate_tasks.py`,
`tests/test_train.py`, `tests/test_e2e_foundation.py` — cover:

- The pipeline runs end-to-end without raising
- CLI knobs are honored (differentials on `n_tasks`, `MIN_SUBJECTS_PER_CODE`, etc.)
- Resume advances `global_step`
- Sampler output has the right schema + label semantics

They all train for **2 optimizer steps on random weights** and never observe the training
dynamics. So silent regressions in the gradient path, label alignment, loss-mask
wiring, or duration-input propagation would pass every one of them.

This test trains for **4000 steps** on a tiny model (~6 minutes on CPU) against a dataset
where the ground-truth labels are a deterministic function of two per-subject marker
tokens. It asserts:

| #   | Assertion                                                                                           | Regression it catches                                                                  |
| --- | --------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| 1   | **Censor-head AUROC ≥ 0.9**                                                                         | Broken censor-head wiring, flag-token tokenisation failure, censor-loss mask inverted  |
| 2   | **Per-(code, duration) occurs AUROC ≥ 0.8** on every non-degenerate cell                            | Broken occurs-head wiring, label flip, pattern-marker tokenisation failure             |
| 3   | **Duration monotonicity** — mean predicted probability strictly increases across `d=1 < d=7 < d=30` | Duration-input feature path detached from gradient, model ignoring the duration scalar |

A gradient-path regression (no parameter updates), label flip, constant-output head, or
collapsed query-token embedding all drop at least one AUROC below threshold.

## Dataset design

See [#123][issue-123] for the full design-space comparison; four candidate approaches were
evaluated against the unified criteria and **Design 2 — oracle markers + fire-time
patterns** was chosen.

### Per-subject markers (emitted at day 0)

Each subject gets exactly one of each marker pair:

- **Fire-time marker** — one of `P_FIRE_D05`, `P_FIRE_D5`, `P_FIRE_D15`, `P_FIRE_D50`,
    `P_NEVER`. Determines whether (and when) the single `TARGET` event fires relative to
    the prediction time.
- **End marker** — one of `P_END_D20`, `P_END_D100`. Determines the subject's
    observation window (20 days or 100 days), which drives censoring at long durations.

### Event stream

- `TARGET` fires once on day `prediction_time + fire_offset` for firing-type subjects
    whose observation window includes that day.
- 5 `NOISE_0..NOISE_4` codes are Poisson-drawn at 0.5/day each to fill the sequence.
    The model has to attend to the two markers rather than treating every history as
    identical.

### Query task

- Query code: `TARGET`
- Query durations: `[1, 7, 30]` days
- Prediction time: day 10

Combined with the fire-time and end markers, the windowing logic produces:

| fire_marker                 | max_time  | d=1 occurs | d=7 occurs | d=30 occurs       | d=30 censored?     |
| --------------------------- | --------- | ---------- | ---------- | ----------------- | ------------------ |
| P_FIRE_D05 (fires day 10.5) | 20 or 100 | ✓          | ✓          | varies on censor  | END_D20 → censored |
| P_FIRE_D5 (fires day 15)    | 20 or 100 | ✗          | ✓          | varies on censor  | END_D20 → censored |
| P_FIRE_D15 (fires day 25)   | 20 or 100 | ✗          | ✗          | if not censored   | END_D20 → censored |
| P_FIRE_D50 (fires day 60)   | 20 or 100 | ✗          | ✗          | event past window | END_D20 → censored |
| P_NEVER (no fire)           | 20 or 100 | ✗          | ✗          | ✗                 | END_D20 → censored |

This produces ~20%/40%/70% `occurs=True` at `d=1/7/30` among non-censored subjects
(hence duration monotonicity), and ~55% censored at `d=30` (driven by `END_D20`).

## Why this particular design

Three of the four #123 candidates were evaluated:

- **Design 1** (pure Poisson + `SUBJECT_FLAG` for censoring) was attempted during
    implementation; per-cell occurs AUROC is inherently capped near chance because the
    Poisson process is memoryless — there's no per-subject signal in history to predict
    the next interval at a fixed rate.
- **Design 1a** (per-subject log-normal latent intensity + flag) got *close* — censor
    AUROC 0.994 in 18m42s of training, but occurs AUROC on the HOT-at-d=1 cell only
    reached 0.697. The Bayes-optimal classifier on that cell is bounded below 1.0 by
    the Bernoulli-sampling-noise floor, and the model's latent-intensity inference from
    a 30-day history didn't approach it in the CPU budget.
- **Design 2** (this test) clears every threshold on first try in ~3 minutes with
    perfect AUROC on every cell. The deterministic-marker → label mapping sidesteps the
    sampling-noise ceiling entirely.

Design 2 is the right trade-off for a *training-validity* test: the model isn't being
asked to estimate a latent variable, it's being asked to attend to a couple of marker
tokens and compose them with the duration input. That's the minimum bar for "the
architecture works end-to-end" — if it can't do this, it can't do anything more
complex.

If we want a stricter test that exercises rate-estimation from history, the Design 1a
branch [`test/e2e-training-validity-d1a`][d1a] remains available as a follow-up.

## Runtime budget

Target: ≤ 10 minutes CPU per [#123][issue-123]. Current runtime on a laptop-class
machine:

- Preprocessing (`EQ_process_data` on ~150 subjects): ~30s
- Training (4000 steps, hidden=128, 4-layer, batch=16): ~5-6 minutes
- Inference + assertions: ~few seconds

Total: ~6-7 minutes. Training step count bumped from 2000 (locally converged in 3
minutes) to 4000 after CI observed a Python-3.12 runner where the model's weight init
produced an unlucky optimization trajectory — the censor head under-converged at 2000
steps while 3.11 on the same commit passed. The root cause is that `train.py` seeds
the RNG *after* model init, making the weight init platform-dependent; see the inline
comment on `oracle_trained_model_dir` for more. A proper fix would move `seed_everything`
before `instantiate(cfg.lightning_module)`, but that's a training-code change outside
the scope of this test.

## Gotchas baked into the test (for future readers)

Three label-semantics and dataset-construction details worth calling out; all are
documented inline in the test module too.

1. **`boolean_value` = *censored*, not *occurs***. The EQ sampler overloads MEDS's
    `boolean_value` label column to mean "censored" (observation ended before we could
    observe the outcome), and uses a separate `occurs` column for the real positive-class
    label. The dataset derives `batch.censor = boolean_value` and the model's
    occurs-loss is masked to `~batch.censor`. Swapping the two labels silently inverts
    training. (See also [#122][issue-122] for the ongoing discussion of collapsing these
    into one nullable column.)
2. **`MEDSDataset.write` treats `data_shards` keys as path stems**, so the key
    `"train/0"` produces `data/train/0.parquet` but `"train"` produces
    `data/train.parquet`. MEDS-transforms expects the sharded layout; flat layout
    breaks the fit_normalization stage's subject_splits lookup.
3. **Query-code attribution uses `dataset.query[i]`**, not a separately-sorted label
    parquet — iteration order is the dataset's internal schema_df, not our sort of the
    source parquet. Re-indexing from a sorted parquet mis-aligns rows.

## Related

- Parent umbrella: [#104][issue-104]
- Design issue: [#123][issue-123]
- PR: [#119](https://github.com/payalchandak/EveryQuery/pull/119)

[d1a]: https://github.com/payalchandak/EveryQuery/tree/test/e2e-training-validity-d1a
[issue-104]: https://github.com/payalchandak/EveryQuery/issues/104
[issue-122]: https://github.com/payalchandak/EveryQuery/issues/122
[issue-123]: https://github.com/payalchandak/EveryQuery/issues/123
[meicar-gen]: https://github.com/mmcdermott/MEDS_EIC_AR/blob/main/tests/test_pattern_generation.py
