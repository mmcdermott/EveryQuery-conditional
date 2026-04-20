# EveryQuery

[![tests](https://github.com/payalchandak/EveryQuery/actions/workflows/tests.yaml/badge.svg?branch=dev)](https://github.com/payalchandak/EveryQuery/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/payalchandak/EveryQuery/branch/dev/graph/badge.svg)](https://codecov.io/gh/payalchandak/EveryQuery)

A framework for training and evaluating foundation models over structured EHR data, built on
the [MEDS](https://github.com/Medical-Event-Data-Standard) ecosystem —
[`meds-torch-data`](https://github.com/mmcdermott/meds-torch-data) for tensorization,
[`MEDS-transforms`](https://github.com/mmcdermott/MEDS_transforms) for preprocessing, PyTorch
Lightning for training.

Given a tensorized MEDS cohort, EveryQuery trains a ModernBERT-style encoder to answer
"query" prediction tasks of the form: *given a subject's history up to time `t`, will code
`c` occur within `d` days?* The same trained model is then evaluated against arbitrary
`(code, duration)` combinations.

> [!NOTE]
> The Phase-1 + Phase-2 refactor from [#54](https://github.com/payalchandak/EveryQuery/issues/54) has landed: `preprocess → generate_tasks → train → predict → evaluate` with a cross-stage [`TaskQuerySchema`](src/every_query/data/schema.py) is the current shape. One CLI surface is still mid-migration: `EQ_evaluate` console script points at the legacy four-stage evaluator; the new single-stage evaluator (#100) is reachable as `python -m every_query.evaluate.evaluate` today and a future release will flip the entry point ([#83](https://github.com/payalchandak/EveryQuery/issues/83) tracks the rewire). See [Roadmap](#roadmap) for what's next.

## Install

**For development** (recommended):

```bash
git clone git@github.com:payalchandak/EveryQuery.git
cd EveryQuery
uv sync --group dev
cp .env.example .env # then edit paths for your machine
```

**As a dependency:**

```bash
# not yet on PyPI — installable from git for now:
pip install "git+https://github.com/payalchandak/EveryQuery.git@main"
```

## Repository layout

Every production module lives under a submodule that reflects its role:

```
src/every_query/
├── preprocessing/      → EQ_process_data        (raw MEDS → tensorized cohort)
├── generate_tasks/     → EQ_generate_tasks      (TaskQuerySchema-conformant task parquets)
├── train/              → EQ_train               (train the model)
├── predict/            → EQ_predict             (inference; consumes TaskQuerySchema, emits PredictionSchema)
│   └── external_tasks/                         (ACES + composite aggregation; entry points on #95)
├── evaluate/           → EQ_evaluate (legacy) + new evaluate.py (#83 rewire pending)
├── model/              (shared: nn.Module + LightningModule)
├── data/               (shared: PyTorch Dataset + Batch types + TaskQuerySchema)
├── paper_experiments/  (research-only: ID/OOD splits, ablations, figure code)
│   └── sample_codes/   (query-code sampling for paper experiments; dataset-agnostic on #97)
└── utils/              (helpers: seeds, code slugs, env-var validation, model_loader)
```

Every submodule has its own `README.md` explaining what belongs there, its pipeline
position, and the tracking issues for remaining work.

## Console scripts

`pip install` exposes the CLIs below, all Hydra-configurable. Run any with `--help` or
`--cfg job` to inspect the resolved config. The **Tests** column summarises the coverage
that lands with each CLI on `dev` today — unit tests (fast, `tests/test_<name>_logic.py`
or `tests/test_<module>.py`), CLI smoke tests (`tests/test_cli_smoke.py`, `--help`-exits-0),
and end-to-end subprocess tests that run the real script against a fixture cohort.

| Script              | Stage         | Purpose                                                                                                                               | Tests                                                                                                                    |
| ------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `EQ_process_data`   | preprocessing | Orchestrate MEDS-transforms + `meds-torch-data` tensorization                                                                         | smoke; E2E via `test_process_data.py` + `test_e2e_foundation.py`                                                         |
| `EQ_generate_tasks` | task labels   | Sample `N` tasks × `M` contexts, label via single-pass asof (`TaskQuerySchema`-conformant)                                            | smoke; unit `test_sample_tasks.py`; E2E `test_generate_tasks.py`                                                         |
| `EQ_train`          | training      | Train the ModernBERT encoder on the labeled tasks                                                                                     | smoke; unit `test_training.py`; E2E `test_train_cli.py` + `test_train.py`; signal test `tests/training_validity/` (slow) |
| `EQ_predict`        | inference     | Consume a `TaskQuerySchema` parquet dir + checkpoint, emit a `PredictionSchema` parquet (`censor_prob`, `occurs_prob`)                | smoke; E2E `test_predict_cli.py` (row-order preserved); also exercised by `tests/training_validity/` (slow)              |
| `EQ_gen_eval_index` | eval (legacy) | Sample held-out prediction times into a deterministic index                                                                           | smoke only                                                                                                               |
| `EQ_gen_eval_tasks` | eval (legacy) | Slice per-duration task matrices by `(code, duration)` using the index                                                                | smoke; unit `test_eval_suite.py`                                                                                         |
| `EQ_evaluate`       | eval (legacy) | **Legacy**: runs a trained checkpoint against the sliced eval tasks, writes per-code AUCs. #83 will rewire to the new evaluator below | smoke; unit `test_eval.py`                                                                                               |
| `EQ_select_model`   | analysis      | Rank models by pairwise win rate over `(code, duration)` pairs                                                                        | smoke only                                                                                                               |

**New single-stage evaluator** (`python -m every_query.evaluate.evaluate`) — consumes a `PredictionSchema` parquet from `EQ_predict`, writes per-`(query, duration_days)` metrics (`n_rows`, `n_occurs_labeled`, `n_positive`, `occurs_auroc`, `censor_auroc`). Not yet on `EQ_evaluate` — the `[project.scripts]` rewire is deferred to the [#83](https://github.com/payalchandak/EveryQuery/issues/83) consolidation wave. Tested via `test_evaluate_cli.py` and `tests/training_validity/` (slow).

## Pipeline

### Current (on `dev`)

```
           MEDS cohort  ──►  EQ_process_data  ──►  tensorized cohort ($FINAL_DATA_DIR)
                                                                     │
                                                                     ▼
                                                         EQ_generate_tasks
                                                                     │  TaskQuerySchema parquets
                                                                     ▼
                                                                EQ_train  ──►  best_model.ckpt
                                                                                     │
                                                                                     ▼
                                                                                EQ_predict  ──►  PredictionSchema parquet
                                                                                                           │
                                                                                     ┌─────────────────────┤
                                                                                     │                     │
                                                                                     ▼                     ▼
                                             python -m every_query.evaluate.evaluate      EQ_evaluate (legacy, ignores
                                                          │                                 the PredictionSchema path —
                                                          ▼                                 runs its own inference loop
                                                 per-(query, duration_days)                 against the sliced task
                                                       metrics                              parquets + sibling CLIs)
```

Both evaluator paths ship in this release; the legacy one is what `EQ_evaluate` resolves to today, and the new path is reachable via `python -m`. [#83](https://github.com/payalchandak/EveryQuery/issues/83) tracks the rewire that consolidates to a single `EQ_evaluate` pointing at the new module.

### 1. Preprocess

```bash
EQ_process_data \
	input_dir="$RAW" \
	intermediate_dir="$INTERMEDIATE" \
	output_dir="$FINAL_DATA_DIR"
```

Produces a tensorized MEDS cohort under `$FINAL_DATA_DIR`. `$INTERMEDIATE` is a staging
directory for the MEDS-transforms stages; `$PROCESSED` holds cross-shard metadata
(`$PROCESSED/metadata/codes.parquet` is the query-code universe the sampler draws from).

### 2. Generate pre-training task labels

```bash
EQ_generate_tasks \
	split=train \
	input_shard=0 \
	task_shard=0 \
	n_tasks=1024 \
	contexts_per_task=1
```

Sweep across shards with
`python -m every_query.generate_tasks.sample_tasks -m input_shard=0,1,2,… task_shard=range(0,K)`.
Each worker writes labeled task parquets under `$TASK_DIR/{split}/*.parquet` idempotently. Output columns conform to [`TaskQuerySchema`](src/every_query/data/schema.py) — `subject_id, prediction_time, query, duration_days, boolean_value` — where `boolean_value` is a nullable three-valued label (`null` = censored, `True` = event occurred in `[prediction_time, prediction_time + duration_days)`, `False` = observed-but-no-event).

### 3. Train

```bash
EQ_train \
	output_dir="$OUTPUT_DIR/outputs/\${run_id:}" \
	datamodule.config.task_labels_dir="$TASK_DIR" \
	datamodule.config.tensorized_cohort_dir="$FINAL_DATA_DIR"
```

`EQ_train` reads the long-format labels written by `EQ_generate_tasks` directly — the
inline collation step that lived in `train.py` was removed in
[#76](https://github.com/payalchandak/EveryQuery/pull/76).

Seeding: `cfg.seed` (default `140799`) is passed through `lightning.seed_everything` *before*
model + datamodule instantiation (fix landed in [#124](https://github.com/payalchandak/EveryQuery/pull/124)),
so model weight initialization is byte-reproducible across Python versions and platforms
for a given seed.

### 4. Predict

```bash
EQ_predict \
	model_run_dir="$OUTPUT_DIR/outputs/YYYY-MM-DD/HH-MM-SS" \
	tasks_dir="$TASK_DIR/tuning" \
	output_parquet="$OUTPUT_DIR/predictions.parquet" \
	split=tuning # or split=held_out (default)
```

Reads every `*.parquet` under `tasks_dir` (`TaskQuerySchema`-conformant), runs the checkpoint's `predict_step` over the chosen split, writes a single `PredictionSchema` parquet with `censor_prob` + `occurs_prob` per input row. See [`predict/README.md`](src/every_query/predict/README.md) for details.

### 5. Evaluate — new single-stage path

```bash
python -m every_query.evaluate.evaluate \
	predictions_parquet="$OUTPUT_DIR/predictions.parquet" \
	metrics_parquet="$OUTPUT_DIR/metrics.parquet"
```

Per-`(query, duration_days)` metrics from the predictions parquet — `n_rows`, `n_occurs_labeled`, `n_positive`, `occurs_auroc` (on non-censored rows), `censor_auroc`. See [`evaluate/README.md`](src/every_query/evaluate/README.md).

`[project.scripts]` `EQ_evaluate` does **not** yet point at this module — the rewire is the remaining work on [#83](https://github.com/payalchandak/EveryQuery/issues/83).

### 6. Evaluate — legacy four-stage path

```bash
EQ_gen_eval_index # sample prediction times into a deterministic eval index
EQ_gen_eval_tasks # slice per-duration task matrices by (code, duration)
EQ_evaluate model_run_dirs='["'"$OUTPUT_DIR"'/outputs/YYYY-MM-DD/HH-MM-SS"]'
EQ_select_model model_run_dirs='["..."]' split=tuning
```

Still what the `EQ_evaluate` console script resolves to today — runs inference + metrics in one shot, with multi-model / id-ood-manual bucketing the new path doesn't have. Preserved until the #83 consolidation wave signs off on migration. `model_run_dirs` is a required override (no default) in both `EQ_evaluate` and `EQ_select_model` as of [#126](https://github.com/payalchandak/EveryQuery/pull/126); Hydra reports "mandatory value missing" on a fresh clone instead of failing later with `FileNotFoundError` on a stale hardcoded path.

## Configuration

All CLIs are `@hydra.main` entry points; every config knob is overridable on the command
line with `key=value` or `+new_key=value`. The config directory is resolved via
`importlib.resources.files("every_query")`, so package-shipped YAMLs work identically
whether you run from a source checkout or a `pip install`ed wheel.

### Environment variables

`ensure_env()` (in `utils/_env.py`) requires these be set before `EQ_train` and the eval
CLIs. Scope of this gate was tightened in [#127](https://github.com/payalchandak/EveryQuery/pull/127)
— `PROCESSED` and `INTERMEDIATE` were dropped because no Hydra config interpolates them
(they were only read by a dotenv fallback in the sampler, which already tolerates missing
env vars when CLI config values are supplied).

| Var              | Purpose                                                |
| ---------------- | ------------------------------------------------------ |
| `PROJECT_DIR`    | Repo root (for relative output paths in a few configs) |
| `OUTPUT_DIR`     | Where training run dirs land                           |
| `TASK_DIR`       | Where task parquets read / write                       |
| `FINAL_DATA_DIR` | Tensorized cohort (output of `EQ_process_data`)        |
| `WANDB_ENTITY`   | W&B entity for training telemetry                      |

`.env.example` is the reference — copy to `.env` and edit. Both Python (via
`python-dotenv`) and the SLURM wrappers under `scripts/` source it. Further phases of
[#117](https://github.com/payalchandak/EveryQuery/issues/117) will migrate the remaining
gated vars to `${oc.env:VAR,???}` / `${oc.env:VAR,default}` form (Hydra-native required
or optional-with-fallback) and eventually retire `ensure_env()` entirely.

### Known gotcha: code-group YAMLs

`train/configs/config.yaml`, `evaluate/conf/eval_config.yaml`, and
`evaluate/conf/gen_tasks_config.yaml` all pull a default `{train,eval}_codes/<hash>.yaml`
that is (a) generated out-of-band and (b) explicitly `.gitignore`d — so a fresh clone
can't compose them. Workaround until [#64](https://github.com/payalchandak/EveryQuery/issues/64)
lands:

- Pass `--config-dir=/path/to/your/codes_dir code_group_name=...`, or
- Generate them locally via
    `python -m every_query.paper_experiments.sample_codes.sample_train_codes` (note: currently
    has a hardcoded MIMIC path — #85 will parameterize it).

The smoke-test fixture in `tests/test_cli_smoke.py` shows the minimal shape of each file.

## Development

```bash
uv sync --group dev
uv run pytest                         # full suite, excluding slow tests (~2 min)
uv run pytest -m 'slow or not slow'   # full suite incl. slow training-validity test (~8-10 min extra)
uv run pytest tests/test_cli_smoke.py # CLI smoke tests only
uv run pre-commit run --all-files     # lint, format, codespell
```

CI runs the full `pytest -m "slow or not slow"` (both `slow`-marked and unmarked tests)
on Python 3.11 and 3.12, plus `ruff check` and `ruff format --check` on every PR; coverage
is uploaded to Codecov. Full CI session: ~10-11 min typical.

### Test layout

```
tests/
├── test_cli_smoke.py               (every EQ_* CLI; --help exits 0)
├── test_process_data.py            (E2E: EQ_process_data output shape + metadata)
├── test_generate_tasks.py          (E2E: ground-truth label recompute + reproducibility + n_tasks differential)
├── test_sample_tasks.py            (unit: sampler primitives, determinism, edge cases)
├── test_train_cli.py               (E2E: EQ_train CLI, resume flow, overwrite flag)
├── test_train.py                   (E2E: resume-actually-loads-ckpt two-stage differential)
├── test_training.py                (unit: single training step, checkpoint roundtrip, demo-mode checks)
├── test_predict_cli.py             (E2E: EQ_predict against a trained checkpoint + row-order preservation)
├── test_evaluate_cli.py            (E2E: python -m every_query.evaluate.evaluate on a synthetic PredictionSchema parquet)
├── test_e2e_foundation.py          (E2E: full preprocess → generate_tasks → train pipeline chains)
├── test_eval.py                    (unit: legacy eval.py helpers)
├── test_eval_suite.py              (unit: legacy gen_task.py, process_eval_tasks)
├── test_dataset_logic.py           (unit: EveryQueryPytorchDataset + EveryQueryBatch)
├── test_lightning_logic.py         (unit: LightningModule loss wiring, mask semantics)
├── test_model_logic.py             (unit: model heads, censored/occurs loss flip sensitivity)
├── test_run_id.py                  (unit: run_id resolver determinism)
└── training_validity/              (E2E @pytest.mark.slow: model actually learns; now runs via EQ_predict → evaluate.evaluate subprocess chain; see its README)
    ├── __init__.py
    ├── conftest.py
    ├── README.md
    └── test_training_validity.py
```

## Roadmap

Overall refactor umbrella: [#54](https://github.com/payalchandak/EveryQuery/issues/54) —
target architecture is `preprocess → generate_tasks → train → predict → evaluate` with a
shared cross-stage task-query schema.

### Phase 2 status

| Sub-phase                      | Issue                                                       | State                                                                                                                                       |
| ------------------------------ | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| 2.1: TaskQuerySchema design    | [#80](https://github.com/payalchandak/EveryQuery/issues/80) | ✅ merged via [#96](https://github.com/payalchandak/EveryQuery/pull/96) (also closes #122)                                                  |
| 2.2: EQ_predict                | [#81](https://github.com/payalchandak/EveryQuery/issues/81) | ✅ merged via [#99](https://github.com/payalchandak/EveryQuery/pull/99)                                                                     |
| 2.3: eval-suite inventory      | [#82](https://github.com/payalchandak/EveryQuery/issues/82) | Decisions captured on the issue + reflected in #100's scope; no code change needed                                                          |
| 2.4: EQ_evaluate consolidation | [#83](https://github.com/payalchandak/EveryQuery/issues/83) | 🟡 new `evaluate.py` merged via [#100](https://github.com/payalchandak/EveryQuery/pull/100); `[project.scripts]` rewire + deletions pending |

### E2E testing status ([#104](https://github.com/payalchandak/EveryQuery/issues/104))

| Subprocess test                  | Issue                                                         | State                                                                                                                                                                            |
| -------------------------------- | ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_process_data.py`           | (pre-104)                                                     | ✅ merged                                                                                                                                                                        |
| `test_generate_tasks.py`         | [#107](https://github.com/payalchandak/EveryQuery/issues/107) | ✅ merged via [#112](https://github.com/payalchandak/EveryQuery/pull/112)                                                                                                        |
| `test_train.py`                  | [#108](https://github.com/payalchandak/EveryQuery/issues/108) | ✅ merged via [#113](https://github.com/payalchandak/EveryQuery/pull/113)                                                                                                        |
| `test_predict_cli.py`            | (part of #99)                                                 | ✅ merged via [#99](https://github.com/payalchandak/EveryQuery/pull/99) (row-order preservation covered)                                                                         |
| `test_evaluate_cli.py`           | [#109](https://github.com/payalchandak/EveryQuery/issues/109) | ✅ merged via [#100](https://github.com/payalchandak/EveryQuery/pull/100) (new single-stage evaluator)                                                                           |
| training-validity (model learns) | [#118](https://github.com/payalchandak/EveryQuery/issues/118) | ✅ merged via [#119](https://github.com/payalchandak/EveryQuery/pull/119); now runs the full `EQ_predict` → `evaluate.evaluate` chain as subprocesses (slow, gated by `-m slow`) |

### Hygiene / follow-ups

| Issue                                                         | Description                                                                                                                     |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| [#62](https://github.com/payalchandak/EveryQuery/issues/62)   | Promote `aces_to_eq` / `process_composite` to entry points — draft PR [#95](https://github.com/payalchandak/EveryQuery/pull/95) |
| [#64](https://github.com/payalchandak/EveryQuery/issues/64)   | Drop gitignored `{train,eval}_codes` defaults (design pick pending)                                                             |
| [#85](https://github.com/payalchandak/EveryQuery/issues/85)   | Rewrite `sample_codes/` dataset-agnostic — draft PR [#97](https://github.com/payalchandak/EveryQuery/pull/97)                   |
| [#117](https://github.com/payalchandak/EveryQuery/issues/117) | Env-var audit — phase 1 merged via [#127](https://github.com/payalchandak/EveryQuery/pull/127); phases 2-4 pending              |
| [#125](https://github.com/payalchandak/EveryQuery/issues/125) | Adopt hypothesis-based property tests for the sampler                                                                           |
| [#129](https://github.com/payalchandak/EveryQuery/issues/129) | Rename `PredictionSchema.occurs_prob` → `label_prob` post-NeurIPS once non-occurrence task types land                           |
| [#59](https://github.com/payalchandak/EveryQuery/issues/59)   | Docs: final rewrite after the refactor settles                                                                                  |

### Model / architecture research (non-blocking)

- [#101](https://github.com/payalchandak/EveryQuery/issues/101) / [#102](https://github.com/payalchandak/EveryQuery/issues/102) — RoPE for time-deltas
- [#103](https://github.com/payalchandak/EveryQuery/issues/103) — Evaluate alternatives to ModernBERT as the encoder backbone

## Acknowledgements

EveryQuery sits on top of [MEDS](https://github.com/Medical-Event-Data-Standard),
[`meds-torch-data`](https://github.com/mmcdermott/meds-torch-data),
[`MEDS-transforms`](https://github.com/mmcdermott/MEDS_transforms), and
[`MEDS_EIC_AR`](https://github.com/mmcdermott/MEDS_EIC_AR) (architectural reference). It
uses [Hydra](https://hydra.cc) for configuration, [PyTorch Lightning](https://lightning.ai)
for training, and [W&B](https://wandb.ai) for telemetry.

## License

MIT — see [LICENSE](LICENSE).
