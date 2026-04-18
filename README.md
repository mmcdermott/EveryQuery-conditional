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
> A substantial refactor is in progress — see [#54](https://github.com/payalchandak/EveryQuery/issues/54).
> The pipeline is being consolidated into fewer, clearer CLIs. This README reflects the
> current state on `dev`.

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
├── generate_tasks/     → EQ_generate_tasks      (task-label parquets for PT)
├── train/              → EQ_train               (train the model)
├── predict/            → (planned: EQ_predict)  (inference; #81)
│   └── external_tasks/                         (ACES + composite aggregation)
├── evaluate/           → EQ_evaluate + 3 sibling CLIs  (metrics, model selection; #83 consolidates)
├── model/              (shared: nn.Module + LightningModule)
├── data/               (shared: PyTorch Dataset + Batch types)
├── paper_experiments/  (research-only: ID/OOD splits, ablations, figure code)
│   └── sample_codes/   (query-code sampling for paper experiments)
└── utils/              (helpers: seeds, code slugs, env-var validation)
```

Every submodule has its own `README.md` explaining what belongs there, its pipeline
position, and the tracking issues for remaining work.

## Console scripts

`pip install` exposes the CLIs below, all Hydra-configurable. Run any with `--help` or
`--cfg job` to inspect the resolved config.

| Script              | Stage         | Purpose                                                                     |
| ------------------- | ------------- | --------------------------------------------------------------------------- |
| `EQ_process_data`   | preprocessing | Orchestrate MEDS-transforms + `meds-torch-data` tensorization               |
| `EQ_generate_tasks` | task labels   | Sample `N` tasks × `M` contexts, label via single-pass asof (PT-ready)      |
| `EQ_train`          | training      | Train the ModernBERT encoder on the labeled tasks                           |
| `EQ_gen_eval_index` | eval setup    | Sample held-out prediction times into a deterministic index                 |
| `EQ_gen_eval_tasks` | eval setup    | Slice per-duration task matrices by `(code, duration)` using the index      |
| `EQ_evaluate`       | eval          | Run a trained checkpoint against the sliced eval tasks, write per-code AUCs |
| `EQ_select_model`   | analysis      | Rank models by pairwise win rate over `(code, duration)` pairs              |

## Pipeline

```
           MEDS cohort  ──►  EQ_process_data  ──►  tensorized cohort ($FINAL_DATA_DIR)
                                                                     │
                                                                     ▼
pre-training:                                              EQ_generate_tasks
                                                                     │  labeled task parquets
                                                                     ▼
                                                                EQ_train  ──►  best_model.ckpt
                                                                                       │
evaluation:                     EQ_gen_eval_index  ──►  EQ_gen_eval_tasks              │
                                                                │                      │
                                                                ▼                      ▼
                                                                             EQ_evaluate
                                                                                       │  per-code AUCs
                                                                                       ▼
                                                                                EQ_select_model
```

> The eval pipeline is being consolidated — `EQ_gen_eval_tasks` currently expects a
> per-duration wide task matrix whose dedicated producer was removed in
> [#76](https://github.com/payalchandak/EveryQuery/pull/76). Phase 2 of
> [#54](https://github.com/payalchandak/EveryQuery/issues/54) replaces these four CLIs with
> a schema-driven `EQ_predict` + a single consolidated `EQ_evaluate`.

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
Each worker writes labeled task parquets under `$TASK_DIR/{split}/*.parquet` idempotently.

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

### 4. Evaluate

```bash
EQ_gen_eval_index # sample prediction times into a deterministic eval index
EQ_gen_eval_tasks # slice per-duration task matrices by (code, duration)
EQ_evaluate model_run_dirs='["'"$OUTPUT_DIR"'/outputs/YYYY-MM-DD/HH-MM-SS"]'
EQ_select_model model_run_dirs='["..."]' split=tuning
```

## Configuration

All CLIs are `@hydra.main` entry points; every config knob is overridable on the command
line with `key=value` or `+new_key=value`. The config directory is resolved via
`importlib.resources.files("every_query")`, so package-shipped YAMLs work identically
whether you run from a source checkout or a `pip install`ed wheel.

### Environment variables

`ensure_env()` (in `utils/_env.py`) requires these be set before `EQ_train` and the eval
CLIs:

| Var              | Purpose                                                       |
| ---------------- | ------------------------------------------------------------- |
| `PROJECT_DIR`    | Repo root (for relative output paths in a few configs)        |
| `OUTPUT_DIR`     | Where training run dirs land                                  |
| `TASK_DIR`       | Where task parquets read / write                              |
| `PROCESSED`      | MEDS cohort `processed/` dir (holds `metadata/codes.parquet`) |
| `INTERMEDIATE`   | MEDS cohort `intermediate/` dir (event shards)                |
| `FINAL_DATA_DIR` | Tensorized cohort (output of `EQ_process_data`)               |
| `WANDB_ENTITY`   | W&B entity for training telemetry                             |

`.env.example` is the reference — copy to `.env` and edit. Both Python (via
`python-dotenv`) and the SLURM wrappers under `scripts/` source it.

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
uv run pytest                         # full suite (~90 s)
uv run pytest tests/test_cli_smoke.py # CLI smoke tests only
uv run pre-commit run --all-files     # lint, format, codespell
```

CI runs the full `pytest` plus `ruff check` and `ruff format --check` on every PR; coverage
is uploaded to Codecov.

## Roadmap

Overall refactor umbrella: [#54](https://github.com/payalchandak/EveryQuery/issues/54) —
target architecture is `preprocess → generate_tasks → train → predict → evaluate` with a
shared cross-stage task-query schema.

Live child issues:

- [#59](https://github.com/payalchandak/EveryQuery/issues/59) — docs: final rewrite after the refactor settles
- [#62](https://github.com/payalchandak/EveryQuery/issues/62) — promote `aces_to_eq` / `process_composite` to entry points (scaffolded here under `predict/external_tasks/`)
- [#64](https://github.com/payalchandak/EveryQuery/issues/64) — drop gitignored `{train,eval}_codes` defaults
- [#66](https://github.com/payalchandak/EveryQuery/issues/66) — unbreak `eval_config.yaml`'s hardcoded run dirs
- [#68](https://github.com/payalchandak/EveryQuery/issues/68) — wheel-install CI + staged CLI functional tests
- [#79](https://github.com/payalchandak/EveryQuery/issues/79) — Phase 1 restructure (this PR)
- [#80](https://github.com/payalchandak/EveryQuery/issues/80) — design cross-stage task-query schema
- [#81](https://github.com/payalchandak/EveryQuery/issues/81) — `EQ_predict` entry point
- [#82](https://github.com/payalchandak/EveryQuery/issues/82) — inventory `evaluate/` code paths
- [#83](https://github.com/payalchandak/EveryQuery/issues/83) — consolidate `EQ_evaluate`
- [#85](https://github.com/payalchandak/EveryQuery/issues/85) — rewrite `sample_codes/` dataset-agnostic
- [#91](https://github.com/payalchandak/EveryQuery/issues/91) — `do_resume` structural-drift check

## Acknowledgements

EveryQuery sits on top of [MEDS](https://github.com/Medical-Event-Data-Standard),
[`meds-torch-data`](https://github.com/mmcdermott/meds-torch-data),
[`MEDS-transforms`](https://github.com/mmcdermott/MEDS_transforms), and
[`MEDS_EIC_AR`](https://github.com/mmcdermott/MEDS_EIC_AR) (architectural reference). It
uses [Hydra](https://hydra.cc) for configuration, [PyTorch Lightning](https://lightning.ai)
for training, and [W&B](https://wandb.ai) for telemetry.

## License

MIT — see [LICENSE](LICENSE).
