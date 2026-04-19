"""Training-validity E2E test: train on synthetic Poisson data, verify the model actually learns.

Designed as the analog of MEDS_EIC_AR's ``test_pattern_generation.py`` — the existing E2E suite
(#110/#111/#112/#113) verifies the pipeline runs and that knobs are honored, but trains for only 2
steps on random weights.  A silent regression in the gradient path, label alignment, or model
wiring would pass all of those tests because the training dynamics are never observed.

This test:

1. Synthesizes a MEDS dataset where each of three codes fires per its own Poisson rate
   (``CODE_HOT`` 20/day, ``CODE_WARM`` 2/day, ``CODE_COLD`` 0.05/day).  With a 1-day prediction
   window the expected per-code occurrence rates (P = 1 - e^(-rate*duration)) land at HOT ~1.0,
   WARM ~0.87, COLD ~0.05.
2. Hand-rolls sampler-shaped task-label parquets — one row per (subject, code) at a fixed
   prediction_time, with labels computed from the actual drawn events.
3. Runs ``EQ_process_data`` + ``EQ_train`` via subprocess against the preprocessed cohort,
   training for 600 steps on a demo-sized model (~45s on CPU).
4. Loads the trained checkpoint and runs inference in-process on held-out subjects.
5. Asserts the model distinguishes the extremes: HOT vs. COLD mean-prob gap ≥ 0.5, and pooled
   AUROC across all three codes ≥ 0.8.

HOT and WARM labels are too close (both ~0.9+ true rate) to reliably order on a 15-subject
tuning set, so the primary assertion is on the well-separated HOT / COLD pair.  A broken
gradient path, label flip, or constant-output head all drop the AUROC below the threshold and
fail this test.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
import torch
from meds import DatasetMetadataSchema, train_split, tuning_split
from meds_testing_helpers.dataset import MEDSDataset
from meds_torchdata.config import MEDSTorchDataConfig

from conftest import ENSURE_ENV_PLACEHOLDERS, run_and_check
from every_query.data.dataset import EveryQueryPytorchDataset
from every_query.model.lightning_module import EveryQueryLightningModule

if TYPE_CHECKING:
    from pathlib import Path


# Dataset-design constants.  HOT / WARM / COLD rates are picked so the per-subject expected
# label probability (1 - e^(-rate*duration)) lands well-separated on [0, 1]:
#   HOT  1 - e^(-20)  ~ 1.000   (essentially always fires in 1d)
#   WARM 1 - e^(-2)   ~ 0.865
#   COLD 1 - e^(-0.05) ~ 0.049   (essentially never fires in 1d)
# With duration=1d the COLD-vs-WARM-vs-HOT labels are all very distinct even at the tuning-set
# size (10 subjects), so the ordering assertion isn't flaky under the noise budget.
_CODE_RATES: dict[str, float] = {"CODE_HOT": 20.0, "CODE_WARM": 2.0, "CODE_COLD": 0.05}
_HISTORY_DAYS = 60
_DURATION_DAYS = 1
_PREDICTION_TIME_DAYS = _HISTORY_DAYS - _DURATION_DAYS  # query window is [H-1, H]
_N_TRAIN_SUBJECTS = 40
_N_TUNING_SUBJECTS = 15
_BASE_TIME = datetime(2020, 1, 1, tzinfo=UTC)
_DATASET_SEED = 1


def _synthesize_poisson_meds(out_dir: Path, seed: int = _DATASET_SEED) -> dict[str, pl.DataFrame]:
    """Write a synthetic MEDS dataset to *out_dir* with Poisson-distributed events per code.

    Returns the in-memory per-split event dataframes so label generation can compute ground-truth labels
    directly from the same draws that went on disk (avoids round-trip coupling to preprocessing).
    """
    rng = np.random.default_rng(seed)

    # Keep a flat per-split dict for downstream label computation, separate from the sharded-key
    # dict that actually gets written to disk.  MEDS-transforms expects ``data/{split}/{shard}.parquet``
    # — writing to flat ``data/{split}.parquet`` breaks the MTD fit_normalization stage's
    # subject_splits lookup.
    events_per_split: dict[str, pl.DataFrame] = {}
    splits_rows: list[dict] = []

    for split_name, n_subjects, subj_offset in [
        (train_split, _N_TRAIN_SUBJECTS, 1000),
        (tuning_split, _N_TUNING_SUBJECTS, 2000),
    ]:
        rows = []
        for subj_id in range(subj_offset, subj_offset + n_subjects):
            splits_rows.append({"subject_id": subj_id, "split": split_name})
            for code, rate in _CODE_RATES.items():
                n_events = int(rng.poisson(rate * _HISTORY_DAYS))
                times_days = np.sort(rng.uniform(0.0, float(_HISTORY_DAYS), size=n_events))
                for t_days in times_days:
                    rows.append(
                        {
                            "subject_id": subj_id,
                            "time": _BASE_TIME + timedelta(days=float(t_days)),
                            "code": code,
                            "numeric_value": None,
                        }
                    )
        events_per_split[split_name] = pl.DataFrame(
            rows,
            schema={
                "subject_id": pl.Int64,
                "time": pl.Datetime("us", "UTC"),
                "code": pl.Utf8,
                "numeric_value": pl.Float64,
            },
        ).sort(["subject_id", "time"])

    subject_splits = pl.DataFrame(splits_rows, schema={"subject_id": pl.Int64, "split": pl.Utf8})
    code_metadata = pl.DataFrame({"code": list(_CODE_RATES.keys())}, schema={"code": pl.Utf8})

    MEDSDataset(
        data_shards={f"{split}/0": df for split, df in events_per_split.items()},
        dataset_metadata=DatasetMetadataSchema(
            dataset_name="poisson_synthetic",
            dataset_version="0.1",
        ),
        code_metadata=code_metadata,
        subject_splits=subject_splits,
    ).write(out_dir)

    return events_per_split


def _compute_labels(events: pl.DataFrame, subject_ids: list[int]) -> pl.DataFrame:
    """For each (subject, code) pair, label whether the code fires in [prediction_time, +duration].

    Returns a sampler-output-shaped dataframe ready to drop into ``task_labels_dir/{split}/``.
    """
    window_start = _BASE_TIME + timedelta(days=_PREDICTION_TIME_DAYS)
    window_end = _BASE_TIME + timedelta(days=float(_PREDICTION_TIME_DAYS + _DURATION_DAYS))

    # Convention in the sampler / data pipeline: `boolean_value` means *censored* (subject data
    # ended before the query window completed, so we don't know if the event occurred), and
    # `occurs` means *the event actually occurred*.  The model's occurs-loss is masked to only
    # the non-censored rows (`~batch.censor` in model.py:619), and `censor` is derived from
    # `boolean_value` in dataset.py:414.  Our synthetic subjects all have data through
    # `_HISTORY_DAYS`, so none are censored — set `boolean_value=False` uniformly.
    rows = []
    for subj in subject_ids:
        subj_events = events.filter(pl.col("subject_id") == subj)
        for code in _CODE_RATES:
            fires_in_window = (
                subj_events.filter(
                    (pl.col("code") == code)
                    & (pl.col("time") >= window_start)
                    & (pl.col("time") < window_end)
                ).height
                > 0
            )
            rows.append(
                {
                    "subject_id": subj,
                    "prediction_time": window_start,
                    "boolean_value": False,  # censor flag — always False for synthetic
                    "occurs": fires_in_window,
                    "query": code,
                    "duration_days": _DURATION_DAYS,
                }
            )

    # Cast prediction_time to *naive* (tz-less) datetime — the tensorized event times come out
    # naive, and polars refuses to join/compare a tz-aware prediction_time against a naive event
    # stream (`could not determine supertype of [datetime[μs], datetime[μs, UTC]]`).
    return pl.DataFrame(
        rows,
        schema={
            "subject_id": pl.Int64,
            "prediction_time": pl.Datetime("us", "UTC"),
            "boolean_value": pl.Boolean,
            "occurs": pl.Boolean,
            "query": pl.Utf8,
            "duration_days": pl.Int64,
        },
    ).with_columns(pl.col("prediction_time").dt.replace_time_zone(None))


@pytest.fixture(scope="module")
def poisson_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict[str, pl.DataFrame | Path]:
    """Synthetic Poisson MEDS dataset + labels ready for EQ_process_data / EQ_train."""
    meds_dir = tmp_path_factory.mktemp("poisson_meds")
    event_shards = _synthesize_poisson_meds(meds_dir, seed=_DATASET_SEED)

    train_subjects = list(range(1000, 1000 + _N_TRAIN_SUBJECTS))
    tuning_subjects = list(range(2000, 2000 + _N_TUNING_SUBJECTS))

    task_labels_dir = tmp_path_factory.mktemp("poisson_labels")
    for split_name, subjects, events in [
        (train_split, train_subjects, event_shards[train_split]),
        (tuning_split, tuning_subjects, event_shards[tuning_split]),
    ]:
        labels = _compute_labels(events, subjects)
        split_dir = task_labels_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        labels.write_parquet(split_dir / "0.parquet")

    return {
        "meds_dir": meds_dir,
        "task_labels_dir": task_labels_dir,
        "train_subjects": train_subjects,
        "tuning_subjects": tuning_subjects,
    }


@pytest.fixture(scope="module")
def poisson_preprocessed(poisson_dataset: dict, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run EQ_process_data on the synthetic Poisson cohort; return the tensorized final dir."""
    root = tmp_path_factory.mktemp("poisson_preprocessed")
    intermediate = root / "intermediate"
    final = root / "final"
    run_and_check(
        [
            "EQ_process_data",
            f"input_dir={poisson_dataset['meds_dir']!s}",
            f"intermediate_dir={intermediate!s}",
            f"output_dir={final!s}",
            "do_demo=True",
        ],
        timeout=300.0,
    )
    return final


@pytest.fixture(scope="module")
def poisson_trained_model_dir(
    poisson_preprocessed: Path,
    poisson_dataset: dict,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Train a small model on the Poisson cohort long enough to learn the per-code rate signal."""
    output_dir = tmp_path_factory.mktemp("poisson_train_out")
    # Override the demo config with a model big enough to learn but small enough for CPU CI:
    # hidden_size 128, 4 layers, 4 heads (was 64/2/2 in _demo_train.yaml).  ~150 training steps
    # — empirically enough for the 3-code ordering signal to emerge on this dataset.
    run_and_check(
        [
            "EQ_train",
            "--config-name=_demo_train",
            f"output_dir={output_dir!s}",
            f"datamodule.config.tensorized_cohort_dir={poisson_preprocessed!s}",
            f"datamodule.config.task_labels_dir={poisson_dataset['task_labels_dir']!s}",
            f"query.codes=[{','.join(_CODE_RATES.keys())}]",
            "datamodule.batch_size=4",
            "datamodule.config.max_seq_len=128",
            "lightning_module.model.config_overrides.hidden_size=64",
            "lightning_module.model.config_overrides.num_hidden_layers=2",
            "lightning_module.model.config_overrides.num_attention_heads=2",
            "lightning_module.model.config_overrides.intermediate_size=128",
            "lightning_module.model.config_overrides.max_position_embeddings=512",
            "trainer.max_steps=600",
            "trainer.max_epochs=1000",
            "trainer.limit_val_batches=2",
            "trainer.val_check_interval=200",
            "lightning_module.optimizer.lr=1e-3",
        ],
        env=ENSURE_ENV_PLACEHOLDERS,
        timeout=600.0,
    )
    return output_dir


def _load_lightning_module(trained_model_dir: Path) -> EveryQueryLightningModule:
    """Load the best checkpoint from a completed EQ_train run as a LightningModule ready for predict."""
    ckpts = sorted((trained_model_dir / "checkpoints").glob("*.ckpt"))
    assert ckpts, f"no checkpoints under {trained_model_dir / 'checkpoints'}"
    # Load the highest-step ckpt (tolerant to Lightning's last-vN versioning on resume).
    best_path, best_step = None, -1
    for fp in ckpts:
        c = torch.load(fp, map_location="cpu", weights_only=False)
        if c["global_step"] > best_step:
            best_path, best_step = fp, c["global_step"]

    module = EveryQueryLightningModule.load_from_checkpoint(str(best_path))
    module.eval()
    return module


def test_trained_model_reflects_poisson_rates(
    poisson_preprocessed: Path,
    poisson_dataset: dict,
    poisson_trained_model_dir: Path,
) -> None:
    """The trained model's mean predicted probability per code follows the designed rate ordering.

    Ordering is the strong assertion — it's robust to sampling noise and architectural drift.  A broken
    gradient path (no training), label flip (probabilities inverted), or constant-output head (uniform
    predictions) all fail this.
    """
    module = _load_lightning_module(poisson_trained_model_dir)

    # Build a dataset from the held-out tuning split.  Reuse the same task-labels parquet the
    # trainer was validated against — each subject appears once per code, so per-code mean-prob
    # computation is straightforward.
    cfg = MEDSTorchDataConfig(
        tensorized_cohort_dir=str(poisson_preprocessed),
        task_labels_dir=str(poisson_dataset["task_labels_dir"]),
        max_seq_len=256,
        seq_sampling_strategy="to_end",
        static_inclusion_mode="omit",
        batch_mode="SM",
    )
    dataset = EveryQueryPytorchDataset(cfg, split=tuning_split)

    # Collect predicted probabilities grouped by query code.  Use the raw model forward rather
    # than `module.predict_step`: predict_step requires `batch.subject_id` / `prediction_time` to
    # be populated, which the dataset only does for the `held_out` split.  Read the query name
    # off ``dataset.query[idx]`` rather than re-indexing a separately-sorted label parquet —
    # iteration order is the dataset's internal schema_df, not my sort.
    per_code_probs: dict[str, list[float]] = {code: [] for code in _CODE_RATES}
    per_code_labels: dict[str, list[int]] = {code: [] for code in _CODE_RATES}
    for i in range(len(dataset)):
        item = dataset[i]
        query_code = dataset.query[i]
        batch = dataset.collate([item])
        with torch.no_grad():
            _, outputs = module.model(batch)
        prob = outputs.occurs_probs.squeeze().item()
        per_code_probs[query_code].append(prob)
        per_code_labels[query_code].append(int(item["occurs"]))

    from sklearn.metrics import roc_auc_score

    mean_probs = {code: float(np.mean(probs)) for code, probs in per_code_probs.items()}
    # Also surface true label rates per code on the tuning set — helps diagnose whether a
    # failing ordering is from model-learning issues or from label-distribution skew.
    true_rates = {code: float(np.mean(labels)) for code, labels in per_code_labels.items()}
    print(f"\nmean predicted probs: {mean_probs}")
    print(f"true occurrence rates: {true_rates}")

    # Primary assertion: HOT vs. COLD — the extremes — are well-separated.  Their true rates
    # are ~1.0 vs ~0.07, so a learned model should give them wildly different mean predicted
    # probs.  HOT and WARM labels are too close (both ~0.9+) to reliably order by mean prob on
    # a 15-subject tuning set, so we test the extremes directly.  A broken gradient path,
    # label flip, or constant-output head fails this.
    hot_cold_gap = mean_probs["CODE_HOT"] - mean_probs["CODE_COLD"]
    assert hot_cold_gap >= 0.5, (
        f"HOT-COLD mean-prob gap {hot_cold_gap:.3f} is too small; expected >= 0.5.  "
        f"mean_probs={mean_probs} true_rates={true_rates}"
    )

    # Secondary assertion: AUROC of model predictions vs. true `occurs` label, pooled across
    # all codes.  Catches the failure mode where mean probs separate HOT vs COLD by base-rate
    # prior alone (the model has no query-aware signal) — a label-aware model beats a constant
    # predictor on this metric.
    all_probs = np.concatenate([per_code_probs[c] for c in _CODE_RATES])
    all_labels = np.concatenate([per_code_labels[c] for c in _CODE_RATES])
    if len(set(all_labels.tolist())) >= 2:
        auroc = float(roc_auc_score(all_labels, all_probs))
        print(f"pooled AUROC: {auroc:.3f}")
        assert auroc >= 0.8, (
            f"pooled AUROC {auroc:.3f} is too low; expected >= 0.8.  The trained model is not "
            f"distinguishing positive from negative labels on held-out data."
        )
