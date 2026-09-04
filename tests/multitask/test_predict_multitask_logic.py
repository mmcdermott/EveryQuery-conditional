"""Logic of ``EQ_predict_multitask`` on a ``QuerySeqSchema`` grid (issue #28), without a checkpoint.

What is tested is the bookkeeping that decides *which* number gets compared to *which* label — get
that wrong and the metrics look plausible and mean nothing:

1.  missing start / bound columns normalize to zeros / ``NO_BOUND_INDEX``;
2.  active duration and event starts tensorize through the explicit multitask opt-in;
3.  query / start / bound codes map to the right vocabulary indices;
4.  unknown codes fail at dataset init, before any model runs;
5.  conditions are all and only the real queries before the final one;
6.  the final code / label is selected per row under variable-length padding;
7.  one-query sequences produce zero conditioning pairs;
8.  target-only logits equal gathered full-vocabulary logits (also in
    ``tests/test_conditional_multitask_ar_model.py``);
9.  the all-vocabulary training forward is unchanged (``tests/test_conditional_multitask_ar_model.py``);
10. prediction output is row-aligned, one row per input row;
11. a grid subject absent from the tensorized cohort is rejected, never silently dropped;
12. no manifest, packed labels or eval-meta sidecar is needed.

The grid rows are written against the session fixture cohort (``tensorized_cohort_dir``), whose
subjects and codes are the real ones the dataset joins against.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from meds import train_split
from meds_torchdata.config import MEDSTorchDataConfig
from omegaconf import DictConfig, OmegaConf

from conftest import _PRED_TIMES, _TRAIN_SUBJECTS
from every_query.data.multitask_dataset import MultitaskBoundaryBatch
from every_query.data.multitask_eval_dataset import MultitaskEvalBatch, QuerySeqMultitaskEvalDataset
from every_query.data.seq_dataset import (
    EVENT_BOUND_DURATION_SENTINEL as SENTINEL,
)
from every_query.data.seq_dataset import (
    NO_BOUND_INDEX,
    ConditionalQueryPytorchDataset,
)
from every_query.predict.predict_multitask import build_dataloader, predictions_to_df, run_inference

# Real codes of the fixture cohort.
Q1, Q2, Q3 = "HR", "TEMP", "DISCHARGE"
ADMIT = "ADMISSION//CARDIAC"
LEGACY_SIDECARS = ("_multitask_manifest.json", "eval_meta", "eval_tasks.parquet")


def _write_grid(root: Path, rows: list[dict], *, split: str = train_split, shard: str = "0") -> Path:
    """Write ``rows`` as one ``QuerySeqSchema`` shard under ``{root}/{split}/``; return ``root``."""
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    cols = {
        "subject_id": pl.Int64,
        "prediction_time": pl.Datetime("us"),
        "queries": pl.List(pl.Utf8),
        "durations": pl.List(pl.Float32),
        "answers": pl.List(pl.Boolean),
    }
    if "bound_events" in rows[0]:
        cols["bound_events"] = pl.List(pl.Utf8)
    if "start_durations" in rows[0]:
        cols["start_durations"] = pl.List(pl.Float32)
        cols["start_events"] = pl.List(pl.Utf8)
    pl.DataFrame(rows, schema=cols).write_parquet(split_dir / f"{shard}.parquet")
    return root


def _row(subject: int, queries: list[str], answers: list[bool], **extra) -> dict:
    return {
        "subject_id": subject,
        "prediction_time": _PRED_TIMES[subject],
        "queries": queries,
        "durations": [float(7 * (i + 1)) for i in range(len(queries))],
        "answers": answers,
        **extra,
    }


def _mixed_rows() -> list[dict]:
    """1-, 2-, 3- and 2-query rows: one per fixture subject, no optional columns."""
    a, b, c, d = _TRAIN_SUBJECTS
    return [
        _row(a, [Q1], [True]),
        _row(b, [Q1, Q2], [False, True]),
        _row(c, [Q2, Q1, Q3], [True, True, False]),
        _row(d, [Q3, Q2], [False, False]),
    ]


def _dataset(cohort: Path, grid: Path, **kw) -> QuerySeqMultitaskEvalDataset:
    cfg = MEDSTorchDataConfig(
        tensorized_cohort_dir=str(cohort),
        task_labels_dir=str(grid),
        max_seq_len=64,
        seq_sampling_strategy="to_end",
        static_inclusion_mode="omit",
        batch_mode="SM",
    )
    return QuerySeqMultitaskEvalDataset(cfg, split=train_split, **kw)


def _collate_all(ds: QuerySeqMultitaskEvalDataset) -> MultitaskEvalBatch:
    return ds.collate([ds[i] for i in range(len(ds))])


# --- 1 / 12: defaults and no sidecars -----------------------------------------------------------


def test_missing_start_and_bound_columns_normalize_to_defaults(tensorized_cohort_dir, tmp_path):
    grid = _write_grid(tmp_path / "grid", _mixed_rows())
    for name in LEGACY_SIDECARS:
        assert not list(grid.rglob(name)), "the grid must carry no legacy sidecar"
    assert not list(grid.rglob("*.labels.npy"))
    ds = _dataset(tensorized_cohort_dir, grid)
    assert not ds.has_starts and not ds.has_bound_events and ds.allow_active_starts
    batch = _collate_all(ds)
    assert isinstance(batch, MultitaskEvalBatch)
    assert batch.num_bounds == 3
    assert torch.equal(batch.q_start_durations, torch.zeros(4, 3))
    assert torch.equal(batch.q_start_codes, torch.full((4, 3), NO_BOUND_INDEX))
    assert torch.equal(batch.q_bound_codes, torch.full((4, 3), NO_BOUND_INDEX))
    assert batch.q_start_durations.dtype == torch.float32 and batch.q_start_codes.dtype == torch.long


# --- 2 / 3: active starts, bounds, and the vocabulary indices ---------------------------------


def test_active_starts_and_bounds_tensorize_with_the_cohort_indices(tensorized_cohort_dir, tmp_path):
    a, b, c, d = _TRAIN_SUBJECTS
    rows = [
        _row(
            a,
            [Q1, Q2],
            [True, False],
            start_durations=[7.0, 0.0],
            start_events=[None, None],
            bound_events=[None, Q3],
        ),
        _row(b, [Q3], [True], start_durations=[SENTINEL], start_events=[ADMIT], bound_events=[None]),
        _row(
            c,
            [Q2, Q1, Q3],
            [True, True, False],
            start_durations=[0.0, SENTINEL, 3.0],
            start_events=[None, ADMIT, None],
            bound_events=[Q3, None, ADMIT],
        ),
        _row(d, [Q1], [False], start_durations=[0.0], start_events=[None], bound_events=[None]),
    ]
    grid = _write_grid(tmp_path / "grid", rows)
    # The ordinary sequence path must refuse this grid; the adapter is the opt-in.
    cfg = MEDSTorchDataConfig(
        tensorized_cohort_dir=str(tensorized_cohort_dir),
        task_labels_dir=str(grid),
        max_seq_len=64,
        seq_sampling_strategy="to_end",
        static_inclusion_mode="omit",
        batch_mode="SM",
    )
    with pytest.raises(ValueError, match="active start"):
        ConditionalQueryPytorchDataset(cfg, split=train_split)

    ds = _dataset(tensorized_cohort_dir, grid)
    assert ds.has_starts and ds.has_bound_events
    idx = ds.code_to_index
    batch = _collate_all(ds)
    nb = NO_BOUND_INDEX
    assert batch.q_start_durations.tolist() == [
        [7.0, 0.0, 0.0],
        [SENTINEL, 0.0, 0.0],
        [0.0, SENTINEL, 3.0],
        [0.0, 0.0, 0.0],
    ]
    assert batch.q_start_codes.tolist() == [
        [nb, nb, nb],
        [idx[ADMIT], nb, nb],
        [nb, idx[ADMIT], nb],
        [nb, nb, nb],
    ]
    assert batch.q_bound_codes.tolist() == [
        [nb, idx[Q3], nb],
        [nb, nb, nb],
        [idx[Q3], nb, idx[ADMIT]],
        [nb, nb, nb],
    ]
    assert batch.scored_codes.tolist() == [idx[Q2], idx[Q3], idx[Q3], idx[Q1]]
    assert batch.condition_codes.tolist() == [[idx[Q1], 0], [0, 0], [idx[Q2], idx[Q1]], [0, 0]]


# --- 4: unknown codes fail before inference ---------------------------------------------------


@pytest.mark.parametrize(
    "column", ["queries", "start_events", "bound_events"], ids=["query", "start", "bound"]
)
def test_unknown_codes_fail_at_init(tensorized_cohort_dir, tmp_path, column):
    a = _TRAIN_SUBJECTS[0]
    row = _row(
        a,
        [Q1, Q2],
        [True, False],
        start_durations=[SENTINEL, 0.0],
        start_events=[ADMIT, None],
        bound_events=[None, Q3],
    )
    if column == "queries":
        row["queries"] = ["NOPE//X", Q2]
    elif column == "start_events":
        row["start_events"] = ["NOPE//X", None]
    else:
        row["bound_events"] = [None, "NOPE//X"]
    grid = _write_grid(tmp_path / "grid", [row])
    with pytest.raises(ValueError, match="not in this run's vocabulary"):
        _dataset(tensorized_cohort_dir, grid)


def test_codes_past_the_checkpoint_vocabulary_width_are_rejected(tensorized_cohort_dir, tmp_path):
    grid = _write_grid(tmp_path / "grid", _mixed_rows())
    ds = _dataset(tensorized_cohort_dir, grid)
    widest = int(ds._q_codes.max())
    _dataset(tensorized_cohort_dir, grid, expected_vocab_size=widest + 1)  # fits
    with pytest.raises(ValueError, match="outside the checkpoint's vocabulary"):
        _dataset(tensorized_cohort_dir, grid, expected_vocab_size=widest)


def test_rows_longer_than_the_checkpoints_window_budget_are_rejected_at_init(tensorized_cohort_dir, tmp_path):
    """A 3-query row fits ``max_windows=3`` and is rejected by ``max_windows=2`` before any model runs (the
    model itself would only fail when that row's batch reached the backbone)."""
    grid = _write_grid(tmp_path / "grid", _mixed_rows())
    ds = _dataset(tensorized_cohort_dir, grid, max_windows=3)
    assert ds.max_windows == 3
    with pytest.raises(ValueError, match="max_windows=2"):
        _dataset(tensorized_cohort_dir, grid, max_windows=2)


def _train_cfg(cohort: Path, *, strip_delta_tokens: bool = False) -> DictConfig:
    """The slice of a checkpoint's ``resolved_config.yaml`` that ``build_eval_dataset`` reads."""
    return OmegaConf.create(
        {
            "datamodule": {
                "config": {
                    "_target_": "meds_torchdata.MEDSTorchDataConfig",
                    "tensorized_cohort_dir": str(cohort),
                    "task_labels_dir": "???",
                    "max_seq_len": 64,
                    "seq_sampling_strategy": "to_end",
                    "static_inclusion_mode": "omit",
                    "batch_mode": "SM",
                },
                "dataset_kwargs": {"strip_delta_tokens": strip_delta_tokens},
                "batch_size": 2,
                "num_workers": 0,
            }
        }
    )


def test_build_eval_dataset_checks_the_checkpoint_against_the_cohort(tensorized_cohort_dir, tmp_path):
    """The predictor-level guards: the cohort's vocabulary width must equal the model's tied
    embedding width, the datamodule's delta-token strip must agree with ``use_rope_time``, and the
    window budget flows through to the dataset."""
    from every_query.predict.predict_multitask import build_eval_dataset

    grid = _write_grid(tmp_path / "grid", _mixed_rows())
    cfg = _train_cfg(tensorized_cohort_dir)
    v = MEDSTorchDataConfig(
        tensorized_cohort_dir=str(tensorized_cohort_dir),
        task_labels_dir=str(grid),
        max_seq_len=64,
        seq_sampling_strategy="to_end",
        static_inclusion_mode="omit",
        batch_mode="SM",
    ).vocab_size
    ds = build_eval_dataset(cfg, grid, train_split, expected_vocab_size=v, use_rope_time=False, max_windows=5)
    assert len(ds) == 4 and ds.expected_vocab_size == v and ds.max_windows == 5
    with pytest.raises(ValueError, match="tied embedding table"):
        build_eval_dataset(cfg, grid, train_split, expected_vocab_size=v + 1, use_rope_time=False)
    with pytest.raises(ValueError, match="use_rope_time"):
        build_eval_dataset(cfg, grid, train_split, expected_vocab_size=v, use_rope_time=True)
    with pytest.raises(ValueError, match="max_windows=2"):
        build_eval_dataset(cfg, grid, train_split, expected_vocab_size=v, use_rope_time=False, max_windows=2)


# --- 5 / 6 / 7: conditions, final code and label under padding -------------------------------


def test_conditions_and_final_query_follow_each_real_row(tensorized_cohort_dir, tmp_path):
    rows = _mixed_rows()
    ds = _dataset(tensorized_cohort_dir, _write_grid(tmp_path / "grid", rows))
    idx = ds.code_to_index
    batch = _collate_all(ds)
    assert batch.num_bounds == 3  # padded to the longest row, not to a fixed K
    assert batch.q_mask.tolist() == [
        [True, False, False],
        [True, True, False],
        [True, True, True],
        [True, True, False],
    ]
    assert batch.n_queries.tolist() == [1, 2, 3, 2]
    for i, row in enumerate(rows):
        n = len(row["queries"])
        # Conditions: all and only the real queries before the final one, PAD / False beyond.
        assert batch.condition_codes[i, : n - 1].tolist() == [idx[c] for c in row["queries"][:-1]]
        assert batch.condition_answers[i, : n - 1].tolist() == row["answers"][:-1]
        assert (batch.condition_codes[i, n - 1 :] == 0).all()
        assert not batch.condition_answers[i, n - 1 :].any()
        # The final query of the row, never the padded width.
        assert int(batch.scored_codes[i]) == idx[row["queries"][-1]]
        assert bool(batch.labels[i]) == row["answers"][-1]
        assert batch.q_durations[i, :n].tolist() == row["durations"]
        assert (batch.q_durations[i, n:] == 0).all()


def test_one_query_rows_have_no_conditioning_pairs_and_k_may_be_one(tensorized_cohort_dir, tmp_path):
    a, b = _TRAIN_SUBJECTS[:2]
    ds = _dataset(
        tensorized_cohort_dir, _write_grid(tmp_path / "grid", [_row(a, [Q1], [True]), _row(b, [Q3], [False])])
    )
    batch = _collate_all(ds)
    assert batch.num_bounds == 1
    assert batch.condition_codes.shape == (2, 0) and batch.condition_answers.shape == (2, 0)
    assert batch.q_mask.tolist() == [[True], [True]]
    assert batch.scored_codes.tolist() == [ds.code_to_index[Q1], ds.code_to_index[Q3]]
    assert batch.labels.tolist() == [True, False]


def test_empty_query_lists_are_rejected(tensorized_cohort_dir, tmp_path):
    a = _TRAIN_SUBJECTS[0]
    grid = _write_grid(tmp_path / "grid", [_row(a, [], [])])
    with pytest.raises(ValueError, match="at least one query"):
        _dataset(tensorized_cohort_dir, grid)


# --- 8: the adapter's batch drives the same scoring path as the training batch ----------------


def test_adapter_batch_scores_like_a_training_batch_with_the_same_windows(tensorized_cohort_dir, tmp_path):
    """Feed the adapter's batch and a ``MultitaskBoundaryBatch`` carrying identical windows to the
    model: the target-only logit equals the gathered full-vocabulary logit of the training path."""
    from every_query.model.conditional_multitask_ar_model import ConditionalMultitaskARModel

    ds = _dataset(tensorized_cohort_dir, _write_grid(tmp_path / "grid", _mixed_rows()))
    batch = _collate_all(ds)
    vocab = max(ds.code_to_index.values()) + 1
    torch.manual_seed(0)
    model = ConditionalMultitaskARModel(
        config_overrides={
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "intermediate_size": 32,
            "max_position_embeddings": 64 + 15,
            "vocab_size": vocab,
            "pad_token_id": 0,
            "attention_dropout": 0.0,
        },
        max_windows=5,
    ).eval()
    target_only = model.score_final_query(batch, batch.scored_codes)

    training_like = MultitaskBoundaryBatch(
        code=batch.code,
        numeric_value=batch.numeric_value,
        numeric_value_mask=batch.numeric_value_mask,
        time_delta_days=batch.time_delta_days,
        q_start_durations=batch.q_start_durations,
        q_start_codes=batch.q_start_codes,
        q_durations=batch.q_durations,
        q_bound_codes=batch.q_bound_codes,
        q_mask=batch.q_mask,
        targets=torch.zeros(batch.batch_size, batch.num_bounds, vocab, dtype=torch.bool),
        condition_codes=batch.condition_codes,
        condition_answers=batch.condition_answers,
    )
    _, out = model(training_like)
    last = batch.q_mask.sum(1) - 1
    gathered = out.logits[torch.arange(batch.batch_size), last, batch.scored_codes]
    torch.testing.assert_close(target_only, gathered, atol=1e-5, rtol=1e-5)


# --- 10 / 11: row alignment and dropped subjects ----------------------------------------------


class _ConstantModel:
    """Stands in for the Lightning module: returns a per-row logit derived from the scored code."""

    def __init__(self):
        self.model = self

    def to(self, device):
        return self

    def eval(self):
        return self

    def score_final_query(self, batch, scored_codes):
        return scored_codes.float() / 100.0 + batch.n_queries.float()


def test_predictions_are_row_aligned_one_per_grid_row(tensorized_cohort_dir, tmp_path):
    rows = _mixed_rows()
    ds = _dataset(tensorized_cohort_dir, _write_grid(tmp_path / "grid", rows))
    loader = build_dataloader(ds, batch_size=3)
    probs, labels = run_inference(_ConstantModel(), loader, torch.device("cpu"), len(ds))
    out = predictions_to_df(ds, probs, labels)

    assert out.height == len(rows) == len(ds)
    assert out.columns == [
        "subject_id",
        "prediction_time",
        "queries",
        "start_durations",
        "start_events",
        "durations",
        "bound_events",
        "answers",
        "target_code",
        "label",
        "prob",
    ]
    assert out["subject_id"].to_list() == [r["subject_id"] for r in rows]
    assert out["queries"].to_list() == [r["queries"] for r in rows]
    assert out["target_code"].to_list() == [r["queries"][-1] for r in rows]
    assert out["label"].to_list() == [r["answers"][-1] for r in rows]
    # Normalized defaults are written even though the grid lacked the optional columns.
    assert out["start_durations"].to_list() == [[0.0] * len(r["queries"]) for r in rows]
    assert out["start_events"].to_list() == [[None] * len(r["queries"]) for r in rows]
    assert out["bound_events"].to_list() == [[None] * len(r["queries"]) for r in rows]
    # Each prob is the one computed for *that* row (the stand-in encodes the scored code and length).
    expected = torch.sigmoid(
        torch.tensor([ds.code_to_index[r["queries"][-1]] / 100.0 + len(r["queries"]) for r in rows])
    )
    assert out["prob"].to_numpy() == pytest.approx(expected.numpy(), abs=1e-6)
    assert out["prob"].dtype == pl.Float32


def test_run_inference_requires_every_row(tensorized_cohort_dir, tmp_path):
    ds = _dataset(tensorized_cohort_dir, _write_grid(tmp_path / "grid", _mixed_rows()))
    loader = build_dataloader(ds, batch_size=2)
    with pytest.raises(RuntimeError, match="yielded 4 prediction"):
        run_inference(_ConstantModel(), loader, torch.device("cpu"), len(ds) + 1)


def test_predictions_to_df_rejects_misaligned_labels(tensorized_cohort_dir, tmp_path):
    ds = _dataset(tensorized_cohort_dir, _write_grid(tmp_path / "grid", _mixed_rows()))
    probs = np.zeros(len(ds), dtype=np.float32)
    labels = np.array([r["answers"][-1] for r in _mixed_rows()])
    predictions_to_df(ds, probs, labels)
    with pytest.raises(RuntimeError, match="disagrees with answers\\[-1\\]"):
        predictions_to_df(ds, probs, ~labels)
    with pytest.raises(RuntimeError, match="for 4 grid row"):
        predictions_to_df(ds, probs[:-1], labels[:-1])


def test_a_grid_subject_absent_from_the_cohort_is_rejected(tensorized_cohort_dir, tmp_path):
    rows = [
        *_mixed_rows(),
        {
            "subject_id": 999_999_999,
            "prediction_time": datetime(2010, 1, 1),
            "queries": [Q1],
            "durations": [7.0],
            "answers": [True],
        },
    ]
    grid = _write_grid(tmp_path / "grid", rows)
    with pytest.raises(RuntimeError, match="1 grid row\\(s\\) were dropped"):
        _dataset(tensorized_cohort_dir, grid)


def test_only_this_splits_shards_are_read(tensorized_cohort_dir, tmp_path):
    """A grid root holding another split's parquets (or several shards) is read per split."""
    grid = _write_grid(tmp_path / "grid", _mixed_rows()[:2], shard="0")
    _write_grid(grid, _mixed_rows()[2:], shard="1")
    _write_grid(grid, [_row(_TRAIN_SUBJECTS[0], [Q1], [True])], split="tuning", shard="0")
    ds = _dataset(tensorized_cohort_dir, grid)
    assert ds.n_grid_rows == 4 and len(ds) == 4


def test_empty_grid_is_rejected(tensorized_cohort_dir, tmp_path):
    grid = tmp_path / "grid"
    (grid / train_split).mkdir(parents=True)
    pl.DataFrame(
        schema={
            "subject_id": pl.Int64,
            "prediction_time": pl.Datetime("us"),
            "queries": pl.List(pl.Utf8),
            "durations": pl.List(pl.Float32),
            "answers": pl.List(pl.Boolean),
        }
    ).write_parquet(grid / train_split / "0.parquet")
    with pytest.raises(ValueError, match="has no rows"):
        _dataset(tensorized_cohort_dir, grid)
    # A grid root that exists but holds no parquet for *this* split.
    other = _write_grid(tmp_path / "other", [_row(_TRAIN_SUBJECTS[0], [Q1], [True])], split="tuning")
    with pytest.raises(FileNotFoundError, match="no QuerySeqSchema parquets"):
        _dataset(tensorized_cohort_dir, other)
