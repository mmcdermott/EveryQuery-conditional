"""Pure helpers of ``EQ_predict_multitask``: sidecar alignment and which code each window scores.

No model is loaded here.  What is tested is the bookkeeping that decides *which* number gets
compared to *which* label - get that wrong and the metrics look plausible and mean nothing.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from every_query.data.multitask_dataset import SOURCE_ROW_COL, SOURCE_SHARD_COL
from every_query.generate_tasks.sample_evaluation_multitask_sequences import (
    END_RESOLVED_COL,
    GROUP_COL,
    START_RESOLVED_COL,
    TARGET_CODE_COL,
    TASK_ID_COL,
    WINDOW_DAYS_COL,
)
from every_query.predict.predict_multitask import (
    align_sidecar,
    check_grid_coverage,
    predictions_to_df,
    read_sidecars,
    scored_code_matrix,
)

K = 3


class _FakeDataset:
    """Enough of ``MultitaskBoundaryPytorchDataset`` for the helpers under test."""

    def __init__(self, condition_codes: np.ndarray, code_to_index: dict[str, int]):
        self._condition_codes = condition_codes
        self.code_to_index = code_to_index
        self.num_bounds = condition_codes.shape[1] + 1


def _write_sidecar(meta_dir: Path, split: str, shard: str, rows: int, first_task: int) -> None:
    d = meta_dir / split
    d.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "subject_id": list(range(rows)),
            "prediction_time": [datetime(2020, 1, 1)] * rows,
            TASK_ID_COL: [first_task + i for i in range(rows)],
            GROUP_COL: ["uniform"] * rows,
            TARGET_CODE_COL: ["T"] * rows,
            START_RESOLVED_COL: [[True] * K] * rows,
            END_RESOLVED_COL: [[False] * K] * rows,
            WINDOW_DAYS_COL: [[1.0] * K] * rows,
        }
    ).write_parquet(d / f"{shard}.parquet")


def test_read_sidecars_keys_rows_like_the_dataset(tmp_path: Path):
    _write_sidecar(tmp_path, "held_out", "0", 3, first_task=0)
    _write_sidecar(tmp_path, "held_out", "1", 2, first_task=10)
    side = read_sidecars(tmp_path, "held_out")
    assert side.height == 5
    assert side[SOURCE_SHARD_COL].to_list() == ["held_out/0"] * 3 + ["held_out/1"] * 2
    assert side[SOURCE_ROW_COL].to_list() == [0, 1, 2, 0, 1]


def test_read_sidecars_requires_the_split(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="no evaluation sidecars"):
        read_sidecars(tmp_path, "held_out")


def _schema_df(pairs: list[tuple[str, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            SOURCE_SHARD_COL: [p[0] for p in pairs],
            SOURCE_ROW_COL: [p[1] for p in pairs],
            "subject_id": list(range(len(pairs))),
            "prediction_time": [datetime(2020, 1, 1)] * len(pairs),
        }
    )


def test_align_sidecar_follows_dataset_order(tmp_path: Path):
    """The dataset's row order is not the sidecar's file order; the join must follow the dataset."""
    _write_sidecar(tmp_path, "held_out", "0", 3, first_task=0)
    _write_sidecar(tmp_path, "held_out", "1", 2, first_task=10)
    side = read_sidecars(tmp_path, "held_out")
    schema_df = _schema_df([("held_out/1", 1), ("held_out/0", 2), ("held_out/1", 0)])
    aligned = align_sidecar(schema_df, side)
    assert aligned.height == 3
    assert aligned[TASK_ID_COL].to_list() == [11, 2, 10]


def test_align_sidecar_rejects_an_unmatched_row(tmp_path: Path):
    _write_sidecar(tmp_path, "held_out", "0", 2, first_task=0)
    side = read_sidecars(tmp_path, "held_out")
    with pytest.raises(RuntimeError, match="no evaluation sidecar entry"):
        align_sidecar(_schema_df([("held_out/0", 0), ("held_out/9", 0)]), side)


def test_grid_coverage_rejects_a_dropped_grid_row(tmp_path: Path):
    """A grid row whose subject is absent from the cohort vanishes from the dataset silently; the
    sidecar still counts it, and that disagreement must be an error, not a shrunken grid."""
    _write_sidecar(tmp_path, "held_out", "0", 3, first_task=0)
    side = read_sidecars(tmp_path, "held_out")
    schema_df = _schema_df([("held_out/0", 0), ("held_out/0", 1)])
    align_sidecar(schema_df, side)  # every dataset row is matched, so alignment alone is silent
    with pytest.raises(RuntimeError, match="1 grid row\\(s\\) were dropped"):
        check_grid_coverage(schema_df, side)
    check_grid_coverage(_schema_df([("held_out/0", i) for i in range(3)]), side)


def test_align_sidecar_rejects_a_non_multitask_frame(tmp_path: Path):
    _write_sidecar(tmp_path, "held_out", "0", 1, first_task=0)
    side = read_sidecars(tmp_path, "held_out")
    with pytest.raises(ValueError, match="not a multitask dataset"):
        align_sidecar(pl.DataFrame({"subject_id": [1]}), side)


def test_scored_codes_are_conditions_then_the_target():
    """Windows 0..K-2 score their conditioning code; the final window scores the task's target."""
    dataset = _FakeDataset(
        condition_codes=np.array([[3, 4], [5, 6]], dtype=np.int64),
        code_to_index={"A": 3, "B": 4, "C": 5, "D": 6, "T": 9, "U": 7},
    )
    sidecar = pl.DataFrame({TARGET_CODE_COL: ["T", "U"]})
    idx, codes = scored_code_matrix(dataset, sidecar, K)
    assert idx.tolist() == [[3, 4, 9], [5, 6, 7]]
    assert codes == [["A", "B", "T"], ["C", "D", "U"]]


def test_scored_codes_reject_an_out_of_vocabulary_target():
    dataset = _FakeDataset(np.array([[1, 2]], dtype=np.int64), {"A": 1, "B": 2})
    with pytest.raises(ValueError, match="outside the cohort vocabulary"):
        scored_code_matrix(dataset, pl.DataFrame({TARGET_CODE_COL: ["NOPE"]}), K)


def test_scored_codes_reject_an_empty_grid():
    dataset = _FakeDataset(np.zeros((0, K - 1), dtype=np.int64), {"T": 1})
    empty = pl.DataFrame({TARGET_CODE_COL: []}, schema={TARGET_CODE_COL: pl.Utf8})
    with pytest.raises(ValueError, match="grid has no rows"):
        scored_code_matrix(dataset, empty, K)


def test_scored_codes_check_the_conditioning_width():
    dataset = _FakeDataset(np.array([[1]], dtype=np.int64), {"A": 1, "T": 2})
    with pytest.raises(ValueError, match="conditioning codes per row"):
        scored_code_matrix(dataset, pl.DataFrame({TARGET_CODE_COL: ["T"]}), K)


def test_predictions_explode_one_row_per_window():
    schema_df = _schema_df([("held_out/0", 0), ("held_out/0", 1)])
    sidecar = pl.DataFrame(
        {
            TASK_ID_COL: [0, 1],
            GROUP_COL: ["uniform", "prevalence"],
            START_RESOLVED_COL: [[True, True, False], [False, True, True]],
            END_RESOLVED_COL: [[True, False, False], [True, True, False]],
            WINDOW_DAYS_COL: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        }
    )
    probs = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    labels = np.array([[True, False, True], [False, False, True]])
    out = predictions_to_df(schema_df, sidecar, [["A", "B", "T"], ["C", "D", "U"]], probs, labels)

    assert out.height == 6
    assert out["position"].to_list() == [0, 1, 2, 0, 1, 2]
    assert out["is_final"].to_list() == [False, False, True, False, False, True]
    assert out["scored_code"].to_list() == ["A", "B", "T", "C", "D", "U"]
    assert out["prob"].to_list() == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], rel=1e-6)
    assert out["label"].to_list() == [True, False, True, False, False, True]
    assert out[GROUP_COL].to_list() == ["uniform"] * 3 + ["prevalence"] * 3
    assert out[WINDOW_DAYS_COL].to_list() == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
