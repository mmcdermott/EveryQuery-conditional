"""Doctest-namespace injection for ``tests/training_validity/README.md``.

The README's ``>>>`` blocks reference ``events`` and ``labels``; this conftest builds
those DataFrames once per session from the *exact* synthesis code the training-validity
test fixture uses (``_synthesize_meds`` + ``_compute_labels``), so the README shows the
literal training data rather than an unrelated mini-example.
"""

from __future__ import annotations

import polars as pl
import pytest

from tests.training_validity.test_training_validity import (
    _DATASET_SEED,
    _N_TRAIN_SUBJECTS,
    _compute_labels,
    _synthesize_meds,
)


@pytest.fixture(scope="session")
def _oracle_doctest_shards(tmp_path_factory: pytest.TempPathFactory) -> dict[str, pl.DataFrame]:
    meds_dir = tmp_path_factory.mktemp("oracle_doctest_meds")
    event_shards = _synthesize_meds(meds_dir, seed=_DATASET_SEED)
    train_subjects = list(range(1000, 1000 + _N_TRAIN_SUBJECTS))
    return {
        "events": event_shards["train"],
        "labels": _compute_labels(event_shards["train"], train_subjects),
    }


@pytest.fixture(autouse=True)
def _inject_oracle_doctest_namespace(
    doctest_namespace: dict, _oracle_doctest_shards: dict[str, pl.DataFrame]
) -> None:
    doctest_namespace["events"] = _oracle_doctest_shards["events"]
    doctest_namespace["labels"] = _oracle_doctest_shards["labels"]
    doctest_namespace["pl"] = pl
    # Default polars display caps at 10 rows; bump it so the README snapshots can show the
    # full 13-row code-counts and 10-row marker-pair tables without truncation.
    pl.Config.set_tbl_rows(20)
