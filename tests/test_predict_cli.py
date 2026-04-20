"""Subprocess integration test for ``EQ_predict``.

Acceptance criterion from #81: *"CPU-only integration test in ``tests/test_predict_cli.py``
that exercises the full subprocess path."*

Uses the session-scoped ``eq_trained_model_dir`` fixture (a real trained demo checkpoint +
``resolved_config.yaml``) and builds a ``TaskQuerySchema``-conformant tasks parquet whose
subjects live in the ``held_out`` split of the training cohort.  Runs ``EQ_predict`` in a
subprocess and verifies the output is ``PredictionSchema``-conformant with probabilities in
``[0, 1]`` and one row per input task.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pytest

from every_query.predict.schema import PredictionSchema

_VENV_BIN = str(Path(sys.executable).parent)

# Subject lives in the ``held_out`` split of the ``simple_static_sharded_by_split``
# testing dataset used by ``eq_preprocessed_dataset``.  Prediction time falls inside
# 1500733's event sequence (2010-06-03).
_HELD_OUT_SUBJECT = 1500733
_PRED_TIME = datetime(2010, 6, 3, 15, 0, 0)
_QUERY_CODES = ["HR", "TEMP"]
_DURATION_DAYS = 30.0


@pytest.fixture(scope="module")
def predict_tasks_parquet(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny TaskQuerySchema-conformant parquet for EQ_predict's input."""
    rows = [
        {
            "subject_id": _HELD_OUT_SUBJECT,
            "prediction_time": _PRED_TIME,
            "query": code,
            "duration_days": _DURATION_DAYS,
            "boolean_value": None,
        }
        for code in _QUERY_CODES
    ]
    df = pl.DataFrame(rows).cast(
        {
            "subject_id": pl.Int64,
            "prediction_time": pl.Datetime("us"),
            "query": pl.Utf8,
            "duration_days": pl.Float32,
            "boolean_value": pl.Boolean,
        }
    )
    out = tmp_path_factory.mktemp("eq_predict_tasks") / "tasks.parquet"
    df.write_parquet(out)
    return out


def test_eq_predict_end_to_end(
    eq_trained_model_dir: Path,
    predict_tasks_parquet: Path,
    tmp_path: Path,
) -> None:
    """``EQ_predict`` runs end-to-end and produces a ``PredictionSchema``-conformant parquet.

    Exercises the full subprocess entry point — console-script resolution, Hydra config
    compose, model checkpoint load, per-``(query, duration_days)`` predict loop, join-back
    to preserve inherited label columns, schema-aligned write.
    """
    output_parquet = tmp_path / "predictions.parquet"

    env = os.environ.copy()
    env["PATH"] = _VENV_BIN + os.pathsep + env.get("PATH", "")

    cmd = [
        "EQ_predict",
        f"model_run_dir={eq_trained_model_dir}",
        f"tasks_parquet={predict_tasks_parquet}",
        f"output_parquet={output_parquet}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    assert result.returncode == 0, (
        f"EQ_predict failed (rc={result.returncode})\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert output_parquet.exists(), "EQ_predict did not produce the output parquet"

    # Schema conformance.
    table = pq.read_table(output_parquet)
    PredictionSchema.align(table)

    df = pl.from_arrow(table)
    assert df.height == len(_QUERY_CODES), (
        f"Expected {len(_QUERY_CODES)} prediction rows (one per query), got {df.height}"
    )

    # Probabilities bounded.
    for col in ("censor_prob", "occurs_prob"):
        col_min = float(df[col].min())
        col_max = float(df[col].max())
        assert 0.0 <= col_min <= col_max <= 1.0, f"{col} not in [0,1]: min={col_min} max={col_max}"

    # Every input query is represented in the output.
    assert set(df["query"].to_list()) == set(_QUERY_CODES)
