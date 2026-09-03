"""Shared synthetic fixtures for the multitask sampler suite (``tests/multitask/``).

Everything here is synthetic: no real cohort is read.  The doctest-namespace override mirrors
``tests/sampler/conftest.py`` so this layer stays offline and never builds the HF demo model.
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

CODES = [f"C//{i}" for i in range(11)] + ["TIMELINE//END"]
K = 5


@pytest.fixture(autouse=True)
def _setup_doctest_namespace():
    yield


def make_codes_parquet(root: Path, codes: list[str] = CODES, *, first_index: int = 1) -> Path:
    """Write ``{root}/metadata/codes.parquet`` with ``code/vocab_index`` starting at ``first_index``."""
    meta = root / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {"code": codes, "code/vocab_index": list(range(first_index, first_index + len(codes)))}
    ).write_parquet(meta / "codes.parquet")
    return meta / "codes.parquet"


def make_events(
    rng: np.random.Generator,
    subject_ids: list[int],
    *,
    n_times: tuple[int, int] = (20, 40),
    horizon_days: int = 400,
    codes: list[str] = CODES,
    with_static: bool = True,
) -> pl.DataFrame:
    """Random events for ``subject_ids``: multiple codes per timestamp, an EOS row, a null-time row."""
    base = datetime(2020, 1, 1)  # noqa: DTZ001 — naive timestamps are fine for synthetic fixtures
    rows = []
    obs = [c for c in codes if c != "TIMELINE//END"]
    for sid in subject_ids:
        n = int(rng.integers(*n_times))
        times = np.sort(rng.integers(0, horizon_days, n))
        for t in times:
            for _ in range(int(rng.integers(1, 3))):
                rows.append(
                    {
                        "subject_id": sid,
                        "time": base + timedelta(days=int(t)),
                        "code": obs[int(rng.integers(0, len(obs)))],
                    }
                )
        if "TIMELINE//END" in codes:
            rows.append(
                {
                    "subject_id": sid,
                    "time": base + timedelta(days=int(times[-1]) + 1),
                    "code": "TIMELINE//END",
                }
            )
        if with_static:
            rows.append({"subject_id": sid, "time": None, "code": obs[0]})
    return pl.DataFrame(rows).with_columns(
        pl.col("time").cast(pl.Datetime("us")), pl.col("subject_id").cast(pl.Int64)
    )


def write_cohort(
    root: Path, shard_to_events: dict[str, pl.DataFrame], split: str = "train", codes: list[str] = CODES
) -> Path:
    """Write a synthetic MEDS root: ``data/{split}/{shard}.parquet`` + ``metadata/codes.parquet``."""
    for shard, df in shard_to_events.items():
        d = root / "data" / split
        d.mkdir(parents=True, exist_ok=True)
        df.write_parquet(d / f"{shard}.parquet")
    make_codes_parquet(root, codes)
    return root


@pytest.fixture
def synthetic_cohort(tmp_path: Path) -> Path:
    """Two-shard synthetic cohort with seven subjects per shard."""
    rng = np.random.default_rng(1)
    shards = {shard: make_events(rng, [int(shard) * 100 + s for s in range(1, 8)]) for shard in ("0", "1")}
    return write_cohort(tmp_path / "cohort", shards)


def base_cfg(cohort: Path, out_dir: Path, **overrides) -> dict:
    cfg = {
        "data_dir": str(cohort),
        "out_dir": str(out_dir),
        "query_codes": str(cohort),
        "split": "train",
        "seed": 3,
        "num_training_examples": 120,
        "num_bounds": K,
        "duration_min": 1,
        "duration_max": 100,
        "duration_distribution": "log-uniform",
        "eventbound_fraction": 0.5,
        "boundary_codes": None,
        "min_prediction_times_per_subject": 5,
        "max_workers": 2,
        "label_chunk_rows": 7,
        "ontology_dir": None,
        "overwrite": False,
    }
    cfg.update(overrides)
    return cfg


def make_index(
    contexts: list[tuple[int, datetime]],
    bounds: list[list[tuple[float, str | None]]],
    conditions: list[list[str]] | None = None,
    *,
    fill_condition: str = "A",
) -> pl.DataFrame:
    """Build a supplied multitask index; ``bounds[i][k]`` is ``(duration_days, bound_event)``.

    ``conditions[i]`` holds the ``K-1`` conditioning codes; by default every slot is ``fill_condition``.
    """
    if conditions is None:
        conditions = [[fill_condition] * (len(row) - 1) for row in bounds]
    return pl.DataFrame(
        {
            "subject_id": pl.Series([c[0] for c in contexts], dtype=pl.Int64),
            "prediction_time": pl.Series([c[1] for c in contexts], dtype=pl.Datetime("us")),
            "durations": pl.Series([[b[0] for b in row] for row in bounds], dtype=pl.List(pl.Float32)),
            "bound_events": pl.Series([[b[1] for b in row] for row in bounds], dtype=pl.List(pl.Utf8)),
            "condition_codes": pl.Series(conditions, dtype=pl.List(pl.Utf8)),
        }
    )


def condition_answers_oracle(meta: pl.DataFrame, dense: np.ndarray, vocab) -> np.ndarray:
    """``(N, K-1)`` bool: ``dense[i, j, index(condition_codes[i, j])]``, computed slot by slot."""
    c2i = vocab.code_to_index()
    rows = meta["condition_codes"].to_list()
    return np.array(
        [[bool(dense[i, j, c2i[c]]) for j, c in enumerate(row)] for i, row in enumerate(rows)], dtype=bool
    ).reshape(meta.height, -1)


def scalar_oracle(
    index_df: pl.DataFrame, events_df: pl.DataFrame, codes: list[str], num_bounds: int
) -> np.ndarray:
    """Every ``(context, boundary, code)`` through ``label_with_event_bounds``; ``(N, K, len(codes))`` bool.

    Rows follow ``index_df``'s order.
    """
    from every_query.generate_tasks.sample_query_sequences import label_with_event_bounds

    recs = []
    for i, r in enumerate(index_df.iter_rows(named=True)):
        for k in range(num_bounds):
            for j, code in enumerate(codes):
                recs.append(
                    {
                        "_ctx_id": i * num_bounds + k,
                        "_position": j,
                        "subject_id": r["subject_id"],
                        "prediction_time": r["prediction_time"],
                        "query": code,
                        "duration_days": r["durations"][k],
                        "bound_event": r["bound_events"][k],
                    }
                )
    idx = pl.DataFrame(recs).with_columns(
        pl.col("_ctx_id").cast(pl.UInt32),
        pl.col("duration_days").cast(pl.Float32),
        pl.col("prediction_time").cast(pl.Datetime("us")),
        pl.col("bound_event").cast(pl.Utf8),
    )
    lab = label_with_event_bounds(idx, events_df.filter(pl.col("time").is_not_null()))
    return np.array(lab["answers"].to_list(), dtype=bool).reshape(index_df.height, num_bounds, len(codes))
