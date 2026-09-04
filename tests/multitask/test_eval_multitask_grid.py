"""The dense multitask evaluation grid: task groups, the cross-joined index, and the sidecar contract.

The grid's whole value rests on two invariants:

* every context answers every task, so per-task metrics are comparable across cohorts and models;
* the ``eval_meta`` sidecar is row-aligned with the label parquet, so a model output can be traced
  back to the task (and the scored code) it belongs to.

Both are asserted here against a synthetic cohort, along with the group machinery that lets one grid
hold a uniformly-drawn and a prevalence-drawn set of tasks over the same contexts.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from every_query.generate_tasks import sample_evaluation_multitask_sequences as sems
from every_query.generate_tasks.interval_table import INF
from every_query.generate_tasks.sample_evaluation_multitask_sequences import (
    END_RESOLVED_COL,
    GROUP_COL,
    START_RESOLVED_COL,
    TARGET_CODE_COL,
    TASK_ID_COL,
    WINDOW_DAYS_COL,
    _group_settings,
    build_dense_multitask_index,
    build_task_table,
    window_diagnostics,
)
from every_query.generate_tasks.sample_multitask_sequences import (
    LABELS_SUFFIX,
    MANIFEST_NAME,
    build_target_vocabulary,
    read_manifest,
)
from omegaconf import OmegaConf

from tests.multitask.conftest import K, make_events, write_cohort

SPEC_COLS = ["start_durations", "start_events", "durations", "bound_events", "condition_codes"]


@pytest.fixture
def eval_cohort(tmp_path: Path) -> Path:
    """A two-shard synthetic cohort on the ``held_out`` split, with a prevalence column."""
    rng = np.random.default_rng(7)
    shards = {s: make_events(rng, [int(s) * 100 + i for i in range(1, 8)]) for s in ("0", "1")}
    root = write_cohort(tmp_path / "cohort", shards, split="held_out")
    meta_fp = root / "metadata" / "codes.parquet"
    meta = pl.read_parquet(meta_fp)
    # Geometric-ish prevalence so weighted and uniform groups draw visibly different pools.
    occ = [10**6 // (2**i) + 1 for i in range(meta.height)]
    meta.with_columns(pl.Series("code/n_occurrences", occ)).write_parquet(meta_fp)
    return root


def eval_cfg(cohort: Path, out_dir: Path, **overrides) -> OmegaConf:
    cfg = {
        "data_dir": str(cohort),
        "out_dir": str(out_dir),
        "query_codes": str(cohort),
        "split": "held_out",
        "seed": 11,
        "prediction_times_per_subject": 1,
        "min_context_per_subject": 5,
        "subject_subsample_fraction": None,
        "num_evaluation_tasks": 3,
        "task_groups": ["uniform", {"name": "prevalence", "code_weighting": "prevalence"}],
        "num_bounds": K,
        "duration_min": 1,
        "duration_max": 100,
        "duration_distribution": "log-uniform",
        "eventbound_fraction": 0.5,
        "boundary_codes": None,
        "eventstart_fraction": 0.2,
        "prediction_time_start_fraction": 0.4,
        "start_duration_min": 1,
        "start_duration_max": 30,
        "start_event_codes": None,
        "code_weight_column": "code/n_occurrences",
        "code_weight_power": 1.0,
        "exclude_boundary_prefixes": [],
        "target_codes": None,
        "target_code_weighting": "prevalence",
        "target_code_weight_power": 1.0,
        "max_workers": 1,
        "label_chunk_rows": 5,
        "ontology_dir": None,
        "overwrite": True,
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


# --- task groups ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "name", "weighting"),
    [
        ("uniform", "uniform", None),
        ("prevalence", "prevalence", "prevalence"),
        ({"name": "p", "code_weighting": "prevalence"}, "p", "prevalence"),
        ({"name": "u"}, "u", None),
    ],
)
def test_group_settings(raw, name, weighting):
    settings = _group_settings(raw, 0)
    assert settings["name"] == name
    assert settings["code_weighting"] == weighting


def test_task_table_shape_and_ids(eval_cohort: Path, tmp_path: Path):
    cfg = eval_cfg(eval_cohort, tmp_path / "grid")
    vocab = build_target_vocabulary(eval_cohort)
    tasks = build_task_table(cfg, vocab, "held_out")
    assert tasks.height == 6  # 3 per group x 2 groups
    assert tasks[TASK_ID_COL].to_list() == list(range(6))
    assert tasks[GROUP_COL].to_list() == ["uniform"] * 3 + ["prevalence"] * 3
    assert tasks["durations"].list.len().to_list() == [K] * 6
    assert tasks["condition_codes"].list.len().to_list() == [K - 1] * 6
    assert tasks[TARGET_CODE_COL].null_count() == 0


def test_groups_are_seeded_independently(eval_cohort: Path, tmp_path: Path):
    """Dropping the second group must not perturb the first group's draw."""
    vocab = build_target_vocabulary(eval_cohort)
    both = build_task_table(eval_cfg(eval_cohort, tmp_path / "a"), vocab, "held_out")
    only = build_task_table(
        eval_cfg(eval_cohort, tmp_path / "b", task_groups=["uniform"]), vocab, "held_out"
    )
    assert both.head(3).drop(TASK_ID_COL).equals(only.drop(TASK_ID_COL))


def test_task_table_is_deterministic_in_the_seed(eval_cohort: Path, tmp_path: Path):
    vocab = build_target_vocabulary(eval_cohort)
    a = build_task_table(eval_cfg(eval_cohort, tmp_path / "a"), vocab, "held_out")
    b = build_task_table(eval_cfg(eval_cohort, tmp_path / "b"), vocab, "held_out")
    c = build_task_table(eval_cfg(eval_cohort, tmp_path / "c", seed=12), vocab, "held_out")
    assert a.equals(b)
    assert not a.equals(c)


def test_exclusions_reach_the_task_pools(eval_cohort: Path, tmp_path: Path):
    cfg = eval_cfg(
        eval_cohort,
        tmp_path / "grid",
        num_evaluation_tasks=25,
        exclude_boundary_prefixes=["TIMELINE//"],
        eventbound_fraction=1.0,
        eventstart_fraction=1.0,
        prediction_time_start_fraction=0.0,
    )
    vocab = build_target_vocabulary(eval_cohort)
    tasks = build_task_table(cfg, vocab, "held_out")
    for col in ("bound_events", "start_events", TARGET_CODE_COL):
        values = tasks[col].explode() if tasks.schema[col] == pl.List(pl.Utf8) else tasks[col]
        assert not any(str(c).startswith("TIMELINE//") for c in values.drop_nulls().to_list())


# --- the dense index ------------------------------------------------------------------------------


def test_dense_index_is_a_full_cross_join(eval_cohort: Path, tmp_path: Path):
    cfg = eval_cfg(eval_cohort, tmp_path / "grid")
    vocab = build_target_vocabulary(eval_cohort)
    tasks = build_task_table(cfg, vocab, "held_out")
    contexts = pl.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "prediction_time": pl.Series(
                [datetime(2020, 1, 1)] * 3,
                dtype=pl.Datetime("us"),
            ),
        }
    )
    idx = build_dense_multitask_index(contexts, tasks, K)
    assert idx.height == 3 * tasks.height
    counts = idx.group_by("subject_id").agg(pl.col(TASK_ID_COL).n_unique().alias("n"))
    assert counts["n"].to_list() == [tasks.height] * 3
    assert idx["_ctx_id"].n_unique() == idx.height


def test_dense_index_rejects_an_empty_task_table():
    with pytest.raises(ValueError, match="task table is empty"):
        build_dense_multitask_index(pl.DataFrame({"subject_id": [], "prediction_time": []}),
                                    pl.DataFrame({"durations": []}), K)


# --- diagnostics ----------------------------------------------------------------------------------


def test_window_diagnostics_flags_unresolved_boundaries():
    day = 86_400_000_000
    start = np.array([[0, INF, 0]], dtype=np.int64)
    end = np.array([[2 * day, INF, INF]], dtype=np.int64)
    sr, er, wd = window_diagnostics(start, end)
    assert sr.tolist() == [[True, False, True]]
    assert er.tolist() == [[True, False, False]]
    assert wd[0, 0] == pytest.approx(2.0)
    assert np.isnan(wd[0, 1]) and np.isnan(wd[0, 2])


# --- end to end -----------------------------------------------------------------------------------


def test_grid_end_to_end(eval_cohort: Path, tmp_path: Path):
    out_dir = tmp_path / "grid"
    cfg = eval_cfg(eval_cohort, out_dir)
    sems.run(cfg)

    eval_dir = out_dir / "eval" / "held_out"
    meta_dir = out_dir / "eval_meta" / "held_out"
    assert (eval_dir / MANIFEST_NAME).is_file()
    manifest = read_manifest(eval_dir)
    assert manifest["eval_grid"] is True
    assert manifest["n_eval_tasks"] == 6
    assert sorted(manifest["eval_task_groups"]) == ["prevalence", "uniform"]

    tasks = pl.read_parquet(out_dir / "eval_tasks.parquet")
    vocab = build_target_vocabulary(eval_cohort)

    total_rows = 0
    for label_fp in sorted(eval_dir.glob("*.parquet")):
        labels = pl.read_parquet(label_fp)
        sidecar = pl.read_parquet(meta_dir / label_fp.name)
        assert labels.height == sidecar.height
        if labels.height == 0:
            continue
        total_rows += labels.height

        # Row alignment: the same (subject, time) in both, and every stored spec is the one the
        # sidecar's task_id names.
        assert labels["subject_id"].equals(sidecar["subject_id"])
        assert labels["prediction_time"].equals(sidecar["prediction_time"])
        expected = sidecar.select(TASK_ID_COL).join(
            tasks.select(TASK_ID_COL, *SPEC_COLS), on=TASK_ID_COL, how="left"
        )
        for col in SPEC_COLS:
            assert labels[col].equals(expected[col]), col

        # Every context answers every task.
        per_context = sidecar.group_by("subject_id", "prediction_time").agg(
            pl.col(TASK_ID_COL).n_unique().alias("n")
        )
        assert per_context["n"].to_list() == [tasks.height] * per_context.height

        packed = np.load(eval_dir / f"{label_fp.stem}{LABELS_SUFFIX}", mmap_mode="r")
        assert packed.shape == (labels.height, K, vocab.packed_width)
        dense = np.unpackbits(np.asarray(packed), axis=-1, count=vocab.size, bitorder="little")
        assert not dense[:, :, 0].any()  # PAD is never a target

        # Diagnostics are per-window and consistent with each other.
        for col in (START_RESOLVED_COL, END_RESOLVED_COL, WINDOW_DAYS_COL):
            assert sidecar[col].list.len().to_list() == [K] * sidecar.height
        flat = sidecar.select(
            pl.col(START_RESOLVED_COL).explode().alias("s"),
            pl.col(END_RESOLVED_COL).explode().alias("e"),
            pl.col(WINDOW_DAYS_COL).explode().alias("d"),
        )
        both = flat["s"] & flat["e"]
        assert flat["d"].filter(both).is_not_nan().all()
        assert flat["d"].filter(~both).is_nan().all()

    assert total_rows > 0
    summary = json.loads((out_dir / "_eval_summary.json").read_text())
    assert summary["n_tasks"] == 6
    assert summary["n_labeled_rows"] == total_rows
    assert summary["num_bounds"] == K


def test_grid_is_reused_unless_overwritten(eval_cohort: Path, tmp_path: Path):
    out_dir = tmp_path / "grid"
    sems.run(eval_cfg(eval_cohort, out_dir))
    stamps = {
        fp: fp.stat().st_mtime_ns for fp in sorted((out_dir / "eval" / "held_out").glob("*.parquet"))
    }
    sems.run(eval_cfg(eval_cohort, out_dir, overwrite=False))
    assert {fp: fp.stat().st_mtime_ns for fp in stamps} == stamps
