"""Tests for ``EQ_evaluate_multitask``'s task grouping, AUROCs and bootstrap intervals.

Two halves:

- **The defensive ``group_by`` suite.**  ``compute_multitask_metrics`` groups directly on five
  *list* columns rather than hashing them to a ``task_id``, which keeps the spec legible in the
  output but leans on a polars path the project does not otherwise exercise.  ``pyproject`` allows
  ``polars>=1.35,<2``, so these tests pin the behaviour across that range: distinct specs must
  partition, nulls inside lists must stay significant, the ``-1.0`` event-bound sentinel must not
  merge with a real horizon, the partition must round-trip against the grid it came from, and the
  result must not depend on input row order.  If a future polars breaks any of them the fallback is
  a derived ``pl.struct(TASK_KEY).hash()`` key.
- **End-to-end metric behaviour** on small synthetic frames: a separable cell scores 1.0, a
  single-class cell comes back null and is counted, the macro is the mean over the non-null cells,
  and the three macro intervals bracket the point estimate with ``_nested`` no narrower than
  ``_subjects``.

Every fixture here is synthetic — no cohort data is read.
"""

from __future__ import annotations

import polars as pl
import pytest

from every_query.evaluate.evaluate_multitask import TASK_KEY, compute_multitask_metrics

# Small enough to keep the suite fast, large enough that the percentile bounds are stable.
_N_RESAMPLES = 64

_GRID_SCHEMA = {
    "subject_id": pl.Int64,
    "queries": pl.List(pl.Utf8),
    "durations": pl.List(pl.Float32),
    "start_durations": pl.List(pl.Float32),
    "start_events": pl.List(pl.Utf8),
    "bound_events": pl.List(pl.Utf8),
    "label": pl.Boolean,
    "prob": pl.Float32,
}


def _grid(specs: list[dict], n_subjects: int = 4) -> pl.DataFrame:
    """Cross every spec with ``n_subjects`` subjects, exactly as the dense evaluation grid does.

    Each spec dict supplies the five ``TASK_KEY`` columns.  Labels alternate and probabilities are
    perfectly separable, so every cell is scorable unless a test overrides it.
    """
    rows = []
    for spec in specs:
        for s in range(n_subjects):
            positive = s < n_subjects // 2
            rows.append(
                {
                    "subject_id": s,
                    **spec,
                    "label": positive,
                    "prob": 0.9 - 0.1 * s,
                }
            )
    return pl.DataFrame(rows, schema=_GRID_SCHEMA)


def _spec(
    queries: list[str],
    durations: list[float],
    start_durations: list[float] | None = None,
    start_events: list[str | None] | None = None,
    bound_events: list[str | None] | None = None,
) -> dict:
    """One ``SequenceSpec``-shaped set of the five list columns, defaults filled in as
    ``EQ_predict_multitask`` normalises them (``0.0`` starts, null start/bound events)."""
    k = len(queries)
    return {
        "queries": queries,
        "durations": durations,
        "start_durations": start_durations if start_durations is not None else [0.0] * k,
        "start_events": start_events if start_events is not None else [None] * k,
        "bound_events": bound_events if bound_events is not None else [None] * k,
    }


def _tuple_set(series: pl.Series) -> set[tuple]:
    """Hashable view of a list column's values — ``sorted`` cannot order lists holding ``None``."""
    return {tuple(v) for v in series.to_list()}


def _cells(predictions: pl.DataFrame) -> pl.DataFrame:
    by_task, _ = compute_multitask_metrics(predictions, n_resamples=_N_RESAMPLES)
    return by_task


# --- 1. no collision --------------------------------------------------------------------------


def test_same_codes_different_durations_do_not_collide() -> None:
    """Identical query codes over different horizons are different questions, so different cells."""
    grid = _grid([_spec(["A", "B"], [1.0, 30.0]), _spec(["A", "B"], [1.0, 7.0])])

    by_task = _cells(grid)

    assert by_task.height == 2
    assert sorted(by_task["durations"].to_list()) == [[1.0, 7.0], [1.0, 30.0]]
    # Same final code, so `target_code` alone would have merged them.
    assert by_task["target_code"].to_list() == ["B", "B"]


def test_event_bounded_and_horizon_bounded_do_not_collide() -> None:
    """One position bounded by an event, the other by a horizon, splits the cell."""
    grid = _grid(
        [
            _spec(["A", "B"], [1.0, 30.0]),
            _spec(["A", "B"], [1.0, -1.0], bound_events=[None, "DISCHARGE"]),
        ]
    )

    by_task = _cells(grid)

    assert by_task.height == 2
    assert sorted(by_task["duration_bucket"].to_list()) == ["8-30d", "event-bound"]


# --- 2. nulls inside lists are significant ----------------------------------------------------


def test_nulls_inside_bound_events_stay_significant() -> None:
    """``[None, None]`` and ``[None, "DISCHARGE"]`` must not group together.

    This is the failure mode most likely to appear from a polars change: a list comparison that
    treats a null element as "absent" rather than as a value would silently pool an event-bounded
    spec with its horizon-bounded twin.
    """
    grid = _grid(
        [
            _spec(["A", "B"], [1.0, -1.0], bound_events=[None, None]),
            _spec(["A", "B"], [1.0, -1.0], bound_events=[None, "DISCHARGE"]),
        ]
    )

    by_task = _cells(grid)

    assert by_task.height == 2
    assert _tuple_set(by_task["bound_events"]) == {(None, None), (None, "DISCHARGE")}


def test_nulls_inside_start_events_stay_significant() -> None:
    """The same guarantee on the other nullable list column, ``start_events``."""
    grid = _grid(
        [
            _spec(["A"], [30.0], start_durations=[-1.0], start_events=["ADMISSION"]),
            _spec(["A"], [30.0], start_durations=[0.0], start_events=[None]),
        ]
    )

    by_task = _cells(grid)

    assert by_task.height == 2
    assert _tuple_set(by_task["start_events"]) == {("ADMISSION",), (None,)}


# --- 3. the -1.0 sentinel groups by exact equality --------------------------------------------


def test_event_bound_sentinel_never_merges_with_a_real_horizon() -> None:
    """``-1.0`` is a sentinel, not a duration: it must never merge with a short real horizon.

    A float comparison that rounded, clamped at zero, or bucketed before grouping would file the sentinel next
    to a sub-day horizon and average two different questions into one AUROC.
    """
    grid = _grid(
        [
            _spec(["A"], [-1.0], bound_events=["DISCHARGE"]),
            _spec(["A"], [0.5]),
            _spec(["A"], [1.0]),
        ]
    )

    by_task = _cells(grid)

    assert by_task.height == 3
    assert sorted(by_task["durations"].to_list()) == [[-1.0], [0.5], [1.0]]
    # Three distinct horizons in, three distinct cells out, each holding the whole cohort.
    assert by_task["n_rows"].to_list() == [4, 4, 4]


def test_event_bound_sentinel_bucket_is_separate() -> None:
    """The sentinel's descriptive bucket is ``event-bound``, not the shortest horizon."""
    grid = _grid([_spec(["A"], [-1.0], bound_events=["DISCHARGE"]), _spec(["A"], [0.5])])

    by_task = _cells(grid).sort("duration_bucket")

    assert by_task["duration_bucket"].to_list() == ["1d", "event-bound"]


def test_start_duration_sentinel_never_merges_with_a_real_delay() -> None:
    """Same exact-equality guarantee on ``start_durations``' sentinel."""
    grid = _grid(
        [
            _spec(["A"], [30.0], start_durations=[-1.0], start_events=["ADMISSION"]),
            _spec(["A"], [30.0], start_durations=[7.0]),
        ]
    )

    by_task = _cells(grid)

    assert by_task.height == 2
    assert sorted(by_task["start_durations"].to_list()) == [[-1.0], [7.0]]


# --- 4. round-trip against the grid -----------------------------------------------------------


def test_cells_round_trip_against_the_grid() -> None:
    """Cell count equals the distinct spec count and ``sum(n_rows)`` equals the input height.

    The strongest of the five: it catches collision (too few cells, or rows double-counted) and
    fragmentation (too many cells) in one assertion, over a grid that exercises every list column.
    """
    specs = [
        _spec(["A"], [30.0]),
        _spec(["A"], [7.0]),
        _spec(["A", "B"], [1.0, 30.0]),
        _spec(["A", "B"], [1.0, -1.0], bound_events=[None, "DISCHARGE"]),
        _spec(["A", "B"], [1.0, -1.0], bound_events=[None, None]),
        _spec(["B"], [30.0], start_durations=[-1.0], start_events=["ADMISSION"]),
        _spec(["B"], [30.0], start_durations=[7.0]),
    ]
    n_subjects = 6
    grid = _grid(specs, n_subjects=n_subjects)

    by_task = _cells(grid)

    assert by_task.height == len(specs)
    assert by_task["n_rows"].sum() == grid.height
    assert by_task["n_rows"].to_list() == [n_subjects] * len(specs)
    assert by_task["n_subjects"].to_list() == [n_subjects] * len(specs)
    # The emitted keys are exactly the input's distinct specs, nothing invented or dropped.
    assert by_task.select(TASK_KEY).sort(TASK_KEY).equals(grid.select(TASK_KEY).unique().sort(TASK_KEY))


def test_grid_invariant_is_asserted_not_trusted() -> None:
    """A cell missing a subject is a malformed grid and must fail loudly, not silently.

    The macro intervals share one subject index across cells, which is only valid when every cell holds the
    same subjects; a partially-written grid would otherwise intersect cells to different degrees and produce
    quietly wrong intervals rather than absent ones.
    """
    grid = _grid([_spec(["A"], [30.0]), _spec(["B"], [30.0])], n_subjects=4)
    # Drop one subject from one cell only.
    holed = grid.filter(~((pl.col("queries").list.first() == "B") & (pl.col("subject_id") == 3)))

    with pytest.raises(ValueError, match="not dense"):
        compute_multitask_metrics(holed, n_resamples=_N_RESAMPLES)


# --- 5. order independence --------------------------------------------------------------------


def test_shuffling_input_rows_changes_nothing() -> None:
    """Shuffled input yields identical cells and identical metrics.

    Guards both the grouping (which uses ``maintain_order=True``) and the bootstrap, whose subject
    slots are derived from the sorted subject array rather than from row order.
    """
    specs = [
        _spec(["A"], [30.0]),
        _spec(["A", "B"], [1.0, -1.0], bound_events=[None, "DISCHARGE"]),
        _spec(["B"], [7.0], start_durations=[-1.0], start_events=["ADMISSION"]),
    ]
    grid = _grid(specs, n_subjects=8)

    by_task, summary = compute_multitask_metrics(grid, n_resamples=_N_RESAMPLES)
    shuffled = grid.sample(fraction=1.0, shuffle=True, seed=17)
    by_task_shuffled, summary_shuffled = compute_multitask_metrics(shuffled, n_resamples=_N_RESAMPLES)

    assert by_task.equals(by_task_shuffled)
    assert summary.equals(summary_shuffled)


# --- end-to-end metric behaviour --------------------------------------------------------------


def _separable_and_degenerate_grid() -> pl.DataFrame:
    """Three cells over six subjects: separable, anti-separable, and single-class (unscorable)."""
    separable = _spec(["A"], [30.0])
    inverted = _spec(["A"], [7.0])
    single_class = _spec(["A"], [1.0])
    rows = []
    for s in range(6):
        positive = s < 3
        rows.append({"subject_id": s, **separable, "label": positive, "prob": 0.9 if positive else 0.1})
        rows.append({"subject_id": s, **inverted, "label": positive, "prob": 0.1 if positive else 0.9})
        rows.append({"subject_id": s, **single_class, "label": True, "prob": 0.5 + 0.01 * s})
    return pl.DataFrame(rows, schema=_GRID_SCHEMA)


def test_separable_cell_scores_one_and_inverted_cell_scores_zero() -> None:
    """A perfectly ordered cell is AUROC 1.0; a perfectly reversed one is 0.0."""
    by_task, _ = compute_multitask_metrics(_separable_and_degenerate_grid(), n_resamples=_N_RESAMPLES)

    aurocs = {
        tuple(d): a for d, a in zip(by_task["durations"].to_list(), by_task["auroc"].to_list(), strict=True)
    }
    assert aurocs[(30.0,)] == 1.0
    assert aurocs[(7.0,)] == 0.0


def test_single_class_cell_is_null_and_counted() -> None:
    """AUROC is undefined on a single-class cell: null in ``by_task``, counted in ``n_tasks_null``."""
    by_task, summary = compute_multitask_metrics(_separable_and_degenerate_grid(), n_resamples=_N_RESAMPLES)

    single = by_task.filter(pl.col("durations").list.first() == 1.0)
    assert single.height == 1
    assert single["auroc"].to_list() == [None]
    # An unscorable cell gets no fabricated interval either.
    assert single["auroc_ci_lo"].to_list() == [None]
    assert single["auroc_ci_hi"].to_list() == [None]

    row = summary.row(0, named=True)
    assert row["n_tasks_scored"] == 2
    assert row["n_tasks_null"] == 1


def test_macro_is_the_mean_over_non_null_cells() -> None:
    """The headline is the mean of the scored cells only — here ``mean(1.0, 0.0) == 0.5``."""
    by_task, summary = compute_multitask_metrics(_separable_and_degenerate_grid(), n_resamples=_N_RESAMPLES)

    scored = [a for a in by_task["auroc"].to_list() if a is not None]
    assert summary["macro_auroc"][0] == pytest.approx(sum(scored) / len(scored))
    assert summary["macro_auroc"][0] == pytest.approx(0.5)


def test_summary_has_exactly_one_row_and_records_its_bootstrap_settings() -> None:
    """The summary is one row and carries the knobs its intervals were produced with."""
    _, summary = compute_multitask_metrics(
        _separable_and_degenerate_grid(), n_resamples=_N_RESAMPLES, bootstrap_seed=7
    )

    assert summary.height == 1
    row = summary.row(0, named=True)
    assert row["n_resamples"] == _N_RESAMPLES
    assert row["bootstrap_seed"] == 7
    assert set(summary.columns) == {
        "macro_auroc",
        "macro_auroc_ci_lo_tasks",
        "macro_auroc_ci_hi_tasks",
        "macro_auroc_ci_lo_subjects",
        "macro_auroc_ci_hi_subjects",
        "macro_auroc_ci_lo_nested",
        "macro_auroc_ci_hi_nested",
        "n_tasks_scored",
        "n_tasks_null",
        "n_resamples",
        "bootstrap_seed",
    }


def _spread_grid(n_subjects: int = 40) -> pl.DataFrame:
    """A grid with genuinely imperfect, per-cell-varying discrimination.

    Perfectly separable cells give degenerate intervals ``[1.0, 1.0]``, which cannot distinguish a wide
    interval from a narrow one.  Here each cell's scores are a noisy, cell-dependent function of the label, so
    both axes of the bootstrap have something to vary.
    """
    specs = [_spec(["A"], [float(d)]) for d in (2, 7, 30, 90, 180)]
    rows = []
    for c, spec in enumerate(specs):
        for s in range(n_subjects):
            positive = s % 2 == 0
            # Overlapping score distributions; the amount of overlap differs per cell.
            base = 0.5 + (0.30 - 0.05 * c) * (1 if positive else -1)
            rows.append(
                {
                    "subject_id": s,
                    **spec,
                    "label": positive,
                    "prob": base + 0.9 * ((s * 7919 % 97) / 97.0 - 0.5),
                }
            )
    return pl.DataFrame(rows, schema=_GRID_SCHEMA)


def test_macro_intervals_bracket_the_point_estimate() -> None:
    """All three macro intervals satisfy ``lo <= macro <= hi``."""
    _, summary = compute_multitask_metrics(_spread_grid(), n_resamples=200)

    row = summary.row(0, named=True)
    macro = row["macro_auroc"]
    for axis in ("tasks", "subjects", "nested"):
        lo, hi = row[f"macro_auroc_ci_lo_{axis}"], row[f"macro_auroc_ci_hi_{axis}"]
        assert lo is not None and hi is not None, axis
        assert lo <= macro <= hi, (axis, lo, macro, hi)


def test_nested_interval_is_no_narrower_than_the_subjects_interval() -> None:
    """``_nested`` resamples subjects *and* tasks, so it cannot be tighter than ``_subjects``.

    ``_subjects`` conditions on the fixed ``N`` specs and sees only patient noise; ``_nested`` adds
    between-task spread on top of exactly the same AUROC grid.  A ``_nested`` interval narrower than
    ``_subjects`` would mean the cell resampling was collapsing rather than adding variance — the
    signature of redrawing the subject index per cell instead of sharing it.
    """
    _, summary = compute_multitask_metrics(_spread_grid(), n_resamples=200)

    row = summary.row(0, named=True)
    subjects_width = row["macro_auroc_ci_hi_subjects"] - row["macro_auroc_ci_lo_subjects"]
    nested_width = row["macro_auroc_ci_hi_nested"] - row["macro_auroc_ci_lo_nested"]
    assert nested_width >= subjects_width


def test_per_cell_intervals_bracket_their_point_estimates() -> None:
    """Every scored cell's subject bootstrap brackets that cell's own AUROC."""
    by_task, _ = compute_multitask_metrics(_spread_grid(), n_resamples=200)

    for row in by_task.iter_rows(named=True):
        assert row["auroc"] is not None
        assert row["auroc_ci_lo"] <= row["auroc"] <= row["auroc_ci_hi"]
        # A cell with plenty of both classes should almost never resample single-class.
        assert row["n_degenerate_replicates"] == 0


def test_bootstrap_is_reproducible_and_seed_dependent() -> None:
    """A fixed seed gives fixed intervals; a different seed moves them."""
    grid = _spread_grid()
    _, a = compute_multitask_metrics(grid, n_resamples=200, bootstrap_seed=0)
    _, b = compute_multitask_metrics(grid, n_resamples=200, bootstrap_seed=0)
    _, c = compute_multitask_metrics(grid, n_resamples=200, bootstrap_seed=1)

    assert a.equals(b)
    assert a["macro_auroc"][0] == c["macro_auroc"][0]  # the point estimate does not depend on the seed
    assert a["macro_auroc_ci_lo_subjects"][0] != c["macro_auroc_ci_lo_subjects"][0]


def test_subject_is_the_resampling_unit_not_the_row() -> None:
    """Duplicating every subject's rows must not shrink the interval the way row resampling would.

    With ``prediction_times_per_subject > 1`` a subject contributes several perfectly correlated
    rows.  Resampling *rows* would treat those as independent and narrow the interval toward
    ``1/sqrt(2)`` of its width; resampling *subjects* keeps them together, so the width is
    essentially unchanged.
    """
    grid = _spread_grid(n_subjects=40)
    # A second prediction time per subject, carrying identical labels and scores.
    doubled = pl.concat([grid, grid])

    _, single = compute_multitask_metrics(grid, n_resamples=200)
    _, double = compute_multitask_metrics(doubled, n_resamples=200)

    def width(summary: pl.DataFrame) -> float:
        row = summary.row(0, named=True)
        return row["macro_auroc_ci_hi_subjects"] - row["macro_auroc_ci_lo_subjects"]

    assert double["macro_auroc"][0] == pytest.approx(single["macro_auroc"][0])
    assert width(double) == pytest.approx(width(single), rel=0.15)


def test_missing_columns_fail_with_the_column_names() -> None:
    """A frame that is not an ``EQ_predict_multitask`` output fails on the schema, not mid-loop."""
    grid = _grid([_spec(["A"], [30.0])]).drop("bound_events")

    with pytest.raises(ValueError, match="bound_events"):
        compute_multitask_metrics(grid, n_resamples=_N_RESAMPLES)


def test_null_label_is_rejected() -> None:
    """The evaluation grid's answers are binary and never null, so a null label is malformed input."""
    grid = _grid([_spec(["A"], [30.0])]).with_columns(
        pl.when(pl.col("subject_id") == 0).then(None).otherwise(pl.col("label")).alias("label")
    )

    with pytest.raises(ValueError, match="null 'label'"):
        compute_multitask_metrics(grid, n_resamples=_N_RESAMPLES)


def test_zero_resamples_is_rejected() -> None:
    """``n_resamples`` trades runtime for resolution; it is not a way to switch intervals off."""
    with pytest.raises(ValueError, match="n_resamples"):
        compute_multitask_metrics(_grid([_spec(["A"], [30.0])]), n_resamples=0)
