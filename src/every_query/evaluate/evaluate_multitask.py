"""Metrics for multitask conditional-query predictions — ``EQ_evaluate_multitask``.

Consumes the one-row-per-grid-row parquet written by ``EQ_predict_multitask`` (``subject_id``,
``prediction_time``, the five window list columns, ``answers``, ``target_code``, ``label``,
``prob``) and emits two metric tables.

**Grouping key — the query specification**, :data:`TASK_KEY`.  That *is* the task:
``EQ_generate_evaluation_query_sequences`` resolves ``N`` ``SequenceSpec``s once and labels every
one of them at every context, so grouping on the spec recovers exactly those ``N`` cells, each
populated by the whole cohort.  Pooling instead would measure cross-query base-rate separation
rather than within-task discrimination.

``answers[:-1]`` is deliberately **not** in the key.  The conditioning answers vary per context, so
adding them would split each spec cell into up to ``2 ** (K - 1)`` sub-buckets skewed hard toward
all-``False`` (most codes are rare), and AUROC is undefined on a single-class cell, so most of the
partition would come back null.  Prior answers are context, not identity: the class label is
``label`` (i.e. ``answers[-1]``, the final query's answer) and the score is ``prob``.

Outputs, both derived from one bootstrap pass (see :func:`compute_multitask_metrics`):

- ``<metrics_stem>.by_task.parquet`` — one row per spec: the spec itself, descriptive
  ``target_code`` / ``n_queries`` / ``duration_bucket``, counts, prevalence, the within-cell AUROC
  (null when single-class) and its 95% subject-cluster bootstrap interval.
- ``<metrics_stem>.summary.parquet`` — one row: ``macro_auroc`` (the mean over non-null cells) and
  three 95% macro intervals — ``_tasks``, ``_subjects``, ``_nested`` — alongside
  ``n_tasks_scored`` / ``n_tasks_null``, so a macro over 12 of 64 cells cannot pass as a macro over
  64.

Bootstrap intervals are **always computed, never opt-in**: a point AUROC without an interval
invites being quoted as if it were precise, and every interval here is a percentile over one
``n_cells x n_resamples`` AUROC grid that the per-cell intervals already pay for.  ``n_resamples``
is the only knob, and it trades runtime for interval resolution rather than switching the intervals
off.
"""

import logging
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score

from every_query.data.query_seq_dataset import (
    BOUND_EVENTS_COL,
    DURATIONS_COL,
    QUERIES_COL,
    START_DURATIONS_COL,
    START_EVENTS_COL,
)
from every_query.evaluate.metrics import _auroc_or_none

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CONFIGS = str(files("every_query") / "evaluate" / "configs")

#: The query specification — grouping on these five list columns recovers the evaluation grid's
#: ``SequenceSpec``s exactly.  ``answers`` is excluded on purpose (see the module docstring).
TASK_KEY = [QUERIES_COL, DURATIONS_COL, START_DURATIONS_COL, START_EVENTS_COL, BOUND_EVENTS_COL]

SUBJECT_ID_COL = "subject_id"
LABEL_COL = "label"
PROB_COL = "prob"

REQUIRED_COLUMNS = [*TASK_KEY, SUBJECT_ID_COL, LABEL_COL, PROB_COL]

# Match `upstream/task-auroc-ci`'s convention so the evaluator's intervals and the training-time
# callback's intervals mean the same thing: percentile method, 1000 resamples, 95%, seed 0.
DEFAULT_N_RESAMPLES = 1000
DEFAULT_BOOTSTRAP_SEED = 0
CONFIDENCE_LEVEL = 0.95

# (upper-exclusive edge in days, bucket label) pairs, copied from the conditional-sequence
# evaluator this CLI is adapted from.  Descriptive only — a rollup axis, never a grouping key.
DURATION_BUCKETS = [
    (2, "1d"),
    (8, "2-7d"),
    (31, "8-30d"),
    (91, "31-90d"),
    (181, "91-180d"),
    (366, "181-365d"),
]

EVENT_BOUND_BUCKET = "event-bound"


def _duration_bucket(duration_days: float | None) -> str | None:
    """Map the final query's horizon to a human-readable bucket label.

    An **event-bounded** query has no horizon: its window ends at a boundary event and its duration
    is the negative event-bound sentinel.  It gets its own bucket rather than falling through the
    ``<`` ladder, where ``-1.0 < 2`` would file every such row under the shortest horizon and
    quietly mix two different questions into one label.

    Examples:
        >>> [_duration_bucket(d) for d in (0.5, 3.0, 20.0, 400.0, -1.0, None)]
        ['1d', '2-7d', '8-30d', '>365d', 'event-bound', None]
    """
    if duration_days is None:
        return None
    if duration_days < 0:
        return EVENT_BOUND_BUCKET
    for hi, label in DURATION_BUCKETS:
        if duration_days < hi:
            return label
    return ">365d"


def _auroc_or_nan(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUROC of one bootstrap replicate, ``nan`` when the resample happened to land single-class.

    The point-estimate path uses :func:`~every_query.evaluate.metrics._auroc_or_none` and yields
    ``None``; replicates live in a float grid, so they use ``nan`` and are read back with
    ``np.nanpercentile``.

    Examples:
        >>> _auroc_or_nan(np.array([True, False]), np.array([0.9, 0.1]))
        1.0
        >>> bool(np.isnan(_auroc_or_nan(np.array([True, True]), np.array([0.9, 0.1]))))
        True
    """
    n_pos = int(y_true.sum())
    if n_pos == 0 or n_pos == y_true.size:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _percentile_bounds(values: np.ndarray) -> tuple[float | None, float | None]:
    """Two-sided ``CONFIDENCE_LEVEL`` percentile bounds, ignoring degenerate (``nan``) replicates.

    ``(None, None)`` when every replicate was degenerate — an interval resting on nothing is not an
    interval, and a null is easier to notice downstream than a silently wide one.

    Examples:
        >>> lo, hi = _percentile_bounds(np.array([0.0, 0.5, 1.0]))
        >>> round(lo, 6), round(hi, 6)
        (0.025, 0.975)
        >>> _percentile_bounds(np.array([0.4, np.nan, 0.4]))  # degenerate replicates skipped
        (0.4, 0.4)
        >>> _percentile_bounds(np.array([np.nan, np.nan]))
        (None, None)
    """
    if values.size == 0 or bool(np.isnan(values).all()):
        return None, None
    tail = (1.0 - CONFIDENCE_LEVEL) / 2.0 * 100.0
    lo, hi = np.nanpercentile(values, [tail, 100.0 - tail])
    return float(lo), float(hi)


def _nanmean_over_cells(grid: np.ndarray) -> np.ndarray:
    """Column-wise mean over cells, skipping ``nan``, without ``np.nanmean``'s all-``nan`` warning.

    ``grid`` is ``(n_cells, n_resamples)``; the result is one macro AUROC per replicate.

    Examples:
        >>> _nanmean_over_cells(np.array([[1.0, np.nan], [0.5, np.nan]])).tolist()
        [0.75, nan]
    """
    if grid.size == 0:
        return np.full(grid.shape[1], np.nan)
    valid = ~np.isnan(grid)
    counts = valid.sum(axis=0)
    totals = np.where(valid, grid, 0.0).sum(axis=0)
    return np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)


@dataclass
class _Cell:
    """One task cell: its spec key, its point metrics, and its rows laid out for resampling.

    ``y`` / ``score`` are sorted by the cell's subject *slot* (the subject's index in the run-wide
    sorted subject array) and ``offsets`` is the resulting CSR-style row index, so the rows of
    subject slot ``s`` are ``y[offsets[s]:offsets[s + 1]]``.  That is what lets one shared subject
    draw be gathered into every cell in vectorised form.
    """

    key: tuple
    key_frame: pl.DataFrame
    n_rows: int
    n_positive: int
    n_subjects: int
    auroc: float | None
    y: np.ndarray
    score: np.ndarray
    offsets: np.ndarray


def _gather_rows(offsets: np.ndarray, draw: np.ndarray) -> np.ndarray:
    """Row indices for a subject draw: every row of every drawn subject, repeats included.

    ``draw`` holds subject slots sampled with replacement, so a subject drawn twice contributes its
    rows twice — which is the point of a *cluster* bootstrap.  When each subject owns exactly one
    row per cell (``prediction_times_per_subject=1``, the default) this collapses to ``offsets[draw]``.

    Examples:
        Three subjects with 1, 2 and 1 rows; drawing subject 1 twice takes both of its rows twice:

        >>> offsets = np.array([0, 1, 3, 4])
        >>> _gather_rows(offsets, np.array([0, 1, 1])).tolist()
        [0, 1, 2, 1, 2]
        >>> _gather_rows(np.array([0, 1, 2, 3]), np.array([2, 0])).tolist()
        [2, 0]
    """
    n_subjects = offsets.size - 1
    if int(offsets[-1]) == n_subjects:  # one row per subject — the common, fast case
        return offsets[draw]
    starts = offsets[draw]
    counts = offsets[draw + 1] - starts
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    ends = np.cumsum(counts)
    return np.arange(total) - np.repeat(ends - counts, counts) + np.repeat(starts, counts)


def _validate_columns(predictions: pl.DataFrame) -> None:
    """Fail with the missing column names rather than a ``KeyError`` deep in the group loop."""
    missing = [c for c in REQUIRED_COLUMNS if c not in predictions.columns]
    if missing:
        raise ValueError(
            f"predictions is missing required column(s) {missing} — EQ_evaluate_multitask needs the "
            f"parquet written by EQ_predict_multitask."
        )
    if predictions[LABEL_COL].null_count():
        raise ValueError(
            f"{predictions[LABEL_COL].null_count()} row(s) have a null {LABEL_COL!r}. The evaluation "
            "grid's answers are binary and never null (censoring is carried by an explicit "
            "TIMELINE//END query), so a null label means the grid or the prediction run is malformed."
        )


def _build_cells(predictions: pl.DataFrame, subjects: np.ndarray) -> list[_Cell]:
    """Partition predictions into task cells, asserting the dense-grid subject invariant.

    Every cell must hold every subject: each spec is labelled at every context, so the grid is
    dense by construction.  The macro intervals depend on it — one shared subject index is only
    valid if it intersects every cell the same way — so a partially-written grid must fail loudly
    here rather than produce quietly wrong intervals.
    """
    cells: list[_Cell] = []
    for key, group in predictions.group_by(TASK_KEY, maintain_order=True):
        slots = np.searchsorted(subjects, group[SUBJECT_ID_COL].to_numpy())
        counts = np.bincount(slots, minlength=subjects.size)
        if not counts.all():
            n_missing = int((counts == 0).sum())
            raise ValueError(
                f"the evaluation grid is not dense: task cell {key[0]!r} covers "
                f"{subjects.size - n_missing} of {subjects.size} subject(s), missing {n_missing}. "
                "Every spec must be labelled at every context; the macro bootstrap intervals assume "
                "one shared subject index intersects every cell identically."
            )
        order = np.argsort(slots, kind="stable")
        y = group[LABEL_COL].to_numpy().astype(bool)[order]
        score = group[PROB_COL].to_numpy().astype(np.float64)[order]
        cells.append(
            _Cell(
                key=key,
                # Carried from the group itself, not rebuilt from python values, so the spec's list
                # dtypes survive to parquet unchanged — and so the key can never drift out of
                # alignment with the metrics, which a second `group_by` pass would risk.
                key_frame=group.select(TASK_KEY).head(1),
                n_rows=group.height,
                n_positive=int(y.sum()),
                n_subjects=int((counts > 0).sum()),
                auroc=_auroc_or_none(y.tolist(), score.tolist()),
                y=y,
                score=score,
                offsets=np.concatenate(([0], np.cumsum(counts))),
            )
        )
    return cells


def _bootstrap_grid(
    cells: list[_Cell],
    scored: list[int],
    n_subjects: int,
    n_resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """The ``(len(scored), n_resamples)`` AUROC grid every interval in this module is read off.

    Per replicate a single subject index is drawn **once and shared across all cells**, then each cell's AUROC
    is recomputed over the rows of its drawn subjects.  Redrawing per cell would destroy the correlation the
    dense grid induces between cells and would narrow every macro interval; it costs the per-cell rows
    nothing, since each row of the grid is still an ordinary subject bootstrap of that cell.
    """
    grid = np.full((len(scored), n_resamples), np.nan)
    for b in range(n_resamples):
        draw = rng.integers(0, n_subjects, n_subjects)
        for k, i in enumerate(scored):
            cell = cells[i]
            rows = _gather_rows(cell.offsets, draw)
            grid[k, b] = _auroc_or_nan(cell.y[rows], cell.score[rows])
    return grid


def _empty_outputs(predictions: pl.DataFrame, n_resamples: int, bootstrap_seed: int) -> tuple:
    """``(by_task, summary)`` for an empty prediction frame, with stable dtypes."""
    by_task = (
        predictions.select(TASK_KEY)
        .clear()
        .with_columns(
            pl.Series("target_code", [], dtype=pl.Utf8),
            pl.Series("n_queries", [], dtype=pl.Int64),
            pl.Series("duration_bucket", [], dtype=pl.Utf8),
            pl.Series("n_rows", [], dtype=pl.Int64),
            pl.Series("n_positive", [], dtype=pl.Int64),
            pl.Series("prevalence", [], dtype=pl.Float64),
            pl.Series("n_subjects", [], dtype=pl.Int64),
            pl.Series("auroc", [], dtype=pl.Float64),
            pl.Series("auroc_ci_lo", [], dtype=pl.Float64),
            pl.Series("auroc_ci_hi", [], dtype=pl.Float64),
            pl.Series("n_degenerate_replicates", [], dtype=pl.Int64),
        )
    )
    return by_task, _summary_frame(None, {}, 0, 0, n_resamples, bootstrap_seed)


def _summary_frame(
    macro_auroc: float | None,
    bounds: dict[str, tuple[float | None, float | None]],
    n_tasks_scored: int,
    n_tasks_null: int,
    n_resamples: int,
    bootstrap_seed: int,
) -> pl.DataFrame:
    """The one-row summary table, with every AUROC column pinned to ``Float64``.

    Pinning matters when nothing was scorable: polars would otherwise infer ``Null`` for the
    all-null columns and two runs of the same CLI would write different on-disk schemas.
    """
    row: dict = {"macro_auroc": macro_auroc}
    for axis in ("tasks", "subjects", "nested"):
        lo, hi = bounds.get(axis, (None, None))
        row[f"macro_auroc_ci_lo_{axis}"] = lo
        row[f"macro_auroc_ci_hi_{axis}"] = hi
    row["n_tasks_scored"] = n_tasks_scored
    row["n_tasks_null"] = n_tasks_null
    row["n_resamples"] = n_resamples
    row["bootstrap_seed"] = bootstrap_seed
    float_cols = [c for c in row if c.startswith("macro_auroc")]
    return pl.DataFrame([row]).with_columns(pl.col(c).cast(pl.Float64) for c in float_cols)


def compute_multitask_metrics(
    predictions: pl.DataFrame,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Compute ``(by_task, summary)`` from an ``EQ_predict_multitask`` prediction frame.

    Args:
        predictions: One row per evaluation-grid row; see the module docstring for the schema.
        n_resamples: Bootstrap replicates.  The only knob — the intervals themselves are not
            optional.
        bootstrap_seed: Seed for both resampling streams, so a fixed prediction frame gives fixed
            intervals.

    Returns:
        ``by_task`` (one row per spec, sorted by the spec so the table does not depend on input row
        order) and ``summary`` (exactly one row).

    Raises:
        ValueError: if a required column is missing, a label is null, ``n_resamples < 1``, or the
            grid is not dense.

    Examples:
        Two specs over four subjects.  ``spec A`` is perfectly separable; ``spec B`` is all-``True``
        and so unscorable.  A tiny ``n_resamples`` keeps the doctest fast.

        >>> from datetime import datetime
        >>> preds = pl.DataFrame({
        ...     "subject_id": [1, 2, 3, 4, 1, 2, 3, 4],
        ...     "prediction_time": [datetime(2024, 1, 1)] * 8,
        ...     "queries": [["A"]] * 4 + [["B"]] * 4,
        ...     "durations": [[30.0]] * 8,
        ...     "start_durations": [[0.0]] * 8,
        ...     "start_events": [[None]] * 8,
        ...     "bound_events": [[None]] * 8,
        ...     "label": [True, True, False, False, True, True, True, True],
        ...     "prob": [0.9, 0.8, 0.2, 0.1, 0.7, 0.6, 0.5, 0.4],
        ... })
        >>> by_task, summary = compute_multitask_metrics(preds, n_resamples=32)
        >>> by_task.select("target_code", "n_queries", "duration_bucket", "n_rows", "auroc").rows()
        [('A', 1, '8-30d', 4, 1.0), ('B', 1, '8-30d', 4, None)]
        >>> summary.select("macro_auroc", "n_tasks_scored", "n_tasks_null").rows()
        [(1.0, 1, 1)]

        The unscorable cell carries a null interval rather than a fabricated one, and the macro is
        the mean over the scored cells only:

        >>> by_task.select("auroc_ci_lo", "auroc_ci_hi").rows()
        [(1.0, 1.0), (None, None)]
        >>> sorted(c for c in summary.columns if c.startswith("macro_auroc_ci"))
        ['macro_auroc_ci_hi_nested', 'macro_auroc_ci_hi_subjects', 'macro_auroc_ci_hi_tasks',
         'macro_auroc_ci_lo_nested', 'macro_auroc_ci_lo_subjects', 'macro_auroc_ci_lo_tasks']
    """
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples} — the intervals are not optional")
    _validate_columns(predictions)

    if predictions.is_empty():
        return _empty_outputs(predictions, n_resamples, bootstrap_seed)

    subjects = np.sort(predictions[SUBJECT_ID_COL].unique().to_numpy())
    cells = _build_cells(predictions, subjects)
    scored = [i for i, c in enumerate(cells) if c.auroc is not None]

    # Two independent streams from one seed, so that changing the number of cells cannot perturb
    # the subject draws (and vice versa) — the two axes are resampled for different reasons.
    subject_seed, task_seed = np.random.SeedSequence(bootstrap_seed).spawn(2)
    grid = _bootstrap_grid(cells, scored, subjects.size, n_resamples, np.random.default_rng(subject_seed))
    task_rng = np.random.default_rng(task_seed)

    rows = []
    ci_by_cell: dict[int, tuple[float | None, float | None]] = {}
    degenerate_by_cell: dict[int, int | None] = {}
    for k, i in enumerate(scored):
        ci_by_cell[i] = _percentile_bounds(grid[k])
        degenerate_by_cell[i] = int(np.isnan(grid[k]).sum())
    for i, cell in enumerate(cells):
        queries, durations = cell.key[0], cell.key[1]
        lo, hi = ci_by_cell.get(i, (None, None))
        rows.append(
            {
                "target_code": queries[-1] if queries else None,
                "n_queries": len(queries),
                "duration_bucket": _duration_bucket(durations[-1] if durations else None),
                "n_rows": cell.n_rows,
                "n_positive": cell.n_positive,
                "prevalence": cell.n_positive / cell.n_rows if cell.n_rows else None,
                "n_subjects": cell.n_subjects,
                "auroc": cell.auroc,
                "auroc_ci_lo": lo,
                "auroc_ci_hi": hi,
                "n_degenerate_replicates": degenerate_by_cell.get(i),
            }
        )

    keys = pl.concat([cell.key_frame for cell in cells])
    by_task = keys.hstack(
        pl.DataFrame(rows).with_columns(
            pl.col("prevalence").cast(pl.Float64),
            pl.col("auroc").cast(pl.Float64),
            pl.col("auroc_ci_lo").cast(pl.Float64),
            pl.col("auroc_ci_hi").cast(pl.Float64),
            pl.col("n_degenerate_replicates").cast(pl.Int64),
        )
    ).sort(TASK_KEY)

    n_scored = len(scored)
    if not n_scored:
        logger.warning("no task cell had both classes present; macro AUROC is undefined")
        return by_task, _summary_frame(None, {}, 0, len(cells), n_resamples, bootstrap_seed)

    point = np.array([cells[i].auroc for i in scored], dtype=np.float64)
    macro_auroc = float(point.mean())

    # `_subjects`: the macro of each replicate's shared-subject draw, cohort noise only.
    macro_subjects = _nanmean_over_cells(grid)
    # `_tasks`: resample the *point* AUROCs — between-task spread only, holding the cohort fixed.
    # This is what upstream/task-auroc-ci's callback computes, so training logs compare directly.
    macro_tasks = point[task_rng.integers(0, n_scored, size=(n_resamples, n_scored))].mean(axis=1)
    # `_nested`: resample cells *within* each subject replicate — both axes at once, and free,
    # since it is a mean over numbers already in the grid.
    cell_idx = task_rng.integers(0, n_scored, size=(n_resamples, n_scored))
    nested_draws = grid[cell_idx, np.arange(n_resamples)[:, None]]
    macro_nested = _nanmean_over_cells(nested_draws.T)

    bounds = {
        "tasks": _percentile_bounds(macro_tasks),
        "subjects": _percentile_bounds(macro_subjects),
        "nested": _percentile_bounds(macro_nested),
    }
    summary = _summary_frame(
        macro_auroc, bounds, n_scored, len(cells) - n_scored, n_resamples, bootstrap_seed
    )
    return by_task, summary


@hydra.main(version_base="1.3", config_path=CONFIGS, config_name="evaluate_multitask")
def main(cfg: DictConfig) -> None:
    predictions_parquet = Path(cfg.predictions_parquet)
    out_stem = Path(cfg.metrics_stem)

    predictions = pl.read_parquet(predictions_parquet)
    logger.info(f"Loaded {predictions.height} grid-row predictions from {predictions_parquet}")

    by_task, summary = compute_multitask_metrics(
        predictions,
        n_resamples=int(cfg.n_resamples),
        bootstrap_seed=int(cfg.bootstrap_seed),
    )

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    by_task.write_parquet(out_stem.with_suffix(".by_task.parquet"))
    summary.write_parquet(out_stem.with_suffix(".summary.parquet"))

    row = summary.row(0, named=True)
    # Always report the task count next to the estimate: a macro over 12 of 64 cells is a different
    # number from a macro over 64, and the two are indistinguishable without it.
    logger.info(
        f"macro AUROC {row['macro_auroc']} over {row['n_tasks_scored']} scored task(s) "
        f"({row['n_tasks_null']} null); 95% CI "
        f"[{row['macro_auroc_ci_lo_nested']}, {row['macro_auroc_ci_hi_nested']}] (nested), "
        f"[{row['macro_auroc_ci_lo_subjects']}, {row['macro_auroc_ci_hi_subjects']}] (subjects), "
        f"[{row['macro_auroc_ci_lo_tasks']}, {row['macro_auroc_ci_hi_tasks']}] (tasks) "
        f"from {row['n_resamples']} resamples, seed {row['bootstrap_seed']}"
    )
    logger.info(
        f"Wrote {by_task.height} task rows to {out_stem}.by_task.parquet and the summary to "
        f"{out_stem}.summary.parquet"
    )


if __name__ == "__main__":
    main()
