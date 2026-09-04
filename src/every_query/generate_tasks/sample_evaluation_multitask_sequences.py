"""Dense evaluation grid for the all-vocabulary multitask model.

``EQ_generate_evaluation_multitask_sequences``.

The multitask counterpart of
:mod:`~every_query.generate_tasks.sample_evaluation_query_sequences`, and it is organized the same
way: a small number ``N`` of *task specifications* is drawn **once**, seeded independently of the
cohort, and then labeled at every sampled prediction time of the split, so every context answers the
identical ``N`` tasks and per-task metrics are comparable across cohorts and across models.

A multitask task specification is one context's worth of the training sampler's own draw - ``K``
windows (each with an explicit start and end spec, per issue #24) plus the ``K-1`` conditioning codes
(issue #22) - with one addition: a **target code**, the vocabulary code whose probability at the
*final* window is the number a metric is computed on.  During training every window's hidden state
predicts every code; at evaluation only the final window's target code is scored, which is what makes
a task a task::

    [W0, C0, A0, W1, C1, A1, ..., W(K-1)]   ->   P(target_code inside window K-1 | history, prior answers)

Tasks are drawn in one or more **groups**, each with its own boundary-code sampling policy (see
``task_groups`` in the config).  That is the point of the endpoint: a grid whose event boundaries are
drawn uniformly over the vocabulary is mostly made of windows whose boundary never occurs, and a grid
whose boundaries are prevalence-weighted is mostly made of windows that genuinely close - a model can
be scored on both, and the two are directly comparable because they share every context.

Output layout (``{out_dir}/eval`` is directly usable as a ``task_labels_dir``)::

    {out_dir}/eval/{split}/{shard}.parquet          MultitaskBoundarySchema metadata rows
    {out_dir}/eval/{split}/{shard}.labels.npy       uint8 (rows, K, ceil(V/8)) packed targets
    {out_dir}/eval/{split}/_multitask_manifest.json vocabulary + semantics the bits were built under
    {out_dir}/eval_meta/{split}/{shard}.parquet     row-aligned sidecar: task_id, target_code, and
                                                    per-window resolution diagnostics
    {out_dir}/eval_tasks.parquet                    the N task specifications themselves

The sidecar lives *outside* the labels root on purpose: a dataset points at ``{out_dir}/eval`` and
reads every parquet under it, so a non-conforming file inside that tree would be a hard error.  It is
row-aligned with ``{shard}.parquet`` - that alignment is the contract, exactly as it is for the
``.labels.npy`` sidecar - which is what lets ``EQ_predict_multitask`` recover which task (and which
scored code) each model output belongs to.

The diagnostics matter as much as the labels here.  ``start_resolved`` / ``end_resolved`` record
whether each window's boundary event *actually occurred*, which no stored label bit reveals: a
window whose end event never occurs runs to the end of the timeline, and one whose start event never
occurs is empty and every target is false.  Metrics stratified on those two flags separate "the model
learned the window semantics" from "the model learned that most windows are degenerate".
"""

from __future__ import annotations

import os

# Pin polars to a single thread BEFORE importing polars, mirroring the sibling samplers: workers
# inherit this env, and process-level fan-out already saturates cores.
os.environ.setdefault("POLARS_MAX_THREADS", "1")

import hashlib
import json
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.resources import files
from pathlib import Path  # noqa: TC003 - runtime use in worker signatures

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig  # noqa: TC002 - Hydra resolves this at runtime

from every_query.generate_tasks.interval_table import INF
from every_query.generate_tasks.sample_evaluation_tasks import (
    sample_prediction_times_per_subject,
    subsample_subject_ids,
)
from every_query.generate_tasks.sample_multitask_sequences import (
    BOUND_EVENTS_COL,
    CONDITION_ANSWERS_COL,
    CONDITION_CODES_COL,
    CTX_ID_COL,
    DURATIONS_COL,
    LABELS_SUFFIX,
    METADATA_COLUMNS,
    PT,
    SID,
    START_DURATIONS_COL,
    START_EVENTS_COL,
    BoundaryDistribution,
    BoundarySample,
    LabelStats,
    TargetVocabulary,
    _bounds_to_columns,
    build_code_weights,
    build_manifest,
    build_target_vocabulary,
    effective_support,
    label_multitask_index,
    normalize_index,
    read_boundary_codes,
    read_exclude_prefixes,
    read_start_event_codes,
    reject_ontology,
    sort_index_for_labeling,
    write_labeled_shard,
    write_manifest,
)
from every_query.generate_tasks.sample_tasks import (
    _atomic_write_parquet,
    _read_event_shard,
    _require_path_arg,
    _split_shards,
    _unique_tmp_path,
    resolve_workers,
)
from every_query.utils.seeds import derive_seed

logger = logging.getLogger(__name__)

TASK_ID_COL = "task_id"
GROUP_COL = "task_group"
TARGET_CODE_COL = "target_code"
EVAL_TASKS_NAME = "eval_tasks.parquet"
EVAL_DIRNAME = "eval"
EVAL_META_DIRNAME = "eval_meta"

# Sidecar diagnostic columns, one list of K entries per row.
START_RESOLVED_COL = "start_resolved"
END_RESOLVED_COL = "end_resolved"
WINDOW_DAYS_COL = "window_days"

TASK_SPEC_COLUMNS = [
    TASK_ID_COL,
    GROUP_COL,
    START_DURATIONS_COL,
    START_EVENTS_COL,
    DURATIONS_COL,
    BOUND_EVENTS_COL,
    CONDITION_CODES_COL,
    TARGET_CODE_COL,
]

US_PER_DAY = 86_400_000_000.0


# ---------------------------------------------------------------------------
# Stage E1 - the N task specifications
# ---------------------------------------------------------------------------


def _group_settings(group: object, index: int) -> dict:
    """Normalize one ``task_groups`` entry to ``{name, code_weighting, code_weight_power}``.

    A bare string is shorthand for a group of that name with no weighting (``"uniform"``) or with
    prevalence weighting at the config's default power (``"prevalence"``).

    Examples:
        >>> _group_settings("uniform", 0)
        {'name': 'uniform', 'code_weighting': None, 'code_weight_power': None}
        >>> _group_settings({"name": "p", "code_weighting": "prevalence", "code_weight_power": 0.5}, 1)
        {'name': 'p', 'code_weighting': 'prevalence', 'code_weight_power': 0.5}
        >>> _group_settings({"code_weighting": "prevalence"}, 2)
        {'name': 'group2', 'code_weighting': 'prevalence', 'code_weight_power': None}
    """
    if isinstance(group, str):
        weighting = None if group.lower() in ("uniform", "none", "null") else group
        return {"name": group, "code_weighting": weighting, "code_weight_power": None}
    g = dict(group)
    name = str(g.get("name", f"group{index}"))
    weighting = g.get("code_weighting")
    power = g.get("code_weight_power")
    return {
        "name": name,
        "code_weighting": None if weighting is None else str(weighting),
        "code_weight_power": None if power is None else float(power),
    }


def _weights_for(
    settings: dict,
    codes: list[str],
    source: object,
    column: str,
    default_power: float,
) -> tuple[float, ...]:
    """The sampling weights one task group draws its boundary / start codes with."""
    weighting = settings["code_weighting"]
    if weighting is None or str(weighting).lower() in ("", "uniform", "none", "null"):
        return ()
    if str(weighting).lower() != "prevalence":
        raise ValueError(f"code_weighting must be null or 'prevalence' (got {weighting!r})")
    power = default_power if settings["code_weight_power"] is None else settings["code_weight_power"]
    return build_code_weights(source, codes, column, power)


def draw_target_codes(
    rng: np.random.Generator, codes: list[str], weights: tuple[float, ...], n: int
) -> list[str]:
    """``n`` scored target codes, iid with replacement, uniform when ``weights`` is empty.

    Examples:
        >>> draw_target_codes(np.random.default_rng(0), ["A", "B"], (1.0, 0.0), 3)
        ['A', 'A', 'A']
        >>> len(draw_target_codes(np.random.default_rng(0), ["A", "B"], (), 5))
        5
    """
    pool = np.asarray(codes, dtype=object)
    if not weights:
        return list(pool[rng.integers(0, len(pool), size=n)])
    return list(pool[rng.choice(len(pool), size=n, p=np.asarray(weights, dtype=np.float64))])


def build_task_table(
    cfg: DictConfig,
    vocab: TargetVocabulary,
    split: str,
) -> pl.DataFrame:
    """Stage E1: draw every task group's specifications once, into one table.

    Each group draws its ``num_evaluation_tasks`` window sequences from a
    :class:`~every_query.generate_tasks.sample_multitask_sequences.BoundaryDistribution` built with
    that group's boundary-code weighting and the config's shared window knobs, so the only thing
    that differs between groups is *which codes bound the windows*.  Every group is seeded on
    ``(seed, "eval_multitask_specs", split, group_name)``, so adding, removing or reordering a group
    never perturbs another group's draw.

    Target codes are drawn from their own stream and their own (usually damped) weighting: they
    decide whether a task's labels are informative at all, and are deliberately not part of what the
    groups vary.
    """
    exclude = read_exclude_prefixes(cfg.get("exclude_boundary_prefixes"))
    boundary_codes = read_boundary_codes(cfg.get("boundary_codes"), vocab, exclude)
    start_event_codes = read_start_event_codes(cfg.get("start_event_codes"), vocab, exclude)
    condition_codes = vocab.boundary_candidates()
    source = cfg.get("query_codes")
    column = str(cfg.get("code_weight_column", "code/n_occurrences"))
    default_power = float(cfg.get("code_weight_power", 1.0))

    target_codes_pool = read_boundary_codes(cfg.get("target_codes"), vocab, exclude)
    target_weights = _weights_for(
        {
            "code_weighting": cfg.get("target_code_weighting"),
            "code_weight_power": cfg.get("target_code_weight_power"),
        },
        target_codes_pool,
        source,
        column,
        default_power,
    )

    n_per_group = int(cfg.num_evaluation_tasks)
    if n_per_group < 1:
        raise ValueError(f"num_evaluation_tasks must be >= 1 (got {n_per_group})")
    groups = list(cfg.get("task_groups") or ["uniform"])
    seed = int(cfg.seed)

    frames: list[pl.DataFrame] = []
    for i, raw in enumerate(groups):
        settings = _group_settings(raw, i)
        name = settings["name"]
        bw = _weights_for(settings, boundary_codes, source, column, default_power)
        sw = _weights_for(settings, start_event_codes, source, column, default_power)
        dist = BoundaryDistribution.from_config(
            cfg, boundary_codes, condition_codes, start_event_codes, bw, sw
        )
        rngs = [
            np.random.default_rng(derive_seed(seed, "eval_multitask_specs", split, name, axis))
            for axis in (
                "bound_forms",
                "bound_durations",
                "bound_codes",
                "condition_codes",
                "start_forms",
                "start_durations",
                "start_codes",
            )
        ]
        sample: BoundarySample = dist.sample(n_per_group, *rngs)
        target_rng = np.random.default_rng(derive_seed(seed, "eval_multitask_targets", split, name))
        frame = pl.DataFrame(
            {
                GROUP_COL: [name] * n_per_group,
                TARGET_CODE_COL: draw_target_codes(
                    target_rng, target_codes_pool, target_weights, n_per_group
                ),
                **_bounds_to_columns(sample),
            }
        )
        logger.info(
            "Stage E1: group %r drew %d task(s) (boundary support %.0f of %d, start support %.0f of %d).",
            name,
            n_per_group,
            effective_support(bw, len(boundary_codes)),
            len(boundary_codes),
            effective_support(sw, len(start_event_codes)),
            len(start_event_codes),
        )
        frames.append(frame)

    tasks = pl.concat(frames, how="vertical").with_row_index(TASK_ID_COL)
    return tasks.with_columns(pl.col(TASK_ID_COL).cast(pl.Int64)).select(TASK_SPEC_COLUMNS)


def tasks_fingerprint(tasks: pl.DataFrame, vocab: TargetVocabulary, num_bounds: int) -> str:
    """Digest of the task table + vocabulary: what fixes the meaning of every stored label bit."""
    h = hashlib.sha256()
    h.update(f"eval-multitask-v1:{num_bounds}:{vocab.fingerprint}\n".encode())
    h.update(tasks.sort(TASK_ID_COL).write_json().encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Stage E2 - the dense (context x task) index
# ---------------------------------------------------------------------------


def build_dense_multitask_index(
    contexts: pl.DataFrame, tasks: pl.DataFrame, num_bounds: int | None = None
) -> pl.DataFrame:
    """Cross-join every context with every task, sorted the way the labeling kernel wants it.

    ``_ctx_id = context_row * n_tasks + task_id`` makes the id context-major and unique, so the
    kernel's ``(subject_id, prediction_time, _ctx_id)`` sort is deterministic and reproducible here -
    which is what lets the sidecar be built from an identically sorted copy rather than guessed at.

    Examples:
        >>> from datetime import datetime
        >>> ctx = pl.DataFrame({"subject_id": [1, 2],
        ...     "prediction_time": [datetime(2024, 1, 1), datetime(2024, 2, 1)]},
        ...     schema={"subject_id": pl.Int64, "prediction_time": pl.Datetime("us")})
        >>> tasks = pl.DataFrame({"task_id": [0, 1], "task_group": ["u", "u"],
        ...     "start_durations": [[0.0], [0.0]], "start_events": [[None], [None]],
        ...     "durations": [[7.0], [30.0]], "bound_events": [[None], [None]],
        ...     "condition_codes": [[], []], "target_code": ["A", "B"]})
        >>> idx = build_dense_multitask_index(ctx, tasks)
        >>> idx.height, idx["_ctx_id"].to_list()
        (4, [0, 1, 2, 3])
        >>> idx.select("subject_id", "task_id", "target_code").rows()
        [(1, 0, 'A'), (1, 1, 'B'), (2, 0, 'A'), (2, 1, 'B')]

        An empty cohort yields an empty frame rather than raising:

        >>> build_dense_multitask_index(ctx.head(0), tasks).height
        0
    """
    if tasks.height == 0:
        raise ValueError("the task table is empty")
    if num_bounds is None:
        num_bounds = int(tasks[DURATIONS_COL].list.len().max())
    n_tasks = tasks.height
    joined = (
        contexts.select(SID, PT)
        .with_row_index("_row")
        .with_columns(pl.col("_row").cast(pl.Int64))
        .join(tasks, how="cross")
        .with_columns((pl.col("_row") * n_tasks + pl.col(TASK_ID_COL)).alias(CTX_ID_COL))
        .drop("_row")
    )
    return sort_index_for_labeling(normalize_index(joined, num_bounds))


# ---------------------------------------------------------------------------
# Stage E3 - per-shard labeling
# ---------------------------------------------------------------------------


def window_diagnostics(
    start_times: np.ndarray, end_times: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(start_resolved, end_resolved, window_days)`` from the kernel's resolved ``(N, K)`` matrices.

    ``INF`` marks a boundary event that never occurred: an unresolved start leaves the window empty,
    an unresolved end lets it run to the end of the timeline.  ``window_days`` is finite only when
    both ends are.

    Examples:
        >>> s = np.array([[0, INF]], dtype=np.int64)
        >>> e = np.array([[86_400_000_000, INF]], dtype=np.int64)
        >>> sr, er, wd = window_diagnostics(s, e)
        >>> sr.tolist(), er.tolist(), wd.tolist()
        ([[True, False]], [[True, False]], [[1.0, nan]])
    """
    start_resolved = start_times != INF
    end_resolved = end_times != INF
    both = start_resolved & end_resolved
    window_days = np.full(start_times.shape, np.nan, dtype=np.float64)
    window_days[both] = (end_times[both] - start_times[both]) / US_PER_DAY
    return start_resolved, end_resolved, window_days


def _list_of(name: str, values: np.ndarray, dtype: pl.DataType) -> pl.Series:
    n, m = values.shape
    return pl.Series(name, values.ravel(), dtype=dtype).reshape((n, m)).arr.to_list()


def build_sidecar(
    sorted_index: pl.DataFrame,
    start_times: np.ndarray,
    end_times: np.ndarray,
) -> pl.DataFrame:
    """The row-aligned evaluation sidecar: which task each labeled row is, and how its windows resolved."""
    start_resolved, end_resolved, window_days = window_diagnostics(start_times, end_times)
    return sorted_index.select(SID, PT, TASK_ID_COL, GROUP_COL, TARGET_CODE_COL).with_columns(
        _list_of(START_RESOLVED_COL, start_resolved, pl.Boolean),
        _list_of(END_RESOLVED_COL, end_resolved, pl.Boolean),
        _list_of(WINDOW_DAYS_COL, window_days.astype(np.float32), pl.Float32),
    )


def label_one_eval_shard(
    shard: str,
    data_dir: Path,
    out_dir: Path,
    meta_dir: Path,
    split: str,
    tasks_fp: Path,
    codes_source: str,
    manifest: dict,
    prediction_times_per_subject: int,
    min_context_per_subject: int,
    subject_subsample_fraction: float | None,
    seed: int,
    overwrite: bool,
    chunk_rows: int,
) -> tuple[str, str, dict]:
    """Worker: sample one shard's cohort, label every task at every context, write the three files.

    Module-level so it pickles under ``spawn``.  Returns ``(shard, status, stats)`` with status
    ``"skipped"`` or ``"labeled"``.
    """
    final_parquet = out_dir / f"{shard}.parquet"
    final_labels = out_dir / f"{shard}{LABELS_SUFFIX}"
    final_meta = meta_dir / f"{shard}.parquet"
    if not overwrite and final_parquet.exists() and final_labels.exists() and final_meta.exists():
        return shard, "skipped", {}

    vocab = build_target_vocabulary(codes_source)
    num_bounds = int(manifest["num_bounds"])
    tasks = pl.read_parquet(tasks_fp)

    events_df = _read_event_shard(data_dir / "data" / split / f"{shard}.parquet")
    if subject_subsample_fraction is not None:
        events_df = subsample_subject_ids(
            events_df, subject_subsample_fraction, derive_seed(seed, "subject_subsample", split, shard)
        )
    contexts = sample_prediction_times_per_subject(
        events_df=events_df,
        k=prediction_times_per_subject,
        min_context_per_subject=min_context_per_subject,
        seed=derive_seed(seed, "prediction_times", split, shard),
    )

    index_df = build_dense_multitask_index(contexts, tasks, num_bounds)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    shape = (index_df.height, num_bounds, vocab.packed_width)
    labels_tmp = _unique_tmp_path(final_labels)
    window_times: dict[str, np.ndarray] = {}
    try:
        if index_df.height == 0:
            # A shard whose subjects all fall below `min_context_per_subject` still writes a
            # complete, well-formed empty pair, so a reader sees every shard of the split.
            metadata = (
                index_df.select(METADATA_COLUMNS)
                .with_columns(pl.Series(CONDITION_ANSWERS_COL, [], dtype=pl.List(pl.Boolean)))
            )
            with open(labels_tmp, "wb") as f:
                np.save(f, np.zeros(shape, dtype=np.uint8))
            stats = LabelStats(vocab_size=vocab.size, packed_width=vocab.packed_width, num_bounds=num_bounds)
            window_times = {
                "start_times": np.zeros((0, num_bounds), dtype=np.int64),
                "end_times": np.zeros((0, num_bounds), dtype=np.int64),
            }
        else:
            mm = np.lib.format.open_memmap(labels_tmp, mode="w+", dtype=np.uint8, shape=shape)
            try:
                metadata, _, stats = label_multitask_index(
                    index_df.drop(TASK_ID_COL, GROUP_COL, TARGET_CODE_COL),
                    events_df,
                    vocab,
                    num_bounds,
                    chunk_rows=chunk_rows,
                    out=mm,
                    window_times_out=window_times,
                )
                mm.flush()
            finally:
                del mm
        write_labeled_shard(metadata, out_dir, shard, labels_tmp=labels_tmp)
    except Exception:
        labels_tmp.unlink(missing_ok=True)
        raise

    _atomic_write_parquet(
        build_sidecar(index_df, window_times["start_times"], window_times["end_times"]), final_meta
    )
    return shard, "labeled", stats.as_dict()


def _log_shard(shard: str, stats: dict) -> None:
    if not stats:
        return
    logger.info(
        "Stage E3 shard %s: contexts=%s events=%s event_starts=%s unresolved_frac=%.3f event_bounds=%s "
        "inf_frac=%.3f empty_windows_frac=%.3f mean_pos/window=%.2f",
        shard,
        f"{stats['n_contexts']:,}",
        f"{stats['n_events']:,}",
        f"{stats['n_event_starts']:,}",
        stats["frac_event_starts_unresolved"],
        f"{stats['n_event_bounds']:,}",
        stats["frac_event_bounds_inf"],
        stats["frac_empty_windows"],
        stats["mean_positives_per_window"],
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(cfg: DictConfig) -> None:
    """Execute Stages E1-E3 for a fully-resolved config (no Hydra side effects)."""
    reject_ontology(cfg.get("ontology_dir"))
    data_dir = _require_path_arg(cfg.get("data_dir"), "data_dir")
    out_root = _require_path_arg(cfg.get("out_dir"), "out_dir")
    split = str(cfg.split)
    seed = int(cfg.seed)

    ssf_raw = cfg.get("subject_subsample_fraction")
    if isinstance(ssf_raw, bool) or (ssf_raw is not None and not isinstance(ssf_raw, int | float)):
        raise TypeError(
            f"subject_subsample_fraction must be a number in (0, 1] or null, got {ssf_raw!r}"
        )
    subject_subsample_fraction = None if ssf_raw is None else float(ssf_raw)

    vocab = build_target_vocabulary(cfg.get("query_codes"))
    tasks = build_task_table(cfg, vocab, split)
    num_bounds = int(cfg.get("num_bounds", 5))
    logger.info(
        "Stage E1: %d task(s) across %d group(s); V=%s, K=%d.",
        tasks.height,
        tasks[GROUP_COL].n_unique(),
        f"{vocab.size:,}",
        num_bounds,
    )

    eval_dir = out_root / EVAL_DIRNAME / split
    meta_dir = out_root / EVAL_META_DIRNAME / split
    eval_dir.mkdir(parents=True, exist_ok=True)
    tasks_fp = out_root / EVAL_TASKS_NAME
    _atomic_write_parquet(tasks, tasks_fp)

    # One representative distribution supplies the window semantics; the fingerprint that actually
    # gates reuse is over the task table, because that - not any single group's policy - is what
    # decides the rows.
    dist = BoundaryDistribution.from_config(
        cfg, vocab.boundary_candidates(), vocab.boundary_candidates(), vocab.boundary_candidates()
    )
    manifest = build_manifest(dist, vocab)
    manifest["config_fingerprint"] = tasks_fingerprint(tasks, vocab, num_bounds)
    manifest["num_bounds"] = num_bounds
    manifest["num_condition_codes"] = num_bounds - 1
    manifest["eval_grid"] = True
    manifest["n_eval_tasks"] = tasks.height
    manifest["eval_task_groups"] = sorted(tasks[GROUP_COL].unique().to_list())
    manifest = write_manifest(eval_dir, manifest)

    shards = _split_shards(data_dir, split)
    n_workers = resolve_workers(cfg.get("max_workers"))
    chunk_rows = int(cfg.get("label_chunk_rows", 2000))
    logger.info(
        "Stage E3: labeling %d shard(s) across %d worker(s), chunk_rows=%d.",
        len(shards),
        n_workers,
        chunk_rows,
    )

    args = {
        "data_dir": data_dir,
        "out_dir": eval_dir,
        "meta_dir": meta_dir,
        "split": split,
        "tasks_fp": tasks_fp,
        "codes_source": str(cfg.query_codes),
        "manifest": manifest,
        "prediction_times_per_subject": int(cfg.prediction_times_per_subject),
        "min_context_per_subject": int(cfg.min_context_per_subject),
        "subject_subsample_fraction": subject_subsample_fraction,
        "seed": seed,
        "overwrite": bool(cfg.get("overwrite", False)),
        "chunk_rows": chunk_rows,
    }
    n_rows = 0
    if n_workers <= 1:
        for shard in shards:
            name, _status, stats = label_one_eval_shard(shard, **args)
            _log_shard(name, stats)
            n_rows += int(stats.get("n_contexts", 0))
    else:
        # Spawn, not the platform default: a forked child inherits polars' thread state and
        # deadlocks before it runs a line of this module, exactly as the training driver's pool does.
        mp_context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context) as pool:
            futures = {pool.submit(label_one_eval_shard, shard, **args): shard for shard in shards}
            for fut in as_completed(futures):
                name, _status, stats = fut.result()
                _log_shard(name, stats)
                n_rows += int(stats.get("n_contexts", 0))

    logger.info(
        "Evaluation grid complete: %s labeled row(s) = contexts x %d task(s) x %d window(s) in %s.",
        f"{n_rows:,}",
        tasks.height,
        num_bounds,
        eval_dir,
    )
    (out_root / "_eval_summary.json").write_text(
        json.dumps(
            {
                "split": split,
                "n_tasks": tasks.height,
                "num_bounds": num_bounds,
                "n_labeled_rows": n_rows,
                "vocab_size": vocab.size,
                "task_groups": manifest["eval_task_groups"],
                "config_fingerprint": manifest["config_fingerprint"],
            },
            indent=2,
        )
    )


CONFIGS = str(files("every_query") / "generate_tasks" / "configs")


@hydra.main(
    version_base=None,
    config_path=CONFIGS,
    config_name="sample_evaluation_multitask_sequences_config",
)
def main(cfg: DictConfig) -> None:
    """Hydra entry point (``EQ_generate_evaluation_multitask_sequences``)."""
    run(cfg)


if __name__ == "__main__":
    main()
