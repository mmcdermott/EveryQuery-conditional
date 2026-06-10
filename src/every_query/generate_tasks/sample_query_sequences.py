"""Sampling-first query-*sequence* label generator for conditional pre-training.

Sibling of :mod:`~every_query.generate_tasks.sample_tasks`, but instead of one scattered
``(code, duration)`` per context this produces, per sampled patient context, an ordered
*sequence* of queries:

    ``[Q1=(__CENSOR__, d1)] [Q2=(c2, d2)] ... [QL=(cL, dL)]``  with ``L <= max_queries``

- **Q1 is always the censor query**: "will any data be present after ``prediction_time + d1``?"
  Its answer is always observed, so the model's downstream conditional predictions are never
  conditioned on a missing first answer — this replaces the separate censor head of the
  single-query model.
- **Q2..QL are iid draws** from (uniform codes × log-uniform durations), in an arbitrary
  (random) order.  The point is not temporal structure but conditional-answering capability:
  the model trained on these sequences learns ``P(A_j | patient, Q_1..Q_j, A_1..A_{j-1})``.
- **Answers** are nullable booleans: ``null`` = censored (window extends past the subject's
  last recorded event).  Censored answers contribute no training loss but are visible to later
  queries via a dedicated "unknown" answer-token class.

Output rows follow :class:`~every_query.data.schema.QuerySeqSchema`.

Worker decomposition, seeding, and atomic-write behavior mirror ``sample_tasks``:
``(input_shard, task_shard, seed)`` parameterize one pure-function worker writing one parquet.
"""

import logging
import os
from importlib.resources import files
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig

from meds import DataSchema

from every_query.data.schema import QuerySeqSchema, TaskQuerySchema
from every_query.data.seq_dataset import EOS_CODE
from every_query.generate_tasks.sample_tasks import (
    _atomic_write_parquet,
    _read_event_shard,
    _resolve_path,
    read_query_codes,
    sample_contexts,
)
from every_query.utils.seeds import derive_seed

logger = logging.getLogger(__name__)

CTX_ID_COL = "_ctx_id"
POSITION_COL = "_position"


def sample_log_uniform_durations(
    n: int, duration_low: int, duration_high: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw ``n`` integer-valued day durations from ``round(exp(U(log low, log high)))``.

    Mirrors the duration distribution of ``sample_tasks.sample_tasks`` (log-uniform favors
    shorter horizons), returned as ``float32`` for ``QuerySeqSchema.durations``.

    Examples:
        >>> rng = np.random.default_rng(0)
        >>> d = sample_log_uniform_durations(1000, 1, 365, rng)
        >>> d.dtype
        dtype('float32')
        >>> bool((d >= 1).all() and (d <= 365).all())
        True
        >>> bool(np.median(d) < (1 + 365) / 2)  # log-uniform skews low
        True
    """
    raw = np.exp(rng.uniform(np.log(duration_low), np.log(duration_high), size=n))
    return np.clip(np.round(raw), duration_low, duration_high).astype(np.float32)


def build_sequence_index_df(
    contexts: pl.DataFrame,
    query_codes: list[str],
    min_queries: int,
    max_queries: int,
    duration_low: int,
    duration_high: int,
    seed: int,
    eos_first_fraction: float = 0.0,
    duration_mode: str = "random",
) -> pl.DataFrame:
    """Expand each sampled context into a flat, fully-random per-query index frame.

    For context ``i`` draws ``L_i ~ Uniform{min_queries..max_queries}`` queries, each an
    independent (code, duration) pair: code uniform over ``query_codes`` (which includes the
    end-of-timeline code ``TIMELINE//END`` like any other), duration log-uniform.  There is **no**
    privileged censor position — the end-of-timeline query is just one possible code.  Two
    optional, default-off reweightings support a later sweep (see issue scope):

      - ``eos_first_fraction``: probability that a sequence's position 0 is forced to be the
        ``TIMELINE//END`` query (upweights the censoring-control pattern).  ``0.0`` = pure random.
      - ``duration_mode``: ``"random"`` (independent durations, default); ``"same"`` (all queries
        in a sequence share one duration); ``"nondecreasing"`` (durations sorted ascending) — the
        latter two reflect that many censoring-style asks reuse one horizon.

    Returns:
        DataFrame ``(_ctx_id, _position, subject_id, prediction_time, query, duration_days)``.

    Examples:
        >>> from datetime import datetime
        >>> ctx = pl.DataFrame({
        ...     "subject_id": [1, 2],
        ...     "prediction_time": [datetime(2024, 1, 1), datetime(2024, 2, 1)],
        ... })
        >>> idx = build_sequence_index_df(ctx, ["A", "B", "TIMELINE//END"], 1, 4, 1, 365, seed=0)
        >>> idx.group_by("_ctx_id").len()["len"].max() <= 4
        True
        >>> bool(idx["query"].is_in(["A", "B", "TIMELINE//END"]).all())
        True

        Determinism:

        >>> build_sequence_index_df(ctx, ["A", "B"], 1, 3, 1, 365, 7).equals(
        ...     build_sequence_index_df(ctx, ["A", "B"], 1, 3, 1, 365, 7))
        True

        ``eos_first_fraction=1.0`` forces position 0 to the end-of-timeline query:

        >>> idx = build_sequence_index_df(
        ...     ctx, ["A", "B", "TIMELINE//END"], 2, 2, 1, 365, 0, eos_first_fraction=1.0)
        >>> idx.filter(pl.col("_position") == 0)["query"].unique().to_list()
        ['TIMELINE//END']
    """
    if min_queries < 1:
        raise ValueError(f"min_queries must be >= 1 (got {min_queries})")
    if max_queries < min_queries:
        raise ValueError(f"max_queries ({max_queries}) must be >= min_queries ({min_queries})")
    if not query_codes:
        raise ValueError("query_codes must be non-empty")
    if duration_mode not in ("random", "same", "nondecreasing"):
        raise ValueError(f"duration_mode must be random|same|nondecreasing (got {duration_mode!r})")

    rng = np.random.default_rng(seed)
    n_ctx = contexts.height
    if n_ctx == 0:
        return contexts.head(0).select(
            pl.lit(0, dtype=pl.UInt32).alias(CTX_ID_COL),
            pl.lit(0, dtype=pl.Int64).alias(POSITION_COL),
            TaskQuerySchema.subject_id_name,
            TaskQuerySchema.prediction_time_name,
            pl.lit("", dtype=pl.Utf8).alias(TaskQuerySchema.query_name),
            pl.lit(0.0, dtype=pl.Float32).alias(TaskQuerySchema.duration_days_name),
        )

    seq_lens = rng.integers(min_queries, max_queries + 1, size=n_ctx)
    total = int(seq_lens.sum())
    ctx_ids = np.repeat(np.arange(n_ctx, dtype=np.int64), seq_lens)
    positions = np.concatenate([np.arange(n) for n in seq_lens])

    codes = np.array(query_codes, dtype=object)[rng.integers(0, len(query_codes), size=total)]
    if eos_first_fraction > 0:
        force = rng.random(n_ctx) < eos_first_fraction
        # position-0 row index of each context within the flattened arrays
        starts = np.concatenate([[0], np.cumsum(seq_lens)[:-1]])
        codes[starts[force]] = EOS_CODE

    durations = sample_log_uniform_durations(total, duration_low, duration_high, rng)
    if duration_mode != "random":
        offsets = np.concatenate([[0], np.cumsum(seq_lens)])
        for i in range(n_ctx):
            sl = slice(offsets[i], offsets[i + 1])
            if duration_mode == "same":
                durations[sl] = durations[offsets[i]]
            elif duration_mode == "nondecreasing":
                durations[sl] = np.sort(durations[sl])

    expanded = contexts.with_row_index(CTX_ID_COL).join(
        pl.DataFrame(
            {
                CTX_ID_COL: pl.Series(ctx_ids).cast(pl.UInt32),
                POSITION_COL: pl.Series(positions).cast(pl.Int64),
                TaskQuerySchema.query_name: pl.Series(codes.tolist(), dtype=pl.Utf8),
                TaskQuerySchema.duration_days_name: pl.Series(durations, dtype=pl.Float32),
            }
        ),
        on=CTX_ID_COL,
        how="inner",
    )
    return expanded.select(
        CTX_ID_COL,
        POSITION_COL,
        TaskQuerySchema.subject_id_name,
        TaskQuerySchema.prediction_time_name,
        TaskQuerySchema.query_name,
        TaskQuerySchema.duration_days_name,
    ).sort(CTX_ID_COL, POSITION_COL)


def label_binary_occurrence(index_df: pl.DataFrame, events_df: pl.DataFrame) -> pl.DataFrame:
    """Label every query with a binary *observed-occurrence* answer and reassemble into list rows.

    ``answer = True`` iff an event whose code equals the query occurs strictly within
    ``(prediction_time, prediction_time + duration_days)`` **and is present in the record**.  There
    is no censoring/null: an event we cannot observe (because the record ends first) is ``False``.
    Censoring is captured separately by the ``TIMELINE//END`` query — which, being the last event
    of each record, answers ``True`` exactly when the record ends within the window.

    The strict ``>`` lower bound is enforced by shifting the asof key ``+1µs`` (microsecond-precision
    datetimes), mirroring ``sample_tasks.evaluate_index_df``.

    Returns:
        DataFrame ``(subject_id, prediction_time, queries, durations, answers)``, one row per
        ``_ctx_id``, lists ordered by ``_position``.

    Examples:
        >>> from datetime import datetime
        >>> events = pl.DataFrame({
        ...     "subject_id": [1, 1, 1],
        ...     "time": [datetime(2024, 1, 1), datetime(2024, 1, 5), datetime(2024, 1, 20)],
        ...     "code": ["A", "B", "TIMELINE//END"],
        ... }).with_columns(pl.col("time").cast(pl.Datetime("us")))
        >>> idx = pl.DataFrame({
        ...     "_ctx_id": [0, 0, 0],
        ...     "_position": [0, 1, 2],
        ...     "subject_id": [1, 1, 1],
        ...     "prediction_time": [datetime(2024, 1, 2)] * 3,
        ...     "query": ["B", "A", "TIMELINE//END"],
        ...     "duration_days": [10.0, 10.0, 10.0],
        ... }).with_columns(
        ...     pl.col("prediction_time").cast(pl.Datetime("us")),
        ...     pl.col("duration_days").cast(pl.Float32),
        ...     pl.col("_ctx_id").cast(pl.UInt32))
        >>> row = label_binary_occurrence(idx, events).row(0, named=True)
        >>> row["answers"]  # B in (2,12]=yes; A at day1 not >day2 =no; END at day20 not in window =no
        [True, False, False]
    """
    sid = TaskQuerySchema.subject_id_name
    pt = TaskQuerySchema.prediction_time_name
    q = TaskQuerySchema.query_name
    d = TaskQuerySchema.duration_days_name

    left = index_df.with_columns((pl.col(pt) + pl.duration(microseconds=1)).alias("_pts")).sort(
        sid, q, "_pts"
    )
    right = (
        events_df.rename({DataSchema.code_name: q})
        .select(sid, q, DataSchema.time_name)
        .sort(sid, q, DataSchema.time_name)
    )
    joined = left.join_asof(
        right, by=[sid, q], left_on="_pts", right_on=DataSchema.time_name, strategy="forward"
    )
    window_end = pl.col(pt) + pl.duration(days=pl.col(d))
    answer = (pl.col(DataSchema.time_name).is_not_null() & (pl.col(DataSchema.time_name) < window_end))

    flat = joined.with_columns(answer.alias("answer")).sort(CTX_ID_COL, POSITION_COL)
    return (
        flat.group_by(CTX_ID_COL, maintain_order=True)
        .agg(
            pl.col(sid).first(),
            pl.col(pt).first(),
            pl.col(q).alias("queries"),
            pl.col(d).alias("durations"),
            pl.col("answer").alias("answers"),
        )
        .drop(CTX_ID_COL)
    )


def run_worker(
    data_dir: Path,
    out_dir: Path,
    query_codes: list[str],
    split: str,
    input_shard: str,
    task_shard: int,
    seed: int,
    n_contexts: int,
    min_queries: int,
    max_queries: int,
    duration_min: int,
    duration_max: int,
    min_context_per_subject: int,
    eos_first_fraction: float = 0.0,
    duration_mode: str = "random",
    overwrite: bool = False,
) -> Path | None:
    """Run the sequence-sampling pipeline for one ``(input_shard, task_shard)`` worker."""
    labels_fp = out_dir / split / f"{input_shard}__{task_shard:04d}.parquet"
    if labels_fp.exists() and not overwrite:
        logger.info("Labels already exist at %s, skipping.", labels_fp)
        return None

    shard_path = data_dir / "data" / split / f"{input_shard}.parquet"
    events_df = _read_event_shard(shard_path)
    logger.info("Loaded %d events from %s", events_df.height, shard_path)

    contexts_seed = derive_seed(seed, "seq_contexts", input_shard, task_shard)
    contexts = sample_contexts(
        events_df=events_df,
        n=n_contexts,
        min_context_per_subject=min_context_per_subject,
        seed=contexts_seed,
    )

    queries_seed = derive_seed(seed, "seq_queries", input_shard, task_shard)
    index_df = build_sequence_index_df(
        contexts=contexts,
        query_codes=query_codes,
        min_queries=min_queries,
        max_queries=max_queries,
        duration_low=duration_min,
        duration_high=duration_max,
        seed=queries_seed,
        eos_first_fraction=eos_first_fraction,
        duration_mode=duration_mode,
    )

    labeled = label_binary_occurrence(index_df, events_df)

    aligned = QuerySeqSchema.align(labeled.to_arrow())
    labels_fp.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_parquet(pl.from_arrow(aligned), labels_fp)
    logger.info("Wrote %d labeled query sequences to %s", labeled.height, labels_fp)
    return labels_fp


CONFIGS = str(files("every_query") / "generate_tasks" / "configs")


@hydra.main(version_base=None, config_path=CONFIGS, config_name="sample_query_sequences_config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point; path fallbacks mirror ``sample_tasks.main``."""
    from dotenv import load_dotenv

    load_dotenv()

    data_dir = _resolve_path(cfg.get("data_dir"), "INTERMEDIATE", "data_dir")
    out_dir = _resolve_path(cfg.get("out_dir"), "TASK_DIR", "out_dir")
    query_codes = read_query_codes(cfg.get("query_codes"))

    run_worker(
        data_dir=data_dir,
        out_dir=out_dir,
        query_codes=query_codes,
        split=str(cfg.split),
        input_shard=str(cfg.input_shard),
        task_shard=int(cfg.task_shard),
        seed=int(cfg.seed),
        n_contexts=int(cfg.n_contexts),
        min_queries=int(cfg.min_queries),
        max_queries=int(cfg.max_queries),
        duration_min=int(cfg.duration_min),
        duration_max=int(cfg.duration_max),
        min_context_per_subject=int(cfg.min_context_per_subject),
        eos_first_fraction=float(cfg.get("eos_first_fraction", 0.0)),
        duration_mode=str(cfg.get("duration_mode", "random")),
        overwrite=bool(cfg.get("overwrite", False)),
    )


if __name__ == "__main__":
    main()
