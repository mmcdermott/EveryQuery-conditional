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

from every_query.data.schema import QuerySeqSchema, TaskQuerySchema
from every_query.data.seq_dataset import CENSOR_QUERY_CODE
from every_query.generate_tasks.sample_tasks import (
    _atomic_write_parquet,
    _read_event_shard,
    _resolve_path,
    compute_max_time_per_subject,
    evaluate_index_df,
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
    min_extra_queries: int,
    max_extra_queries: int,
    duration_low: int,
    duration_high: int,
    seed: int,
) -> pl.DataFrame:
    """Expand each sampled context into a flat per-query index frame.

    For context ``i``, draws ``L_i ~ Uniform{min_extra_queries..max_extra_queries}`` code
    queries (codes uniform over ``query_codes``, durations log-uniform), prepends the censor
    query at position 0, and emits one row per query with ``(_ctx_id, _position)`` identity
    columns so the labeled rows can be reassembled into ordered lists.

    Returns:
        DataFrame with columns ``(_ctx_id, _position, subject_id, prediction_time, query,
        duration_days)``.  Position 0 of every context has ``query == CENSOR_QUERY_CODE``.

    Examples:
        >>> from datetime import datetime
        >>> ctx = pl.DataFrame({
        ...     "subject_id": [1, 2],
        ...     "prediction_time": [datetime(2024, 1, 1), datetime(2024, 2, 1)],
        ... })
        >>> idx = build_sequence_index_df(ctx, ["A", "B"], 1, 3, 1, 365, seed=0)
        >>> idx.group_by("_ctx_id").len().sort("_ctx_id")["len"].min() >= 2  # censor + >=1
        True
        >>> idx.filter(pl.col("_position") == 0)["query"].unique().to_list()
        ['__CENSOR__']
        >>> bool((idx.filter(pl.col("_position") > 0)["query"].is_in(["A", "B"])).all())
        True

        Determinism:

        >>> build_sequence_index_df(ctx, ["A", "B"], 1, 3, 1, 365, 7).equals(
        ...     build_sequence_index_df(ctx, ["A", "B"], 1, 3, 1, 365, 7))
        True
    """
    if min_extra_queries < 1:
        raise ValueError(f"min_extra_queries must be >= 1 (got {min_extra_queries})")
    if max_extra_queries < min_extra_queries:
        raise ValueError(
            f"max_extra_queries ({max_extra_queries}) must be >= min_extra_queries ({min_extra_queries})"
        )
    if not query_codes:
        raise ValueError("query_codes must be non-empty")

    rng = np.random.default_rng(seed)
    n_ctx = contexts.height

    n_extra = rng.integers(min_extra_queries, max_extra_queries + 1, size=n_ctx)
    seq_lens = n_extra + 1  # + censor query at position 0
    total = int(seq_lens.sum())

    ctx_ids = np.repeat(np.arange(n_ctx, dtype=np.int64), seq_lens)
    # Position within each sequence: 0..L_i
    positions = np.concatenate([np.arange(n) for n in seq_lens]) if n_ctx else np.array([], dtype=np.int64)

    is_censor = positions == 0
    codes = np.empty(total, dtype=object)
    codes[is_censor] = CENSOR_QUERY_CODE
    n_code_rows = int((~is_censor).sum())
    codes[~is_censor] = np.array(query_codes, dtype=object)[
        rng.integers(0, len(query_codes), size=n_code_rows)
    ]
    durations = sample_log_uniform_durations(total, duration_low, duration_high, rng)

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


def label_sequence_index_df(
    index_df: pl.DataFrame,
    events_df: pl.DataFrame,
    max_time_per_subject: pl.DataFrame,
) -> pl.DataFrame:
    """Label a flat sequence index and reassemble it into ``QuerySeqSchema``-shaped list rows.

    Code-query rows are labeled with the same single-pass ``join_asof`` as the single-query
    sampler (nullable ``boolean_value``; null = censored).  Censor-query rows (position 0) get
    ``answers[0] = (prediction_time + duration <= max_time)`` — i.e. *not* censored — which is
    always observed.

    Returns:
        DataFrame with columns ``(subject_id, prediction_time, queries, durations, answers)``,
        one row per ``_ctx_id``, lists ordered by ``_position``.
    """
    code_rows = index_df.filter(pl.col(POSITION_COL) > 0)
    censor_rows = index_df.filter(pl.col(POSITION_COL) == 0)

    labeled_codes = evaluate_index_df(
        code_rows, events_df, max_time_per_subject, id_cols=(CTX_ID_COL, POSITION_COL)
    )

    # Censor-query label: data present after prediction_time + duration.  `fill_null(False)`
    # resolves subjects missing from max_time_per_subject to "no data after horizon".
    duration_expr = pl.duration(days=pl.col(TaskQuerySchema.duration_days_name))
    window_end = pl.col(TaskQuerySchema.prediction_time_name) + duration_expr
    labeled_censor = (
        censor_rows.join(max_time_per_subject, on=TaskQuerySchema.subject_id_name, how="left")
        .with_columns(
            (window_end <= pl.col("max_time"))
            .fill_null(False)
            .alias(TaskQuerySchema.boolean_value_name)
        )
        .select(labeled_codes.columns)
    )

    flat = pl.concat([labeled_censor, labeled_codes], how="vertical").sort(CTX_ID_COL, POSITION_COL)

    return (
        flat.group_by(CTX_ID_COL, maintain_order=True)
        .agg(
            pl.col(TaskQuerySchema.subject_id_name).first(),
            pl.col(TaskQuerySchema.prediction_time_name).first(),
            pl.col(TaskQuerySchema.query_name).alias("queries"),
            pl.col(TaskQuerySchema.duration_days_name).alias("durations"),
            pl.col(TaskQuerySchema.boolean_value_name).alias("answers"),
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
    min_extra_queries: int,
    max_extra_queries: int,
    duration_min: int,
    duration_max: int,
    min_context_per_subject: int,
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
        min_extra_queries=min_extra_queries,
        max_extra_queries=max_extra_queries,
        duration_low=duration_min,
        duration_high=duration_max,
        seed=queries_seed,
    )

    max_time_df = compute_max_time_per_subject(events_df)
    labeled = label_sequence_index_df(index_df, events_df, max_time_df)

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
        min_extra_queries=int(cfg.min_extra_queries),
        max_extra_queries=int(cfg.max_extra_queries),
        duration_min=int(cfg.duration_min),
        duration_max=int(cfg.duration_max),
        min_context_per_subject=int(cfg.min_context_per_subject),
        overwrite=bool(cfg.get("overwrite", False)),
    )


if __name__ == "__main__":
    main()
