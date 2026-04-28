"""ETHOS predict stage: stream trajectories and compute occurs_prob (Phase 2).

For every (subject_id, prediction_time, query, duration_days, tier) row that has
a ground-truth label in the held-out eval task set, this stage computes::

    occurs_prob = (# of the 20 trajectory samples in which the mapped pattern
                   fires within `duration_days` of prediction_time) / 20

Patterns and their match-kinds come from ``mapping_table.parquet`` (Phase 1).
Outputs ``ethos_predictions.parquet`` and uploads it to
``gs://every-query-runs/baselines/ethos/``.
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from google.cloud import storage

from every_query.paper_experiments.ethos.build_mapping import (
    BUCKET,
    CACHE_DIR,
    EQ_AUCS_BLOB,
    ETHOS_TRAJ_PREFIX,
    MAPPING_TABLE_BLOB,
    _client,
    _download,
    _list,
    _upload,
)

EVAL_TASKS_PREFIX = "eval_tasks/7573f855c4b050a9d79d57fefd8a139c/held_out/"
PREDICTIONS_BLOB = "baselines/ethos/ethos_predictions.parquet"
DURATIONS = (30, 180)
N_SIMS = 20
MAX_DURATION_DAYS = max(DURATIONS)


def load_mapping() -> pl.DataFrame:
    p = _download(MAPPING_TABLE_BLOB, CACHE_DIR / "mapping_table.parquet")
    return pl.read_parquet(p)


def load_eq_buckets() -> pl.DataFrame:
    """``code -> bucket`` lookup from EQ AUC parquet (bucket is per-code)."""
    p = _download(EQ_AUCS_BLOB, CACHE_DIR / "eval_aucs_held_out.parquet")
    return (
        pl.read_parquet(p)
        .select("code", "bucket")
        .unique()
        .rename({"code": "query"})
    )


def load_eval_labels(queries: list[str]) -> pl.DataFrame:
    """Read eval-task parquets for the given queries at durations {30, 180}.

    Returns columns: ``subject_id, prediction_time, query, duration_days, boolean_value``.

    Label convention in this eval_tasks set follows the older EQ semantics
    (per ``every_query.generate_tasks.sample_tasks`` docstring, lines 25-27):
    ``boolean_value=True`` indicates the row is censored and ``occurs`` is the
    actual event indicator. We therefore project ``occurs`` into the
    ``boolean_value`` column for downstream AUROC computation, while keeping
    censored rows null so they get dropped at evaluation time -- matching how
    the working EIC baseline at ``baselines/eic/`` was computed against the
    same eval-task hash.
    """
    bucket = _client().bucket(BUCKET)
    parts: list[pl.DataFrame] = []
    for d in DURATIONS:
        prefix = f"{EVAL_TASKS_PREFIX}{d}/"
        for blob_name in _list(prefix):
            if not blob_name.endswith(".parquet"):
                continue
            data = bucket.blob(blob_name).download_as_bytes()
            df = pl.read_parquet(io.BytesIO(data))
            if df.is_empty() or df["query"][0] not in queries:
                continue
            # In this convention boolean_value=True flags a censored row; for
            # those rows ``occurs`` is unreliable (always False per the
            # generator). Surface them as null so evaluation drops them.
            label = (
                pl.when(pl.col("boolean_value")).then(None).otherwise(pl.col("occurs"))
            ).alias("boolean_value_truth")
            parts.append(
                df.with_columns(label)
                .select(
                    "subject_id",
                    "prediction_time",
                    "query",
                    "duration_days",
                    pl.col("boolean_value_truth").alias("boolean_value"),
                )
            )
    return pl.concat(parts, how="vertical") if parts else pl.DataFrame()


def _pattern_index(mapping: pl.DataFrame) -> pl.DataFrame:
    """Add a pattern_id column to the mapping table for joining later."""
    return mapping.with_row_index("pattern_id")


def _relevant_tokens(mapping: pl.DataFrame) -> set[str]:
    """All ETHOS tokens we need to keep when filtering trajectories.

    For literal patterns: just the pattern itself. For code+next_token: both
    halves (since both must be retained so the next-row check still finds a
    match).
    """
    out: set[str] = set()
    for row in mapping.iter_rows(named=True):
        if row["match_kind"] == "literal":
            out.add(row["token_pattern"])
        elif row["match_kind"] == "code+next_token":
            lab, qk = row["token_pattern"].split("|", 1)
            out.add(lab)
            out.add(qk)
    return out


def _hits_for_sim(traj: pl.DataFrame, mapping: pl.DataFrame, sim_idx: int) -> pl.DataFrame:
    """Return per-(subject, prediction_time, pattern_id) first-hit-days for one sim file.

    Memory plan: shift+filter inside a single ``with_columns`` so we never
    materialise both ``next_code`` and the unfiltered 30M rows at once.
    """
    keep_tokens = _relevant_tokens(mapping)
    df = (
        traj.lazy()
        .sort(["subject_id", "prediction_time", "time"])
        .with_columns(
            pl.col("code").shift(-1).over(["subject_id", "prediction_time"]).alias("next_code"),
            (
                (pl.col("time") - pl.col("prediction_time"))
                .dt.total_seconds()
                .truediv(86_400.0)
                .alias("delta_days")
            ),
        )
        .filter(
            (pl.col("delta_days") >= 0.0)
            & (pl.col("delta_days") <= float(MAX_DURATION_DAYS))
            & pl.col("code").is_in(list(keep_tokens))
        )
        .collect()
    )

    pieces: list[pl.DataFrame] = []
    for row in mapping.iter_rows(named=True):
        if row["match_kind"] == "literal":
            mask = pl.col("code") == row["token_pattern"]
        elif row["match_kind"] == "code+next_token":
            lab, qk = row["token_pattern"].split("|", 1)
            mask = (pl.col("code") == lab) & (pl.col("next_code") == qk)
        else:
            raise ValueError(f"unknown match_kind: {row['match_kind']}")
        hits = (
            df.filter(mask)
            .group_by(["subject_id", "prediction_time"])
            .agg(pl.col("delta_days").min().alias("first_hit_days"))
            .with_columns(
                pl.lit(row["pattern_id"], dtype=pl.UInt32).alias("pattern_id"),
                pl.lit(sim_idx, dtype=pl.UInt8).alias("sim_idx"),
            )
        )
        if not hits.is_empty():
            pieces.append(hits)

    if not pieces:
        return pl.DataFrame(
            schema={
                "subject_id": pl.Int64,
                "prediction_time": pl.Datetime("us"),
                "first_hit_days": pl.Float64,
                "pattern_id": pl.UInt32,
                "sim_idx": pl.UInt8,
            }
        )
    return pl.concat(pieces, how="vertical")


def stream_predictions(mapping: pl.DataFrame) -> pl.DataFrame:
    """Stream all 20 trajectory files; return per-sim per-pattern first-hit-days."""
    bucket = _client().bucket(BUCKET)
    flat_files = sorted(
        n
        for n in _list(ETHOS_TRAJ_PREFIX)
        if n.count("/") == ETHOS_TRAJ_PREFIX.count("/") and n.endswith(".parquet")
    )
    if len(flat_files) != N_SIMS:
        raise RuntimeError(f"expected {N_SIMS} flat trajectory files, found {len(flat_files)}")

    out: list[pl.DataFrame] = []
    t0 = time.time()
    for sim_idx, name in enumerate(flat_files):
        data = bucket.blob(name).download_as_bytes()
        traj = pl.read_parquet(
            io.BytesIO(data),
            columns=["subject_id", "time", "prediction_time", "code"],
        )
        hits = _hits_for_sim(traj, mapping, sim_idx)
        out.append(hits)
        print(
            f"  predict [{sim_idx+1}/{N_SIMS}] {name.split('/')[-1]} "
            f"-> {len(hits)} hit-rows, t={time.time()-t0:.1f}s"
        )
    return pl.concat(out, how="vertical")


def aggregate_predictions(
    sim_hits: pl.DataFrame,
    mapping: pl.DataFrame,
    eval_labels: pl.DataFrame,
    eq_buckets: pl.DataFrame,
) -> pl.DataFrame:
    """Roll per-sim hits into ``occurs_prob`` and join in ground-truth labels."""

    # pattern_id -> (query, tier)
    pat = mapping.select("pattern_id", "query", "tier")

    # per (sim, subj, pred_time, query, tier): min over patterns -> first_hit_days
    per_sim_qt = (
        sim_hits.join(pat, on="pattern_id", how="inner")
        .group_by(["sim_idx", "subject_id", "prediction_time", "query", "tier"])
        .agg(pl.col("first_hit_days").min())
    )

    # For each duration: did this sim hit?
    rows: list[pl.DataFrame] = []
    for d in DURATIONS:
        rows.append(
            per_sim_qt.with_columns(
                pl.lit(d, dtype=pl.Int32).alias("duration_days"),
                (pl.col("first_hit_days") <= float(d)).alias("hit"),
            ).select("sim_idx", "subject_id", "prediction_time", "query", "tier", "duration_days", "hit")
        )
    sim_hit_long = pl.concat(rows, how="vertical")

    # n_sims_hit per (subj, pred_time, query, tier, duration). Subjects without any
    # hit-row for this pattern across any sim contribute zero hits implicitly --
    # we re-introduce them via an outer join with the eval-label keys below.
    aggregated = (
        sim_hit_long.filter(pl.col("hit"))
        .group_by(["subject_id", "prediction_time", "query", "tier", "duration_days"])
        .agg(pl.col("sim_idx").n_unique().alias("n_sims_hit"))
    )

    # Cartesian frame of (eval label) x (tier) so unmapped predictions are 0/N_SIMS rather than missing
    tiers = mapping.select("tier").unique()
    base = eval_labels.join(tiers, how="cross")

    preds = (
        base.join(
            aggregated,
            on=["subject_id", "prediction_time", "query", "tier", "duration_days"],
            how="left",
        )
        .with_columns(pl.col("n_sims_hit").fill_null(0))
        .with_columns(
            pl.lit(N_SIMS, dtype=pl.UInt32).alias("n_sims"),
            (pl.col("n_sims_hit").cast(pl.Float64) / float(N_SIMS)).alias("occurs_prob"),
        )
        .join(eq_buckets, on="query", how="left")
        .select(
            "subject_id",
            "prediction_time",
            "query",
            "duration_days",
            "tier",
            "bucket",
            "n_sims",
            "n_sims_hit",
            "occurs_prob",
            "boolean_value",
        )
    )
    return preds


def main() -> None:
    print("Loading mapping table ...")
    mapping = _pattern_index(load_mapping())
    queries = sorted(set(mapping["query"].to_list()))
    print(f"  {len(mapping)} mapping rows, {len(queries)} unique queries")

    print("Loading eval labels ...")
    labels = load_eval_labels(queries)
    print(f"  {len(labels)} label rows across {labels['query'].n_unique()} queries")

    print("Loading EQ buckets ...")
    buckets = load_eq_buckets()

    print("Streaming ETHOS trajectories ...")
    sim_hits = stream_predictions(mapping)
    print(f"  total sim hits across all 20 files: {len(sim_hits)}")

    print("Aggregating predictions ...")
    preds = aggregate_predictions(sim_hits, mapping, labels, buckets)
    print(f"  {len(preds)} prediction rows")
    print()
    print("Hit-rate by tier x duration:")
    print(
        preds.group_by(["tier", "duration_days"])
        .agg(
            pl.len().alias("n_rows"),
            pl.col("n_sims_hit").sum().alias("total_hits"),
            (pl.col("n_sims_hit").sum() / (pl.len() * N_SIMS)).alias("hit_rate"),
        )
        .sort(["tier", "duration_days"])
    )

    out = CACHE_DIR / "ethos_predictions.parquet"
    preds.write_parquet(out)
    _upload(out, PREDICTIONS_BLOB)
    print(f"\nWrote {out}")
    print(f"Uploaded gs://{BUCKET}/{PREDICTIONS_BLOB}")


if __name__ == "__main__":
    main()
