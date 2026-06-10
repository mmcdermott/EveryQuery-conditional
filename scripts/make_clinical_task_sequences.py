#!/usr/bin/env python
"""Build designed, clinically meaningful query-sequence tasks over MIMIC-IV held-out subjects.

Each task anchors prediction times at 24h after a (sampled) hospital admission and asks a fixed
conditional query sequence.  All sequences start with the censor query, per the architecture's
convention.  Output: one ``QuerySeqSchema`` parquet per task at ``{out_dir}/{task}/tasks.parquet``,
directly consumable by ``EQ_predict_sequences tasks_dir={out_dir}/{task}``.

Tasks:
  mortality_30d            [(CENSOR, 30), (MEDS_DEATH, 30)]
  icu_then_death           [(CENSOR, 30), (MICU admission, 7), (MEDS_DEATH, 30)]
  discharge_then_readmit   [(CENSOR, 90), (HOME discharge, 14), (ER readmission, 90)]
  readmit_90d              [(CENSOR, 90), (ER readmission, 90)]
  home_discharge_then_death [(CENSOR, 180), (HOME discharge, 30), (MEDS_DEATH, 180)]

Usage:
    python scripts/make_clinical_task_sequences.py \
        --data-dir /path/to/intermediate --out-dir /path/to/tasks_clinical \
        --split held_out --max-anchors-per-subject 1 --seed 7
"""

import argparse
import os
from datetime import timedelta
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "8")

import numpy as np
import polars as pl

from every_query.data.schema import QuerySeqSchema
from every_query.data.seq_dataset import CENSOR_QUERY_CODE
from every_query.generate_tasks.sample_query_sequences import (
    CTX_ID_COL,
    POSITION_COL,
    label_sequence_index_df,
)
from every_query.generate_tasks.sample_tasks import _read_event_shard, compute_max_time_per_subject

DEATH = "MEDS_DEATH"
MICU_ADM = "ICU_ADMISSION//Medical Intensive Care Unit (MICU)"
HOME_DISCHARGE = "HOSPITAL_DISCHARGE//HOME"
ER_ADMISSION = "HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM"

TASKS: dict[str, list[tuple[str, float]]] = {
    "mortality_30d": [(CENSOR_QUERY_CODE, 30.0), (DEATH, 30.0)],
    "icu_then_death": [(CENSOR_QUERY_CODE, 30.0), (MICU_ADM, 7.0), (DEATH, 30.0)],
    "discharge_then_readmit": [
        (CENSOR_QUERY_CODE, 90.0),
        (HOME_DISCHARGE, 14.0),
        (ER_ADMISSION, 90.0),
    ],
    "readmit_90d": [(CENSOR_QUERY_CODE, 90.0), (ER_ADMISSION, 90.0)],
    "home_discharge_then_death": [
        (CENSOR_QUERY_CODE, 180.0),
        (HOME_DISCHARGE, 30.0),
        (DEATH, 180.0),
    ],
}

ANCHOR_OFFSET_HOURS = 24
MIN_PRIOR_EVENTS = 10


def build_anchors(events: pl.DataFrame, max_per_subject: int, seed: int) -> pl.DataFrame:
    """Anchor prediction times at admission + 24h, requiring MIN_PRIOR_EVENTS history."""
    admissions = events.filter(pl.col("code").str.starts_with("HOSPITAL_ADMISSION//")).select(
        "subject_id", pl.col("time").alias("admission_time")
    )
    if admissions.height == 0:
        return pl.DataFrame(
            schema={"subject_id": pl.Int64, "prediction_time": pl.Datetime("us")}
        )

    anchors = admissions.with_columns(
        (pl.col("admission_time") + pl.duration(hours=ANCHOR_OFFSET_HOURS)).alias("prediction_time")
    )

    # Count prior events at each anchor (events with time <= prediction_time).
    counts = (
        anchors.sort("subject_id", "prediction_time")
        .join_asof(
            events.sort("subject_id", "time")
            .with_columns(pl.int_range(pl.len()).over("subject_id").alias("_n_prior")),
            left_on="prediction_time",
            right_on="time",
            by="subject_id",
            strategy="backward",
        )
        .filter(pl.col("_n_prior") >= MIN_PRIOR_EVENTS)
        .select("subject_id", "prediction_time")
        .unique()
        .sort("subject_id", "prediction_time")
    )

    if max_per_subject:
        rng_seed = seed % (2**31)
        counts = (
            counts.with_columns(pl.col("prediction_time").hash(seed=rng_seed).alias("_h"))
            .sort("subject_id", "_h")
            .group_by("subject_id", maintain_order=True)
            .head(max_per_subject)
            .drop("_h")
        )
    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--split", default="held_out")
    p.add_argument("--max-anchors-per-subject", type=int, default=1)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    shard_dir = args.data_dir / "data" / args.split
    shards = sorted(shard_dir.glob("*.parquet"))
    print(f"{len(shards)} shards in {shard_dir}")

    per_task_frames: dict[str, list[pl.DataFrame]] = {t: [] for t in TASKS}

    for shard_fp in shards:
        events = _read_event_shard(shard_fp)
        anchors = build_anchors(events, args.max_anchors_per_subject, args.seed)
        if anchors.height == 0:
            continue
        max_time = compute_max_time_per_subject(events)

        for task, spec in TASKS.items():
            n_q = len(spec)
            n_ctx = anchors.height
            index_df = (
                anchors.with_row_index(CTX_ID_COL)
                .join(
                    pl.DataFrame(
                        {
                            POSITION_COL: pl.Series(range(n_q), dtype=pl.Int64),
                            "query": [c for c, _ in spec],
                            "duration_days": pl.Series([d for _, d in spec], dtype=pl.Float32),
                        }
                    ),
                    how="cross",
                )
                .select(
                    CTX_ID_COL, POSITION_COL, "subject_id", "prediction_time", "query", "duration_days"
                )
                .sort(CTX_ID_COL, POSITION_COL)
            )
            labeled = label_sequence_index_df(index_df, events, max_time)
            per_task_frames[task].append(labeled)

    for task, frames in per_task_frames.items():
        df = pl.concat(frames, how="vertical")
        out_fp = args.out_dir / task / "tasks.parquet"
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        aligned = QuerySeqSchema.align(df.to_arrow())
        pl.from_arrow(aligned).write_parquet(out_fp)

        # Quick prevalence summary for the final (target) query.
        target = df.select(pl.col("answers").list.last().alias("t"))["t"]
        print(
            f"{task}: {df.height} sequences; target prevalence "
            f"{target.drop_nulls().mean():.4f} (censored {target.null_count() / df.height:.3f})"
        )


if __name__ == "__main__":
    main()
