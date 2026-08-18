#!/usr/bin/env python
"""Build designed, clinically meaningful query-sequence tasks over MIMIC-IV held-out subjects.

Each task anchors prediction times at 24h after a (sampled) hospital admission and asks a fixed
conditional query sequence.  Every sequence opens with the end-of-record query
(``TIMELINE//END``), which is how censoring is expressed since the v2 redesign.  Output: one
``QuerySeqSchema`` parquet per task at ``{out_dir}/{task}/tasks.parquet``, directly consumable by
``EQ_predict_sequences tasks_dir={out_dir}/{task}``.

Tasks (EOS = ``TIMELINE//END``, at the short horizon below):
  mortality_30d            [(EOS, 1), (MEDS_DEATH, 30)]
  icu_then_death           [(EOS, 1), (MICU admission, 7), (MEDS_DEATH, 30)]
  discharge_then_readmit   [(EOS, 1), (HOME discharge, 14), (ER readmission, 90)]
  readmit_90d              [(EOS, 1), (ER readmission, 90)]
  home_discharge_then_death [(EOS, 1), (HOME discharge, 30), (MEDS_DEATH, 180)]

Usage:
    python scripts/make_clinical_task_sequences.py \
        --data-dir /path/to/intermediate --out-dir /path/to/tasks_clinical \
        --split held_out --max-anchors-per-subject 1 --seed 7

Status (conditional-v2): repaired against the v2 binary labels.  The deleted ``__CENSOR__``
sentinel ("is there data after t+d?") is now the real ``TIMELINE//END`` query ("does the record end
within d?") -- its logical complement, and an ordinary answerable query rather than a sentinel.
``label_binary_occurrence`` replaces the 3-valued ``label_sequence_index_df``: there is no null
answer class any more, so nothing is dropped from downstream metrics and censoring must be read off
the EOS answer itself.  Anchoring is unchanged (this script never used ``sample_contexts``).
"""

import argparse
import os
from datetime import timedelta
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "8")

import numpy as np
import polars as pl

from every_query.data.schema import QuerySeqSchema
from every_query.data.seq_dataset import EOS_CODE
from every_query.generate_tasks.sample_query_sequences import (
    CTX_ID_COL,
    POSITION_COL,
    label_binary_occurrence,
)
from every_query.generate_tasks.sample_tasks import _read_event_shard

DEATH = "MEDS_DEATH"
MICU_ADM = "ICU_ADMISSION//Medical Intensive Care Unit (MICU)"
HOME_DISCHARGE = "HOSPITAL_DISCHARGE//HOME"
ER_ADMISSION = "HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM"

# End-of-record query horizon for position 0.  IMPORTANT: this must NOT equal the target horizon.
# ``[TIMELINE//END, h]`` answers "does the record end within h?", which — for a terminal target like
# death, and structurally for any occurrence label at the same horizon — is near-deterministically
# tied to the target answer at that *same* horizon.  Teacher-forcing a same-horizon EOS answer leaks
# the label; that is the v1 failure this constant exists to avoid (30-day mortality AUROC 0.991, of
# which 0.996 came from the censor answer alone).  A short, fixed horizon (default 1 day) keeps
# position 0 in distribution without revealing whether the multi-day target window is observable.
#
# v2 note: under ``label_binary_occurrence`` there is no null/censored answer class — an
# unobservable window is labeled False, not dropped.  So censoring of the TARGET is no longer
# handled by the labeler; it must be read off the EOS query in the sequence.
EOS_HORIZON_DAYS = 1.0

TASKS: dict[str, list[tuple[str, float]]] = {
    "mortality_30d": [(EOS_CODE, EOS_HORIZON_DAYS), (DEATH, 30.0)],
    "icu_then_death": [(EOS_CODE, EOS_HORIZON_DAYS), (MICU_ADM, 7.0), (DEATH, 30.0)],
    "discharge_then_readmit": [
        (EOS_CODE, EOS_HORIZON_DAYS),
        (HOME_DISCHARGE, 14.0),
        (ER_ADMISSION, 90.0),
    ],
    "readmit_90d": [(EOS_CODE, EOS_HORIZON_DAYS), (ER_ADMISSION, 90.0)],
    "home_discharge_then_death": [
        (EOS_CODE, EOS_HORIZON_DAYS),
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
            labeled = label_binary_occurrence(index_df, events)
            per_task_frames[task].append(labeled)

    for task, frames in per_task_frames.items():
        df = pl.concat(frames, how="vertical")
        out_fp = args.out_dir / task / "tasks.parquet"
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        aligned = QuerySeqSchema.align(df.to_arrow())
        pl.from_arrow(aligned).write_parquet(out_fp)

        # Quick prevalence summary for the final (target) query.  v2 answers are non-null
        # booleans, so the old "censored fraction" is gone; the position-0 EOS answer is the
        # closest replacement -- the share of anchors whose record ends inside EOS_HORIZON_DAYS.
        target = df.select(pl.col("answers").list.last().alias("t"))["t"]
        record_ends = df.select(pl.col("answers").list.first().alias("e"))["e"]
        print(
            f"{task}: {df.height} sequences; target prevalence {target.mean():.4f} "
            f"(record ends within {EOS_HORIZON_DAYS:g}d for {record_ends.mean():.3f})"
        )


if __name__ == "__main__":
    main()
