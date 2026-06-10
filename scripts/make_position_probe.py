#!/usr/bin/env python
"""Dense per-position conditioning probe: same target code placed at positions 1..P across many patients.

Motivation: pooled AUROC across heterogeneous query codes is inflated by cross-query base-rate
separation. To measure the model's *within-query* skill and isolate the effect of conditioning on
prior answers, we want, for each curated target code C and each target position p, many
(patient, prediction_time) examples of the sequence

    [ __CENSOR__(d) ]  [ filler_1 ] ... [ filler_{p-1} ]  [ C(d) @ position p ]

so that per-(C, p) AUROC has enough positives, and comparing a fixed C across p=1..P varies *only*
the number of prior teacher-forced answers the target conditions on.

Output: one QuerySeqSchema parquet at ``{out_dir}/probe/tasks.parquet``; the target of every
sequence is its last query (position = sequence length - 1).
"""

import argparse
import os
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
    sample_log_uniform_durations,
)
from every_query.generate_tasks.sample_tasks import (
    _read_event_shard,
    compute_max_time_per_subject,
    read_query_codes,
    sample_contexts,
)

# Curated, prevalence-spanning target codes selected by frequency within recognizable prefixes.
CURATED_PREFIXES = [
    ("MEDS_DEATH", 1),
    ("HOSPITAL_ADMISSION//", 3),
    ("HOSPITAL_DISCHARGE//", 3),
    ("ICU_ADMISSION//", 2),
    ("ICU_DISCHARGE//", 2),
    ("MEDICATION//", 4),
    ("LAB//", 4),
    ("DIAGNOSIS//", 3),
]


def curated_codes(processed: Path) -> list[str]:
    df = pl.read_parquet(processed / "metadata" / "codes.parquet", columns=["code", "code/n_subjects"])
    out: list[str] = []
    for pref, k in CURATED_PREFIXES:
        sub = (
            df.filter(pl.col("code").str.starts_with(pref) | (pl.col("code") == pref))
            .sort("code/n_subjects", descending=True)
            .head(k)
        )
        out += sub["code"].to_list()
    seen: set[str] = set()
    return [c for c in out if not (c in seen or seen.add(c))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--processed", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--split", default="held_out")
    ap.add_argument("--n-contexts", type=int, default=1024)
    ap.add_argument("--max-position", type=int, default=4)
    ap.add_argument("--target-duration", type=float, default=90.0)
    ap.add_argument("--min-context-per-subject", type=int, default=10)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--limit-shards", type=int, default=None)
    args = ap.parse_args()

    targets = curated_codes(args.processed)
    filler_vocab = read_query_codes(args.processed)
    print(f"{len(targets)} target codes: {targets}")

    shard_dir = args.data_dir / "data" / args.split
    shards = sorted(shard_dir.glob("*.parquet"))
    if args.limit_shards:
        shards = shards[: args.limit_shards]

    frames = []
    for si, shard_fp in enumerate(shards):
        events = _read_event_shard(shard_fp)
        ctx = sample_contexts(
            events,
            n=args.n_contexts,
            min_context_per_subject=args.min_context_per_subject,
            seed=args.seed + si,
        )
        if ctx.height == 0:
            continue
        maxt = compute_max_time_per_subject(events)
        rng = np.random.default_rng(args.seed * 1000 + si)

        rows = []
        cid = 0
        for c in ctx.iter_rows(named=True):
            for p in range(1, args.max_position + 1):
                tcode = targets[int(rng.integers(0, len(targets)))]
                codes = [CENSOR_QUERY_CODE]
                durs = [args.target_duration]
                for _ in range(1, p):  # filler queries at positions 1..p-1
                    codes.append(filler_vocab[int(rng.integers(0, len(filler_vocab)))])
                    durs.append(float(sample_log_uniform_durations(1, 1, 365, rng)[0]))
                codes.append(tcode)
                durs.append(args.target_duration)
                for pos, (cc, dd) in enumerate(zip(codes, durs, strict=True)):
                    rows.append((cid, pos, c["subject_id"], c["prediction_time"], cc, dd))
                cid += 1

        idf = pl.DataFrame(
            rows,
            schema=[CTX_ID_COL, POSITION_COL, "subject_id", "prediction_time", "query", "duration_days"],
            orient="row",
        ).with_columns(
            pl.col(CTX_ID_COL).cast(pl.UInt32),
            pl.col(POSITION_COL).cast(pl.Int64),
            pl.col("prediction_time").cast(pl.Datetime("us")),
            pl.col("duration_days").cast(pl.Float32),
        )
        frames.append(label_sequence_index_df(idf, events, maxt))

    df = pl.concat(frames, how="vertical")
    out = args.out_dir / "probe"
    out.mkdir(parents=True, exist_ok=True)
    pl.from_arrow(QuerySeqSchema.align(df.to_arrow())).write_parquet(out / "tasks.parquet")
    # quick coverage report
    tgt = df.with_columns(
        pl.col("queries").list.last().alias("target"),
        (pl.col("queries").list.len() - 1).alias("target_pos"),
        pl.col("answers").list.last().alias("target_ans"),
    )
    cov = (
        tgt.group_by("target", "target_pos")
        .agg(pl.len().alias("n"), pl.col("target_ans").drop_nulls().mean().alias("prev"))
        .sort("target", "target_pos")
    )
    print(f"wrote {df.height} probe sequences across {tgt['target'].n_unique()} target codes x "
          f"{tgt['target_pos'].n_unique()} positions; median cell size {cov['n'].median():.0f}")


if __name__ == "__main__":
    main()
