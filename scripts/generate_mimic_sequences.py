#!/usr/bin/env python
"""Drive EQ_generate_query_sequences workers across all MIMIC-IV shards in-process.

Usage:
    python scripts/generate_mimic_sequences.py \
        --data-dir /path/to/intermediate --out-dir /path/to/tasks \
        --processed /path/to/preprocessed --split train --n-contexts 2048 --jobs 8

One worker per (shard, task_shard); workers are pure functions writing one parquet each, so
re-running skips existing outputs (resume-friendly).
"""

import argparse
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "4")


def _worker(args_tuple):
    from every_query.generate_tasks.sample_query_sequences import run_worker

    kwargs = args_tuple
    return str(run_worker(**kwargs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--processed", required=True, type=Path, help="dir holding metadata/codes.parquet")
    p.add_argument("--split", default="train")
    p.add_argument("--n-contexts", type=int, default=2048)
    p.add_argument("--task-shards", type=int, default=1)
    p.add_argument("--min-queries", type=int, default=1)
    p.add_argument("--max-queries", type=int, default=5)
    p.add_argument("--duration-min", type=int, default=1)
    p.add_argument("--duration-max", type=int, default=365)
    p.add_argument("--min-context-per-subject", type=int, default=10)
    p.add_argument("--eos-first-fraction", type=float, default=0.0)
    p.add_argument("--duration-mode", default="random", choices=["random", "same", "nondecreasing"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--limit-shards", type=int, default=None)
    args = p.parse_args()

    from every_query.generate_tasks.sample_tasks import read_query_codes

    query_codes = read_query_codes(args.processed)
    print(f"query universe: {len(query_codes)} codes")

    shard_dir = args.data_dir / "data" / args.split
    shards = sorted(fp.stem for fp in shard_dir.glob("*.parquet"))
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    print(f"{len(shards)} shards x {args.task_shards} task shards for split={args.split}")

    jobs = [
        dict(
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            query_codes=query_codes,
            split=args.split,
            input_shard=shard,
            task_shard=ts,
            seed=args.seed,
            n_contexts=args.n_contexts,
            min_queries=args.min_queries,
            max_queries=args.max_queries,
            duration_min=args.duration_min,
            duration_max=args.duration_max,
            min_context_per_subject=args.min_context_per_subject,
            eos_first_fraction=args.eos_first_fraction,
            duration_mode=args.duration_mode,
        )
        for shard in shards
        for ts in range(args.task_shards)
    ]

    done = 0
    # ``spawn`` (not fork): the parent has live polars/torch thread pools by this point, and
    # fork-inheriting their locks deadlocks the children on first use.
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as ex:
        futures = [ex.submit(_worker, j) for j in jobs]
        for fut in as_completed(futures):
            fut.result()  # re-raise worker errors
            done += 1
            if done % 20 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} workers done")


if __name__ == "__main__":
    main()
