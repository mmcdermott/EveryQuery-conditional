"""Pick the ~6 demo patients that the notebook will hard-code.

Run this script once locally (not in Colab) to materialize
``demo_patients.json``. The notebook reads that JSON instead of redoing the
selection at runtime — this keeps Colab fast, deterministic, and free of
curation code bloat.

Usage:
    uv run --with pandas --with pyarrow --with numpy --with gcsfs \\
        python paper_experiments/eq_vs_ar_demo/select_demo_patients.py

Reads from ``gs://every-query-runs/`` directly (gcsfs + Application Default
Credentials). Writes ``paper_experiments/eq_vs_ar_demo/demo_patients.json``
beside this script.

The selection logic mirrors what the notebook used to do inline:
1. Pull a few MIMIC-IV held-out shards (10 MB each).
2. Restrict to subjects that have full EQ-prediction coverage for all 8 demo
   tasks AND at least 80 events of history.
3. Score candidates by ground-truth diversity across the 8 tasks (peaks at
   half-positive / half-negative) and total event count, take the top 6 with
   pairwise-distinct label vectors.
"""

from __future__ import annotations
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional

import gcsfs
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "demo_patients.json"
BUCKET = "every-query-runs"
SHARD_INDICES = [0, 1, 2]
EQ_PREDS_PATH = "outputs/2026-04-02/23-43-54/eval/best/eval_preds_10000_ID__8db2be6fadf8_20260413_201311.parquet"
MEDS_HELDOUT_SHARD_TEMPLATE = "meds/MIMIC_MEDS/MEDS_cohort/intermediate/data/held_out/{shard}.parquet"

# Demo task universe — mirrors notebook's TASKS list.
TASKS: list[tuple[str, int]] = [
    ("MEDS_DEATH",                                                 30),
    ("MEDS_DEATH",                                                180),
    ("DIAGNOSIS//ICD//9//4271",                                    90),
    ("DIAGNOSIS//ICD//9//5856",                                   365),
    ("DIAGNOSIS//ICD//10//I25118",                                180),
    ("INFUSION_START//221794//value_[8.004926,10.000001)",         30),
    ("MEDICATION//Carbidopa-Levodopa (25-100)//Administered",      90),
    ("LAB//51274//sec//value_[14.1,15.3)",                         30),
]


def gread_parquet(fs: gcsfs.GCSFileSystem, rel: str) -> pd.DataFrame:
    cache = Path(f"/tmp/eq_demo_select/{rel.replace('/', '__')}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        with fs.open(f"{BUCKET}/{rel}", "rb") as src, cache.open("wb") as dst:
            dst.write(src.read())
    return pd.read_parquet(cache)


def compute_ground_truth(events: pd.DataFrame, prediction_time_us: int, code: str, duration_days: int) -> int:
    pt = pd.Timestamp(prediction_time_us, unit="us")
    end = pt + pd.Timedelta(days=duration_days)
    return int(((events["time"] > pt) & (events["time"] <= end) & (events["code"] == code)).any())


def main() -> None:
    fs = gcsfs.GCSFileSystem()
    print("Loading EQ predictions and MEDS held-out shards from GCS...", flush=True)
    preds = gread_parquet(fs, EQ_PREDS_PATH)
    shard = pd.concat(
        [gread_parquet(fs, MEDS_HELDOUT_SHARD_TEMPLATE.format(shard=i)) for i in SHARD_INDICES],
        ignore_index=True,
    )
    print(f"  preds rows: {len(preds):,}, shard subjects: {shard['subject_id'].nunique():,}", flush=True)

    shard_subjects = set(shard["subject_id"].unique())
    preds_in_shard = preds[preds["subject_id"].isin(shard_subjects)].copy()

    needed_df = pd.DataFrame(TASKS, columns=["code", "duration_days"])
    covered_counts = (
        preds_in_shard.merge(needed_df, on=["code", "duration_days"]).groupby("subject_id").size()
    )
    fully_covered = set(covered_counts[covered_counts == len(TASKS)].index)

    event_counts = shard.groupby("subject_id").size().rename("n_events").reset_index()
    candidate_subjects = [
        sid for sid in event_counts[event_counts["n_events"] >= 80]["subject_id"].tolist()
        if sid in fully_covered
    ]
    print(f"  candidates with full coverage + >= 80 events: {len(candidate_subjects):,}", flush=True)

    events_by_subject = {sid: g for sid, g in shard.sort_values(["subject_id", "time"]).groupby("subject_id", sort=False)}

    def label_vector(sid: int) -> Optional[list[int]]:
        sub_preds = preds_in_shard[preds_in_shard["subject_id"] == sid]
        if sub_preds.empty:
            return None
        pt_us = sub_preds["prediction_time"].iloc[0]
        sub_events = events_by_subject.get(sid)
        if sub_events is None or len(sub_events) < 80:
            return None
        return [compute_ground_truth(sub_events, pt_us, code, dur) for code, dur in TASKS]

    rng = random.Random(20260428)
    sampled = rng.sample(candidate_subjects, min(400, len(candidate_subjects)))
    scored = []
    for sid in sampled:
        lv = label_vector(sid)
        if lv is None:
            continue
        diversity = sum(lv) * (len(lv) - sum(lv))
        n_events = int(event_counts.loc[event_counts["subject_id"] == sid, "n_events"].iloc[0])
        scored.append((sid, lv, diversity, n_events))
    scored.sort(key=lambda x: (-x[2], -x[3]))

    chosen: list[dict] = []
    for sid, lv, diversity, n_events in scored:
        if any(sum(a != b for a, b in zip(lv, prev["labels"])) < 1 for prev in chosen):
            continue  # skip exact duplicate label-vector
        sub_preds = preds_in_shard[preds_in_shard["subject_id"] == sid]
        pt_us = int(sub_preds["prediction_time"].iloc[0])
        chosen.append({
            "subject_id": int(sid),
            "prediction_time_us": pt_us,
            "labels": lv,
            "n_events": int(n_events),
        })
        if len(chosen) >= 6:
            break

    OUT.write_text(json.dumps({
        "tasks": [{"code": c, "duration_days": d} for c, d in TASKS],
        "patients": chosen,
        "selection_seed": 20260428,
        "shards_used": SHARD_INDICES,
    }, indent=2))
    print(f"\nWrote {OUT}: {len(chosen)} patients.")
    for p in chosen:
        print(f"  subject {p['subject_id']:>10}: {p['n_events']:>5} events, labels {p['labels']}")


if __name__ == "__main__":
    main()
