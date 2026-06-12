#!/usr/bin/env python
"""Designed, clinically meaningful evaluation of the conditional model on held-out MIMIC-IV.

Three prediction-anchor families (a task's positives/negatives are the anchors where the target
code does / does not occur in (t, t+d]):

  - POST-ADMISSION  : 24 h after a HOSPITAL_ADMISSION event  (acute in-hospital risk).
  - POST-DISCHARGE  : at a HOSPITAL_DISCHARGE//HOME event     (genuine post-discharge outcomes).
  - RANDOM-TIME     : a uniformly sampled valid event time    (anytime risk, matches training).

All anchors require >= 10 prior events and are capped at 3 / subject.

Reported metrics:
  A. Single-query within-task AUROC (bootstrap 95% CI) — the model as an ordinary risk predictor.
  B. Conditioning demonstrations — score a target under a teacher-forced prior answer (YES vs NO)
     on the same anchors, showing the model updates downstream risk sensibly.

Code choices are SINGLE codes (the model answers single-code queries), named precisely:
  MEDS_DEATH; ICU_ADMISSION//Medical Intensive Care Unit (MICU); HOSPITAL_DISCHARGE//HOME;
  HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM (ED-route readmission); TIMELINE//END.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from eval_v2 import load_model, score_last
from every_query.data.schema import QuerySeqSchema
from every_query.generate_tasks.sample_query_sequences import (
    CTX_ID_COL, POSITION_COL, label_binary_occurrence,
)
from every_query.generate_tasks.sample_tasks import _read_event_shard

DEATH = "MEDS_DEATH"
MICU = "ICU_ADMISSION//Medical Intensive Care Unit (MICU)"
ER_ADM = "HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM"
HOME_DC = "HOSPITAL_DISCHARGE//HOME"
END = "TIMELINE//END"
MIN_PRIOR = 10
MAX_PER_SUBJECT = 3

# anchor_family -> { task_name: (code, horizon_days) }
SINGLE_TASKS = {
    "post_admission": {
        "mortality_30d (post-admission +24h)": (DEATH, 30.0),
        "mortality_7d (post-admission +24h)": (DEATH, 7.0),
        "MICU admission 7d (post-admission +24h)": (MICU, 7.0),
        "home discharge 7d (post-admission +24h)": (HOME_DC, 7.0),
    },
    "post_discharge": {
        "ED readmission 30d (post-discharge)": (ER_ADM, 30.0),
        "mortality 30d (post-discharge)": (DEATH, 30.0),
    },
    "random_time": {
        "mortality 30d (random time)": (DEATH, 30.0),
        "mortality 7d (random time)": (DEATH, 7.0),
    },
}
# anchor_family -> { name: (prior_code, prior_d, target_code, target_d) }
COND_TASKS = {
    "post_admission": {
        "30d mortality | record ends in 30d (TIMELINE//END)": (END, 30.0, DEATH, 30.0),
        "30d mortality | MICU admission in 7d": (MICU, 7.0, DEATH, 30.0),
    },
    "post_discharge": {
        "90d mortality | ED readmission in 30d": (ER_ADM, 30.0, DEATH, 90.0),
    },
}


def _prior_count(events, anchors):
    """Keep anchors with >= MIN_PRIOR events at/<= prediction_time."""
    return (
        anchors.sort("subject_id", "prediction_time")
        .join_asof(
            events.sort("subject_id", "time").with_columns(
                pl.int_range(pl.len()).over("subject_id").alias("_n")
            ),
            left_on="prediction_time", right_on="time", by="subject_id", strategy="backward",
        )
        .filter(pl.col("_n") >= MIN_PRIOR)
        .select("subject_id", "prediction_time").unique()
    )


def _cap_per_subject(anchors, seed):
    return (
        anchors.with_columns(pl.col("prediction_time").hash(seed=seed % (2**31)).alias("_h"))
        .sort("subject_id", "_h").group_by("subject_id", maintain_order=True)
        .head(MAX_PER_SUBJECT).drop("_h").sort("subject_id", "prediction_time")
    )


def build_anchor_sets(events, seed):
    out = {}
    # post-admission: admission + 24h
    adm = events.filter(pl.col("code").str.starts_with("HOSPITAL_ADMISSION//")).select(
        "subject_id", (pl.col("time") + pl.duration(hours=24)).alias("prediction_time")
    )
    out["post_admission"] = _cap_per_subject(_prior_count(events, adm), seed) if adm.height else adm
    # post-discharge: at HOME discharge events
    dc = events.filter(pl.col("code") == HOME_DC).select(
        "subject_id", pl.col("time").alias("prediction_time")
    )
    out["post_discharge"] = _cap_per_subject(_prior_count(events, dc), seed + 1) if dc.height else dc
    # random-time: uniformly sampled valid event times
    valid = (
        events.with_columns(pl.int_range(pl.len()).over("subject_id").alias("_n"))
        .filter(pl.col("_n") >= MIN_PRIOR).select("subject_id", pl.col("time").alias("prediction_time"))
        .unique()
    )
    if valid.height:
        valid = valid.with_columns(pl.col("prediction_time").hash(seed=(seed + 2) % (2**31)).alias("_h"))
        out["random_time"] = valid.sort("subject_id", "_h").group_by("subject_id", maintain_order=True).head(
            MAX_PER_SUBJECT
        ).drop("_h").sort("subject_id", "prediction_time")
    else:
        out["random_time"] = valid
    return out


def _auroc(y, p):
    y = np.asarray(y)
    return float(roc_auc_score(y, np.asarray(p))) if len(np.unique(y)) > 1 else None


def _boot_ci(y, p, n=2000, seed=0):
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            v.append(roc_auc_score(y[i], p[i]))
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (None, None)


def make_task_df(anchors_by_shard, family, spec):
    frames = []
    for si, (anchsets, ev) in anchors_by_shard.items():
        anc = anchsets[family]
        if anc.height == 0:
            continue
        rows = []
        for cid, c in enumerate(anc.iter_rows(named=True)):
            for pos, (code, d) in enumerate(spec):
                rows.append((cid, pos, c["subject_id"], c["prediction_time"], code, float(d)))
        idf = pl.DataFrame(
            rows, orient="row",
            schema=[CTX_ID_COL, POSITION_COL, "subject_id", "prediction_time", "query", "duration_days"],
        ).with_columns(
            pl.col(CTX_ID_COL).cast(pl.UInt32), pl.col(POSITION_COL).cast(pl.Int64),
            pl.col("prediction_time").cast(pl.Datetime("us")), pl.col("duration_days").cast(pl.Float32),
        )
        frames.append(label_binary_occurrence(idf, ev))
    return pl.concat(frames, how="vertical") if frames else None


def write_one(df, d, split):
    (d / split).mkdir(parents=True, exist_ok=True)
    pl.from_arrow(QuerySeqSchema.align(df.to_arrow())).write_parquet(d / split / "tasks.parquet")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--cohort-dir", required=True, type=Path)
    ap.add_argument("--intermediate", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--split", default="held_out")
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    model = load_model(args.run_dir)
    anchors_by_shard, n_anc = {}, {k: 0 for k in ("post_admission", "post_discharge", "random_time")}
    for si, fp in enumerate(sorted((args.intermediate / "data" / args.split).glob("*.parquet"))):
        ev = _read_event_shard(fp)
        sets = build_anchor_sets(ev, seed=31 + si)
        anchors_by_shard[si] = (sets, ev)
        for k in n_anc:
            n_anc[k] += sets[k].height
    print("anchors:", n_anc)

    work = args.out.parent / "_clinical_tasks"
    summary = {"n_anchors": n_anc, "single_query": {}, "conditional": {}}

    # A. single-query within-task AUROC
    for fam, tasks in SINGLE_TASKS.items():
        for name, (c, d) in tasks.items():
            df = make_task_df(anchors_by_shard, fam, [(c, d)])
            if df is None or df.height == 0:
                continue
            td = write_one(df, work / f"single_{abs(hash(name))}", args.split)
            sc = score_last(model, args.cohort_dir, td, args.split, force_prior=None, batch_size=args.batch_size)
            y, p = sc["true_answer"].to_list(), sc["prob"].to_list()
            lo, hi = _boot_ci(y, p)
            summary["single_query"][name] = {
                "family": fam, "code": c, "horizon_d": d, "n": len(y),
                "prevalence": float(np.mean(y)), "auroc": _auroc(y, p), "auroc_ci95": [lo, hi],
            }
            print(f"  [single] {name:42s} prev={np.mean(y):.4f} AUROC={_auroc(y,p)} n={len(y)}")

    # B. conditioning demonstrations
    for fam, tasks in COND_TASKS.items():
        for name, (pc, pd, tc, td_) in tasks.items():
            df = make_task_df(anchors_by_shard, fam, [(pc, pd), (tc, td_)])
            if df is None or df.height == 0:
                continue
            tdir = write_one(df, work / f"cond_{abs(hash(name))}", args.split)
            nat = score_last(model, args.cohort_dir, tdir, args.split, force_prior=None, batch_size=args.batch_size)
            no = score_last(model, args.cohort_dir, tdir, args.split, force_prior=0, batch_size=args.batch_size)
            yes = score_last(model, args.cohort_dir, tdir, args.split, force_prior=1, batch_size=args.batch_size)
            y = nat["true_answer"].to_list()
            lo, hi = _boot_ci(y, nat["prob"].to_list())
            summary["conditional"][name] = {
                "family": fam, "prior": f"{pc}@{pd}d", "target": f"{tc}@{td_}d", "n": len(y),
                "target_prevalence": float(np.mean(y)),
                "target_auroc_natural": _auroc(y, nat["prob"].to_list()), "auroc_ci95": [lo, hi],
                "mean_P_target_given_prior_NO": float(np.mean(no["prob"].to_numpy())),
                "mean_P_target_given_prior_YES": float(np.mean(yes["prob"].to_numpy())),
            }
            s = summary["conditional"][name]
            print(f"  [cond]  {name:42s} AUROC={s['target_auroc_natural']} "
                  f"P|NO={s['mean_P_target_given_prior_NO']:.4f} P|YES={s['mean_P_target_given_prior_YES']:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
