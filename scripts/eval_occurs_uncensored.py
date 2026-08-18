#!/usr/bin/env python
"""Original-EveryQuery-comparable metric: macro per-task occurs-AUROC on the UNCENSORED cohort.

Original EveryQuery reports occurrence AUC only where censoring is false (the outcome window is
fully observed).  We replicate that exactly: for a random task (code C, duration D) we build the
sequence ``[TIMELINE//END, D]=0  [C, D]`` — telling the model, via the teacher-forced EOS=NO prior,
that the record does NOT end within D (uncensored) — and evaluate the C-prediction ONLY on contexts
where that is actually true (the subject has data past t+D).  Positives are uncensored contexts
where C occurs in (t, t+D]; negatives are uncensored contexts where it does not.  We report the
within-task AUROC, macro-averaged over tasks (patient-uniform), with a bootstrap CI.

We also report the marginal ``[C, D]`` (no EOS prefix) on the same uncensored cohort, to isolate
what the EOS=NO conditioning prefix contributes.

Note: terminal codes (e.g. MEDS_DEATH) have NO uncensored positives by construction (death ends the
record), so they are structurally absent here — exactly the regime original EveryQuery cannot
evaluate, and the reason the model's separate EOS=YES ("record ends") capability matters.
"""

import argparse
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
from meds import DataSchema

from eval_v2 import load_model, sample_eval_contexts, score_last
from every_query.data.schema import QuerySeqSchema
from every_query.data.seq_dataset import EOS_CODE
from every_query.generate_tasks.sample_query_sequences import (
    CTX_ID_COL, POSITION_COL, label_binary_occurrence, sample_log_uniform_durations,
)
from every_query.generate_tasks.sample_tasks import read_query_codes

SID, TIME, CODE = DataSchema.subject_id_name, DataSchema.time_name, DataSchema.code_name


def build_uncensored_triples(ce, n_tasks, dmin, dmax, min_ctx, seed, max_valid=8000):
    """Patient-uniform (query, uncensored pos-ctx, uncensored neg-ctx) triples.

    A context (subj, t) is UNCENSORED for horizon D iff the subject has data past t+D
    (max_time > t+D).  Positive: uncensored context with C in (t, t+D].  Negative: uncensored
    context with no C in (t, t+D].  Patient-uniform within each.
    """
    rng = np.random.default_rng(seed)
    triples = []
    shards = list(ce["events"])
    per_shard = n_tasks // len(shards) + 1
    for si in shards:
        if len(triples) >= n_tasks:
            break
        ev = ce["events"][si]
        evx = ev.with_columns(pl.col(TIME).cum_count().over(SID).alias("_n"))
        mx = ev.group_by(SID).agg(pl.col(TIME).max().alias("mt"))
        V = evx.filter(pl.col("_n") >= min_ctx).select(SID, pl.col(TIME).alias("t")).join(mx, on=SID)
        if V.height == 0:
            continue
        if V.height > max_valid:
            V = V.sample(n=max_valid, seed=int(rng.integers(1 << 31)))
        codes_present = ev[CODE].unique().to_list()
        for _ in range(per_shard):
            if len(triples) >= n_tasks:
                break
            C = codes_present[int(rng.integers(len(codes_present)))]
            D = float(sample_log_uniform_durations(1, dmin, dmax, rng)[0])
            occ = ev.filter(pl.col(CODE) == C).select(SID, pl.col(TIME).alias("tau"))
            win = pl.duration(days=D)
            # uncensored: data past t+D
            Vu = V.filter(pl.col("mt") > pl.col("t") + win)
            if Vu.height == 0:
                continue
            # positive: uncensored ctx with C in (t, t+D]
            pos = Vu.join(occ, on=SID).filter(
                (pl.col("t") < pl.col("tau")) & (pl.col("tau") <= pl.col("t") + win)
            ).select(SID, "t").unique()
            if pos.height == 0:
                continue
            psub = pos[SID].unique()
            sp = psub[int(rng.integers(psub.len()))]
            tp = pos.filter(pl.col(SID) == sp)["t"]
            pos_t = tp[int(rng.integers(tp.len()))]
            # negative: uncensored ctx with NO C in window (patient-uniform)
            neg = None
            for _ in range(40):
                subs = Vu[SID].unique()
                s = subs[int(rng.integers(subs.len()))]
                tv = Vu.filter(pl.col(SID) == s)["t"]
                s_t = tv[int(rng.integers(tv.len()))]
                hit = occ.filter(
                    (pl.col(SID) == s) & (pl.col("tau") > s_t) & (pl.col("tau") <= s_t + timedelta(days=D))
                )
                if hit.height == 0:
                    neg = (s, s_t)
                    break
            if neg is None:
                continue
            triples.append((C, D, sp, pos_t, neg[0], neg[1]))
    return triples


def build_seq_df(triples, with_eos):
    """Rows ordered [t0_pos, t0_neg, t1_pos, ...]; sequence is [EOS@D, C@D] (with_eos) or [C@D]."""
    rows, cid = [], 0
    for (C, D, ps, pt, ns, nt) in triples:
        for (subj, t) in [(ps, pt), (ns, nt)]:
            seq = ([(EOS_CODE, D)] if with_eos else []) + [(C, D)]
            for pos, (cc, dd) in enumerate(seq):
                rows.append((cid, pos, subj, t, cc, float(dd)))
            cid += 1
    return pl.DataFrame(
        rows, orient="row",
        schema=[CTX_ID_COL, POSITION_COL, SID, "prediction_time", "query", "duration_days"],
    ).with_columns(
        pl.col(CTX_ID_COL).cast(pl.UInt32), pl.col(POSITION_COL).cast(pl.Int64),
        pl.col("prediction_time").cast(pl.Datetime("us")), pl.col("duration_days").cast(pl.Float32),
    )


def macro_auc(margins):
    return float(margins.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--cohort-dir", required=True, type=Path)
    ap.add_argument("--intermediate", required=True, type=Path)
    ap.add_argument("--processed", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--split", default="held_out")
    ap.add_argument("--n-tasks", type=int, default=40000)
    ap.add_argument("--n-per-shard", type=int, default=700)
    ap.add_argument("--dmin", type=int, default=1)
    ap.add_argument("--dmax", type=int, default=365)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--boot", type=int, default=4000)
    args = ap.parse_args()

    model = load_model(args.run_dir)
    ce = sample_eval_contexts(args.intermediate, args.split, args.n_per_shard, seed=23)
    triples = build_uncensored_triples(ce, args.n_tasks, args.dmin, args.dmax, args.min_ctx if hasattr(args, "min_ctx") else 10, seed=9)
    T = len(triples)
    print(f"{T} uncensored (query, pos, neg) triples")

    work = args.out.parent / "_uncens_tasks"
    all_ev = pl.concat([ce["events"][s] for s in ce["events"]], how="vertical")
    tsubs = set()
    for (_C, _D, ps, _pt, ns, _nt) in triples:
        tsubs.add(ps); tsubs.add(ns)
    all_ev = all_ev.filter(pl.col(SID).is_in(list(tsubs))).unique().sort([SID, TIME])

    results = {}
    for label, with_eos, force in [("with_EOS=NO_prefix", True, 0), ("marginal_no_prefix", False, None)]:
        df = build_seq_df(triples, with_eos)
        labeled = label_binary_occurrence(df, all_ev)
        td = work / label
        (td / args.split).mkdir(parents=True, exist_ok=True)
        pl.from_arrow(QuerySeqSchema.align(labeled.to_arrow())).write_parquet(td / args.split / "tasks.parquet")
        sc = score_last(model, args.cohort_dir, td, args.split, force_prior=force, batch_size=args.batch_size)
        assert sc.height == 2 * T, f"{label}: {sc.height} != {2*T}"
        probs = sc["prob"].to_numpy()
        margins = (probs[0::2] > probs[1::2]).astype(float)
        rng = np.random.default_rng(0)
        boot = [margins[rng.integers(0, T, T)].mean() for _ in range(args.boot)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        results[label] = {"macro_occurs_auroc": macro_auc(margins),
                          "ci95": [float(lo), float(hi)], "n_tasks": T}
        print(f"  {label:22s} macro occurs-AUROC = {macro_auc(margins):.4f}  95%CI [{lo:.4f}, {hi:.4f}]")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
