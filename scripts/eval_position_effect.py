#!/usr/bin/env python
"""Clean, controlled test of the conditioning effect: same target query, varying # of prior queries.

For each curated target code C and each context (subject, prediction_time), build sequences that end
in the SAME (C, d) but are preceded by 0, 2, or 4 random teacher-forced prior queries:
    L1: [C d]                          (0 priors)
    L3: [f1 .. f2, C d]                (2 priors)
    L5: [f1 .. f4, C d]                (4 priors)
The target duration d is drawn once per (context, C) and reused across L1/L3/L5 so the target query
is identical; only the amount of prior conditioning context differs.  Priors are random codes with
their TRUE answers teacher-forced (the realistic conditional setting).

We then compute, per code, the within-code AUROC of the target prediction at 0/2/4 priors, with a
bootstrap CI over contexts on the macro (over codes) delta AUROC(4 priors) - AUROC(0 priors).  This
isolates the value of conditioning information with proper error bars — unlike the pooled
per-position training metric, which mixes codes of very different base rates and is too noisy.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from every_query.generate_tasks.sample_query_sequences import (
    CTX_ID_COL,
    POSITION_COL,
    label_binary_occurrence,
)
from every_query.generate_tasks.sample_tasks import read_query_codes
from every_query.utils.seeds import derive_seed
from eval_v2 import (
    add_context_sampling_args,
    contexts_from_args,
    curated_codes,
    load_model,
    log_uniform_durations,
    score_last,
)

PRIORS = [0, 2, 4]


def _auroc(y, p):
    y = np.asarray(y)
    return float(roc_auc_score(y, np.asarray(p))) if len(np.unique(y)) > 1 else None


def build_sets(ce, codes, vocab, dmin, dmax, seed):
    """Return {n_priors: labeled_df} where each target sequence ends in (code, d) preceded by
    n_priors random queries; d matched per (context, code) across the three settings."""
    frames = {n: [] for n in PRIORS}
    for si, ev in ce["events"].items():
        ctx = ce["contexts"].filter(pl.col("_shard") == si).drop("_shard")
        if not ctx.height:
            continue
        rng = np.random.default_rng(derive_seed(seed, si))
        rows = {n: [] for n in PRIORS}
        cid = 0
        for c in ctx.iter_rows(named=True):
            subj, t = c["subject_id"], c["prediction_time"]
            for code in codes:
                d = float(log_uniform_durations(1, dmin, dmax, rng)[0])
                # a pool of random fillers (codes + durations) for this (context, code)
                fcodes = [vocab[int(rng.integers(len(vocab)))] for _ in range(max(PRIORS))]
                fdurs = log_uniform_durations(max(PRIORS), dmin, dmax, rng).tolist()
                for n in PRIORS:
                    seq = [(fcodes[j], fdurs[j]) for j in range(n)] + [(code, d)]
                    cidn = cid  # same cid space per setting is fine (separate frames)
                    for pos, (cc, dd) in enumerate(seq):
                        rows[n].append((cidn, pos, subj, t, cc, float(dd)))
                cid += 1
        for n in PRIORS:
            idf = pl.DataFrame(
                rows[n], orient="row",
                schema=[CTX_ID_COL, POSITION_COL, "subject_id", "prediction_time", "query", "duration_days"],
            ).with_columns(
                pl.col(CTX_ID_COL).cast(pl.UInt32), pl.col(POSITION_COL).cast(pl.Int64),
                pl.col("prediction_time").cast(pl.Datetime("us")), pl.col("duration_days").cast(pl.Float32),
            )
            frames[n].append(label_binary_occurrence(idf, ev))
    return {n: pl.concat(frames[n], how="vertical") for n in PRIORS}


def write_tasks(df, d):
    d = Path(d); (d / "held_out").mkdir(parents=True, exist_ok=True)
    from every_query.data.schema import QuerySeqSchema
    pl.from_arrow(QuerySeqSchema.align(df.to_arrow())).write_parquet(d / "held_out" / "tasks.parquet")
    return d


def per_code_auroc(df):
    """df has columns target_query, true_answer, prob. Returns {code: (auroc, n, n_pos)}."""
    out = {}
    for (code,), g in df.group_by(["target_query"]):
        y = g["true_answer"].to_list()
        out[code] = (_auroc(y, g["prob"].to_list()), len(y), int(sum(y)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--cohort-dir", required=True, type=Path)
    ap.add_argument("--intermediate", required=True, type=Path)
    ap.add_argument("--processed", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--split", default="held_out")
    ap.add_argument("--dmin", type=int, default=1)
    ap.add_argument("--dmax", type=int, default=365)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--boot", type=int, default=1000)
    add_context_sampling_args(ap, default_n_contexts=2048)
    args = ap.parse_args()

    model = load_model(args.run_dir)
    codes = curated_codes(args.processed)
    vocab = read_query_codes(args.processed)
    ce = contexts_from_args(args, args.out.parent, seed=11)
    print(f"{len(codes)} codes, {ce['contexts'].height} contexts")

    sets = build_sets(ce, codes, vocab, args.dmin, args.dmax, seed=101)
    work = args.out.parent / "_poseffect_tasks"
    scored = {}
    for n, df in sets.items():
        td = write_tasks(df, work / f"p{n}")
        scored[n] = score_last(model, args.cohort_dir, td, args.split, batch_size=args.batch_size)
        print(f"  scored {n} priors: {scored[n].height} target rows")

    # per-code AUROC at each prior count
    pc = {n: per_code_auroc(scored[n]) for n in PRIORS}
    common = [c for c in codes if all(pc[n].get(c, (None,))[0] is not None for n in PRIORS)]
    rows = []
    for c in common:
        row = {"code": c, "n": pc[0][c][1], "prevalence": pc[0][c][2] / pc[0][c][1]}
        for n in PRIORS:
            row[f"auroc_{n}prior"] = pc[n][c][0]
        rows.append(row)
    tbl = pl.DataFrame(rows).sort("code")
    print(tbl)

    macro = {n: float(np.mean([pc[n][c][0] for c in common])) for n in PRIORS}
    delta = macro[4] - macro[0]

    # bootstrap CI on macro delta (resample codes — the unit we macro over)
    rng = np.random.default_rng(0)
    arr0 = np.array([pc[0][c][0] for c in common])
    arr4 = np.array([pc[4][c][0] for c in common])
    boot = []
    for _ in range(args.boot):
        idx = rng.integers(0, len(common), len(common))
        boot.append(arr4[idx].mean() - arr0[idx].mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])

    summary = {
        "n_codes": len(common), "n_contexts": ce["contexts"].height,
        "macro_auroc_by_priors": macro,
        "delta_4_minus_0": delta, "delta_ci95": [float(lo), float(hi)],
        "per_code": tbl.to_dicts(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nMACRO within-code AUROC: 0 priors={macro[0]:.4f}  2 priors={macro[2]:.4f}  4 priors={macro[4]:.4f}")
    print(f"DELTA (4 priors - 0 priors) = {delta:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"({'significant' if lo > 0 else 'NOT significant (CI crosses 0)'})")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
