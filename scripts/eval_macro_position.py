#!/usr/bin/env python
"""Macro (per-task) AUC vs sequence position, via the triple-paired estimator.

WHY: pooled AUROC measures "rank any positive above any negative" — dominated by cross-task
base-rate differences.  We want the *per-task* AUC (within a single query), macro-averaged over
tasks, and whether it rises with position (more prior conditioning).  AUC = P(score_pos >
score_neg) (Mann-Whitney), so for one positive/negative patient pair drawn for the SAME query,
1[score_pos > score_neg] is an unbiased estimate of that query's AUC; averaging over many queries
estimates macro-AUC.

DESIGN (fully paired): sample T tasks = queries Q=(code, duration).  For each, draw one positive
context (Q occurs in (t, t+d]) and one negative context.  Place Q at every position p (p random
filler queries before it, their TRUE answers teacher-forced), scoring the SAME pos/neg pair at
each position.  macro_AUC(p) = mean_T 1[score_pos(p) > score_neg(p)].  Bootstrap over tasks gives
a CI on the slope (and Spearman rho) of macro_AUC vs p; the conditioning trend is "confirmed" only
if that CI excludes 0.  Holding query AND patient-pair fixed across positions isolates the effect
of prior-query conditioning with maximum power.
"""

import argparse
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats as sstats

from eval_v2 import (
    add_context_sampling_args,
    contexts_from_args,
    load_model,
    log_uniform_durations,
    score_last,
)
from every_query.generate_tasks.sample_query_sequences import (
    CTX_ID_COL,
    POSITION_COL,
    label_binary_occurrence,
)
from every_query.generate_tasks.sample_tasks import read_query_codes
from every_query.data.schema import QuerySeqSchema
from meds import DataSchema


def occurrence_labels(flat: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    """Per-row binary occurrence: for each (subject_id, prediction_time, query, duration_days) row,
    did the query code occur in (prediction_time, prediction_time + duration_days)?  Same strict-> asof
    join as label_binary_occurrence, but row-wise (no per-sequence aggregation)."""
    sid, time = DataSchema.subject_id_name, DataSchema.time_name
    left = flat.with_row_index("_row").with_columns(
        (pl.col("prediction_time") + pl.duration(microseconds=1)).alias("_pts")
    ).sort(sid, "query", "_pts")
    right = events.rename({DataSchema.code_name: "query"}).select(sid, "query", time).sort(sid, "query", time)
    joined = left.join_asof(right, by=[sid, "query"], left_on="_pts", right_on=time, strategy="forward")
    win_end = pl.col("prediction_time") + pl.duration(days=pl.col("duration_days"))
    return joined.with_columns(
        (pl.col(time).is_not_null() & (pl.col(time) < win_end)).alias("occurred")
    ).sort("_row")


def build_triples_occurrence(ce, n_tasks, dmin, dmax, min_ctx, seed, scheme="patient",
                             max_valid=8000, max_occ=4000):
    """Occurrence-driven triple sampler (covers ALL codes, rare included).

    Two explicit, both-valid label-1 sampling schemes (``scheme``):
      - ``"pair"``   (context-level): pick one positive (patient, prediction_time) pair uniformly
        from ALL such pairs — patients are weighted by how many positive contexts they have.
      - ``"patient"`` (patient-level, default): pick a patient uniformly among those that permit a
        positive, then a positive prediction time for that patient uniformly — each patient counts
        once, so long-stay patients with many positive windows do not dominate.
    The NEGATIVE uses the matching scheme over valid contexts where C does not occur in (t, t+T].
    """
    rng = np.random.default_rng(seed)
    triples = []
    shards = list(ce["events"])
    per_shard = n_tasks // len(shards) + 1
    sid, time = DataSchema.subject_id_name, DataSchema.time_name
    for si in shards:
        if len(triples) >= n_tasks:
            break
        ev = ce["events"][si]
        evx = ev.with_columns(pl.col(time).cum_count().over(sid).alias("_n"))
        V = evx.filter(pl.col("_n") >= min_ctx).select(sid, pl.col(time).alias("t"))
        if V.height == 0:
            continue
        if V.height > max_valid:
            V = V.sample(n=max_valid, seed=int(rng.integers(1 << 31)))
        codes_present = ev[DataSchema.code_name].unique().to_list()
        for _ in range(per_shard):
            if len(triples) >= n_tasks:
                break
            C = codes_present[int(rng.integers(len(codes_present)))]
            T = float(log_uniform_durations(1, dmin, dmax, rng)[0])
            occ = ev.filter(pl.col(DataSchema.code_name) == C).select(sid, pl.col(time).alias("tau"))
            if occ.height > max_occ:
                occ = occ.sample(n=max_occ, seed=int(rng.integers(1 << 31)))
            win = pl.duration(days=T)
            # POSITIVE label-1 contexts: valid (subj, t) with a C-occurrence in (t, t+T].
            pos_ctx = V.join(occ, on=sid).filter(
                (pl.col("t") < pl.col("tau")) & (pl.col("tau") <= pl.col("t") + win)
            ).select(sid, "t").unique()
            if pos_ctx.height == 0:
                continue
            if scheme == "patient":
                ps = pos_ctx[sid].unique()
                subj_p = ps[int(rng.integers(ps.len()))]
                tp = pos_ctx.filter(pl.col(sid) == subj_p)["t"]
                pos_subj, pos_t = subj_p, tp[int(rng.integers(tp.len()))]
            else:  # pair-uniform
                pr = pos_ctx.row(int(rng.integers(pos_ctx.height)), named=True)
                pos_subj, pos_t = pr[sid], pr["t"]
            # NEGATIVE: valid context where C does NOT occur in (t, t+T], matching the scheme.
            neg = None
            for _ in range(40):
                if scheme == "patient":
                    subs = V[sid].unique()
                    s = subs[int(rng.integers(subs.len()))]
                    tv = V.filter(pl.col(sid) == s)["t"]
                    s_t = tv[int(rng.integers(tv.len()))]
                else:
                    vr = V.row(int(rng.integers(V.height)), named=True)
                    s, s_t = vr[sid], vr["t"]
                hit = occ.filter(
                    (pl.col(sid) == s) & (pl.col("tau") > s_t) & (pl.col("tau") <= s_t + timedelta(days=T))
                )
                if hit.height == 0:
                    neg = (s, s_t)
                    break
            if neg is None:
                continue
            triples.append((C, T, pos_subj, pos_t, neg[0], neg[1]))
    return triples


def build_triples(ce, vocab, n_tasks, dmin, dmax, min_pos, min_neg, seed):
    """Find T tasks each with >=1 pos and >=1 neg context; return one (query, pos_ctx, neg_ctx) triple
    per task.  Occurrence is labeled per shard against that shard's contexts (a context's pos/neg for
    a query Q is whether Q's code occurs in (t, t+d])."""
    rng = np.random.default_rng(seed)
    triples = []  # (code, dur, pos_subj, pos_time, neg_subj, neg_time)
    # draw a big candidate query list; we keep those that yield a pos and a neg within a shard
    for si, ev in ce["events"].items():
        if len(triples) >= n_tasks:
            break
        ctx = ce["contexts"].filter(pl.col("_shard") == si).drop("_shard")
        if ctx.height < (min_pos + min_neg):
            continue
        # candidate queries for this shard
        n_cand = max(64, n_tasks // max(1, len(ce["events"])) * 4)
        codes = [vocab[int(rng.integers(len(vocab)))] for _ in range(n_cand)]
        durs = log_uniform_durations(n_cand, dmin, dmax, rng)
        # build (query x context) grid, label per-row occurrence in one pass
        ctx_rows = ctx.select("subject_id", "prediction_time").to_dicts()
        rows = []
        for qi, (code, d) in enumerate(zip(codes, durs, strict=True)):
            for c in ctx_rows:
                rows.append((qi, c["subject_id"], c["prediction_time"], code, float(d)))
        flat = pl.DataFrame(
            rows, orient="row",
            schema=["qi", "subject_id", "prediction_time", "query", "duration_days"],
        ).with_columns(
            pl.col("prediction_time").cast(pl.Datetime("us")), pl.col("duration_days").cast(pl.Float32),
        )
        lab = occurrence_labels(flat, ev)
        for (qi,), g in lab.group_by(["qi"]):
            pos = g.filter(pl.col("occurred")); neg = g.filter(~pl.col("occurred"))
            if pos.height >= min_pos and neg.height >= min_neg:
                pr = pos.row(int(rng.integers(pos.height)), named=True)
                nr = neg.row(int(rng.integers(neg.height)), named=True)
                triples.append((pr["query"], float(pr["duration_days"]),
                                pr["subject_id"], pr["prediction_time"],
                                nr["subject_id"], nr["prediction_time"]))
                if len(triples) >= n_tasks:
                    break
    return triples


def build_position_tasks(triples, max_pos, vocab, dmin, dmax, seed):
    """For each position p, a task parquet with rows ordered [t0_pos, t0_neg, t1_pos, t1_neg, ...],
    each a length-(p+1) sequence [p random fillers, Q]. Returns {p: df} ready for labeling+scoring."""
    rng = np.random.default_rng(seed)
    out = {}
    for p in range(max_pos):
        rows = []
        cid = 0
        for (code, d, ps, pt, ns, nt) in triples:
            fcodes = [vocab[int(rng.integers(len(vocab)))] for _ in range(p)]
            fdurs = log_uniform_durations(p, dmin, dmax, rng).tolist() if p else []
            for (subj, t) in [(ps, pt), (ns, nt)]:
                seq = [(fcodes[j], fdurs[j]) for j in range(p)] + [(code, d)]
                for pos, (cc, dd) in enumerate(seq):
                    rows.append((cid, pos, subj, t, cc, float(dd)))
                cid += 1
        out[p] = pl.DataFrame(
            rows, orient="row",
            schema=[CTX_ID_COL, POSITION_COL, "subject_id", "prediction_time", "query", "duration_days"],
        ).with_columns(
            pl.col(CTX_ID_COL).cast(pl.UInt32), pl.col(POSITION_COL).cast(pl.Int64),
            pl.col("prediction_time").cast(pl.Datetime("us")), pl.col("duration_days").cast(pl.Float32),
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--cohort-dir", required=True, type=Path)
    ap.add_argument("--intermediate", required=True, type=Path)
    ap.add_argument("--processed", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--split", default="held_out")
    ap.add_argument("--n-tasks", type=int, default=2000)
    ap.add_argument("--max-pos", type=int, default=5)
    ap.add_argument("--sampling", choices=["patient", "pair", "context-pool"], default="patient",
                    help="positive/negative sampling scheme: 'patient' (patient-uniform, default), "
                         "'pair' (context-uniform over positive pairs), 'context-pool' (legacy random pool)")
    ap.add_argument("--min-ctx", type=int, default=10)
    ap.add_argument("--min-pos", type=int, default=1)
    ap.add_argument("--min-neg", type=int, default=1)
    ap.add_argument("--dmin", type=int, default=1)
    ap.add_argument("--dmax", type=int, default=365)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--boot", type=int, default=3000)
    add_context_sampling_args(ap, default_n_contexts=4096)
    args = ap.parse_args()

    model = load_model(args.run_dir)
    vocab = read_query_codes(args.processed)
    ce = contexts_from_args(args, args.out.parent, seed=21)
    print(f"sampling triples (scheme={args.sampling}) over {ce['contexts'].height} contexts...")
    if args.sampling == "context-pool":
        triples = build_triples(ce, vocab, args.n_tasks, args.dmin, args.dmax, args.min_pos, args.min_neg, seed=5)
    else:
        triples = build_triples_occurrence(ce, args.n_tasks, args.dmin, args.dmax, args.min_ctx,
                                            seed=5, scheme=args.sampling)
    T = len(triples)
    print(f"got {T} (query, pos-ctx, neg-ctx) triples")

    pos_tasks = build_position_tasks(triples, args.max_pos, vocab, args.dmin, args.dmax, seed=7)
    work = args.out.parent / "_macro_tasks"
    # Label every position's sequences against the events of ONLY the subjects used by the triples
    # (filtering keeps memory bounded — the full held-out event union is tens of millions of rows
    # and would OOM alongside the live training process).
    triple_subjects = set()
    for (_c, _d, ps, _pt, ns, _nt) in triples:
        triple_subjects.add(ps); triple_subjects.add(ns)
    all_ev = (
        pl.concat([ce["events"][s] for s in ce["events"]], how="vertical")
        .filter(pl.col("subject_id").is_in(list(triple_subjects)))
        .unique()
        .sort(["subject_id", "time"])
    )
    print(f"labeling against {all_ev.height:,} events for {len(triple_subjects):,} subjects")
    margins = np.zeros((T, args.max_pos))  # 1 if score_pos > score_neg at that position, else 0
    for p, df in pos_tasks.items():
        labeled = label_binary_occurrence(df, all_ev)
        td = work / f"p{p}"
        (td / args.split).mkdir(parents=True, exist_ok=True)
        pl.from_arrow(QuerySeqSchema.align(labeled.to_arrow())).write_parquet(td / args.split / "tasks.parquet")
        scored = score_last(model, args.cohort_dir, td, args.split, force_prior=None, batch_size=args.batch_size)
        assert scored.height == 2 * T, (
            f"position {p}: expected {2*T} scored rows but got {scored.height} — a context subject was "
            f"dropped by the cohort, breaking the pos/neg interleaving."
        )
        probs = scored["prob"].to_numpy()  # rows ordered [t0_pos, t0_neg, t1_pos, t1_neg, ...]
        margins[:, p] = (probs[0::2] > probs[1::2]).astype(float)
        print(f"  position {p}: macro-AUC = {margins[:, p].mean():.4f}")

    macro = margins.mean(axis=0)
    xs = np.arange(args.max_pos)
    slope = float(np.polyfit(xs, macro, 1)[0])
    rho = float(sstats.spearmanr(xs, macro).statistic)

    rng = np.random.default_rng(0)
    bsl, brho = [], []
    for _ in range(args.boot):
        idx = rng.integers(0, T, T)
        m = margins[idx].mean(axis=0)
        bsl.append(np.polyfit(xs, m, 1)[0]); brho.append(sstats.spearmanr(xs, m).statistic)
    slo, shi = np.percentile(bsl, [2.5, 97.5])
    rlo, rhi = np.nanpercentile(brho, [2.5, 97.5])

    summary = {
        "n_tasks": T, "sampling_scheme": args.sampling,
        "macro_auc_by_position": {int(p): float(macro[p]) for p in xs},
        "slope_per_position": slope, "slope_ci95": [float(slo), float(shi)],
        "spearman_rho": rho, "spearman_ci95": [float(rlo), float(rhi)],
        "slope_significant_positive": bool(slo > 0),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print("\nMACRO (per-task) AUC by position:")
    for p in xs:
        print(f"  pos {p} ({p} priors): {macro[p]:.4f}")
    print(f"slope/pos = {slope:+.5f}  95%CI [{slo:+.5f}, {shi:+.5f}]")
    print(f"Spearman rho(pos, macroAUC) = {rho:+.3f}  95%CI [{rlo:+.3f}, {rhi:+.3f}]")
    print("CONFIRMED positive trend" if slo > 0 else "NOT significant (slope CI includes 0)")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
