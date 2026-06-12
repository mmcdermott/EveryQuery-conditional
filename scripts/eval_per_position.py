#!/usr/bin/env python
"""Low-noise per-position AUROC with bootstrap CIs on a fixed-length held-out set.

The per-position pooled AUROC is a valid conditioning metric: every position draws its (code,
duration) from the same iid distribution, so the base-rate inflation is identical across positions
and cancels in the position-to-position comparison; the only thing that differs is the amount of
prior (query, answer) conditioning context, so Bayes error should be monotonically non-increasing
with position (AUROC non-decreasing).  The effect may be weak, so we measure it on the full
held-out set (≫ the 100 val batches used during training) and attach bootstrap CIs per position and
on the linear slope, reporting "confirmed" only if the slope CI excludes 0.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from lightning.fabric.utilities.apply_func import move_data_to_device
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from eval_v2 import load_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_dataset(cohort_dir, tasks_dir, split, max_seq_len=256):
    from meds_torchdata.config import MEDSTorchDataConfig

    from every_query.data.seq_dataset import ConditionalQueryPytorchDataset

    cfg = MEDSTorchDataConfig(
        tensorized_cohort_dir=str(cohort_dir), task_labels_dir=str(tasks_dir), max_seq_len=max_seq_len,
        seq_sampling_strategy="to_end", static_inclusion_mode="omit", batch_mode="SM",
    )
    return ConditionalQueryPytorchDataset(cfg, split=split)


@torch.no_grad()
def predict_all_positions(model, ds, batch_size):
    """Return a long DataFrame (position, true_answer, prob) over every query of every sequence."""
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=ds.collate)
    pos_list, true_list, prob_list, seq_list = [], [], [], []
    seq_off = 0
    for batch in dl:
        batch = move_data_to_device(batch, DEVICE)
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            _, out = model.model(batch)
        probs = out.answer_probs.float().cpu().numpy()
        mask = batch.q_mask.cpu().numpy()
        ans = (batch.q_answers.cpu().numpy() == 1).astype(int)
        B, L = mask.shape
        for i in range(B):
            for j in range(L):
                if mask[i, j]:
                    pos_list.append(j); true_list.append(int(ans[i, j]))
                    prob_list.append(float(probs[i, j])); seq_list.append(seq_off + i)
        seq_off += B
    return pl.DataFrame({"seq": seq_list, "position": pos_list, "true_answer": true_list, "prob": prob_list})


def _auroc(y, p):
    y = np.asarray(y)
    return float(roc_auc_score(y, np.asarray(p))) if len(np.unique(y)) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--cohort-dir", required=True, type=Path)
    ap.add_argument("--tasks-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--split", default="held_out")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    model = load_model(args.run_dir)
    ds = make_dataset(args.cohort_dir, args.tasks_dir, args.split)
    print(f"held-out sequences: {len(ds)} on {DEVICE}")
    df = predict_all_positions(model, ds, args.batch_size)
    positions = sorted(df["position"].unique().to_list())

    # point estimates
    by_pos = {p: df.filter(pl.col("position") == p) for p in positions}
    auroc = {p: _auroc(by_pos[p]["true_answer"].to_list(), by_pos[p]["prob"].to_list()) for p in positions}
    n_per = {p: by_pos[p].height for p in positions}
    prev = {p: float(np.mean(by_pos[p]["true_answer"].to_list())) for p in positions}

    # bootstrap over SEQUENCES (resample whole sequences so positions stay correlated within a seq)
    seqs = df["seq"].unique().to_numpy()
    # pre-split arrays per position keyed by seq for fast resampling
    rng = np.random.default_rng(0)
    # build per-position (seq -> (y,p)) via pandas-like grouping
    pos_arrays = {}
    for p in positions:
        g = by_pos[p]
        pos_arrays[p] = (g["seq"].to_numpy(), g["true_answer"].to_numpy(), g["prob"].to_numpy())

    boot_auroc = {p: [] for p in positions}
    boot_slope = []
    n_seq = len(seqs)
    seq_index = {s: i for i, s in enumerate(seqs)}
    # map each position's rows to a resample-able structure: for a bootstrap of sequences, select rows whose seq is in the sample.
    # Efficient approach: for each position, sort by seq and use np.isin on a sampled seq set is O(n*boot) -> too slow.
    # Instead: since fixed-length, every seq has exactly one row per position in order; build matrix [n_seq, n_pos].
    mat_true = np.full((n_seq, len(positions)), np.nan)
    mat_prob = np.full((n_seq, len(positions)), np.nan)
    pcol = {p: k for k, p in enumerate(positions)}
    s_idx = df["seq"].to_numpy()
    p_idx = df["position"].to_numpy()
    y_all = df["true_answer"].to_numpy()
    q_all = df["prob"].to_numpy()
    remap = np.vectorize(seq_index.get)(s_idx)
    for r in range(len(df)):
        mat_true[remap[r], pcol[p_idx[r]]] = y_all[r]
        mat_prob[remap[r], pcol[p_idx[r]]] = q_all[r]

    xs = np.array(positions, dtype=float)
    for _ in range(args.boot):
        samp = rng.integers(0, n_seq, n_seq)
        a = []
        for k, p in enumerate(positions):
            yt = mat_true[samp, k]; pp = mat_prob[samp, k]
            ok = ~np.isnan(yt)
            au = _auroc(yt[ok], pp[ok])
            boot_auroc[p].append(au); a.append(au)
        boot_slope.append(np.polyfit(xs, a, 1)[0])

    ci = {p: [float(np.nanpercentile(boot_auroc[p], 2.5)), float(np.nanpercentile(boot_auroc[p], 97.5))] for p in positions}
    slope_pt = float(np.polyfit(xs, [auroc[p] for p in positions], 1)[0])
    slope_lo, slope_hi = np.nanpercentile(boot_slope, [2.5, 97.5])

    summary = {
        "n_sequences": len(ds), "positions": positions,
        "auroc": {int(p): auroc[p] for p in positions},
        "auroc_ci95": {int(p): ci[p] for p in positions},
        "n_per_position": {int(p): n_per[p] for p in positions},
        "prevalence": {int(p): prev[p] for p in positions},
        "slope_per_position": slope_pt,
        "slope_ci95": [float(slope_lo), float(slope_hi)],
        "slope_significant_positive": bool(slope_lo > 0),
        "delta_lastpos_minus_pos0": auroc[positions[-1]] - auroc[positions[0]],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print("\nper-position AUROC (n/pos = %d):" % n_per[positions[0]])
    for p in positions:
        print(f"  pos {p}: {auroc[p]:.4f}  95%CI [{ci[p][0]:.4f}, {ci[p][1]:.4f}]  prev={prev[p]:.3f}")
    print(f"slope/position = {slope_pt:+.5f}  95%CI [{slope_lo:+.5f}, {slope_hi:+.5f}]  "
          f"{'SIGNIFICANT (+)' if slope_lo>0 else 'not significant (CI includes 0)'}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
