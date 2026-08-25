"""Per-task and per-category macro AUROC for the designed evaluation grid.

`EQ_evaluate_sequences` has no macro-average, no support gate, and does not group `by_query` by
position -- so at length 3 it pools the task under test (position 2) with the random filler
queries at positions 0 and 1.  This script does the grouping the experiment actually needs.

Row identity: the eval grid is context-major, so sequence k = (contexts[k // N], specs[k % N]).
The predictions parquet is one row per (sequence, position) in that same order, so the sequence
ordinal is `(position == 0).cum_sum() - 1`.  That reconstruction is ASSERTED against the spec
YAML -- every row's `query` must equal the code the spec names at that position -- because a
single dropped label row would shift every downstream assignment silently.

Prints spec NAMES (dur_00 ... anc_19) and metrics only; never code strings.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from sklearn.metrics import roc_auc_score

MIN_POS = 10
MIN_NEG = 10


def auroc_ci(y: np.ndarray, s: np.ndarray) -> tuple[float, float, float]:
    """AUROC with a Hanley-McNeil standard error and a normal 95% CI."""
    a = float(roc_auc_score(y, s))
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    q1 = a / (2.0 - a)
    q2 = 2.0 * a * a / (1.0 + a)
    se = math.sqrt(max((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0), 0.0))
    return a, a - 1.96 * se, a + 1.96 * se


def score_one(tag: str, preds_path: Path, spec_path: Path, manifest: pl.DataFrame, out_dir: Path) -> dict:
    specs: dict[str, list] = yaml.safe_load(spec_path.read_text())
    names = sorted(specs)
    n_specs = len(names)
    seq_len = len(specs[names[0]])
    assert all(len(v) == seq_len for v in specs.values()), "ragged spec file"

    p = pl.read_parquet(preds_path)
    print(f"\n{'='*70}\n{tag}: rows={p.height} n_specs={n_specs} seq_len={seq_len}")
    print(f"columns: {p.columns}")

    # --- reconstruct sequence ordinal and spec identity -------------------------------------
    p = p.with_columns(((pl.col("position") == 0).cum_sum() - 1).alias("seq_ord"))
    n_seq = p["seq_ord"].max() + 1
    print(f"sequences={n_seq}  contexts={n_seq // n_specs}  rows/seq={p.height / n_seq:.3f}")
    if p.height != n_seq * seq_len:
        print(f"  !! row count {p.height} != n_seq*seq_len {n_seq * seq_len} — grid is ragged")
    p = p.with_columns((pl.col("seq_ord") % n_specs).alias("spec_idx"))

    # --- ASSERT the reconstruction against the spec file --------------------------------------
    expected = pl.DataFrame(
        {
            "spec_idx": [i for i in range(n_specs) for _ in range(seq_len)],
            "position": [j for _ in range(n_specs) for j in range(seq_len)],
            "expected_query": [specs[names[i]][j][0] for i in range(n_specs) for j in range(seq_len)],
        }
    )
    chk = p.join(expected, on=["spec_idx", "position"], how="left")
    match_rate = float((chk["query"] == chk["expected_query"]).mean())
    print(f"spec-reconstruction match rate: {match_rate:.6f}  (must be 1.0)")
    if match_rate < 1.0:
        print("  !! ABORTING scoring for this file — row identity is not trustworthy")
        return {"tag": tag, "error": "spec reconstruction failed", "match_rate": match_rate}

    # --- the task under test is the LAST position ---------------------------------------------
    tgt = p.filter(pl.col("position") == seq_len - 1)

    rows = []
    for (idx,), g in tgt.group_by(["spec_idx"], maintain_order=True):
        y = np.asarray(g["answer"].to_list(), dtype=int)
        s = np.asarray(g["answer_prob"].to_list(), dtype=float)
        npos, nneg = int(y.sum()), int(len(y) - y.sum())
        rec = {
            "spec": names[idx],
            "category": names[idx].split("_")[0],
            "n": len(y),
            "n_pos": npos,
            "prevalence": npos / len(y) if len(y) else float("nan"),
            "auroc": None,
            "ci_lo": None,
            "ci_hi": None,
        }
        if npos >= MIN_POS and nneg >= MIN_NEG:
            a, lo, hi = auroc_ci(y, s)
            rec.update(auroc=a, ci_lo=lo, ci_hi=hi)
        rows.append(rec)

    t = pl.DataFrame(rows).sort("spec")
    t.write_parquet(out_dir / f"by_task_{tag}.parquet")
    t.write_csv(out_dir / f"by_task_{tag}.csv")

    # --- per-category macro -------------------------------------------------------------------
    summary = {}
    for cat in ("dur", "evt", "anc"):
        sub = t.filter(pl.col("category") == cat)
        scored = sub.filter(pl.col("auroc").is_not_null())
        a = scored["auroc"].to_numpy() if scored.height else np.array([])
        n_above = int((a > 0.5).sum()) if a.size else 0
        # sign test vs the 0.5 coin-flip null
        pval = None
        if a.size:
            from scipy.stats import binomtest

            pval = float(binomtest(n_above, a.size, 0.5, alternative="two-sided").pvalue)
        summary[cat] = {
            "n_total": sub.height,
            "n_scored": scored.height,
            "macro_auroc": float(a.mean()) if a.size else None,
            "median_auroc": float(np.median(a)) if a.size else None,
            "min_auroc": float(a.min()) if a.size else None,
            "max_auroc": float(a.max()) if a.size else None,
            "n_above_0.5": n_above,
            "sign_test_p": pval,
            "n_ci_excludes_0.5": int(
                ((scored["ci_lo"].to_numpy() > 0.5) | (scored["ci_hi"].to_numpy() < 0.5)).sum()
            )
            if scored.height
            else 0,
            "mean_prevalence": float(sub["prevalence"].mean()),
        }

    # --- per-position AUROC (pooled) — shows the conditioning effect at length 3 --------------
    per_pos = []
    for pos in range(seq_len):
        g = p.filter(pl.col("position") == pos)
        y = np.asarray(g["answer"].to_list(), dtype=int)
        s = np.asarray(g["answer_prob"].to_list(), dtype=float)
        if y.sum() >= MIN_POS and (len(y) - y.sum()) >= MIN_NEG:
            per_pos.append({"position": pos, "n": len(y), "prevalence": float(y.mean()),
                            "pooled_auroc": float(roc_auc_score(y, s))})
    print("\n--- pooled AUROC by position (base-rate inflated; for dynamics only) ---")
    for r in per_pos:
        print(f"  pos {r['position']}: n={r['n']:>7d} prev={r['prevalence']:.4f} auroc={r['pooled_auroc']:.4f}")

    print(f"\n--- per-task AUROC at the target position ({tag}) ---")
    print(f"{'spec':<10}{'n':>8}{'n_pos':>8}{'prev':>9}{'auroc':>9}{'95% CI':>20}")
    for r in t.iter_rows(named=True):
        a = "n/a" if r["auroc"] is None else f"{r['auroc']:.4f}"
        ci = "" if r["auroc"] is None else f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
        print(f"{r['spec']:<10}{r['n']:>8}{r['n_pos']:>8}{r['prevalence']:>9.4f}{a:>9}{ci:>20}")

    print(f"\n--- category macro summary ({tag}) ---")
    for cat, d in summary.items():
        m = "n/a" if d["macro_auroc"] is None else f"{d['macro_auroc']:.4f}"
        print(f"  {cat}: macro={m} scored={d['n_scored']}/{d['n_total']} "
              f"above0.5={d['n_above_0.5']} ci_excl_0.5={d['n_ci_excludes_0.5']} "
              f"sign_p={d['sign_test_p']} prev={d['mean_prevalence']:.4f}")

    return {"tag": tag, "seq_len": seq_len, "n_sequences": int(n_seq),
            "n_contexts": int(n_seq // n_specs), "match_rate": match_rate,
            "by_position": per_pos, "by_category": summary}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--spec-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    pred_dir, spec_dir, out_dir = Path(args.pred_dir), Path(args.spec_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pl.read_parquet(spec_dir / "task_manifest.parquet")

    results = []
    for tag in ("len1", "len3"):
        pp = pred_dir / f"preds_{tag}.parquet"
        if not pp.exists():
            print(f"{tag}: MISSING {pp}")
            continue
        results.append(score_one(tag, pp, spec_dir / f"designed_{tag}.yaml", manifest, out_dir))

    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))

    # --- headline comparison ------------------------------------------------------------------
    by = {r["tag"]: r.get("by_category", {}) for r in results if "by_category" in r}

    def cell(tag: str, cat: str, key: str) -> str:
        d = by.get(tag, {}).get(cat)
        if not d or d.get(key) is None:
            return "n/a"
        v = d[key]
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    def scored(tag: str, cat: str) -> str:
        d = by.get(tag, {}).get(cat)
        return f"{d['n_scored']}/{d['n_total']}" if d else "n/a"

    print("\n" + "=" * 70)
    print("HEADLINE - macro AUROC at the target position, by category and sequence length")
    print(f"{'category':<14}{'len1':>9}{'len3':>9}{'scored1':>10}{'scored3':>10}{'>0.5 (1/3)':>13}")
    for cat, label in (("dur", "duration"), ("evt", "event-bound"), ("anc", "DAG/ancestor")):
        above = f"{cell('len1', cat, 'n_above_0.5')}/{cell('len3', cat, 'n_above_0.5')}"
        print(
            f"{label:<14}{cell('len1', cat, 'macro_auroc'):>9}{cell('len3', cat, 'macro_auroc'):>9}"
            f"{scored('len1', cat):>10}{scored('len3', cat):>10}{above:>13}"
        )

    print(f"\nwrote {out_dir/'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
