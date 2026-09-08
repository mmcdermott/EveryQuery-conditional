#!/usr/bin/env python
"""Training-side check: the retrained small models (old code x2, dev code x1) vs each other and vs the
stored 2026-09-05 small run.

Reads each run's final logged losses from the offline W&B summary (aggregates only) and the per-task
AUROCs of every retrained checkpoint on the canonical tuning grid (all scored through the dev CLI at
its default precision, so only training differs).  The old-vs-old pair is the run-to-run noise floor
of GPU training; the old-vs-new pairs are the refactor's fit path.  Writes
``metrics/train_check.json`` and ``metrics/train_check.md``.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl

EXP = Path("/home/gkondas/eq-new-exps/dev-lightning-refactor-check-in")
OLD_EXP = Path("/home/gkondas/eq-new-exps/multitask-canonical-eval")
RUNS = {
    "small_old_a": EXP / "runs" / "small_old_a",
    "small_old_b": EXP / "runs" / "small_old_b",
    "small_new_a": EXP / "runs" / "small_new_a",
    "small_stored": OLD_EXP / "runs" / "small",
}
PAIRS = [
    ("small_old_a", "small_old_b", "old code vs old code, same seed (run-to-run noise floor)"),
    ("small_new_a", "small_old_a", "dev code vs old code, same seed"),
    ("small_new_a", "small_old_b", "dev code vs old code (second run), same seed"),
    ("small_new_a", "small_stored", "dev code vs the stored 2026-09-05 small run"),
    ("small_old_a", "small_stored", "old code today vs the stored 2026-09-05 small run"),
]


def run_dir(root: Path) -> Path:
    return sorted(root.glob("*/*/"))[-1]


def logged_losses(name: str) -> dict:
    """The validation-loss trajectory and final epoch train loss, read from the stage log's progress bar.

    Offline W&B runs write no summary JSON, so the values are regex-extracted from the training
    stage's log (``logs/50_train_<run>.log`` here, the old suite's ``logs/30_train_small.log`` for the
    stored run).  Only the loss values are read; nothing else from the log is touched.
    """
    import re

    log = OLD_EXP / "logs" / "30_train_small.log" if name == "small_stored" else EXP / "logs" / f"50_train_{name}.log"
    if not log.is_file():
        return {}
    text = log.read_text(errors="replace")
    tuning = [float(x) for x in re.findall(r"tuning/loss=([0-9.e-]+)", text)]
    traj = [v for i, v in enumerate(tuning) if i == 0 or v != tuning[i - 1]]
    train_epoch = re.findall(r"train/loss_epoch=([0-9.e-]+)", text)
    return {
        "tuning/loss": tuning[-1] if tuning else float("nan"),
        "tuning/loss_trajectory": traj,
        "train/loss_epoch": float(train_epoch[-1]) if train_epoch else float("nan"),
    }


def n_params(rd: Path) -> int | None:
    import torch

    ckpt = rd / "best_model.ckpt"
    if not ckpt.is_file():
        return None
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    return int(sum(v.numel() for v in sd.values()))


def task_metrics(name: str) -> pl.DataFrame | None:
    size = "small" if name == "small_stored" else name
    p = EXP / "metrics" / "new_bf16" / "canonical_tuning" / size / "task_metrics.parquet"
    return pl.read_parquet(p).sort("task_index") if p.is_file() else None


def main() -> None:
    out = {"runs": {}, "pairs": {}}
    for name, root in RUNS.items():
        if not root.is_dir() or not list(root.glob("*/*/")):
            continue
        rd = run_dir(root)
        t = task_metrics(name)
        out["runs"][name] = {
            "run_dir": str(rd),
            "final_logged": logged_losses(name),
            "n_params": n_params(rd),
            "macro_auroc": float(np.nanmean(t["auroc"].to_numpy())) if t is not None else None,
            "median_auroc": float(np.nanmedian(t["auroc"].to_numpy())) if t is not None else None,
            "macro_auprc": float(np.nanmean(t["auprc"].to_numpy())) if t is not None else None,
        }
    for a, b, why in PAIRS:
        ta, tb = task_metrics(a), task_metrics(b)
        if ta is None or tb is None:
            continue
        d = ta["auroc"].to_numpy() - tb["auroc"].to_numpy()
        d = d[np.isfinite(d)]
        boots = np.random.default_rng(0).choice(d, size=(5000, d.size), replace=True).mean(axis=1)
        out["pairs"][f"{a} - {b}"] = {
            "why": why,
            "delta_macro_auroc": float(d.mean()),
            "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "max_abs_dauroc_task": float(np.abs(d).max()),
            "mean_abs_dauroc_task": float(np.abs(d).mean()),
            "tasks_a_higher": int((d > 0).sum()),
            "tasks_b_higher": int((d < 0).sum()),
            "n_tasks": int(d.size),
        }
    (EXP / "metrics" / "train_check.json").write_text(json.dumps(out, indent=2))
    md = ["# Training-side check: retrained small models (canonical grid, tuning split, dev CLI bf16-mixed)", ""]
    md += ["| run | params | tuning/loss trajectory (4 validations) | final train/loss_epoch | macro AUROC | median AUROC | macro AUPRC |", "|---|---|---|---|---|---|---|"]
    for name, r in out["runs"].items():
        fl = r["final_logged"]
        traj = " -> ".join(f"{v:.4f}" for v in fl.get("tuning/loss_trajectory", [])) or "n/a"
        trl = fl.get("train/loss_epoch", float("nan"))
        md.append(
            f"| {name} | {r['n_params']} | {traj} | {trl:.4f} | "
            f"{(r['macro_auroc'] if r['macro_auroc'] is not None else float('nan')):.4f} | "
            f"{(r['median_auroc'] if r['median_auroc'] is not None else float('nan')):.4f} | "
            f"{(r['macro_auprc'] if r['macro_auprc'] is not None else float('nan')):.4f} |"
        )
    md += ["", "| pair | why | d macro AUROC [95% CI over tasks] | max abs dAUROC (task) | mean abs dAUROC (task) | a higher / b higher |", "|---|---|---|---|---|---|"]
    for k, p in out["pairs"].items():
        md.append(
            f"| {k} | {p['why']} | {p['delta_macro_auroc']:+.4f} [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}] | "
            f"{p['max_abs_dauroc_task']:.4f} | {p['mean_abs_dauroc_task']:.4f} | {p['tasks_a_higher']}/{p['tasks_b_higher']} of {p['n_tasks']} |"
        )
    (EXP / "metrics" / "train_check.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
