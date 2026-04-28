# EveryQuery vs. Autoregressive Trajectory Model — Side-by-Side Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/payalchandak/EveryQuery/blob/eq-vs-ar-demo/paper_experiments/eq_vs_ar_demo/everyquery_demo.ipynb)

A Colab-runnable notebook that compares **EveryQuery** (task-conditioned prediction) to **MEDS-EIC-AR** (trajectory-simulation prediction) on the same MIMIC-IV held-out cohort.

Click the badge above to open the notebook directly in Colab. The notebook itself only hard-codes the (subject_id, prediction_time) pairs to surface; everything else — the helper module, the task definitions, the MIMIC item-id vocabulary — is fetched from this folder at runtime.

## Headline result

EveryQuery wins **85% of held-out (code, duration) tasks** by AUC over the full MIMIC-IV held-out cohort (160 cells with comparable EQ and AR AUCs — mean EQ AUC 0.839, mean AR AUC 0.674, Δ = +0.165 in EQ's favor). The notebook computes this from artifacts at runtime.

## Two latency numbers, both measured

| Model | Per-prediction latency | Source |
| --- | :---: | --- |
| EveryQuery | **0.9–3.1 ms** | Per-cell evaluation wall-clock ÷ per-cell subject count |
| MEDS-EIC-AR (20-trajectory ensemble) | **~6.0 s** (0.30 s × 20) | Trajectory-generation log: 20 trajectory-shards × 15,031 subjects in 24 h 59 min on a single A100 |

The notebook UI offers a 3-way replay-speed toggle (`Real-time (~6 s)` / `Fast (~2 s)` / `Instant`) so you can either feel the actual latency difference in the room or run the demo more punchily without changing the prediction values.

## Files

| File | Purpose |
| --- | --- |
| `everyquery_demo.ipynb` | The Colab notebook. Minimal — hard-codes only patient IDs and prediction times. |
| `demo_helpers.py` | All UI rendering, model classes, vocab translation, and analysis logic. Downloaded from this folder by the notebook at runtime. |
| `tasks.json` | The 8 demo tasks (code, duration, label, question). |
| `mimic_itemid_map.json` | Consolidated MIMIC-IV `itemid → human label` lookup (~2 100 entries) built from MIT-LCP/mimic-code mapping CSVs. |
| `demo_patients.json` | The selected demo cohort (subject IDs, prediction times, ground-truth label vectors) from the offline curation step. |
| `select_demo_patients.py` | Local-only curation script that produces `demo_patients.json`. Not used at notebook runtime. |
| `build_demo_notebook.py` | Source of truth for the `.ipynb`. Run with `uv run --no-project python paper_experiments/eq_vs_ar_demo/build_demo_notebook.py` to regenerate. |
| `README.md` | This file. |

## Running

**Colab:** click the badge at the top. You'll be prompted for an interactive Google auth so the notebook can read the underlying patient data — accept the popup. You need read access to the demo's data location.

**Locally:** open `everyquery_demo.ipynb` in Jupyter / VS Code. Make sure `gcloud auth application-default login` has been run once on this machine.

## Regenerating the notebook

```
uv run --no-project python paper_experiments/eq_vs_ar_demo/build_demo_notebook.py
```

This rewrites `everyquery_demo.ipynb` from `build_demo_notebook.py` with empty cell outputs. Edit the `.py` file, never the `.ipynb` directly.

## Re-running patient curation

```
uv run --no-project --with pandas --with pyarrow --with numpy --with gcsfs \
    python paper_experiments/eq_vs_ar_demo/select_demo_patients.py
```

This rewrites `demo_patients.json` with a fresh selection. After running it, copy the new `subject_id` / `prediction_time_us` pairs into the `DEMO_SUBJECTS` list in `build_demo_notebook.py` and rebuild the notebook.
