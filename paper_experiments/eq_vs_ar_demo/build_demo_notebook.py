"""Emit ``everyquery_demo.ipynb`` — a minimal Colab-runnable demo notebook.

The notebook hard-codes only (subject_id, prediction_time_us) pairs. All other
state — model classes, vocab translation, UI rendering, data loading,
aggregate analysis — lives in ``demo_helpers.py`` and is downloaded from this
GitHub folder at runtime.

Run with:
    uv run --no-project python paper_experiments/eq_vs_ar_demo/build_demo_notebook.py

The committed notebook has empty ``outputs`` lists in every cell so we never
ship generated MIMIC data in version control.
"""

from __future__ import annotations
import json
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "everyquery_demo.ipynb"

# Branch + folder where ``demo_helpers.py`` etc. live. The notebook downloads
# from raw.githubusercontent.com using this prefix.
GH_USER = "payalchandak"
GH_REPO = "EveryQuery"
GH_BRANCH = "eq-vs-ar-demo"
GH_FOLDER = "paper_experiments/eq_vs_ar_demo"


_CELLS: list[dict] = []
_CID = [0]


def _id() -> str:
    _CID[0] += 1
    return f"c{_CID[0]:02d}"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _id(),
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "id": _id(),
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": textwrap.dedent(text).strip("\n").splitlines(keepends=True),
    }


# Selected subjects + their prediction times (µs since epoch).
# These were chosen offline by ``select_demo_patients.py`` — see that script
# for the curation logic. The notebook does not redo curation at runtime.
DEMO_SUBJECTS = [
    (13336307, 6051391200000000),
    (11495932, 6635161440000000),
    (13999026, 5392922400000000),
    (13753717, 6315084000000000),
    (15605355, 5753333700000000),
    (18242530, 6576321600000000),
]


_CELLS.append(md(r"""
# EveryQuery vs. Autoregressive EHR FM — Demo

A side-by-side demo of two ways to answer clinical prediction questions about ICU patients:

- **Task-conditioned model (EveryQuery / "EQ"):** answers the question directly in a single forward pass.
- **Trajectory-simulation model (autoregressive / "AR"):** samples N event trajectories from the current state and tallies outcomes empirically.

The architectural difference is the demo. The latency difference is the headline. Per-task AUC over the full held-out cohort (shown in the scoreboard cell) is the supporting evidence.
"""))

_CELLS.append(md("## 1. Setup"))

_CELLS.append(code(rf"""
# Download helper module + JSON inputs from this folder on GitHub, then import.
import os, sys, urllib.request

_RAW_BASE = "https://raw.githubusercontent.com/{GH_USER}/{GH_REPO}/{GH_BRANCH}/{GH_FOLDER}"
_LOCAL = "/tmp/eq_demo_helpers"
os.makedirs(_LOCAL, exist_ok=True)

for _f in ("demo_helpers.py", "tasks.json", "mimic_itemid_map.json"):
    urllib.request.urlretrieve(f"{{_RAW_BASE}}/{{_f}}", f"{{_LOCAL}}/{{_f}}")

if _LOCAL not in sys.path:
    sys.path.insert(0, _LOCAL)

import demo_helpers
demo_helpers.setup()
"""))

_CELLS.append(md(r"""
## 2. Demo subjects

These are the patients the side-by-side UI will surface. The list is fixed for reproducibility — to add or remove patients, edit `select_demo_patients.py` in the GitHub folder, run it locally, and update this list.
"""))

_subj_lines = ",\n    ".join(f"({sid}, {pt})" for sid, pt in DEMO_SUBJECTS)
_CELLS.append(code(f"""
DEMO_SUBJECTS = [
    {_subj_lines},
]
"""))

_CELLS.append(md(r"""
## 3. Side-by-side comparison

Pick a patient, pick a clinical question, and click **Compare models**. EQ resolves on the right almost instantly; the AR panel streams 20 sampled trajectories on the left, with the running event-occurrence tally and a scrollable list of every trajectory.

Use the *AR replay* toggle to feel real-time latency (~6 s per query) or to fast-forward the demo without changing the predictions.
"""))

_CELLS.append(code("""
demo_helpers.run_demo(DEMO_SUBJECTS)
"""))

_CELLS.append(md(r"""
## 4. Headline result — per-task AUC over the full held-out cohort

The 6-patient slice above is too small to give a faithful AUC read on rare events. The cell below pulls the per-(code, duration) AUCs that were already computed during the original evaluations across thousands of held-out subjects per cell, and reports the EQ-vs-AR head-to-head.
"""))

_CELLS.append(code("""
demo_helpers.show_scoreboard()
"""))

_CELLS.append(md(r"""
## 5. Multi-task efficiency

When the same patient is queried for multiple tasks at once, the AR model amortizes trajectory generation but EQ stays linear and fast. Numbers below use the measured AR per-(subject, trajectory) cost.
"""))

_CELLS.append(code("""
demo_helpers.show_multi_task_costs()
"""))

_CELLS.append(md(r"""
## 6. About this demo

| Component | Source |
| --- | --- |
| EQ probabilities | **Real.** Pre-computed by the 22-layer ModernBERT-EQ best checkpoint on the held-out split. |
| EQ latency | **Measured.** From the per-cell evaluation wall-clock divided by per-cell subject count. ~0.9–3.1 ms per prediction. |
| AR probabilities | **Real.** Empirical event-occurrence rate over 20 independent [MEDS-EIC-AR](https://github.com/mmcdermott/MEDS_EIC_AR) trajectories per patient. |
| AR latency | **Measured.** From the trajectory-generation log. 0.30 s per (subject, trajectory) → ~6.0 s for one patient × 20 trajectories on a single A100 in bf16. |
| Patient histories + ground truth | **Real.** Computed from the MIMIC-IV held-out MEDS extract used to train and evaluate both models. |
| Vocab mapping | MIMIC-IV `d_items` / `d_labitems` mappings from MIT-LCP/mimic-code, packaged as `mimic_itemid_map.json` in this folder. |

**Architecture in two sentences.** EveryQuery answers a clinical question directly by conditioning a transformer on `(patient_history, query_code, duration)` and producing a single probability. Autoregressive EHR foundation models answer the same question by generating N future event trajectories and tallying empirical outcome rates — flexible, but quadratic in tokens and Monte-Carlo-noisy.
"""))


def build() -> None:
    nb = {
        "cells": _CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11", "mimetype": "text/x-python",
                              "file_extension": ".py"},
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1))
    print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes, {len(_CELLS)} cells, all outputs empty)")


if __name__ == "__main__":
    build()
