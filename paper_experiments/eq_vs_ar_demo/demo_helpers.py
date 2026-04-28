"""Helper module for the EveryQuery vs AR demo notebook.

The notebook stays minimal and just hard-codes (subject_id, prediction_time)
pairs. All UI rendering, data loading, model interfaces, vocab translation,
and analysis live here. The notebook downloads this file from GitHub at
runtime so changes here propagate without re-uploading the .ipynb.

Public entry points the notebook calls:
    setup()                   — interactive auth + data-loading bootstrap
    run_demo(subjects)        — render the side-by-side patient/task UI
    show_scoreboard()         — per-task AUC over the full held-out cohort
    show_multi_task_costs()   — multi-task efficiency table
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# Asset locations (private to this module — the notebook doesn't see these)
# ---------------------------------------------------------------------------
_BUCKET = "every-query-runs"
_EQ_RUN_DIR = "outputs/2026-04-02/23-43-54"
_EQ_PREDS_PATH = f"{_EQ_RUN_DIR}/eval/best/eval_preds_10000_ID__8db2be6fadf8_20260413_201311.parquet"
_EQ_AUC_PATH = f"{_EQ_RUN_DIR}/eval/best/eval_aucs_held_out_20260410_192213.parquet"
_MEDS_HELDOUT_TEMPLATE = "meds/MIMIC_MEDS/MEDS_cohort/intermediate/data/held_out/{shard}.parquet"
_MEDS_SHARDS = [0, 1, 2]
_AR_TRAJ_DIR = "eic/trajectories/medium_seq_1024/7573f855c4b050a9d79d57fefd8a139c/held_out"
_AR_TRAJ_SHARDS = list(range(20))
_PREVALENCE_PATH = "held_out_prevalence_7573f855c4b050a9d79d57fefd8a139c.csv"
_AR_AUC_PATH = "baselines/eic/temporal_auc_results_10000_ID__8db2be6fadf8.parquet"

_HERE = Path(__file__).resolve().parent
_CACHE = Path("/tmp/eq_demo_cache")

# Lazy-loaded module-globals populated by setup()
_state: dict[str, Any] = {}


def _pip(*pkgs: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])


def setup() -> None:
    """Install runtime deps, authenticate, and warm caches.

    Safe to call repeatedly. In Colab it triggers ``google.colab`` interactive
    auth; otherwise it relies on ``gcloud auth application-default login``.
    """
    try:
        import gcsfs, pandas, ipywidgets, numpy  # noqa: F401
    except ImportError:
        _pip("gcsfs", "pandas", "pyarrow", "ipywidgets", "numpy", "scikit-learn")

    if "google.colab" in sys.modules:
        from google.colab import auth as colab_auth
        colab_auth.authenticate_user()

    import gcsfs
    import pandas as pd

    _state["fs"] = gcsfs.GCSFileSystem()
    _CACHE.mkdir(parents=True, exist_ok=True)

    # Eagerly load the small assets the notebook needs.
    _state["preds_df"] = _gread_parquet(_EQ_PREDS_PATH)
    _state["auc_df"] = _gread_parquet(_EQ_AUC_PATH)
    _state["prevalence_df"] = _gread_csv(_PREVALENCE_PATH)
    _state["base_rate"] = (
        _state["prevalence_df"]
        .assign(code=lambda d: d["code"].astype(str))
        .set_index(["code", "duration_days"])["prevalence"]
        .to_dict()
    )
    _state["tasks"] = _load_tasks()
    _state["itemid_map"] = _load_itemid_map()
    print(f"Ready. {len(_state['tasks'])} demo tasks loaded.")


def _gread_parquet(path: str):
    import pandas as pd
    cache = _CACHE / path.replace("/", "__")
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        with _state["fs"].open(f"{_BUCKET}/{path}", "rb") as src, cache.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return pd.read_parquet(cache)


def _gread_csv(path: str):
    import pandas as pd
    cache = _CACHE / path.replace("/", "__")
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        with _state["fs"].open(f"{_BUCKET}/{path}", "rb") as src, cache.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return pd.read_csv(cache)


def _load_tasks() -> list["Task"]:
    """Load tasks.json from the same directory as this module."""
    p = _HERE / "tasks.json"
    if not p.exists():
        # Running in Colab — file was downloaded next to demo_helpers.py
        p = Path(sys.path[0]) / "tasks.json"
    data = json.loads(p.read_text())
    return [Task(**t) for t in data["tasks"]]


def _load_itemid_map() -> dict:
    p = _HERE / "mimic_itemid_map.json"
    if not p.exists():
        p = Path(sys.path[0]) / "mimic_itemid_map.json"
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Task:
    code: str
    duration_days: int
    label: str
    question: str
    valence: str

    @property
    def id(self) -> str:
        return self.code


@dataclass
class PatientEvent:
    t_hours: float
    kind: str
    description: str


@dataclass
class DemoPatient:
    id: str
    age: Optional[int]
    sex: Optional[str]
    primary_issue: str
    icu_day: Optional[int]
    summary: str
    history: list = field(default_factory=list)
    prediction_time_us: int = 0
    ground_truth: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Vocab mapping (MEDS code → human-readable string)
# ---------------------------------------------------------------------------
_ICD9_DX = {
    "3320": "Parkinson's disease",
    "4271": "Paroxysmal supraventricular tachycardia",
    "5856": "End-stage renal disease (CKD stage V)",
    "7295": "Pain in limb / soft-tissue pain",
}
_ICD10_DX = {"I25118": "Atherosclerotic heart disease, native coronary, unstable angina"}
_ICD9_PROC = {"7936": "Open reduction of fracture without internal fixation"}


def _parse_bin(parts: list[str]) -> str:
    for p in parts:
        if p.startswith("value_["):
            inner = p[len("value_["):]
            return f"in {inner.replace(',', ', ')}"
    return ""


def humanize(code: str) -> str:
    """Return a clinician-readable rendering of a MEDS code."""
    if code == "MEDS_DEATH":
        return "Death (in-hospital or out-of-hospital)"
    if code == "MEDS_BIRTH":
        return "Birth"
    if code.startswith("TIMELINE//"):
        return "Timeline marker"

    parts = code.split("//")
    head = parts[0]
    bin_str = _parse_bin(parts)
    item_map: dict = _state.get("itemid_map", {})

    if head == "DIAGNOSIS" and len(parts) >= 4 and parts[1] == "ICD":
        version, dxcode = parts[2], parts[3]
        table = _ICD9_DX if version == "9" else _ICD10_DX if version == "10" else {}
        return f"Diagnosis: {table.get(dxcode, f'ICD-{version} {dxcode}')}"
    if head == "PROCEDURE" and len(parts) >= 4 and parts[1] == "ICD":
        version, pxcode = parts[2], parts[3]
        table = _ICD9_PROC if version == "9" else {}
        return f"Procedure: {table.get(pxcode, f'ICD-{version} procedure {pxcode}')}"
    if head in {"INFUSION_START", "INFUSION_END"} and len(parts) >= 2:
        item = parts[1]
        name = item_map.get(item, f"itemid {item}")
        verb = "started" if head == "INFUSION_START" else "ended"
        return f"{name} infusion {verb} {bin_str}".strip()
    if head == "LAB" and len(parts) >= 2:
        item = parts[1]
        name = item_map.get(item, f"itemid {item}")
        return f"Lab — {name} {bin_str}".strip()
    if head == "SUBJECT_FLUID_OUTPUT" and len(parts) >= 2:
        item = parts[1]
        name = item_map.get(item, f"itemid {item}")
        return f"Fluid output — {name} {bin_str}".strip()
    if head == "MEDICATION":
        return f"Medication: {' '.join(parts[1:]).strip()}"
    if head in {"HOSPITAL_ADMISSION", "HOSPITAL_DISCHARGE", "ICU_ADMISSION", "ICU_DISCHARGE"}:
        return head.replace("_", " ").title()
    if head in {"GENDER", "Weight", "Weight (Lbs)", "Height", "Blood Pressure"}:
        return head
    return code


def _kind_for(code: str) -> str:
    head = code.split("//", 1)[0]
    return {
        "DIAGNOSIS": "diagnosis", "PROCEDURE": "procedure", "MEDICATION": "med",
        "INFUSION_START": "med", "INFUSION_END": "med", "LAB": "lab",
        "SUBJECT_FLUID_OUTPUT": "lab", "MEDS_DEATH": "diagnosis",
        "ICU_ADMISSION": "note", "ICU_DISCHARGE": "note",
        "HOSPITAL_ADMISSION": "note", "HOSPITAL_DISCHARGE": "note",
    }.get(head, "note")


# ---------------------------------------------------------------------------
# Patient assembly
# ---------------------------------------------------------------------------
def _shard_for_subjects(sids: list[int]):
    """Concatenate the small set of MEDS held-out shards we need for these subjects."""
    import pandas as pd
    if "_meds_shards" not in _state:
        frames = [_gread_parquet(_MEDS_HELDOUT_TEMPLATE.format(shard=i)) for i in _MEDS_SHARDS]
        _state["_meds_shards"] = pd.concat(frames, ignore_index=True)
    full = _state["_meds_shards"]
    return full[full["subject_id"].isin(sids)].copy()


def _events_by_subject(sids: list[int]):
    df = _shard_for_subjects(sids)
    return {sid: g for sid, g in df.sort_values(["subject_id", "time"]).groupby("subject_id", sort=False)}


def _compute_ground_truth(events, prediction_time_us: int, code: str, duration_days: int) -> int:
    import pandas as pd
    pt = pd.Timestamp(prediction_time_us, unit="us")
    end = pt + pd.Timedelta(days=duration_days)
    return int(((events["time"] > pt) & (events["time"] <= end) & (events["code"] == code)).any())


def _generate_narrative(events_pre) -> tuple[str, str]:
    diagnoses = [c for c in events_pre["code"] if c.startswith("DIAGNOSIS//")]
    infusions = [c for c in events_pre["code"] if c.startswith("INFUSION_START//")]
    medications = [c for c in events_pre["code"] if c.startswith("MEDICATION//")]
    labs = [c for c in events_pre["code"] if c.startswith("LAB//")]
    icu = [c for c in events_pre["code"] if c.startswith("ICU_ADMISSION")]
    hosp = [c for c in events_pre["code"] if c.startswith("HOSPITAL_ADMISSION")]

    seen, recent_dx = set(), []
    for c in reversed(diagnoses):
        if c in seen:
            continue
        seen.add(c)
        recent_dx.append(c)
        if len(recent_dx) >= 3:
            break

    primary = "Complex critical illness"
    if recent_dx:
        cleaned = humanize(recent_dx[0]).replace("Diagnosis: ", "")
        primary = cleaned[:60] if len(cleaned) <= 60 else cleaned[:57] + "..."

    parts = [
        f"{len(events_pre):,} events on file across {len(hosp)} hospital admission(s) and {len(icu)} ICU admission(s).",
    ]
    if recent_dx:
        parts.append(
            "Recent diagnoses: " + "; ".join(humanize(c).replace("Diagnosis: ", "") for c in recent_dx) + "."
        )
    if infusions:
        parts.append(f"{len(infusions)} infusion start events; most recent: {humanize(infusions[-1])}.")
    if medications:
        parts.append(f"{len(set(medications))} distinct medications administered.")
    if labs:
        parts.append(f"{len(labs):,} lab values recorded in the available history.")
    return primary, " ".join(parts)


def build_demo_patient(subject_id: int, prediction_time_us: int) -> DemoPatient:
    import pandas as pd
    events = _events_by_subject([subject_id])[subject_id].copy()
    pt = pd.Timestamp(prediction_time_us, unit="us")
    pre = events[events["time"] <= pt].copy()

    age = None
    if (events["code"] == "MEDS_BIRTH").any():
        b = events[events["code"] == "MEDS_BIRTH"].iloc[0]
        age = int((pt - b["time"]).total_seconds() / (365.25 * 86400))
    sex = None
    sex_rows = events[events["code"].str.startswith("GENDER//", na=False)]
    if not sex_rows.empty:
        sex = sex_rows["code"].iloc[0].split("//", 1)[1][:1]
    icu_admits = events[events["code"].str.startswith("ICU_ADMISSION", na=False) & (events["time"] <= pt)]
    icu_day = None
    if not icu_admits.empty:
        last = icu_admits.iloc[-1]["time"]
        icu_day = int((pt - last).total_seconds() // 86400) + 1

    primary, narrative = _generate_narrative(pre)

    history = []
    for _, row in pre.tail(25).iterrows():
        delta_h = (row["time"] - pt).total_seconds() / 3600.0
        history.append(PatientEvent(t_hours=float(delta_h), kind=_kind_for(row["code"]),
                                    description=humanize(row["code"])))

    gt = {(t.id, t.duration_days): _compute_ground_truth(events, prediction_time_us, t.id, t.duration_days)
          for t in _state["tasks"]}

    return DemoPatient(
        id=f"H{subject_id}",
        age=age,
        sex=sex,
        primary_issue=primary,
        icu_day=icu_day,
        summary=narrative,
        history=history,
        prediction_time_us=int(prediction_time_us),
        ground_truth=gt,
    )


# ---------------------------------------------------------------------------
# Models — both replay real artifacts; latency numbers are measured.
# ---------------------------------------------------------------------------
class _ReplayEQModel:
    def __init__(self):
        import pandas as pd
        preds = _state["preds_df"]
        self._preds = preds.set_index(["subject_id", "code", "duration_days"])["occurs_probs"]
        cell_size = preds.groupby(["code", "duration_days"]).size()
        self._latency_per_pred = (
            _state["auc_df"].set_index(["code", "duration_days"])["eval_time"] / cell_size
        ).rename("latency_s")

    def predict(self, patient: DemoPatient, task: Task) -> dict:
        key = (int(patient.id.lstrip("H")), task.id, task.duration_days)
        p = float(self._preds.loc[key])
        try:
            lat = float(self._latency_per_pred.loc[(task.id, task.duration_days)])
        except KeyError:
            lat = 0.04
        time.sleep(min(lat, 0.05))
        return {"prediction": p, "latency_s": lat, "is_real": True}


class _ReplayARModel:
    """Reads sampled trajectories and returns empirical event rate."""

    PER_SUBJ_TRAJ_SECONDS_MEASURED: float = 0.299  # from generation log

    REPLAY_PRESETS = {
        "real-time": PER_SUBJ_TRAJ_SECONDS_MEASURED,
        "fast":      0.10,
        "instant":   0.0,
    }

    def __init__(self, subjects: list[int]):
        self._subjects = list(subjects)
        self.per_traj_seconds_real = self.PER_SUBJ_TRAJ_SECONDS_MEASURED
        self.per_traj_seconds_replay = self.PER_SUBJ_TRAJ_SECONDS_MEASURED
        self._cache: dict[int, Any] = {}

    def _load_shard(self, shard_idx: int):
        import pandas as pd
        import pyarrow.parquet as pq
        if shard_idx in self._cache:
            return self._cache[shard_idx]
        local = _CACHE / f"ar_shard_{shard_idx}.parquet"
        local.parent.mkdir(parents=True, exist_ok=True)
        if local.exists():
            df = pd.read_parquet(local)
        else:
            try:
                with _state["fs"].open(f"{_BUCKET}/{_AR_TRAJ_DIR}/{shard_idx}.parquet", "rb") as f:
                    table = pq.read_table(f, filters=[("subject_id", "in", self._subjects)])
                df = table.to_pandas()
            except Exception:
                full = _gread_parquet(f"{_AR_TRAJ_DIR}/{shard_idx}.parquet")
                df = full[full["subject_id"].isin(self._subjects)].copy()
            df.to_parquet(local, index=False)
        self._cache[shard_idx] = df
        return df

    @staticmethod
    def _to_events(traj_df, anchor_us: int, max_events: int = 30) -> list[PatientEvent]:
        import pandas as pd
        events = []
        anchor = pd.Timestamp(anchor_us, unit="us")
        for _, row in traj_df.head(max_events).iterrows():
            delta_h = (row["time"] - anchor).total_seconds() / 3600.0
            events.append(PatientEvent(
                t_hours=float(delta_h),
                kind=_kind_for(row["code"]),
                description=humanize(row["code"]),
            ))
        return events

    def predict_streaming(
        self,
        patient: DemoPatient,
        task: Task,
        n_samples: int = 20,
        callback: Optional[Callable] = None,
    ) -> dict:
        import numpy as np
        import pandas as pd
        sid = int(patient.id.lstrip("H"))
        anchor_us = patient.prediction_time_us
        anchor = pd.Timestamp(anchor_us, unit="us")
        end = anchor + pd.Timedelta(days=task.duration_days)

        outcomes, trajectories = [], []
        n_iter = min(n_samples, len(_AR_TRAJ_SHARDS))
        t_start = time.time()
        for i, shard_idx in enumerate(_AR_TRAJ_SHARDS[:n_iter]):
            df = self._load_shard(shard_idx)
            traj = df[df["subject_id"] == sid].sort_values("time")
            if len(traj) == 0:
                outcome = 0
                rendered = []
            else:
                in_window = traj[(traj["time"] > anchor) & (traj["time"] <= end)]
                outcome = int((in_window["code"] == task.id).any())
                source = in_window if len(in_window) else traj
                rendered = self._to_events(source, anchor_us)
            outcomes.append(outcome)
            trajectories.append(rendered)
            time.sleep(self.per_traj_seconds_replay)
            if callback:
                callback(i + 1, n_iter, outcomes, rendered)

        replay_wall = time.time() - t_start
        return {
            "prediction": float(np.mean(outcomes)),
            "samples": outcomes,
            "trajectories": trajectories,
            "latency_s": self.per_traj_seconds_real * n_iter,
            "replay_wall_s": replay_wall,
            "n_samples": n_iter,
            "is_real": True,
        }


# ---------------------------------------------------------------------------
# UI — HTML helpers + ipywidgets demo
# ---------------------------------------------------------------------------
_KIND_COLORS = {
    "vital":     "#0a7",
    "lab":       "#06a",
    "med":       "#a06",
    "procedure": "#960",
    "diagnosis": "#a30",
    "note":      "#666",
}


def _render_patient_card(p: DemoPatient) -> str:
    age_sex = " · ".join(filter(None, [
        f"{p.age}y" if p.age else None,
        p.sex or None,
        f"ICU day {p.icu_day}" if p.icu_day else None,
    ]))
    header = p.id + ((" — " + age_sex) if age_sex else "")
    return (
        "<div style='border:1px solid #ddd; padding:14px; border-radius:8px; margin:6px 0; background:#fafafa;'>"
        f"<div style='font-size:18px; font-weight:600;'>{header}</div>"
        f"<div style='margin:4px 0; color:#a30; font-weight:500;'>{p.primary_issue}</div>"
        f"<div style='margin-top:8px; color:#222; line-height:1.5;'>{p.summary}</div>"
        "</div>"
    )


def _render_event_timeline(events, anchor_t: float = 0.0, title: Optional[str] = None) -> str:
    rows = []
    if title:
        rows.append(f"<div style='font-weight:600; margin:8px 0 4px;'>{title}</div>")
    for e in sorted(events, key=lambda x: x.t_hours):
        delta = e.t_hours - anchor_t
        sign = "+" if delta >= 0 else ""
        color = _KIND_COLORS.get(e.kind, "#444")
        rows.append(
            "<div style='margin:1px 0; font-family:ui-monospace,Menlo,monospace; font-size:13px;'>"
            f"<span style='color:#888;'>{sign}{delta:>7.1f}h</span> "
            f"<span style='color:{color}; min-width:90px; display:inline-block;'>[{e.kind}]</span> "
            f"<span style='color:#222;'>{e.description}</span>"
            "</div>"
        )
    return "<div>" + "".join(rows) + "</div>"


def _render_prob_block(p: float, label: str, color: str, latency_s: float, footer: str,
                       base_rate: Optional[float] = None) -> str:
    import numpy as np
    bar_pct = int(float(np.clip(p, 0, 1)) * 100)
    base_html = ""
    if base_rate is not None and base_rate > 0:
        base_html = (
            "<div style='color:#666; font-size:13px; margin-bottom:8px;'>"
            f"Base rate in held-out cohort: <b>{base_rate*100:.2f}%</b> &middot; "
            f"this prediction = <b>{p/base_rate:.1f}×</b> base rate</div>"
        )
    lat_str = f"{latency_s:.1f} s" if latency_s >= 1 else f"{latency_s*1000:.0f} ms"
    return (
        f"<div style='font-size:32px; font-weight:600;'>{p:.3f}</div>"
        f"<div style='color:#666; margin-bottom:4px;'>P({label})</div>"
        f"{base_html}"
        f"<div style='height:14px; background:#eee; border-radius:7px; overflow:hidden; margin:8px 0 16px;'>"
        f"<div style='width:{bar_pct}%; height:100%; background:{color};'></div></div>"
        f"<div style='color:#444;'>Latency: <b>{lat_str}</b> &middot; {footer}</div>"
    )


def _render_speedup(eq_s: float, ar_s: float) -> str:
    speedup = ar_s / max(eq_s, 1e-6)
    extrap = (
        f"On 1 000 patients × 5 tasks: AR ≈ <b>{(ar_s * 5000)/3600:.1f} GPU-h</b>, "
        f"EQ ≈ <b>{(eq_s * 5000)/60:.1f} GPU-min</b>."
    )
    return (
        "<div style='margin-top:14px; padding:12px; background:#f3f8ff; border:1px solid #cdd; "
        "border-radius:6px;'>"
        f"<div style='font-size:18px; font-weight:600;'>EQ ≈ {speedup:.0f}× faster ({eq_s*1000:.0f} ms vs {ar_s:.1f} s)</div>"
        f"<div style='margin-top:6px; color:#444;'>{extrap}</div>"
        "</div>"
    )


def run_demo(subjects: list[tuple[int, int]], ar_n_samples: int = 20) -> None:
    """Render the side-by-side EQ vs AR demo UI.

    Args:
        subjects: list of (subject_id, prediction_time_us) pairs to surface.
        ar_n_samples: how many AR trajectories to stream (default 20 = full set).
    """
    import ipywidgets as widgets
    from IPython.display import display, clear_output, HTML

    sids = [s for s, _ in subjects]
    patients = [build_demo_patient(s, pt) for s, pt in subjects]

    eq_model = _ReplayEQModel()
    ar_model = _ReplayARModel(sids)

    patient_picker = widgets.Dropdown(
        options=[(f"{p.id} — {p.primary_issue}", p) for p in patients],
        description="Patient:", layout=widgets.Layout(width="640px"),
    )
    task_picker = widgets.Dropdown(
        options=[(t.question, t) for t in _state["tasks"]],
        description="Question:", layout=widgets.Layout(width="640px"),
    )
    replay_speed = widgets.ToggleButtons(
        options=[("Real-time (~6 s)", "real-time"), ("Fast (~2 s)", "fast"), ("Instant", "instant")],
        value="real-time", description="AR replay:",
        tooltips=[
            "Wait the actual measured per-trajectory time (0.30 s × 20 = 6 s)",
            "Shortened wait (0.10 s × 20 = 2 s) — same predictions, less waiting",
            "No sleep between samples",
        ],
        style={"description_width": "initial"},
    )
    run_btn = widgets.Button(description="Compare models", button_style="primary")
    out_patient = widgets.Output()
    out_compare = widgets.Output()

    def update_patient(_=None):
        with out_patient:
            clear_output()
            p = patient_picker.value
            display(HTML(_render_patient_card(p)))
            display(HTML(_render_event_timeline(p.history, anchor_t=0.0,
                                                title="Recent events (anchored at prediction time, t=0)")))

    def on_run(_):
        with out_compare:
            clear_output()
            _render_comparison(patient_picker.value, task_picker.value, eq_model, ar_model,
                               replay_speed.value, ar_n_samples)

    patient_picker.observe(update_patient, names="value")
    run_btn.on_click(on_run)
    update_patient()

    display(widgets.VBox([
        widgets.HTML("<h2 style='margin:6px 0;'>EveryQuery vs. autoregressive trajectory model</h2>"
                     "<div style='color:#666; margin-bottom:12px;'>Pick a patient, pick a clinical question, "
                     "and watch a task-conditioned model and a trajectory-simulation model answer the same question.</div>"),
        patient_picker, out_patient,
        task_picker, replay_speed, run_btn, out_compare,
    ]))


def _render_comparison(patient, task, eq_model, ar_model, replay_speed_value, n_samples):
    import ipywidgets as widgets
    from IPython.display import display, clear_output, HTML

    ar_out = widgets.Output(layout=widgets.Layout(border="1px solid #ccc", padding="12px", min_height="320px"))
    eq_out = widgets.Output(layout=widgets.Layout(border="1px solid #ccc", padding="12px", min_height="320px"))
    cols = widgets.HBox([
        widgets.VBox([
            widgets.HTML("<h3 style='margin:0;'>AR trajectory model</h3>"
                         "<div style='color:#888; font-size:12px;'>Samples N trajectories, derives answer empirically</div>"),
            ar_out,
        ], layout=widgets.Layout(width="50%")),
        widgets.VBox([
            widgets.HTML("<h3 style='margin:0;'>EveryQuery</h3>"
                         "<div style='color:#888; font-size:12px;'>Task-conditioned, single forward pass</div>"),
            eq_out,
        ], layout=widgets.Layout(width="50%")),
    ])
    speedup_holder = widgets.Output()
    gt_btn = widgets.Button(description="Reveal ground truth")
    gt_out = widgets.Output()

    def reveal(_):
        with gt_out:
            clear_output()
            gt = patient.ground_truth.get((task.id, task.duration_days), "unknown")
            display(HTML(
                "<div style='margin-top:12px; padding:10px; background:#fffbe6; border:1px solid #f0c14b; "
                f"border-radius:6px;'><b>Ground truth ({task.label}):</b> "
                f"<code>{'positive' if gt == 1 else 'negative'}</code> &mdash; observed in held-out data within {task.duration_days} days.</div>"
            ))

    gt_btn.on_click(reveal)
    display(cols)
    display(speedup_holder)
    display(gt_btn)
    display(gt_out)

    br = _state["base_rate"].get((task.id, task.duration_days))

    with eq_out:
        display(HTML("<div style='color:#888;'>Running EveryQuery&hellip;</div>"))
        eq_result = eq_model.predict(patient, task)
        clear_output()
        display(HTML(_render_prob_block(
            eq_result["prediction"], task.label, "#0a7",
            latency_s=eq_result["latency_s"],
            footer="1 forward pass &middot; measured per-prediction wall-clock",
            base_rate=br,
        )))

    ar_model.per_traj_seconds_replay = _ReplayARModel.REPLAY_PRESETS[replay_speed_value]
    with ar_out:
        ar_result = _render_ar_streaming(patient, task, ar_model, n_samples, br)

    with speedup_holder:
        display(HTML(_render_speedup(eq_result["latency_s"], ar_result["latency_s"])))


def _render_ar_streaming(patient, task, ar_model, n_samples, base_rate):
    import ipywidgets as widgets
    from IPython.display import display, clear_output, HTML
    import numpy as np

    progress = widgets.IntProgress(min=0, max=n_samples, description="samples",
                                   layout=widgets.Layout(width="320px"))
    status = widgets.HTML(value=f"Sampling 0/{n_samples}&hellip;")
    hist_box = widgets.Output()
    # Scrollable container that grows as new trajectories arrive — viewers can
    # scroll back through every sample once streaming completes.
    traj_box = widgets.Output(layout=widgets.Layout(
        max_height="380px", overflow_y="auto",
        border="1px dotted #bbb", padding="6px",
    ))
    display(widgets.VBox([progress, status, hist_box, traj_box]))

    def cb(i, n, running, traj):
        progress.value = i
        status.value = f"Sampling {i}/{n}&hellip;"
        with hist_box:
            clear_output()
            pos = sum(running)
            pct = pos / max(len(running), 1)
            display(HTML(
                f"<div style='margin:6px 0; color:#444;'>"
                f"Running: <b>{pos}/{len(running)}</b> samples show event "
                f"(<b>{pct:.0%}</b>)</div>"
            ))
        with traj_box:
            display(HTML(_render_event_timeline(
                traj, anchor_t=0.0,
                title=f"Sample {i} trajectory",
            )))

    result = ar_model.predict_streaming(patient, task, n_samples=n_samples, callback=cb)
    p = result["prediction"]
    n = result["n_samples"]
    display(HTML(
        "<hr style='margin:14px 0;'>" + _render_prob_block(
            p, task.label, "#06a", latency_s=result["latency_s"],
            footer=(
                f"{n} sampled trajectories &middot; "
                f"{result['latency_s']/n*1000:.0f} ms/trajectory (measured) &middot; "
                f"replayed in {result.get('replay_wall_s', 0):.1f} s"
            ),
            base_rate=base_rate,
        )
    ))
    return result


# ---------------------------------------------------------------------------
# Aggregate scoreboard + multi-task efficiency
# ---------------------------------------------------------------------------
def show_scoreboard() -> None:
    """Per-task AUC over the *full* held-out cohort (not the demo slice)."""
    import numpy as np
    import pandas as pd
    from IPython.display import display

    eic_auc = _gread_parquet(_AR_AUC_PATH)
    auc_cols = [c for c in eic_auc.columns if c.startswith("AUC/")]
    eic_long = eic_auc.melt(id_vars=["duration"], value_vars=auc_cols,
                            var_name="code_slug", value_name="ar_auc")
    eic_long["code_slug"] = eic_long["code_slug"].str.removeprefix("AUC/")
    eic_long["duration_days"] = eic_long["duration"].dt.days
    eic_long = eic_long[["code_slug", "duration_days", "ar_auc"]]

    eq_long = _state["auc_df"][["code", "code_slug", "duration_days", "occurs_auc", "bucket"]].rename(
        columns={"occurs_auc": "eq_auc"}
    )
    merged = eq_long.merge(eic_long, on=["code_slug", "duration_days"], how="inner")
    merged["delta"] = merged["eq_auc"] - merged["ar_auc"]
    merged["winner"] = np.where(merged["delta"] > 0, "EQ",
                                np.where(merged["delta"] < 0, "AR", "tie"))

    n = len(merged)
    n_eq = int((merged["winner"] == "EQ").sum())
    n_ar = int((merged["winner"] == "AR").sum())
    n_ties = int((merged["winner"] == "tie").sum())
    print(f"Full held-out cohort: {n} (code, duration) cells.")
    print(f"  EQ wins: {n_eq} ({n_eq/n:.1%})")
    print(f"  AR wins: {n_ar} ({n_ar/n:.1%})")
    print(f"  Ties:    {n_ties}")
    print(f"  Mean EQ AUC: {merged['eq_auc'].mean():.3f}")
    print(f"  Mean AR AUC: {merged['ar_auc'].mean():.3f}")
    print(f"  Mean delta:  {merged['delta'].mean():+.3f} AUC points in EQ's favor")
    print()
    print("Per duration:")
    per_dur = (
        merged.groupby("duration_days")
        .agg(n_tasks=("code_slug", "count"),
             eq_auc=("eq_auc", "mean"),
             ar_auc=("ar_auc", "mean"),
             eq_win_rate=("winner", lambda s: (s == "EQ").mean()))
        .round(3)
    )
    print(per_dur.to_string())
    print()
    demo_keys = pd.DataFrame([(t.id, t.duration_days) for t in _state["tasks"]],
                             columns=["code", "duration_days"])
    label_lookup = pd.DataFrame([{"code": t.id, "duration_days": t.duration_days, "label": t.label}
                                 for t in _state["tasks"]])
    demo_aucs = (merged.merge(demo_keys, on=["code", "duration_days"], how="inner")
                       .merge(label_lookup, on=["code", "duration_days"], how="left"))
    print("AUC over the full held-out cohort for the 8 tasks featured in this demo:")
    display(demo_aucs[["label", "duration_days", "eq_auc", "ar_auc", "delta", "winner"]].round(3))


def show_multi_task_costs() -> None:
    """How AR's amortization advantage saturates as #tasks grows."""
    ar_per_traj_real = _ReplayARModel.PER_SUBJ_TRAJ_SECONDS_MEASURED
    ar_shared_s = ar_per_traj_real * 20      # 20 trajectories per patient
    ar_per_task_s = 0.05                     # tally cost per additional task
    eq_lat_per_call_s = 0.05                 # representative EQ wall-clock

    print(f"{'#tasks':>7s}  {'AR total':>10s}  {'EQ total':>10s}  {'speedup':>8s}")
    for n in [1, 2, 4, 8, 16, 32]:
        ar_total = ar_shared_s + n * ar_per_task_s
        eq_total = eq_lat_per_call_s * n
        print(f"{n:>7d}  {ar_total:>9.1f}s  {eq_total:>9.3f}s  {ar_total / eq_total:>7.0f}×")
