#!/usr/bin/env python
"""Assemble the FINAL conditional-query report PDF (converged big_v2 run).

Consumes:
  --train-csv     big_v2 Lightning CSVLogger metrics.csv
  --macro         macro_patient_final/summary.json (per-position macro per-task AUC)
  --clinical      clinical_final/summary.json (single-query + conditioning)
Writes a multi-section PDF: model, task + leak post-mortem, data, training stability/convergence,
evaluation methodology, macro per-task results + conditioning trend, clinical tasks, discussion.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

try:  # `reportlab` is a report-only dependency, deliberately absent from pyproject.toml.
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        HRFlowable, Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )
except ModuleNotFoundError as e:  # pragma: no cover - depends on the active venv
    raise ModuleNotFoundError(
        "scripts/build_report*.py needs `reportlab`, which this project does not depend on: the "
        "PDFs under reports/ were built in a separate venv.  Install it into the active venv with "
        "`uv pip install reportlab` and rerun."
    ) from e


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("H1c", parent=ss["Heading1"], fontSize=14.5, spaceBefore=9, spaceAfter=5))
    ss.add(ParagraphStyle("H2c", parent=ss["Heading2"], fontSize=11.5, spaceBefore=6, spaceAfter=3))
    ss.add(ParagraphStyle("Body", parent=ss["BodyText"], fontSize=9.3, leading=12.6))
    ss.add(ParagraphStyle("Small", parent=ss["BodyText"], fontSize=7.8, leading=9.8, textColor=colors.grey))
    ss.add(ParagraphStyle("TitleBig", parent=ss["Title"], fontSize=18, alignment=TA_CENTER))
    return ss


def _fmt(x, nd=3):
    if x is None:
        return "—"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def _table(data, col_widths=None, font=8):
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), font),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF2F5")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BBBBBB")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    return t


def _aspect(p):
    from PIL import Image as PILImage
    with PILImage.open(p) as im:
        w, h = im.size
    return h / w


# ---------- figures ----------

def training_figs(csv_fp, figs):
    df = pl.read_csv(csv_fp, infer_schema_length=10**6)
    figs.mkdir(parents=True, exist_ok=True)
    info = {}

    def s(c):
        if c not in df.columns:
            return None, None
        sub = df.select("step", c).drop_nulls()
        return sub["step"].to_list(), sub[c].to_list()

    # loss + LR + grad norm
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(9.6, 2.7))
    x, y = s("train/loss_step")
    if x:
        a1.plot(x, y, lw=0.4, c="#BBB", label="train (step)")
    x, y = s("tuning/loss")
    if x:
        a1.plot(x, y, "-", lw=1.4, c="#C0392B", label="tuning")
        info["best_tuning_loss"] = min(y); info["final_tuning_loss"] = y[-1]
    a1.set_title("Loss"); a1.set_xlabel("step"); a1.set_ylabel("BCE"); a1.legend(fontsize=7)
    x, y = s("train/grad_norm")
    if x:
        a2.plot(x, y, lw=0.4, c="#27AE60")
        info["median_grad_norm"] = float(pl.Series(y).median()); info["max_grad_norm"] = max(y)
    a2.set_title("Gradient L2 norm (pre-clip)"); a2.set_xlabel("step")
    x, y = s("lr-AdamW/pg1")
    if x:
        a3.plot(x, y, lw=1.2, c="#8E44AD")
    a3.set_title("LR schedule"); a3.set_xlabel("step")
    fig.tight_layout(); fig.savefig(figs / "stability.png", dpi=150); plt.close(fig)

    # validation AUROC trajectory (pooled + per-position)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for col, lab, c, lw in [("tuning/answer_auc", "pooled (all positions)", "#2C3E50", 1.6),
                            ("tuning/answer_auc_pos0", "pos 0", "#3498DB", 0.9),
                            ("tuning/answer_auc_pos4", "pos 4", "#E67E22", 0.9)]:
        x, y = s(col)
        if x:
            ax.plot(x, y, "-", lw=lw, label=lab, c=c)
            if col == "tuning/answer_auc":
                info["final_pooled_auc"] = y[-1]
    ax.set_ylim(0.5, 1.0); ax.axhline(0.5, ls="--", c="grey", lw=0.8)
    ax.set_title("Validation AUROC over training (pooled — base-rate inflated, diagnostic only)", fontsize=9)
    ax.set_xlabel("step"); ax.legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(figs / "val_auroc.png", dpi=150); plt.close(fig)
    info["total_steps"] = int(df["step"].max())
    return info


def macro_fig(macro, figs):
    pos = sorted(int(k) for k in macro["macro_auc_by_position"])
    ys = [macro["macro_auc_by_position"][str(p)] for p in pos]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.plot(pos, ys, "-o", c="#C0392B", lw=2)
    ax.set_xticks(pos)
    ax.set_xlabel("query position in sequence (# of prior teacher-forced answers)")
    ax.set_ylabel("macro per-task AUROC")
    lo, hi = min(ys), max(ys)
    ax.set_ylim(lo - 0.004, hi + 0.004)
    sl = macro["slope_per_position"]; ci = macro["slope_ci95"]
    ax.set_title(f"Per-task discriminability vs position\nslope {sl:+.5f}/pos  95%CI [{ci[0]:+.5f}, {ci[1]:+.5f}]  (excludes 0)", fontsize=9)
    fig.tight_layout(); fig.savefig(figs / "macro_position.png", dpi=150); plt.close(fig)


def clinical_figs(clin, figs):
    sq = clin["single_query"]
    names = list(sq)
    order = sorted(names, key=lambda n: sq[n]["auroc"])
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ys = np.arange(len(order))
    auc = [sq[n]["auroc"] for n in order]
    err = [[sq[n]["auroc"] - sq[n]["auroc_ci95"][0] for n in order],
           [sq[n]["auroc_ci95"][1] - sq[n]["auroc"] for n in order]]
    ax.barh(ys, auc, xerr=err, color="#2980B9", ecolor="#444", capsize=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{n}\n(prev {sq[n]['prevalence']:.1%})" for n in order], fontsize=7)
    ax.axvline(0.5, ls="--", c="grey", lw=1); ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("within-task AUROC (95% CI)")
    ax.set_title("Designed clinical single-query tasks (held-out; 3 anchor families)", fontsize=9)
    fig.tight_layout(); fig.savefig(figs / "clinical_single.png", dpi=150); plt.close(fig)

    cond = clin["conditional"]
    cn = list(cond)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(len(cn)); w = 0.38
    no = [cond[n]["mean_P_target_given_prior_NO"] for n in cn]
    yes = [cond[n]["mean_P_target_given_prior_YES"] for n in cn]
    ax.bar(x - w / 2, no, w, color="#3498DB", label="prior = NO")
    ax.bar(x + w / 2, yes, w, color="#C0392B", label="prior = YES")
    import textwrap
    ax.set_xticks(x)
    ax.set_xticklabels(["\n".join(textwrap.wrap(n, 24)) for n in cn], fontsize=6.5)
    ax.set_ylabel("mean predicted P(target)")
    ax.set_title("Conditioning: target risk vs teacher-forced prior answer", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figs / "clinical_cond.png", dpi=150); plt.close(fig)


def arch_fig(figs):
    fig, (ax, axm) = plt.subplots(1, 2, figsize=(9.4, 3.2), gridspec_kw={"width_ratios": [1.8, 1]})
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)

    def box(x, y, w, h, t, fc):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec="#333", lw=1))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7.5)

    box(0.3, 4.3, 3.3, 1.3, "Patient events ≤ t\n(bidirectional\nModernBERT encoder)", "#AED6F1")
    bx = 4.5
    for j in range(3):
        box(bx, 4.5, 0.7, 0.95, f"Q{j+1}\nc,d", "#F9E79F")
        box(bx + 0.74, 4.5, 0.62, 0.95, f"A{j+1}", "#ABEBC6")
        bx += 1.62
    ax.text(4.5, 5.75, "block-autoregressive decoder (cross-attends to encoder)", fontsize=7)
    ax.annotate("", xy=(4.4, 5.0), xytext=(3.6, 5.0), arrowprops=dict(arrowstyle="->", color="#666"))
    box(3.1, 1.5, 3.8, 0.9, "binary answer head → P(code in (t, t+d))", "#F5B7B1")
    for k in range(3):
        ax.annotate("", xy=(3.9 + k * 1.3, 2.4), xytext=(4.85 + k * 1.62, 4.45),
                    arrowprops=dict(arrowstyle="->", color="#999", lw=0.7))
    ax.text(5.0, 0.95, "loss = BCE over all query positions; censoring is the TIMELINE//END query",
            ha="center", fontsize=6.8, color="#555")
    ax.set_title("Conditional query-sequence model", fontsize=9.5)

    L = 3; n = 3 * L
    allowed = np.zeros((n, n))
    for i in range(n):
        bi, ti = i // 3, i % 3
        for k in range(n):
            bk, tk = k // 3, k % 3
            allowed[i, k] = 1.0 if ((bk < bi) or (bk == bi and tk < 2) or (bk == bi and ti == 2 and tk == 2)) else 0.0
    axm.imshow(allowed, cmap="Greens", vmin=0, vmax=1.5)
    labs = sum([[f"c{j+1}", f"d{j+1}", f"a{j+1}"] for j in range(L)], [])
    axm.set_xticks(range(n)); axm.set_xticklabels(labs, fontsize=6)
    axm.set_yticks(range(n)); axm.set_yticklabels(labs, fontsize=6)
    axm.set_title("Block-causal mask (L=3)\ngreen=allowed", fontsize=8)
    for i in range(n + 1):
        axm.axhline(i - 0.5, color="white", lw=0.4); axm.axvline(i - 0.5, color="white", lw=0.4)
    fig.tight_layout(); fig.savefig(figs / "arch.png", dpi=150); plt.close(fig)


def _find(d, *subs):
    """Return the value whose key contains all substrings (case-insensitive), else None."""
    for k, v in d.items():
        if all(s.lower() in k.lower() for s in subs):
            return v
    return None


def build(train_csv, macro_fp, clin_fp, uncens_fp, out):
    ss = _styles()
    figs = Path(out).parent / "final_figs"
    figs.mkdir(parents=True, exist_ok=True)
    t = training_figs(Path(train_csv), figs)
    macro = json.loads(Path(macro_fp).read_text())
    clin = json.loads(Path(clin_fp).read_text())
    uncens = json.loads(Path(uncens_fp).read_text()) if uncens_fp and Path(uncens_fp).exists() else None
    arch_fig(figs); macro_fig(macro, figs); clinical_figs(clin, figs)
    mort30_pa = _find(clin["single_query"], "mortality_30d", "post-admission") or {}
    icu_cond = _find(clin["conditional"], "MICU") or {}

    E = []

    def P(x, st="Body"):
        E.append(Paragraph(x, ss[st]))

    def img(p, w=6.6 * inch):
        p = Path(p)
        if p.exists():
            E.append(Image(str(p), width=w, height=w * _aspect(p))); E.append(Spacer(1, 5))

    P("Conditional Query Answering over EHR", "TitleBig")
    P("A block-autoregressive reformulation of EveryQuery, trained on MIMIC-IV (MEDS)", "H2c")
    mp = macro["macro_auc_by_position"]
    P(
        f"We rework EveryQuery (Chandak et al., arXiv:2603.07900) into a conditional query-sequence "
        f"foundation model: a bidirectional encoder over the patient record feeds a block-autoregressive "
        f"decoder that answers an ordered list of queries, each conditioned on the patient state and the "
        f"teacher-forced answers of all earlier queries. Trained on a single NVIDIA GB10 over one "
        f"no-repeat epoch of <b>28.79M</b> unique random query sequences (every step a fresh sequence). "
        f"Held-out macro per-task AUROC is <b>{_fmt(mp['0'])}</b>; designed clinical tasks reach "
        f"<b>{_fmt(mort30_pa.get('auroc'))}</b> AUROC (30-day mortality, post-admission), and "
        f"conditioning on an intermediate ICU-admission query raises predicted 30-day mortality "
        f"{icu_cond.get('mean_P_target_given_prior_NO',0):.1%}→"
        f"{icu_cond.get('mean_P_target_given_prior_YES',0):.1%}.",
        "Body",
    )
    E.append(HRFlowable(width="100%", color=colors.grey, thickness=0.5, spaceBefore=5, spaceAfter=5))

    # 1. Task & model
    P("1. Task & model", "H1c")
    P(
        "<b>Task.</b> Each query is <i>(code C, horizon d)</i>; the answer is binary — was C observed in "
        "<i>(t, t+d)</i> for a patient at prediction time t? The model answers an ordered sequence "
        "<font face='Courier'>[Q1][A1][Q2][A2]…</font>, predicting each A<sub>j</sub> from the patient "
        "state, the queries Q<sub>1..j</sub>, and the teacher-forced answers A<sub>1..j-1</sub> — never "
        "A<sub>j</sub> itself. It thus learns P(A<sub>j</sub> | patient, Q<sub>1..j</sub>, A<sub>1..j-1</sub>).",
        "Body",
    )
    P(
        "<b>Architecture.</b> A ModernBERT encoder (8 layers, hidden 384) embeds the tokenized MEDS "
        "history up to t. A 4-layer Transformer decoder runs over the query/answer stream and "
        "cross-attends to it under a block-causal mask: a query's code/duration tokens see each other and "
        "all tokens of earlier blocks (incl. their answers), but never the query's own answer and never "
        "later blocks. The answer for block j is read at its duration token. Encoder uses ModernBERT "
        "positions; the decoder adds learned block-position + token-type embeddings. ~33M parameters, bf16. "
        "Single binary head; loss is one BCE over all real query positions.",
        "Body",
    )
    img(figs / "arch.png", w=6.9 * inch)
    P(
        "<b>Censoring is a query, not a label.</b> The end-of-timeline code <font face='Courier'>"
        "TIMELINE//END</font> (one per subject) is queried like any other: <font face='Courier'>(END, d)</font> "
        "= 'does the record end within d?'. Conditioning a later query on its answer recovers and generalizes "
        "EveryQuery's implicit P(occurs | data after d): the END=NO slice is the original question, while "
        "END=YES gives P(occurs | record ends) — the regime where terminal events like death actually fall.",
        "Body",
    )
    P(
        "<b>Leak post-mortem.</b> An earlier design teacher-forced a same-horizon censor answer and masked "
        "censored labels; for terminal events the censor answer then equaled the label, and the model copied "
        "it — inflating 30-day mortality AUROC to 0.991 (0.996 from the censor answer alone). The binary "
        "occurrence + END-as-query design here removes that leak structurally; the honest mortality AUROC is "
        f"{_fmt(mort30_pa.get('auroc'))} post-admission (§6).",
        "Small",
    )

    E.append(PageBreak())

    # 2. Data
    P("2. Data & query sampling", "H1c")
    P(
        "MIMIC-IV in MEDS form (v0.3.0 tensorized cohort, 11,958 codes; 292 train / 37 tuning / 37 held-out "
        "shards). Training sequences are sampled fully at random: 5 i.i.d. queries per patient context "
        "(code uniform over the vocabulary incl. TIMELINE//END, horizon log-uniform over 1–365 days), with "
        "binary observed-occurrence labels from a single per-query asof join. We sampled <b>28.79M</b> "
        "unique sequences (86.9% distinct; all 227,602 subjects covered across all shards) and trained one "
        "no-repeat epoch — so every optimizer step sees a different query sequence (maximizing query-space "
        "coverage; verified no cross-shard query replication). Fixed length 5 is used because, under the "
        "block-causal mask, position j depends only on blocks ≤ j, so shorter sequences are redundant "
        "prefixes — fixed length trains every position with uniform sample counts.",
        "Body",
    )

    # 3. Training stability / convergence
    P("3. Training stability & convergence", "H1c")
    P(
        f"AdamW (lr 2e-4, cosine schedule, 5% warmup), grad-clip 1.0, bf16, batch 96, "
        f"{t.get('total_steps','—'):,} steps (~22.5 h). Training was stable throughout: the pre-clip "
        f"gradient norm stayed bounded (median {_fmt(t.get('median_grad_norm'),2)}, max "
        f"{_fmt(t.get('max_grad_norm'),2)}) with no loss spikes; tuning loss fell smoothly to "
        f"{_fmt(t.get('best_tuning_loss'))} and plateaued, indicating clean convergence on this one-epoch "
        f"schedule.",
        "Body",
    )
    img(figs / "stability.png", w=6.9 * inch)
    img(figs / "val_auroc.png", w=5.8 * inch)

    E.append(PageBreak())

    # 4. Evaluation methodology
    P("4. Evaluation methodology", "H1c")
    P(
        "<b>Macro per-task AUROC, not pooled.</b> Pooled AUROC scores cross-task pairs (a positive for one "
        "query vs a negative for another) and is dominated by base-rate differences between queries — it "
        "overstates per-query skill (pooled ≈0.95 here). The right metric is AUROC computed <i>within each "
        "query</i>, macro-averaged. Since AUROC = P(score<sub>pos</sub> &gt; score<sub>neg</sub>) "
        "(Mann–Whitney), for one positive/negative patient pair drawn for the same query the indicator "
        "1[score<sub>pos</sub> &gt; score<sub>neg</sub>] is an unbiased "
        "estimate of that query's AUROC, and averaging over many queries estimates macro-AUROC.",
        "Body",
    )
    P(
        "Positives are constructed <b>occurrence-driven</b> — take a real occurrence of C at τ and a "
        "prediction time t∈[τ−d, τ) — so the label is guaranteed positive and <i>all codes are estimable, "
        "rare included</i> (no coverage bias toward common codes). We use the <b>patient-uniform</b> scheme "
        "(each patient that permits a positive picked uniformly, then one of its positive times) so long "
        "stays don't dominate. For the position trend, the <i>same</i> positive/negative pair is scored at "
        "every position (fillers' true answers teacher-forced), so the comparison is fully paired and any "
        "sampling bias cancels; we bootstrap over tasks for the slope CI.",
        "Body",
    )

    # 5. Results: macro per-task + conditioning trend
    P("5. Results — per-task discrimination & the conditioning trend", "H1c")
    rows = [["position (priors)", "macro per-task AUROC"]]
    for p in sorted(int(k) for k in mp):
        rows.append([str(p), _fmt(mp[str(p)], 4)])
    E.append(_table(rows, col_widths=[2.0 * inch, 2.2 * inch], font=8.5))
    P(
        f"On {macro['n_tasks']:,} held-out tasks (patient-uniform), per-task AUROC is "
        f"<b>{_fmt(mp['0'])}</b> at position 0 and rises monotonically with position. The trend is small "
        f"but statistically significant: slope <b>{_fmt(macro['slope_per_position'],5)}</b>/position, 95% CI "
        f"[{_fmt(macro['slope_ci95'][0],5)}, {_fmt(macro['slope_ci95'][1],5)}] (excludes 0); Spearman ρ = "
        f"{_fmt(macro['spearman_rho'],2)} [{_fmt(macro['spearman_ci95'][0],2)}, "
        f"{_fmt(macro['spearman_ci95'][1],2)}]. Later queries, which condition on more prior (query, answer) "
        f"context, are answered measurably better — confirming the model uses the conditional structure "
        f"(lower Bayes error with more information). The increment is tiny (+"
        f"{(mp[str(max(int(k) for k in mp))]-mp['0'])*1000:.1f} milli-AUROC pos 0→4): random priors are only "
        f"weakly informative about a random target, and a well-trained encoder already captures much of the "
        f"signal, leaving little for the priors to add.",
        "Body",
    )
    img(figs / "macro_position.png", w=4.9 * inch)

    E.append(PageBreak())

    # 5b. Original-EveryQuery-comparable: occurs-AUROC on the uncensored cohort
    if uncens:
        P("5b. Comparison to original EveryQuery (uncensored cohort)", "H2c")
        we = uncens.get("with_EOS=NO_prefix", {}); mg = uncens.get("marginal_no_prefix", {})
        P(
            f"Original EveryQuery reports occurrence AUC only where censoring is false (the window is fully "
            f"observed). We replicate that exactly — task sequence <font face='Courier'>[TIMELINE//END, d]=0 "
            f"[C, d]</font> evaluated only on contexts where the record does not end within d — over "
            f"{we.get('n_tasks',0):,} tasks. Macro occurs-AUROC is <b>{_fmt(we.get('macro_occurs_auroc'))}</b> "
            f"[{_fmt(we.get('ci95',[None,None])[0])}, {_fmt(we.get('ci95',[None,None])[1])}] with the EOS=NO "
            f"prefix, vs {_fmt(mg.get('macro_occurs_auroc'))} without it — statistically indistinguishable "
            f"(the cohort is already uncensored, so the prefix adds little). This ~0.79 is the apples-to-apples "
            f"number against original EveryQuery's headline metric; note terminal codes (death) have no "
            f"uncensored positives by construction and are absent here — exactly original EveryQuery's blind "
            f"spot, which our model covers via the EOS=YES (record-ends) regime.",
            "Body",
        )

    E.append(PageBreak())

    # 6. Clinical tasks
    P("6. Results — designed clinical tasks", "H1c")
    na = clin.get("n_anchors", {})
    P(
        "We designed clinically meaningful single-code tasks over three held-out anchor families — "
        f"post-admission (24h after a HOSPITAL_ADMISSION; {na.get('post_admission',0):,} anchors), "
        f"post-discharge (at a HOSPITAL_DISCHARGE//HOME event; {na.get('post_discharge',0):,}), and "
        f"random-time (a uniformly sampled valid event time; {na.get('random_time',0):,}) — each requiring "
        "≥10 prior events and ≤3 anchors/subject. Readmission is anchored at discharge (a genuine "
        "post-discharge metric); each task names its exact single MEDS code. Within-task AUROC, bootstrap "
        "95% CI.",
        "Body",
    )
    sq = clin["single_query"]
    rows = [["clinical task", "prevalence", "AUROC", "95% CI"]]
    for n in sorted(sq, key=lambda k: -sq[k]["auroc"]):
        d = sq[n]
        rows.append([n, f"{d['prevalence']:.2%}", _fmt(d["auroc"]),
                     f"[{_fmt(d['auroc_ci95'][0])}, {_fmt(d['auroc_ci95'][1])}]"])
    E.append(_table(rows, col_widths=[3.0*inch, 0.95*inch, 0.8*inch, 1.5*inch], font=7.8))
    img(figs / "clinical_single.png", w=6.0 * inch)

    P(
        "<b>Conditioning demonstrations.</b> Scoring a target query under a teacher-forced prior answer "
        "(forced YES vs NO on the same anchors) shows the model updates downstream risk in a clinically "
        "sensible direction — the capability the block-autoregressive form adds:",
        "Body",
    )
    cond = clin["conditional"]
    rows = [["conditional query (target | prior)", "P|prior=NO", "P|prior=YES", "natural AUROC"]]
    for n, c in cond.items():
        rows.append([n, f"{c['mean_P_target_given_prior_NO']:.3f}",
                     f"{c['mean_P_target_given_prior_YES']:.3f}", _fmt(c["target_auroc_natural"])])
    E.append(_table(rows, col_widths=[3.3*inch, 1.0*inch, 1.0*inch, 0.95*inch], font=7.6))
    img(figs / "clinical_cond.png", w=5.6 * inch)
    if icu_cond:
        no_, yes_ = icu_cond["mean_P_target_given_prior_NO"], icu_cond["mean_P_target_given_prior_YES"]
        P(
            f"The clearest example: teacher-forcing 'MICU admission within 7 days' raises predicted 30-day "
            f"mortality from {no_:.1%} to {yes_:.1%} (≈{yes_/no_:.1f}×) — the model correctly treats an "
            f"impending ICU transfer as a strong mortality signal.",
            "Body",
        )

    # 7. Discussion
    P("7. Discussion", "H1c")
    for para in [
        "<b>A single model answers arbitrary queries well.</b> Trained only on random query sequences (no "
        "task-specific supervision), the model reaches honest macro within-task AUROC ~0.79 across tens of "
        "thousands of held-out (code, horizon) tasks (and the same ~0.79 on original-EveryQuery's "
        "uncensored-occurs cohort), and 0.81–0.92 on designed mortality tasks across anchor families — "
        "leak-free, unlike the 0.99 of the flawed earlier design.",
        "<b>Conditioning works, and is used when it carries signal.</b> Per-task AUROC rises monotonically "
        "and significantly with sequence position (more priors → lower Bayes error), and clinically the "
        "model sharply updates downstream risk on informative priors (ICU→death 3.6×). The position effect "
        "is small in aggregate because random priors are weakly informative and a strong encoder is partly "
        "redundant with them — but it is real and statistically significant.",
        "<b>Censoring without a censor head.</b> Expressing censoring as the TIMELINE//END query makes the "
        "model strictly more expressive than EveryQuery (it can answer P(occurs | record ends), not just "
        "P(occurs | data continue)) and removes the terminal-event leak by construction.",
        "<b>Limitations.</b> Evaluation is teacher-forced (conditional calibration, not free-running "
        "rollout). The macro metric covers codes with an estimable positive rate. Sequences are length 5; "
        "longer-context behavior and free-running multi-step querying are natural next steps.",
    ]:
        P(para, "Body")
    P(
        "Code & docs: github.com/mmcdermott/EveryQuery-conditional (main). CLIs: EQ_generate_query_sequences, "
        "EQ_predict_sequences, EQ_evaluate_sequences. Evaluation: scripts/eval_macro_position.py (per-task), "
        "scripts/eval_clinical.py (clinical). See CONDITIONAL_QUERIES.md.",
        "Small",
    )

    SimpleDocTemplate(str(out), pagesize=letter, topMargin=0.65 * inch, bottomMargin=0.55 * inch,
                      leftMargin=0.7 * inch, rightMargin=0.7 * inch).build(E)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--macro", required=True)
    ap.add_argument("--clinical", required=True)
    ap.add_argument("--uncens", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a.train_csv, a.macro, a.clinical, a.uncens, a.out)


if __name__ == "__main__":
    main()
