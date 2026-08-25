# Tiny all-features conditional model — held-out AUROC

Branch `worktree-nf-train-eval` (from `dev` @ `a59f951`). Scripts: `scripts/new_features/`.
W&B: project `EQ-conditional-new-features-test`, run `wu8xbdh0`.

**Headline: not random.** All three new query forms score well above chance on the held-out
split, at both sequence lengths, from a model trained for 1,638 steps.

## Macro AUROC at the target position

Macro-averaged over 20 randomly drawn tasks per category. Every task has ≥10 positives and
≥10 negatives, so all 60 are scored at both lengths — nothing was silently dropped.

| category | len 1 | len 3 | tasks scored | tasks > 0.5 | 95% CIs excluding 0.5 | sign-test p |
|---|---|---|---|---|---|---|
| duration-bounded | **0.815** | **0.821** | 20/20 | 20/20 | 20/20 | 1.9 × 10⁻⁶ |
| event-bounded | **0.768** | **0.777** | 20/20 | 17/20 | 17/20 | 2.6 × 10⁻³ |
| DAG / ancestor | **0.776** | **0.783** | 20/20 | 18/20 | 18/20 | 4.0 × 10⁻⁴ |

Pooled AUROC by position at length 3 — **monotonic**, the same conditioning effect the v2
report found (0.769 → 0.782 across positions 0→4 for the full model):

| position | n | prevalence | pooled AUROC |
|---|---|---|---|
| 0 | 238,740 | 0.150 | 0.659 |
| 1 | 238,740 | 0.166 | 0.696 |
| 2 | 238,740 | 0.124 | 0.703 |

(Pooled numbers are base-rate inflated and are for dynamics only; the macro table above is the
honest level.)

Every category also gains from length 1 → 3 (+0.006, +0.010, +0.006). The gain is small but
consistent in sign across all three, and inference is teacher-forced, so position 2 sees the
true answers at positions 0 and 1.

`HOSPITAL_ADMISSION` — the ancestor the request named, a pure ancestor rolling up 70 child
codes — is `anc_00`: AUROC **0.543** at len 1 and **0.546** at len 3, CI [0.525, 0.567].
Above chance, but the weakest ancestor in the panel: at 23.8% prevalence it is a broad,
near-ambient roll-up, which is the hard end of the ancestor range. Other ancestors reach 0.95.

Per-task numbers: `by_task_len1.csv`, `by_task_len3.csv`. Spec names are category+index; the
code strings live in `$NF_ROOT/eval_specs/designed_len{1,3}.yaml`.

## What was actually trained

Tiny: hidden 256, 4 encoder layers, 2 decoder layers, 4 heads, ffn 1024, batch 64,
`max_seq_len` 256, bf16. Labels: 300,000 train + 20,000 tuning query sequences, lengths
U{1..5}, log-uniform horizons over [1, 731] days.

**All three features were on**, verified three ways (`09_verify_run.sh`, 22 checks):

| feature | evidence |
|---|---|
| RoPE time | `strip_delta_tokens=true` + `use_rope_time=true` in the saved config and in the checkpoint hparams; the run log shows `RoPE time: stripping 25 delta-token vocab ids`; a live batch carries `time_pos_ids` with the same shape as `code`, nondecreasing over real tokens, in elapsed hours |
| event-bounded | 50.1% of the 900,512 training queries are event-bounded, with the `-1` duration sentinel aligned on every one (0 violations in either direction); a live batch has `q_bound_codes` with 48% bounded; `bound_marker` is in the state dict |
| DAG-aware | encoder sized to `V_ext = 21,072` rather than the 13,909 cohort vocab; query vocabulary extended to 21,071 nodes (7,163 ancestors); 34.0% of training queries and 33.9% of event boundaries are ancestor nodes; the input embedding is an `OntologyEmbedding` with 21,072 rows |

The ontology: 21,071 nodes = 13,908 leaves + 7,163 ancestors (399 dual-role `//ANY`), built
from the cohort's real `parent_codes` column, not from `//`-prefix structure alone.

## The evaluation set

3,979 held-out contexts × 60 designed specs = 238,740 labeled sequences per length; the two
lengths use the *same* contexts. Tasks are drawn randomly from the model's own query universe
under an occurrence floor, since a code nobody ever has yields an undefined AUROC. At length 3
the target sits at position 2 behind two randomly drawn filler queries (themselves a random mix
of the three forms), so len 1 and len 3 differ only in conditioning depth.

Row identity was **asserted, not assumed**: each row's spec is reconstructed from row order and
checked against the spec YAML. Match rate 1.000000 on both files.

Event bounds behave as documented: 79,580 bounded eval queries over 20 distinct boundary nodes,
45.2% of which never fire after the prediction time, so those windows run to the end of the
record and the query degenerates to "does this ever occur again".

## Caveat — the training run crashed

The run died at step 1,638 of 4,688 in the **backward** pass:

```
RuntimeError: merge_sort: failed to synchronize: cudaErrorIllegalAddress
```

All numbers above come from `checkpoints/last.ckpt` at step 1,638, which is a valid checkpoint
— `EQ_predict_sequences` falls back to it when `best_model.ckpt` is absent. Nothing about the
evaluation is compromised; the model is simply undertrained, so treat these AUROCs as a floor.

It is **not** an out-of-range embedding index: `probe_index_ranges.py` scanned 500 real batches
and every index was in range (`q_codes` max 21,071 vs `V_ext` 21,072; `q_answers` ∈ {0,1};
`q_bound_codes` in range; `q_answers` padding is `ANSWER_NO`=0 and `block_pos_embed` is clamped).

The most likely site is the ontology embedding itself: `OntologyEmbedding` holds the mix matrix
as a **21,072 × 21,072 sparse COO tensor on GPU**, and the backward through `A @ W` sorts
indices — which is exactly where `merge_sort` runs. Two other users' training jobs were sharing
this GB10 at the time, so memory pressure is a plausible trigger. Worth reproducing before
treating it as a hardware flake.
