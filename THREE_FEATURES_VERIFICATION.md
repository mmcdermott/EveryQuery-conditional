# Verifying the three-feature port

How to check a large diff that adds three features to `ConditionalQueryModel` — RoPE time
representation with delta-token stripping, event-bounded duration queries, and
ontology/hierarchical embeddings with ancestor queries — and what checking it actually found.

## Why "the tests pass" is not the bar here

The port adds dozens of new test functions across several files, and the suite was green when
the branch was pushed. A real data run then found bugs anyway — several of which produced
*silently wrong labels* rather than errors. That is the empirical starting point, and it has a
specific cause:

**the same author wrote the code and the tests.** A green test proves the code matches the
author's belief about the spec. It says nothing about whether the belief is right. Where the
belief was wrong, the test encodes the same wrong belief and passes.

Worse, a test can pass for a reason unrelated to what it claims to check. A bitwise
`assert not torch.equal(a, b)` is satisfied by **one ULP of float32 rounding** — around 1.19e-07
on these tensors — so a "the feature changes the output" test written that way is green from the
moment it exists, whether or not the feature does anything. That failure mode was observed
directly on this branch's sibling work, and it is why nothing here asserts liveness with a bare
inequality.

So the checks below are chosen specifically to be ones the author's own tests *cannot* be.

## The four checks, in decreasing order of what they buy you

### 1. Differential testing against a brute-force oracle — the highest-value check

`tests/test_rope_strip_oracle.py`

Write a second implementation from the *spec*, in plain Python loops, with no polars and no
torch, deliberately without consulting the optimised code. Run both over randomised synthetic
data and compare every output. An independent implementation cannot share an implementation bug,
so a disagreement localises a real defect in one of the two.

This is worth most where the optimised code is hardest to read. `strip_delta_tokens` compacts
four parallel tensors at once, re-bases each row's clock to its first surviving token, and
*recomputes* rather than compacts `time_delta_days`. A misalignment between any two of those
outputs would not raise — it would hand the encoder a stream whose values belong to different
tokens than its codes, corrupting every training sequence while every shape assertion still
passed.

Result: `strip_delta_tokens` matches a naive per-row oracle across 8 seeds × 2 `protect_first_n`
settings, including all-delta rows and degenerate padding, plus monotonic-rebased-position and
total-elapsed-time invariants. Runs in a few seconds, so it stays in the default suite.

**Coverage gap, stated rather than papered over:** only Feature 1 has an oracle. Event-bounded
labeling (`label_with_event_bounds`) and the ontology closure explode (`explode_events_to_closure`)
are both asserted against hand-written expectations, not an independent implementation. Those are
the two places a second oracle would buy the most.

### 2. Liveness probes — is the feature *live*, or merely wired?

`tests/test_feature_liveness.py`

A feature can be plumbed through collate, reach the forward pass, and then be multiplied by zero.
It passes every shape, dtype and "runs without error" assertion while contributing nothing. No
existing test could detect that, because they assert the model *runs* with the new tensors, not
that it *responds* to them.

Three probes a dead feature cannot pass:

- **Gradient** — each new parameter receives a non-zero gradient from a batch exercising it.
- **Sensitivity** — perturbing one new input field alone moves the output, by a margin well
  above float noise (`LIVE = 1e-6`, against ~1e-7 rounding and ~1e-4 real effects).
- **Atom invariance** — an unbounded batch is **bit-identical** with and without the new tensors
  attached.

That last one is the property every reported AUROC rests on. `tasks/eval_full` is entirely
atomic, time-bounded single-code queries, so if attaching the feature machinery perturbed them,
every score would describe a different model than the one trained. It holds at exactly `0.0`.

A measurement note that cost a false negative: a randomly-initialised decoder and head compress
an 8e-05 encoder-output difference down to ~1e-07 at the logits — the same magnitude as float
noise. **Assert at the level where the effect lives**, not downstream of an untrained head. The
RoPE probe therefore reads the encoder's `last_hidden_state` directly.

**Coverage gap:** the probes cover Feature 1 (RoPE positions reach the encoder; a RoPE model
without `time_pos_ids` refuses rather than falling back) and Feature 2 (`bound_marker` gets
gradient, boundary identity moves the output, a boundary embeds differently from the same code
asked about). Feature 3 has **no liveness probe** — `tests/test_feature_composition.py` asserts
the ontology raw table receives gradient in the combined configuration, and `tests/test_ontology.py`
asserts the mix arithmetic, but nothing measures ancestor-query sensitivity the way the RoPE and
bound probes do.

### 3. Adversarial review — reviewers who must refute, not confirm

Reviewers spread across the diff's dimensions, each briefed that the demonstrated failure mode is
a silent wrong label rather than a crash, and told to rank accordingly. Then a second pass whose
only job is to *refute* each finding by execution, defaulting to "refuted" when uncertain.

The refutations mattered: dismissed findings were plausible-sounding and wrong — one was a real
logging gap with no wrong-output consequence, another was already instrumented by the author's own
degenerate-bound reporter. Without the refute pass they would have been reported as defects.

### 4. Blast-radius arithmetic — size every finding before acting on it

A confirmed bug is not yet an actionable one. For each, compute how much it actually touches.
Example, for the prefix-absorption bug below: 399 of 13,908 leaf codes (2.87%) are affected,
5.97% of closure rows, and **2 of the 100 scored eval codes** — which is what establishes that
the published AUROC comparison survives it.

## What the checks found

The defects below are the ones that bear on these three features. All are silent; none crashes.

### Fixed on this branch

| # | Defect | Where |
|---|--------|-------|
| A | **The ancestor-mixed query universe silently dropped codes.** It was built by *sampling* a fixed 20,000 slots, which lost a long tail of the vocabulary — and on the real cohort lost `TIMELINE//END`, the code the censoring mechanism runs through. The universe is exact now: every leaf appears once, ancestors get the multiplicity that hits the target share. | `sample_query_sequences.py:build_query_universe` |
| B | **`RESERVED_CHARS` reserved parentheses.** They are structural only at the ends of a query string and the operator regex is anchored, so a name containing them round-trips fine. Reserving them dropped the ancestor names of 7,804 of 13,908 MIMIC-IV codes — `value_[4.0,6.0)`, `(MICU)` — to guard against the 10 codes carrying a real separator. | `query_vocab.py:RESERVED_CHARS` |
| C | **`bound_events` accepted only a literal list.** Real boundary codes carry spaces, periods and parentheses, which Hydra's override grammar cannot parse as a bare list, so the documented codes were unusable from the CLI. A YAML path works now, with the same vocabulary check. | `sample_query_sequences.py:resolve_bound_events` |

### Open — confirmed, not fixed here

These were confirmed during verification and are **deliberately out of scope** for this branch,
which is a feature port and a Feature-4 removal, not an ontology-correctness fix. They are
recorded here so they are not rediscovered from scratch.

| # | Defect | Where |
|---|--------|-------|
| 1 | **Prefix absorption.** A leaf code that is a strict `//`-prefix of another code stays a leaf but still receives closure rows from every descendant, so `explode_events_to_closure` makes its events include all descendants'. Every ordinary leaf query naming such a code silently changes meaning from "this exact code occurred" to "this code or any descendant occurred", flipping labels False→True. Fires whenever `ontology_dir` is set, **even at `ancestor_fraction=0.0`**. | `ontology.py` (closure construction) |
| 2 | **`parent_codes` edges are never transitively closed.** Declared parent edges are followed exactly one hop; the fixed-point loop closes *string prefixes*, not declared grouper chains. Any ancestor reachable only via a second declared edge is missing from both mix matrix and closure, so ancestor queries are labelled False for descendants more than one hop away. This truncates exactly the multi-level DAG that `parent_codes` exists to express. | `ontology.py` (ancestor map construction) |
| 3 | **The eval sampler ignores the ontology knobs.** `ontology_dir` and `ancestor_fraction` are shipped and documented in `sample_evaluation_query_sequences_config.yaml` and read by *nothing*. Hydra accepts the overrides silently. | `sample_evaluation_query_sequences.py` |
| 4 | **The eval grid never explodes events through the closure**, so ancestor queries in a grid would be labelled all-False while training labels them correctly. | `sample_evaluation_query_sequences.py` |
| 5 | **`strip_delta_tokens=True` + `use_rope_time=False` is silently accepted**, producing an encoder with *zero* elapsed-time information — delta tokens deleted, token-index positions. Trains, validates and checkpoints with normal-looking numbers. The reverse mismatch hard-errors; this direction has no check, even though `time_pos_ids` on a batch is an unambiguous signal the strip happened. | `conditional_model.py` |

3 and 4 are one root cause. Together they mean **you cannot currently build an eval grid that
exercises Feature 3.** That is why the earlier evaluation was atomic-only — not a design choice,
but the only thing the eval path can do.

## Does any of this invalidate the reported AUROC?

No — checked rather than assumed.

- **Atom invariance holds at exactly 0.0**, so the scored path is the trained model.
- The training run set **both** `strip_delta_tokens=true` and `use_rope_time=true`, so #5 did not
  fire.
- The eval grid was built without `ontology_dir` (#3 guarantees that), so its labels are the
  correct narrow ones.
- Prefix absorption touches **2 of the 100 scored codes** — too small to move a +0.007 macro gap
  across 100 tasks.

What it *does* change is the interpretation. The feature model was trained on partly-corrupted
ontology labels and still scored higher. So the result stands as "the diets did not hurt atomic
performance", and the ceiling for a correct implementation is probably higher — but "the features
work" remains unestablished, and #3–#4 are what currently prevent testing it.

## Running the checks

```bash
.venv/bin/python -m pytest tests/test_rope_strip_oracle.py \
  tests/test_feature_liveness.py \
  tests/test_feature_composition.py -q
```

A few seconds. The end-to-end CLI run is marked `slow` and excluded by default:

```bash
.venv/bin/python -m pytest -m slow tests/test_features_e2e_cli.py -q
```

## Suggested order of work

1. **#1 prefix absorption** — silently redefines ordinary leaf queries, fires at any
   `ancestor_fraction`, and corrupts training data. Cheapest to fix, widest reach.
2. **#3–#4 eval plumbing** — until this is wired, no experiment can measure Feature 3.
   Everything else about that feature is unmeasurable without it.
3. **#2 transitive closure**, then the footgun #5.
4. The two coverage gaps above: an oracle for event-bounded labeling, and a liveness probe for
   ancestor queries.
