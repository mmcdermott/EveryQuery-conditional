# Ontology & evaluation audit

Branch `test/ontology-evaluation-suite`, from `dev` @ `5061e54`.
Cohort: MIMIC-IV MEDS 0.2.0, tensorized at `/home/gkondas/eq-vs-eq-cond/cohort/processed`
(13,908 codes, 228,945 / 28,513 / 28,709 subjects, 292 / 37 / 37 shards).

Everything below was established by reading and running the code on this branch, not from
documentation. Where the two disagree it is called out.

---

## 1. Discovered semantics

### 1.1 Duration-bounded queries

`answer = True` iff an event whose code matches the query occurs at a time strictly inside the
**open** interval `(t, t + d)`.

`label_binary_occurrence` — `sample_query_sequences.py:765`:

```python
left = index_df.with_columns((pl.col(pt) + pl.duration(microseconds=1)).alias("_pts"))
joined = left.join_asof(right, by=[sid, q], left_on="_pts", right_on=time, strategy="forward")
window_end = pl.col(pt) + pl.duration(days=pl.col(d))
answer = pl.col(time).is_not_null() & (pl.col(time) < window_end)
```

- **Lower bound open.** The asof key is shifted `+1µs`; timestamps are microsecond-precision, so
  `time >= t + 1µs` is exactly `time > t`. An event *at* `t` does not count.
- **Upper bound open.** The comparison is `<`, not `<=`. An event landing exactly on the horizon
  does not count.

Only the *first* occurrence at or after `t` is fetched, which is sound for an "any occurrence"
question: if the earliest candidate is beyond the horizon, none are inside.

**Units are days**, as a continuous `float32` — there is no discrete set of supported durations.
`sample_evaluation_query_sequences_config.yaml:107-109` draws log-uniformly over `[1, 731]`; the
config notes the median lands near 27d and ~52% of horizons are ≤ 30d. Consequence for the suite:
duration strata must be **buckets**, and designed specs must state explicit horizons.

### 1.2 Event-bounded queries

`label_with_event_bounds` — `sample_query_sequences.py:866`. Window is `(t, b)`, again open at both
ends, where `b` is the first boundary occurrence **strictly after** `t` (same `+1µs` shift, via
`_first_occurrence_after` at `:842`).

| question | answer |
|---|---|
| which boundary occurrence? | the **first** strictly after `t`; later ones are ignored |
| is the boundary event itself in the window? | **no** — upper bound is open |
| target and boundary share a timestamp? | target does **not** count |
| boundary never occurs? | window is left **open to the end of the record** — the query degenerates to "does this code ever occur again" |
| boundary code == query code? | unconditionally `False` (nothing is strictly before itself) |

The same-timestamp rule matters far more here than on the horizon: MEDS clusters many codes onto
one instant, so a discharge and everything charted with it routinely share a timestamp. The code
says so explicitly and pins it with one test, `test_a_query_at_the_exact_boundary_instant_does_not_count`.

The degenerate case is logged per boundary code at generation time by `log_degenerate_bounds`
(`:978`), which is the right call — a boundary that never fires produces a well-formed grid that
silently measures a different question.

### 1.3 Censoring

There is **no censor label and no censor head** in the conditional model. This contradicts the
framing in the task brief, which describes a separate censor loss and a censor head that overfits
around 32k steps. On this branch:

- answers are binary, never null (`schema.py`: `answers: large_list(bool)`);
- censoring is expressed as an ordinary query on `TIMELINE//END` — "does the record end within
  `d`?";
- the loss is a single masked BCE over real query positions (`conditional_model.py:475`), with
  padding the only thing masked out.

So "which examples contribute to the occurrence loss" is: **all unpadded query positions**, and
there is no separate denominator to get wrong. The single-query sampler `sample_tasks.py` *does*
use a three-valued `boolean_value` with `null` = censored, but that feeds the non-conditional
model and is not what this branch trains.

`docs/CONDITIONAL_QUERIES.md` documents the v1 leak post-mortem that led here, and the archived
`/experiments/.../README.md` states the window as `(t, t+d]` — **closed** upper bound. That is
stale: commit `038fc77` opened it at both ends. The in-repo docs were updated; the archive was not.

### 1.4 Query-form dispatch

`label_query_sequences` (`:1032`) is the single seam both training and evaluation flow through.
Dispatch is on the **frame**, not a config flag: an index carrying a `bound_event` column is
labelled with bounds, anything else with plain occurrence. Event-bounded rows carry
`EVENT_BOUND_DURATION_SENTINEL = -1.0` in place of a horizon (`assign_event_bounds`, `:1065`).

This answers "are training and evaluation labelling implementations different?" — **no, they are
the same function**, which is the right design and removes a whole class of drift.

### 1.5 Ontology

Two node kinds: **leaves** (real codes in `codes.parquet`) and **ancestor nodes** (names that exist
only as some leaf's parent). Ancestor indices are appended above the highest leaf index so leaf
indices are preserved (`ontology.py:65`). Edges come from two sources, both one hop:

- `//`-prefix parents (`string_ancestors`, `:47`) — the separator is the two-character `//`, so a
  single `/` is not a separator and `ICD10CM/A04.72` has no ancestors;
- declared `parent_codes` from cohort metadata — 5,123 of 13,908 codes carry them, max 2 parents,
  pointing at 2,498 pure groupers (`APR-DRG/021` and similar).

Labeling works by exploding each event into itself plus every ancestor above it
(`explode_events_to_closure`, `:457`), so "did any descendant of X occur" becomes an ordinary
occurrence question about X. The join is on `code` only, so **closure expansion cannot leak events
across subjects**. Codes absent from the closure are passed through unexploded with a warning
rather than being silently dropped by the inner join — a good guard.

Both target and boundary codes can be ancestors: the boundary search runs over the same exploded
stream.

`build_query_universe` (`:354`) draws ancestor query slots from **non-leaf nodes only**, and drops
the `TIMELINE` namespace as tautological.

---

## 2. Bugs found

### BUG-1 — prefix absorption in the closure (severity: high, corrupts training labels)

`build_closure` kept every mix component whose *node* was a leaf, so `(A//B//C → A//B)` survived
into the closure and `explode_events_to_closure` duplicated the event under `A//B`. The ordinary
leaf query `A//B` therefore changed meaning from "this exact code occurred" to "this code **or any
descendant** occurred" — no crash, no warning, a well-formed parquet of wrong labels. Fires
whenever `ontology_dir` is set, including at `ancestor_fraction=0.0`.

Measured on this cohort: **399 of 13,908 leaf codes (2.87%)** affected, **3,259 closure rows
(5.08%)**, and all 399 carry real events — 11.25M of them.

### BUG-2 — `parent_codes` never transitively closed (severity: medium, silently dead feature)

The fixed-point loop followed a declared edge exactly one hop and then only *string* prefixes, so a
chain `X → P → GRP//G` stopped at `P`: `GRP//G` never entered `X`'s closure and the ancestor query
`GRP//G` labelled `False` for every `X` event. This truncates exactly the multi-level DAG that
`parent_codes` exists to express.

This cohort has no declared chains of length ≥ 2 today, so the label impact here is zero — but the
fix still added **6 ancestor nodes and 2,935 mix entries**, because the corrected walk also reaches
groupers' own prefix parents that the old one missed.

### BUG-3 — addressability fused to sampling rate (severity: medium, blocks designed ancestor evals)

`build_query_universe` returns the vocabulary unchanged when `ancestor_fraction <= 0`
(`sample_query_sequences.py:385`), *even with `ontology_dir` set*. `validate_spec_codes` was
handed that same list, so at the default `ancestor_fraction=0` **no ancestor node was addressable
and every hand-written spec naming one was rejected as an unknown code.**

The only ancestor queries that got through were names that also happened to be real codes — i.e.
exactly the ones BUG-1 was silently widening. The prefix-absorption defect was *load-bearing* for
the designed-ancestor-spec capability, which is why the existing eval fixture was built on
dual-role codes.

A designed grid does not sample at all; it needs a name to resolve, nothing more. Fixed by
`addressable_codes` (`sample_evaluation_query_sequences.py:574`), which resolves addressability
against the ontology's node list and leaves the sampling universe untouched.

### Not a bug, but a coverage hole

7 ancestor nodes are dropped because their names contain grammar-reserved characters (`|`, `&`) —
e.g. `MEDICATION//Bupivacaine 0.1%|HYDROmorphone (Dilaudid)`. Those drug-class ancestors are not
queryable at all. It is logged as a warning, not silent, but worth knowing.

---

## 3. The dual-role decision

BUG-1's fix forces a semantic choice, because 399 names are **both** a real code and another
code's ancestor:

```
INFUSION_START//220949                    365,723 events   ← the unvalued infusion
  └─ INFUSION_START//220949//value_[…]    ~620k events     ← 10 quantile-binned variants
```

One string cannot mean both "exactly this code" and "the whole subtree" without one meaning
becoming unaskable. The nearest pure ancestor, `INFUSION_START`, covers all **635** infusion codes,
so it is not a substitute for the drug level.

**Decision (user, this session): mint a separate subtree node.** The leaf keeps its exact meaning
and a fresh ancestor node `<name>//ANY` means the subtree, so the whole ladder stays addressable:

| query | kind | codes | events |
|---|---|---|---|
| `INFUSION_START` | ancestor | 635 | 8,526,435 |
| `INFUSION_START//220949//ANY` | ancestor | 11 | 994,619 |
| `INFUSION_START//220949` | leaf | 1 | 365,723 |
| `INFUSION_START//220949//value_[-inf,4.50)` | leaf | 1 | 62,276 |

Cost is +399 nodes (20,665 → 21,064, +1.9%); `V_ext` 21,065. Controlled by
`subtree_suffix` in `build_ontology.yaml`; `null` restores purely-exact dual-role names.

Note this is *not* how the `HOSPITAL_ADMISSION` family behaves, because no bare
`HOSPITAL_ADMISSION` code exists — it is a pure ancestor and rolls up its 70 descendants under
every option. The same is true of `ICU_ADMISSION` (16), `TRANSFER_TO` (71) and
`HOSPITAL_DISCHARGE` (14). Trailing-slash spellings (`HOSPITAL_ADMISSION//`, `…//*`) are **not**
nodes and match nothing; the bare stem is the addressable name.

---

## 4. Correctness evidence

| artifact | what it establishes |
|---|---|
| `tests/ontology_suite/oracle.py` | independent labeller written from the prose contract; imports no production labelling or closure helper |
| `tests/ontology_suite/golden.py` | 5-subject fixture, 47-row hand-computed truth table covering every window edge case |
| `tests/test_ontology_golden.py` | oracle reproduces the table; production reproduces it; enabling the ontology moves no leaf answer |
| `tests/test_ontology_differential.py` | 120 randomised worlds (random DAGs incl. cycles, leaf-parents, multi-parents) compared row by row |
| `tests/test_ontology_embedding.py` | 22 tests: orientation, normalisation, identity mixing, gradient routing, cache staleness, HF install contract, checkpoint round-trip, dtype/device |

**Red-proof.** Against the genuine pre-fix module (`git show HEAD~1:…/ontology.py`), the
differential suite disagrees on **40 of 120 worlds, 71 of 2,880 query rows (2.47%)**, and the golden
suite fails 6 assertions. A suite that cannot fail is not evidence.

The truth table also caught an error in **my own** hand-computed censor column (two `d=30` rows
where the record ends inside the window), which is what the three-way check exists for.

---

## 5. Environment trap worth repeating

A plain `python script.py` from a worktree imports `every_query` from the **main checkout** via the
editable-install `.pth`, not the worktree. My first blast-radius measurement compared the old code
against itself and reported "0 rows changed". `pyproject.toml`'s `pythonpath = ["src"]` covers
pytest, but **not** ad-hoc scripts — those need an explicit `PYTHONPATH`. This is the same trap
documented in `docs/history/2026-08-21-ontology-handoff.md` §6, and it is still live.

---

## 6. Remaining risks

- **Task parquets generated with `ontology_dir` set before this branch are wrong** and must be
  regenerated. `_ontology_fingerprint` (`sample_evaluation_query_sequences.py:748`) exists to
  detect exactly this; it still trips, and the eval-plumbing suite covers it.
- **`V_ext` changed** (20,666 → 21,065 with `subtree_suffix=ANY`). Any checkpoint trained against
  the old ontology cannot be loaded against the new one — the encoder is sized from `V_ext`.
- **Durations are continuous**, so any "supported durations" list is a bucketing convention, not a
  property of the pipeline. Stated explicitly wherever the suite stratifies by duration.
- **DDP equality is asserted only as single-device determinism.** This box has one GB10, so a real
  two-rank comparison could not be run.
