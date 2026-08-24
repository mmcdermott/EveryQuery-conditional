# Ontology handoff — defects #1 and #2

Paused at your request on 2026-08-21. Nothing in `src/every_query/data/ontology.py` has been
touched. This is everything needed to start cold tomorrow.

---

## 1. The design question is mostly already answered — read this first

The previous session flagged a decision it thought had to be made before code was written:

> does a prefix-leaf keep its narrow meaning and get a separate ancestor node, or does the
> ancestor namespace get disambiguated?

**An invariant already in the code settles it.** `sample_query_sequences.py:396`:

```python
ancestors = sorted(nodes.filter(~pl.col("is_leaf"))["node"].to_list())
```

`build_query_universe` draws ancestor slots from **non-leaf nodes only**. So a leaf that happens
to be a `//`-prefix of another code was *never addressable as a subtree query in the first place*.
Its closure rows could only ever widen its own leaf query — which is precisely the bug.

Consequence: **excluding leaf-named nodes from the closure loses no capability.** There is no
tradeoff to weigh, no new node to mint, no namespace to relocate.

> **Correction to what I said in chat.** I claimed the closure-only option would "remove exactly
> the capability you want." That was wrong — I had not yet found this invariant. It removes
> nothing. It is now the recommended fix, and it is the smallest of the three options I offered.

What this buys you versus the alternatives:

| | closure-only (recommended) | separate ancestor node | global `//*` namespace |
|---|---|---|---|
| Fixes wrong labels | yes | yes | yes |
| Capability lost | **none** | none | none |
| New vocab nodes | 0 | ~399 | 0 (all renamed) |
| Ancestor indices move | no | appended only | **entire range** |
| Mix matrix / embedding sharing | unchanged | unchanged | unchanged |
| Existing ancestor query strings | unchanged | unchanged | **all change** |

The author's own regression test states the reasoning explicitly — see
`test_only_non_leaf_nodes_are_addressable_as_ancestors` in §5.

**`READMISSION//*` is optional ergonomics, not a requirement.** Since there is no bare
`READMISSION` code, `READMISSION` is already a pure ancestor node and already means all-cause.
If you still want the explicit spelling, accept `<prefix>//*` as an alias resolving to the same
node — `*` is not in `RESERVED_CHARS` (only `|>&`) and `READMISSION//*` does not match the
aggregate-operator regex, so it parses as a plain atom today. Quote it in shell/Hydra overrides
(`'READMISSION//*'`) or bash will try to glob it.

---

## 2. Defect #1 — prefix absorption

**Severity: highest. Corrupts training labels. Fires whenever `ontology_dir` is set, including at
`ancestor_fraction=0.0`.**

### Mechanism

`ontology.py:118-136` builds each code's ancestor map from `string_ancestors`. A name is added to
`ancestor_names` only when it is *not* a leaf (line 136):

```python
ancestor_names.update(a for a in amap if a not in leaf_names)
```

That correctly keeps a prefix-leaf out of the ancestor *node list*. But `name_to_index` still
resolves it — to its **leaf** index — so the mix row built at the bottom of `build_ontology`
still emits a component pointing at that leaf.

`build_closure` (`ontology.py:217`) then keeps every mix row whose `node_index` is a leaf:

```python
rows = mix_df.filter(pl.col("node_index").is_in(list(leaf_indices)))
```

So for leaf `A//B//C`, the pair `(code=A//B//C, node=A//B)` survives into the closure.
`explode_events_to_closure` (`ontology.py:297`) then duplicates every `A//B//C` event as an
`A//B` event.

Net effect: the ordinary leaf query `A//B` silently changes meaning from *"this exact code
occurred"* to *"this code **or any descendant** occurred"*, flipping labels False → True. No
crash, no warning, well-formed output parquet.

Blast radius (from the verification doc, measured on MIMIC-IV): 399 of 13,908 leaf codes (2.87%),
5.97% of closure rows, 2 of the 100 scored eval codes.

### Fix sketch

Exclude closure rows whose `node` resolves to a leaf name, keeping the self-pair. The self-pair
must survive — `(code=A//B, node=A//B)` is what makes a leaf query answerable at all. Guard
against "fixing" it by dropping `component_index == node_index`, which would silently break every
leaf query.

Leave `mix_df` alone. The embedding *sharing* between `A//B//C` and `A//B` is desirable and is not
what is broken — only the labeling closure is.

---

## 3. Defect #2 — `parent_codes` never transitively closed

### Mechanism

`ontology.py:126-134` follows a declared parent edge exactly one hop, then walks the *string*
prefixes of that grouper:

```python
if has_parents and row.get("parent_codes"):
    for pc in row["parent_codes"]:
        ...
        amap[pc] = min(amap.get(pc, 10**9), 1)           # one declared hop
        for d2, anc2 in enumerate(string_ancestors(pc), start=2):
            amap[anc2] = min(amap.get(anc2, 10**9), d2)  # string prefixes only
```

The fixed-point loop at `ontology.py:143-157` looks like it closes this, but it only calls
`string_ancestors(anc)` — it never consults `anc`'s own `parent_codes`. So a chain
`X -> P -> GRP//G` stops at `P`: `GRP//G` never enters `X`'s mix row or closure, and the ancestor
query `GRP//G` is labelled **False** for `X`.

This truncates exactly the multi-level DAG that `parent_codes` exists to express.

### Fix sketch

Make the fixed-point loop follow declared edges too, not just string prefixes, accumulating
minimum distance. Two things to get right:

- **Cycles.** `parent_codes` is caller-supplied data. `A -> B -> A` must terminate and must not
  make a node its own ancestor. There is already a test for this (§5).
- **Distance semantics.** The existing code treats a declared hop as distance 1 and the grouper's
  string prefixes as starting at 2. Preserve that — `decay**dist` sets the mix weights, so
  changing it silently reweights every embedding.

---

## 4. The trap — regenerate task parquets

Fixing #1 **changes labels**. Any task parquet already generated with `ontology_dir` set is now
wrong and must be regenerated.

`_ontology_fingerprint` in `sample_query_sequences.py` exists to detect exactly this. **Verify it
still trips after your change** — a stale-artifact detector that stops detecting is worse than
none. There is a ready-made test for it (§5).

---

## 5. Ready-made tests — salvage these first

`tests/test_four_features_regression.py` (untracked, in the main checkout at
`/home/gkondas/EveryQuery-conditional/`, **not** in this worktree) already contains red-test
demonstrations of both defects, written as properties rather than transcripts of a fix.

**Blocker:** line 42 imports `every_query.generate_tasks.aggregate_labeling`, which does not exist
on this branch (Feature 4 was removed). The file will not import as-is.

**Salvage:** drop the `aggregate_labeling` import and the aggregate-only tests, keep these:

| Test | Line | Covers |
|---|---|---|
| `test_a_leaf_that_prefixes_another_leaf_keeps_its_exact_meaning` | 244 | defect #1, with controls |
| `test_only_non_leaf_nodes_are_addressable_as_ancestors` | 265 | the invariant in §1 |
| `test_declared_parent_edges_are_transitively_closed` | 282 | defect #2, mix weights |
| `test_ancestor_query_two_declared_hops_up_is_labelled_true` | 305 | defect #2, label consequence |
| `test_a_cycle_in_declared_parents_terminates_and_excludes_self` | 319 | cycle safety |
| `test_relabel_fingerprint_distinguishes_two_ontologies` | 530 | the §4 trap |

Helpers `_events` (131) and `_answers` (142) come along with them.

The #1 test is well constructed — note it pairs the probe with **controls** (`A//B//C` still
labels True, ancestor `A` still rolls up), so it cannot pass by the closure simply being empty.
Keep that structure.

---

## 6. Environment defect found today — it silently invalidated test runs

**A bare `pytest` launched from a git worktree tested the MAIN checkout's code, not the worktree's.**

The editable install is a plain `.pth` file containing one absolute path — the main checkout's
`src`. A worktree shares that venv, so `import every_query` resolved to
`/home/gkondas/EveryQuery-conditional/src`, which sits on a *different branch*. Verified directly:

```
BARE   -> /home/gkondas/EveryQuery-conditional/src/every_query/__init__.py
PYPATH -> .../worktrees/three-feature-fixes/src/every_query/__init__.py
```

Consequences, both observed: green suites that proved nothing about the branch under test, and
mutation-testing runs that reported a mutation "caught" when the mutated file was never imported.
Nothing about it is visible in the output — the suite passes either way. Same silent-wrong-answer
shape as the defects this branch exists to fix.

**Fixed** by adding `pythonpath = ["src"]` to `[tool.pytest.ini_options]` in `pyproject.toml`, so
pytest always prefers the checkout it was launched from. Proof it works: `tests/test_rope_strip_guard.py`
asserts a `ValueError` that only this worktree's code raises, and it now passes under a bare
`pytest` with no `PYTHONPATH` set.

**If you work in a different worktree tomorrow, make sure that pyproject line came with you.**
Otherwise your ontology fix will appear to change nothing, because your tests will be importing
the old `ontology.py` from the main checkout.

Two more tooling traps, both of which produced fabricated results today:

- **`pytest-timeout` is NOT installed.** Passing `--timeout=...` makes pytest exit `rc=4` having
  run **zero** tests. An agent that treats a nonzero exit as "the test caught the mutation"
  produces an all-green mutation report out of thin air. This happened twice before it was caught.
- **`pytest-randomly` is absent**, so `-p no:randomly` is a silent no-op rather than an error.

---

## 7. The quality bar

Every defect on this branch produced a **silently wrong label** or a **silently dead feature** —
never a crash. The suite was green the whole time because it asserted shapes, dtypes and "runs
without error".

So: a test that only asserts a shape or that code does not raise is worthless here. The bar is a
test that **fails on a plausible wrong implementation**. Prove it by actually breaking the code
and watching it go red.

Two specific traps already paid for on this branch:

- Never assert liveness with a bare `assert not torch.equal(a, b)` — one ULP of float32 rounding
  (~1.19e-07) satisfies it. Use a margin (`LIVE = 1e-4` in the regression file).
- Assert at the level where the effect lives. A randomly-initialised decoder and head compress an
  8e-05 encoder difference to ~1e-07 at the logits — indistinguishable from noise. Read
  `last_hidden_state` directly. This already caused one false negative.

---

## 8. Facts established today

- **No bare `READMISSION` code exists** — only `READMISSION//<child>` variants. So `READMISSION`
  is a pure ancestor node, defect #1 **never fired on the all-cause use case**, and any existing
  all-cause labels for it are trustworthy.
- `RESERVED_CHARS` is `frozenset("|>&")`. `*` is legal in a query string.
- `string_ancestors` splits on the two-character `//` only; a single `/` is not a separator, so
  `ICD10CM/A04.72` has no ancestors.
- What actually blocked the all-cause experiment was **not** the ontology bugs but defects #3/#4
  (the eval sampler ignored `ontology_dir` entirely). Those are fixed — see below.

---

## 9. What landed while this was paused

Branch `fix/three-feature-defects`, worktree `.claude/worktrees/three-feature-fixes`.

- **Defects #3 + #4 — eval plumbing.** `sample_evaluation_query_sequences.py` had *zero* ontology
  code; `ontology_dir` and `ancestor_fraction` were documented in its config and read by nothing,
  and the eval grid never exploded events through the closure. Two independent silent failures:
  the sampled universe drew only leaf codes, and a designed spec naming an ancestor node was
  labelled all-False. Now wired to mirror the training path.
- **Defect #5 — RoPE footgun.** `strip_delta_tokens=True` with `use_rope_time=False` was silently
  accepted, producing an encoder with zero elapsed-time information that trains and checkpoints
  with normal-looking numbers. Now a hard error, symmetric with the existing opposite guard.
- **Coverage gap — event-bounds differential oracle.** An independent plain-Python
  reimplementation of `label_with_event_bounds` compared against the optimised one over
  randomised inputs, following the `test_rope_strip_oracle.py` pattern.

Deliberately **not** done, because it would bake in pre-fix closure semantics: the Feature 3
ancestor-query liveness probe. That is the remaining coverage gap once #1 and #2 land.

---

## 10. Suggested order tomorrow

1. Salvage the tests in §5 into `tests/test_ontology_defects.py`. Confirm they go **red** against
   today's code — that is your definition of done.
2. Fix #1 (closure-only, §2). Cheapest, widest reach, corrupts training data.
3. Fix #2 (§3). Watch the cycle case and the distance semantics.
4. Confirm `_ontology_fingerprint` still trips (§4), then regenerate any ontology task parquet.
5. Add the Feature 3 liveness probe (§9), following `tests/test_feature_liveness.py`.

Before step 1, confirm `pyproject.toml` in whatever checkout you use carries `pythonpath = ["src"]`
(§6). Without it your tests import `ontology.py` from the main checkout and your fix will appear
to do nothing.
