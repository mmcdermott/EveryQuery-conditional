# Ontology support for `ConditionalMultitaskARModel`: the derived-at-load plan

Status: plan only, nothing implemented (2026-09-07). Base branch: `origin/dev`.

## The invariant this plan is built on

Under the multitask window rule (`resolved_start < t_v < resolved_end` for *some* occurrence of
`v`), an ancestor node's bit is exactly the OR of its descendant leaves' bits:

```
anc[b, k, a] = OR_{v in descendants(a)} leaf[b, k, v]
```

`descendants(a)` is the `event_to_query_nodes.parquet` closure, the same table the scalar
QuerySeq sampler already explodes events through. So ancestor bits carry no information beyond
the leaf row plus the closure, and storing them (+51% disk: 1,739 -> 2,634 bytes per window on
the 13,908 / 21,065 cohort) is pure redundancy.

Design consequences:

1. **`.labels.npy` stays leaf-only and byte-identical** to today. `vocab_size` in the manifest
   stays the cohort's V. `FORMAT_VERSION` is not bumped for leaf-only files.
2. **Ancestor targets are derived per batch, on the GPU, inside the model's dense forward.** The
   batch that crosses host to device stays `(B, K, V)`.
3. **The sampler learns about the ontology only where an ancestor must act as an *event*** (an
   ancestor-valued `start_event` / `bound_event`), and optionally as a conditioning code. Both are
   separate, later PRs. Training with ancestor targets and evaluating ancestor queries needs no
   sampler change at all.

## Delivery order

Four PRs into `dev`, each independently mergeable and each leaving every guard test below green.

| PR | Scope | What it unlocks |
|---|---|---|
| A | closure index + derivation, model, dataset/datamodule widths, configs | train the multitask model with ancestor targets on today's leaf labels |
| B | eval + predict plumbing, cross-path agreement test | score ancestor queries from QuerySeq grids |
| C | sampler: ancestor-valued boundaries, manifest provenance | windows bounded by "next occurrence of any X" |
| D (optional) | ancestor conditioning codes | teacher-forcing on ancestor answers |

C and D are deferrable; A + B is the headline.

## PR A: derived ancestor targets

### A1. `every_query/data/ontology.py`: closure as index pairs

- `load_closure_index(ontology_dir, base_vocab_size) -> ClosureIndex(leaf_ids, ancestor_ids, v_ext)`.
  Built from `event_to_query_nodes.parquet` joined to `ontology_vocab.parquet` token ids. Drops
  self pairs (`event_code == query_node`), keeps the dual-role `X//ANY` pairs (which include `X`
  itself). Validates: every `leaf_id < base_vocab_size`, every `ancestor_id` in
  `[base_vocab_size, v_ext)`, and the number of `is_observed_code` nodes equals `base_vocab_size`
  (the ontology was built from this cohort's `codes.parquet`).
- `derive_ancestor_targets(leaf: BoolTensor(B, K, V), closure, v_ext) -> BoolTensor(B, K, v_ext)`.
  Pure torch: gather the `leaf_ids` columns, `index_add_` into a zero `(B, K, n_anc)` buffer keyed
  by `ancestor_ids - V`, threshold `> 0`, concatenate after the leaf block. Works on CPU and GPU,
  no autograd needed. About 50k pairs on the real cohort, sub-millisecond per batch.
- `closure_fingerprint(ontology_dir) -> str`: lift the closure half of
  `sample_evaluation_query_sequences._ontology_fingerprint` into `ontology.py` so the sampler
  (PR C), the eval sampler, and checkpoints digest the closure the same way.

### A2. `ConditionalMultitaskARModel`

- Accept `ontology_dir`; call `wrap_tok_embeddings(self, load_mix_matrix(ontology_dir))` exactly as
  `ConditionalARModel` does; register `closure.leaf_ids` / `closure.ancestor_ids` as
  non-persistent buffers; record `ontology_dir` in hparams (already in `self._init_kwargs`).
- Expose `base_vocab_size` (V, the cohort) beside `vocab_size` (V_ext, the table width). Without an
  ontology they are equal.
- `forward`: if `batch.targets.shape[-1] == base_vocab_size < vocab_size`, replace `targets` with
  `derive_ancestor_targets(...)` before the existing width check. A `(B, K, V_ext)` batch is also
  accepted unchanged (tests feed those directly).
- Readout uses the **effective** table: `OntologyEmbedding.weight` deliberately returns the raw
  table, so both `forward` and `score_final_query` must switch to `mixed_weight()` when the
  embedding is wrapped. Otherwise ancestor logits would read raw rows while the input side reads
  mixed rows. Add a small `_readout_weight()` helper used by both.
- `code_bias` is sized to `vocab_size` (V_ext) already via `HF_model_config.vocab_size`.

### A3. Dataset and datamodule widths

Today `dataset_kwargs.expected_vocab_size` is interpolated from `config_overrides.vocab_size`,
which `train.py` overwrites with V_ext when an ontology is set. That would make the training
dataset reject its own leaf manifest.

- `MultitaskBoundaryPytorchDataset.expected_vocab_size` keeps its meaning: the manifest's leaf V.
- `ConditionalMultitaskDataModule` gains `ontology_dir` (in `dataset_kwargs`, mirroring the scalar
  datamodule). It passes `expected_vocab_size` = leaf V to the training dataset and
  `expected_vocab_size` = V_ext plus `ontology_dir` to the eval adapter (PR B).
- `train.py`: after computing `v_ext`, write both numbers explicitly into the resolved config
  (`config_overrides.vocab_size = v_ext`, `datamodule.dataset_kwargs.expected_vocab_size =
  cohort V`) instead of relying on the interpolation. `resolved_config.yaml` then records both
  widths, which `predict_multitask` needs (PR B).
- Batches, `collate`, `gather_packed`, `unpack_targets`, and the `condition_answers`
  consistency check are untouched.

### A4. Configs and docs

- `conditional_multitask_ar_config.yaml` and the demo config: `lightning_module.model.ontology_dir:
  null` and `datamodule.dataset_kwargs.ontology_dir: ${lightning_module.model.ontology_dir}`.
- Update the "reserved / not supported" prose in `conditional_multitask_ar_model.py`,
  `predict_multitask.py` (line ~147), `sample_multitask_sequences.py` module docstring, and the
  model `README.md`.

### A5. Tests for PR A

Flip (the guards that pin the old behaviour):

- `tests/test_conditional_multitask_ar_model.py::test_validation_errors_without_dataclass_reconstruction`
  (line 373: `tiny_model(ontology_dir=...)` raises) becomes a positive construction test.
- `tests/test_conditional_multitask_ar_model.py::test_configs_and_position_budget` (line 448:
  `"ontology_dir" not in dataset_kwargs`) asserts the interpolation instead.

New:

- `test_derive_ancestor_targets_matches_brute_force_or`: random leaf bits, compare against a
  Python loop over the closure. Include a dual-role `X` / `X//ANY` node, a multi-parent leaf
  (`parent_codes`), a leaf with no ancestors, and the PAD column (must stay false and never be an
  ancestor).
- `test_derive_ancestor_targets_agrees_with_the_oracle`: on the `tests/ontology_suite` golden
  cohort, label leaf bits with the multitask labeler (no ontology), derive, and compare each
  ancestor query against `tests/ontology_suite/oracle.py` for the same window. This is the
  proof that derivation equals the scalar path's explosion semantics.
- `test_closure_index_rejects_a_foreign_ontology`: mismatched leaf count / leaf id >= V raise.
- `test_forward_with_leaf_targets_equals_forward_with_extended_targets`: same batch, once at V
  (derived in-model) and once pre-derived to V_ext, identical loss and logits.
- `test_tied_readout_uses_the_mixed_table_under_an_ontology`: extend
  `test_tied_readout_identity_and_gradient`; logits equal `hidden @ (A @ W).T + bias`; gradient
  reaches raw ancestor rows through the mix.
- `test_score_final_query_on_an_ancestor_code_matches_the_dense_forward`: extend
  `test_score_final_query_matches_gathered_full_vocab_logits` with `scored_codes >= V`.
- `test_checkpoint_round_trip_with_an_ontology`: variant of
  `test_lightning_predict_and_checkpoint_round_trip`; `ontology_dir` survives in hparams and the
  reloaded model derives identically.
- `test_training_dataset_width_is_the_leaf_manifest_under_an_ontology`: datamodule with
  `ontology_dir` set constructs the training dataset against a leaf manifest and passes V_ext
  only to the eval adapter.

## PR B: evaluating ancestor queries

- `QuerySeqMultitaskEvalDataset.__init__` passes `ontology_dir` through to the parent instead of
  the hard-coded `None`, so `extend_code_map` makes ancestor query / start / bound names
  resolvable, and takes `expected_vocab_size` = V_ext for its `idx.max()` check.
- `predict_multitask.build_predict_datamodule`: read `ontology_dir` and both widths from the
  checkpoint's `resolved_config.yaml`; check `data_cfg.vocab_size == base V` (cohort) and
  `model.vocab_size == V_ext` (ontology), with distinct error messages.
- `EQ_predict_multitask` refuses a grid whose provenance sidecar records a different
  `ontology_fingerprint` than the checkpoint's closure (the eval sampler already writes it).

Tests for PR B:

- `test_multitask_eval_and_scalar_grid_agree_on_the_same_ancestor_query` (the multitask analog of
  `tests/test_eval_ontology_plumbing.py::test_eval_and_training_paths_agree_on_the_same_ancestor_query`):
  generate a QuerySeq grid with an ontology, generate multitask training labels without one,
  derive, and assert the derived training bit equals the grid label for matching
  (subject, prediction time, window, ancestor) rows.
- `test_predict_multitask_scores_an_ancestor_query` (extend
  `tests/test_conditional_multitask_cli.py::test_predict_multitask_scores_a_queryseq_grid_with_active_starts`).
- `test_build_predict_datamodule_separates_cohort_width_from_table_width` (extend
  `tests/multitask/test_predict_multitask_logic.py::test_build_predict_datamodule_checks_the_checkpoint_against_the_cohort`).
- `test_codes_past_the_checkpoint_vocabulary_width_are_rejected` keeps passing with V_ext as the
  width.

## PR C: ancestor-valued boundaries in the sampler

Only needed for windows like "until the next occurrence of any `LAB//220645//*`".

- Seam 1, `build_target_vocabulary`: bits stay leaf V. Return an additional boundary vocabulary
  (`extend_code_map` over the ontology) so `_encode_bounds` can resolve ancestor names to ids in
  `[V, V_ext)`.
- Seam 2, `prepare_events_for_labeling`: with an ontology, `expand_events_to_query_nodes` (the
  scalar sampler's function, already oracle-tested). `_encode_events` must map ancestor names via
  the extended map, and `build_interval_table` is built with `vocab_size = V_ext` so ancestor
  intervals exist for boundary resolution. The packed kernel allocates its dense chunk
  `(rows, K, vocab_size)` and scatters by code index, so it must keep receiving the leaf V and
  the ancestor rows (`code_index >= V`) must be masked out of the table it labels from. Two
  options: one V_ext table plus a leaf mask on the labeling side (recommended), or a leaf-only
  table for labeling and a second expanded table used only by `resolve_*`. Stage 4M RAM grows by
  the mean closure depth (~3 to 4x per shard); disk does not.
- Seam 3, `resolve_event_boundaries`: unchanged in shape; `resolve_start_times` /
  `resolve_end_times` already resolve "first occurrence of code id after t" and the expanded
  table makes an ancestor id an ordinary code.
- Manifest: `ontology_mode` gains `"boundaries"` beside `"none"`, plus `ontology_fingerprint`
  (from A1's `closure_fingerprint`) and `boundary_vocab_size` (V_ext). `vocab_size`,
  `packed_width_bytes`, `vocab_fingerprint` and `VOCAB_FINGERPRINT_VERSION` do not change.
  `validate_manifest` accepts both modes; the reuse gate compares `ontology_fingerprint`.
- `MultitaskBoundaryPytorchDataset` accepts `q_start_codes` / `q_bound_codes` in `[V, V_ext)` when
  the manifest is in `"boundaries"` mode and the model width covers them.
- `reject_ontology` is removed; the hard error becomes "manifest mode vs. model ontology mismatch".

Tests for PR C:

- Flip `tests/multitask/test_multibound_labeling.py::test_ontology_dir_raises` and
  `tests/multitask/test_multitask_orchestration.py::test_ontology_dir_raises_before_stage0` into
  positive tests.
- Flip `test_events_are_not_closure_expanded` into "not expanded without an ontology, expanded
  through the closure with one".
- `test_ontology_leaves_leaf_bits_byte_identical`: run the sampler on the synthetic cohort with
  `ontology_dir=None` and with an ontology whose boundary draws are disabled; `.labels.npy` files
  are byte-equal and the manifests differ only in the ontology keys. This is the permanent proof
  of the invariant at the top.
- `test_ancestor_bounded_window_matches_the_scalar_labeler`: extend
  `test_differential_against_scalar_label_with_event_bounds` with an ancestor bound code; the
  scalar oracle is `label_with_event_bounds` on the exploded stream.
- `test_ancestor_interval_never_writes_a_leaf_column`.
- Manifest tests: `test_config_fingerprint_covers_every_start_parameter_and_vocab_salt_is_pinned`
  stays green (salt untouched); add `test_ontology_fingerprint_change_relabels`.

## PR D (optional): ancestor conditioning codes

- Stage 1M draws `condition_codes` from the extended node pool; Stage 4M computes an ancestor's
  `condition_answers[i, j]` as the OR of the row's descendant bits (the packed row is in hand).
- `collate`'s consistency check derives the same OR for slots with `condition_codes >= V`
  (reuse `derive_ancestor_targets` on the `(B, K-1)` slots only).
- Tests: extend `test_condition_answer_is_the_target_bit_at_the_matching_boundary` and
  `test_corrupted_condition_answers_fail_in_collate` with ancestor slots.

## Guard tests: keep green before and after every PR

Run the three suite splits on the base commit first and record the pass counts; re-run after
each PR. Commands (from the repo root, see the per-machine notes in memory):

```
uv run pytest tests/multitask tests/sampler
uv run pytest tests --ignore=tests/multitask --ignore=tests/sampler
uv run pytest src
```

**Label bits and window semantics** (`tests/multitask/test_multibound_labeling.py`, 43 tests, all
but `test_ontology_dir_raises` unchanged):

- `test_zero_start_duration_reproduces_existing_labels_byte_identically`
- `test_differential_against_scalar_label_with_event_bounds`
- `test_fuzz_window_starts_against_naive_and_scalar_oracles`
- `test_output_identical_for_different_initial_context_order_and_chunk_sizes`
- `test_chunked_equals_unchunked_and_supplied_order_invariance_with_starts`
- `test_supplied_out_memmap_is_filled_in_place`
- `test_condition_answer_is_the_target_bit_at_the_matching_boundary`
- `test_pad_bit_stays_false_and_unknown_event_codes_are_ignored`
- `test_vocab_size_not_divisible_by_eight_and_absent_codes_keep_bits_false`
- `test_all_six_start_end_combinations_in_one_context`

**Sampler orchestration, manifest, reuse gate** (`tests/multitask/test_multitask_orchestration.py`):

- `test_run_writes_layout_manifest_and_exact_count` (asserts `ontology_mode == "none"` for a
  no-ontology run; must stay true)
- `test_fixed_seed_reproducibility_and_reuse`
- `test_stale_fingerprint_relabels`, `test_config_change_invalidates_labels`,
  `test_vocabulary_change_invalidates_output`, `test_worker_rejects_manifest_vocab_mismatch`
- `test_config_fingerprint_covers_every_start_parameter_and_vocab_salt_is_pinned`
- `test_driver_owns_manifest_worker_never_writes_it`, `test_interrupted_write_recovery`
- `test_vocabulary_is_base_codes_only_and_rejects_lists`

**Dataset alignment** (`tests/test_multitask_dataset_integration.py`, all 17):

- `test_end_to_end_cli_run_to_batch_matches_raw_events`
- `test_end_to_end_run_with_starts_to_batch_matches_raw_events`
- `test_metadata_row_maps_to_matching_packed_row`, `test_global_shuffle_preserves_alignment_and_batch_shape`
- `test_corrupted_condition_answers_fail_in_collate`
- `test_manifest_vocab_mismatch_raises`, `test_dataset_matches_manifest_vocabulary` (meaning: leaf V)
- `test_split_mixing_format_2_and_3_shards_loads` (legacy files keep loading; no format bump)
- `test_upstream_reordering_is_an_error_not_a_silent_misalignment`

**Model** (`tests/test_conditional_multitask_ar_model.py`, 24):

- `test_tied_readout_identity_and_gradient` (no-ontology identity must hold exactly)
- `test_pad_target_is_excluded_from_loss`, `test_no_valid_elements_has_differentiable_zero_loss`
- `test_score_final_query_matches_gathered_full_vocab_logits`,
  `test_score_final_query_never_reads_targets_or_builds_dense_logits`,
  `test_score_final_query_validates_codes_and_mask`
- `test_lightning_predict_and_checkpoint_round_trip`
- `test_training_forward_is_the_window_hidden_states_projection`
- `test_lazy_package_exports_do_not_cycle` (the closure loader must stay a lazy import)

**Lightning loops** (`tests/multitask/test_conditional_multitask_lightning.py`, all 7). The
derivation lives in `forward`, so these two are the ones that would catch it leaking:

- `test_evaluation_loops_never_call_the_dense_forward`
- `test_evaluation_loops_never_build_a_vocabulary_wide_projection`

**Datamodule and predict** (`tests/multitask/test_conditional_multitask_datamodule.py` 10,
`tests/multitask/test_predict_multitask_logic.py` 23, `tests/test_conditional_multitask_cli.py` 8):

- `test_dataset_kwargs_and_max_windows_reach_the_eval_adapter`
- `test_demo_train_config_instantiates_the_datamodule`, `test_hyperparameters_record_the_grid_settings`
- `test_build_predict_datamodule_checks_the_checkpoint_against_the_cohort`
- `test_codes_past_the_checkpoint_vocabulary_width_are_rejected`
- `test_run_inference_never_calls_the_dense_forward_or_projects_onto_the_vocabulary`
- `test_conditional_multitask_train_checkpoint_and_reload`,
  `test_predict_multitask_scores_a_queryseq_grid_with_active_starts`,
  `test_predict_multitask_accepts_a_pre_issue_30_run_dir`

**Ontology core, unchanged code but shared helpers** (`tests/test_ontology.py` 28,
`tests/test_ontology_golden.py` 7, `tests/test_ontology_differential.py` 2,
`tests/test_ontology_embedding.py` 21, `tests/test_eval_ontology_plumbing.py` 17). In particular:

- `test_closure_pairs_each_leaf_with_itself_and_its_ancestors`
- `test_a_name_that_is_both_code_and_prefix_keeps_its_leaf_index`
- `test_enabling_ontology_does_not_move_leaf_answers`
- `test_production_agrees_with_oracle`, `test_ontology_off_matches_identity_ontology`
- `test_eval_and_training_paths_agree_on_the_same_ancestor_query`
- `test_ontology_fingerprint_separates_both_of_its_halves` (if `_ontology_fingerprint` is split,
  its closure half must digest identically)
- `test_eval_ontology_config_keys_match_training_and_are_actually_consumed` (checks only the
  scalar sampler pair today; PR C adds the same `ontology_dir` default assertion for
  `sample_multitask_sequences_config.yaml`)

**Scalar sampler regression** (`tests/sampler/test_stage4_labeling.py` 31,
`tests/sampler/test_eval_grid_per_shard.py` 16): untouched code, but PR C shares
`expand_events_to_query_nodes` and PR A moves the fingerprint helper.

## Byte-identity check outside the test suite

Before PR A lands, generate multitask labels for the demo synthetic cohort with the current code
and record `sha256sum` of every `.labels.npy` and `_multitask_manifest.json` in the PR
description. After each PR, regenerate and compare. The hashes must be equal through PR A and B
(no sampler change) and, for PR C with `ontology_dir=None`, equal as well. Record patient-free
hashes only; never paste shard contents.

## Modeling caveats to carry into the first experiment

- Ancestor columns are denser positives, so the dense BCE mix shifts toward them. Consider a
  prevalence-based per-column init for `code_bias` instead of the flat `-3.0`.
- The +17.5% embedding-parameter capacity confound noted for the scalar ontology applies here
  too; the flat-vocab control has still not been run.
- The zero-sum query budget does not apply to PR A (no query slots are spent), which is a point
  in favour of derived targets: every leaf window now also supervises its ancestors for free.
