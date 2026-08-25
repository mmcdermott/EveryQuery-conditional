"""Tests for ontology / hierarchical embeddings.

Covered in pipeline order:

1. DAG construction — prefix semantics, index layout (leaves untouched, ancestors appended),
   the "every indexed node has a mix row" invariant, decay, and the two hazards this port fixes
   (reserved characters in ancestor names, unknown codes silently dropped by the closure join).
2. Embedding wrapper — the mix arithmetic, gradient reaching ancestor rows, and the per-forward
   cache actually being a cache.
3. Query addressing — ancestors becoming queryable, and the universe mixing that keeps the
   sampler's RNG contract intact.
4. Model integration — that the wrapper composes with the query slots rather than only the
   patient encoder.
"""

import polars as pl
import pytest
import torch

from every_query.data.ontology import (
    CLOSURE_FILE,
    MIX_FILE,
    NODES_FILE,
    build_closure,
    build_ontology,
    explode_events_to_closure,
    extend_code_map,
    extended_vocab_size,
    load_mix_matrix,
    string_ancestors,
)
from every_query.generate_tasks.sample_query_sequences import build_query_universe
from every_query.model.conditional_model import ANSWER_NO, ANSWER_YES, ConditionalQueryModel
from every_query.model.ontology_embedding import OntologyEmbedding, wrap_tok_embeddings


def _codes(codes, parents=None) -> pl.DataFrame:
    data = {"code": codes, "code/vocab_index": list(range(1, len(codes) + 1))}
    if parents is not None:
        data["parent_codes"] = parents
    return pl.DataFrame(data)


def _write_ontology(tmp_path, codes_df, decay: float = 0.5):
    nodes, mix = build_ontology(codes_df, decay=decay)
    closure = build_closure(nodes, mix)
    nodes.write_parquet(tmp_path / NODES_FILE)
    mix.write_parquet(tmp_path / MIX_FILE)
    closure.write_parquet(tmp_path / CLOSURE_FILE)
    return nodes, mix, closure


# ── 1. DAG construction ─────────────────────────────────────────────────


def test_single_slash_is_not_a_separator():
    """MEDS uses `//` between levels; a lone `/` inside a code is part of the name."""
    assert string_ancestors("ICD10CM/A04.72") == []
    assert string_ancestors("LAB//220645//mEq/L") == ["LAB//220645", "LAB"]


def test_leaf_indices_are_preserved():
    """An ontology must be droppable onto an existing cohort without renumbering it."""
    nodes, _ = build_ontology(_codes(["A//B//C", "A//B//D", "E"]))
    leaves = nodes.filter(pl.col("is_leaf")).sort("vocab_index")
    assert leaves["node"].to_list() == ["A//B//C", "A//B//D", "E"]
    assert leaves["vocab_index"].to_list() == [1, 2, 3]


def test_ancestors_are_appended_above_the_highest_leaf():
    nodes, _ = build_ontology(_codes(["A//B//C", "E"]))
    max_leaf = nodes.filter(pl.col("is_leaf"))["vocab_index"].max()
    assert (nodes.filter(~pl.col("is_leaf"))["vocab_index"] > max_leaf).all()


def test_every_indexed_node_has_a_mix_row():
    """Otherwise the node embeds to the zero vector, permanently and silently."""
    nodes, mix = build_ontology(_codes(["A//B//C", "X//Y"], parents=[["G//H//I"], None]))
    assert set(nodes["vocab_index"].to_list()) == set(mix["node_index"].unique().to_list())


def test_parent_codes_prefixes_are_closed_to_a_fixed_point():
    """A grouper's own prefixes, and their prefixes, must all become nodes."""
    nodes, _ = build_ontology(_codes(["LEAF"], parents=[["G//H//I"]]))
    ancestors = set(nodes.filter(~pl.col("is_leaf"))["node"].to_list())
    assert {"G//H//I", "G//H", "G"} <= ancestors


def test_decay_controls_ancestor_weight():
    """Decay=0 is the structure-without-mixing control: each node is only itself."""
    _, mix_half = build_ontology(_codes(["A//B//C"]), decay=0.5)
    _, mix_zero = build_ontology(_codes(["A//B//C"]), decay=0.0)

    def leaf_components(mix):
        rows = mix.filter((pl.col("node_index") == 1) & (pl.col("weight") > 0))
        return set(rows["component_index"].to_list())

    assert leaf_components(mix_zero) == {1}, "no ancestor may carry weight at decay=0"
    # At decay=0.5 the leaf mixes itself plus both prefixes (A//B, A).
    assert len(leaf_components(mix_half)) == 3


def test_reserved_characters_are_kept_out_of_the_ancestor_pool():
    """An ancestor whose name carries a grammar separator could never be queried back."""
    nodes, _ = build_ontology(_codes(["A&B//C", "PLAIN//X"]))
    ancestors = nodes.filter(~pl.col("is_leaf"))["node"].to_list()
    assert "A&B" not in ancestors
    assert "PLAIN" in ancestors


def test_parenthesised_ancestor_names_are_not_dropped():
    """Parentheses are not separators, and most of a real MEDS vocabulary contains them.

    Reserving them cost the ancestor names of 7,804 of 13,908 MIMIC-IV codes: value-bin
    segments like ``value_[4.0,6.0)`` and unit names like ``(MICU)`` are ordinary parts of a
    code, not grammar, and an ancestor built from them is perfectly addressable.
    """
    codes = ["ICU//STAY (MICU)//LOS", "LAB//GLUCOSE//value_[4.0,6.0)"]
    nodes, _ = build_ontology(_codes(codes))
    ancestors = set(nodes.filter(~pl.col("is_leaf"))["node"].to_list())
    assert "ICU//STAY (MICU)" in ancestors
    assert "LAB//GLUCOSE//value_[4.0,6.0)" not in ancestors, "that one is a leaf, not an ancestor"
    assert "LAB//GLUCOSE" in ancestors


def test_mix_rows_are_normalised(tmp_path):
    _write_ontology(tmp_path, _codes(["A//B//C", "A//B//D"]))
    mix = load_mix_matrix(tmp_path)
    row_sums = torch.sparse.sum(mix, dim=1).to_dense()
    nonzero = row_sums[row_sums > 0]
    assert torch.allclose(nonzero, torch.ones_like(nonzero), atol=1e-6)


def test_closure_pairs_each_leaf_with_itself_and_its_ancestors(tmp_path):
    _, _, closure = _write_ontology(tmp_path, _codes(["A//B//C"]))
    pairs = set(zip(closure["code"].to_list(), closure["node"].to_list(), strict=True))
    assert ("A//B//C", "A//B//C") in pairs
    assert ("A//B//C", "A//B") in pairs
    assert ("A//B//C", "A") in pairs


def test_explode_keeps_events_the_ontology_does_not_know():
    """An inner join here would silently DELETE data when the ontology is out of date."""
    from datetime import datetime

    events = pl.DataFrame(
        {"subject_id": [1, 1], "time": [datetime(2024, 1, 1)] * 2, "code": ["A//B", "ORPHAN"]}
    )
    closure = pl.DataFrame({"code": ["A//B", "A//B"], "node": ["A//B", "A"]})
    out = explode_events_to_closure(events, closure)
    assert "ORPHAN" in out["code"].to_list(), "an unknown code must survive, not vanish"
    assert set(out["code"].to_list()) == {"A//B", "A", "ORPHAN"}


# ── 2. embedding wrapper ────────────────────────────────────────────────


def _two_node_wrapper():
    raw = torch.nn.Embedding(2, 3)
    raw.weight.data.copy_(torch.tensor([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]]))
    mix = torch.sparse_coo_tensor(
        torch.tensor([[0, 1, 1], [0, 0, 1]]), torch.tensor([1.0, 0.5, 0.5]), (2, 2)
    ).coalesce()
    return OntologyEmbedding(raw, mix), raw


def test_wrapper_returns_the_ancestor_mixed_average():
    emb, _ = _two_node_wrapper()
    assert torch.allclose(emb(torch.tensor([1])), torch.full((1, 3), 2.0))


def test_wrapper_handles_multidimensional_ids():
    """Aggregate component blocks are (B, L, K); the wrapper must index them in one call."""
    emb, _ = _two_node_wrapper()
    assert emb(torch.zeros(2, 4, 3, dtype=torch.long)).shape == (2, 4, 3, 3)


def test_gradient_reaches_ancestor_rows():
    """An ancestor never appears in a patient stream; it can only learn through the mix."""
    emb, raw = _two_node_wrapper()
    emb(torch.tensor([1])).sum().backward()
    assert raw.weight.grad[0].abs().sum() > 0, "the ancestor row must receive gradient"


def test_cache_is_reused_within_a_forward_and_cleared_between():
    emb, _ = _two_node_wrapper()
    first = emb.mixed_weight()
    assert emb.mixed_weight() is first, "the mixed table must be computed once per forward"
    emb.clear_cache()
    assert emb.mixed_weight() is not first, "clear_cache must drop the cached product"


def _identity_mix(v_ext: int) -> torch.Tensor:
    return torch.sparse_coo_tensor(torch.tensor([[0], [0]]), torch.tensor([1.0]), (v_ext, v_ext)).coalesce()


def test_wrap_rejects_an_undersized_table():
    """Sizing the encoder to V rather than V_ext puts every ancestor index out of range."""
    model = _tiny_model(vocab_size=4)
    with pytest.raises(ValueError, match="V_ext"):
        wrap_tok_embeddings(model, _identity_mix(16))


def test_wrap_rejects_an_oversized_table():
    """``(V_ext, V_ext) @ (V_model, H)`` needs equality, not just enough rows.

    An encoder left at ModernBERT's own 50k vocabulary satisfies "big enough" while being just
    as wrong as an undersized one — and would fail deep inside ``torch.sparse.mm`` instead.
    """
    model = _tiny_model(vocab_size=99)
    with pytest.raises(ValueError, match="V_ext"):
        wrap_tok_embeddings(model, _identity_mix(16))


# ── 3. query addressing ─────────────────────────────────────────────────


def test_ancestors_become_queryable(tmp_path):
    _write_ontology(tmp_path, _codes(["A//B//C"]))
    extended = extend_code_map({"A//B//C": 1}, tmp_path)
    assert "A//B" in extended and "A" in extended


def test_a_name_that_is_both_code_and_prefix_keeps_its_leaf_index(tmp_path):
    _write_ontology(tmp_path, _codes(["A", "A//B"]))
    extended = extend_code_map({"A": 1, "A//B": 2}, tmp_path)
    assert extended["A"] == 1, "a real code's canonical index must win over any node index"


def test_extended_vocab_size_covers_every_node(tmp_path):
    nodes, _, _ = _write_ontology(tmp_path, _codes(["A//B//C", "D//E"]))
    assert extended_vocab_size(tmp_path) == int(nodes["vocab_index"].max()) + 1


def test_query_universe_is_untouched_when_the_feature_is_off():
    assert build_query_universe(["A", "B"]) == ["A", "B"]
    assert build_query_universe(["A", "B"], ontology_dir=None, ancestor_fraction=0.5) == ["A", "B"]


def test_query_universe_mixes_in_ancestors(tmp_path):
    _write_ontology(tmp_path, _codes(["A//B//C", "D//E//F"]))
    universe = build_query_universe(
        ["A//B//C", "D//E//F"], ontology_dir=tmp_path, ancestor_fraction=0.5, seed=3
    )
    ancestors = {"A//B", "A", "D//E", "D"}
    share = sum(1 for c in universe if c in ancestors) / len(universe)
    assert abs(share - 0.5) < 0.05, f"expected ~50% ancestors, got {share:.2f}"


def test_query_universe_never_drops_a_leaf_code(tmp_path):
    """A leaf missing from the universe is a code the model is never asked about.

    An earlier, sampled construction lost a long tail of the vocabulary — and on the real
    cohort it lost ``TIMELINE//END``, which is this model's entire censoring mechanism.
    """
    leaves = [f"LAB//{i}" for i in range(200)] + ["TIMELINE//END"]
    _write_ontology(tmp_path, _codes(leaves))
    for fraction in (0.05, 0.15, 0.5, 0.9):
        universe = build_query_universe(leaves, ontology_dir=tmp_path, ancestor_fraction=fraction, seed=1)
        assert set(leaves) <= set(universe), f"leaves dropped at ancestor_fraction={fraction}"


def test_query_universe_rejects_an_all_ancestor_fraction(tmp_path):
    """Every leaf stays in, so ancestors cannot be 100% of the universe."""
    _write_ontology(tmp_path, _codes(["A//B//C"]))
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        build_query_universe(["A//B//C"], ontology_dir=tmp_path, ancestor_fraction=1.0, seed=1)


def test_query_universe_excludes_tautological_timeline_ancestors(tmp_path):
    """'did any TIMELINE event occur' is free positives and teaches nothing."""
    leaves = ["TIMELINE//END", "TIMELINE//DELTA//1d", "LAB//X//Y"]
    _write_ontology(tmp_path, _codes(leaves))
    universe = build_query_universe(leaves, ontology_dir=tmp_path, ancestor_fraction=0.5, seed=1)
    assert "TIMELINE" not in set(universe)


# ── 4. model integration ────────────────────────────────────────────────


def _tiny_model(vocab_size: int = 16, **kwargs) -> ConditionalQueryModel:
    model = ConditionalQueryModel(
        num_hidden_layers=2,
        config_overrides={
            "hidden_size": 32,
            "num_attention_heads": 2,
            "intermediate_size": 64,
            "vocab_size": vocab_size,
            "max_position_embeddings": 64,
            "pad_token_id": 0,
        },
        decoder_layers=1,
        decoder_heads=2,
        decoder_ffn_mult=2,
        max_queries=8,
        mlp_dropout=0.0,
        **kwargs,
    )
    model.eval()
    return model


def _batch(q_codes=(7, 8)):
    from every_query.data.seq_dataset import ConditionalQueryBatch

    return ConditionalQueryBatch(
        code=torch.tensor([[3, 4, 5, 6]]),
        numeric_value=torch.zeros(1, 4),
        numeric_value_mask=torch.zeros(1, 4, dtype=torch.bool),
        time_delta_days=torch.zeros(1, 4),
        q_codes=torch.tensor([list(q_codes)]),
        q_durations=torch.tensor([[30.0, 7.0]]),
        q_answers=torch.tensor([[ANSWER_YES, ANSWER_NO]]),
        q_mask=torch.tensor([[True, True]]),
    )


def test_ontology_dir_is_recorded_in_hparams(tmp_path):
    """Checkpoints must round-trip it: the wrapper changes state-dict keys."""
    _write_ontology(tmp_path, _codes([f"G//{i}" for i in range(1, 15)]))
    model = _tiny_model(vocab_size=extended_vocab_size(tmp_path), ontology_dir=str(tmp_path))
    assert model.hparams["ontology_dir"] == str(tmp_path)
    assert _tiny_model().hparams["ontology_dir"] is None


def test_wrapper_is_installed_on_the_shared_table(tmp_path):
    """It must be the encoder's input embedding, which is what the query slots also read."""
    _write_ontology(tmp_path, _codes([f"G//{i}" for i in range(1, 15)]))
    model = _tiny_model(vocab_size=extended_vocab_size(tmp_path), ontology_dir=str(tmp_path))
    assert isinstance(model.HF_model.get_input_embeddings(), OntologyEmbedding)


def test_ontology_model_runs_and_trains(tmp_path):
    _write_ontology(tmp_path, _codes([f"G//{i}" for i in range(1, 15)]))
    model = _tiny_model(vocab_size=extended_vocab_size(tmp_path), ontology_dir=str(tmp_path))
    model.train()
    loss, out = model(_batch())
    assert loss.isfinite() and out.answer_logits.shape == (1, 2)
    loss.backward()
    raw = model.HF_model.get_input_embeddings().tok
    assert raw.weight.grad is not None and torch.isfinite(raw.weight.grad).all()


def test_cache_is_cleared_between_forwards(tmp_path):
    """Without the pre-hook, a cached product would be reused across backward passes."""
    _write_ontology(tmp_path, _codes([f"G//{i}" for i in range(1, 15)]))
    model = _tiny_model(vocab_size=extended_vocab_size(tmp_path), ontology_dir=str(tmp_path))
    model.train()
    for _ in range(2):
        loss, _ = model(_batch())
        loss.backward()  # would raise "backward through the graph a second time" if stale
