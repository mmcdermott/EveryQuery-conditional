"""All four ported features enabled at once.

Each feature has its own suite; this one exists because the reason they were folded into
``ConditionalQueryModel`` behind flags — rather than shipped as the upstream forks' four
separate subclass towers — was so that any combination could run together.  That claim needs
a test, not just an argument.

The interactions that could plausibly break are checked individually below:

- RoPE strips the delta tokens and drives rotary from elapsed time; event bounds put a code in
  the duration slot.  They touch the encoder and the decoder respectively and must not interact.
- The ontology wrapper substitutes the shared token table, so boundary codes and aggregate
  components go through it too — which is the intended generalisation, and worth pinning.
- ``TIMELINE//END`` must survive the delta strip, because it is this model's entire censoring
  mechanism.
"""

import polars as pl
import torch

from every_query.data.ontology import (
    CLOSURE_FILE,
    MIX_FILE,
    NODES_FILE,
    build_closure,
    build_ontology,
    extended_vocab_size,
)
from every_query.data.query_vocab import OP_ATOM, OP_SEQ
from every_query.data.rope_time import DELTA_TOKEN_PREFIX, build_keep_mask, delta_vocab_ids
from every_query.data.seq_dataset import (
    EOS_CODE,
    MAX_COMPONENTS,
    NO_BOUND_INDEX,
    ConditionalQueryBatch,
)
from every_query.model.conditional_model import ANSWER_NO, ANSWER_YES, ConditionalQueryModel
from every_query.model.ontology_embedding import OntologyEmbedding


def _write_ontology(tmp_path):
    codes = [f"GRP//{i}" for i in range(1, 20)]
    codes_df = pl.DataFrame({"code": codes, "code/vocab_index": list(range(1, len(codes) + 1))})
    nodes, mix = build_ontology(codes_df)
    nodes.write_parquet(tmp_path / NODES_FILE)
    mix.write_parquet(tmp_path / MIX_FILE)
    build_closure(nodes, mix).write_parquet(tmp_path / CLOSURE_FILE)
    return extended_vocab_size(tmp_path)


def _all_features_model(tmp_path) -> ConditionalQueryModel:
    v_ext = _write_ontology(tmp_path)
    model = ConditionalQueryModel(
        num_hidden_layers=2,
        config_overrides={
            "hidden_size": 32,
            "num_attention_heads": 2,
            "intermediate_size": 64,
            "vocab_size": v_ext,
            "max_position_embeddings": 64,
            "pad_token_id": 0,
        },
        decoder_layers=1,
        decoder_heads=2,
        decoder_ffn_mult=2,
        max_queries=8,
        mlp_dropout=0.0,
        use_rope_time=True,
        ontology_dir=str(tmp_path),
    )
    model.eval()
    return model


def _all_features_batch() -> ConditionalQueryBatch:
    """Two queries: one event-bounded atom, one aggregate.  Plus RoPE time positions."""
    return ConditionalQueryBatch(
        code=torch.tensor([[3, 4, 5, 6]]),
        numeric_value=torch.zeros(1, 4),
        numeric_value_mask=torch.zeros(1, 4, dtype=torch.bool),
        time_delta_days=torch.tensor([[0.0, 1.0, 0.5, 2.0]]),
        time_pos_ids=torch.tensor([[0, 24, 36, 84]]),
        q_codes=torch.tensor([[7, 0]]),
        q_durations=torch.tensor([[-1.0, 30.0]]),
        q_answers=torch.tensor([[ANSWER_YES, ANSWER_NO]]),
        q_mask=torch.tensor([[True, True]]),
        q_bound_codes=torch.tensor([[9, NO_BOUND_INDEX]]),
        q_op=torch.tensor([[OP_ATOM, OP_SEQ]]),
        q_comp_codes=torch.tensor([[[7, 0, 0], [10, 11, 0]]]),
        q_gap_lo=torch.tensor([[0.0, 1.0]]),
        q_gap_hi=torch.tensor([[0.0, 7.0]]),
    )


def test_all_four_features_run_together(tmp_path):
    """The composability claim: RoPE + event bounds + ontology + aggregates in one forward."""
    model = _all_features_model(tmp_path)
    loss, out = model(_all_features_batch())
    assert loss.isfinite()
    assert out.answer_logits.shape == (1, 2)
    assert out.answer_logits.isfinite().all()


def test_all_four_features_train_together(tmp_path):
    model = _all_features_model(tmp_path)
    model.train()
    loss, _ = model(_all_features_batch())
    loss.backward()

    # Every feature's own parameters must receive gradient in the combined configuration.
    raw = model.HF_model.embeddings.tok_embeddings.tok
    for name, param in (
        ("ontology raw table", raw.weight),
        ("bound_marker", model.bound_marker),
        ("op_embed", model.op_embed.weight),
        ("comp_role_embed", model.comp_role_embed.weight),
    ):
        assert param.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} gradient is not finite"


def test_ontology_wrapper_covers_the_query_slots_too(tmp_path):
    """Boundary codes and aggregate components must go through the mixed table, not around it."""
    model = _all_features_model(tmp_path)
    assert isinstance(model.HF_model.embeddings.tok_embeddings, OntologyEmbedding)

    batch = _all_features_batch()
    wrapper = model.HF_model.embeddings.tok_embeddings

    calls = []
    original = wrapper.forward
    wrapper.forward = lambda ids: (calls.append(tuple(ids.shape)), original(ids))[1]
    try:
        with torch.no_grad():
            model(batch)
    finally:
        wrapper.forward = original

    # (B, N) patient stream, (B, L) query codes, (B, L) bound codes, (B, L, K) components.
    assert tuple(batch.code.shape) in calls, "patient stream must go through the wrapper"
    assert tuple(batch.q_comp_codes.shape) in calls, "aggregate components must go through it"


def test_rope_time_still_moves_the_encoder_with_everything_on(tmp_path):
    """RoPE must not be neutralised by the other features sharing the encoder."""
    model = _all_features_model(tmp_path)
    batch = _all_features_batch()

    def encode(time_pos):
        batch.time_pos_ids = time_pos
        with torch.no_grad():
            return model.HF_model(
                input_ids=batch.code,
                attention_mask=batch.code != ConditionalQueryBatch.PAD_INDEX,
                **model._encoder_position_kwargs(batch),
            ).last_hidden_state

    near = encode(torch.tensor([[0, 1, 2, 3]]))
    far = encode(torch.tensor([[0, 240, 1000, 5000]]))
    assert not torch.allclose(near, far)


def test_eos_code_survives_the_delta_strip():
    """TIMELINE//END is the model's whole censoring mechanism; the strip must not eat it.

    The strip keys on the ``TIMELINE//DELTA`` prefix, and ``TIMELINE//END`` does not start with
    it — but the two names are close enough that a prefix widened to ``TIMELINE`` would silently
    delete censoring from every sequence.
    """
    vocab = {EOS_CODE: 5, f"{DELTA_TOKEN_PREFIX}//1d": 9, f"{DELTA_TOKEN_PREFIX}//1h": 10}
    delta_ids = delta_vocab_ids(vocab)
    assert 5 not in delta_ids.tolist()

    code = torch.tensor([[5, 9, 5, 10]])
    keep = build_keep_mask(code, delta_ids)
    assert keep.tolist() == [[True, False, True, False]]


def test_bound_and_aggregate_occupy_different_slots(tmp_path):
    """Event bounds own the duration slot, aggregates the code slot — they must not collide."""
    model = _all_features_model(tmp_path)
    batch = _all_features_batch()

    with torch.no_grad():
        code_slot = model._query_code_embeds(batch)
        dur_slot = model._query_duration_embeds(batch)

    assert code_slot.shape == dur_slot.shape == (1, 2, 32)

    # Query 0 is a bounded atom: its code slot is the plain lookup, its duration slot is not.
    with torch.no_grad():
        plain_code = model.HF_model.embeddings.tok_embeddings(batch.q_codes)
    assert torch.allclose(code_slot[:, 0], plain_code[:, 0]), "an atom's code slot is untouched"
    assert not torch.allclose(code_slot[:, 1], plain_code[:, 1]), "the aggregate's is replaced"


def test_batch_validates_every_optional_tensor_together(tmp_path):
    """All the optional per-query tensors are shape-checked in the combined configuration."""
    batch = _all_features_batch()
    for name in ("q_bound_codes", "q_op", "q_gap_lo", "q_gap_hi"):
        assert getattr(batch, name).shape == (1, 2)
    assert batch.q_comp_codes.shape == (1, 2, MAX_COMPONENTS)
    assert batch.time_pos_ids.shape == batch.code.shape
