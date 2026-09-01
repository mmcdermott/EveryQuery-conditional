"""Unit tests for the decoder-only (Llama) conditional query-sequence model (issue #14).

Covers, in order:

1. ``ConditionalQueryARModel`` — forward shape, loss masking, and the *functional*
   information-flow invariants the token-level causal mask must provide: a query's prediction
   is invariant to its own teacher-forced answer and to everything later, sensitive to its own
   query, earlier answers and the patient context, and identical whether computed from the full
   sequence or the prefix ending at its own ``d_i``.
2. Padding — neither padded query blocks nor padded patient positions may influence real
   predictions or the loss.
3. Feature parity — event-bounded duration slots, rope-time positions (and the paired-flag
   guard), and ontology-mixed embeddings shared by patient events, query codes and boundary
   codes.
4. Architecture selection — the renamed ``ConditionalQueryEncoderDecoderModel`` (with its
   backward-compatible ``ConditionalQueryModel`` alias), checkpoint round-trips through
   ``ConditionalQueryLightningModule`` for both architectures, and the training configs /
   ``train.py`` position-budget arithmetic that make ``max_position_embeddings`` cover patient
   *and* query tokens.
"""

from pathlib import Path

import polars as pl
import pytest
import torch
import yaml
from omegaconf import OmegaConf

from every_query.data.seq_dataset import ConditionalQueryBatch
from every_query.model.conditional_ar_model import (
    N_TOKEN_TYPES,
    TYPE_QUERY_ANSWER,
    ConditionalQueryARModel,
)
from every_query.model.conditional_model import (
    ANSWER_NO,
    ANSWER_YES,
    TOKENS_PER_QUERY,
    ConditionalQueryEncoderDecoderModel,
    ConditionalQueryModel,
)


@pytest.fixture(autouse=True)
def _setup_doctest_namespace():
    """Override the repo-root autouse fixture; nothing here downloads a HuggingFace model.

    ``ConditionalQueryARModel`` builds its backbone from a fresh ``LlamaConfig()`` — never
    ``from_pretrained`` — so this module stays offline-runnable.
    """
    yield


VOCAB = 50
EOS_IDX = VOCAB - 1

AR_OVERRIDES = {
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "intermediate_size": 64,
    "max_position_embeddings": 64,
    "vocab_size": VOCAB,
    "pad_token_id": 0,
    "attention_dropout": 0.0,
}

ENCDEC_OVERRIDES = {
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 64,
    "max_position_embeddings": 64,
    "vocab_size": VOCAB,
    "pad_token_id": 0,
    "cls_token_id": 1,
    "bos_token_id": 1,
    "sep_token_id": 2,
    "eos_token_id": 2,
}


def _tiny_ar_model(**kwargs) -> ConditionalQueryARModel:
    torch.manual_seed(0)
    overrides = dict(AR_OVERRIDES, **kwargs.pop("config_overrides", {}))
    model = ConditionalQueryARModel(config_overrides=overrides, mlp_dropout=0.0, **kwargs)
    model.eval()
    return model


@pytest.fixture(scope="module")
def tiny_model() -> ConditionalQueryARModel:
    return _tiny_ar_model()


def make_batch(
    q_answers: list[list[int]],
    patient_codes: list[list[int]] | None = None,
    q_codes: list[list[int]] | None = None,
    q_durations: list[list[float]] | None = None,
    q_mask: list[list[bool]] | None = None,
    **kwargs,
) -> ConditionalQueryBatch:
    """Build a small ConditionalQueryBatch with sensible defaults.

    The default two patient rows have *different real lengths* (4 and 2 tokens), so every test routinely
    exercises the re-packing that places each row's query stream immediately after its own last real event.
    """
    B, L = len(q_answers), len(q_answers[0])
    if patient_codes is None:
        patient_codes = [[3, 4, 5, 6], [10, 11, 0, 0]][:B] if B <= 2 else [[3, 4, 5, 6]] * B
    if q_codes is None:
        q_codes = [[EOS_IDX, *range(7, 6 + L)]] * B
    if q_durations is None:
        q_durations = [[30.0] * L] * B
    if q_mask is None:
        q_mask = [[True] * L] * B
    S = len(patient_codes[0])
    return ConditionalQueryBatch(
        code=torch.tensor(patient_codes),
        numeric_value=torch.zeros(B, S),
        numeric_value_mask=torch.zeros(B, S, dtype=torch.bool),
        time_delta_days=torch.zeros(B, S),
        q_codes=torch.tensor(q_codes),
        q_durations=torch.tensor(q_durations),
        q_answers=torch.tensor(q_answers),
        q_mask=torch.tensor(q_mask),
        **kwargs,
    )


# ── 1. Forward / information flow ───────────────────────────────────────


def test_forward_shapes_and_finite_loss(tiny_model):
    batch = make_batch([[ANSWER_YES, ANSWER_NO, ANSWER_YES], [ANSWER_NO, ANSWER_NO, ANSWER_NO]])
    loss, out = tiny_model(batch)
    assert out.answer_logits.shape == (2, 3)
    assert out.valid_mask.shape == (2, 3)
    assert loss.isfinite()


def test_own_answer_does_not_change_own_or_earlier_logits(tiny_model):
    """Flipping A_i must be invisible to d_1..d_i — only later blocks may move."""
    for flip_pos in range(3):
        answers_a = [ANSWER_YES, ANSWER_NO, ANSWER_YES]
        answers_b = list(answers_a)
        answers_b[flip_pos] = 1 - answers_b[flip_pos]
        _, out_a = tiny_model(make_batch([answers_a, answers_a]))
        _, out_b = tiny_model(make_batch([answers_b, answers_b]))
        torch.testing.assert_close(
            out_a.answer_logits[:, : flip_pos + 1],
            out_b.answer_logits[:, : flip_pos + 1],
            msg=f"flipping A_{flip_pos + 1} leaked into d_1..d_{flip_pos + 1}",
        )


def test_prior_answer_changes_later_logit(tiny_model):
    """Conditioning: flipping A_1 must change the prediction for A_2."""
    a = make_batch([[ANSWER_YES, ANSWER_NO, ANSWER_YES]] * 2)
    b = make_batch([[ANSWER_NO, ANSWER_NO, ANSWER_YES]] * 2)
    _, out_a = tiny_model(a)
    _, out_b = tiny_model(b)
    torch.testing.assert_close(out_a.answer_logits[:, 0], out_b.answer_logits[:, 0])
    assert not torch.allclose(out_a.answer_logits[:, 1], out_b.answer_logits[:, 1])


def test_later_query_does_not_change_earlier_logit(tiny_model):
    """Changing Q_3's code must not affect predictions for A_1 / A_2 (but must affect A_3)."""
    answers = [[ANSWER_YES, ANSWER_NO, ANSWER_YES]] * 2
    a = make_batch(answers, q_codes=[[EOS_IDX, 7, 8]] * 2)
    b = make_batch(answers, q_codes=[[EOS_IDX, 7, 20]] * 2)
    _, out_a = tiny_model(a)
    _, out_b = tiny_model(b)
    torch.testing.assert_close(out_a.answer_logits[:, :2], out_b.answer_logits[:, :2])
    assert not torch.allclose(out_a.answer_logits[:, 2], out_b.answer_logits[:, 2])


def test_later_duration_does_not_change_earlier_logit(tiny_model):
    answers = [[ANSWER_YES, ANSWER_NO, ANSWER_YES]] * 2
    a = make_batch(answers, q_durations=[[30.0, 7.0, 30.0]] * 2)
    b = make_batch(answers, q_durations=[[30.0, 7.0, 300.0]] * 2)
    _, out_a = tiny_model(a)
    _, out_b = tiny_model(b)
    torch.testing.assert_close(out_a.answer_logits[:, :2], out_b.answer_logits[:, :2])


def test_prefix_forward_matches_full_sequence(tiny_model):
    """The logit at d_i from the full sequence equals the one from the prefix ending at d_i."""
    full = make_batch([[ANSWER_YES, ANSWER_NO, ANSWER_YES]] * 2)
    for i in (1, 2):
        prefix = make_batch(
            [[ANSWER_YES, ANSWER_NO, ANSWER_YES][:i]] * 2,
            q_codes=[[EOS_IDX, 7, 8][:i]] * 2,
            q_durations=[[30.0] * i] * 2,
        )
        _, out_full = tiny_model(full)
        _, out_prefix = tiny_model(prefix)
        torch.testing.assert_close(out_full.answer_logits[:, :i], out_prefix.answer_logits)


def test_own_query_changes_own_logit(tiny_model):
    """The prediction for A_j must depend on Q_j's code and duration."""
    answers = [[ANSWER_YES, ANSWER_NO, ANSWER_YES]] * 2
    base = make_batch(answers)
    diff_code = make_batch(answers, q_codes=[[EOS_IDX, 30, 8]] * 2)
    diff_dur = make_batch(answers, q_durations=[[30.0, 300.0, 30.0]] * 2)
    _, out = tiny_model(base)
    _, out_code = tiny_model(diff_code)
    _, out_dur = tiny_model(diff_dur)
    assert not torch.allclose(out.answer_logits[:, 1], out_code.answer_logits[:, 1])
    assert not torch.allclose(out.answer_logits[:, 1], out_dur.answer_logits[:, 1])


def test_patient_context_changes_logits(tiny_model):
    a = make_batch([[ANSWER_YES, ANSWER_NO, ANSWER_YES]], patient_codes=[[3, 4, 5, 6]])
    b = make_batch([[ANSWER_YES, ANSWER_NO, ANSWER_YES]], patient_codes=[[10, 11, 12, 13]])
    _, out_a = tiny_model(a)
    _, out_b = tiny_model(b)
    assert not torch.allclose(out_a.answer_logits, out_b.answer_logits)


# ── 2. Padding ──────────────────────────────────────────────────────────


def test_padded_queries_do_not_change_real_logits(tiny_model):
    """Appending a padded (masked-out) query block must leave real predictions unchanged."""
    short = make_batch(
        [[ANSWER_YES, ANSWER_NO]] * 2, q_codes=[[EOS_IDX, 7]] * 2, q_durations=[[30.0, 7.0]] * 2
    )
    padded = make_batch(
        [[ANSWER_YES, ANSWER_NO, ANSWER_NO]] * 2,
        q_codes=[[EOS_IDX, 7, 9]] * 2,
        q_durations=[[30.0, 7.0, 99.0]] * 2,
        q_mask=[[True, True, False]] * 2,
    )
    _, out_short = tiny_model(short)
    _, out_padded = tiny_model(padded)
    torch.testing.assert_close(out_short.answer_logits, out_padded.answer_logits[:, :2])


def test_padded_queries_do_not_contribute_to_loss(tiny_model):
    """Only the ``q_mask``-selected positions carry loss; the padded block's target is inert."""
    kwargs = {"q_codes": [[EOS_IDX, 7, 9]] * 2, "q_mask": [[True, True, False]] * 2}
    loss_no, out = tiny_model(make_batch([[ANSWER_YES, ANSWER_NO, ANSWER_NO]] * 2, **kwargs))
    loss_yes, _ = tiny_model(make_batch([[ANSWER_YES, ANSWER_NO, ANSWER_YES]] * 2, **kwargs))
    torch.testing.assert_close(loss_no, loss_yes)
    assert out.valid_mask.tolist() == [[True, True, False]] * 2

    # And the loss equals the manual BCE over exactly the real positions.
    manual = torch.nn.functional.binary_cross_entropy_with_logits(
        out.answer_logits[:, :2], torch.tensor([[1.0, 0.0]] * 2)
    )
    torch.testing.assert_close(loss_no, manual)


def test_patient_padding_cannot_affect_predictions(tiny_model):
    """Widening the patient tensor with extra padding must leave every logit unchanged.

    This is the re-packing contract: each row's query stream starts at its own last real
    event, so batch-level padding width is invisible to the model.
    """
    narrow = make_batch(
        [[ANSWER_YES, ANSWER_NO, ANSWER_YES]] * 2, patient_codes=[[3, 4, 5, 6], [10, 11, 0, 0]]
    )
    wide = make_batch(
        [[ANSWER_YES, ANSWER_NO, ANSWER_YES]] * 2,
        patient_codes=[[3, 4, 5, 6, 0, 0, 0], [10, 11, 0, 0, 0, 0, 0]],
    )
    _, out_narrow = tiny_model(narrow)
    _, out_wide = tiny_model(wide)
    torch.testing.assert_close(out_narrow.answer_logits, out_wide.answer_logits)


def test_combined_length_must_fit_position_budget(tiny_model):
    """``max_position_embeddings`` covers patient AND query tokens; overflow is a hard error."""
    n_queries = 6
    patient = [list(range(3, 3 + 50))] * 1  # 50 patient + 18 query tokens > 64 positions
    batch = make_batch(
        [[ANSWER_NO] * n_queries],
        patient_codes=patient,
        q_codes=[list(range(7, 7 + n_queries))],
        q_durations=[[30.0] * n_queries],
    )
    with pytest.raises(ValueError, match="max_position_embeddings"):
        tiny_model(batch)


# ── 3. Feature parity: event bounds, rope time, ontology ────────────────


def test_event_bound_changes_duration_slot(tiny_model):
    """A bound code replaces the scalar-horizon content; NO_BOUND rows keep the scalar path."""
    answers = [[ANSWER_YES, ANSWER_NO]] * 2
    common = {"q_codes": [[EOS_IDX, 7]] * 2, "q_durations": [[30.0, 7.0]] * 2}
    no_bounds_col = make_batch(answers, **common)
    all_unbound = make_batch(answers, q_bound_codes=torch.zeros(2, 2, dtype=torch.long), **common)
    bound = make_batch(answers, q_bound_codes=torch.tensor([[0, 12]] * 2), **common)
    other_bound = make_batch(answers, q_bound_codes=torch.tensor([[0, 13]] * 2), **common)

    _, out_none = tiny_model(no_bounds_col)
    _, out_unbound = tiny_model(all_unbound)
    _, out_bound = tiny_model(bound)
    _, out_other = tiny_model(other_bound)

    # An all-NO_BOUND column is bit-identical to a dataset with no bounds column at all.
    torch.testing.assert_close(out_none.answer_logits, out_unbound.answer_logits)
    # A real bound changes that query's prediction, and *which* code bounds it matters.
    assert not torch.allclose(out_none.answer_logits[:, 1], out_bound.answer_logits[:, 1])
    assert not torch.allclose(out_bound.answer_logits[:, 1], out_other.answer_logits[:, 1])
    # The bound is invisible to the earlier query.
    torch.testing.assert_close(out_none.answer_logits[:, 0], out_bound.answer_logits[:, 0])


def test_rope_time_flag_and_positions_must_arrive_as_a_pair():
    model_off = _tiny_ar_model(use_rope_time=False)
    model_on = _tiny_ar_model(use_rope_time=True)
    answers = [[ANSWER_YES, ANSWER_NO, ANSWER_YES]] * 2
    time_pos = torch.tensor([[0, 24, 48, 100], [0, 5, 0, 0]])

    with pytest.raises(ValueError, match="time_pos_ids"):
        model_on(make_batch(answers))
    with pytest.raises(ValueError, match="strip_delta_tokens"):
        model_off(make_batch(answers, time_pos_ids=time_pos))


def test_rope_time_positions_are_consumed():
    """With ``use_rope_time`` the elapsed-hour spacing of events must change predictions."""
    model = _tiny_ar_model(use_rope_time=True)
    answers = [[ANSWER_YES, ANSWER_NO, ANSWER_YES]] * 2
    close = make_batch(answers, time_pos_ids=torch.tensor([[0, 1, 2, 3], [0, 1, 0, 0]]))
    spread = make_batch(answers, time_pos_ids=torch.tensor([[0, 240, 480, 720], [0, 240, 0, 0]]))
    _, out_close = model(close)
    _, out_spread = model(spread)
    assert not torch.allclose(out_close.answer_logits, out_spread.answer_logits)


def _combined_position_ids(model, batch) -> torch.Tensor:
    """Reproduce ``forward``'s bookkeeping and return the model's combined rotary positions."""
    n_patient = (batch.code != batch.PAD_INDEX).sum(dim=1)
    n_query_tokens = TOKENS_PER_QUERY * batch.q_codes.shape[1]
    query_positions = n_patient.unsqueeze(1) + torch.arange(n_query_tokens).unsqueeze(0)
    total_len = batch.code.shape[1] + n_query_tokens
    return model._position_ids(batch, n_patient, query_positions, total_len)


def test_query_tokens_share_the_prediction_time_rope_position():
    """Every query token sits at the final real patient event's hour — clinical time does not advance while
    the queries are asked, and rows with different lengths use their own hour."""
    model = _tiny_ar_model(use_rope_time=True)
    batch = make_batch(
        [[ANSWER_YES, ANSWER_NO]] * 2,
        patient_codes=[[3, 4, 5], [10, 11, 0]],
        time_pos_ids=torch.tensor([[0, 24, 72], [0, 10, 0]]),
    )
    pos = _combined_position_ids(model, batch)

    # Row 0: 3 real events (last hour 72) then 6 query tokens, all at 72.
    assert pos[0].tolist() == [0, 24, 72, 72, 72, 72, 72, 72, 72]
    # Row 1 is shorter and ends at hour 10; its query stream packs at index 2.
    assert pos[1, 2:8].tolist() == [10] * 6
    # No artificial +1, +2, ... progression anywhere in either query stream.
    n_patient = (batch.code != batch.PAD_INDEX).sum(dim=1)
    for row, n in enumerate(n_patient.tolist()):
        q = pos[row, n : n + 6]
        assert q.unique().numel() == 1


def test_query_blocks_get_shared_block_position_embeddings(tiny_model):
    """c_i, d_i and a_i share one block embedding; different blocks differ; patients get none."""
    B, L = 1, 3
    batch = make_batch([[ANSWER_YES, ANSWER_NO, ANSWER_YES]][:B])
    with torch.no_grad():
        tokens = tiny_model._query_tokens(batch).view(B, L, TOKENS_PER_QUERY, -1)
        # Same code/duration/answer content in every block would confound this, so instead
        # compare against the same model with the block table zeroed.
        saved = tiny_model.block_pos_embed.weight.clone()
        tiny_model.block_pos_embed.weight.zero_()
        base = tiny_model._query_tokens(batch).view(B, L, TOKENS_PER_QUERY, -1)
        tiny_model.block_pos_embed.weight.copy_(saved)

    delta = tokens - base  # (1, L, 3, H) — exactly the block-position contribution
    for block in range(L):
        for slot in range(1, TOKENS_PER_QUERY):
            assert torch.allclose(delta[0, block, 0], delta[0, block, slot], atol=1e-6)
    assert not torch.allclose(delta[0, 0, 0], delta[0, 1, 0])
    assert not torch.allclose(delta[0, 1, 0], delta[0, 2, 0])
    assert torch.allclose(delta[0, 0, 0], saved[0], atol=1e-6)


def test_patient_tokens_carry_no_block_position_embedding(tiny_model):
    """Patient positions of the packed stream are code + patient-type embedding, nothing more."""
    from every_query.model.conditional_ar_model import TYPE_PATIENT

    batch = make_batch([[ANSWER_YES, ANSWER_NO, ANSWER_YES]])
    captured = {}

    def _capture(_module, _args, kwargs):
        captured["inputs_embeds"] = kwargs["inputs_embeds"]

    handle = tiny_model.HF_model.register_forward_pre_hook(_capture, with_kwargs=True)
    try:
        with torch.no_grad():
            tiny_model(batch)
    finally:
        handle.remove()

    n_patient = (batch.code != batch.PAD_INDEX).sum(dim=1)
    with torch.no_grad():
        expected = tiny_model.HF_model.get_input_embeddings()(batch.code)
        expected = expected + tiny_model.token_type_embed.weight[TYPE_PATIENT]
    for row, n in enumerate(n_patient.tolist()):
        assert torch.allclose(captured["inputs_embeds"][row, :n], expected[row, :n], atol=1e-6)


def test_block_position_embeddings_receive_finite_nonzero_gradients():
    model = _tiny_ar_model()
    batch = make_batch([[ANSWER_YES, ANSWER_NO, ANSWER_NO], [ANSWER_NO, ANSWER_YES, ANSWER_NO]])
    model.train()
    loss, _ = model(batch)
    loss.backward()
    grad = model.block_pos_embed.weight.grad
    assert grad is not None and grad.isfinite().all()
    assert grad.abs().sum() > 0


def _uniform_mix(vocab: int) -> torch.Tensor:
    """Sparse mix matrix mapping *every* node's row to raw row 0 — all codes embed identically."""
    idx = torch.tensor([list(range(vocab)), [0] * vocab], dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.ones(vocab), size=(vocab, vocab)).coalesce()


def test_ontology_wrapper_feeds_patient_query_and_boundary_codes():
    """Under a mix that collapses every code to one row, code *identity* must become invisible on all three
    lookup paths — proof each path reads the wrapped table, not a private one."""
    from every_query.model.ontology_embedding import wrap_tok_embeddings

    model = _tiny_ar_model()
    wrap_tok_embeddings(model, _uniform_mix(VOCAB))
    answers = [[ANSWER_YES, ANSWER_NO]] * 2

    with torch.no_grad():
        # Patient path: swapping patient codes (same count) changes nothing.
        _, a = model(make_batch(answers, patient_codes=[[3, 4, 5, 6]] * 2))
        _, b = model(make_batch(answers, patient_codes=[[20, 21, 22, 23]] * 2))
        torch.testing.assert_close(a.answer_logits, b.answer_logits)

        # Query-code path.
        _, c = model(make_batch(answers, q_codes=[[EOS_IDX, 7]] * 2))
        _, d = model(make_batch(answers, q_codes=[[EOS_IDX, 8]] * 2))
        torch.testing.assert_close(c.answer_logits, d.answer_logits)

        # Boundary-code path.
        _, e = model(make_batch(answers, q_bound_codes=torch.tensor([[0, 12]] * 2)))
        _, f = model(make_batch(answers, q_bound_codes=torch.tensor([[0, 13]] * 2)))
        torch.testing.assert_close(e.answer_logits, f.answer_logits)


def test_ontology_dir_installs_wrapped_embeddings(tmp_path):
    """``ConditionalQueryARModel(ontology_dir=...)`` swaps in ``OntologyEmbedding``."""
    from every_query.data.ontology import (
        EMBEDDING_MIX_FILE,
        EVENT_TO_QUERY_NODES_FILE,
        ONTOLOGY_VOCAB_FILE,
        build_event_to_query_nodes,
        build_ontology,
        extended_vocab_size,
    )
    from every_query.model.ontology_embedding import OntologyEmbedding

    codes = ["LAB//A//x", "LAB//A//y", "LAB//B//z"]
    frame = pl.DataFrame({"code": codes, "code/vocab_index": [1, 2, 3]})
    nodes, mix = build_ontology(frame)
    nodes.write_parquet(tmp_path / ONTOLOGY_VOCAB_FILE)
    mix.write_parquet(tmp_path / EMBEDDING_MIX_FILE)
    build_event_to_query_nodes(nodes, mix).write_parquet(tmp_path / EVENT_TO_QUERY_NODES_FILE)

    v_ext = extended_vocab_size(tmp_path)
    model = _tiny_ar_model(config_overrides={"vocab_size": v_ext}, ontology_dir=str(tmp_path))
    assert isinstance(model.HF_model.get_input_embeddings(), OntologyEmbedding)


# ── 4. Training signal ──────────────────────────────────────────────────


def test_gradients_flow_and_are_finite(tiny_model):
    batch = make_batch([[ANSWER_YES, ANSWER_NO, ANSWER_NO], [ANSWER_NO, ANSWER_YES, ANSWER_NO]])
    tiny_model.train()
    loss, _ = tiny_model(batch)
    loss.backward()
    grads = [p.grad for p in tiny_model.parameters() if p.grad is not None]
    assert grads, "no gradients populated"
    assert all(g.isfinite().all() for g in grads)
    # Token-type embeddings are live: the answer-type row only sees gradient through a_i
    # tokens, which condition later blocks.
    assert tiny_model.token_type_embed.weight.grad.abs().sum() > 0
    tiny_model.zero_grad()
    tiny_model.eval()


def test_tiny_model_overfits_one_batch():
    """Sanity training signal: a tiny model should drive loss down on one fixed batch."""
    torch.manual_seed(1)
    model = ConditionalQueryARModel(config_overrides=dict(AR_OVERRIDES, num_hidden_layers=1), mlp_dropout=0.0)
    batch = make_batch(
        [[ANSWER_YES, ANSWER_NO, ANSWER_YES], [ANSWER_NO, ANSWER_YES, ANSWER_NO]],
        patient_codes=[[3, 4, 5, 6], [10, 11, 12, 13]],
    )
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    first = None
    for _ in range(80):
        opt.zero_grad()
        loss, _ = model(batch)
        if first is None:
            first = loss.item()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.6, f"loss did not decrease enough: {first} -> {loss.item()}"


# ── 5. Architecture selection / renaming / checkpoints ──────────────────


def test_backward_compatible_alias():
    """The old name must resolve to the renamed encoder-decoder class (same state-dict keys)."""
    assert ConditionalQueryModel is ConditionalQueryEncoderDecoderModel


def test_token_type_vocabulary_covers_all_roles():
    assert N_TOKEN_TYPES == 4
    assert TYPE_QUERY_ANSWER == 1 + 2  # patient + the three per-block slots


def _save_module_ckpt(module, path):
    import lightning as L

    torch.save(
        {
            "state_dict": module.state_dict(),
            "hyper_parameters": dict(module.hparams),
            "pytorch-lightning_version": L.__version__,
        },
        path,
    )


def test_ar_checkpoint_round_trip(tiny_model, tmp_path):
    """A checkpoint written by the Lightning module restores a ConditionalQueryARModel."""
    from functools import partial

    from every_query.model.conditional_lightning import ConditionalQueryLightningModule

    module = ConditionalQueryLightningModule(model=tiny_model, optimizer=partial(torch.optim.AdamW, lr=1e-4))
    assert module.hparams["model"]["architecture"] == "autoregressive"

    ckpt = tmp_path / "ar.ckpt"
    _save_module_ckpt(module, ckpt)
    loaded = ConditionalQueryLightningModule.load_from_checkpoint(str(ckpt))

    assert isinstance(loaded.model, ConditionalQueryARModel)
    assert loaded.model.hparams["max_queries"] == tiny_model.hparams["max_queries"]
    assert torch.equal(loaded.model.token_type_embed.weight, tiny_model.token_type_embed.weight)
    assert torch.equal(loaded.model.block_pos_embed.weight, tiny_model.block_pos_embed.weight)
    # The dispatch key must not leak into (or get popped from) the live module's hparams.
    assert module.hparams["model"]["architecture"] == "autoregressive"


def test_encoder_decoder_checkpoint_loads_under_renamed_class(tmp_path):
    """Pre-rename checkpoints carry no ``architecture`` key and restore the renamed class."""
    from functools import partial

    from transformers import ModernBertConfig

    from every_query.model.conditional_lightning import ConditionalQueryLightningModule

    # Seed the encoder config locally so this test (unlike the fixture-based enc-dec suite)
    # stays offline-runnable; ``model_name`` accepts any AutoConfig-loadable path.
    encoder_dir = tmp_path / "encoder"
    ModernBertConfig(**ENCDEC_OVERRIDES).save_pretrained(encoder_dir)

    torch.manual_seed(0)
    encdec = ConditionalQueryModel(
        model_name=str(encoder_dir),
        config_overrides=ENCDEC_OVERRIDES,
        decoder_layers=1,
        decoder_heads=2,
        decoder_ffn_mult=2,
        mlp_dropout=0.0,
    )
    module = ConditionalQueryLightningModule(model=encdec, optimizer=partial(torch.optim.AdamW, lr=1e-4))
    assert "architecture" not in module.hparams["model"]

    ckpt = tmp_path / "encdec.ckpt"
    _save_module_ckpt(module, ckpt)
    loaded = ConditionalQueryLightningModule.load_from_checkpoint(str(ckpt))
    assert isinstance(loaded.model, ConditionalQueryEncoderDecoderModel)
    assert torch.equal(next(loaded.model.parameters()), next(encdec.parameters()))


def test_unknown_architecture_is_rejected(tiny_model, tmp_path):
    from functools import partial

    from every_query.model.conditional_lightning import ConditionalQueryLightningModule

    module = ConditionalQueryLightningModule(model=tiny_model, optimizer=partial(torch.optim.AdamW, lr=1e-4))
    hparams = dict(module.hparams)
    hparams["model"] = dict(hparams["model"], architecture="hypercube")
    import lightning as L

    ckpt = tmp_path / "bad.ckpt"
    torch.save(
        {
            "state_dict": module.state_dict(),
            "hyper_parameters": hparams,
            "pytorch-lightning_version": L.__version__,
        },
        ckpt,
    )
    with pytest.raises(KeyError, match="hypercube"):
        ConditionalQueryLightningModule.load_from_checkpoint(str(ckpt))


# ── 6. Configs + train.py position budget ───────────────────────────────


def _train_config(name: str) -> dict:
    from every_query.train.train import CONFIGS

    return yaml.safe_load((Path(CONFIGS) / name).read_text())


def test_configs_select_architectures_explicitly():
    """Each config's ``_target_`` names its architecture; the retired name appears nowhere."""
    targets = {
        "conditional_config.yaml": "every_query.model.conditional_model.ConditionalQueryEncoderDecoderModel",
        "_demo_train_conditional.yaml": (
            "every_query.model.conditional_model.ConditionalQueryEncoderDecoderModel"
        ),
        "conditional_ar_config.yaml": "every_query.model.conditional_ar_model.ConditionalQueryARModel",
        "_demo_train_conditional_ar.yaml": "every_query.model.conditional_ar_model.ConditionalQueryARModel",
    }
    for name, target in targets.items():
        assert _train_config(name)["lightning_module"]["model"]["_target_"] == target, name


@pytest.mark.parametrize("name", ["conditional_ar_config.yaml", "_demo_train_conditional_ar.yaml"])
def test_ar_configs_leave_backbone_sizing_to_the_data(name):
    overrides = _train_config(name)["lightning_module"]["model"]["config_overrides"]
    assert overrides["vocab_size"] == "???"
    assert overrides["max_position_embeddings"] == "???"
    # The issue-#14 LlamaConfig surface is exposed explicitly.
    for key in (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "attention_dropout",
    ):
        assert key in overrides, f"{name} must expose {key}"


def test_position_budget_covers_patient_and_query_tokens():
    """train.py must size the AR backbone to max_seq_len + 3 * max_queries."""
    from every_query.train.train import required_position_embeddings

    ar_cfg = OmegaConf.create(
        {
            "_target_": "every_query.model.conditional_ar_model.ConditionalQueryARModel",
            "max_queries": 8,
        }
    )
    assert required_position_embeddings(ar_cfg, 256) == 256 + TOKENS_PER_QUERY * 8

    encdec_cfg = OmegaConf.create(
        {"_target_": "every_query.model.conditional_model.ConditionalQueryEncoderDecoderModel"}
    )
    assert required_position_embeddings(encdec_cfg, 256) == 258

    # The production AR config's knobs feed that arithmetic.
    cfg = _train_config("conditional_ar_config.yaml")
    assert cfg["lightning_module"]["model"]["max_queries"] == 8
    assert cfg["datamodule"]["config"]["max_seq_len"] == 256


def test_dataset_end_to_end_forward(seq_dataset, seq_sample_batch):
    """A real collated ``ConditionalQueryBatch`` runs through an AR model sized to the cohort."""
    torch.manual_seed(0)
    vocab = max(seq_dataset.code_to_index.values()) + 1
    model = ConditionalQueryARModel(
        config_overrides=dict(AR_OVERRIDES, vocab_size=vocab, max_position_embeddings=128),
        mlp_dropout=0.0,
    )
    model.eval()
    loss, out = model(seq_sample_batch)
    assert loss.isfinite()
    assert out.answer_logits.shape == (seq_sample_batch.batch_size, seq_sample_batch.n_queries)
