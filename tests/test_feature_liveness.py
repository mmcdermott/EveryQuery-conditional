"""Are the new features LIVE in the model, or merely wired to it?

A feature can be plumbed through collate, reach the forward pass, and then be multiplied by
zero -- passing every shape, dtype and "runs without error" assertion in the suite while
contributing nothing to the output.  Nothing in ``test_event_bounded.py`` or
``test_feature_composition.py`` can detect that, because they assert the model *runs* with the
new tensors, not that it *responds* to them.

Three probes here that a dead feature cannot pass:

1. **Gradient** -- each new parameter receives a non-zero gradient from a batch exercising it.
2. **Sensitivity** -- perturbing one new input field alone moves the output.
3. **Atom invariance** -- an unbounded batch is bit-identical with and without the new tensors
   attached.  The evaluation grid is entirely atomic single-code, time-bounded queries, so this
   is the property every reported AUROC number rests on: if attaching the machinery perturbed
   those queries, the scores would describe a different model than the one that was trained.

A note on measurement level, learned the hard way: a randomly-initialised decoder and output
head compress an 8e-05 encoder-output difference down to ~1e-07 at the logits, which is the
same magnitude as float32 rounding noise.  Assertions about the *encoder* therefore measure the
encoder's output directly, and no assertion anywhere here uses bare ``torch.equal`` inequality
to mean "responds to" -- a bitwise inequality is satisfied by one ULP of rounding, which is how
a test can be green from the moment it is written while the feature it names does nothing.
"""

import pytest
import torch

from every_query.data.seq_dataset import ConditionalQueryBatch
from every_query.model.conditional_model import ConditionalQueryModel

B, L, N = 2, 2, 6
# Comfortably above float32 rounding on these tensors (~1e-7) and below any real effect (~1e-5).
LIVE = 1e-6


def _tiny(**kw) -> ConditionalQueryModel:
    torch.manual_seed(0)
    model = ConditionalQueryModel(
        num_hidden_layers=2,
        config_overrides={
            "hidden_size": 32,
            "num_attention_heads": 2,
            "intermediate_size": 64,
            "vocab_size": 16,
            "max_position_embeddings": 64,
            "pad_token_id": 0,
        },
        decoder_layers=1,
        decoder_heads=2,
        decoder_ffn_mult=2,
        max_queries=8,
        mlp_dropout=0.0,
        **kw,
    )
    model.eval()
    return model


def _batch(**over) -> ConditionalQueryBatch:
    kw = {
        "code": torch.tensor([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 0]]),
        "numeric_value": torch.zeros(B, N),
        "numeric_value_mask": torch.zeros(B, N, dtype=torch.bool),
        "time_delta_days": torch.ones(B, N),
        "q_codes": torch.tensor([[7, 8], [9, 7]]),
        "q_durations": torch.tensor([[30.0, 7.0], [365.0, 14.0]]),
        "q_answers": torch.tensor([[1, 0], [0, 1]]),
        "q_mask": torch.ones(B, L, dtype=torch.bool),
    }
    kw.update(over)
    return ConditionalQueryBatch(**kw)


def _logits(model, **over):
    logits, _ = model(_batch(**over))
    return logits


# ── 1. gradient ────────────────────────────────────────────────────────────────────────


def test_bound_marker_receives_gradient():
    """The one parameter these three features add to the model must not be inert."""
    model = _tiny()
    model.train()
    logits, _ = model(_batch(q_bound_codes=torch.tensor([[3, 0], [0, 5]])))
    logits.sum().backward()

    grad = model.bound_marker.grad
    total = 0.0 if grad is None else grad.abs().sum().item()
    assert total > 0, "bound_marker received no gradient -- it is wired in but inert"


# ── 2. sensitivity ─────────────────────────────────────────────────────────────────────


def test_bound_code_identity_changes_output():
    model = _tiny()
    ref = _logits(model, q_bound_codes=torch.tensor([[3, 0], [0, 5]]))
    moved = _logits(model, q_bound_codes=torch.tensor([[4, 0], [0, 5]]))
    assert (moved - ref).abs().max().item() > LIVE


def test_bound_marker_separates_bounding_from_being_asked_about():
    """The same code id must not embed identically as a boundary and as a query subject."""
    model = _tiny()
    ids = torch.tensor([[3, 0], [0, 5]])
    with torch.no_grad():
        bounded = model._query_duration_embeds(_batch(q_bound_codes=ids))
        plain = model.HF_model.get_input_embeddings()(ids)
    assert (bounded[0, 0] - plain[0, 0]).abs().max().item() > LIVE


# ── 3. RoPE ────────────────────────────────────────────────────────────────────────────


def test_time_positions_reach_the_encoder():
    model = _tiny(use_rope_time=True)
    even = _batch(time_pos_ids=torch.tensor([[0, 1, 2, 3, 4, 5]] * B))
    uneven = _batch(time_pos_ids=torch.tensor([[0, 10, 40, 90, 160, 250], [0, 5, 9, 30, 44, 60]]))
    mask = torch.ones(B, N, dtype=torch.long)
    with torch.no_grad():
        # Directly at the encoder: the untrained decoder+head would compress this to ~1e-7.
        h_even = model.HF_model(
            input_ids=even.code, attention_mask=mask, **model._encoder_position_kwargs(even)
        ).last_hidden_state
        h_uneven = model.HF_model(
            input_ids=uneven.code, attention_mask=mask, **model._encoder_position_kwargs(uneven)
        ).last_hidden_state
    assert (h_even - h_uneven).abs().max().item() > LIVE


def test_rope_without_time_positions_refuses_rather_than_falling_back():
    model = _tiny(use_rope_time=True)
    with pytest.raises(ValueError, match="time_pos_ids"):
        model(_batch())


# ── 4. atom invariance ─────────────────────────────────────────────────────────────────


def test_unbounded_batch_is_bit_identical_with_and_without_feature_tensors():
    """The property every reported AUROC rests on -- the eval grid is entirely atomic queries."""
    model = _tiny()
    plain = _logits(model)
    with_machinery = _logits(model, q_bound_codes=torch.zeros(B, L, dtype=torch.long))
    assert torch.equal(plain, with_machinery), (
        "attaching the feature tensors perturbs a purely unbounded batch; every eval_full score "
        "would then describe a different model than the one trained"
    )
