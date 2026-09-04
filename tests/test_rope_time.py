"""Tests for the RoPE time representation and the delta-token strip.

Covers, in pipeline order:

1. :mod:`every_query.data.rope_time` — strip semantics, elapsed-time preservation, row
   isolation, and agreement between the keep mask and the strip.
2. :class:`~every_query.data.seq_dataset.ConditionalQueryPytorchDataset` — that
   ``strip_delta_tokens=True`` removes the delta tokens from the collated encoder input,
   emits aligned ``time_pos_ids``, and leaves the query tensors untouched.
3. :class:`~every_query.model.conditional_model.ConditionalQueryModel` — that
   ``use_rope_time`` actually reaches ModernBERT's rotary machinery *through the real
   ``forward`` path* (the encoder output moves when only the times move), and that **both**
   half-configurations are refused rather than silently falling back to token-index
   positions: a RoPE model handed a batch with no times, and a non-RoPE model handed a
   batch that carries them.  The second direction was written down here as a *desirable*
   property once ("a non-RoPE model answers a timed batch identically") — that configuration
   leaves the encoder with zero elapsed-time information, so the correct behaviour is a
   refusal.  ``use_rope_time=False`` is inert only on a batch that carries no times, which is
   the half of that claim kept below.

Measurement level, throughout section 3: assertions about whether times *reached* the encoder
read ``last_hidden_state`` and use the ``LIVE`` margin.  A randomly-initialised decoder and
answer head squash the encoder difference to ~1e-07 at ``answer_logits``, i.e. into float32
rounding noise, so a bare ``torch.equal`` inequality there is satisfied by one ULP and passes
just as happily when RoPE is dead.
"""

import pytest
import torch
from meds import train_split
from meds_torchdata import MEDSTorchDataConfig

from every_query.data.rope_time import (
    DELTA_TOKEN_PREFIX,
    build_keep_mask,
    compact_by_keep,
    delta_vocab_ids,
    strip_delta_tokens,
)
from every_query.data.seq_dataset import ConditionalQueryBatch, ConditionalQueryPytorchDataset
from every_query.model.conditional_model import ANSWER_NO, ANSWER_YES, ConditionalQueryModel

# Imported rather than re-declared so the margin that separates "RoPE is live" from float32
# rounding cannot drift between the three files that measure it.
from tests.test_feature_liveness import LIVE

DELTA_ID = 90


def _rope_batch(time_pos_ids=None, patient_codes=None):
    """A 1x4-token, 1x2-query batch, optionally carrying rotary time positions."""
    if patient_codes is None:
        patient_codes = [[3, 4, 5, 6]]
    B, S = len(patient_codes), len(patient_codes[0])
    return ConditionalQueryBatch(
        code=torch.tensor(patient_codes),
        numeric_value=torch.zeros(B, S),
        numeric_value_mask=torch.zeros(B, S, dtype=torch.bool),
        time_delta_days=torch.zeros(B, S),
        q_codes=torch.tensor([[7, 8]] * B),
        q_durations=torch.tensor([[30.0, 7.0]] * B),
        q_answers=torch.tensor([[ANSWER_YES, ANSWER_NO]] * B),
        q_mask=torch.tensor([[True, True]] * B),
        time_pos_ids=None if time_pos_ids is None else torch.tensor(time_pos_ids),
    )


def _tiny_model(**kwargs) -> ConditionalQueryModel:
    model = ConditionalQueryModel(
        model_name="answerdotai/ModernBERT-base",
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
        **kwargs,
    )
    model.eval()
    return model


# ── 1. strip semantics ──────────────────────────────────────────────────


def test_delta_vocab_ids_selects_by_prefix():
    vocab = {"A": 1, f"{DELTA_TOKEN_PREFIX}//1h": 5, f"{DELTA_TOKEN_PREFIX}//1d": 2, "B": 3}
    assert delta_vocab_ids(vocab).tolist() == [2, 5]


def test_strip_removes_deltas_and_preserves_elapsed_time():
    """The delta tokens vanish, but the time they encoded survives in ``time_pos_ids``."""
    code = torch.tensor([[5, DELTA_ID, 6, DELTA_ID, 7, 0]])
    tdd = torch.tensor([[0.0, 1.0, 0.0, 0.5, 0.0, 0.0]])
    out_code, _, _, _, pos = strip_delta_tokens(
        code,
        torch.zeros(1, 6),
        torch.zeros(1, 6, dtype=torch.bool),
        tdd,
        torch.tensor([DELTA_ID]),
    )
    assert out_code.tolist() == [[5, 6, 7]], "delta tokens and padding must be gone"
    # 0d, then +1d = 24h, then +0.5d = 36h.
    assert pos.tolist() == [[0, 24, 36]]


def test_strip_is_row_isolated():
    """One row's delta pattern must not leak into another row's positions."""
    code = torch.tensor([[5, DELTA_ID, 6, 0], [5, 6, 0, 0]])
    tdd = torch.tensor([[0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    _, _, _, _, pos = strip_delta_tokens(
        code,
        torch.zeros(2, 4),
        torch.zeros(2, 4, dtype=torch.bool),
        tdd,
        torch.tensor([DELTA_ID]),
    )
    assert pos[0].tolist() == [0, 48]
    assert pos[1].tolist() == [0, 0], "row without deltas stays at time zero"


def test_strip_rebases_each_row_to_zero():
    """Absolute offset is arbitrary under RoPE; equal relative spacing must give equal ids."""
    code = torch.tensor([[DELTA_ID, 5, DELTA_ID, 6], [DELTA_ID, 5, DELTA_ID, 6]])
    tdd = torch.tensor([[1.0, 0.0, 1.0, 0.0], [40.0, 0.0, 1.0, 0.0]])
    _, _, _, _, pos = strip_delta_tokens(
        code,
        torch.zeros(2, 4),
        torch.zeros(2, 4, dtype=torch.bool),
        tdd,
        torch.tensor([DELTA_ID]),
    )
    assert pos[0].tolist() == pos[1].tolist() == [0, 24]


def test_strip_recomputes_time_delta_days_from_survivors():
    """After the strip, ``time_delta_days`` must still mean "days since the previous token".

    The gaps live on the delta tokens being dropped, so a naive compaction would leave this
    field all zeros — a silent trap sitting next to ``time_pos_ids``, which holds the truth.
    """
    code = torch.tensor([[5, DELTA_ID, 6, DELTA_ID, 7, 0]])
    tdd = torch.tensor([[0.0, 1.0, 0.0, 0.5, 0.0, 0.0]])
    _, _, _, out_tdd, pos = strip_delta_tokens(
        code,
        torch.zeros(1, 6),
        torch.zeros(1, 6, dtype=torch.bool),
        tdd,
        torch.tensor([DELTA_ID]),
    )
    assert out_tdd[0].tolist() == [0.0, 1.0, 0.5]
    assert out_tdd.abs().sum() > 0, "gaps must survive the strip somewhere other than time_pos_ids"
    # The two representations must agree: cumulative days == positions in hours.
    assert torch.allclose(out_tdd.cumsum(1) * 24.0, pos.float())


def test_strip_zeroes_time_delta_at_padding():
    """Padded tails carry no gap — in particular no negative gap at the pad boundary."""
    code = torch.tensor([[5, DELTA_ID, 6, 0], [5, 0, 0, 0]])
    tdd = torch.tensor([[0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    _, _, _, out_tdd, _ = strip_delta_tokens(
        code,
        torch.zeros(2, 4),
        torch.zeros(2, 4, dtype=torch.bool),
        tdd,
        torch.tensor([DELTA_ID]),
    )
    assert out_tdd[0].tolist() == [0.0, 3.0]
    assert out_tdd[1].tolist() == [0.0, 0.0], "shorter row's padded tail must stay zero"
    assert (out_tdd >= 0).all(), "no negative gap may appear at the padding boundary"


def test_keep_mask_agrees_with_strip():
    code = torch.tensor([[5, DELTA_ID, 6, 0]])
    keep = build_keep_mask(code, torch.tensor([DELTA_ID]))
    assert keep.tolist() == [[True, False, True, False]]
    out_code, _, _, _, _ = strip_delta_tokens(
        code,
        torch.zeros(1, 4),
        torch.zeros(1, 4, dtype=torch.bool),
        torch.zeros(1, 4),
        torch.tensor([DELTA_ID]),
    )
    assert compact_by_keep(code, keep, out_code.shape[1]).tolist() == out_code.tolist()


def test_strip_handles_nan_time_deltas():
    """meds_torchdata leaves NaN at unknown deltas; those must not poison the cumsum."""
    code = torch.tensor([[5, DELTA_ID, 6]])
    tdd = torch.tensor([[float("nan"), 1.0, 0.0]])
    _, _, _, _, pos = strip_delta_tokens(
        code,
        torch.zeros(1, 3),
        torch.zeros(1, 3, dtype=torch.bool),
        tdd,
        torch.tensor([DELTA_ID]),
    )
    assert pos.isfinite().all() and pos.tolist() == [[0, 24]]


# ── 2. dataset integration ──────────────────────────────────────────────


def _seq_cfg(tensorized_cohort_dir, seq_task_labels_dir) -> MEDSTorchDataConfig:
    return MEDSTorchDataConfig(
        tensorized_cohort_dir=str(tensorized_cohort_dir),
        task_labels_dir=str(seq_task_labels_dir),
        max_seq_len=64,
        seq_sampling_strategy="to_end",
        static_inclusion_mode="omit",
        batch_mode="SM",
    )


def test_dataset_without_strip_emits_no_time_pos_ids(seq_sample_batch):
    """Default behaviour is unchanged: no rotary times, delta tokens left in the stream."""
    assert seq_sample_batch.time_pos_ids is None


def test_dataset_strip_emits_aligned_time_pos_ids(tensorized_cohort_dir, seq_task_labels_dir):
    """With stripping on, ``time_pos_ids`` aligns to ``code`` and no delta token survives."""
    ds = ConditionalQueryPytorchDataset(
        _seq_cfg(tensorized_cohort_dir, seq_task_labels_dir),
        split=train_split,
        strip_delta_tokens=True,
    )
    batch = ds.collate([ds[i] for i in range(len(ds))])

    assert batch.time_pos_ids is not None
    assert batch.time_pos_ids.shape == batch.code.shape
    assert not torch.isin(batch.code, ds.delta_ids).any(), "no delta token may survive the strip"
    # Elapsed time never runs backwards within a row.
    real = batch.code != ConditionalQueryBatch.PAD_INDEX
    for i in range(batch.code.shape[0]):
        row = batch.time_pos_ids[i][real[i]]
        assert (row.diff() >= 0).all() if row.numel() > 1 else True


def test_dataset_strip_compacts_the_real_collated_stream(tensorized_cohort_dir, seq_task_labels_dir):
    """Exercise the collate-time strip on real batches.

    The fixture cohort carries no ``TIMELINE//DELTA*`` codes, which would make a "nothing
    survives the strip" assertion vacuous.  So we nominate a code that *is* present as the
    delta id and check the stream is genuinely compacted, every other per-token field stays
    aligned, and the surviving tokens keep their order.
    """
    cfg = _seq_cfg(tensorized_cohort_dir, seq_task_labels_dir)
    plain = ConditionalQueryPytorchDataset(cfg, split=train_split)
    before = plain.collate([plain[i] for i in range(len(plain))])

    real = before.code[before.code != ConditionalQueryBatch.PAD_INDEX]
    victim = int(real.mode().values.item())  # the most common real token

    ds = ConditionalQueryPytorchDataset(cfg, split=train_split, strip_delta_tokens=True)
    ds.delta_ids = torch.tensor([victim])
    after = ds.collate([ds[i] for i in range(len(ds))])

    assert (after.code == victim).sum() == 0, "the nominated delta token must be gone"
    assert after.code.shape[1] < before.code.shape[1], "the stream must actually get shorter"
    assert after.time_pos_ids.shape == after.code.shape
    assert after.numeric_value.shape == after.code.shape
    assert after.numeric_value_mask.shape == after.code.shape

    for i in range(before.code.shape[0]):
        kept = [int(c) for c in before.code[i] if int(c) not in (victim, ConditionalQueryBatch.PAD_INDEX)]
        got = [int(c) for c in after.code[i] if int(c) != ConditionalQueryBatch.PAD_INDEX]
        assert got == kept, "surviving tokens must keep their original order"


def test_strip_emits_times_even_when_the_cohort_has_no_delta_tokens(
    tensorized_cohort_dir, seq_task_labels_dir
):
    """``time_pos_ids`` means "the strip was requested", not "delta tokens were deleted".

    ``ConditionalQueryPytorchDataset`` handles the empty-``delta_ids`` cohort explicitly — it
    warns and carries on, emitting ``time_pos_ids`` while deleting nothing — so the presence of
    the field is not by itself proof that ``batch.code`` was rewritten.  Pinned here because
    ``_encoder_position_kwargs``'s docstring reasons about what that presence implies, and
    because the obvious "tidy-up" (suppressing ``time_pos_ids`` when nothing was stripped)
    would silently turn a caught misconfiguration back into a silent one: a user who asked for
    the strip against a cohort whose delta tokens are missing or differently named still needs
    the mismatch reported, not smoothed over.
    """
    cfg = _seq_cfg(tensorized_cohort_dir, seq_task_labels_dir)
    plain = ConditionalQueryPytorchDataset(cfg, split=train_split)
    ds = ConditionalQueryPytorchDataset(cfg, split=train_split, strip_delta_tokens=True)
    assert ds.delta_ids.numel() == 0, "this fixture cohort must have no TIMELINE//DELTA* codes"

    before = plain.collate([plain[i] for i in range(len(plain))])
    after = ds.collate([ds[i] for i in range(len(ds))])

    assert after.time_pos_ids is not None, "the positions are emitted even with nothing to strip"
    assert after.time_pos_ids.shape == after.code.shape
    pad = ConditionalQueryBatch.PAD_INDEX
    for i in range(before.code.shape[0]):
        kept = [int(c) for c in before.code[i] if int(c) != pad]
        got = [int(c) for c in after.code[i] if int(c) != pad]
        assert got == kept, "no delta ids means no token may be removed from the stream"

    # The guard is still right in this case — the user asked for the strip, and a
    # use_rope_time=False model would drop the hours it produced on the floor.
    with pytest.raises(ValueError, match="strip_delta_tokens"):
        _tiny_model(use_rope_time=False)._encoder_position_kwargs(after)


def test_dataset_strip_never_touches_the_static_table(tensorized_cohort_dir, seq_task_labels_dir):
    """``static_code`` & friends are a separate table, not per-token fields of the dynamic stream.

    The old width heuristic compacted *any* ``(B, n_old)`` tensor with the dynamic keep mask, so
    with ``static_inclusion_mode=include`` the static table was corrupted whenever it happened to
    be exactly as wide as the padded dynamic stream.  Pick ``max_seq_len`` so that it is.
    """

    def cfg(max_seq_len: int) -> MEDSTorchDataConfig:
        return MEDSTorchDataConfig(
            tensorized_cohort_dir=str(tensorized_cohort_dir),
            task_labels_dir=str(seq_task_labels_dir),
            max_seq_len=max_seq_len,
            seq_sampling_strategy="to_end",
            static_inclusion_mode="include",
            batch_mode="SM",
        )

    plain = before = None
    for max_seq_len in range(1, 9):
        plain = ConditionalQueryPytorchDataset(cfg(max_seq_len), split=train_split)
        before = plain.collate([plain[i] for i in range(len(plain))])
        if before.code.shape[1] == before.static_code.shape[1]:
            break
    assert before.code.shape[1] == before.static_code.shape[1], "fixture never lines the widths up"

    real = before.code[before.code != ConditionalQueryBatch.PAD_INDEX]
    victim = int(real.mode().values.item())
    ds = ConditionalQueryPytorchDataset(
        cfg(plain.config.max_seq_len), split=train_split, strip_delta_tokens=True
    )
    ds.delta_ids = torch.tensor([victim])
    after = ds.collate([ds[i] for i in range(len(ds))])

    assert after.code.shape[1] < before.code.shape[1], "the dynamic stream must actually get shorter"
    assert torch.equal(after.static_code, before.static_code)
    assert torch.equal(after.static_numeric_value, before.static_numeric_value)
    assert torch.equal(after.static_numeric_value_mask, before.static_numeric_value_mask)


def test_dataset_strip_leaves_query_tensors_untouched(tensorized_cohort_dir, seq_task_labels_dir):
    """Stripping touches the encoder stream only; the decoder's query blocks are unaffected."""
    cfg = _seq_cfg(tensorized_cohort_dir, seq_task_labels_dir)
    plain = ConditionalQueryPytorchDataset(cfg, split=train_split)
    stripped = ConditionalQueryPytorchDataset(cfg, split=train_split, strip_delta_tokens=True)

    a = plain.collate([plain[i] for i in range(len(plain))])
    b = stripped.collate([stripped[i] for i in range(len(stripped))])

    assert torch.equal(a.q_codes, b.q_codes)
    assert torch.equal(a.q_durations, b.q_durations)
    assert torch.equal(a.q_answers, b.q_answers)
    assert torch.equal(a.q_mask, b.q_mask)


# ── 3. model wiring ─────────────────────────────────────────────────────


def _encode(model, batch) -> torch.Tensor:
    """The encoder memory the decoder cross-attends to — where RoPE actually acts."""
    with torch.no_grad():
        return model.HF_model(
            input_ids=batch.code,
            attention_mask=batch.code != ConditionalQueryBatch.PAD_INDEX,
            **model._encoder_position_kwargs(batch),
        ).last_hidden_state


def _record_encoder_calls(model) -> list[tuple[dict, torch.Tensor]]:
    """Record ``(kwargs, last_hidden_state)`` for every encoder call ``forward`` itself makes.

    Every other measurement in this file calls ``model.HF_model(...)`` with position kwargs the
    test assembled, which proves the *seam* works and says nothing about whether ``forward``
    uses it.  A ``forward`` that computed the kwargs and then dropped them — the guards firing,
    the seam correct, RoPE stone dead in the only path training and evaluation take — would be
    invisible to all of them.  Hooking the encoder is what closes that.
    """
    calls: list[tuple[dict, torch.Tensor]] = []

    def hook(module, args, kwargs, output):
        calls.append((dict(kwargs), output.last_hidden_state.detach().clone()))

    model.HF_model.register_forward_hook(hook, with_kwargs=True)
    return calls


def test_rope_time_reaches_rotary():
    """Same tokens, different elapsed times must give a different encoder representation.

    Asserted on the encoder output rather than on ``answer_logits``: the randomly-initialised
    decoder and answer head attenuate the difference to ~1e-7, which would make a logit-level
    assertion a test of initialisation luck rather than of the wiring.  The margin is ``LIVE``
    rather than a bare inequality for the same reason — the real effect here is ~1e-4.
    """
    model = _tiny_model(use_rope_time=True)
    near = _encode(model, _rope_batch(time_pos_ids=[[0, 1, 2, 3]]))
    far = _encode(model, _rope_batch(time_pos_ids=[[0, 240, 1000, 5000]]))
    assert (near - far).abs().max().item() > LIVE, (
        "time_pos_ids must change the encoder geometry when use_rope_time=True"
    )


def test_rope_time_is_the_only_thing_that_moved():
    """Holding times fixed reproduces the encoder output exactly — the change is time, not noise."""
    model = _tiny_model(use_rope_time=True)
    once = _encode(model, _rope_batch(time_pos_ids=[[0, 24, 48, 72]]))
    twice = _encode(model, _rope_batch(time_pos_ids=[[0, 24, 48, 72]]))
    assert torch.equal(once, twice)


def test_forward_hands_the_times_to_the_encoder():
    """The real ``forward`` path — not a hand-assembled encoder call — must use the positions.

    Replaces an assertion that read ``assert not torch.equal(near.answer_logits,
    far.answer_logits)`` after two ``model(batch)`` calls.  That was the repo's only defence
    against ``forward`` ignoring the position kwargs, and it was a one-ULP defence: an untrained
    head compresses this difference to ~1e-07, so *any* two non-identical float paths satisfy
    it.  Here the batches go through ``model(batch)`` exactly as training does, and the claim is
    measured where the effect lives — at the tensor the decoder cross-attends to.
    """
    model = _tiny_model(use_rope_time=True)
    calls = _record_encoder_calls(model)
    near_batch = _rope_batch(time_pos_ids=[[0, 1, 2, 3]])
    far_batch = _rope_batch(time_pos_ids=[[0, 240, 1000, 5000]])

    with torch.no_grad():
        model(near_batch)
        model(far_batch)

    assert len(calls) == 2, "forward must call the encoder exactly once per batch"
    for (kwargs, _), batch in zip(calls, [near_batch, far_batch], strict=True):
        assert "position_ids" in kwargs, (
            "forward computed the rotary positions and did not pass them to the encoder"
        )
        assert torch.equal(kwargs["position_ids"], batch.time_pos_ids)

    (_, near), (_, far) = calls
    assert (near - far).abs().max().item() > LIVE, (
        "the encoder forward actually ran must move when only the elapsed times move"
    )


def test_non_rope_model_refuses_a_batch_carrying_times():
    """The mirror of ``test_rope_model_refuses_a_batch_without_times``, and just as necessary.

    This test used to assert the opposite — that such a batch is "answered identically" by a
    non-RoPE model — which wrote the defect down as intended behaviour.  ``time_pos_ids`` is
    emitted only by the strip path, so answering that batch normally means answering with an
    encoder that has *no* elapsed-time signal at all: the delta tokens gone from ``code`` and
    the hours that replaced them discarded, while training, validating and checkpointing with
    entirely normal-looking numbers.  See ``test_rope_strip_guard.py`` for the measurement
    proving that blindness.
    """
    model = _tiny_model(use_rope_time=False)
    with pytest.raises(ValueError, match="time_pos_ids"):
        model(_rope_batch(time_pos_ids=[[0, 240, 1000, 5000]]))


def test_non_rope_model_without_times_is_unperturbed():
    """The half of the old claim that survives: no times, no RoPE, no change and no refusal.

    The guard must be narrow.  A refusal that fired on every ``use_rope_time=False`` batch, or
    a fallback that started handing the encoder positions of its own, would both be caught
    here: the encoder output must be bit-identical to a plain call passing no position kwargs
    at all, and the model must still answer.
    """
    model = _tiny_model(use_rope_time=False)
    batch = _rope_batch()
    assert batch.time_pos_ids is None

    assert model._encoder_position_kwargs(batch) == {}
    with torch.no_grad():
        loss, out = model(batch)
        plain = model.HF_model(
            input_ids=batch.code, attention_mask=batch.code != ConditionalQueryBatch.PAD_INDEX
        ).last_hidden_state
    assert torch.equal(_encode(model, batch), plain), (
        "a non-RoPE model must reach the encoder exactly as it did before the feature existed"
    )
    assert loss.isfinite() and out.answer_logits.isfinite().all()


def test_rope_model_refuses_a_batch_without_times():
    """A RoPE model handed an ordinary batch must fail loudly, not silently use token indices.

    Falling back is indistinguishable from working: the upstream experiment scored an entire
    eval grid against a model that never received its time positions before noticing.
    """
    model = _tiny_model(use_rope_time=True)
    with pytest.raises(ValueError, match="time_pos_ids"):
        model(_rope_batch())


def test_rope_positions_beyond_max_position_embeddings_are_finite():
    """Hour-scale positions exceed max_position_embeddings; rotary computes them on the fly."""
    model = _tiny_model(use_rope_time=True)
    with torch.no_grad():
        loss, out = model(_rope_batch(time_pos_ids=[[0, 20_000, 60_000, 90_000]]))
    assert loss.isfinite() and out.answer_logits.isfinite().all()


def test_use_rope_time_is_recorded_in_hparams():
    """Checkpoints must round-trip the flag, or a reloaded model silently changes semantics."""
    assert _tiny_model(use_rope_time=True).hparams["use_rope_time"] is True
    assert _tiny_model().hparams["use_rope_time"] is False
