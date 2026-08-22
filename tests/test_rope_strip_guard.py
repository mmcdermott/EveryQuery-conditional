"""The other half of the RoPE/strip guard: ``use_rope_time=False`` on a *stripped* batch.

``use_rope_time`` (model) and ``strip_delta_tokens`` (dataset) are two halves of one setting.
The model has always refused ``use_rope_time=True`` with no ``time_pos_ids``.  The reverse
mismatch used to be accepted in silence, and it is the worse of the two:

- ``strip_delta_tokens=True`` deletes the quantized ``TIMELINE//DELTA*`` tokens from
  ``batch.code`` and moves the time they encoded into ``time_pos_ids``.
- The encoder reads *only* ``code``, the attention mask, and ``position_ids``.  ``time_delta_days``
  survives on the batch but is never passed to it.
- So ``use_rope_time=False`` on such a batch throws the positions away and leaves an encoder with
  **zero** elapsed-time information.  It trains, validates and checkpoints with normal-looking
  numbers; nothing in a loss curve distinguishes it from a healthy run.

The tests below lock six things:

1. the new guard fires, and its message names the config key to change;
2. the original opposite guard still fires (regression lock);
3. neither correctly-configured pair is disturbed — and for the ``use_rope_time=False`` +
   no-times pair, the forward output is *bit-identical* to the pre-guard implementation;
4. the semantics: the caught configuration is provably blind to time, measured at the encoder's
   ``last_hidden_state``;
5. each row of a batch is encoded against *its own* timeline — the whole suite used to hand
   both rows identical ``time_pos_ids``, under which broadcasting patient 0's timeline over the
   entire batch is indistinguishable from correct;
6. ``forward`` — the only non-test caller of ``_encoder_position_kwargs`` — actually passes the
   positions to the encoder.  Every other measurement here calls ``model.HF_model(...)`` with
   kwargs the test assembled itself, so a ``forward`` that computed the kwargs and discarded
   them would leave the seam provably correct and RoPE provably dead.

On (4)'s measurement level, following ``test_feature_liveness``: a randomly-initialised decoder
and answer head compress an ~8e-05 encoder difference down to ~1e-07 at the logits, i.e. into
float32 rounding noise.  Every assertion here therefore reads ``last_hidden_state`` directly and
uses a margin (``LIVE``), never a bare ``torch.equal`` inequality — except where *equality* is
the claim, which is exact and needs no margin.
"""

import pytest
import torch

from every_query.data.seq_dataset import ConditionalQueryBatch

# Reused deliberately rather than re-declared: the tiny-model config and the liveness margin
# must not drift between the file that proves RoPE is live and the file that proves the
# misconfiguration is dead.
from tests.test_feature_liveness import LIVE, N, _batch, _tiny
from tests.test_rope_time import _record_encoder_calls

# Two batches whose patient streams are token-for-token identical and whose *only* difference is
# how much time elapsed between those events.  This is what a stripped batch looks like: no
# TIMELINE//DELTA* tokens left in `code`, the gaps living in `time_pos_ids` (elapsed hours).
_HOURS_DENSE = [0, 1, 2, 3, 4, 5]  # six events within six hours
_HOURS_SPARSE = [0, 24, 300, 1200, 4000, 9000]  # the same six events spread over a year

# Two patients with identical code streams and no padding, so that when their timelines differ
# the *only* thing that can separate their encodings is the elapsed time.
_SAME_CODES = torch.tensor([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])


def _stripped_batch(hours: list[int] | list[list[int]], code=None) -> ConditionalQueryBatch:
    """A ``strip_delta_tokens=True``-shaped batch with the given per-token elapsed hours.

    ``hours`` is either one timeline shared by both rows, or one timeline per row.  Per-row
    timelines matter: with both rows identical, ``time_pos[:1].expand_as(batch.code)`` — a
    plausible broadcast slip applying patient 0's timeline to every patient in the batch — is
    numerically indistinguishable from the correct implementation.

    ``time_delta_days`` is filled in consistently with ``hours`` rather than left constant, so
    two batches built from different timelines differ in *every* field that carries time.  That
    makes the blindness result below stronger: not even the surviving ``time_delta_days``
    rescues the encoder, because the encoder never reads it.
    """
    rows = hours if isinstance(hours[0], list) else [hours] * 2
    per_row = torch.tensor(rows, dtype=torch.long)
    assert per_row.shape[1] == N
    deltas = torch.diff(per_row.float(), dim=1, prepend=torch.zeros(len(rows), 1)) / 24.0
    over = {"time_pos_ids": per_row, "time_delta_days": deltas}
    if code is not None:
        over["code"] = code
    return _batch(**over)


def _one_row(batch: ConditionalQueryBatch, i: int) -> ConditionalQueryBatch:
    """Row ``i`` of ``batch`` on its own — the same patient, encoded without the other one."""
    return ConditionalQueryBatch(
        code=batch.code[i : i + 1],
        numeric_value=batch.numeric_value[i : i + 1],
        numeric_value_mask=batch.numeric_value_mask[i : i + 1],
        time_delta_days=batch.time_delta_days[i : i + 1],
        q_codes=batch.q_codes[i : i + 1],
        q_durations=batch.q_durations[i : i + 1],
        q_answers=batch.q_answers[i : i + 1],
        q_mask=batch.q_mask[i : i + 1],
        time_pos_ids=batch.time_pos_ids[i : i + 1],
    )


def _encode(model, batch, position_kwargs) -> torch.Tensor:
    """The encoder memory the decoder cross-attends to — where RoPE actually acts."""
    with torch.no_grad():
        return model.HF_model(
            input_ids=batch.code,
            attention_mask=batch.code != ConditionalQueryBatch.PAD_INDEX,
            **position_kwargs,
        ).last_hidden_state


def _pre_guard_position_kwargs(model, batch) -> dict[str, torch.Tensor]:
    """``_encoder_position_kwargs`` exactly as it read before the symmetric guard was added.

    Kept verbatim so the "correctly configured users see no change" test compares against the
    real previous behaviour rather than against a re-derivation of it.
    """
    if not model.use_rope_time:
        return {}
    time_pos = getattr(batch, "time_pos_ids", None)
    if time_pos is None:
        raise ValueError("use_rope_time=True but the batch carries no time_pos_ids.")
    return {"position_ids": time_pos.to(batch.code.device)}


# ── 1. the new guard fires ─────────────────────────────────────────────────────────────


def test_non_rope_model_refuses_a_stripped_batch():
    """``use_rope_time=False`` + ``time_pos_ids`` is a misconfiguration, not a no-op."""
    model = _tiny(use_rope_time=False)
    with pytest.raises(ValueError) as excinfo:
        model(_stripped_batch(_HOURS_SPARSE))

    message = str(excinfo.value)
    # Both halves of the pair are named, because either one fixes it and only the user knows
    # which they meant.  A message that just said "mismatch" would leave them guessing.
    assert "datamodule.dataset_kwargs.strip_delta_tokens" in message, message
    assert "use_rope_time" in message, message
    assert "time_pos_ids" in message, message
    # It must say *why*, not merely that the two flags disagree.
    assert "elapsed" in message.lower(), message


def test_the_guard_fires_before_the_encoder_runs():
    """The refusal comes from the position-kwargs seam itself, not from a check inside ``forward``.

    ``forward`` is today the only non-test caller of ``_encoder_position_kwargs`` (the eval
    scripts go through ``forward``/``predict_step``, contrary to what this docstring used to
    claim).  The seam is still the right place: it is where the two halves of the setting are
    both in scope, it cannot be bypassed by a second encoder entry point being added later, and
    it fires before any tensor reaches the encoder — so a misconfigured run dies at the first
    batch instead of after an epoch of normal-looking loss.
    """
    model = _tiny(use_rope_time=False)
    with pytest.raises(ValueError, match="strip_delta_tokens"):
        model._encoder_position_kwargs(_stripped_batch(_HOURS_DENSE))


# ── 2. regression lock on the original guard ───────────────────────────────────────────


def test_rope_model_still_refuses_a_batch_without_times():
    """The pre-existing opposite guard must survive the symmetric one being added."""
    model = _tiny(use_rope_time=True)
    with pytest.raises(ValueError) as excinfo:
        model(_batch())

    message = str(excinfo.value)
    assert "time_pos_ids" in message, message
    assert "datamodule.dataset_kwargs.strip_delta_tokens" in message, message


# ── 3. neither correct configuration is disturbed ──────────────────────────────────────


def test_rope_model_with_times_still_runs_and_still_uses_them():
    """(a) ``use_rope_time=True`` + ``time_pos_ids``: runs, and the positions are handed over."""
    model = _tiny(use_rope_time=True)
    batch = _stripped_batch(_HOURS_SPARSE)

    kwargs = model._encoder_position_kwargs(batch)
    assert set(kwargs) == {"position_ids"}
    assert torch.equal(kwargs["position_ids"], batch.time_pos_ids)

    with torch.no_grad():
        loss, out = model(batch)
    assert loss.isfinite() and out.answer_logits.isfinite().all()


def test_non_rope_model_without_times_is_bit_identical_to_the_pre_guard_model():
    """(b) ``use_rope_time=False`` + no ``time_pos_ids``: not one bit moves for correct users.

    Compared against ``_pre_guard_position_kwargs`` — the literal previous implementation —
    rather than against a hardcoded number, so this stays a statement about the guard rather
    than about this machine's float arithmetic.
    """
    model = _tiny(use_rope_time=False)
    batch = _batch()
    assert batch.time_pos_ids is None

    assert model._encoder_position_kwargs(batch) == _pre_guard_position_kwargs(model, batch) == {}

    with torch.no_grad():
        guarded_loss, guarded_out = model(batch)
        reference = _encode(model, batch, _pre_guard_position_kwargs(model, batch))
        guarded = _encode(model, batch, model._encoder_position_kwargs(batch))

    assert torch.equal(reference, guarded), "the guard perturbed a correctly-configured encoder"
    assert guarded_loss.isfinite() and guarded_out.answer_logits.isfinite().all()


# ── 4. why the caught configuration is bad ─────────────────────────────────────────────


def test_the_caught_configuration_is_provably_blind_to_elapsed_time():
    """Two runs differing only in elapsed time encode *identically* without RoPE positions.

    This is the whole justification for making the mismatch an error rather than a warning.
    The two batches hold the same six codes in the same order; one spans six hours, the other a
    year.  Their delta tokens have been stripped, so the token stream cannot express that.

    Both measurements come from a single model instance, so the weights are literally the same
    tensors and the only independent variable is whether ``position_ids`` is passed.  Read at
    ``last_hidden_state``: an untrained decoder and answer head would squash this difference to
    ~1e-07, which is float32 noise, and that measurement error has already produced one false
    negative on this branch.
    """
    model = _tiny(use_rope_time=True)
    dense, sparse = _stripped_batch(_HOURS_DENSE), _stripped_batch(_HOURS_SPARSE)
    assert torch.equal(dense.code, sparse.code), "the codes must be the only thing held fixed"
    assert (dense.time_pos_ids != sparse.time_pos_ids).any()

    # What `use_rope_time=False` did with these batches: token-index positions, i.e. no extra
    # kwargs at all.  Bit-for-bit identical — six hours and a year are the same patient.
    blind_dense = _encode(model, dense, {})
    blind_sparse = _encode(model, sparse, {})
    assert torch.equal(blind_dense, blind_sparse), (
        "the point of the guard: without time positions the encoder cannot distinguish six "
        "hours from a year, and every checkpoint trained that way looks healthy"
    )

    # The correctly-configured pair does see it, at a margin far above rounding noise.
    seeing_dense = _encode(model, dense, model._encoder_position_kwargs(dense))
    seeing_sparse = _encode(model, sparse, model._encoder_position_kwargs(sparse))
    assert (seeing_dense - seeing_sparse).abs().max().item() > LIVE

    # And the blind path is genuinely the one a `use_rope_time=False` model takes: same weights
    # (same seed), same empty kwargs, same numbers — it is only reachable now via a batch that
    # carries no times, which is exactly what the guard forces.
    off = _tiny(use_rope_time=False)
    untimed = _batch()
    assert torch.equal(untimed.code, dense.code)
    assert off._encoder_position_kwargs(untimed) == {}
    assert torch.equal(_encode(off, untimed, {}), blind_dense)


# ── 5. every row is encoded against its own timeline ───────────────────────────────────


def test_each_row_is_encoded_against_its_own_timeline():
    """Patient 1's encoding must follow patient 1's clock, not patient 0's.

    Until this test, every RoPE fixture in the repo gave both rows of the batch *identical*
    ``time_pos_ids``, so ``{"position_ids": time_pos[:1].expand_as(batch.code)}`` — patient 0's
    timeline broadcast over the whole batch — passed the entire RoPE suite.  In training that is
    a silently wrong label generator: every patient but the first is scored against someone
    else's elapsed time, and nothing in a loss curve shows it.

    Two checks, because either alone is weak.  The rows must differ from *each other* (a
    broadcast collapses them to bit-identical), and each row must match what that patient's
    timeline produces when encoded alone (which pins *whose* clock each row got, not merely
    that the rows are not equal).
    """
    model = _tiny(use_rope_time=True)
    batch = _stripped_batch([_HOURS_DENSE, _HOURS_SPARSE], code=_SAME_CODES)
    assert torch.equal(batch.code[0], batch.code[1]), "codes held fixed; only the clocks differ"
    assert (batch.time_pos_ids[0] != batch.time_pos_ids[1]).any()

    both = _encode(model, batch, model._encoder_position_kwargs(batch))
    assert (both[0] - both[1]).abs().max().item() > LIVE, (
        "same codes, different timelines: the two rows must not encode alike — a broadcast of "
        "row 0's positions over the batch makes them identical"
    )

    for i in range(2):
        alone = _one_row(batch, i)
        solo = _encode(model, alone, model._encoder_position_kwargs(alone))
        assert (both[i] - solo[0]).abs().max().item() < LIVE, (
            f"row {i} must be encoded against its own hours, whoever it shares the batch with"
        )


# ── 6. the call site: forward really hands the positions over ──────────────────────────


def test_forward_passes_the_positions_to_the_encoder():
    """The guarded seam is useless if ``forward`` throws its result away.

    Everything above measures ``model.HF_model(...)`` called with kwargs the test assembled, so
    all of it stays green if ``forward`` computes ``_encoder_position_kwargs(batch)`` — guards
    firing and all — and then passes the encoder nothing.  That is the headline defect wearing a
    different hat: a RoPE run that trains and checkpoints on token-index positions.  Here the
    batches go through ``model(batch)``, and the encoder call ``forward`` actually made is
    recorded: the kwargs it received, and the memory it returned.
    """
    model = _tiny(use_rope_time=True)
    calls = _record_encoder_calls(model)
    dense, sparse = _stripped_batch(_HOURS_DENSE), _stripped_batch(_HOURS_SPARSE)

    with torch.no_grad():
        model(dense)
        model(sparse)

    assert len(calls) == 2, "forward must call the encoder exactly once per batch"
    for (kwargs, _), batch in zip(calls, [dense, sparse], strict=True):
        assert "position_ids" in kwargs, "forward dropped the rotary positions on the floor"
        assert torch.equal(kwargs["position_ids"], batch.time_pos_ids)

    (_, memory_dense), (_, memory_sparse) = calls
    assert (memory_dense - memory_sparse).abs().max().item() > LIVE, (
        "the memory the decoder cross-attends to must move when only the elapsed times move"
    )
