"""Unit tests for the conditional query-sequence pipeline.

Covers, in order of pipeline position:

1. ``build_block_causal_mask`` — structural properties of the block-autoregressive mask.
2. ``ConditionalQueryModel`` — forward shape, loss masking, and *functional* information-flow
   tests: a query's prediction must be invariant to its own answer and to later queries, but
   sensitive to earlier answers and the patient context.
3. ``ConditionalQueryPytorchDataset`` — label parquet loading, censor-sentinel encoding, and
   collation (padding, answer classes, masks).
4. ``sample_query_sequences`` — sequence sampling structure and ground-truth labeling logic
   on a hand-built events frame.
"""

from datetime import datetime

import numpy as np
import polars as pl
import pytest
import torch

from every_query.data.seq_dataset import (
    ANSWERS_COL,
    CENSOR_QUERY_CODE,
    ConditionalQueryBatch,
    ConditionalQueryPytorchDataset,
)
from every_query.generate_tasks.sample_query_sequences import (
    build_sequence_index_df,
    label_sequence_index_df,
)
from every_query.generate_tasks.sample_tasks import compute_max_time_per_subject
from every_query.model.conditional_model import (
    ANSWER_CENSORED,
    ANSWER_NO,
    ANSWER_YES,
    TOKENS_PER_QUERY,
    ConditionalQueryModel,
    build_block_causal_mask,
)

# ── 1. Block-causal mask properties ─────────────────────────────────────


@pytest.mark.parametrize("n_queries", [1, 2, 3, 5, 8])
def test_mask_no_self_answer_leakage(n_queries):
    """No code/duration token may attend to its own block's answer token."""
    allowed = ~build_block_causal_mask(n_queries)
    for j in range(n_queries):
        code_pos, dur_pos, ans_pos = 3 * j, 3 * j + 1, 3 * j + 2
        assert not allowed[code_pos, ans_pos], f"code token of block {j} sees own answer"
        assert not allowed[dur_pos, ans_pos], f"duration token of block {j} sees own answer"


@pytest.mark.parametrize("n_queries", [2, 3, 5])
def test_mask_full_visibility_of_prior_blocks(n_queries):
    """Every token sees *all* tokens (incl. answers) of strictly earlier blocks."""
    allowed = ~build_block_causal_mask(n_queries)
    for j in range(1, n_queries):
        for tok in range(TOKENS_PER_QUERY):
            dst = 3 * j + tok
            assert allowed[dst, : 3 * j].all(), f"block {j} token {tok} misses an earlier token"


@pytest.mark.parametrize("n_queries", [1, 2, 5])
def test_mask_no_future_visibility(n_queries):
    """No token sees any token of a strictly later block."""
    allowed = ~build_block_causal_mask(n_queries)
    for j in range(n_queries - 1):
        for tok in range(TOKENS_PER_QUERY):
            dst = 3 * j + tok
            assert not allowed[dst, 3 * (j + 1) :].any(), f"block {j} token {tok} sees the future"


def test_mask_no_fully_masked_rows():
    for n in (1, 4, 8):
        assert (~build_block_causal_mask(n)).any(dim=1).all()


# ── 2. Model forward / information flow ─────────────────────────────────

VOCAB = 50
CENSOR_IDX = VOCAB - 1

MODEL_OVERRIDES = {
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


@pytest.fixture(scope="module")
def tiny_model() -> ConditionalQueryModel:
    torch.manual_seed(0)
    model = ConditionalQueryModel(
        config_overrides=MODEL_OVERRIDES,
        decoder_layers=2,
        decoder_heads=2,
        decoder_ffn_mult=2,
        max_queries=8,
        mlp_dropout=0.0,
    )
    model.eval()
    return model


def make_batch(
    q_answers: list[list[int]],
    patient_codes: list[list[int]] | None = None,
    q_codes: list[list[int]] | None = None,
    q_durations: list[list[float]] | None = None,
    q_mask: list[list[bool]] | None = None,
) -> ConditionalQueryBatch:
    """Build a small ConditionalQueryBatch with sensible defaults (2 samples × 3 queries)."""
    B, L = len(q_answers), len(q_answers[0])
    if patient_codes is None:
        patient_codes = [[3, 4, 5, 6]] * B
    if q_codes is None:
        q_codes = [[CENSOR_IDX] + list(range(7, 6 + L))] * B
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
    )


def test_forward_shapes_and_finite_loss(tiny_model):
    batch = make_batch([[ANSWER_YES, ANSWER_NO, ANSWER_YES], [ANSWER_NO, ANSWER_CENSORED, ANSWER_NO]])
    loss, out = tiny_model(batch)
    assert out.answer_logits.shape == (2, 3)
    assert loss.isfinite()
    assert out.censor_loss.isfinite() and out.occurs_loss.isfinite()


def test_loss_masks_censored_answers(tiny_model):
    """Flipping a censored answer's *target interpretation* must not change the loss, because
    censored positions are excluded — but flipping an observed one must."""
    base = make_batch([[ANSWER_YES, ANSWER_CENSORED, ANSWER_NO]])
    loss_base, out_base = tiny_model(base)
    assert not out_base.valid_mask[0, 1], "censored position should be invalid for loss"
    assert out_base.valid_mask[0, 0] and out_base.valid_mask[0, 2]


def test_own_answer_does_not_change_own_logit(tiny_model):
    """Teacher-forced A_j must be invisible to the prediction of A_j."""
    a = make_batch([[ANSWER_YES, ANSWER_NO, ANSWER_YES]])
    b = make_batch([[ANSWER_YES, ANSWER_NO, ANSWER_NO]])  # flip A_3 only
    _, out_a = tiny_model(a)
    _, out_b = tiny_model(b)
    torch.testing.assert_close(out_a.answer_logits[0, 2], out_b.answer_logits[0, 2])
    # And earlier positions are also unaffected (A_3 is in their future).
    torch.testing.assert_close(out_a.answer_logits[0, :2], out_b.answer_logits[0, :2])


def test_prior_answer_changes_later_logit(tiny_model):
    """Conditioning: flipping A_1 must change the prediction for A_2 (and typically A_3)."""
    a = make_batch([[ANSWER_YES, ANSWER_NO, ANSWER_YES]])
    b = make_batch([[ANSWER_NO, ANSWER_NO, ANSWER_YES]])  # flip A_1 only
    _, out_a = tiny_model(a)
    _, out_b = tiny_model(b)
    # A_1's own logit must be unchanged...
    torch.testing.assert_close(out_a.answer_logits[0, 0], out_b.answer_logits[0, 0])
    # ...but A_2's prediction must differ (it conditions on A_1).
    assert not torch.allclose(out_a.answer_logits[0, 1], out_b.answer_logits[0, 1])


def test_later_query_does_not_change_earlier_logit(tiny_model):
    """Changing Q_3's code must not affect predictions for A_1 / A_2."""
    a = make_batch([[ANSWER_YES, ANSWER_NO, ANSWER_YES]], q_codes=[[CENSOR_IDX, 7, 8]])
    b = make_batch([[ANSWER_YES, ANSWER_NO, ANSWER_YES]], q_codes=[[CENSOR_IDX, 7, 20]])
    _, out_a = tiny_model(a)
    _, out_b = tiny_model(b)
    torch.testing.assert_close(out_a.answer_logits[0, :2], out_b.answer_logits[0, :2])
    assert not torch.allclose(out_a.answer_logits[0, 2], out_b.answer_logits[0, 2])


def test_own_query_changes_own_logit(tiny_model):
    """The prediction for A_j must depend on Q_j's code and duration."""
    base = make_batch([[ANSWER_YES, ANSWER_NO, ANSWER_YES]])
    diff_code = make_batch([[ANSWER_YES, ANSWER_NO, ANSWER_YES]], q_codes=[[CENSOR_IDX, 30, 8]])
    diff_dur = make_batch(
        [[ANSWER_YES, ANSWER_NO, ANSWER_YES]], q_durations=[[30.0, 300.0, 30.0]]
    )
    _, out = tiny_model(base)
    _, out_code = tiny_model(diff_code)
    _, out_dur = tiny_model(diff_dur)
    assert not torch.allclose(out.answer_logits[0, 1], out_code.answer_logits[0, 1])
    assert not torch.allclose(out.answer_logits[0, 1], out_dur.answer_logits[0, 1])


def test_patient_context_changes_logits(tiny_model):
    a = make_batch([[ANSWER_YES, ANSWER_NO, ANSWER_YES]], patient_codes=[[3, 4, 5, 6]])
    b = make_batch([[ANSWER_YES, ANSWER_NO, ANSWER_YES]], patient_codes=[[10, 11, 12, 13]])
    _, out_a = tiny_model(a)
    _, out_b = tiny_model(b)
    assert not torch.allclose(out_a.answer_logits, out_b.answer_logits)


def test_padded_queries_do_not_change_real_logits(tiny_model):
    """Appending a padded (masked-out) query block must leave real predictions unchanged."""
    short = make_batch([[ANSWER_YES, ANSWER_NO]], q_codes=[[CENSOR_IDX, 7]], q_durations=[[30.0, 7.0]])
    padded = make_batch(
        [[ANSWER_YES, ANSWER_NO, ANSWER_NO]],
        q_codes=[[CENSOR_IDX, 7, 9]],
        q_durations=[[30.0, 7.0, 99.0]],
        q_mask=[[True, True, False]],
    )
    _, out_short = tiny_model(short)
    _, out_padded = tiny_model(padded)
    torch.testing.assert_close(out_short.answer_logits[0, :2], out_padded.answer_logits[0, :2])


def test_gradients_flow_and_are_finite(tiny_model):
    batch = make_batch([[ANSWER_YES, ANSWER_NO, ANSWER_CENSORED], [ANSWER_NO, ANSWER_YES, ANSWER_NO]])
    tiny_model.train()
    loss, _ = tiny_model(batch)
    loss.backward()
    grads = [p.grad for p in tiny_model.parameters() if p.grad is not None]
    assert grads, "no gradients populated"
    assert all(g.isfinite().all() for g in grads)
    tiny_model.zero_grad()
    tiny_model.eval()


def test_tiny_model_overfits_one_batch():
    """Sanity training signal: a tiny model should drive loss down on one fixed batch."""
    torch.manual_seed(1)
    model = ConditionalQueryModel(
        config_overrides=MODEL_OVERRIDES,
        decoder_layers=1,
        decoder_heads=2,
        decoder_ffn_mult=2,
        mlp_dropout=0.0,
    )
    batch = make_batch(
        [[ANSWER_YES, ANSWER_NO, ANSWER_YES], [ANSWER_NO, ANSWER_YES, ANSWER_NO]],
        patient_codes=[[3, 4, 5, 6], [10, 11, 12, 13]],
    )
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    first = None
    for _ in range(40):
        opt.zero_grad()
        loss, _ = model(batch)
        if first is None:
            first = loss.item()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.6, f"loss did not decrease enough: {first} -> {loss.item()}"


# ── 3. Dataset / collation ──────────────────────────────────────────────


def test_seq_dataset_loads_and_encodes(seq_dataset):
    assert len(seq_dataset) > 0
    assert seq_dataset.censor_query_index == max(seq_dataset.code_to_index.values()) + 1
    assert seq_dataset.encode_query(CENSOR_QUERY_CODE) == seq_dataset.censor_query_index
    with pytest.raises(KeyError):
        seq_dataset.encode_query("NOT_A_REAL_CODE")


def test_seq_dataset_getitem_carries_sequences(seq_dataset):
    item = seq_dataset[0]
    assert item["queries"][0] == CENSOR_QUERY_CODE
    assert len(item["queries"]) == len(item["durations"]) == len(item["answers"])
    assert "dynamic" in item


def test_seq_collate_shapes_and_padding(seq_dataset, seq_sample_batch):
    batch = seq_sample_batch
    assert isinstance(batch, ConditionalQueryBatch)
    B = batch.batch_size
    L = batch.n_queries
    assert batch.q_codes.shape == (B, L)
    assert batch.q_durations.shape == (B, L)
    assert batch.q_answers.shape == (B, L)
    assert batch.q_mask.shape == (B, L)

    # First position of every sample is the censor sentinel and real.
    assert (batch.q_codes[:, 0] == seq_dataset.censor_query_index).all()
    assert batch.q_mask[:, 0].all()

    # Padded positions carry zeros / mask False; the fixture mixes lengths 2 and 3.
    lengths = batch.q_mask.sum(dim=1)
    assert lengths.min() == 2 and lengths.max() == 3
    pad = ~batch.q_mask
    assert (batch.q_codes[pad] == 0).all()
    assert (batch.q_durations[pad] == 0).all()


def test_seq_collate_answer_classes(seq_dataset, seq_sample_batch):
    """Fixture rows alternate: even rows end censored (None); odd rows fully observed."""
    batch = seq_sample_batch
    raw = seq_dataset.schema_df[ANSWERS_COL].to_list()
    for i, answers in enumerate(raw):
        for j, ans in enumerate(answers):
            expected = ANSWER_CENSORED if ans is None else (ANSWER_YES if ans else ANSWER_NO)
            assert batch.q_answers[i, j].item() == expected, f"row {i} pos {j}"


def test_seq_dataset_end_to_end_forward(seq_dataset, seq_sample_batch):
    """The collated fixture batch runs through a tiny model sized to the fixture vocab."""
    torch.manual_seed(0)
    vocab = seq_dataset.censor_query_index + 1
    overrides = dict(MODEL_OVERRIDES, vocab_size=vocab)
    model = ConditionalQueryModel(
        config_overrides=overrides, decoder_layers=1, decoder_heads=2, decoder_ffn_mult=2
    )
    model.eval()
    loss, out = model(seq_sample_batch)
    assert loss.isfinite()
    assert out.answer_logits.shape == (seq_sample_batch.batch_size, seq_sample_batch.n_queries)


# ── 4. Sequence sampling + labeling ─────────────────────────────────────


def _events(rows: list[tuple[int, datetime, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "subject_id": [r[0] for r in rows],
            "time": [r[1] for r in rows],
            "code": [r[2] for r in rows],
        }
    ).with_columns(pl.col("time").cast(pl.Datetime("us")))


def test_build_sequence_index_structure():
    ctx = pl.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "prediction_time": [datetime(2024, 1, 1), datetime(2024, 2, 1), datetime(2024, 3, 1)],
        }
    ).with_columns(pl.col("prediction_time").cast(pl.Datetime("us")))
    idx = build_sequence_index_df(ctx, ["A", "B", "C"], 1, 4, 1, 365, seed=3)

    per_ctx = idx.group_by("_ctx_id").len()
    assert per_ctx["len"].min() >= 2 and per_ctx["len"].max() <= 5

    pos0 = idx.filter(pl.col("_position") == 0)
    assert pos0.height == 3
    assert pos0["query"].unique().to_list() == [CENSOR_QUERY_CODE]
    assert not idx.filter(pl.col("_position") > 0)["query"].is_in([CENSOR_QUERY_CODE]).any()


def test_label_sequence_known_answers():
    """Hand-checkable labeling: subject 1 has events at known times.

    Events for subject 1: code A at day 10, code B at day 100; last event day 100.
    Context at day 5:
      - censor query, 30d  → data after day 35?  yes (event at day 100)         → True
      - A within 10d       → A at day 10 < day 15 window end                     → True
      - B within 10d       → no B before day 15; window inside observation       → False
      - C within 200d      → no C ever; window end day 205 > max_time day 100    → null (censored)
      - A within 200d      → A observed in window; occurrence overrides the
        truncated window (terminal-event semantics)                              → True
    """
    t0 = datetime(2024, 1, 1)

    def day(n):
        return datetime(2024, 1, 1 + n) if n < 30 else datetime(2024, 1 + n // 30, 1 + n % 28)

    events = _events(
        [
            (1, day(0), "X"),
            (1, day(10), "A"),
            (1, datetime(2024, 4, 10), "B"),  # ~day 100
        ]
    )
    pred_time = datetime(2024, 1, 6)  # day 5

    index_df = pl.DataFrame(
        {
            "_ctx_id": [0, 0, 0, 0, 0],
            "_position": [0, 1, 2, 3, 4],
            "subject_id": [1, 1, 1, 1, 1],
            "prediction_time": [pred_time] * 5,
            "query": [CENSOR_QUERY_CODE, "A", "B", "C", "A"],
            "duration_days": [30.0, 10.0, 10.0, 200.0, 200.0],
        }
    ).with_columns(
        pl.col("prediction_time").cast(pl.Datetime("us")),
        pl.col("duration_days").cast(pl.Float32),
        pl.col("_ctx_id").cast(pl.UInt32),
    )

    labeled = label_sequence_index_df(index_df, events, compute_max_time_per_subject(events))
    assert labeled.height == 1
    row = labeled.row(0, named=True)
    assert row["queries"] == [CENSOR_QUERY_CODE, "A", "B", "C", "A"]
    assert row["answers"][0] is True, "data exists after day 35"
    assert row["answers"][1] is True, "A occurred at day 10 within (day5, day15)"
    assert row["answers"][2] is False, "no B in window, window fully observed"
    assert row["answers"][3] is None, "no C ever; window extends past last event -> censored"
    assert row["answers"][4] is True, "A observed in window; occurrence overrides truncation"


def test_label_sequence_censor_query_false_when_no_later_data():
    events = _events([(1, datetime(2024, 1, 1), "X"), (1, datetime(2024, 1, 10), "A")])
    index_df = pl.DataFrame(
        {
            "_ctx_id": [0],
            "_position": [0],
            "subject_id": [1],
            "prediction_time": [datetime(2024, 1, 5)],
            "query": [CENSOR_QUERY_CODE],
            "duration_days": [30.0],
        }
    ).with_columns(
        pl.col("prediction_time").cast(pl.Datetime("us")),
        pl.col("duration_days").cast(pl.Float32),
        pl.col("_ctx_id").cast(pl.UInt32),
    )
    labeled = label_sequence_index_df(index_df, events, compute_max_time_per_subject(events))
    assert labeled.row(0, named=True)["answers"] == [False], "no data after day 35 -> censor answer False"


def test_sampler_determinism():
    ctx = pl.DataFrame(
        {"subject_id": [1, 2], "prediction_time": [datetime(2024, 1, 1), datetime(2024, 2, 1)]}
    ).with_columns(pl.col("prediction_time").cast(pl.Datetime("us")))
    a = build_sequence_index_df(ctx, ["A", "B"], 1, 4, 1, 365, seed=11)
    b = build_sequence_index_df(ctx, ["A", "B"], 1, 4, 1, 365, seed=11)
    c = build_sequence_index_df(ctx, ["A", "B"], 1, 4, 1, 365, seed=12)
    assert a.equals(b)
    assert not a.equals(c)
