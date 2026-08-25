"""Unit tests for the conditional query-sequence pipeline.

Covers, in order of pipeline position:

1. ``build_block_causal_mask`` — structural properties of the block-autoregressive mask.
2. ``ConditionalQueryModel`` — forward shape, loss masking, and *functional* information-flow
   tests: a query's prediction must be invariant to its own answer and to later queries, but
   sensitive to earlier answers and the patient context.
3. ``ConditionalQueryPytorchDataset`` — label parquet loading, code encoding, and collation
   (padding, binary answer classes, masks).
4. ``sample_query_sequences`` — fully-random sequence sampling and binary observed-occurrence
   labeling on a hand-built events frame.
5. ``sample_query_sequences`` Stages 1'/3' — the 5-stage pipeline's sequence-specific stages:
   distribution parity with the training sampler, and per-shard index resolution.
"""

import inspect
from datetime import datetime
from functools import partial
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import polars as pl
import pytest
import torch
import yaml
from meds import held_out_split, tuning_split

from every_query.data.seq_dataset import (
    ANSWERS_COL,
    EOS_CODE,
    ConditionalQueryBatch,
)
from every_query.generate_tasks import sample_evaluation_query_sequences
from every_query.generate_tasks.sample_evaluation_query_sequences import (
    SequenceSpec,
    build_dense_sequence_index_df,
    read_sequence_specs,
    sample_sequence_specs,
    validate_spec_codes,
)
from every_query.generate_tasks.sample_query_sequences import (
    CTX_ID_COL,
    POSITION_COL,
    QuerySequenceDistribution,
    build_sequence_index,
    build_sequence_index_df,
    label_binary_occurrence,
    resolve_prediction_times,
)
from every_query.generate_tasks.sample_tasks import (
    QueryDistribution,
    QuerySpec,
    index_path,
    prediction_times_path,
)
from every_query.model.conditional_model import (
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
    """Every token sees *all* tokens (including answers) of strictly earlier blocks."""
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
EOS_IDX = VOCAB - 1

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
    """Build a small ConditionalQueryBatch with sensible defaults (2 samples x 3 queries)."""
    B, L = len(q_answers), len(q_answers[0])
    if patient_codes is None:
        patient_codes = [[3, 4, 5, 6]] * B
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
    )


def test_forward_shapes_and_finite_loss(tiny_model):
    batch = make_batch([[ANSWER_YES, ANSWER_NO, ANSWER_YES], [ANSWER_NO, ANSWER_NO, ANSWER_NO]])
    loss, out = tiny_model(batch)
    assert out.answer_logits.shape == (2, 3)
    assert loss.isfinite()


def test_loss_covers_all_real_positions_only(tiny_model):
    """Every real (non-padding) position is valid for loss; padding positions are excluded."""
    batch = make_batch([[ANSWER_YES, ANSWER_NO, ANSWER_YES]], q_mask=[[True, True, False]])
    _, out = tiny_model(batch)
    assert out.valid_mask[0, 0] and out.valid_mask[0, 1]
    assert not out.valid_mask[0, 2], "padding position must be excluded from loss"


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
    a = make_batch([[ANSWER_YES, ANSWER_NO, ANSWER_YES]], q_codes=[[EOS_IDX, 7, 8]])
    b = make_batch([[ANSWER_YES, ANSWER_NO, ANSWER_YES]], q_codes=[[EOS_IDX, 7, 20]])
    _, out_a = tiny_model(a)
    _, out_b = tiny_model(b)
    torch.testing.assert_close(out_a.answer_logits[0, :2], out_b.answer_logits[0, :2])
    assert not torch.allclose(out_a.answer_logits[0, 2], out_b.answer_logits[0, 2])


def test_own_query_changes_own_logit(tiny_model):
    """The prediction for A_j must depend on Q_j's code and duration."""
    base = make_batch([[ANSWER_YES, ANSWER_NO, ANSWER_YES]])
    diff_code = make_batch([[ANSWER_YES, ANSWER_NO, ANSWER_YES]], q_codes=[[EOS_IDX, 30, 8]])
    diff_dur = make_batch([[ANSWER_YES, ANSWER_NO, ANSWER_YES]], q_durations=[[30.0, 300.0, 30.0]])
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
    short = make_batch([[ANSWER_YES, ANSWER_NO]], q_codes=[[EOS_IDX, 7]], q_durations=[[30.0, 7.0]])
    padded = make_batch(
        [[ANSWER_YES, ANSWER_NO, ANSWER_NO]],
        q_codes=[[EOS_IDX, 7, 9]],
        q_durations=[[30.0, 7.0, 99.0]],
        q_mask=[[True, True, False]],
    )
    _, out_short = tiny_model(short)
    _, out_padded = tiny_model(padded)
    torch.testing.assert_close(out_short.answer_logits[0, :2], out_padded.answer_logits[0, :2])


def test_gradients_flow_and_are_finite(tiny_model):
    batch = make_batch([[ANSWER_YES, ANSWER_NO, ANSWER_NO], [ANSWER_NO, ANSWER_YES, ANSWER_NO]])
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
    for _ in range(80):
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
    # EOS may be absent in the tiny test cohort; encode_query is a plain vocab lookup.
    a_code = next(iter(seq_dataset.code_to_index))
    assert seq_dataset.encode_query(a_code) == seq_dataset.code_to_index[a_code]
    with pytest.raises(KeyError):
        seq_dataset.encode_query("NOT_A_REAL_CODE")


def test_seq_dataset_getitem_carries_sequences(seq_dataset):
    item = seq_dataset[0]
    assert len(item["queries"]) == len(item["durations"]) == len(item["answers"])
    assert all(isinstance(a, bool) for a in item["answers"]), "answers are binary, never None"
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

    # Padded positions carry zeros / mask False; the fixture mixes lengths 2 and 3.
    lengths = batch.q_mask.sum(dim=1)
    assert lengths.min() == 2 and lengths.max() == 3
    pad = ~batch.q_mask
    assert (batch.q_codes[pad] == 0).all()
    assert (batch.q_durations[pad] == 0).all()


def test_seq_collate_answer_classes(seq_dataset, seq_sample_batch):
    """Binary answers: True -> ANSWER_YES, False -> ANSWER_NO; padding holds ANSWER_NO."""
    batch = seq_sample_batch
    raw = seq_dataset.schema_df[ANSWERS_COL].to_list()
    for i, answers in enumerate(raw):
        for j, ans in enumerate(answers):
            expected = ANSWER_YES if ans else ANSWER_NO
            assert batch.q_answers[i, j].item() == expected, f"row {i} pos {j}"
        # padding beyond the real length is ANSWER_NO
        for j in range(len(answers), batch.n_queries):
            assert batch.q_answers[i, j].item() == ANSWER_NO


def test_seq_dataset_end_to_end_forward(seq_dataset, seq_sample_batch):
    """The collated fixture batch runs through a tiny model sized to the fixture vocab."""
    torch.manual_seed(0)
    vocab = max(seq_dataset.code_to_index.values()) + 1
    overrides = dict(MODEL_OVERRIDES, vocab_size=vocab)
    model = ConditionalQueryModel(
        config_overrides=overrides, decoder_layers=1, decoder_heads=2, decoder_ffn_mult=2
    )
    model.eval()
    loss, out = model(seq_sample_batch)
    assert loss.isfinite()
    assert out.answer_logits.shape == (seq_sample_batch.batch_size, seq_sample_batch.n_queries)


# ── 4. Sequence sampling + binary labeling ──────────────────────────────


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
    codes = ["A", "B", "C", EOS_CODE]
    idx = build_sequence_index_df(ctx, codes, 1, 5, 1, 365, seed=3)

    per_ctx = idx.group_by("_ctx_id").len()
    assert per_ctx["len"].min() >= 1 and per_ctx["len"].max() <= 5
    # All queries are ordinary codes (no privileged censor position / sentinel).
    assert bool(idx["query"].is_in(codes).all())
    # Positions within each context start at 0 and are contiguous.
    assert idx.filter(pl.col("_position") == 0)["_ctx_id"].n_unique() == 3


def test_eos_first_fraction_forces_eos_at_position_0():
    ctx = pl.DataFrame(
        {"subject_id": [1, 2], "prediction_time": [datetime(2024, 1, 1), datetime(2024, 2, 1)]}
    ).with_columns(pl.col("prediction_time").cast(pl.Datetime("us")))
    idx = build_sequence_index_df(ctx, ["A", "B", EOS_CODE], 2, 2, 1, 365, 0, eos_first_fraction=1.0)
    assert idx.filter(pl.col("_position") == 0)["query"].unique().to_list() == [EOS_CODE]


def test_label_binary_known_answers():
    """Hand-checkable binary observed-occurrence labeling.

    Subject 1: A at day 10, B at day 100 (last event = TIMELINE//END at day 100).
    Context at day 5:
      - A within 10d   → A at day 10 in (day5, day15)                         → True
      - B within 10d   → no B before day 15                                   → False
      - C within 200d  → no C ever                                            → False (not null!)
      - A within 200d  → A observed in window                                 → True
      - END within 10d → record ends day 100, not within (day5, day15)        → False
      - END within 200d→ record ends day 100, within (day5, day205)           → True
    """
    events = _events(
        [
            (1, datetime(2024, 1, 1), "X"),
            (1, datetime(2024, 1, 11), "A"),  # day 10
            (1, datetime(2024, 4, 10), "B"),  # ~day 100
            (1, datetime(2024, 4, 10), EOS_CODE),  # record ends with B's day
        ]
    )
    pred_time = datetime(2024, 1, 6)  # day 5
    index_df = pl.DataFrame(
        {
            "_ctx_id": [0] * 6,
            "_position": list(range(6)),
            "subject_id": [1] * 6,
            "prediction_time": [pred_time] * 6,
            "query": ["A", "B", "C", "A", EOS_CODE, EOS_CODE],
            "duration_days": [10.0, 10.0, 200.0, 200.0, 10.0, 200.0],
        }
    ).with_columns(
        pl.col("prediction_time").cast(pl.Datetime("us")),
        pl.col("duration_days").cast(pl.Float32),
        pl.col("_ctx_id").cast(pl.UInt32),
    )
    labeled = label_binary_occurrence(index_df, events)
    assert labeled.height == 1
    row = labeled.row(0, named=True)
    assert row["answers"] == [True, False, False, True, False, True]
    assert all(a is not None for a in row["answers"]), "binary labels are never null"


def test_sampler_determinism():
    ctx = pl.DataFrame(
        {"subject_id": [1, 2], "prediction_time": [datetime(2024, 1, 1), datetime(2024, 2, 1)]}
    ).with_columns(pl.col("prediction_time").cast(pl.Datetime("us")))
    a = build_sequence_index_df(ctx, ["A", "B"], 1, 4, 1, 365, seed=11)
    b = build_sequence_index_df(ctx, ["A", "B"], 1, 4, 1, 365, seed=11)
    c = build_sequence_index_df(ctx, ["A", "B"], 1, 4, 1, 365, seed=12)
    assert a.equals(b)
    assert not a.equals(c)


# ── 5. sample_evaluation_query_sequences (dense grid) ────────────────────


def test_read_sequence_specs_yaml_mapping_and_list(tmp_path):
    """Both YAML forms parse; the mapping form keeps its names, the list form generates them."""
    mapping_fp = tmp_path / "mapping.yaml"
    mapping_fp.write_text(yaml.safe_dump({"mortality_30d": [[EOS_CODE, 1], ["MEDS_DEATH", 30]]}))
    (spec,) = read_sequence_specs(mapping_fp)
    assert spec.name == "mortality_30d"
    assert spec.queries == (EOS_CODE, "MEDS_DEATH")
    assert spec.durations == (1, 30)

    list_fp = tmp_path / "list.yaml"
    list_fp.write_text(yaml.safe_dump([[["A", 7]], [["B", 14], ["C", 30]]]))
    specs = read_sequence_specs(list_fp)
    assert [s.name for s in specs] == ["seq_0000", "seq_0001"]
    assert [len(s) for s in specs] == [1, 2]


def test_read_sequence_specs_parquet_orders_by_position(tmp_path):
    """The long-format parquet form is ordered by ``position``, not by row order."""
    fp = tmp_path / "specs.parquet"
    pl.DataFrame(
        {
            "seq_id": ["s", "s", "s"],
            "position": [2, 0, 1],
            "query": ["third", "first", "second"],
            "duration_days": [3.0, 1.0, 2.0],
        }
    ).write_parquet(fp)
    (spec,) = read_sequence_specs(fp)
    assert spec.queries == ("first", "second", "third")
    assert spec.durations == (1.0, 2.0, 3.0)


def test_read_sequence_specs_rejects_malformed(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"s": [["A", 7, 9]]}))
    with pytest.raises(ValueError, match=r"\[code, duration\] pair"):
        read_sequence_specs(bad)

    empty = tmp_path / "empty.yaml"
    empty.write_text(yaml.safe_dump({}))
    with pytest.raises(ValueError, match="contains no sequences"):
        read_sequence_specs(empty)

    unsupported = tmp_path / "specs.txt"
    unsupported.write_text("A 7")
    with pytest.raises(ValueError, match=r"must be a \.yaml"):
        read_sequence_specs(unsupported)


def test_spec_names_sanitised_for_output_dirs(tmp_path):
    """MEDS codes contain ``/``, so names become path-safe; post-sanitisation clashes are errors."""
    fp = tmp_path / "s.yaml"
    fp.write_text(yaml.safe_dump({"MEDS_DEATH//30d": [["A", 30]]}))
    (spec,) = read_sequence_specs(fp)
    assert spec.name == "MEDS_DEATH_30d", "runs of unsafe chars collapse to a single underscore"

    clash = tmp_path / "clash.yaml"
    clash.write_text(yaml.safe_dump({"a/b": [["A", 1]], "a|b": [["B", 1]]}))
    with pytest.raises(ValueError, match="sanitise to"):
        read_sequence_specs(clash)


def test_dense_grid_shares_one_spec_set_across_contexts():
    """The defining property: every context is asked the identical N sequences, in the same order."""
    ctx = pl.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "prediction_time": [datetime(2024, 1, 1), datetime(2024, 2, 1), datetime(2024, 3, 1)],
        }
    ).with_columns(pl.col("prediction_time").cast(pl.Datetime("us")))
    specs = sample_sequence_specs(4, ["A", "B", EOS_CODE], 2, 3, 1, 365, seed=3)
    idx = build_dense_sequence_index_df(ctx, specs)

    assert idx.height == ctx.height * sum(len(s) for s in specs)
    # _ctx_id is unique per (context, spec) and context-major.
    assert idx[CTX_ID_COL].n_unique() == ctx.height * len(specs)
    per_ctx = idx.group_by("subject_id").agg(pl.col("query").alias("qs"), pl.col("duration_days").alias("ds"))
    assert per_ctx["qs"].n_unique() == 1 and per_ctx["ds"].n_unique() == 1


def test_validate_spec_codes_rejects_out_of_vocab():
    specs = [SequenceSpec("s", ("A", "NOPE"), (1.0, 2.0))]
    validate_spec_codes([SequenceSpec("ok", ("A",), (1.0,))], ["A", "B"])  # no error
    with pytest.raises(ValueError, match="absent from the query vocabulary"):
        validate_spec_codes(specs, ["A", "B"])


def test_sequence_spec_rejects_bad_durations():
    with pytest.raises(ValueError, match="finite number > 0"):
        SequenceSpec("s", ("A",), (-1.0,))
    with pytest.raises(ValueError, match="finite number > 0"):
        SequenceSpec("s", ("A",), (float("inf"),))
    # ``bool`` is an int subclass; True must not silently become 1.0 days.
    with pytest.raises(TypeError, match="must be a number"):
        SequenceSpec("s", ("A",), (True,))


def test_eval_sampling_defaults_stay_in_training_distribution():
    """A sampled evaluation grid must default to the distribution the model was pretrained on.

    ``sample_evaluation_query_sequences`` reuses ``build_sequence_index_df``, so these knobs mean
    exactly the same thing on both sides; a drifted default would silently produce
    out-of-distribution queries and read as an unexplained metric shift rather than an error.

    The code/duration knobs must match training *exactly* — a wider ``duration_max`` samples
    horizons the model never saw. Sequence length is different: the eval grid deliberately fixes
    one length (``min_queries == max_queries``) so every sequence has the same positions, which
    keeps per-position comparisons clean. That only needs to sit *inside* training's range.
    """
    configs = Path(sample_evaluation_query_sequences.CONFIGS)
    train_cfg = yaml.safe_load((configs / "sample_query_sequences_config.yaml").read_text())
    eval_cfg = yaml.safe_load((configs / "sample_evaluation_query_sequences_config.yaml").read_text())

    exact = [
        "duration_min",
        "duration_max",
        "duration_distribution",
        "eos_first_fraction",
        "duration_mode",
    ]
    assert {k: eval_cfg[k] for k in exact} == {k: train_cfg[k] for k in exact}

    assert eval_cfg["min_queries"] == eval_cfg["max_queries"], "eval grid uses one fixed length"
    assert train_cfg["min_queries"] <= eval_cfg["min_queries"] <= train_cfg["max_queries"], (
        f"eval length {eval_cfg['min_queries']} is outside training's "
        f"{train_cfg['min_queries']}..{train_cfg['max_queries']} range"
    )

    # The Python-level defaults on ``run_worker`` must agree with the YAML too — programmatic
    # callers bypass Hydra entirely.
    defaults = inspect.signature(sample_evaluation_query_sequences.run_worker).parameters
    for k in ("min_queries", "max_queries", "duration_min", "duration_max"):
        assert defaults[k].default == eval_cfg[k], f"run_worker default for {k} != config"


# ── 6. Stage 1'/3': the 5-stage sequence pipeline ───────────────────────


def _seq_dist(**overrides) -> QuerySequenceDistribution:
    kwargs = {
        "query_codes": ["A", "B", "C", EOS_CODE],
        "min_duration": 1.0,
        "max_duration": 365.0,
        "duration_distribution": "log-uniform",
        "min_queries": 1,
        "max_queries": 5,
    }
    kwargs.update(overrides)
    return QuerySequenceDistribution(**kwargs)


def test_sequence_draw_is_identical_to_the_training_query_distribution():
    """Stage 1' must draw from *exactly* the training sampler's distribution.

    This is the distribution-parity anchor the integration plan calls for.  Because sequence
    *structure* (lengths, the two sweep knobs) is drawn from a separate generator, the flattened
    ``(code, duration)`` stream is identical to ``QueryDistribution.sample`` for the same generator
    and total — so a future change to either sampler's draw breaks this test rather than silently
    retraining the conditional model on a shifted distribution.
    """
    dist = _seq_dist()
    seqs = dist.sample_sequences(50, np.random.default_rng(0), np.random.default_rng(1))
    flat = [q for s in seqs for q in s]

    base = QueryDistribution(dist.query_codes, 1.0, 365.0, "log-uniform")
    assert flat == base.sample(len(flat), np.random.default_rng(0))


def test_structure_rng_does_not_perturb_the_query_draw():
    """Changing only the structure seed must leave the code/duration stream untouched.

    The two axes are independent by construction; if they ever share a generator, a change to a sweep knob
    would silently move the query distribution too.
    """
    dist = _seq_dist(min_queries=3, max_queries=3)  # fixed length => same total either way
    a = dist.sample_sequences(20, np.random.default_rng(0), np.random.default_rng(1))
    b = dist.sample_sequences(20, np.random.default_rng(0), np.random.default_rng(99))
    assert [q for s in a for q in s] == [q for s in b for q in s]


def test_sequence_durations_are_floats_not_whole_days():
    """The fork rounded durations to integer days; ``QueryDistribution`` does not.

    Guards the documented distribution shift that makes the pre-port checkpoint unusable
    (docs/history/2026-08-18-conditional-v2-integration-plan.md §3) — a silent return to rounding would be a regression.
    """
    seqs = _seq_dist().sample_sequences(50, np.random.default_rng(0), np.random.default_rng(1))
    durations = [q.duration_days for s in seqs for q in s]
    assert any(d != round(d) for d in durations)


def test_sequence_lengths_span_the_configured_range():
    seqs = _seq_dist(min_queries=2, max_queries=4).sample_sequences(
        200, np.random.default_rng(0), np.random.default_rng(1)
    )
    assert {len(s) for s in seqs} == {2, 3, 4}


def test_duration_mode_nondecreasing_sorts_within_each_sequence():
    seqs = _seq_dist(min_queries=4, max_queries=4, duration_mode="nondecreasing").sample_sequences(
        20, np.random.default_rng(0), np.random.default_rng(1)
    )
    for s in seqs:
        durations = [q.duration_days for q in s]
        assert durations == sorted(durations)


def test_duration_mode_same_shares_one_horizon_per_sequence():
    seqs = _seq_dist(min_queries=4, max_queries=4, duration_mode="same").sample_sequences(
        20, np.random.default_rng(0), np.random.default_rng(1)
    )
    assert all(len({q.duration_days for q in s}) == 1 for s in seqs)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"min_queries": 0}, "min_queries must be >= 1"),
        ({"min_queries": 3, "max_queries": 2}, "must be >= min_queries"),
        ({"eos_first_fraction": 1.5}, r"must be in \[0, 1\]"),
        ({"duration_mode": "sorted"}, "duration_mode must be one of"),
        ({"query_codes": ["A"], "eos_first_fraction": 0.5}, "not in query_codes"),
    ],
)
def test_sequence_distribution_rejects_bad_config(kwargs, match):
    """Config errors must fail at construction, not three stages downstream."""
    with pytest.raises(ValueError, match=match):
        _seq_dist(**kwargs)


def _write_prediction_times_map(artifacts: Path, split: str, shard: str, rows: list[tuple]) -> None:
    """Write a Stage 0 ``_prediction_times/{shard}.parquet`` map with upstream's exact dtypes."""
    fp = prediction_times_path(artifacts, split, shard)
    fp.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "subject_id": [r[0] for r in rows],
            "prediction_time_index": [r[1] for r in rows],
            "time": [r[2] for r in rows],
        },
        schema={
            "subject_id": pl.Int64,
            "prediction_time_index": pl.Int64,
            "time": pl.Datetime("us"),
        },
    ).write_parquet(fp)


def _stage2_contexts(rows: list[tuple]) -> pl.DataFrame:
    """A Stage 2 output frame: ``(subject_id, shard, prediction_time_index)``."""
    return pl.DataFrame(
        {
            "subject_id": [r[0] for r in rows],
            "shard": [r[1] for r in rows],
            "prediction_time_index": [r[2] for r in rows],
        },
        schema={"subject_id": pl.Int64, "shard": pl.Utf8, "prediction_time_index": pl.Int64},
    )


def test_build_sequence_index_partitions_by_shard_and_resolves_times(tmp_path: Path):
    """Stage 3': one index partition per shard, ranks resolved, sequences kept intact."""
    artifacts, split = tmp_path / "tasks_artifacts", "train"
    _write_prediction_times_map(
        artifacts, split, "0", [(1, 0, datetime(2024, 1, 1)), (1, 1, datetime(2024, 1, 8))]
    )
    _write_prediction_times_map(artifacts, split, "1", [(2, 0, datetime(2024, 2, 1))])

    sequences = [
        [QuerySpec("A", 3.0), QuerySpec("B", -1.0, "DISCHARGE")],  # -> shard 0, subject 1, rank 1
        [QuerySpec("C", 5.0)],  # -> shard 1, subject 2, rank 0
        [QuerySpec("A", 1.0), QuerySpec(EOS_CODE, 9.0)],  # -> shard 0, subject 1, rank 0
    ]
    contexts = _stage2_contexts([(1, "0", 1), (2, "1", 0), (1, "0", 0)])

    assert build_sequence_index(sequences, contexts, artifacts, split) == 2

    shard0 = pl.read_parquet(index_path(artifacts, split, "0"))
    shard1 = pl.read_parquet(index_path(artifacts, split, "1"))
    assert shard0.height == 4 and shard1.height == 1

    # Ranks resolved to the map's timestamps, and each sequence's queries stayed together, in
    # order, under its own _ctx_id.
    by_ctx = {
        key[0]: (grp["query"].to_list(), grp["prediction_time"].to_list())
        for key, grp in shard0.sort(CTX_ID_COL, POSITION_COL).group_by(CTX_ID_COL, maintain_order=True)
    }
    assert by_ctx[0] == (["A", "B"], [datetime(2024, 1, 8)] * 2)
    assert by_ctx[2] == (["A", EOS_CODE], [datetime(2024, 1, 1)] * 2)
    assert shard1["prediction_time"].to_list() == [datetime(2024, 2, 1)]

    # _ctx_id is globally unique, so no two sequences collide when Stage 4' regroups.
    assert len(set(shard0[CTX_ID_COL].to_list() + shard1[CTX_ID_COL].to_list())) == 3

    # Stage 3' samples nothing: the bound Stage 1' put on a spec is carried into the index as-is,
    # and every shard of the run agrees on the column, bounded rows or not.
    ordered = shard0.sort(CTX_ID_COL, POSITION_COL)
    assert ordered["bound_event"].to_list() == [None, "DISCHARGE", None, None]
    assert ordered["duration_days"].to_list()[1] == -1.0
    assert shard1["bound_event"].to_list() == [None]


def test_build_sequence_index_rejects_length_mismatch(tmp_path: Path):
    with pytest.raises(ValueError, match=r"must equal contexts\.height"):
        build_sequence_index(
            [[QuerySpec("A", 1.0)]],
            _stage2_contexts([(1, "0", 0), (2, "0", 0)]),
            tmp_path,
            "train",
        )


def test_resolve_prediction_times_raises_on_unresolvable_rank(tmp_path: Path):
    """The join is total by design — a null timestamp is a hard error, never a silent drop."""
    artifacts, split = tmp_path / "a", "train"
    _write_prediction_times_map(artifacts, split, "0", [(1, 0, datetime(2024, 1, 1))])
    with pytest.raises(ValueError, match="null prediction_time"):
        resolve_prediction_times(_stage2_contexts([(1, "0", 99)]), artifacts, split, "0")


def test_resolve_prediction_times_raises_on_join_key_dtype_drift(tmp_path: Path):
    """A dtype mismatch silently produces an all-null join in polars; it must fail loudly."""
    artifacts, split = tmp_path / "a", "train"
    _write_prediction_times_map(artifacts, split, "0", [(1, 0, datetime(2024, 1, 1))])
    drifted = _stage2_contexts([(1, "0", 0)]).with_columns(pl.col("subject_id").cast(pl.UInt32))
    with pytest.raises(ValueError, match="dtype mismatch"):
        resolve_prediction_times(drifted, artifacts, split, "0")


# ── ConditionalQueryLightningModule: training-loop hooks ────────────────
#
# Every method below is either overridden by the conditional module or inherited from
# ``EveryQueryLightningModule``.  The inherited ones were written for the two-headed single-query
# model and reach for ``outputs.occurs_loss`` / ``outputs.censor_loss`` / ``batch.occurs`` /
# ``batch.censor``, none of which exist on ``ConditionalQueryOutput`` / ``ConditionalQueryBatch``.
# These tests run each hook against a real conditional batch so a re-introduced two-head assumption
# fails here instead of at step 0 of a real training run.


@pytest.fixture
def tiny_lightning_module(tiny_model):
    from every_query.model.conditional_lightning import ConditionalQueryLightningModule

    return ConditionalQueryLightningModule(
        model=tiny_model,
        optimizer=partial(torch.optim.AdamW, lr=1e-4, weight_decay=0.01),
    )


@pytest.mark.parametrize("hook", ["training_step", "validation_step", "test_step", "predict_step"])
def test_lightning_step_hooks_run_on_a_conditional_batch(tiny_lightning_module, hook):
    """No step hook may touch a two-head-only attribute."""
    batch = make_batch([[ANSWER_YES, ANSWER_NO, ANSWER_YES], [ANSWER_NO, ANSWER_NO, ANSWER_YES]])
    with patch.object(tiny_lightning_module, "log"):
        out = getattr(tiny_lightning_module, hook)(batch)
    if hook == "predict_step":
        assert set(out) == {"answer_probs", "q_mask", "q_answers", "q_codes", "q_durations"}
    else:
        assert out.isfinite()


def test_epoch_end_hooks_compute_only_conditional_metrics(tiny_lightning_module):
    """``_on_epoch_end`` walks ``self.metrics``; the conditional set has no occurs/censor entries."""
    for split in (tuning_split, held_out_split):
        names = set(tiny_lightning_module.metrics[split])
        assert "occurs_auc" not in names and "censor_auc" not in names
        assert names == {"answer_auc"} | {f"answer_auc_pos{j}" for j in range(8)}

    batch = make_batch([[ANSWER_YES, ANSWER_NO, ANSWER_YES], [ANSWER_NO, ANSWER_NO, ANSWER_YES]])
    logged = {}
    with patch.object(tiny_lightning_module, "log", side_effect=lambda k, v, **kw: logged.setdefault(k, v)):
        tiny_lightning_module.validation_step(batch)
        tiny_lightning_module.on_validation_epoch_end()
        tiny_lightning_module.test_step(batch)
        tiny_lightning_module.on_test_epoch_end()
        tiny_lightning_module.on_train_epoch_end()
    assert f"{tuning_split}/answer_auc" in logged
    assert f"{held_out_split}/answer_auc" in logged


def test_grad_norm_hook_is_throttled(tiny_model):
    """``grad_norm_log_every_n_steps`` gates the total-grad-norm log (0-indexed, so step 0 fires)."""
    from every_query.model.conditional_lightning import ConditionalQueryLightningModule

    module = ConditionalQueryLightningModule(model=tiny_model, grad_norm_log_every_n_steps=4)
    # ``grad_norm`` reads ``.grad``; without a backward there is nothing to norm.
    loss, _ = module.model(make_batch([[ANSWER_YES, ANSWER_NO, ANSWER_YES]]))
    loss.backward()

    fired = []
    for step in (0, 1, 4, 5, 8):
        with (
            patch.object(type(module), "global_step", property(lambda _s, v=step: v)),
            patch.object(module, "log", side_effect=lambda k, *a, **kw: fired.append(k)),
        ):
            module.on_before_optimizer_step(None)
    assert fired == ["train/grad_norm", "train/grad_norm", "train/grad_norm"]


def test_warmup_ratio_drives_the_lr_schedule(tiny_model):
    """``warmup_ratio`` x ``estimated_stepping_batches`` -> ``num_warmup_steps`` at fit time."""
    from transformers import get_cosine_schedule_with_warmup

    from every_query.model.conditional_lightning import ConditionalQueryLightningModule

    module = ConditionalQueryLightningModule(
        model=tiny_model,
        optimizer=partial(torch.optim.AdamW, lr=1e-4, weight_decay=0.01),
        LR_scheduler=partial(get_cosine_schedule_with_warmup),
        warmup_ratio=0.1,
    )
    module.trainer = Mock(estimated_stepping_batches=10_000)
    result = module.configure_optimizers()
    # 10% of 10,000 = 1,000 warmup steps, so the linear ramp is halfway done at step 500.
    assert result["lr_scheduler"]["scheduler"].lr_lambdas[0](500) == pytest.approx(0.5)
    assert result["lr_scheduler"]["interval"] == "step"


def test_warmup_ratio_is_validated(tiny_model):
    from every_query.model.conditional_lightning import ConditionalQueryLightningModule

    with pytest.raises(ValueError, match=r"warmup_ratio must be in \[0.0, 1.0\]"):
        ConditionalQueryLightningModule(model=tiny_model, warmup_ratio=1.5)


def test_warmup_ratio_survives_checkpoint_round_trip(tiny_model, tmp_path):
    import lightning as L

    from every_query.model.conditional_lightning import ConditionalQueryLightningModule

    module = ConditionalQueryLightningModule(
        model=tiny_model,
        optimizer=partial(torch.optim.AdamW, lr=1e-4, weight_decay=0.01),
        warmup_ratio=0.25,
    )
    ckpt = tmp_path / "cq.ckpt"
    torch.save(
        {
            "state_dict": module.state_dict(),
            "hyper_parameters": dict(module.hparams),
            "pytorch-lightning_version": L.__version__,
        },
        ckpt,
    )
    loaded = ConditionalQueryLightningModule.load_from_checkpoint(str(ckpt))
    assert loaded.warmup_ratio == 0.25
    assert loaded.model.hparams["max_queries"] == tiny_model.hparams["max_queries"]

    # Checkpoints written before warmup_ratio existed load with no warmup rather than KeyError.
    old_hparams = dict(module.hparams)
    old_hparams.pop("warmup_ratio")
    old_ckpt = tmp_path / "old.ckpt"
    torch.save(
        {
            "state_dict": module.state_dict(),
            "hyper_parameters": old_hparams,
            "pytorch-lightning_version": L.__version__,
        },
        old_ckpt,
    )
    assert ConditionalQueryLightningModule.load_from_checkpoint(str(old_ckpt)).warmup_ratio == 0.0


# ── conditional_config.yaml / _demo_train_conditional.yaml invariants ───


def _train_config(name: str) -> dict:
    from every_query.train.train import CONFIGS

    return yaml.safe_load((Path(CONFIGS) / name).read_text())


@pytest.mark.parametrize("name", ["conditional_config.yaml", "_demo_train_conditional.yaml"])
def test_conditional_configs_leave_encoder_sizing_to_the_data(name):
    """train.py overwrites both from the tensorized cohort; a literal here would be a lie."""
    overrides = _train_config(name)["lightning_module"]["model"]["config_overrides"]
    assert overrides["vocab_size"] == "???"
    assert overrides["max_position_embeddings"] == "???"


@pytest.mark.parametrize("name", ["conditional_config.yaml", "_demo_train_conditional.yaml"])
def test_conditional_configs_have_one_source_of_truth_for_precision_and_steps(name):
    cfg = _train_config(name)
    assert cfg["lightning_module"]["model"]["precision"] == "${trainer.precision}"
    assert cfg["trainer"]["precision"] == "bf16-mixed"
    # Removed upstream (8f83423): total steps come from the data via max_epochs.
    assert "max_steps" not in cfg["trainer"]
    assert cfg["trainer"]["max_epochs"] == 1


def test_conditional_config_writes_into_the_hydra_run_dir():
    """``${output_dir}`` is the shared base; a second run there would raise FileExistsError."""
    cfg = _train_config("conditional_config.yaml")
    assert cfg["trainer"]["default_root_dir"] == "${hydra:runtime.output_dir}"
    assert cfg["hydra"]["run"]["dir"].startswith("${output_dir}/")
    for key in ("logger", "callbacks"):
        assert "${trainer.default_root_dir}" in yaml.safe_dump(cfg["trainer"][key])


def test_conditional_config_declares_warmup_ratio_not_dead_step_counts():
    """``configure_optimizers`` overrides these as call-site kwargs, so declaring them is dead."""
    lm = _train_config("conditional_config.yaml")["lightning_module"]
    assert lm["warmup_ratio"] == 0.05
    assert "num_warmup_steps" not in lm["LR_scheduler"]
    assert "num_training_steps" not in lm["LR_scheduler"]
