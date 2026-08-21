"""Tests for aggregate queries.

An aggregate query asks about a combination of codes: ``ANY``/``ALL``/``GE2``/``XOR`` over
component occurrence, and the three temporal operators ``CO`` (same timestamp), ``WITHIN``
(unordered, bounded gap) and ``SEQ`` (ordered, gap range).

Covered in pipeline order:

1. Labeling — a truth table per operator, plus the properties that distinguish them from each
   other (``CO`` is not ``WITHIN`` at zero width; ``SEQ`` is ordered and excludes co-timed
   pairs; the gap range is half-open).
2. Sampling — round-tripping through the grammar, reserved-character exclusion, seed-axis
   independence.
3. Batch/model — that an aggregate reaches the code slot, that ordering and gaps matter, and
   that an all-atomic batch is answered *identically* to a model without the feature.
"""

from datetime import datetime

import numpy as np
import polars as pl
import pytest
import torch

from every_query.data.query_vocab import (
    OP_ALL,
    OP_ANY,
    OP_ATOM,
    OP_CO,
    OP_GE2,
    OP_SEQ,
    OP_WITHIN,
    OP_XOR,
    is_aggregate,
    parse_query,
)
from every_query.data.seq_dataset import MAX_COMPONENTS, ConditionalQueryBatch
from every_query.generate_tasks.aggregate_labeling import (
    assign_aggregate_queries,
    draw_aggregate_query,
    eligible_components,
    format_query,
    has_aggregates,
    label_aggregates,
)
from every_query.generate_tasks.sample_query_sequences import build_sequence_index_df
from every_query.model.conditional_model import ANSWER_NO, ANSWER_YES, ConditionalQueryModel

PT = datetime(2024, 1, 2)


def _events(rows) -> pl.DataFrame:
    return pl.DataFrame(
        {"subject_id": [r[0] for r in rows], "time": [r[1] for r in rows], "code": [r[2] for r in rows]},
        schema={"subject_id": pl.Int64, "time": pl.Datetime("us"), "code": pl.Utf8},
    )


def _answer(query: str, events: pl.DataFrame, duration: float = 10.0) -> bool:
    idx = pl.DataFrame(
        {
            "_ctx_id": pl.Series([0], dtype=pl.UInt32),
            "_position": [0],
            "subject_id": [1],
            "prediction_time": [PT],
            "query": [query],
            "duration_days": pl.Series([duration], dtype=pl.Float32),
        }
    )
    return label_aggregates(idx, events).row(0, named=True)["answers"][0]


# ── 1. labeling ─────────────────────────────────────────────────────────

_BASE = _events(
    [
        (1, datetime(2024, 1, 3), "A"),
        (1, datetime(2024, 1, 3), "B"),  # co-timed with A
        (1, datetime(2024, 1, 6), "A"),
        (1, datetime(2024, 1, 9), "B"),
    ]
)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("ANY(A|Z)", True),
        ("ANY(Y|Z)", False),
        ("ALL(A|B)", True),
        ("ALL(A|Z)", False),
        ("GE2(A|B|Z)", True),
        ("GE2(A|Y|Z)", False),
        ("XOR(A|Z)", True),
        ("XOR(A|B)", False),
        ("CO(A&B)", True),
    ],
)
def test_set_operator_truth_table(query, expected):
    assert _answer(query, _BASE) is expected


def test_co_requires_the_same_timestamp_not_merely_a_small_gap():
    """CO is an event predicate, not a narrow WITHIN — this is what makes it its own operator."""
    events = _events(
        [(1, datetime(2024, 1, 3), "A"), (1, datetime(2024, 1, 3, 0, 0, 1), "B")]  # 1s apart
    )
    assert _answer("CO(A&B)", events) is False
    assert _answer("WITHIN(A&B|W=1.0)", events) is True


def test_within_is_unordered():
    """B-then-A must answer the same as A-then-B."""
    events = _events([(1, datetime(2024, 1, 5), "B"), (1, datetime(2024, 1, 6), "A")])
    assert _answer("WITHIN(A&B|W=2.0)", events) is True
    assert _answer("WITHIN(B&A|W=2.0)", events) is True


def test_within_respects_its_width():
    events = _events([(1, datetime(2024, 1, 3), "A"), (1, datetime(2024, 1, 9), "B")])
    assert _answer("WITHIN(A&B|W=7.0)", events) is True
    assert _answer("WITHIN(A&B|W=3.0)", events) is False


def test_seq_is_ordered():
    """A-then-B is true; B-then-A over the same events is not."""
    events = _events([(1, datetime(2024, 1, 3), "A"), (1, datetime(2024, 1, 6), "B")])
    assert _answer("SEQ(A>B|gap=7.0)", events) is True
    assert _answer("SEQ(B>A|gap=7.0)", events) is False


def test_seq_excludes_co_timed_pairs():
    """SEQ needs tau1 < tau2 strictly; a simultaneous pair is CO's business."""
    same = datetime(2024, 1, 3)
    assert _answer("SEQ(A>B|gap=7.0)", _events([(1, same, "A"), (1, same, "B")])) is False


def test_seq_gap_range_is_half_open():
    """[Gs, Ge): the lower bound is included, the upper excluded."""
    events = _events([(1, datetime(2024, 1, 3), "A"), (1, datetime(2024, 1, 6), "B")])  # 3-day gap
    assert _answer("SEQ(A>B|gap=3.0:4.0)", events) is True, "lower bound is inclusive"
    assert _answer("SEQ(A>B|gap=1.0:3.0)", events) is False, "upper bound is exclusive"


def test_seq_finds_a_later_valid_pair():
    """Any valid pair suffices — the first c1 need not be the one that works."""
    events = _events(
        [
            (1, datetime(2024, 1, 3), "A"),  # A -> B(Jan 9) is 6 days, outside [2, 4)
            (1, datetime(2024, 1, 6), "A"),  # A -> B(Jan 9) is 3 days, inside
            (1, datetime(2024, 1, 9), "B"),
        ]
    )
    assert _answer("SEQ(A>B|gap=2.0:4.0)", events) is True


def test_everything_is_bounded_by_the_window():
    """An otherwise-valid pair outside (t, t+d) does not count."""
    events = _events([(1, datetime(2024, 3, 1), "A"), (1, datetime(2024, 3, 2), "B")])
    assert _answer("SEQ(A>B|gap=7.0)", events, duration=10.0) is False
    assert _answer("SEQ(A>B|gap=7.0)", events, duration=120.0) is True


def test_atomic_rows_are_unaffected():
    """A frame with no aggregates must come back exactly as plain occurrence labeling."""
    from every_query.generate_tasks.sample_query_sequences import label_binary_occurrence

    idx = pl.DataFrame(
        {
            "_ctx_id": pl.Series([0, 0], dtype=pl.UInt32),
            "_position": [0, 1],
            "subject_id": [1, 1],
            "prediction_time": [PT, PT],
            "query": ["A", "Z"],
            "duration_days": pl.Series([10.0, 10.0], dtype=pl.Float32),
        }
    )
    assert (
        label_aggregates(idx, _BASE).row(0, named=True)["answers"]
        == label_binary_occurrence(idx, _BASE).row(0, named=True)["answers"]
    )


def test_has_aggregates_detects_the_frame():
    def idx(q):
        return pl.DataFrame({"query": [q]})

    assert has_aggregates(idx("ANY(A|B)")) is True
    assert has_aggregates(idx("LAB//X")) is False


# ── 2. sampling ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("op", [OP_ANY, OP_ALL, OP_GE2, OP_XOR, OP_CO, OP_WITHIN, OP_SEQ])
def test_every_operator_round_trips_through_the_grammar(op):
    comps = ["A", "B", "C"] if op == OP_GE2 else ["A", "B"]
    q = format_query(op, comps, gap_lo=1.0, gap_hi=7.0)
    parsed = parse_query(q)
    assert parsed.op == op
    assert parsed.components == comps


def test_reserved_character_codes_are_excluded_from_the_pool():
    """Upstream shipped this as a live bugfix after & collided with the CO separator."""
    assert eligible_components(["OK", "BAD&CODE", "BAD|CODE", "A>B"]) == ["OK"]


def test_drawn_queries_always_parse_back():
    rng = np.random.default_rng(0)
    pool = ["A", "B", "C", "D"]
    for _ in range(200):
        q = draw_aggregate_query(pool, rng)
        parsed = parse_query(q)
        assert parsed.op != OP_ATOM
        assert all(c in pool for c in parsed.components), f"{q} -> {parsed.components}"


def test_drawn_components_are_distinct():
    """ALL(A|A) or SEQ(A>A) would be degenerate questions."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        comps = parse_query(draw_aggregate_query(["A", "B", "C"], rng)).components
        assert len(set(comps)) == len(comps)


def test_assign_is_deterministic_and_respects_the_fraction():
    idx = pl.DataFrame({"query": list("ABCDEFGH") * 5, "duration_days": [30.0] * 40}).with_columns(
        pl.col("duration_days").cast(pl.Float32)
    )
    a = assign_aggregate_queries(idx, ["X", "Y", "Z"], 0.5, np.random.default_rng(7))
    b = assign_aggregate_queries(idx, ["X", "Y", "Z"], 0.5, np.random.default_rng(7))
    assert a["query"].to_list() == b["query"].to_list()
    share = sum(is_aggregate(q) for q in a["query"].to_list()) / a.height
    assert 0.3 < share < 0.7


def test_assign_needs_enough_components():
    idx = pl.DataFrame({"query": ["A"], "duration_days": [30.0]}).with_columns(
        pl.col("duration_days").cast(pl.Float32)
    )
    with pytest.raises(ValueError, match="at least 3"):
        assign_aggregate_queries(idx, ["X", "Y"], 1.0, np.random.default_rng(0))


def test_aggregates_do_not_perturb_the_duration_draw():
    """Own seed axis: switching aggregates on must not move the sampler's other draws."""
    ctx = pl.DataFrame(
        {"subject_id": [1, 2], "prediction_time": [PT, PT]},
        schema={"subject_id": pl.Int64, "prediction_time": pl.Datetime("us")},
    )
    plain = build_sequence_index_df(ctx, ["A", "B", "C"], 3, 3, 1, 365, seed=5)
    agg = build_sequence_index_df(
        ctx, ["A", "B", "C"], 3, 3, 1, 365, seed=5, aggregate_fraction=0.5, component_codes=["A", "B", "C"]
    )
    assert plain["duration_days"].to_list() == agg["duration_days"].to_list()


# ── 3. batch / model ────────────────────────────────────────────────────


def _tiny_model() -> ConditionalQueryModel:
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
    )
    model.eval()
    return model


def _batch(ops=None, comps=None, gap_lo=None, gap_hi=None, q_codes=(7, 8)):
    kwargs = {}
    if ops is not None:
        kwargs = {
            "q_op": torch.tensor([ops]),
            "q_comp_codes": torch.tensor([comps]),
            "q_gap_lo": torch.tensor([gap_lo or [0.0] * len(ops)]),
            "q_gap_hi": torch.tensor([gap_hi or [0.0] * len(ops)]),
        }
    return ConditionalQueryBatch(
        code=torch.tensor([[3, 4, 5, 6]]),
        numeric_value=torch.zeros(1, 4),
        numeric_value_mask=torch.zeros(1, 4, dtype=torch.bool),
        time_delta_days=torch.zeros(1, 4),
        q_codes=torch.tensor([list(q_codes)]),
        q_durations=torch.tensor([[30.0, 7.0]]),
        q_answers=torch.tensor([[ANSWER_YES, ANSWER_NO]]),
        q_mask=torch.tensor([[True, True]]),
        **kwargs,
    )


_ATOM_COMPS = [[7, 0, 0], [8, 0, 0]]


def test_all_atomic_batch_is_identical_to_no_feature():
    """The safety property: an all-atom op column changes nothing."""
    model = _tiny_model()
    with torch.no_grad():
        _, plain = model(_batch())
        _, atoms = model(_batch(ops=[OP_ATOM, OP_ATOM], comps=_ATOM_COMPS))
    assert torch.equal(plain.answer_logits, atoms.answer_logits)


def test_aggregate_reaches_the_code_slot():
    model = _tiny_model()
    with torch.no_grad():
        _, atom = model(_batch(ops=[OP_ATOM, OP_ATOM], comps=_ATOM_COMPS))
        _, agg = model(_batch(ops=[OP_ALL, OP_ATOM], comps=[[7, 9, 0], [8, 0, 0]]))
    assert not torch.equal(atom.answer_logits, agg.answer_logits)


def test_operator_identity_matters():
    """ALL(A|B) and ANY(A|B) are different questions over the same components."""
    model = _tiny_model()
    comps = [[7, 9, 0], [8, 0, 0]]
    with torch.no_grad():
        _, all_op = model(_batch(ops=[OP_ALL, OP_ATOM], comps=comps))
        _, any_op = model(_batch(ops=[OP_ANY, OP_ATOM], comps=comps))
    assert not torch.equal(all_op.answer_logits, any_op.answer_logits)


def test_component_order_matters_for_seq():
    """The role embedding is what makes SEQ(A>B) differ from SEQ(B>A)."""
    model = _tiny_model()
    with torch.no_grad():
        _, ab = model(_batch(ops=[OP_SEQ, OP_ATOM], comps=[[7, 9, 0], [8, 0, 0]], gap_hi=[7.0, 0.0]))
        _, ba = model(_batch(ops=[OP_SEQ, OP_ATOM], comps=[[9, 7, 0], [8, 0, 0]], gap_hi=[7.0, 0.0]))
    assert not torch.equal(ab.answer_logits, ba.answer_logits)


def test_gap_bounds_matter_for_temporal_operators():
    model = _tiny_model()
    comps = [[7, 9, 0], [8, 0, 0]]
    with torch.no_grad():
        _, narrow = model(_batch(ops=[OP_SEQ, OP_ATOM], comps=comps, gap_hi=[1.0, 0.0]))
        _, wide = model(_batch(ops=[OP_SEQ, OP_ATOM], comps=comps, gap_hi=[30.0, 0.0]))
    assert not torch.equal(narrow.answer_logits, wide.answer_logits)


def test_gap_is_ignored_for_non_temporal_operators():
    """ALL has no gap; a stray value must not leak into its representation."""
    model = _tiny_model()
    comps = [[7, 9, 0], [8, 0, 0]]
    with torch.no_grad():
        _, a = model(_batch(ops=[OP_ALL, OP_ATOM], comps=comps, gap_hi=[0.0, 0.0]))
        _, b = model(_batch(ops=[OP_ALL, OP_ATOM], comps=comps, gap_hi=[30.0, 0.0]))
    assert torch.equal(a.answer_logits, b.answer_logits)


def test_padding_components_do_not_drag_the_mean():
    """A 2-component query must embed the same whether or not the third slot exists."""
    model = _tiny_model()
    with torch.no_grad():
        emb2 = model._query_code_embeds(_batch(ops=[OP_ALL, OP_ATOM], comps=[[7, 9, 0], [8, 0, 0]]))
        emb3 = model._query_code_embeds(_batch(ops=[OP_ALL, OP_ATOM], comps=[[7, 9, 10], [8, 0, 0]]))
    assert not torch.equal(emb2[:, 0], emb3[:, 0]), "a real third component must count"


def test_aggregate_parameters_receive_gradient():
    model = _tiny_model()
    model.train()
    loss, _ = model(_batch(ops=[OP_SEQ, OP_ATOM], comps=[[7, 9, 0], [8, 0, 0]], gap_hi=[7.0, 0.0]))
    loss.backward()
    for name, param in (
        ("op_embed", model.op_embed.weight),
        ("comp_role_embed", model.comp_role_embed.weight),
    ):
        assert param.grad is not None and param.grad.abs().sum() > 0, f"{name} got no gradient"


def test_batch_validates_component_shape():
    with pytest.raises(ValueError, match="q_comp_codes"):
        ConditionalQueryBatch(
            code=torch.tensor([[3, 4]]),
            numeric_value=torch.zeros(1, 2),
            numeric_value_mask=torch.zeros(1, 2, dtype=torch.bool),
            time_delta_days=torch.zeros(1, 2),
            q_codes=torch.tensor([[7, 8]]),
            q_durations=torch.tensor([[30.0, 7.0]]),
            q_answers=torch.tensor([[ANSWER_YES, ANSWER_NO]]),
            q_mask=torch.tensor([[True, True]]),
            q_op=torch.tensor([[OP_ATOM, OP_ATOM]]),
            q_comp_codes=torch.zeros(1, 5, MAX_COMPONENTS, dtype=torch.long),  # wrong L
            q_gap_lo=torch.zeros(1, 2),
            q_gap_hi=torch.zeros(1, 2),
        )
