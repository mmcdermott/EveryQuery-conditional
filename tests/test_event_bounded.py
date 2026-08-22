"""Tests for event-bounded duration queries.

A query's window may end at the next occurrence of a **boundary event** rather than after a
fixed horizon.  Covered here, in pipeline order:

1. Labeling — that an occurrence after the boundary is outside the window while one sharing the
   boundary's instant is inside it, that time- and event-bounded queries can coexist in one
   sequence, and that the degenerate "boundary never fires" case behaves as documented
   (window runs to the end of the record) and is *reported* rather than hidden.
2. Sampling — that bounds are drawn on their own seed axis, so turning the feature on does not
   perturb the code/duration draw that the sampler's parity contract depends on.
3. Dataset/batch — that ``bound_events`` is optional on disk, and that an unknown boundary code
   raises instead of silently decaying into an unbounded query.
4. Model — that the boundary reaches the duration slot, that it is block-local, and that a
   bound-free batch is answered *identically* to a model without the feature.
"""

from datetime import datetime

import numpy as np
import polars as pl
import pytest
import torch
import yaml
from omegaconf import OmegaConf

from every_query.data.seq_dataset import (
    EVENT_BOUND_DURATION_SENTINEL,
    NO_BOUND_INDEX,
    ConditionalQueryBatch,
)
from every_query.generate_tasks.sample_query_sequences import (
    BOUND_COL,
    assign_event_bounds,
    build_sequence_index_df,
    label_binary_occurrence,
    label_query_sequences,
    label_with_event_bounds,
    log_degenerate_bounds,
    resolve_bound_events,
)
from every_query.model.conditional_model import ANSWER_NO, ANSWER_YES, ConditionalQueryModel

PT = datetime(2024, 1, 1)


def _events(rows: list[tuple[int, datetime, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        {"subject_id": [r[0] for r in rows], "time": [r[1] for r in rows], "code": [r[2] for r in rows]},
        schema={"subject_id": pl.Int64, "time": pl.Datetime("us"), "code": pl.Utf8},
    )


def _index(queries, durations, bounds=None, subject: int = 1) -> pl.DataFrame:
    n = len(queries)
    data = {
        "_ctx_id": [0] * n,
        "_position": list(range(n)),
        "subject_id": [subject] * n,
        "prediction_time": [PT] * n,
        "query": list(queries),
        "duration_days": [float(d) for d in durations],
    }
    schema = {
        "_ctx_id": pl.UInt32,
        "_position": pl.Int64,
        "subject_id": pl.Int64,
        "prediction_time": pl.Datetime("us"),
        "query": pl.Utf8,
        "duration_days": pl.Float32,
    }
    if bounds is not None:
        data[BOUND_COL] = list(bounds)
        schema[BOUND_COL] = pl.Utf8
    return pl.DataFrame(data, schema=schema)


# ── 1. labeling semantics ───────────────────────────────────────────────


def test_boundary_closes_the_window():
    """An occurrence after the boundary does not count, even well inside any horizon."""
    events = _events(
        [
            (1, datetime(2024, 1, 5), "DISCHARGE"),
            (1, datetime(2024, 1, 9), "SEPSIS"),  # after the discharge
        ]
    )
    idx = _index(["SEPSIS"], [EVENT_BOUND_DURATION_SENTINEL], ["DISCHARGE"])
    assert label_with_event_bounds(idx, events).row(0, named=True)["answers"] == [False]


def test_occurrence_before_boundary_counts():
    events = _events(
        [
            (1, datetime(2024, 1, 3), "SEPSIS"),
            (1, datetime(2024, 1, 5), "DISCHARGE"),
        ]
    )
    idx = _index(["SEPSIS"], [EVENT_BOUND_DURATION_SENTINEL], ["DISCHARGE"])
    assert label_with_event_bounds(idx, events).row(0, named=True)["answers"] == [True]


def test_boundary_is_inclusive():
    """An occurrence exactly *at* the boundary instant is inside the window.

    The window is closed at the top everywhere (see the RESOLUTION note in
    ``tests/test_event_bounds_oracle.py``), and for a bounded query the top IS the boundary
    event's timestamp -- so a SEPSIS charted in the same instant as the DISCHARGE counts as
    having happened before it.  MEDS clusters codes onto one timestamp, so this is a common
    shape rather than an edge case.
    """
    same = datetime(2024, 1, 5)
    events = _events([(1, same, "SEPSIS"), (1, same, "DISCHARGE")])
    idx = _index(["SEPSIS"], [EVENT_BOUND_DURATION_SENTINEL], ["DISCHARGE"])
    assert label_with_event_bounds(idx, events).row(0, named=True)["answers"] == [True]


def test_mixed_sequence_labels_each_query_by_its_own_rule():
    """One sequence may mix both kinds; each query must use its own window."""
    events = _events(
        [
            (1, datetime(2024, 1, 3), "SEPSIS"),
            (1, datetime(2024, 1, 4), "DISCHARGE"),
            (1, datetime(2024, 1, 10), "LATE"),
        ]
    )
    idx = _index(
        ["SEPSIS", "LATE", "LATE"],
        [30.0, 30.0, EVENT_BOUND_DURATION_SENTINEL],
        [None, None, "DISCHARGE"],
    )
    answers = label_with_event_bounds(idx, events).row(0, named=True)["answers"]
    # SEPSIS within 30d: yes.  LATE within 30d: yes.  LATE before discharge (day 4): no.
    assert answers == [True, True, False]


def test_unbounded_rows_match_the_plain_labeler_exactly():
    """With every bound null, the bound-aware labeler must agree with the original."""
    events = _events(
        [
            (1, datetime(2024, 1, 3), "A"),
            (1, datetime(2024, 1, 20), "B"),
        ]
    )
    queries, durations = ["A", "B", "A"], [30.0, 5.0, 1.0]
    plain = label_binary_occurrence(_index(queries, durations), events).row(0, named=True)
    bounded = label_with_event_bounds(_index(queries, durations, [None, None, None]), events).row(
        0, named=True
    )
    assert plain["answers"] == bounded["answers"]


def test_missing_boundary_runs_to_end_of_record():
    """Documented degenerate case: no boundary occurrence -> 'does it ever occur again'."""
    events = _events([(1, datetime(2024, 6, 1), "SEPSIS")])  # no DISCHARGE at all
    idx = _index(["SEPSIS"], [EVENT_BOUND_DURATION_SENTINEL], ["DISCHARGE"])
    assert label_with_event_bounds(idx, events).row(0, named=True)["answers"] == [True]


def test_degenerate_rate_is_reported():
    """The degenerate case is legitimate but misleading, so it must be measured, not hidden."""
    events = _events([(1, datetime(2024, 1, 4), "DISCHARGE")])
    idx = _index(
        ["A", "A"],
        [EVENT_BOUND_DURATION_SENTINEL] * 2,
        ["DISCHARGE", "NEVER_HAPPENS"],
    )
    rates = log_degenerate_bounds(idx, events)
    assert rates["DISCHARGE"] == 0.0
    assert rates["NEVER_HAPPENS"] == 1.0


def test_dispatch_keys_on_the_frame_not_a_flag():
    """An index carrying bounds is always labelled bound-aware, flag or no flag."""
    events = _events([(1, datetime(2024, 1, 9), "SEPSIS"), (1, datetime(2024, 1, 5), "DISCHARGE")])
    bounded = label_query_sequences(
        _index(["SEPSIS"], [EVENT_BOUND_DURATION_SENTINEL], ["DISCHARGE"]), events
    )
    assert "bound_events" in bounded.columns
    assert bounded.row(0, named=True)["answers"] == [False]

    plain = label_query_sequences(_index(["SEPSIS"], [30.0]), events)
    assert "bound_events" not in plain.columns


# ── 2. sampling ─────────────────────────────────────────────────────────


def test_bounds_do_not_perturb_the_code_duration_draw():
    """The whole point of the separate seed axis: parity with an unbounded run is preserved."""
    ctx = pl.DataFrame(
        {"subject_id": [1, 2], "prediction_time": [PT, PT]},
        schema={"subject_id": pl.Int64, "prediction_time": pl.Datetime("us")},
    )
    plain = build_sequence_index_df(ctx, ["A", "B", "C"], 3, 3, 1, 365, seed=11)
    bounded = build_sequence_index_df(
        ctx, ["A", "B", "C"], 3, 3, 1, 365, seed=11, eventbound_fraction=0.5, bound_events=["X"]
    )
    assert plain["query"].to_list() == bounded["query"].to_list(), (
        "turning bounds on must not change which codes were drawn"
    )
    # Durations change only where a bound replaced them with the sentinel.
    unbounded_rows = bounded[BOUND_COL].is_null()
    assert (
        bounded.filter(unbounded_rows)["duration_days"].to_list()
        == plain.filter(unbounded_rows)["duration_days"].to_list()
    )


def test_assign_event_bounds_requires_a_pool():
    idx = pl.DataFrame({"query": ["A"], "duration_days": [30.0]}).with_columns(
        pl.col("duration_days").cast(pl.Float32)
    )
    with pytest.raises(ValueError, match="at least one boundary code"):
        assign_event_bounds(idx, [], 0.5, np.random.default_rng(0))


def test_assign_event_bounds_rejects_a_bad_fraction():
    idx = pl.DataFrame({"query": ["A"], "duration_days": [30.0]}).with_columns(
        pl.col("duration_days").cast(pl.Float32)
    )
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        assign_event_bounds(idx, ["X"], 1.5, np.random.default_rng(0))


def test_bound_events_may_be_a_yaml_path_not_only_a_literal_list(tmp_path):
    """Real boundary codes are unusable as a Hydra CLI list, so a file has to work too.

    They carry spaces, periods and parentheses ("HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM"), which Hydra's
    override grammar cannot parse as a bare list — so a literal-list-only knob means the documented codes
    cannot be passed at all.
    """
    codes = ["HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM", "ICU_DISCHARGE//STAY (MICU)"]
    fp = tmp_path / "bounds.yaml"
    fp.write_text(yaml.safe_dump(codes))

    cfg = OmegaConf.create({"eventbound_fraction": 0.3, "bound_events": str(fp)})
    assert resolve_bound_events(cfg, [*codes, "LAB//X"]) == codes

    # The literal-list form keeps working unchanged.
    literal = OmegaConf.create({"eventbound_fraction": 0.3, "bound_events": codes})
    assert resolve_bound_events(literal, [*codes, "LAB//X"]) == codes


def test_bound_events_from_a_path_are_still_vocabulary_checked(tmp_path):
    """Reading from a file must not become a way to smuggle an unknown boundary code in."""
    fp = tmp_path / "bounds.yaml"
    fp.write_text(yaml.safe_dump(["NOT//IN//VOCAB"]))
    cfg = OmegaConf.create({"eventbound_fraction": 0.3, "bound_events": str(fp)})
    with pytest.raises(ValueError, match="NOT//IN//VOCAB"):
        resolve_bound_events(cfg, ["LAB//X"])


def test_bound_events_is_required_when_the_fraction_is_on():
    cfg = OmegaConf.create({"eventbound_fraction": 0.3, "bound_events": None})
    with pytest.raises(ValueError, match="bound_events"):
        resolve_bound_events(cfg, ["LAB//X"])


def test_bound_draw_is_deterministic():
    idx = pl.DataFrame({"query": list("ABCDEFGH"), "duration_days": [30.0] * 8}).with_columns(
        pl.col("duration_days").cast(pl.Float32)
    )
    a = assign_event_bounds(idx, ["X", "Y"], 0.5, np.random.default_rng(4))
    b = assign_event_bounds(idx, ["X", "Y"], 0.5, np.random.default_rng(4))
    assert a[BOUND_COL].to_list() == b[BOUND_COL].to_list()


# ── 3. dataset / batch ──────────────────────────────────────────────────


def test_batch_without_bounds_is_still_valid():
    """The column is optional end to end; a pre-feature dataset must keep working."""
    batch = ConditionalQueryBatch(
        code=torch.tensor([[3, 4]]),
        numeric_value=torch.zeros(1, 2),
        numeric_value_mask=torch.zeros(1, 2, dtype=torch.bool),
        time_delta_days=torch.zeros(1, 2),
        q_codes=torch.tensor([[7, 8]]),
        q_durations=torch.tensor([[30.0, 7.0]]),
        q_answers=torch.tensor([[ANSWER_YES, ANSWER_NO]]),
        q_mask=torch.tensor([[True, True]]),
    )
    assert batch.q_bound_codes is None


def test_batch_validates_bound_shape():
    with pytest.raises(ValueError, match="q_bound_codes"):
        ConditionalQueryBatch(
            code=torch.tensor([[3, 4]]),
            numeric_value=torch.zeros(1, 2),
            numeric_value_mask=torch.zeros(1, 2, dtype=torch.bool),
            time_delta_days=torch.zeros(1, 2),
            q_codes=torch.tensor([[7, 8]]),
            q_durations=torch.tensor([[30.0, 7.0]]),
            q_answers=torch.tensor([[ANSWER_YES, ANSWER_NO]]),
            q_mask=torch.tensor([[True, True]]),
            q_bound_codes=torch.tensor([[1]]),  # wrong width
        )


# ── 4. model ────────────────────────────────────────────────────────────


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


def _batch(q_bound_codes=None, n_queries: int = 2) -> ConditionalQueryBatch:
    return ConditionalQueryBatch(
        code=torch.tensor([[3, 4, 5, 6]]),
        numeric_value=torch.zeros(1, 4),
        numeric_value_mask=torch.zeros(1, 4, dtype=torch.bool),
        time_delta_days=torch.zeros(1, 4),
        q_codes=torch.tensor([[7, 8][:n_queries]]),
        q_durations=torch.tensor([[30.0, 7.0][:n_queries]]),
        q_answers=torch.tensor([[ANSWER_YES, ANSWER_NO][:n_queries]]),
        q_mask=torch.tensor([[True, True][:n_queries]]),
        q_bound_codes=None if q_bound_codes is None else torch.tensor([q_bound_codes]),
    )


def test_no_bounds_is_identical_to_the_feature_being_absent():
    """The safety property: an all-zero bound column changes nothing at all."""
    model = _tiny_model()
    with torch.no_grad():
        _, without = model(_batch())
        _, all_zero = model(_batch(q_bound_codes=[NO_BOUND_INDEX, NO_BOUND_INDEX]))
    assert torch.equal(without.answer_logits, all_zero.answer_logits)


def test_boundary_code_changes_the_prediction():
    model = _tiny_model()
    with torch.no_grad():
        _, unbounded = model(_batch(q_bound_codes=[NO_BOUND_INDEX, NO_BOUND_INDEX]))
        _, bounded = model(_batch(q_bound_codes=[9, NO_BOUND_INDEX]))
    assert not torch.equal(unbounded.answer_logits, bounded.answer_logits)


def test_different_boundaries_give_different_predictions():
    """'before discharge' and 'before death' must not be the same question."""
    model = _tiny_model()
    with torch.no_grad():
        _, a = model(_batch(q_bound_codes=[9, NO_BOUND_INDEX]))
        _, b = model(_batch(q_bound_codes=[10, NO_BOUND_INDEX]))
    assert not torch.equal(a.answer_logits, b.answer_logits)


def test_bound_does_not_leak_backwards_across_blocks():
    """Block-causal structure holds: bounding block 1 must not move block 0's answer."""
    model = _tiny_model()
    with torch.no_grad():
        _, base = model(_batch(q_bound_codes=[NO_BOUND_INDEX, NO_BOUND_INDEX]))
        _, later = model(_batch(q_bound_codes=[NO_BOUND_INDEX, 9]))
    assert torch.equal(base.answer_logits[:, 0], later.answer_logits[:, 0]), (
        "a bound on a later query must not change an earlier query's answer"
    )
    assert not torch.equal(base.answer_logits[:, 1], later.answer_logits[:, 1])


def test_bound_marker_receives_gradient():
    """The marker is what separates 'bounded by X' from 'asking about X'; it must train."""
    model = _tiny_model()
    model.train()
    loss, _ = model(_batch(q_bound_codes=[9, NO_BOUND_INDEX]))
    loss.backward()
    assert model.bound_marker.grad is not None
    assert torch.isfinite(model.bound_marker.grad).all()
    assert model.bound_marker.grad.abs().sum() > 0
