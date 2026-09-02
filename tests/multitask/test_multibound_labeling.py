"""Semantic + differential tests for the multi-bound packed labeling kernel (``label_multitask_index``).

Every window rule from issue #20 is pinned directly, and the whole kernel is then compared - for every
``(context, boundary, code)`` - against the scalar ``label_with_event_bounds`` oracle on randomized
inputs.
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from every_query.generate_tasks import sample_multitask_sequences as sms
from every_query.generate_tasks.sample_multitask_sequences import (
    TargetVocabulary,
    label_multitask_index,
    validate_index,
)
from tests.multitask.conftest import CODES, make_events, make_index, scalar_oracle

T0 = datetime(2024, 1, 1)
VOCAB = TargetVocabulary.from_pairs(["A", "B", "DISCHARGE", "TIMELINE//END"], [1, 2, 3, 4])


def _events(rows: list[tuple[int, datetime | None, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "subject_id": pl.Series([r[0] for r in rows], dtype=pl.Int64),
            "time": pl.Series([r[1] for r in rows], dtype=pl.Datetime("us")),
            "code": pl.Series([r[2] for r in rows], dtype=pl.Utf8),
        }
    )


def _dense(
    index_df: pl.DataFrame, events_df: pl.DataFrame, vocab: TargetVocabulary = VOCAB, k: int = 1, **kw
):
    meta, packed, _ = label_multitask_index(index_df, events_df, vocab, k, **kw)
    return meta, np.unpackbits(packed, axis=-1, count=vocab.size, bitorder="little").astype(bool)


def _row(code: str, dense: np.ndarray, vocab: TargetVocabulary = VOCAB, k: int = 0) -> bool:
    return bool(dense[0, k, vocab.code_to_index()[code]])


# --- semantic tests -------------------------------------------------------------------------------


def test_occurrence_at_prediction_time_is_false_and_just_after_is_true() -> None:
    ev = _events([(1, T0, "A"), (1, T0 + timedelta(microseconds=1), "B")])
    _, d = _dense(make_index([(1, T0)], [[(10.0, None)]]), ev)
    assert _row("A", d) is False
    assert _row("B", d) is True


def test_occurrence_at_duration_endpoint_is_false_and_just_before_is_true() -> None:
    ev = _events([(1, T0 + timedelta(days=10), "A"), (1, T0 + timedelta(days=10, microseconds=-1), "B")])
    _, d = _dense(make_index([(1, T0)], [[(10.0, None)]]), ev)
    assert _row("A", d) is False
    assert _row("B", d) is True


def test_fractional_day_duration() -> None:
    ev = _events([(1, T0 + timedelta(hours=11), "A"), (1, T0 + timedelta(hours=13), "B")])
    _, d = _dense(make_index([(1, T0)], [[(0.5, None)]]), ev)
    assert _row("A", d) is True
    assert _row("B", d) is False


def test_event_bound_excludes_boundary_instant_and_boundary_code_itself() -> None:
    ev = _events(
        [
            (1, T0 + timedelta(days=1), "A"),
            (1, T0 + timedelta(days=3), "DISCHARGE"),
            (1, T0 + timedelta(days=3), "B"),  # co-timestamped with the boundary
            (1, T0 + timedelta(days=5), "TIMELINE//END"),
        ]
    )
    _, d = _dense(make_index([(1, T0)], [[(-1.0, "DISCHARGE")]]), ev)
    assert _row("A", d) is True
    assert _row("B", d) is False
    assert _row("DISCHARGE", d) is False
    assert _row("TIMELINE//END", d) is False


def test_every_code_co_timestamped_with_boundary_is_excluded() -> None:
    t = T0 + timedelta(days=2)
    ev = _events([(1, t, "A"), (1, t, "B"), (1, t, "DISCHARGE"), (1, t, "TIMELINE//END")])
    _, d = _dense(make_index([(1, T0)], [[(-1.0, "DISCHARGE")]]), ev)
    assert d.sum() == 0


def test_missing_boundary_event_means_infinity() -> None:
    ev = _events([(1, T0 + timedelta(days=900), "A"), (1, T0 + timedelta(days=901), "TIMELINE//END")])
    _, d = _dense(make_index([(1, T0)], [[(-1.0, "DISCHARGE")]]), ev)
    assert _row("A", d) is True
    assert _row("TIMELINE//END", d) is True
    assert _row("DISCHARGE", d) is False


def test_boundary_occurrence_at_prediction_time_does_not_close_window() -> None:
    ev = _events(
        [(1, T0, "DISCHARGE"), (1, T0 + timedelta(days=1), "A"), (1, T0 + timedelta(days=4), "DISCHARGE")]
    )
    _, d = _dense(make_index([(1, T0)], [[(-1.0, "DISCHARGE")]]), ev)
    assert _row("A", d) is True


def test_no_later_occurrence_is_false() -> None:
    ev = _events([(1, T0 - timedelta(days=1), "A"), (1, T0 + timedelta(days=3), "B")])
    _, d = _dense(make_index([(1, T0)], [[(30.0, None)]]), ev)
    assert _row("A", d) is False
    assert _row("B", d) is True


def test_duplicate_events_and_null_time_events() -> None:
    ev = _events(
        [
            (1, T0 + timedelta(days=1), "A"),
            (1, T0 + timedelta(days=1), "A"),
            (1, None, "B"),  # static: ignored
            (1, None, "DISCHARGE"),
        ]
    )
    _, d = _dense(make_index([(1, T0)], [[(10.0, None), (-1.0, "DISCHARGE")]]), ev, k=2)
    assert d[0, 0].tolist() == [False, True, False, False, False]
    assert d[0, 1].tolist() == [False, True, False, False, False]  # boundary never occurs in time => inf


def test_timeline_end_is_an_ordinary_target() -> None:
    ev = _events([(1, T0 + timedelta(days=2), "TIMELINE//END")])
    _, d = _dense(make_index([(1, T0), (1, T0)], [[(3.0, None)], [(1.0, None)]]), ev)
    assert d[:, 0, VOCAB.code_to_index()["TIMELINE//END"]].tolist() == [True, False]


def test_pad_bit_stays_false_and_unknown_event_codes_are_ignored() -> None:
    vocab = TargetVocabulary.from_pairs(["PAD", "A"], [0, 1])
    ev = _events(
        [
            (1, T0 + timedelta(days=1), "PAD"),
            (1, T0 + timedelta(days=1), "A"),
            (1, T0 + timedelta(days=1), "ZZZ"),
        ]
    )
    _, packed, stats = label_multitask_index(make_index([(1, T0)], [[(5.0, None)]]), ev, vocab, 1)
    d = np.unpackbits(packed, axis=-1, count=vocab.size, bitorder="little").astype(bool)
    assert d[0, 0].tolist() == [False, True]  # an observed PAD event never sets bit 0
    assert stats.n_unknown_code_events == 1
    assert vocab.boundary_candidates() == ["A"]


def test_unknown_boundary_code_is_a_hard_error() -> None:
    ev = _events([(1, T0 + timedelta(days=1), "A")])
    with pytest.raises(ValueError, match="not in the base vocabulary"):
        label_multitask_index(make_index([(1, T0)], [[(-1.0, "NOPE")]]), ev, VOCAB, 1)


def test_pad_boundary_code_is_a_hard_error() -> None:
    vocab = TargetVocabulary.from_pairs(["PAD", "A"], [0, 1])
    ev = _events([(1, T0 + timedelta(days=1), "A")])
    with pytest.raises(ValueError, match="PAD"):
        label_multitask_index(make_index([(1, T0)], [[(-1.0, "PAD")]]), ev, vocab, 1)


def test_validate_index_rejects_wrong_arity_and_mixed_representation() -> None:
    with pytest.raises(ValueError, match="exactly 2"):
        validate_index(make_index([(1, T0)], [[(1.0, None)]]), 2)
    with pytest.raises(ValueError, match="either duration"):
        validate_index(make_index([(1, T0)], [[(5.0, "A")]]), 1)
    with pytest.raises(ValueError, match="either duration"):
        validate_index(make_index([(1, T0)], [[(-1.0, None)]]), 1)
    with pytest.raises(ValueError, match="missing required column"):
        validate_index(make_index([(1, T0)], [[(1.0, None)]]).drop("durations"), 1)


def test_ontology_dir_raises() -> None:
    ev = _events([(1, T0 + timedelta(days=1), "A")])
    with pytest.raises(NotImplementedError, match="observable leaf codes only"):
        label_multitask_index(make_index([(1, T0)], [[(1.0, None)]]), ev, VOCAB, 1, ontology_dir="x")
    with pytest.raises(NotImplementedError):
        sms.prepare_events_for_labeling(ev, ontology_dir="x")
    with pytest.raises(NotImplementedError):
        sms.build_target_vocabulary("anything", ontology_dir="x")


def test_vocab_size_not_divisible_by_eight_and_absent_codes_keep_bits_false() -> None:
    vocab = TargetVocabulary.from_pairs(["A", "B", "Z"], [1, 2, 12])  # V = 13 -> 2 packed bytes
    ev = _events([(1, T0 + timedelta(days=1), "A"), (1, T0 + timedelta(days=2), "B")])
    _, packed, _ = label_multitask_index(make_index([(1, T0)], [[(5.0, None)]]), ev, vocab, 1)
    assert packed.shape == (1, 1, 2)
    d = np.unpackbits(packed, axis=-1, count=vocab.size, bitorder="little").astype(bool)
    assert d.shape == (1, 1, 13)
    assert d[0, 0, [1, 2]].tolist() == [True, True]
    assert not d[0, 0, 12] and not d[0, 0, 0]


# --- ordering / chunking / supplied-index contract -------------------------------------------------


def test_output_identical_for_different_initial_context_order_and_chunk_sizes() -> None:
    rng = np.random.default_rng(7)
    events = make_events(rng, [1, 2, 3], codes=CODES)
    vocab = TargetVocabulary.from_pairs(CODES, list(range(1, len(CODES) + 1)))
    contexts = [(int(rng.integers(1, 4)), T0 + timedelta(days=int(rng.integers(0, 400)))) for _ in range(30)]
    bounds = [
        [
            (-1.0, CODES[int(rng.integers(0, 11))])
            if rng.random() < 0.5
            else (float(rng.uniform(1, 60)), None)
            for _ in range(3)
        ]
        for _ in contexts
    ]
    idx = make_index(contexts, bounds)
    meta_a, packed_a, _ = label_multitask_index(idx, events, vocab, 3, chunk_rows=4)
    perm = rng.permutation(idx.height)
    shuffled = idx.with_row_index("_ctx_id").with_columns(pl.col("_ctx_id").cast(pl.Int64))[perm.tolist()]
    meta_b, packed_b, _ = label_multitask_index(shuffled, events, vocab, 3, chunk_rows=1000)
    assert meta_a.equals(meta_b)
    assert np.array_equal(packed_a, packed_b)
    assert meta_a.sort("subject_id", "prediction_time").equals(meta_a)


def test_supplied_out_memmap_is_filled_in_place(tmp_path) -> None:
    ev = _events([(1, T0 + timedelta(days=1), "A")])
    idx = make_index([(1, T0)], [[(5.0, None)]])
    mm = np.lib.format.open_memmap(
        tmp_path / "x.npy", mode="w+", dtype=np.uint8, shape=(1, 1, VOCAB.packed_width)
    )
    _, out, _ = label_multitask_index(idx, ev, VOCAB, 1, out=mm)
    assert out is mm
    mm.flush()
    del mm
    assert np.load(tmp_path / "x.npy")[0, 0, 0] == 0b10
    with pytest.raises(ValueError, match="out must be uint8"):
        label_multitask_index(idx, ev, VOCAB, 1, out=np.zeros((2, 1, 1), np.uint8))


# --- differential oracle ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_differential_against_scalar_label_with_event_bounds(seed: int) -> None:
    rng = np.random.default_rng(seed)
    k = 5
    subjects = [1, 2, 3, 4]
    events = make_events(rng, subjects, n_times=(5, 25), horizon_days=60, codes=CODES)
    vocab = TargetVocabulary.from_pairs(CODES, list(range(1, len(CODES) + 1)))
    obs = CODES[:11]
    n = 24
    contexts = []
    for _ in range(n):
        sid = int(rng.integers(1, 6))  # 5 is an unknown subject: every bit false
        # Prediction times deliberately land on event instants often, to exercise the strict bounds.
        sub = events.filter(pl.col("subject_id") == sid)["time"].drop_nulls()
        if sub.len() and rng.random() < 0.7:
            pt = sub[int(rng.integers(0, sub.len()))]
        else:
            pt = T0 + timedelta(days=int(rng.integers(0, 60)))
        contexts.append((sid, pt))

    def pick_bound():
        if rng.random() < 0.5:
            return float(rng.uniform(0.1, 40)), None
        pool = CODES if rng.random() < 0.2 else obs  # TIMELINE//END is an eligible boundary too
        return -1.0, pool[int(rng.integers(0, len(pool)))]

    bounds = [[pick_bound() for _ in range(k)] for _ in range(n)]
    idx = make_index(contexts, bounds)
    meta, packed, stats = label_multitask_index(idx, events, vocab, k, chunk_rows=5)
    got = np.unpackbits(packed, axis=-1, count=vocab.size, bitorder="little").astype(bool)[
        :, :, vocab.indices
    ]
    expect = scalar_oracle(meta, events, list(vocab.codes), k)
    assert np.array_equal(got, expect)
    assert stats.n_contexts == n
