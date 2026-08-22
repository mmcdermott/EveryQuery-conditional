"""Differential test: `label_with_event_bounds` against a naive per-query oracle.

Companion to ``tests/test_rope_strip_oracle.py``, for the second of the two coverage gaps that
``THREE_FEATURES_VERIFICATION.md`` records: event-bounded labelling was asserted only against
hand-written expectations written by the author of the code, which proves the code matches the
author's belief, not that the belief is right.  Every defect this branch found produced a
silently wrong *label*, never a crash, so a shape assertion buys nothing here.

The oracle below answers one query at a time with plain Python loops over plain tuples -- no
polars, no ``join_asof``, no shared code path with the vectorised implementation.  It was written
from the written spec transcribed below *before* the body of ``label_with_event_bounds`` was
read, so the two cannot share an implementation bug: any disagreement localises a real defect in
one of them.

================================================================================================
THE SPEC, in my own words, from the sources named -- written down BEFORE the oracle
================================================================================================

Sources: the ``label_with_event_bounds`` docstring; the ``label_binary_occurrence`` docstring it
delegates to; ``QuerySeqSchema.bound_events`` in ``src/every_query/data/schema.py``; the
``eventbound_fraction`` / ``bound_events`` block of
``src/every_query/generate_tasks/configs/sample_query_sequences_config.yaml``.

Inputs.  ``index_df`` is one row per query: ``_ctx_id``, ``_position``, ``subject_id``,
``prediction_time``, ``query``, ``duration_days`` (float32), ``bound_event`` (nullable string).
``events_df`` is the observed record: ``subject_id``, ``time``, ``code``.

S1. A query whose ``bound_event`` is null is an ordinary time-bounded query and "behaves exactly
    as ``label_binary_occurrence``": the answer is True iff an event of the queried code occurs
    strictly inside ``(prediction_time, prediction_time + duration_days)``.  The lower bound is
    strict (docstring: enforced by shifting the asof key +1us).  See the AMBIGUITY note below on
    the upper bound.

S2. A query with a boundary code is answered over ``(prediction_time, boundary)`` instead, and
    ``duration_days`` is ignored -- it carries the ``EVENT_BOUND_DURATION_SENTINEL`` (-1.0), not
    a horizon.

S3. ``boundary`` is the FIRST occurrence of the boundary code STRICTLY AFTER the prediction time.
    This settles "which occurrence defines the bound" for repeated boundary events (the first
    eligible one) and for boundary events at or before the prediction time (not eligible -- the
    search starts strictly after ``prediction_time``, so earlier occurrences are invisible).

S4. The bound is STRICT: "an occurrence *at* the boundary instant is not inside the window."

S5. When no occurrence of the boundary code exists strictly after the prediction time, "the
    window runs to the end of the record", degenerating the query into "does this code ever
    occur again" -- i.e. True iff the queried code occurs at any time strictly after
    ``prediction_time``.  Documented as deliberate upstream semantics, not a bug.

S6. Consequences the spec forces, which the oracle re-derives rather than special-cases:
    - No events for the subject at all => False for every query form.
    - Zero-length window (``duration_days == 0``) or inverted window (``duration_days < 0``) on
      an UNBOUNDED query => the open interval is empty => always False.
    - ``bound_event == query`` => the first occurrence after ``prediction_time`` is simultaneously
      the boundary and the earliest candidate answer, and S4 excludes the boundary instant =>
      always False, whether or not the code recurs.

S7. Output: one row per ``_ctx_id`` -- ``subject_id``, ``prediction_time``, and the aligned list
    columns ``queries`` / ``durations`` / ``answers`` / ``bound_events``, each ordered by
    ``_position``.

AMBIGUITY, since RECONCILED: when this oracle was written the sources disagreed on the UPPER
bound of the time-bounded window.  Both labellers' prose said "strictly inside
``(prediction_time, prediction_time + duration_days)``" (open), while
``label_binary_occurrence``'s worked example annotated the same window ``(2,12]`` and
``QuerySeqSchema.answers`` documented ``(prediction_time, prediction_time + durations[j]]``
(closed).  The oracle implemented the OPEN reading, because that is the prose of the function
under test and the reading consistent with S4's strict event bound -- and the code agreed: both
labellers compare ``<``, so no emitted label was ever wrong.  The two closed-reading sources
have since been corrected to the open reading, so all four now say the same thing.

That reconciliation is what ``test_the_documented_upper_bound_matches_the_implemented_one``
defends.  It does not assert fixed prose: it *measures* the code's behaviour at the horizon
instant and then requires the docstrings' interval notation to agree with what it measured.  So
it stays honest if the behaviour is ever deliberately changed -- the docs would then have to
change with it -- and it goes red if either half drifts alone.  That matters more than it looks:
the risk here was never a wrong label today, it was a downstream consumer reimplementing the
closed reading from ``schema.py`` and flipping every event that lands exactly on the horizon.

NOT reconciled, and out of this file's edit scope -- reported rather than silently fixed:
``data/seq_dataset.py``, ``model/conditional_model.py``, ``generate_tasks/sample_tasks.py`` and
``generate_tasks/redesign-spec.md`` still spell the window closed.  The doc-consistency test
below is deliberately scoped to the sources that were reconciled, so it passes today; widen its
``_DOC_SOURCES`` tuple once those four are corrected.

RESULT: the implementation agrees with the oracle on every field of every context, across all
seeds.  No labelling defect found -- but a green oracle is only worth what it would have caught,
so the suite was run against ten hand-built wrong implementations of
``label_with_event_bounds``.  All ten go red here: `<` -> `<=` at the bound; dropping
``window_open`` so a never-recurring boundary closes the window; asof ``forward`` -> ``backward``
(last occurrence before, instead of first after); dropping the +1us shift that makes the lower
bound strict; falling back to the ``duration_days`` horizon for bounded queries (the feature
wired but dead); losing the ``_position`` sort so the list columns misalign; aggregating
``answers`` out of order; losing ``subject_id`` from the boundary join; looking the boundary up
by the query code; and writing the query into the ``bound_events`` column.

LATER PASS -- three thin spots closed, each proven by replaying the mutation it exists to catch:

1. ``label_binary_occurrence`` is the anchor this whole feature is judged against, yet the
   randomised sweep never called it; its only guard was one differential test against
   ``label_with_event_bounds``, which two separate mutations reduced to a single failure.  It
   now gets its own twelve-seed sweep AGAINST THE ORACLE (not against the other labeller, which
   shares its conventions and would move with it).  Dropping the ``+1us`` shift now fails 9
   seeds instead of 1; an inclusive upper bound fails 5; truncating the horizon to whole days
   fails a seed plus a dedicated test.
2. The sweep's unbounded durations were drawn from ``{0, -2, 0.25, 0.5, 1, 2, 5}``, which never
   produced an unbounded row carrying the ``EVENT_BOUND_DURATION_SENTINEL``.  That shape is the
   only one separating "is this bounded?" asked of the bound column from the same question
   asked of the duration, so that confusion rested on one hand-built row: with the sentinel
   removed from the pool, the mutation fails ONLY ``test_s2_...``; with it added, 7 of 12 seeds
   catch it.
3. ``assign_event_bounds`` had no coverage outside its own module doctest -- dropping the
   sentinel duration and inverting the bounded fraction both left this file and
   ``tests/test_event_bounded.py`` fully green (47 passed under each, replayed and confirmed).
   ``test_bound_draw_is_deterministic`` there cannot see the inversion: it compares two calls
   that any mutation perturbs identically.  Five tests now cover the draw itself.

Also added: ``test_the_documented_upper_bound_matches_the_implemented_one``, the only test in
the suite that fails when the DOCS alone regress -- reverting ``QuerySeqSchema.answers`` to the
closed reading is otherwise invisible to every test in this repo.
"""

import inspect
import itertools
import random
import re
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from every_query.data.schema import QuerySeqSchema
from every_query.generate_tasks.sample_query_sequences import (
    EVENT_BOUND_DURATION_SENTINEL,
    assign_event_bounds,
    label_binary_occurrence,
    label_with_event_bounds,
)

# ------------------------------------------------------------------------------------------
# The oracle: plain Python, one query at a time, straight from the spec above.
# ------------------------------------------------------------------------------------------


def _oracle_answer(subject_events, prediction_time, query, duration_days, bound_event):
    """Answer ONE query by hand.  `subject_events` is a list of (time, code) for this subject."""
    if bound_event is None:
        # S1: strictly inside the open horizon window.
        window_end = prediction_time + timedelta(days=float(duration_days))
        return any(code == query and prediction_time < t < window_end for t, code in subject_events)

    # S3: the boundary is the first occurrence of the bound code strictly after the pred time.
    boundary = None
    for t, code in subject_events:
        if code == bound_event and t > prediction_time and (boundary is None or t < boundary):
            boundary = t

    if boundary is None:
        # S5: no boundary ahead -> the window runs to the end of the record.
        return any(code == query and t > prediction_time for t, code in subject_events)

    # S4: strict on both ends.
    return any(code == query and prediction_time < t < boundary for t, code in subject_events)


def _oracle_label(index_rows, event_rows):
    """Whole-frame oracle.  Returns {ctx_id: dict of the row S7 describes}, plain Python."""
    by_subject: dict[int, list[tuple[datetime, str]]] = {}
    for subject_id, time, code in event_rows:
        by_subject.setdefault(subject_id, []).append((time, code))

    out: dict[int, dict] = {}
    for row in sorted(index_rows, key=lambda r: (r["_ctx_id"], r["_position"])):
        ctx = out.setdefault(
            row["_ctx_id"],
            {
                "subject_id": row["subject_id"],
                "prediction_time": row["prediction_time"],
                "queries": [],
                "durations": [],
                "answers": [],
                "bound_events": [],
            },
        )
        ctx["queries"].append(row["query"])
        ctx["durations"].append(row["duration_days"])
        ctx["bound_events"].append(row["bound_event"])
        ctx["answers"].append(
            _oracle_answer(
                by_subject.get(row["subject_id"], []),
                row["prediction_time"],
                row["query"],
                row["duration_days"],
                row["bound_event"],
            )
        )
    return out


# ------------------------------------------------------------------------------------------
# Adapter: the same inputs, shaped for the implementation.
# ------------------------------------------------------------------------------------------

INDEX_SCHEMA = {
    "_ctx_id": pl.UInt32,
    "_position": pl.Int64,
    "subject_id": pl.Int64,
    "prediction_time": pl.Datetime("us"),
    "query": pl.String,
    "duration_days": pl.Float32,
    "bound_event": pl.String,
}
EVENT_SCHEMA = {"subject_id": pl.Int64, "time": pl.Datetime("us"), "code": pl.String}


def _index_df(index_rows):
    return pl.DataFrame({k: [r[k] for r in index_rows] for k in INDEX_SCHEMA}, schema=INDEX_SCHEMA)


def _events_df(event_rows):
    return pl.DataFrame(
        {
            "subject_id": [r[0] for r in event_rows],
            "time": [r[1] for r in event_rows],
            "code": [r[2] for r in event_rows],
        },
        schema=EVENT_SCHEMA,
    )


# ------------------------------------------------------------------------------------------
# Randomised synthetic cases, degenerate ones included by construction.
# ------------------------------------------------------------------------------------------

SEEDS = range(12)
CODES = ["A", "B", "C", "DISCHARGE", "TIMELINE//END"]
BOUND_CODES = ["DISCHARGE", "TIMELINE//END", "A", "NEVER_HAPPENS"]
EPOCH = datetime(2024, 1, 1)
# Quarter-day grid: exactly representable in float32 days and in microsecond datetimes, so
# neither side can win or lose an edge case to rounding.
STEP = timedelta(hours=6)


def _random_case(seed, n_subjects=8, max_queries=6):
    """Build one (index_rows, event_rows) pair, deliberately dense with edge cases."""
    rng = random.Random(seed)
    event_rows: list[tuple[int, datetime, str]] = []
    index_rows: list[dict] = []
    ctx_id = 0

    for subject_id in range(1, n_subjects + 1):
        # One subject in five has an empty record.
        n_events = 0 if rng.random() < 0.2 else rng.randrange(1, 13)
        # A coarse grid over a short span, so repeated codes and simultaneous events -- an
        # occurrence landing exactly ON the boundary instant -- are common, not measure-zero.
        times = sorted(EPOCH + STEP * rng.randrange(0, 10) for _ in range(n_events))
        subject_events = [(subject_id, t, rng.choice(CODES)) for t in times]
        event_rows.extend(subject_events)

        for _ in range(rng.randrange(1, 4)):
            # Prediction times land ON the event grid half the time, so "exactly at the edge"
            # is a common case rather than a measure-zero one.
            if subject_events and rng.random() < 0.5:
                prediction_time = rng.choice(subject_events)[1]
            else:
                prediction_time = EPOCH + STEP * rng.randrange(0, 24)

            for position in range(rng.randrange(1, max_queries + 1)):
                bounded = rng.random() < 0.5
                if bounded:
                    bound_event = rng.choice(BOUND_CODES)
                    duration_days = -1.0
                else:
                    bound_event = None
                    # Zero-length and inverted windows included on purpose -- and so is the
                    # EVENT_BOUND_DURATION_SENTINEL itself on an UNBOUNDED row.  That shape is
                    # not a curiosity: it is the one input that separates "is this query
                    # bounded?" asked of the bound column (correct) from the same question
                    # asked of the duration (wrong).  Both readings agree on every other row
                    # in this sweep, because bounded rows always carry the sentinel and
                    # unbounded rows otherwise never do.  Here they disagree loudly: the
                    # bound column says unbounded, so the window is inverted and the answer is
                    # always False, while the sentinel reading would call the row bounded,
                    # find no boundary, and fall through to S5's "does it ever recur".
                    duration_days = rng.choice(
                        [0.0, -2.0, 0.25, 0.5, 1.0, 2.0, 5.0, EVENT_BOUND_DURATION_SENTINEL]
                    )
                index_rows.append(
                    {
                        "_ctx_id": ctx_id,
                        "_position": position,
                        "subject_id": subject_id,
                        "prediction_time": prediction_time,
                        "query": rng.choice(CODES),
                        "duration_days": duration_days,
                        "bound_event": bound_event,
                    }
                )
            ctx_id += 1

    return index_rows, event_rows


def _assert_matches_oracle(index_rows, event_rows, context=""):
    expected = _oracle_label(index_rows, event_rows)
    got = label_with_event_bounds(_index_df(index_rows), _events_df(event_rows))

    # `_ctx_id` is dropped from the output, but the rows come out in ascending `_ctx_id` order
    # (the implementation sorts by it before a `maintain_order` group_by), so zip them and
    # cross-check the identity columns -- a context whose rows leaked into a neighbour would
    # show up here rather than hiding behind a key-based match.
    assert got.height == len(expected), f"{context}: {got.height} output rows, {len(expected)} contexts"
    # No join scratch (`_pts`, `_q_time`, `_b_time`) may survive into the written parquet.
    assert set(got.columns) == {
        "subject_id",
        "prediction_time",
        "queries",
        "durations",
        "answers",
        "bound_events",
    }, f"{context}: unexpected output columns {sorted(got.columns)}"
    for row, ctx_id in zip(got.iter_rows(named=True), sorted(expected), strict=True):
        exp = expected[ctx_id]
        where = f"{context}: ctx {ctx_id}, subject {row['subject_id']} @ {row['prediction_time']}"
        assert row["subject_id"] == exp["subject_id"], f"{where}: subject_id"
        assert row["prediction_time"] == exp["prediction_time"], f"{where}: prediction_time"
        assert list(row["queries"]) == exp["queries"], f"{where}: queries"
        assert list(row["durations"]) == pytest.approx(exp["durations"]), f"{where}: durations"
        assert list(row["bound_events"]) == exp["bound_events"], f"{where}: bound_events"
        assert list(row["answers"]) == exp["answers"], (
            f"{where}: answers\n"
            f"  queries      {exp['queries']}\n"
            f"  bound_events {exp['bound_events']}\n"
            f"  durations    {exp['durations']}\n"
            f"  oracle       {exp['answers']}\n"
            f"  impl         {list(row['answers'])}"
        )


@pytest.mark.parametrize("seed", SEEDS)
def test_matches_naive_oracle_across_random_cases(seed):
    """Every field of every context row, against an independent implementation of the spec."""
    index_rows, event_rows = _random_case(seed)
    _assert_matches_oracle(index_rows, event_rows, context=f"seed {seed}")


def _unbounded_rows(index_rows):
    """The rows of a generated case that `label_binary_occurrence` can be handed directly."""
    return [dict(r) for r in index_rows if r["bound_event"] is None]


def _assert_plain_matches_oracle(index_rows, event_rows, context=""):
    """Same comparison as `_assert_matches_oracle`, but for the PLAIN labeller.

    Deliberately against the oracle, not against `label_with_event_bounds`.  The one existing
    cross-check of the plain path (`test_s1_null_bound_agrees_with_label_binary_occurrence...`)
    is a differential between two implementations that share a convention; a change made to
    both -- exactly what "fixing" the upper bound from the schema docstring would be -- passes
    it while relabelling real data.  The oracle is written from the spec and shares no code, so
    it does not move when the implementations do.
    """
    expected = _oracle_label(index_rows, event_rows)
    got = label_binary_occurrence(_index_df(index_rows).drop("bound_event"), _events_df(event_rows))

    assert got.height == len(expected), f"{context}: {got.height} rows, {len(expected)} contexts"
    assert set(got.columns) == {
        "subject_id",
        "prediction_time",
        "queries",
        "durations",
        "answers",
    }, f"{context}: unexpected output columns {sorted(got.columns)}"
    for row, ctx_id in zip(got.iter_rows(named=True), sorted(expected), strict=True):
        exp = expected[ctx_id]
        where = f"{context}: ctx {ctx_id}, subject {row['subject_id']} @ {row['prediction_time']}"
        assert row["subject_id"] == exp["subject_id"], f"{where}: subject_id"
        assert row["prediction_time"] == exp["prediction_time"], f"{where}: prediction_time"
        assert list(row["queries"]) == exp["queries"], f"{where}: queries"
        assert list(row["durations"]) == pytest.approx(exp["durations"]), f"{where}: durations"
        assert list(row["answers"]) == exp["answers"], (
            f"{where}: answers\n"
            f"  queries   {exp['queries']}\n"
            f"  durations {exp['durations']}\n"
            f"  oracle    {exp['answers']}\n"
            f"  impl      {list(row['answers'])}"
        )


@pytest.mark.parametrize("seed", SEEDS)
def test_plain_labeller_matches_naive_oracle_across_random_cases(seed):
    """The regression anchor gets its own sweep, not one hand-built differential.

    `label_with_event_bounds` documents itself as matching `label_binary_occurrence` on null
    bounds, which makes the plain labeller the reference the whole feature is judged against.
    Before this, the entire randomised sweep never called it: it was one deleted test away from
    no coverage at all.  Now every unbounded row of every seed is checked against the oracle,
    so the strict `+1us` lower bound and the fractional-day horizon are held by twelve seeds'
    worth of edge cases rather than by a single assertion.
    """
    index_rows, event_rows = _random_case(seed)
    unbounded = _unbounded_rows(index_rows)
    assert unbounded, f"seed {seed} generated no unbounded rows -- the sweep is not testing this"
    _assert_plain_matches_oracle(unbounded, event_rows, context=f"seed {seed} (plain)")


def _shapes_in(index_rows, event_rows):
    """Classify every generated query by which degenerate trap it lands in."""
    by_subject: dict[int, list] = {}
    for subject_id, time, code in event_rows:
        by_subject.setdefault(subject_id, []).append((time, code))

    kinds: list[str] = []
    for row in index_rows:
        events = by_subject.get(row["subject_id"], [])
        if not events:
            kinds.append("no_events")
        if row["bound_event"] is None:
            kinds.append("unbounded")
            if row["duration_days"] == 0:
                kinds.append("zero_length_window")
            if row["duration_days"] < 0:
                kinds.append("inverted_window")
            if row["duration_days"] == EVENT_BOUND_DURATION_SENTINEL:
                # The row that distinguishes "bounded" read off the bound column from
                # "bounded" read off the duration.  If the generator ever stops emitting it,
                # the sweep silently stops covering that confusion -- hence the guard.
                kinds.append("unbounded_carrying_the_sentinel_duration")
            # The two shapes that give the PLAIN labeller's sweep its teeth.  Both are about
            # rows where a wrong implementation returns a DIFFERENT answer, not merely a
            # different intermediate -- counting rows that merely *could* differ would let the
            # guard pass while the sweep had gone toothless.
            pt = row["prediction_time"]
            if any(t == pt and c == row["query"] for t, c in events):
                # Drop the +1us shift and this row's answer flips False -> True.
                kinds.append("query_exactly_at_prediction_time")
            horizon = float(row["duration_days"])
            if horizon > 0 and horizon != int(horizon):
                full = pt + timedelta(days=float(int(horizon)))
                end = pt + timedelta(days=horizon)
                if any(c == row["query"] and full <= t < end for t, c in events):
                    # Truncate the horizon to whole days and this row flips True -> False.
                    kinds.append("fractional_horizon_decides_the_answer")
            continue
        kinds.append("bounded")
        hits = sorted(t for t, c in events if c == row["bound_event"])
        after = [t for t in hits if t > row["prediction_time"]]
        if not hits:
            kinds.append("bound_never_occurs")
        if hits and not after:
            kinds.append("bound_only_before_window")
        if len(after) > 1:
            kinds.append("bound_repeats_after")
        if any(t == row["prediction_time"] for t in hits):
            kinds.append("bound_at_prediction_time")
        if row["bound_event"] == row["query"]:
            kinds.append("bound_is_query")
        if after and any(t == after[0] for t, c in events if c == row["query"]):
            kinds.append("query_exactly_at_boundary")
        if after and any(t > after[0] for t, c in events if c == row["query"]):
            kinds.append("query_only_after_boundary")
    return kinds


REQUIRED_SHAPES = (
    "no_events",
    "unbounded",
    "zero_length_window",
    "inverted_window",
    "unbounded_carrying_the_sentinel_duration",
    "query_exactly_at_prediction_time",
    "fractional_horizon_decides_the_answer",
    "bounded",
    "bound_never_occurs",
    "bound_only_before_window",
    "bound_repeats_after",
    "bound_at_prediction_time",
    "bound_is_query",
    "query_exactly_at_boundary",
    "query_only_after_boundary",
)


def test_the_random_sweep_actually_reaches_every_degenerate_shape():
    """A differential sweep is worth its runtime only if it reaches the traps.

    Guards the guard: if a future edit to the generator stops producing (say) simultaneous
    query-and-boundary events, the comparison above would keep passing while silently no
    longer testing the `<` vs `<=` flip.  Each shape must appear several times, not once.
    """
    counts: dict[str, int] = {}
    for seed in SEEDS:
        for kind in _shapes_in(*_random_case(seed)):
            counts[kind] = counts.get(kind, 0) + 1

    thin = {k: counts.get(k, 0) for k in REQUIRED_SHAPES if counts.get(k, 0) < 3}
    assert not thin, f"the sweep barely reaches {thin} across {len(SEEDS)} seeds -- it is toothless"


# ------------------------------------------------------------------------------------------
# Hand-built degenerate cases: one spec clause each, so a failure names the clause it broke.
# ------------------------------------------------------------------------------------------


def _one(query, duration_days, bound_event, prediction_time=EPOCH, subject_id=1):
    return [
        {
            "_ctx_id": 0,
            "_position": 0,
            "subject_id": subject_id,
            "prediction_time": prediction_time,
            "query": query,
            "duration_days": duration_days,
            "bound_event": bound_event,
        }
    ]


def _answer(index_rows, event_rows):
    return list(label_with_event_bounds(_index_df(index_rows), _events_df(event_rows))["answers"][0])


def _plain_answer(index_rows, event_rows):
    """Same, through `label_binary_occurrence` -- which takes no `bound_event` column."""
    out = label_binary_occurrence(_index_df(index_rows).drop("bound_event"), _events_df(event_rows))
    return list(out["answers"][0])


def test_the_plain_labeller_honours_a_fractional_day_horizon():
    """A sub-day horizon must not be floored to whole days.

    `durations` are float32 DAYS and the sampler draws values like 0.25 and 0.5, so a `days=`
    conversion that truncated to an integer would turn every sub-day horizon into a zero-length
    window -- every such query silently answering False, with no crash and no shape change.
    The randomised sweep does reach this, but on one seed out of twelve, which is too thin a
    thread for a whole class of query; this pins it outright.
    """
    six_hours = [(1, EPOCH + timedelta(hours=6), "A")]
    # 6h is inside a 0.5-day (12h) window, but outside the 0-day window truncation produces.
    assert _plain_answer(_one("A", 0.5, None), six_hours) == [True], (
        "a 0.5-day horizon was truncated to 0 days -- sub-day query windows are dead"
    )
    # A fractional horizon ABOVE one day, so the failure is not merely "zero vs non-zero":
    # truncating 1.5 -> 1 day would drop an event at 1.25 days.
    day_and_a_quarter = [(1, EPOCH + timedelta(days=1, hours=6), "A")]
    assert _plain_answer(_one("A", 1.5, None), day_and_a_quarter) == [True], (
        "a 1.5-day horizon was truncated to 1 day -- the fractional part is being discarded"
    )
    # ...and the horizon really is being read, rather than ignored altogether: at exactly
    # 0.25 days the 6h event lands ON the horizon instant, which the open window excludes.
    assert _plain_answer(_one("A", 0.25, None), six_hours) == [False]

    for rows, events in (
        (_one("A", 0.5, None), six_hours),
        (_one("A", 1.5, None), day_and_a_quarter),
        (_one("A", 0.25, None), six_hours),
    ):
        _assert_plain_matches_oracle(rows, events, context="fractional horizon")


def test_s3_boundary_is_the_first_occurrence_after_not_the_last():
    """Two boundaries ahead: the query sits between them.  First-occurrence => False."""
    events = [
        (1, EPOCH + timedelta(days=1), "DISCHARGE"),
        (1, EPOCH + timedelta(days=2), "A"),
        (1, EPOCH + timedelta(days=3), "DISCHARGE"),
    ]
    rows = _one("A", -1.0, "DISCHARGE")
    assert _answer(rows, events) == [False], "the LAST boundary was used, not the first"
    _assert_matches_oracle(rows, events, context="S3 first-vs-last")


def test_s3_boundary_before_the_prediction_time_is_invisible():
    """A boundary that already happened must not close the window retroactively (S3+S5)."""
    events = [
        (1, EPOCH - timedelta(days=5), "DISCHARGE"),
        (1, EPOCH + timedelta(days=40), "A"),
    ]
    rows = _one("A", -1.0, "DISCHARGE")
    # No boundary ahead => S5's open window => the far-future A still counts.
    assert _answer(rows, events) == [True], "a past boundary closed the window"
    _assert_matches_oracle(rows, events, context="S3 past boundary")


def test_s4_occurrence_exactly_at_the_boundary_instant_is_outside():
    """The classic silent flip: `<` vs `<=` at the bound."""
    at_bound = [
        (1, EPOCH + timedelta(days=2), "A"),
        (1, EPOCH + timedelta(days=2), "DISCHARGE"),
    ]
    just_inside = [
        (1, EPOCH + timedelta(days=2) - timedelta(microseconds=1), "A"),
        (1, EPOCH + timedelta(days=2), "DISCHARGE"),
    ]
    rows = _one("A", -1.0, "DISCHARGE")
    assert _answer(rows, at_bound) == [False], "an occurrence AT the boundary was counted"
    assert _answer(rows, just_inside) == [True], "1us inside the boundary was excluded"
    _assert_matches_oracle(rows, at_bound, context="S4 at bound")
    _assert_matches_oracle(rows, just_inside, context="S4 1us inside")


def test_lower_bound_is_strict_for_both_query_and_boundary():
    """An event AT the prediction time is neither an answer nor a boundary."""
    events = [
        (1, EPOCH, "A"),
        (1, EPOCH, "DISCHARGE"),
        (1, EPOCH + timedelta(days=3), "A"),
    ]
    # The A at the prediction instant does not answer; the DISCHARGE at it does not bound,
    # so S5 opens the window and the day-3 A does answer.
    rows = _one("A", -1.0, "DISCHARGE")
    assert _answer(rows, events) == [True], (
        "a boundary AT the prediction instant closed the window, or the A at the prediction "
        "instant was counted -- the lower bound is strict for both"
    )
    _assert_matches_oracle(rows, events, context="strict lower bound")


def test_s2_the_boundary_not_the_sentinel_duration_defines_the_window():
    """The feature must be live, not merely wired.

    `durations` carries the -1.0 sentinel for a bounded query, so an implementation that
    quietly fell back to the horizon would label every bounded query False and still produce
    perfectly-shaped output.  Here the only thing that can make the answer True is the
    boundary at day 100 genuinely widening the window past the sentinel.
    """
    events = [
        (1, EPOCH + timedelta(days=40), "A"),
        (1, EPOCH + timedelta(days=100), "DISCHARGE"),
    ]
    assert _answer(_one("A", -1.0, "DISCHARGE"), events) == [True], (
        "bounded query fell back to the -1.0 sentinel horizon; the boundary is not driving "
        "the window and the feature is dead"
    )
    # And the same query without the boundary is False, so True above is not trivially free.
    assert _answer(_one("A", -1.0, None), events) == [False]


def test_a_boundary_belonging_to_another_subject_does_not_close_the_window():
    """The boundary lookup is per subject; a shared code must not leak across the join key."""
    events = [
        (1, EPOCH + timedelta(days=5), "A"),
        (2, EPOCH + timedelta(days=1), "DISCHARGE"),  # subject 2's discharge, not subject 1's
    ]
    rows = _one("A", -1.0, "DISCHARGE", subject_id=1)
    # Subject 1 has no DISCHARGE, so S5 opens the window and the day-5 A answers True.  If the
    # join lost `subject_id`, subject 2's day-1 DISCHARGE would close it and the answer flips.
    assert _answer(rows, events) == [True], "another subject's event closed this subject's window"
    _assert_matches_oracle(rows, events, context="cross-subject boundary")


def test_s5_missing_boundary_opens_the_window_to_the_end_of_record():
    """Documented (and trap-prone) upstream semantics: no boundary => 'does it ever recur'."""
    events = [(1, EPOCH + timedelta(days=900), "A")]
    rows = _one("A", -1.0, "MEDS_DEATH")
    assert _answer(rows, events) == [True], "a never-occurring boundary must NOT close the window"
    _assert_matches_oracle(rows, events, context="S5 open window")


def test_s6_subject_with_no_events_is_false_for_every_query_form():
    events = [(2, EPOCH + timedelta(days=1), "A")]  # a different subject entirely
    assert _answer(_one("A", -1.0, "DISCHARGE"), events) == [False]
    assert _answer(_one("A", 30.0, None), events) == [False]


def test_s6_zero_length_and_inverted_windows_are_always_false():
    events = [(1, EPOCH, "A"), (1, EPOCH + timedelta(days=1), "A")]
    assert _answer(_one("A", 0.0, None), events) == [False], "zero-length window answered True"
    assert _answer(_one("A", -3.0, None), events) == [False], "inverted window answered True"
    _assert_matches_oracle(_one("A", 0.0, None), events, context="zero window")
    _assert_matches_oracle(_one("A", -3.0, None), events, context="inverted window")


def test_s6_bound_event_equal_to_the_query_is_always_false():
    """Self-bounded queries are degenerate by construction; they must not label True."""
    for events in (
        [(1, EPOCH + timedelta(days=1), "A")],
        [(1, EPOCH + timedelta(days=1), "A"), (1, EPOCH + timedelta(days=2), "A")],
        [(1, EPOCH - timedelta(days=1), "A")],
        [],
    ):
        rows = _one("A", -1.0, "A")
        assert _answer(rows, events) == [False], f"self-bounded query True for {events}"
        _assert_matches_oracle(rows, events or [(2, EPOCH, "Z")], context="self-bound")


def test_s1_null_bound_agrees_with_label_binary_occurrence_query_for_query():
    """The spec's own words: a null `bound_event` "behaves exactly as label_binary_occurrence".

    An upper-bound convention that drifted between the two labellers would silently relabel
    every ordinary query the moment `eventbound_fraction` was turned on -- and nothing else
    compares the two paths.
    """
    for seed in SEEDS:
        index_rows, event_rows = _random_case(seed)
        unbounded = [dict(r) for r in index_rows if r["bound_event"] is None]
        if not unbounded:
            continue
        # Both labellers group in ascending `_ctx_id` order, so the rows line up directly.
        events = _events_df(event_rows)
        bounded_out = label_with_event_bounds(_index_df(unbounded), events)
        plain_out = label_binary_occurrence(_index_df(unbounded).drop("bound_event"), events)
        assert bounded_out["queries"].to_list() == plain_out["queries"].to_list(), (
            f"seed {seed}: the two labellers did not even emit the same contexts"
        )
        assert bounded_out["answers"].to_list() == plain_out["answers"].to_list(), (
            f"seed {seed}: the null-bound path disagrees with label_binary_occurrence"
        )


def _measure_upper_bound_is_closed():
    """Ask the CODE, not the docs, whether an event on the horizon instant counts.

    Returns True if the window is closed at the top (`<=`), False if open (`<`).  Also checks
    the two labellers answer alike, since a drift between them is its own defect.
    """
    events = [(1, EPOCH + timedelta(days=2), "A")]
    rows = _one("A", 2.0, None)  # the event lands exactly on prediction_time + duration
    bounded = _answer(rows, events)
    plain = list(
        label_binary_occurrence(_index_df(rows).drop("bound_event"), _events_df(events))["answers"][0]
    )
    assert bounded == plain, (
        f"the two labellers disagree on the upper edge of the same window: "
        f"label_with_event_bounds says {bounded}, label_binary_occurrence says {plain}"
    )
    return bounded == [True]


# The four sources that were reconciled onto the open reading.  Each entry is
# (name, docstring, pattern) where the pattern's single group captures the closing bracket of
# the labelling window's interval notation.  Deliberately NOT the full prose: this must go red
# on a reverted bracket, not on an unrelated reword.
_DOC_SOURCES = (
    (
        "QuerySeqSchema.answers (src/every_query/data/schema.py)",
        inspect.getdoc(QuerySeqSchema),
        r"\(prediction_time,\s*prediction_time \+ durations\[j\]\s*([)\]])",
    ),
    (
        "label_binary_occurrence prose",
        inspect.getdoc(label_binary_occurrence),
        r"\(prediction_time,\s*prediction_time \+ duration_days\s*([)\]])",
    ),
    (
        "label_binary_occurrence worked example",
        inspect.getdoc(label_binary_occurrence),
        r"\(2,\s*12\s*([)\]])",
    ),
    (
        "label_with_event_bounds prose",
        inspect.getdoc(label_with_event_bounds),
        r"\(prediction_time,\s*prediction_time \+ duration_days\s*([)\]])",
    ),
)


def test_the_documented_upper_bound_matches_the_implemented_one():
    """The docs must describe the window the code actually labels.

    This is the whole point of the reconciliation.  No label is wrong today -- both labellers
    compare `<` and always have -- so nothing about this is a live bug, and a test that merely
    pinned the behaviour would have caught nothing.  The failure mode is entirely forward
    looking: `QuerySeqSchema` is the document a downstream consumer reads to reimplement or
    "fix" this labelling, and it used to promise a CLOSED upper bound.  Anyone who believed it
    would flip the answer for every event landing exactly on the horizon instant, silently, on
    a code path this repo's tests never run.

    So the assertion is a consistency one, not a constant one: measure what the code does, then
    require every source to say that.  Change the behaviour deliberately and this goes red until
    the docs follow -- which is the intended workflow, not an obstacle.
    """
    closed = _measure_upper_bound_is_closed()
    expected = "]" if closed else ")"

    wrong = {}
    for name, doc, pattern in _DOC_SOURCES:
        assert doc, f"{name}: docstring missing (running under python -OO?)"
        found = re.findall(pattern, doc)
        # An empty match list would let this pass vacuously if the notation were simply
        # deleted, so absence is a failure, not a skip.
        assert found, f"{name}: no labelling-window interval notation found -- pattern {pattern!r}"
        bad = [b for b in found if b != expected]
        if bad:
            wrong[name] = bad

    assert not wrong, (
        f"the code labels the window {'CLOSED' if closed else 'OPEN'} at the top (an event "
        f"exactly on prediction_time + duration is {'INSIDE' if closed else 'OUTSIDE'}), so "
        f"every source must write the interval with {expected!r}, but these do not: {wrong}.\n"
        "Either the docs drifted from the code, or the code changed and the docs were not "
        "updated with it.  Reconcile them; do not delete this test."
    )


def test_an_event_on_the_horizon_instant_is_outside_the_window():
    """Pin today's convention outright, so the consistency test above cannot drift quietly.

    `test_the_documented_upper_bound_matches_the_implemented_one` compares docs to code, so it
    would stay green if BOTH flipped together.  This one records which reading was actually
    chosen, and is the test a deliberate change is supposed to have to edit.
    """
    events = [(1, EPOCH + timedelta(days=2), "A")]
    on_horizon = _one("A", 2.0, None)
    assert _answer(on_horizon, events) == [False], (
        "an event landing exactly on prediction_time + duration_days was counted as inside; "
        "the window is documented open at the top"
    )
    # ...and 1us earlier it genuinely is inside, so the False above is an edge effect rather
    # than the query simply never matching.
    just_inside = _one("A", 2.0, None, prediction_time=EPOCH + timedelta(microseconds=1))
    assert _answer(just_inside, events) == [True], "1us inside the horizon was excluded too"
    _assert_matches_oracle(on_horizon, events, context="horizon instant")


# ------------------------------------------------------------------------------------------
# `assign_event_bounds`: the draw that decides WHICH queries become event-bounded.
# ------------------------------------------------------------------------------------------
#
# Phase-1 verification found this function guarded by nothing but its own module doctest:
# dropping the sentinel duration, and inverting the bounded fraction, both survived the whole
# of `tests/test_event_bounds_oracle.py` and `tests/test_event_bounded.py` (replayed and
# confirmed -- 47 passed under each).  `test_bound_draw_is_deterministic` in the other file is
# structurally blind to the inversion: it compares two calls that any mutation perturbs
# identically, so it asserts reproducibility and nothing about correctness.
#
# Both mutations are silent rather than loud.  Drop the sentinel and every bounded query
# carries its ORIGINAL horizon into the labeller instead of -1.0; because the labeller
# dispatches on the bound column, the labels stay right and only the emitted `durations` are
# wrong -- so the model trains on a horizon number for a query whose window is not a horizon at
# all.  Invert the fraction and `eventbound_fraction: 0.1` silently produces 90% event-bounded
# queries: the run converges and reports on a query diet nobody chose.


def _bounds_frame(n):
    """`n` query rows, each with a DISTINCT horizon.

    Distinct on purpose: with a constant horizon, an implementation that overwrote every
    duration with the same value -- or shuffled them -- would be indistinguishable from one
    that correctly left the unbounded rows alone.
    """
    return pl.DataFrame(
        {
            "query": [f"Q{i}" for i in range(n)],
            "duration_days": [float(i + 1) for i in range(n)],
        }
    ).with_columns(pl.col("duration_days").cast(pl.Float32))


def test_the_sentinel_lands_on_exactly_the_bounded_rows():
    """Bounded rows get the sentinel; unbounded rows keep the horizon they came in with.

    Goes red if the sentinel is dropped (bounded rows keep a horizon), if it is written to the
    wrong side of the `when/otherwise` (unbounded rows get -1.0), or if the durations are
    rebuilt in a way that loses the row-to-horizon correspondence.
    """
    n = 400
    idx = _bounds_frame(n)
    out = assign_event_bounds(idx, ["X", "Y"], 0.5, np.random.default_rng(0))

    assert out.height == n, "rows were added or dropped"
    assert out["query"].to_list() == idx["query"].to_list(), "the query column was disturbed"

    bounds = out["bound_event"].to_list()
    durations = out["duration_days"].to_list()
    originals = idx["duration_days"].to_list()
    # A mixed draw is the only one that can tell the two sides apart: all-bound and all-null
    # both survive a sentinel written to the wrong branch.
    assert any(b is not None for b in bounds), "no row was bounded -- this proves nothing"
    assert any(b is None for b in bounds), "every row was bounded -- this proves nothing"

    for i, (bound, got, want) in enumerate(zip(bounds, durations, originals, strict=True)):
        if bound is None:
            assert got == pytest.approx(want), (
                f"row {i} is unbounded but its horizon changed {want} -> {got}; an unbounded "
                "query must keep the duration it was sampled with"
            )
        else:
            assert got == pytest.approx(EVENT_BOUND_DURATION_SENTINEL), (
                f"row {i} is bounded by {bound!r} but carries horizon {got} instead of the "
                f"{EVENT_BOUND_DURATION_SENTINEL} sentinel; its window is defined by the "
                "boundary event, so a horizon here is a number that means nothing"
            )


# 0.5 is deliberately absent: it is the one fraction an inversion maps to itself, so a suite
# that only ever tested 0.5 would be blind to exactly the defect this is here to catch.
@pytest.mark.parametrize("fraction", [0.0, 0.1, 0.25, 0.75, 0.9, 1.0])
def test_the_bounded_fraction_is_the_one_that_was_asked_for(fraction):
    """The realised rate must track `eventbound_fraction`, not merely vary with it.

    An inverted comparison yields `1 - fraction`, which is off by at least 0.5 for every
    asymmetric value here.  At n=4000 the binomial spread is ~0.008, so the 0.03 tolerance is
    ~4 sigma of slack while still far tighter than the smallest error it must catch.
    """
    n = 4000
    out = assign_event_bounds(_bounds_frame(n), ["X"], fraction, np.random.default_rng(7))
    observed = sum(b is not None for b in out["bound_event"].to_list()) / n
    inverted = abs(observed - (1.0 - fraction)) < 0.03
    assert observed == pytest.approx(fraction, abs=0.03), (
        f"asked for {fraction:.0%} event-bounded queries, got {observed:.1%}"
        + (f" -- which is 1 - {fraction}, the signature of an inverted comparison" if inverted else "")
    )


def test_the_draw_is_per_query_not_per_sequence():
    """The docstring's headline contract: a single sequence mixes both kinds of ask.

    "Draws i.i.d. per query, not per sequence, so a single sequence mixes time- and
    event-bounded asks -- which is the point: the model has to read the slot to know which kind
    it is being given."  One draw broadcast over the frame would make every slot agree, and the
    model could then infer the kind once per sequence instead of per query.
    """
    out = assign_event_bounds(_bounds_frame(400), ["X"], 0.5, np.random.default_rng(3))
    flags = [b is not None for b in out["bound_event"].to_list()]
    assert 0 < sum(flags) < len(flags), "the whole frame shared one draw -- it is per sequence"
    # Independent fair draws switch on ~half of adjacent pairs; any per-block scheme (one draw
    # broadcast, or runs) collapses this toward zero.
    switches = sum(a != b for a, b in itertools.pairwise(flags))
    assert switches > 0.3 * len(flags), (
        f"only {switches} of {len(flags) - 1} adjacent slots differ; the draw is correlated "
        "along the frame rather than i.i.d. per query"
    )


def test_bounds_are_drawn_from_across_the_whole_pool():
    """Every supplied boundary code must actually be reachable, not just the first one.

    The module doctest only asserts the drawn codes are a SUBSET of the pool, which a constant
    "always pick pool[0]" satisfies -- and that would quietly train a whole run on a single
    boundary code while the config advertised several.
    """
    pool = ["W", "X", "Y", "Z"]
    n = 600
    out = assign_event_bounds(_bounds_frame(n), pool, 1.0, np.random.default_rng(11))
    drawn = out["bound_event"].to_list()

    assert set(drawn) == set(pool), (
        f"only {sorted(set(drawn))} were ever drawn from the pool {pool}; the rest of the pool is unreachable"
    )
    # Roughly uniform, not merely present once: a 597/1/1/1 split is still "all reachable".
    per_code = {c: drawn.count(c) for c in pool}
    assert min(per_code.values()) > 0.6 * n / len(pool), (
        f"boundary codes are drawn very unevenly: {per_code} (each should be near {n // len(pool)})"
    )
