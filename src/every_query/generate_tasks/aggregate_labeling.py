"""Exact labelers for aggregate queries, and the sampler that draws them.

An aggregate query asks about a *combination* of codes rather than one code.  The grammar lives
in :mod:`every_query.data.query_vocab` (it has to, because vocabulary validation needs it); this
module answers the questions it expresses, and draws them.

Every bound is strict/open, matching the repo's convention that an occurrence exactly at a
window edge is outside it:

- ``ANY(c1|c2|...)``          — some component occurs in ``(t, t+d)``;
- ``ALL(c1|c2|...)``          — every component occurs;
- ``GE2(c1|c2|c3)``           — at least two of the three occur;
- ``XOR(c1|c2)``              — exactly one occurs;
- ``CO(c1&c2)``               — some single timestamp in ``(t, t+d)`` carries **both** codes;
- ``WITHIN(c1&c2|W=W)``       — occurrences of each, both in window, at most ``W`` days apart
                                in either direction;
- ``SEQ(c1>c2|gap=Gs:Ge)``    — ordered: ``t < τ1 < τ2 < t+d`` with ``τ2 - τ1`` in ``[Gs, Ge)``.

The four set operators reduce to per-component first-occurrence tests, so they cost one asof
join over the exploded component frame.  The three temporal operators do not, and each gets its
own exact treatment: ``CO`` builds a derived co-timed event stream per distinct pair, ``WITHIN``
checks the nearest occurrence in *both* directions from each in-window ``τ1``, and ``SEQ``
shifts the asof key by ``Gs`` so the first match is already past the lower bound.

A note on what upstream found, since it bears on whether native aggregate heads are worth
training at all: **composing** single-query predictions under an independence assumption beat
the native aggregate head overall (0.7568 vs 0.7396).  The native model won specifically on
``CO`` and ``WITHIN`` — the two operators whose truth is not a function of the components'
marginal occurrence — which is where a native head can express something composition cannot.
"""

import logging

import numpy as np
import polars as pl
from meds import DataSchema

from every_query.data.query_vocab import (
    OP_ALL,
    OP_ANY,
    OP_ATOM,
    OP_CO,
    OP_GE2,
    OP_SEQ,
    OP_WITHIN,
    OP_XOR,
    has_reserved_chars,
    parse_query,
)
from every_query.data.schema import TaskQuerySchema

logger = logging.getLogger(__name__)


def _first_occurrence_in_window(
    left: pl.DataFrame, events_df: pl.DataFrame, code_col: str, out_col: str
) -> pl.DataFrame:
    """Forward-asof each row to the first event matching ``code_col`` strictly after ``pt``."""
    sid = TaskQuerySchema.subject_id_name
    pt = TaskQuerySchema.prediction_time_name
    time_col = DataSchema.time_name

    right = (
        events_df.rename({DataSchema.code_name: code_col})
        .select(sid, code_col, time_col)
        .sort(sid, code_col, time_col)
    )
    return (
        left.with_columns((pl.col(pt) + pl.duration(microseconds=1)).alias("_pts"))
        .sort(sid, code_col, "_pts")
        .join_asof(right, by=[sid, code_col], left_on="_pts", right_on=time_col, strategy="forward")
        .rename({time_col: out_col})
    )


def _co_timed_stream(events_df: pl.DataFrame, pairs: list[tuple[str, str]]) -> pl.DataFrame:
    """A synthetic event stream whose 'code' is a pair key, emitted at co-timed instants only.

    Turning "both codes at the same timestamp" into a derived single code means the ordinary
    first-occurrence machinery answers ``CO`` with no special casing downstream.
    """
    sid = TaskQuerySchema.subject_id_name
    time_col = DataSchema.time_name
    code_col = DataSchema.code_name

    streams = []
    for c1, c2 in sorted(set(pairs)):
        e1 = events_df.filter(pl.col(code_col) == c1).select(sid, time_col)
        e2 = events_df.filter(pl.col(code_col) == c2).select(sid, time_col)
        both = e1.join(e2, on=[sid, time_col], how="semi").unique()
        streams.append(both.with_columns(pl.lit(pair_key(c1, c2)).alias(code_col)))

    if not streams:
        return events_df.head(0).select(sid, time_col, code_col)
    return pl.concat(streams).select(sid, time_col, code_col)


def pair_key(c1: str, c2: str) -> str:
    """Key for a co-timed pair's derived stream.

    Uses NUL, which cannot occur in a MEDS code, so the key can never collide with a real one.

    Examples:
        >>> pair_key("A", "B") == "A\\x00B"
        True
    """
    return f"{c1}\x00{c2}"


def label_aggregates(index_df: pl.DataFrame, events_df: pl.DataFrame) -> pl.DataFrame:
    """Label a frame that mixes atomic and aggregate queries.

    Atomic rows are answered by exactly the same first-occurrence rule
    :func:`~every_query.generate_tasks.sample_query_sequences.label_binary_occurrence` uses, so
    a frame with no aggregates in it comes back identical.

    Examples:
        >>> from datetime import datetime
        >>> events = pl.DataFrame({
        ...     "subject_id": [1] * 6,
        ...     "time": [datetime(2024, 1, 3), datetime(2024, 1, 3), datetime(2024, 1, 6),
        ...              datetime(2024, 1, 9), datetime(2024, 1, 9, 12), datetime(2024, 1, 20)],
        ...     "code": ["A", "B", "A", "B", "A", "C"],
        ... }).with_columns(pl.col("time").cast(pl.Datetime("us")))
        >>> idx = pl.DataFrame({
        ...     "_ctx_id": pl.Series([0] * 7, dtype=pl.UInt32),
        ...     "_position": list(range(7)),
        ...     "subject_id": [1] * 7,
        ...     "prediction_time": [datetime(2024, 1, 2)] * 7,
        ...     "query": ["CO(A&B)", "CO(A&C)", "WITHIN(A&B|W=1.0)", "WITHIN(B&C|W=2.0)",
        ...               "SEQ(A>B|gap=2.0:4.0)", "GE2(A|B|C)", "XOR(A|C)"],
        ...     "duration_days": pl.Series([10.0, 30.0, 10.0, 10.0, 10.0, 5.0, 5.0],
        ...                                dtype=pl.Float32),
        ... })
        >>> label_aggregates(idx, events).row(0, named=True)["answers"]
        [True, False, True, False, True, True, True]

        Reading those: ``CO(A&B)`` — Jan 3 carries both, so True.  ``CO(A&C)`` — never
        co-timed, False.  ``WITHIN(A&B|1d)`` — B at Jan 9 and A at Jan 9 12:00 are half a day
        apart, True.  ``WITHIN(B&C|2d)`` — C at Jan 20, nearest B at Jan 9, eleven days, False.
        ``SEQ(A>B|2:4)`` — A at Jan 6 then B at Jan 9 is a 3-day gap inside ``[2, 4)``, True.
        ``GE2(A|B|C)`` over ``(Jan 2, Jan 7)`` — A and B occur, C does not, so two of three,
        True.  ``XOR(A|C)`` over the same window — A only, True.
    """
    sid = TaskQuerySchema.subject_id_name
    pt = TaskQuerySchema.prediction_time_name
    q = TaskQuerySchema.query_name
    d = TaskQuerySchema.duration_days_name
    time_col = DataSchema.time_name
    code_col = DataSchema.code_name

    parsed = [parse_query(x) for x in index_df[q].to_list()]
    df = index_df.with_row_index("_row").with_columns(
        pl.Series("_op", [p.op for p in parsed], dtype=pl.Int8),
        pl.Series("_gap_hi", [p.gap_days or 0.0 for p in parsed], dtype=pl.Float64),
        pl.Series("_gap_lo", [p.gap_lo for p in parsed], dtype=pl.Float64),
    )
    df = df.with_columns((pl.col(pt) + pl.duration(days=pl.col(d))).alias("_wend"))
    ev = events_df.select(sid, time_col, code_col)

    # --- set operators: per-component first occurrence is sufficient -------------------------
    comp_rows = df.with_columns(pl.Series("_comps", [p.components for p in parsed])).explode("_comps")
    comp_hits = (
        _first_occurrence_in_window(
            comp_rows.select("_row", sid, pt, "_wend", pl.col("_comps").alias("_comp")),
            events_df,
            "_comp",
            "_t_comp",
        )
        .with_columns((pl.col("_t_comp").is_not_null() & (pl.col("_t_comp") < pl.col("_wend"))).alias("_hit"))
        .group_by("_row")
        .agg(
            pl.col("_hit").any().alias("_any_hit"),
            pl.col("_hit").all().alias("_all_hit"),
            pl.col("_hit").cast(pl.Int32).sum().alias("_n_hits"),
        )
    )

    ops = df["_op"].to_list()

    # --- CO: first occurrence in a derived co-timed stream ------------------------------------
    co_hits = None
    co_df = df.filter(pl.col("_op") == OP_CO)
    if co_df.height:
        co_pairs = [
            (p.components[0], p.components[1]) for p, o in zip(parsed, ops, strict=True) if o == OP_CO
        ]
        co_df = co_df.with_columns(
            pl.Series("_comp", [pair_key(a, b) for a, b in co_pairs]),
        )
        co_hits = _first_occurrence_in_window(
            co_df.select("_row", sid, pt, "_wend", "_comp"),
            _co_timed_stream(ev, co_pairs),
            "_comp",
            "_t_co",
        ).select(
            "_row",
            (pl.col("_t_co").is_not_null() & (pl.col("_t_co") < pl.col("_wend"))).alias("_co_hit"),
        )

    # --- WITHIN: nearest c2 in BOTH directions from each in-window c1 -------------------------
    wi_hits = None
    wi_df = df.filter(pl.col("_op") == OP_WITHIN)
    if wi_df.height:
        wi_pairs = [
            (p.components[0], p.components[1]) for p, o in zip(parsed, ops, strict=True) if o == OP_WITHIN
        ]
        wi_df = wi_df.with_columns(
            pl.Series("_c1", [a for a, _ in wi_pairs]), pl.Series("_c2", [b for _, b in wi_pairs])
        )
        occ1 = (
            wi_df.select("_row", sid, pt, "_wend", "_gap_hi", "_c1", "_c2")
            .join(ev.rename({code_col: "_c1"}), on=[sid, "_c1"], how="inner")
            .filter((pl.col(time_col) > pl.col(pt)) & (pl.col(time_col) < pl.col("_wend")))
            .rename({time_col: "_tau1"})
        )
        if occ1.height:
            right = ev.rename({code_col: "_c2"}).sort(sid, "_c2", time_col)
            base = occ1.sort(sid, "_c2", "_tau1")
            fwd = base.join_asof(
                right, by=[sid, "_c2"], left_on="_tau1", right_on=time_col, strategy="forward"
            ).rename({time_col: "_t_f"})
            bwd = base.join_asof(
                right, by=[sid, "_c2"], left_on="_tau1", right_on=time_col, strategy="backward"
            ).rename({time_col: "_t_b"})
            merged = fwd.with_row_index("_oi").join(bwd.with_row_index("_oi").select("_oi", "_t_b"), on="_oi")
            near = (
                pl.col("_t_f").is_not_null()
                & (pl.col("_t_f") < pl.col("_wend"))
                & ((pl.col("_t_f") - pl.col("_tau1")) < pl.duration(days=pl.col("_gap_hi")))
            ) | (
                pl.col("_t_b").is_not_null()
                & (pl.col("_t_b") > pl.col(pt))
                & ((pl.col("_tau1") - pl.col("_t_b")) < pl.duration(days=pl.col("_gap_hi")))
            )
            wi_hits = (
                merged.with_columns(near.alias("_v"))
                .group_by("_row")
                .agg(pl.col("_v").any().alias("_wi_hit"))
            )

    # --- SEQ: key-shifted forward asof so the first match already clears the lower bound ------
    seq_hits = None
    seq_df = df.filter(pl.col("_op") == OP_SEQ)
    if seq_df.height:
        sq_pairs = [
            (p.components[0], p.components[1]) for p, o in zip(parsed, ops, strict=True) if o == OP_SEQ
        ]
        seq_df = seq_df.with_columns(
            pl.Series("_c1", [a for a, _ in sq_pairs]), pl.Series("_c2", [b for _, b in sq_pairs])
        )
        occ1 = (
            seq_df.select("_row", sid, pt, "_wend", "_gap_hi", "_gap_lo", "_c1", "_c2")
            .join(ev.rename({code_col: "_c1"}), on=[sid, "_c1"], how="inner")
            .filter((pl.col(time_col) > pl.col(pt)) & (pl.col(time_col) < pl.col("_wend")))
            .rename({time_col: "_tau1"})
        )
        if occ1.height:
            # Shift by Gs, or by 1us when Gs is 0 so that tau2 == tau1 is excluded (SEQ is
            # strictly ordered; a co-timed pair is CO's job, not SEQ's).
            occ1 = occ1.with_columns(
                (
                    pl.col("_tau1")
                    + pl.duration(days=pl.col("_gap_lo"))
                    + pl.duration(microseconds=(pl.col("_gap_lo") == 0.0).cast(pl.Int64))
                ).alias("_key")
            )
            right = ev.rename({code_col: "_c2"}).sort(sid, "_c2", time_col)
            paired = (
                occ1.sort(sid, "_c2", "_key")
                .join_asof(right, by=[sid, "_c2"], left_on="_key", right_on=time_col, strategy="forward")
                .rename({time_col: "_tau2"})
            )
            valid = (
                pl.col("_tau2").is_not_null()
                & (pl.col("_tau2") < pl.col("_wend"))
                & ((pl.col("_tau2") - pl.col("_tau1")) < pl.duration(days=pl.col("_gap_hi")))
            )
            seq_hits = (
                paired.with_columns(valid.alias("_v"))
                .group_by("_row")
                .agg(pl.col("_v").any().alias("_sq_hit"))
            )

    labeled = df.join(comp_hits, on="_row", how="left")
    for hits, col in ((co_hits, "_co_hit"), (wi_hits, "_wi_hit"), (seq_hits, "_sq_hit")):
        labeled = (
            labeled.join(hits, on="_row", how="left")
            if hits is not None
            else labeled.with_columns(pl.lit(False).alias(col))
        )

    answer = (
        pl.when(pl.col("_op") == OP_ALL)
        .then(pl.col("_all_hit"))
        .when(pl.col("_op") == OP_SEQ)
        .then(pl.col("_sq_hit").fill_null(False))
        .when(pl.col("_op") == OP_CO)
        .then(pl.col("_co_hit").fill_null(False))
        .when(pl.col("_op") == OP_WITHIN)
        .then(pl.col("_wi_hit").fill_null(False))
        .when(pl.col("_op") == OP_GE2)
        .then(pl.col("_n_hits") >= 2)
        .when(pl.col("_op") == OP_XOR)
        .then(pl.col("_n_hits") == 1)
        # ATOM and ANY are the same test: did any (the only) component occur.
        .otherwise(pl.col("_any_hit"))
        .fill_null(False)
    )

    from every_query.generate_tasks.sample_query_sequences import CTX_ID_COL, POSITION_COL

    flat = labeled.with_columns(answer.alias("answer")).sort(CTX_ID_COL, POSITION_COL)
    return (
        flat.group_by(CTX_ID_COL, maintain_order=True)
        .agg(
            pl.col(sid).first(),
            pl.col(pt).first(),
            pl.col(q).alias("queries"),
            pl.col(d).alias("durations"),
            pl.col("answer").alias("answers"),
        )
        .drop(CTX_ID_COL)
    )


def has_aggregates(index_df: pl.DataFrame) -> bool:
    """True when any query in the frame is a composite expression rather than a bare code."""
    q = TaskQuerySchema.query_name
    if q not in index_df.columns or index_df.height == 0:
        return False
    return any(parse_query(x).op != OP_ATOM for x in index_df[q].to_list())


# ---------------------------------------------------------------------------
# Drawing aggregate queries
# ---------------------------------------------------------------------------

# The op set drawn from, and how many components each takes.
_DRAWABLE_OPS = (
    (OP_ANY, 2),
    (OP_ALL, 2),
    (OP_SEQ, 2),
    (OP_CO, 2),
    (OP_WITHIN, 2),
    (OP_GE2, 3),
    (OP_XOR, 2),
)

# Gap bounds drawn for the two temporal operators, in days.
_GAP_CHOICES_DAYS = (1.0, 3.0, 7.0, 30.0)


def eligible_components(codes: list[str]) -> list[str]:
    """Codes usable as aggregate components.

    Excludes anything carrying a character the grammar reserves — such a code could not be
    parsed back out of the query string it was written into.  Upstream shipped this as a live
    bugfix (5fd5acb) after ``&``-containing codes collided with the ``CO`` separator.

    Examples:
        >>> eligible_components(["LAB//A", "DRUG//B&C", "OK"])
        ['LAB//A', 'OK']
    """
    return [c for c in codes if not has_reserved_chars(c)]


def format_query(op: int, components: list[str], gap_lo: float = 0.0, gap_hi: float = 0.0) -> str:
    """Render an aggregate query back into its string form.

    Examples:
        >>> format_query(OP_ALL, ["A", "B"])
        'ALL(A|B)'
        >>> format_query(OP_CO, ["A", "B"])
        'CO(A&B)'
        >>> format_query(OP_WITHIN, ["A", "B"], gap_hi=2.0)
        'WITHIN(A&B|W=2.0)'
        >>> format_query(OP_SEQ, ["A", "B"], gap_hi=7.0)
        'SEQ(A>B|gap=7.0)'
        >>> format_query(OP_SEQ, ["A", "B"], gap_lo=1.0, gap_hi=7.0)
        'SEQ(A>B|gap=1.0:7.0)'

        Every rendering round-trips through the parser:

        >>> from every_query.data.query_vocab import parse_query
        >>> p = parse_query(format_query(OP_SEQ, ["A", "B"], 1.0, 7.0))
        >>> p.components, p.gap_lo, p.gap_days
        (['A', 'B'], 1.0, 7.0)
    """
    names = {OP_ANY: "ANY", OP_ALL: "ALL", OP_GE2: "GE2", OP_XOR: "XOR"}
    if op in names:
        return f"{names[op]}({'|'.join(components)})"
    if op == OP_CO:
        return f"CO({components[0]}&{components[1]})"
    if op == OP_WITHIN:
        return f"WITHIN({components[0]}&{components[1]}|W={gap_hi})"
    if op == OP_SEQ:
        body = f"{components[0]}>{components[1]}"
        rng = f"{gap_lo}:{gap_hi}" if gap_lo > 0 else f"{gap_hi}"
        return f"SEQ({body}|gap={rng})"
    raise ValueError(f"op {op} is not an aggregate operator")


def draw_aggregate_query(components_pool: list[str], rng: np.random.Generator) -> str:
    """Draw one aggregate query string uniformly over the op set."""
    op, k = _DRAWABLE_OPS[rng.integers(0, len(_DRAWABLE_OPS))]
    picks = rng.choice(len(components_pool), size=k, replace=False)
    comps = [components_pool[int(i)] for i in picks]

    gap_lo = gap_hi = 0.0
    if op in (OP_SEQ, OP_WITHIN):
        gap_hi = float(_GAP_CHOICES_DAYS[rng.integers(0, len(_GAP_CHOICES_DAYS))])
        if op == OP_SEQ and rng.random() < 0.5:
            # Half the SEQ draws get a non-zero lower bound, which is what makes a *range*
            # ("between 1 and 7 days later") expressible rather than only "within 7 days".
            gap_lo = float(rng.choice([g for g in _GAP_CHOICES_DAYS if g < gap_hi] or [0.0]))
    return format_query(op, comps, gap_lo, gap_hi)


def assign_aggregate_queries(
    index_df: pl.DataFrame,
    components_pool: list[str],
    aggregate_fraction: float,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """Replace that fraction of the per-query rows with aggregate expressions.

    Drawn i.i.d. per query, like event bounds, so a sequence mixes atomic and aggregate asks.
    ``rng`` is supplied by the caller so this can sit on its own seed axis and leave the
    code/duration draw — and its parity contract with ``sample_tasks`` — untouched.

    Examples:
        >>> import numpy as np
        >>> idx = pl.DataFrame({"query": ["A", "B", "C", "D"],
        ...                     "duration_days": [30.0] * 4}).with_columns(
        ...     pl.col("duration_days").cast(pl.Float32))
        >>> out = assign_aggregate_queries(idx, ["X", "Y", "Z"], 0.0, np.random.default_rng(0))
        >>> out["query"].to_list()
        ['A', 'B', 'C', 'D']

        At fraction 1.0 every row becomes an aggregate over the pool:

        >>> from every_query.data.query_vocab import is_aggregate
        >>> out = assign_aggregate_queries(idx, ["X", "Y", "Z"], 1.0, np.random.default_rng(0))
        >>> all(is_aggregate(x) for x in out["query"].to_list())
        True
    """
    if aggregate_fraction <= 0.0:
        return index_df
    if not 0.0 <= aggregate_fraction <= 1.0:
        raise ValueError(f"aggregate_fraction must be in [0, 1], got {aggregate_fraction}")

    pool = eligible_components(components_pool)
    if len(pool) < 3:
        raise ValueError(
            f"aggregate queries need at least 3 usable component codes, got {len(pool)} "
            f"(after excluding codes containing reserved characters)."
        )

    q = TaskQuerySchema.query_name
    n = index_df.height
    chosen = rng.random(n) < aggregate_fraction
    queries = index_df[q].to_list()
    out = [draw_aggregate_query(pool, rng) if chosen[i] else queries[i] for i in range(n)]
    return index_df.with_columns(pl.Series(q, out, dtype=pl.Utf8))
