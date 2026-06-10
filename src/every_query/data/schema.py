"""Cross-stage schema for EveryQuery task-query rows.

``TaskQuerySchema`` is the contract shared between ``generate_tasks/`` (producer) and the
future ``predict/`` + ``evaluate/`` stages (consumers).  Each row specifies: *"for
subject_id at prediction_time, will ``query`` occur within ``duration_days``?"*

Extends the MEDS ``LabelSchema`` — so it inherits ``subject_id``, ``prediction_time``, and
the optional label columns (``boolean_value``, ``integer_value``, etc.) — and adds the two
required query fields ``query`` and ``duration_days``.  Mirrors the pattern
``meds-evaluation``'s `PredictionSchema
<https://github.com/kamilest/meds-evaluation/blob/main/src/meds_evaluation/schema.py>`_ uses.

The ``query`` column holds the MEDS code the query asks about.  The field is named
``query`` (not ``code``) to match the existing column name used by ``sample_tasks``
output, ``EveryQueryPytorchDataset``, and the ``EveryQueryBatch.query`` tensor —
renaming to ``code`` would (a) collide with the inherited ``EveryQueryBatch.code``
sequence-token tensor, which is a different thing semantically, and (b) churn every
downstream consumer for no functional win.

Initial scope (per #80) is intentionally narrow: a flat single code + a continuous (float)
duration.  Extensions — compound ANY/ALL queries, structured task payloads — are out of
scope for the initial schema and will be added as the inference / evaluation pipelines
evolve.
"""

import polars as pl
import pyarrow as pa
from flexible_schema import Optional, Required
from meds import LabelSchema


class TaskQuerySchema(LabelSchema):
    """An EveryQuery task-query row: a MEDS prediction-time label plus the query that defines it.

    Each row is a single ``(subject_id, prediction_time, query, duration_days)`` tuple with
    optional label columns inherited from ``LabelSchema``.  When the ground-truth label is
    present it lives on the inherited ``boolean_value`` column — *"did ``query`` occur for
    ``subject_id`` within ``duration_days`` of ``prediction_time``?"* — so the schema serves
    both inference input (no label) and evaluation input (label filled in) without a branch.

    Attributes:
        query: The MEDS code the query asks about.  Stored as ``pa.large_string``
            (polars' ``Utf8`` serializes to ``large_string`` when a DataFrame is
            converted to arrow, so this matches producer output natively and
            ``TaskQuerySchema.align()`` works without type coercion; also matches
            MEDS's own ``DataSchema.code`` convention which uses ``large_string``
            for the same 2 GB-offset reason).  Named ``query`` rather than ``code``
            to match the column name already used throughout the sampler / dataset /
            batch layer (``EveryQueryBatch.code`` is the distinct event-sequence-token
            tensor).
        duration_days: The horizon, in days (continuous — ``float32``) within which the
            ``query`` code must occur for the query to be positive.  Allowing fractional
            days keeps the contract flexible for future finer-grained horizons.

    Examples:
        A row with just the query (inference input) validates:

        >>> from datetime import datetime
        >>> import pyarrow as pa
        >>> data = pa.Table.from_pylist([
        ...     {"subject_id": 1, "prediction_time": datetime(2023, 1, 1),
        ...      "query": "ICD//I10", "duration_days": 30.0},
        ... ])
        >>> aligned = TaskQuerySchema.align(data)
        >>> [f.name for f in aligned.schema]
        ['subject_id', 'prediction_time', 'query', 'duration_days']

        A row with the inherited ``boolean_value`` label filled in (evaluation input) also
        validates:

        >>> data = pa.Table.from_pylist([
        ...     {"subject_id": 1, "prediction_time": datetime(2023, 1, 1),
        ...      "query": "ICD//I10", "duration_days": 30.0, "boolean_value": True},
        ...     {"subject_id": 2, "prediction_time": datetime(2023, 1, 1),
        ...      "query": "ICD//I10", "duration_days": 30.0, "boolean_value": False},
        ... ])
        >>> aligned = TaskQuerySchema.align(data)
        >>> [f.name for f in aligned.schema]
        ['subject_id', 'prediction_time', 'boolean_value', 'query', 'duration_days']

        Fractional durations are supported:

        >>> data = pa.Table.from_pylist([
        ...     {"subject_id": 1, "prediction_time": datetime(2023, 1, 1),
        ...      "query": "ICD//I10", "duration_days": 0.5},
        ... ])
        >>> _ = TaskQuerySchema.align(data)
    """

    query: Required(pa.large_string(), nullable=False)
    duration_days: Required(pa.float32(), nullable=False)
    # Override ``boolean_value`` from ``LabelSchema`` (which declares it
    # ``Optional(bool, nullable=NONE)``) to allow nulls — the EveryQuery task-label
    # convention is to use null as the "censored" sentinel (closes #122).  The column
    # stays optional at the schema level so inference-only inputs (no ground truth)
    # continue to validate.
    boolean_value: Optional(pa.bool_(), nullable=True)


class QuerySeqSchema(LabelSchema):
    """A conditional query-sequence row: one patient context plus an ordered list of queries.

    Each row is a ``(subject_id, prediction_time)`` context with three aligned list columns:

    Attributes:
        queries: Ordered list of MEDS code strings.  Element 0 is always the censor sentinel
            ``__CENSOR__`` (see ``every_query.data.seq_dataset.CENSOR_QUERY_CODE``), asking
            "will any data be present after ``prediction_time + durations[0]``?".
        durations: Per-query horizons in days (``float32``), aligned with ``queries``.
        answers: Per-query nullable booleans aligned with ``queries``.  ``null`` means the
            answer is unobserved (the window extends past the subject's last recorded event);
            element 0 is never null.

    Examples:
        >>> from datetime import datetime
        >>> import pyarrow as pa
        >>> data = pa.Table.from_pylist([
        ...     {"subject_id": 1, "prediction_time": datetime(2023, 1, 1),
        ...      "queries": ["__CENSOR__", "ICD//I10"], "durations": [30.0, 7.0],
        ...      "answers": [True, None]},
        ... ])
        >>> aligned = QuerySeqSchema.align(data)
        >>> [f.name for f in aligned.schema]
        ['subject_id', 'prediction_time', 'queries', 'durations', 'answers']
    """

    queries: Required(pa.large_list(pa.large_string()), nullable=False)
    durations: Required(pa.large_list(pa.float32()), nullable=False)
    answers: Required(pa.large_list(pa.bool_()), nullable=False)


def empty_task_query_df() -> pl.DataFrame:
    """Build an empty polars DataFrame shaped like ``TaskQuerySchema``'s required columns plus the inherited
    ``boolean_value`` (the collapsed label column).

    Only the required columns + ``boolean_value`` are included — not every
    ``LabelSchema`` optional column — because (a) that's what the sampler's empty-input
    fast path needs, and (b) a polars-arrow round-trip coerces ``pa.string`` →
    ``pa.large_string`` on the inherited ``categorical_value`` column, so a schema-
    complete empty frame would fail ``TaskQuerySchema.validate`` after the round-trip
    unless we bypassed polars entirely.  Keeping the shape focused on what downstream
    writers actually emit avoids that type-drift landmine.

    Polars dtypes are derived from ``TaskQuerySchema``'s arrow types at call time
    (``pl.from_arrow`` on an empty arrow table) rather than hardcoded — so any future
    change to the PyArrow type declarations flows through automatically instead of
    drifting silently.  ``pa.large_string`` coerces to ``pl.Utf8`` in polars' type
    system, which is the correct mapping here.

    Callers use this at the empty-input fast path (e.g., ``evaluate_index_df`` when no
    tasks were sampled) so the produced parquet still aligns to the schema via
    ``TaskQuerySchema.align`` at the write boundary.

    Examples:
        >>> df = empty_task_query_df()
        >>> df.height
        0
        >>> for name, dtype in df.schema.items():
        ...     print(f"{name}: {dtype}")
        subject_id: Int64
        prediction_time: Datetime(time_unit='us', time_zone=None)
        query: String
        duration_days: Float32
        boolean_value: Boolean
    """
    field_names = [
        TaskQuerySchema.subject_id_name,
        TaskQuerySchema.prediction_time_name,
        TaskQuerySchema.query_name,
        TaskQuerySchema.duration_days_name,
        TaskQuerySchema.boolean_value_name,
    ]
    pa_fields = [pa.field(name, getattr(TaskQuerySchema, f"{name}_dtype")) for name in field_names]
    return pl.from_arrow(pa.schema(pa_fields).empty_table())
