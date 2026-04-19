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

import pyarrow as pa
from flexible_schema import Required
from meds import LabelSchema


class TaskQuerySchema(LabelSchema):
    """An EveryQuery task-query row: a MEDS prediction-time label plus the query that defines it.

    Each row is a single ``(subject_id, prediction_time, query, duration_days)`` tuple with
    optional label columns inherited from ``LabelSchema``.  When the ground-truth label is
    present it lives on the inherited ``boolean_value`` column — *"did ``query`` occur for
    ``subject_id`` within ``duration_days`` of ``prediction_time``?"* — so the schema serves
    both inference input (no label) and evaluation input (label filled in) without a branch.

    Attributes:
        query: The MEDS code the query asks about.  Named ``query`` rather than ``code`` to
            match the column name already used throughout the sampler / dataset / batch
            layer (``EveryQueryBatch.code`` is the distinct event-sequence-token tensor).
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
        >>> set(f.name for f in aligned.schema) >= {
        ...     "subject_id", "prediction_time", "query", "duration_days", "boolean_value"
        ... }
        True

        Fractional durations are supported:

        >>> data = pa.Table.from_pylist([
        ...     {"subject_id": 1, "prediction_time": datetime(2023, 1, 1),
        ...      "query": "ICD//I10", "duration_days": 0.5},
        ... ])
        >>> _ = TaskQuerySchema.align(data)
    """

    query: Required(pa.large_string(), nullable=False)
    duration_days: Required(pa.float32(), nullable=False)
