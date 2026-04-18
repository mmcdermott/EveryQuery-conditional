"""Prediction-output schema.

``PredictionSchema`` extends ``TaskQuerySchema`` (the inference *input* schema defined in
``every_query.data.schema``) with the model's per-row predicted probability.  Mirrors
``meds-evaluation.PredictionSchema``'s ``predicted_boolean_probability`` column so downstream
tools that consume MEDS predictions (including meds-evaluation itself) work unmodified on
EveryQuery output.

Current scope matches ``TaskQuerySchema``: flat single code + continuous duration.
Embedding output (when ``EQ_predict --write-embeddings`` is on) uses a separate schema kept
out of this PR.
"""

import pyarrow as pa
from flexible_schema import Optional

from every_query.data.schema import TaskQuerySchema


class PredictionSchema(TaskQuerySchema):
    """EveryQuery prediction-output row.

    Inherits every column from :class:`every_query.data.schema.TaskQuerySchema` —
    ``subject_id``, ``prediction_time``, ``code``, ``duration_days``, plus the optional label
    columns.  Adds:

    Attributes:
        predicted_boolean_probability: Model-reported probability (``sigmoid(logit)``) that
            ``code`` occurs for ``subject_id`` within ``duration_days`` of ``prediction_time``.

    Examples:
        A prediction-only row (no ground truth) validates:

        >>> from datetime import datetime
        >>> import pyarrow as pa
        >>> data = pa.Table.from_pylist([
        ...     {"subject_id": 1, "prediction_time": datetime(2023, 1, 1),
        ...      "code": "ICD//I10", "duration_days": 30.0,
        ...      "predicted_boolean_probability": 0.73},
        ... ])
        >>> aligned = PredictionSchema.align(data)
        >>> [f.name for f in aligned.schema]
        ['subject_id', 'prediction_time', 'code', 'duration_days', 'predicted_boolean_probability']

        A row with both ground-truth label and prediction (eval-time shape) also validates:

        >>> data = pa.Table.from_pylist([
        ...     {"subject_id": 1, "prediction_time": datetime(2023, 1, 1),
        ...      "code": "ICD//I10", "duration_days": 30.0,
        ...      "boolean_value": True, "predicted_boolean_probability": 0.91},
        ... ])
        >>> set(f.name for f in PredictionSchema.align(data).schema) >= {
        ...     "subject_id", "prediction_time", "code", "duration_days",
        ...     "boolean_value", "predicted_boolean_probability",
        ... }
        True
    """

    predicted_boolean_probability: Optional(pa.float32(), nullable=False)
