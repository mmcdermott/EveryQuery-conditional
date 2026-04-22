"""Unit tests for ``EveryQueryPytorchDataset.get_task_seq_bounds_and_labels``.

Guards against silent misalignment between the upstream-computed end indices
and the EveryQuery-specific annotation columns (``occurs``, ``query``,
``duration_days``) that the subclass hstacks onto the result.
"""

from datetime import datetime

import polars as pl
from meds import DataSchema, LabelSchema

from every_query.data.dataset import EveryQueryPytorchDataset


def _schema_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            DataSchema.subject_id_name: [1, 2, 3],
            DataSchema.time_name: [
                [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
                [datetime(2020, 1, 1), datetime(2020, 1, 5)],
                [datetime(2020, 1, 10)],
            ],
        }
    )


class TestExtraColumnAlignment:
    def test_extras_align_to_original_label_rows(self):
        label_df = pl.DataFrame(
            {
                DataSchema.subject_id_name: [1, 1, 2, 3, 99],
                LabelSchema.prediction_time_name: [
                    datetime(2020, 1, 2),
                    datetime(2020, 1, 3),
                    datetime(2020, 1, 5),
                    datetime(2020, 1, 10),
                    datetime(2020, 1, 1),
                ],
                "boolean_value": [True, False, True, False, True],
                "occurs": [10, 20, 30, 40, 50],
                "query": [101, 102, 103, 104, 105],
                "duration_days": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )

        result = EveryQueryPytorchDataset.get_task_seq_bounds_and_labels(label_df, _schema_df())

        # Subject 99 is absent from schema_df → inner-join semantics drop it.
        assert result.height == 4
        assert set(result.columns) >= {
            DataSchema.subject_id_name,
            LabelSchema.prediction_time_name,
            EveryQueryPytorchDataset.END_IDX,
            "boolean_value",
            "occurs",
            "query",
            "duration_days",
        }

        # For each surviving row, the extras must match the row in label_df with the
        # same (subject_id, prediction_time).
        label_lookup = {
            (row[DataSchema.subject_id_name], row[LabelSchema.prediction_time_name]): row
            for row in label_df.iter_rows(named=True)
        }
        for out_row in result.iter_rows(named=True):
            key = (out_row[DataSchema.subject_id_name], out_row[LabelSchema.prediction_time_name])
            expected = label_lookup[key]
            assert out_row["occurs"] == expected["occurs"]
            assert out_row["query"] == expected["query"]
            assert out_row["duration_days"] == expected["duration_days"]
            assert out_row["boolean_value"] == expected["boolean_value"]

    def test_no_extras_passes_through(self):
        label_df = pl.DataFrame(
            {
                DataSchema.subject_id_name: [1, 2],
                LabelSchema.prediction_time_name: [datetime(2020, 1, 2), datetime(2020, 1, 5)],
                "boolean_value": [True, False],
            }
        )

        result = EveryQueryPytorchDataset.get_task_seq_bounds_and_labels(label_df, _schema_df())

        assert result.height == 2
        assert "occurs" not in result.columns
        assert "query" not in result.columns
        assert "duration_days" not in result.columns

    def test_partial_extras(self):
        label_df = pl.DataFrame(
            {
                DataSchema.subject_id_name: [1, 2],
                LabelSchema.prediction_time_name: [datetime(2020, 1, 2), datetime(2020, 1, 5)],
                "boolean_value": [True, False],
                "query": [7, 8],
            }
        )

        result = EveryQueryPytorchDataset.get_task_seq_bounds_and_labels(label_df, _schema_df())

        assert "query" in result.columns
        assert "occurs" not in result.columns
        assert "duration_days" not in result.columns
        by_sid = {r[DataSchema.subject_id_name]: r["query"] for r in result.iter_rows(named=True)}
        assert by_sid == {1: 7, 2: 8}
