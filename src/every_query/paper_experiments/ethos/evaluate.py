"""ETHOS evaluate stage: compute occurs_auroc per (query, duration, tier, bucket) (Phase 3).

Reads ``ethos_predictions.parquet`` from Phase 2, drops censored rows (null
``boolean_value``), and computes ``occurs_auroc`` per
``(query, duration_days, tier, bucket)`` using the same
``sklearn.metrics.roc_auc_score`` call EQ uses in
``src/every_query/evaluate/metrics.py``. Emits ``ethos_aucs_held_out.parquet``
shaped to align with EQ's ``eval_aucs_held_out.parquet`` so the two tables can
be unioned for the comparison notebook in Phase 4.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from sklearn.metrics import roc_auc_score

from every_query.paper_experiments.ethos.build_mapping import (
    BUCKET,
    CACHE_DIR,
    MAPPING_TABLE_BLOB,
    _download,
    _upload,
)
from every_query.paper_experiments.ethos.predict import PREDICTIONS_BLOB

AUCS_BLOB = "baselines/ethos/ethos_aucs_held_out.parquet"


def _auroc_or_none(y_true: list[bool], y_score: list[float]) -> float | None:
    if len(set(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _slug(code: str) -> str:
    """Match EQ's ``every_query.utils.codes.code_slug`` -- prefix + sha1[:10]."""
    import hashlib

    safe = (
        code.replace("//", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("[", "_")
        .replace("(", "_")
        .replace(")", "_")
    )
    h = hashlib.sha1(code.encode("utf-8")).hexdigest()[:10]
    return f"{safe[:24]}__{h}"


def compute_aucs(predictions: pl.DataFrame, mapping: pl.DataFrame) -> pl.DataFrame:
    """AUROC per ``(query, duration_days, tier, bucket)`` over labeled rows.

    Only ``(query, tier)`` pairs that have at least one mapping row are scored;
    the cross-join in :func:`predict.aggregate_predictions` yields
    ``occurs_prob=0`` for unmapped ``(query, tier)`` cells which would
    artificially inflate every tier's mean toward 0.5.

    Each row is annotated with the ``tier_quality`` (``primary`` or
    ``secondary``) inherited from the mapping table so downstream consumers
    can split the AUC table into primary-tier and secondary-tier views per
    Phase 10. ``union`` rows propagate the underlying primary-tier
    annotation when present, otherwise fall back to ``secondary``.
    """
    if "tier_quality" not in mapping.columns:
        mapping = mapping.with_columns(pl.lit("primary").alias("tier_quality"))

    quality = (
        mapping.select("query", "tier", "tier_quality")
        .unique()
        .group_by(["query", "tier"])
        .agg(
            pl.when(pl.col("tier_quality").eq("primary").any())
            .then(pl.lit("primary"))
            .otherwise(pl.lit("secondary"))
            .alias("tier_quality"),
        )
    )

    valid = mapping.select("query", "tier").unique()
    labeled = predictions.join(valid, on=["query", "tier"], how="inner").filter(
        pl.col("boolean_value").is_not_null()
    )
    rows = []
    for (query, duration_days, tier, bucket), grp in labeled.group_by(
        ["query", "duration_days", "tier", "bucket"]
    ):
        y = grp["boolean_value"].to_list()
        p = grp["occurs_prob"].to_list()
        rows.append(
            {
                "model": "ethos",
                "duration_days": int(duration_days),
                "code": query,
                "code_slug": _slug(query),
                "bucket": bucket,
                "tier": tier,
                "occurs_auc": _auroc_or_none(y, p),
                "n_rows": grp.height,
                "n_positive": int(sum(y)),
            }
        )
    auc_df = pl.DataFrame(rows)
    if auc_df.is_empty():
        return auc_df.with_columns(pl.lit("primary").alias("tier_quality"))
    auc_df = (
        auc_df.join(
            quality.rename({"query": "code"}), on=["code", "tier"], how="left"
        )
        .with_columns(pl.col("tier_quality").fill_null("secondary"))
        .sort(["tier", "bucket", "duration_days", "code"])
    )
    return auc_df


def main() -> None:
    print("Loading predictions ...")
    p = _download(PREDICTIONS_BLOB, CACHE_DIR / "ethos_predictions.parquet")
    preds = pl.read_parquet(p)
    print(f"  {len(preds)} prediction rows")

    print("Loading mapping table ...")
    mp = _download(MAPPING_TABLE_BLOB, CACHE_DIR / "mapping_table.parquet")
    mapping = pl.read_parquet(mp)

    print("Computing AUCs ...")
    aucs = compute_aucs(preds, mapping)
    print(f"  {len(aucs)} AUC rows")
    print()
    print(aucs)
    print()
    print("Per-tier mean occurs_auc (where defined):")
    print(
        aucs.filter(pl.col("occurs_auc").is_not_null())
        .group_by("tier")
        .agg(
            pl.len().alias("n"),
            pl.col("occurs_auc").mean().alias("mean_auc"),
            pl.col("occurs_auc").median().alias("median_auc"),
        )
        .sort("tier")
    )

    print()
    print("Primary vs secondary tier split (excluding union):")
    print(
        aucs.filter(
            pl.col("occurs_auc").is_not_null() & (pl.col("tier") != "union")
        )
        .group_by("tier_quality")
        .agg(
            pl.len().alias("n"),
            pl.col("occurs_auc").mean().alias("mean_auc"),
            pl.col("occurs_auc").median().alias("median_auc"),
        )
        .sort("tier_quality")
    )
    print()
    print("Union tier (per-code OR; reported separately):")
    print(
        aucs.filter(
            pl.col("occurs_auc").is_not_null() & (pl.col("tier") == "union")
        )
        .group_by("tier_quality")
        .agg(
            pl.len().alias("n"),
            pl.col("occurs_auc").mean().alias("mean_auc"),
            pl.col("occurs_auc").median().alias("median_auc"),
        )
        .sort("tier_quality")
    )

    out = CACHE_DIR / "ethos_aucs_held_out.parquet"
    aucs.write_parquet(out)
    _upload(out, AUCS_BLOB)
    print(f"\nWrote {out}")
    print(f"Uploaded gs://{BUCKET}/{AUCS_BLOB}")


if __name__ == "__main__":
    main()
