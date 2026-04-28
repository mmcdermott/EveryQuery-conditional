"""``EQ_reprocess_ethos`` — collapse Ethos's split-token MEDS shards to singleton tokens.

The Ethos tokenizer splits each ICD10/CM diagnosis code into three events at the same
``(subject_id, time)`` (chars 0-3 → ``ICD//CM//<head>``, chars 3-6 → ``ICD//CM//3-6//<mid>``,
chars 6+ → ``ICD//CM//SFX//<sfx>``) and each ATC drug code into three events similarly
(``ATC//<head>``, ``ATC//4//<mid>``, ``ATC//SFX//<sfx>``).  EQ's task grammar
(:class:`every_query.data.schema.TaskQuerySchema`) treats ``code`` as an atom, so a query
for "did ``I10`` occur" against Ethos's split-token corpus would require conjunctive
reasoning over the triplet — which the load-bearing
:func:`every_query.generate_tasks.sample_tasks.evaluate_index_df` cannot express.

This module reverses the splits.  For every code family Ethos splits, the rewriter:

1. Identifies the role (``head`` / ``mid`` / ``sfx``) of each row by code prefix.
2. Within each ``(subject_id, time, role)`` group, assigns a per-role rank.
3. Pairs ``head[k]`` with ``mid[k]`` and ``sfx[k]`` (the assumption is that Ethos's
   per-original-event ``head, mid, sfx`` triple stays contiguous after its
   ``polars.explode``; see ``ipolharvard/ethos-ares``).
4. Emits a single combined code per pairing (``ICD//CM//I10`` alone, or
   ``ICD//CM//I10//3-6//00//SFX//1`` for full-triplet).  The combined code is
   deterministic and reversible: a downstream reader can recover the original
   triplet by splitting on ``//3-6//`` and ``//SFX//``.

Output codes are intentionally not the raw ICD10/ATC strings — tokenization is a
property of the preprocessing run, not of EQ.  See #174 for the design discussion.

The ``ethos_code_mapping.parquet`` written under ``{output_dir}/metadata/`` records every
distinct ``(input_codes_set → output_code)`` rewrite the run performed.  Pass-through codes
(no recombination) are not included.

Implementation notes:

* Atomic event types Ethos emits without splitting (labs, vitals, BMI, SOFA, admissions,
  discharges, demographics, time-deltas) pass through unchanged.
* If multiple heads of the same family co-occur at one ``(subject_id, time)`` (e.g., a
  patient has two ICD codes recorded at the same instant), the rank-based pairing
  preserves Ethos's row order.  This is correct as long as Ethos preserves the
  per-original-event ordering during ``.explode()``, which it does in the current
  implementation.  Deviations would surface as visibly garbled output codes; tests cover
  the round-trip identity for canonical inputs.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import polars as pl

from every_query.utils._env import ensure_env  # noqa: F401  (parity with other CLIs)

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Family definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Family:
    """One code family Ethos splits.

    Attributes:
        name: Short identifier for logging / mapping output.
        head_prefix: Codes starting with this prefix (and *not* the mid/sfx prefixes)
            are head tokens.  Carries the original code's chars 0-3.
        mid_prefix: Codes starting with this prefix are mid tokens.  Carries chars 3-6
            (ICD) or char 3 (ATC).
        sfx_prefix: Codes starting with this prefix are suffix tokens.  Carries chars 6+
            (ICD) or chars 4+ (ATC).
        mid_marker: Inserted between head and mid when recombining (e.g., ``"3-6//"`` for
            ICD, ``"4//"`` for ATC) so the output preserves the role provenance.
        sfx_marker: Likewise for suffix.  Always ``"SFX//"`` in Ethos's scheme today.
    """

    name: str
    head_prefix: str
    mid_prefix: str
    sfx_prefix: str
    mid_marker: str
    sfx_marker: str = "SFX//"


# Order matters only for logging (per-family stats are emitted in this order).
FAMILIES: tuple[_Family, ...] = (
    _Family(
        name="ICD_CM",
        head_prefix="ICD//CM//",
        mid_prefix="ICD//CM//3-6//",
        sfx_prefix="ICD//CM//SFX//",
        mid_marker="3-6//",
    ),
    _Family(
        name="ATC",
        head_prefix="ATC//",
        mid_prefix="ATC//4//",
        sfx_prefix="ATC//SFX//",
        mid_marker="4//",
    ),
)


# ---------------------------------------------------------------------------
# Recombination
# ---------------------------------------------------------------------------


def _classify_role(events: pl.DataFrame, family: _Family) -> pl.DataFrame:
    """Tag each row with its role in the family (``head`` / ``mid`` / ``sfx`` / null).

    A row whose ``code`` starts with the mid prefix is ``mid``; with the sfx prefix is
    ``sfx``; with the head prefix but neither of those two is ``head``; everything else
    is null (not part of this family — pass-through).
    """
    code = pl.col("code")
    is_mid = code.str.starts_with(family.mid_prefix)
    is_sfx = code.str.starts_with(family.sfx_prefix)
    is_head = code.str.starts_with(family.head_prefix) & ~is_mid & ~is_sfx
    return events.with_columns(
        pl.when(is_mid)
        .then(pl.lit("mid"))
        .when(is_sfx)
        .then(pl.lit("sfx"))
        .when(is_head)
        .then(pl.lit("head"))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
        .alias("_role")
    )


def _strip_role_prefix(code_expr: pl.Expr, role_prefix: str) -> pl.Expr:
    """Strip a known role prefix from a code column expression."""
    return code_expr.str.slice(len(role_prefix))


def _build_combined_code(family: _Family) -> pl.Expr:
    """Polars expression: combine ``head`` + optional ``mid`` + optional ``sfx`` into one code.

    Inputs are columns ``head``, ``mid``, ``sfx`` (any of mid/sfx may be null).  Output:
    ``head`` if mid and sfx are both null; else ``head//<mid_marker><mid_value>`` and/or
    ``head//<mid_marker><mid_value>//<sfx_marker><sfx_value>``.
    """
    mid_value = _strip_role_prefix(pl.col("mid"), family.mid_prefix)
    sfx_value = _strip_role_prefix(pl.col("sfx"), family.sfx_prefix)
    mid_part = (
        pl.when(pl.col("mid").is_not_null())
        .then(pl.lit("//") + pl.lit(family.mid_marker) + mid_value)
        .otherwise(pl.lit(""))
    )
    sfx_part = (
        pl.when(pl.col("sfx").is_not_null())
        .then(pl.lit("//") + pl.lit(family.sfx_marker) + sfx_value)
        .otherwise(pl.lit(""))
    )
    return pl.concat_str([pl.col("head"), mid_part, sfx_part])


def _recombine_family(events: pl.DataFrame, family: _Family) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Recombine all rows of one family into singleton-code rows.

    Args:
        events: A shard of MEDS events (must have at least ``subject_id``, ``time``, ``code``;
            other columns are preserved on head rows and dropped on mid/sfx rows).
        family: Which family's prefixes to handle this pass.

    Returns:
        A tuple ``(rewritten_family_rows, mapping_records)``.

        ``rewritten_family_rows`` is the family's recombined rows, one per
        ``(subject_id, time, head_rank)`` triplet.  Numeric / text auxiliary columns
        come from the head row (Ethos's split sub-tokens carry no numeric value, so
        this is non-lossy for any sane input).

        ``mapping_records`` is a long-form frame ``(input_code, output_code)``,
        one row per input sub-token that contributed to a recombination — only
        emitted for codes that *actually changed*.
    """
    if events.height == 0:
        return events.head(0), pl.DataFrame(schema={"input_code": pl.Utf8, "output_code": pl.Utf8})

    classified = _classify_role(events, family)
    family_rows = classified.filter(pl.col("_role").is_not_null())
    if family_rows.height == 0:
        return events.head(0), pl.DataFrame(schema={"input_code": pl.Utf8, "output_code": pl.Utf8})

    # Within each (subject_id, time, role), assign a 0-based rank that pairs head[k]
    # with mid[k] and sfx[k].  ``cum_count`` enumerates rows in their original order
    # within the partition — relying on Ethos's polars-native ordering.
    family_rows = family_rows.with_columns(
        (pl.cum_count("_role").over("subject_id", "time", "_role") - 1).alias("_rank")
    )

    # Pivot roles into columns: head/mid/sfx as separate columns, indexed by
    # (subject_id, time, _rank).  Auxiliary columns (numeric_value, text_value) are
    # carried forward via a separate self-join below — pivot on its own only handles
    # the code column.
    head_rows = family_rows.filter(pl.col("_role") == "head").drop("_role")
    mid_rows = (
        family_rows.filter(pl.col("_role") == "mid")
        .select(["subject_id", "time", "_rank", "code"])
        .rename({"code": "mid"})
    )
    sfx_rows = (
        family_rows.filter(pl.col("_role") == "sfx")
        .select(["subject_id", "time", "_rank", "code"])
        .rename({"code": "sfx"})
    )

    combined = (
        head_rows.rename({"code": "head"})
        .join(mid_rows, on=["subject_id", "time", "_rank"], how="left")
        .join(sfx_rows, on=["subject_id", "time", "_rank"], how="left")
        .with_columns(_build_combined_code(family).alias("code"))
    )

    # Diagnostic: count orphan mid/sfx tokens that didn't pair with a head (shouldn't
    # happen under Ethos's pipeline; defensive log).
    n_heads = head_rows.height
    n_mids = mid_rows.height
    n_sfxs = sfx_rows.height
    orphan_mids = max(0, n_mids - n_heads)
    orphan_sfxs = max(0, n_sfxs - n_heads)
    if orphan_mids or orphan_sfxs:
        logger.warning(
            "Family %s: %d orphan mid token(s), %d orphan sfx token(s) "
            "without matching head — they are dropped from the recombined output.",
            family.name,
            orphan_mids,
            orphan_sfxs,
        )

    # Mapping: for codes that actually changed, emit (input_code, output_code) pairs.
    mapping_long = combined.filter(pl.col("head") != pl.col("code")).select(
        [
            pl.col("head").alias("input_code"),
            pl.col("code").alias("output_code"),
        ]
    )
    if "mid" in combined.columns:
        mapping_long = pl.concat(
            [
                mapping_long,
                combined.filter(pl.col("mid").is_not_null()).select(
                    [pl.col("mid").alias("input_code"), pl.col("code").alias("output_code")]
                ),
            ],
            how="vertical",
        )
    if "sfx" in combined.columns:
        mapping_long = pl.concat(
            [
                mapping_long,
                combined.filter(pl.col("sfx").is_not_null()).select(
                    [pl.col("sfx").alias("input_code"), pl.col("code").alias("output_code")]
                ),
            ],
            how="vertical",
        )

    # Drop the helper columns; output mirrors the original schema.
    out_cols = [c for c in events.columns if c in combined.columns]
    return combined.select(out_cols), mapping_long


def reprocess_shard(events: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Reprocess one MEDS shard: recombine all known split families.

    Returns:
        ``(rewritten_events, mapping_records)`` where ``rewritten_events`` has the same
        schema as ``events`` but with every ``(head, mid, sfx)`` triplet collapsed into
        a single combined-code row, and ``mapping_records`` is the long-form
        ``(input_code, output_code)`` mapping for every changed code.
    """
    if events.height == 0:
        return events, pl.DataFrame(schema={"input_code": pl.Utf8, "output_code": pl.Utf8})

    # Classify each row by the *first* family whose prefixes match.  Family prefixes
    # are non-overlapping in Ethos's scheme (``ICD//CM//`` vs ``ATC//``), so first-match
    # is unambiguous.
    in_any_family = pl.lit(False)
    for fam in FAMILIES:
        in_any_family = in_any_family | pl.col("code").str.starts_with(fam.head_prefix)

    untouched = events.filter(~in_any_family)

    rewritten_per_family: list[pl.DataFrame] = []
    mapping_per_family: list[pl.DataFrame] = []
    for fam in FAMILIES:
        # Filter to rows whose code starts with this family's broadest prefix
        # (head_prefix is the most general, matches mid + sfx + head all at once).
        family_subset = events.filter(pl.col("code").str.starts_with(fam.head_prefix))
        rewritten, mapping = _recombine_family(family_subset, fam)
        if rewritten.height > 0:
            rewritten_per_family.append(rewritten)
        if mapping.height > 0:
            mapping_per_family.append(mapping)

    if rewritten_per_family:
        rewritten = pl.concat([untouched, *rewritten_per_family], how="diagonal_relaxed")
    else:
        rewritten = untouched

    rewritten = rewritten.sort(["subject_id", "time"])

    if mapping_per_family:
        mapping = pl.concat(mapping_per_family, how="vertical").unique()
    else:
        mapping = pl.DataFrame(schema={"input_code": pl.Utf8, "output_code": pl.Utf8})

    return rewritten, mapping


# ---------------------------------------------------------------------------
# I/O orchestration
# ---------------------------------------------------------------------------


def _discover_shards(input_dir: Path) -> list[tuple[Path, Path]]:
    """Find all parquet shards under ``{input_dir}/data/``.

    Returns:
        List of ``(absolute_input_path, relative_path)`` pairs.  ``relative_path`` is
        relative to ``input_dir`` so the same layout can be reproduced under
        ``output_dir``.
    """
    data_root = input_dir / "data"
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"Expected MEDS shards under {data_root!s}, but it does not exist or is not a directory."
        )
    shards = sorted(p for p in data_root.rglob("*.parquet"))
    return [(p, p.relative_to(input_dir)) for p in shards]


def reprocess_directory(input_dir: Path, output_dir: Path, overwrite: bool = False) -> Path:
    """Recombine every parquet under ``input_dir/data/`` and write to ``output_dir/data/``.

    Side effects:
        * Writes recombined shards to ``output_dir/data/{relative_path}``.
        * Writes the cumulative code mapping to
          ``output_dir/metadata/ethos_code_mapping.parquet``.
        * Copies ``input_dir/metadata/codes.parquet`` to ``output_dir/metadata/codes.parquet``
          *with codes rewritten via the same mapping*, if the source file exists.  Other
          metadata files (subject_splits, dataset_metadata) are copied through unchanged.

    Returns:
        The output directory path (for chaining / logging).
    """
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"output_dir {output_dir!s} already exists and is non-empty; "
                f"pass ``overwrite=true`` to clobber."
            )
        logger.info("overwrite=true — clearing %s", output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shards = _discover_shards(input_dir)
    if not shards:
        raise ValueError(f"No parquet shards found under {input_dir / 'data'}.")

    logger.info("Reprocessing %d shard(s) from %s → %s", len(shards), input_dir, output_dir)

    cumulative_mapping: list[pl.DataFrame] = []
    n_rows_in_total = 0
    n_rows_out_total = 0
    for src, rel in shards:
        events = pl.read_parquet(src)
        n_in = events.height
        rewritten, mapping = reprocess_shard(events)
        n_out = rewritten.height
        n_rows_in_total += n_in
        n_rows_out_total += n_out

        dst = output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        rewritten.write_parquet(dst)
        if mapping.height > 0:
            cumulative_mapping.append(mapping)

        logger.info(
            "  %s: %d → %d rows (%+d, %d unique mappings)",
            rel,
            n_in,
            n_out,
            n_out - n_in,
            mapping.height,
        )

    # Cumulative mapping output
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    if cumulative_mapping:
        full_mapping = pl.concat(cumulative_mapping, how="vertical").unique()
    else:
        full_mapping = pl.DataFrame(schema={"input_code": pl.Utf8, "output_code": pl.Utf8})
    mapping_path = metadata_dir / "ethos_code_mapping.parquet"
    full_mapping.write_parquet(mapping_path)
    logger.info("Wrote %d unique code mappings to %s", full_mapping.height, mapping_path)

    # Carry through metadata.  We rewrite ``codes.parquet`` if present (so the universe
    # downstream sees combined codes); other files are copied as-is.
    src_metadata_dir = input_dir / "metadata"
    if src_metadata_dir.is_dir():
        for src_meta in src_metadata_dir.iterdir():
            if not src_meta.is_file():
                continue
            dst_meta = metadata_dir / src_meta.name
            if src_meta.name == "codes.parquet" and full_mapping.height > 0:
                _rewrite_codes_metadata(src_meta, dst_meta, full_mapping)
            else:
                shutil.copy2(src_meta, dst_meta)

    logger.info(
        "Total: %d → %d rows across %d shard(s); %d unique input→output code rewrites.",
        n_rows_in_total,
        n_rows_out_total,
        len(shards),
        full_mapping.height,
    )
    return output_dir


def _rewrite_codes_metadata(src: Path, dst: Path, mapping: pl.DataFrame) -> None:
    """Rewrite ``codes.parquet`` to use combined codes; uncovered rows pass through.

    The source ``codes.parquet`` has one row per code in the input vocabulary.  After
    recombination, mid/sfx codes no longer appear standalone — they're absorbed into
    the combined codes.  We replace each input-side row with its combined-code row,
    deduplicating on the output code.  Heads that contribute to multiple combined codes
    keep the *last* seen output code; in practice every head/mid/sfx contributes to
    exactly one canonical combined code per (subject, time) site, but the dedup is
    defensive.
    """
    codes = pl.read_parquet(src)
    if "code" not in codes.columns:
        logger.warning("codes.parquet at %s has no ``code`` column; passing through unchanged.", src)
        shutil.copy2(src, dst)
        return

    # Right-join the mapping onto codes to swap any input_code → output_code.  Rows
    # whose code isn't in the mapping pass through (output_code is null → coalesce
    # back to original code).
    rewritten = codes.join(
        mapping.rename({"input_code": "code"}),
        on="code",
        how="left",
    ).with_columns(pl.coalesce(["output_code", "code"]).alias("code"))
    rewritten = rewritten.drop("output_code")

    # The mid/sfx tokens are now redundant (their content is folded into the combined
    # head row).  Drop them by filtering out rows whose original code was a non-head
    # token; equivalently, every row whose code starts with a known mid/sfx prefix.
    drop_predicate = pl.lit(False)
    for fam in FAMILIES:
        drop_predicate = (
            drop_predicate
            | pl.col("code").str.starts_with(fam.mid_prefix)
            | pl.col("code").str.starts_with(fam.sfx_prefix)
        )
    rewritten = rewritten.filter(~drop_predicate).unique(subset=["code"], keep="first")
    rewritten.write_parquet(dst)


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


CONFIGS = str(files("every_query") / "paper_experiments" / "ethos_compat" / "configs")


@hydra.main(version_base=None, config_path=CONFIGS, config_name="reprocess")
def main(cfg: DictConfig) -> None:
    """Hydra entry point for ``EQ_reprocess_ethos``.

    Reads parquets at ``input_dir/data/{split}/*.parquet``, rewrites split-token
    triplets into singleton codes, writes to ``output_dir`` mirroring the input layout.
    Downstream MTD_preprocess + EQ_train consume the output dir unchanged.
    """
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s")

    input_dir = Path(str(cfg.input_dir)).expanduser().resolve()
    output_dir = Path(str(cfg.output_dir)).expanduser().resolve()
    overwrite = bool(cfg.get("overwrite", False))

    if not input_dir.is_dir():
        raise FileNotFoundError(f"input_dir {input_dir!s} does not exist or is not a directory.")
    if input_dir == output_dir:
        raise ValueError("input_dir and output_dir must differ — refusing to rewrite in place.")

    reprocess_directory(input_dir, output_dir, overwrite=overwrite)


if __name__ == "__main__":
    main()
