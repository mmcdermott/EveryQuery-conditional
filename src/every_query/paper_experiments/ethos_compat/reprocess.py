"""``EQ_reprocess_ethos`` — make Ethos-tokenized MEDS shards ingestible by EQ training.

The Ethos tokenizer (``ipolharvard/ethos-ares``) splits codes across multiple events at
the same ``(subject_id, time)``: each ICD10/CM diagnosis becomes three rows (chars 0-3 →
``ICD//CM//<head>``, chars 3-6 → ``ICD//CM//3-6//<mid>``, chars 6+ →
``ICD//CM//SFX//<sfx>``); each ATC drug code becomes three rows (``ATC//<head>``,
``ATC//4//<mid>``, ``ATC//SFX//<sfx>``).  EQ's task grammar treats ``code`` as an atom,
so querying "did ``I10`` occur" against the split corpus would require conjunctive
reasoning that :func:`every_query.generate_tasks.sample_tasks.evaluate_index_df` cannot
express.

This module reverses the splits.  For every split family at every shared timestamp, the
``(head, optional mid, optional sfx)`` rows collapse into one row whose code is a
deterministic opaque identifier ``EQ_TOK//<hash>``.  The mapping file
(``metadata/ethos_code_mapping.parquet``) records ``(input_code, output_code, family)``
rows for every changed code so the reverse lookup is recoverable.

Atomic event types Ethos emits without splitting (labs, vitals, BMI, SOFA quantile
tokens, admissions, discharges, time-deltas) pass through unchanged.

Optional static-data reinjection: when ``static_data_path`` points at a parquet of
``(subject_id, code, [numeric_value])`` rows, every entry is written as a ``time=null``
MEDS row into the shard its subject lives in.  Ethos's native ``StaticDataCollector``
emits a pickle whose schema is not yet inspected — only parquet input is supported here
for now (convert externally if you have a pickle).  ``scripts/setup_ethos_demo_data.py``
documents how to reproduce a real Ethos run and surface the actual pickle layout.

See #174 for the design discussion.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import polars as pl

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
            are head tokens.
        mid_prefix: Codes starting with this prefix are mid tokens.
        sfx_prefix: Codes starting with this prefix are suffix tokens.
    """

    name: str
    head_prefix: str
    mid_prefix: str
    sfx_prefix: str


# Order matters only for logging.  Family head prefixes must be non-overlapping in
# Ethos's scheme (``ICD//CM//`` vs ``ATC//``) so first-match classification is
# unambiguous; this is asserted in the test suite.
FAMILIES: tuple[_Family, ...] = (
    _Family(
        name="ICD_CM",
        head_prefix="ICD//CM//",
        mid_prefix="ICD//CM//3-6//",
        sfx_prefix="ICD//CM//SFX//",
    ),
    _Family(
        name="ATC",
        head_prefix="ATC//",
        mid_prefix="ATC//4//",
        sfx_prefix="ATC//SFX//",
    ),
)


# ---------------------------------------------------------------------------
# Code naming
# ---------------------------------------------------------------------------


COMBINED_PREFIX = "EQ_TOK//"
_HASH_DIGEST_BYTES = 8  # → 16 hex chars (64 bits) — collision-free at any reasonable EHR vocab size.


def make_combined_code(input_codes: tuple[str, ...] | list[str]) -> str:
    """Stable opaque identifier for a recombined triplet.

    The output is ``EQ_TOK//<16-char-hex>`` where the hex is a blake2b digest of the
    sorted input-codes tuple.  Sorting makes the function commutative — identical
    triplets in any row order produce the same identifier.

    The identifier is not reversible from the string alone — the
    ``ethos_code_mapping.parquet`` written by :func:`reprocess_directory` is the
    canonical reverse lookup.

    Examples:
        >>> a = make_combined_code(("ICD//CM//I10", "ICD//CM//3-6//00", "ICD//CM//SFX//A"))
        >>> b = make_combined_code(("ICD//CM//SFX//A", "ICD//CM//I10", "ICD//CM//3-6//00"))
        >>> a == b
        True
        >>> a.startswith("EQ_TOK//")
        True
        >>> make_combined_code(("ICD//CM//I10",)) != make_combined_code(("ICD//CM//I11",))
        True
        >>> code = make_combined_code(("ICD//CM//I10",))
        >>> code.startswith("EQ_TOK//") and len(code) == len("EQ_TOK//") + 16
        True
    """
    h = hashlib.blake2b(digest_size=_HASH_DIGEST_BYTES)
    for c in sorted(input_codes):
        h.update(c.encode("utf-8"))
        h.update(b"\x1f")  # unit separator — avoid prefix collisions across components
    return COMBINED_PREFIX + h.hexdigest()


# ---------------------------------------------------------------------------
# Recombination
# ---------------------------------------------------------------------------


def _classify_role(events: pl.DataFrame, family: _Family) -> pl.DataFrame:
    """Tag each row with its role in the family (``head`` / ``mid`` / ``sfx`` / null).

    Mid- and sfx-prefix checks come first; whatever's left of the family's broad prefix is treated as a head.
    Family prefixes are designed to be non-overlapping (asserted in tests) so first-match classification is
    unambiguous.
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


def _recombine_family(events: pl.DataFrame, family: _Family) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Recombine all rows of one family into singleton-code rows.

    The recombination pairs ``head[k]`` with ``mid[k]`` and ``sfx[k]`` within each
    ``(subject_id, time)`` based on Ethos's row order.  This relies on Ethos's polars
    ``.explode()`` preserving the per-original-event grouping, which it does in the
    current upstream implementation.

    Returns:
        ``(rewritten_family_rows, mapping_records)`` — ``mapping_records`` has columns
        ``(input_code, output_code, family)``, one row per input sub-token that
        contributed to a recombination.
    """
    if events.height == 0:
        return events.head(0), _empty_mapping()

    classified = _classify_role(events, family)
    family_rows = classified.filter(pl.col("_role").is_not_null())
    if family_rows.height == 0:
        return events.head(0), _empty_mapping()

    # Per-(subject_id, time, role) rank — pair head[k] with mid[k] and sfx[k].
    family_rows = family_rows.with_columns(
        (pl.cum_count("_role").over("subject_id", "time", "_role") - 1).alias("_rank")
    )

    head_rows = family_rows.filter(pl.col("_role") == "head").drop("_role")
    if head_rows.height == 0:
        logger.warning(
            "Family %s: %d row(s) but no head tokens — entire family dropped from output.",
            family.name,
            family_rows.height,
        )
        return events.head(0), _empty_mapping()

    mid_rows = (
        family_rows.filter(pl.col("_role") == "mid")
        .select(["subject_id", "time", "_rank", "code"])
        .rename({"code": "_mid"})
    )
    sfx_rows = (
        family_rows.filter(pl.col("_role") == "sfx")
        .select(["subject_id", "time", "_rank", "code"])
        .rename({"code": "_sfx"})
    )

    n_heads = head_rows.height
    orphan_mids = max(0, mid_rows.height - n_heads)
    orphan_sfxs = max(0, sfx_rows.height - n_heads)
    if orphan_mids or orphan_sfxs:
        logger.warning(
            "Family %s: %d orphan mid token(s), %d orphan sfx token(s) without matching head — "
            "they are dropped from the recombined output.",
            family.name,
            orphan_mids,
            orphan_sfxs,
        )

    joined = (
        head_rows.rename({"code": "_head"})
        .join(mid_rows, on=["subject_id", "time", "_rank"], how="left")
        .join(sfx_rows, on=["subject_id", "time", "_rank"], how="left")
    )

    # Hash each unique (head, mid, sfx) triplet *once*, then join back — avoids a
    # Python-per-row call for every event in the shard.  Hashing happens in plain Python
    # on the small unique set rather than in a polars expression because polars'
    # ``map_elements`` over structs is fiddly across versions and not worth the risk.
    unique_triplets = joined.select(["_head", "_mid", "_sfx"]).unique()
    triplet_codes = [
        make_combined_code(tuple(c for c in (h, m, s) if c is not None))
        for h, m, s in zip(
            unique_triplets["_head"].to_list(),
            unique_triplets["_mid"].to_list(),
            unique_triplets["_sfx"].to_list(),
            strict=True,
        )
    ]
    hashed = unique_triplets.with_columns(pl.Series("code", triplet_codes, dtype=pl.Utf8))
    # Polars join treats null != null by default, which would drop rows whose mid/sfx
    # are null (the head-only and head+mid cases).  ``nulls_equal=True`` matches
    # null-to-null so every row in ``joined`` finds its hashed counterpart.
    joined = joined.join(hashed, on=["_head", "_mid", "_sfx"], how="left", nulls_equal=True)

    # Mapping rows — one per input sub-token that contributed to a recombination.
    family_lit = pl.lit(family.name)
    mapping_parts: list[pl.DataFrame] = [
        joined.select(
            [
                pl.col("_head").alias("input_code"),
                pl.col("code").alias("output_code"),
                family_lit.alias("family"),
            ]
        ),
    ]
    if "_mid" in joined.columns:
        mapping_parts.append(
            joined.filter(pl.col("_mid").is_not_null()).select(
                [
                    pl.col("_mid").alias("input_code"),
                    pl.col("code").alias("output_code"),
                    family_lit.alias("family"),
                ]
            )
        )
    if "_sfx" in joined.columns:
        mapping_parts.append(
            joined.filter(pl.col("_sfx").is_not_null()).select(
                [
                    pl.col("_sfx").alias("input_code"),
                    pl.col("code").alias("output_code"),
                    family_lit.alias("family"),
                ]
            )
        )
    mapping = pl.concat(mapping_parts, how="vertical").unique()

    out_cols = [c for c in events.columns if c in joined.columns]
    return joined.select(out_cols), mapping


def _empty_mapping() -> pl.DataFrame:
    """Empty mapping frame with the canonical schema."""
    return pl.DataFrame(schema={"input_code": pl.Utf8, "output_code": pl.Utf8, "family": pl.Utf8})


def reprocess_shard(events: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Reprocess one MEDS shard: recombine all known split families.

    Atomic events (codes outside the known split families) pass through unchanged.
    """
    if events.height == 0:
        return events, _empty_mapping()

    in_any_family = pl.lit(False)
    for fam in FAMILIES:
        in_any_family = in_any_family | pl.col("code").str.starts_with(fam.head_prefix)

    untouched = events.filter(~in_any_family)

    rewritten_per_family: list[pl.DataFrame] = []
    mapping_per_family: list[pl.DataFrame] = []
    for fam in FAMILIES:
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
        mapping = _empty_mapping()

    return rewritten, mapping


# ---------------------------------------------------------------------------
# Static reinjection
# ---------------------------------------------------------------------------


def load_static_table(path: Path) -> pl.DataFrame:
    """Load a parquet-format static-data table.

    Schema requirement: ``subject_id`` and ``code`` columns are required; ``numeric_value``
    is optional.  Other columns pass through.

    Ethos's native ``StaticDataCollector`` emits a pickle whose schema this PR has not
    yet inspected — until that's resolved, only the parquet format is supported here.
    Convert your Ethos pickle externally (see ``scripts/setup_ethos_demo_data.py``).

    Raises:
        ValueError: if the file extension is not ``.parquet``, or required columns are
            missing.
        FileNotFoundError: if the path doesn't exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Static-data file does not exist: {path!s}")
    if path.suffix.lower() != ".parquet":
        raise ValueError(
            f"Static-data file at {path!s} has unsupported extension {path.suffix!r}; "
            f"only .parquet is supported.  Convert your Ethos pickle externally — see "
            f"scripts/setup_ethos_demo_data.py for the recommended workflow."
        )
    df = pl.read_parquet(path)
    missing = {"subject_id", "code"} - set(df.columns)
    if missing:
        raise ValueError(
            f"Static parquet at {path!s} is missing required column(s): {sorted(missing)}; got {df.columns!r}"
        )
    return df


def _route_statics_to_shards(
    static_table: pl.DataFrame,
    subject_to_shard: dict[int, Path],
) -> dict[Path, pl.DataFrame]:
    """Group static rows by their subject's home shard.

    Subjects in the static table that don't appear in any timeline shard are dropped with a warning (a static-
    only subject has no events for EQ to train on anyway).
    """
    table_with_shard = static_table.with_columns(
        pl.col("subject_id")
        .map_elements(lambda sid: str(subject_to_shard.get(sid, "")), return_dtype=pl.Utf8)
        .alias("_shard_path")
    )
    orphan_count = table_with_shard.filter(pl.col("_shard_path") == "").height
    if orphan_count:
        logger.warning(
            "%d static row(s) reference subjects not present in any timeline shard — dropping them.",
            orphan_count,
        )
    table_with_shard = table_with_shard.filter(pl.col("_shard_path") != "")

    routed: dict[Path, pl.DataFrame] = {}
    for key, group in table_with_shard.group_by("_shard_path"):
        # Polars group_by yields ``(key_tuple, group_df)`` for both single- and multi-key groupings.
        shard_path_str = key[0] if isinstance(key, tuple) else key
        routed[Path(shard_path_str)] = group.drop("_shard_path")
    return routed


def _build_subject_shard_map(shard_paths: list[Path]) -> dict[int, Path]:
    """One pass over each shard to learn which shard each subject lives in."""
    sid_to_shard: dict[int, Path] = {}
    for sp in shard_paths:
        sids = pl.read_parquet(sp, columns=["subject_id"])["subject_id"].unique().to_list()
        for sid in sids:
            sid_to_shard[sid] = sp
    return sid_to_shard


# ---------------------------------------------------------------------------
# I/O orchestration
# ---------------------------------------------------------------------------


def _discover_shards(input_dir: Path) -> list[tuple[Path, Path]]:
    """Find all parquet shards under ``{input_dir}/data/``."""
    data_root = input_dir / "data"
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"Expected MEDS shards under {data_root!s}, but it does not exist or is not a directory."
        )
    shards = sorted(p for p in data_root.rglob("*.parquet"))
    return [(p, p.relative_to(input_dir)) for p in shards]


def reprocess_directory(
    input_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
    static_data_path: Path | None = None,
) -> Path:
    """Recombine every parquet under ``input_dir/data/`` and optionally reinject statics.

    Side effects:
        * Writes recombined shards to ``output_dir/data/{relative_path}``.
        * Writes the cumulative code mapping (sorted, deterministic) to
          ``output_dir/metadata/ethos_code_mapping.parquet``.
        * Optionally appends ``time=null`` static rows from a parquet, routed into each
          subject's home shard.
        * Copies ``input_dir/metadata/codes.parquet`` to ``output_dir/metadata/codes.parquet``
          with codes rewritten via the same mapping.  Other metadata files pass through.

    Returns:
        The output directory path.
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

    statics_per_shard: dict[Path, pl.DataFrame] = {}
    if static_data_path is not None:
        static_table = load_static_table(static_data_path)
        logger.info(
            "Loaded %d static row(s) covering %d subject(s) from %s",
            static_table.height,
            static_table["subject_id"].n_unique(),
            static_data_path,
        )
        sid_to_shard = _build_subject_shard_map([src for src, _ in shards])
        statics_per_shard = _route_statics_to_shards(static_table, sid_to_shard)

    cumulative_mapping: list[pl.DataFrame] = []
    n_rows_in_total = 0
    n_rows_out_total = 0
    n_static_rows_total = 0
    for src, rel in shards:
        events = pl.read_parquet(src)
        n_in = events.height
        rewritten, mapping = reprocess_shard(events)

        statics = statics_per_shard.get(src)
        if statics is not None and statics.height > 0:
            statics_for_shard = statics.with_columns(
                pl.lit(None, dtype=rewritten.schema.get("time", pl.Datetime("us"))).alias("time")
            )
            rewritten = pl.concat([rewritten, statics_for_shard], how="diagonal_relaxed")
            n_static_rows_total += statics.height

        n_out = rewritten.height
        n_rows_in_total += n_in
        n_rows_out_total += n_out

        dst = output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        rewritten.write_parquet(dst)
        if mapping.height > 0:
            cumulative_mapping.append(mapping)

        n_static = statics.height if statics is not None else 0
        logger.info(
            "  %s: %d → %d rows (recombined %+d, +%d static, %d unique mappings)",
            rel,
            n_in,
            n_out,
            (n_out - n_in - n_static),
            n_static,
            mapping.height,
        )

    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    if cumulative_mapping:
        full_mapping = (
            pl.concat(cumulative_mapping, how="vertical")
            .unique()
            .sort(["family", "input_code", "output_code"])
        )
    else:
        full_mapping = _empty_mapping()
    mapping_path = metadata_dir / "ethos_code_mapping.parquet"
    full_mapping.write_parquet(mapping_path)
    logger.info("Wrote %d unique code mappings to %s", full_mapping.height, mapping_path)

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
        "Total: %d → %d rows across %d shard(s); recombined %+d, +%d static, "
        "%d unique input→output code rewrites.",
        n_rows_in_total,
        n_rows_out_total,
        len(shards),
        (n_rows_out_total - n_rows_in_total - n_static_rows_total),
        n_static_rows_total,
        full_mapping.height,
    )
    return output_dir


def _rewrite_codes_metadata(src: Path, dst: Path, mapping: pl.DataFrame) -> None:
    """Rewrite ``codes.parquet`` to use combined codes; uncovered rows pass through."""
    codes = pl.read_parquet(src)
    if "code" not in codes.columns:
        logger.warning("codes.parquet at %s has no ``code`` column; passing through unchanged.", src)
        shutil.copy2(src, dst)
        return

    rewritten = codes.join(
        mapping.select(["input_code", "output_code"]).rename({"input_code": "code"}),
        on="code",
        how="left",
    ).with_columns(pl.coalesce(["output_code", "code"]).alias("code"))
    rewritten = rewritten.drop("output_code")

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
    """Hydra entry point for ``EQ_reprocess_ethos``."""
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s")

    input_dir = Path(str(cfg.input_dir)).expanduser().resolve()
    output_dir = Path(str(cfg.output_dir)).expanduser().resolve()
    overwrite = bool(cfg.get("overwrite", False))

    static_data_path: Path | None = None
    if cfg.get("static_data_path") is not None:
        static_data_path = Path(str(cfg.static_data_path)).expanduser().resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"input_dir {input_dir!s} does not exist or is not a directory.")
    if input_dir == output_dir:
        raise ValueError("input_dir and output_dir must differ — refusing to rewrite in place.")

    reprocess_directory(
        input_dir,
        output_dir,
        overwrite=overwrite,
        static_data_path=static_data_path,
    )


if __name__ == "__main__":
    main()
