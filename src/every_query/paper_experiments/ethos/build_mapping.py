"""Build the EQ -> ETHOS code mapping table (Phase 1).

For every EQ query in ``eval_aucs_held_out.parquet`` this script emits zero or
more mapping rows, one per (tier, ETHOS token pattern) the query can be
matched against:

* ``exact`` -- EQ code string is a literal ETHOS token.
* ``drop_bin`` -- for value-binned families (LAB / INFUSION_START / INFUSION_END
  / SUBJECT_FLUID_OUTPUT) strip the trailing ``//value_[..)`` segment, upper-case
  the units, match the resulting prefix as a literal ETHOS token (any value
  decile counts as a hit).
* ``quantile`` -- same prefix as ``drop_bin``, additionally constrained so the
  next ETHOS token must be the ``Qk`` decile that EQ's value bin corresponds to.
  Decile boundaries come from ``meds_codes.parquet`` ``values/quantiles``;
  empirically EQ's bins ARE the deciles, so the mapping is the bin's positional
  index 1..10 -> Q1..Q10.
* ``union`` -- per-code OR of all patterns from prior tiers.

Outputs ``mapping_table.parquet`` and ``mapping_coverage.parquet`` and uploads
them to ``gs://every-query-runs/baselines/ethos/``.
"""

from __future__ import annotations

import io
import os
import re
import time
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml
from google.cloud import storage

BUCKET = "every-query-runs"
EQ_AUCS_BLOB = "eval_aucs_held_out.parquet"
MEDS_CODES_BLOB = "meds/MIMIC_MEDS/MEDS_cohort/processed/metadata/codes.parquet"
ETHOS_TRAJ_PREFIX = "ethos/trajectories/7573f855c4b050a9d79d57fefd8a139c/"
ETHOS_VOCAB_BLOB = "baselines/ethos/ethos_vocab.parquet"
MAPPING_TABLE_BLOB = "baselines/ethos/mapping_table.parquet"
MAPPING_COVERAGE_BLOB = "baselines/ethos/mapping_coverage.parquet"

VALUE_BIN_FAMILIES = ("LAB", "INFUSION_START", "INFUSION_END", "SUBJECT_FLUID_OUTPUT")
VALUE_BIN_RE = re.compile(r"//value_\[[^)]*\)$")

CACHE_DIR = Path(os.environ.get("ETHOS_MAPPING_CACHE", "/tmp/eq_ethos_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CROSSWALK_DIR = Path(__file__).parent / "crosswalks"
ICD_CROSSWALK_PATH = CROSSWALK_DIR / "icd.yaml"
ATC_CROSSWALK_PATH = CROSSWALK_DIR / "atc.yaml"
MIMIC_ITEMS_CROSSWALK_PATH = CROSSWALK_DIR / "mimic_items.yaml"


def _client() -> storage.Client:
    return storage.Client(project="every-query")


def _download(blob_name: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    bucket = _client().bucket(BUCKET)
    bucket.blob(blob_name).download_to_filename(str(dest))
    return dest


def _upload(local: Path, blob_name: str) -> None:
    bucket = _client().bucket(BUCKET)
    bucket.blob(blob_name).upload_from_filename(str(local))


def _list(prefix: str) -> list[str]:
    return [b.name for b in _client().list_blobs(BUCKET, prefix=prefix)]


def build_ethos_vocab(force: bool = False) -> pl.DataFrame:
    """Stream the 20 flat ETHOS trajectory parquets and accumulate a vocab.

    Caches to ``gs://every-query-runs/baselines/ethos/ethos_vocab.parquet``.
    """
    cache = CACHE_DIR / "ethos_vocab.parquet"
    if not force and cache.exists():
        return pl.read_parquet(cache)

    bucket = _client().bucket(BUCKET)
    try:
        bucket.blob(ETHOS_VOCAB_BLOB).reload()
        bucket.blob(ETHOS_VOCAB_BLOB).download_to_filename(cache)
        return pl.read_parquet(cache)
    except Exception:
        pass

    counts: dict[str, int] = {}
    flat_files = sorted(
        n for n in _list(ETHOS_TRAJ_PREFIX) if n.count("/") == ETHOS_TRAJ_PREFIX.count("/")
    )
    t0 = time.time()
    for i, name in enumerate(flat_files):
        data = bucket.blob(name).download_as_bytes()
        col = pq.read_table(pa.BufferReader(data), columns=["code"]).column("code").combine_chunks()
        for entry in pc.value_counts(col).to_pylist():
            counts[entry["values"]] = counts.get(entry["values"], 0) + entry["counts"]
        print(f"  vocab [{i+1}/{len(flat_files)}] {len(counts)} tokens cumulative t={time.time()-t0:.1f}s")

    df = (
        pl.DataFrame({"token": list(counts.keys()), "count": list(counts.values())})
        .sort("count", descending=True)
    )
    df.write_parquet(cache)
    _upload(cache, ETHOS_VOCAB_BLOB)
    return df


def load_eq_codes() -> list[str]:
    p = _download(EQ_AUCS_BLOB, CACHE_DIR / "eval_aucs_held_out.parquet")
    return sorted(pl.read_parquet(p)["code"].unique().to_list())


def load_meds_quantiles() -> dict[str, list[float]]:
    """Map non-binned base code (e.g. ``LAB//51274//sec``) -> list of 9 decile boundaries.

    Primary source: ``values/quantiles`` struct on the parent code row.
    Fallback: when the parent has no populated quantiles, derive deciles from
    the *sibling bin codes* (e.g. ``LAB//51274//sec//value_[lo,hi)``) but only
    if there are exactly 10 contiguous bins -- otherwise the EQ tokenizer used
    fewer than 10 buckets and the result wouldn't be a decile alignment.
    """
    p = _download(MEDS_CODES_BLOB, CACHE_DIR / "meds_codes.parquet")
    df = pl.read_parquet(p)
    out: dict[str, list[float]] = {}
    bins_by_base: dict[str, list[tuple[float, float]]] = {}
    for row in df.iter_rows(named=True):
        code = row["code"]
        q = row["values/quantiles"]
        if q is not None:
            deciles = [q.get(f"values/quantile/0.{i}") for i in range(1, 10)]
            if all(d is not None for d in deciles):
                out[code] = [float(d) for d in deciles]
                continue
        parsed = parse_value_bin(code)
        if parsed is not None:
            base, lo, hi = parsed
            bins_by_base.setdefault(base, []).append((lo, hi))

    for base, bins in bins_by_base.items():
        if base in out or len(bins) != 10:
            continue
        bins_sorted = sorted(bins)
        if bins_sorted[0][0] != float("-inf") or bins_sorted[-1][1] != float("inf"):
            continue
        # contiguity: each bin's hi == next bin's lo
        if not all(bins_sorted[i][1] == bins_sorted[i + 1][0] for i in range(9)):
            continue
        # deciles = the 9 internal boundaries
        out[base] = [bins_sorted[i][1] for i in range(9)]
    return out


def family_of(code: str) -> str:
    return code.split("//", 1)[0]


def has_value_bin(code: str) -> bool:
    return bool(VALUE_BIN_RE.search(code))


def parse_value_bin(code: str) -> tuple[str, float, float] | None:
    """Return (base_code_no_bin, lo, hi) or None if not a value-binned code."""
    m = VALUE_BIN_RE.search(code)
    if not m:
        return None
    base = code[: m.start()]
    inside = m.group(0)[len("//value_[") : -1]  # "lo,hi"
    lo_s, hi_s = inside.split(",")
    lo = float("-inf") if lo_s == "-inf" else float(lo_s)
    hi = float("inf") if hi_s == "inf" else float(hi_s)
    return base, lo, hi


def upper_units(base_code: str) -> str:
    """Upper-case the trailing units segment to match ETHOS lab token convention.

    ``LAB//51274//sec``  -> ``LAB//51274//SEC``
    ``LAB//50912//mg/dL`` -> ``LAB//50912//MG/DL``
    ``INFUSION_START//220862`` -> unchanged (no units segment).
    """
    parts = base_code.split("//")
    fam = parts[0]
    if fam == "LAB" and len(parts) == 3:
        parts[2] = parts[2].upper()
        return "//".join(parts)
    if fam == "SUBJECT_FLUID_OUTPUT" and len(parts) == 3:
        parts[2] = parts[2].upper()
        return "//".join(parts)
    return base_code


def build_exact(eq_codes: list[str], ethos_vocab: set[str]) -> pl.DataFrame:
    rows = [
        {
            "query": c,
            "tier": "exact",
            "token_pattern": c,
            "match_kind": "literal",
            "provenance": "exact-match",
            "mapping_source": "code:string_equality",
        }
        for c in eq_codes
        if c in ethos_vocab
    ]
    return pl.DataFrame(rows, schema=_schema())


def build_drop_bin(eq_codes: list[str], ethos_vocab: set[str]) -> pl.DataFrame:
    rows: list[dict] = []
    for c in eq_codes:
        if family_of(c) not in VALUE_BIN_FAMILIES:
            continue
        parsed = parse_value_bin(c)
        if parsed is None:
            continue
        base, _, _ = parsed
        token = upper_units(base)
        if token in ethos_vocab:
            rows.append(
                {
                    "query": c,
                    "tier": "drop_bin",
                    "token_pattern": token,
                    "match_kind": "literal",
                    "provenance": "drop-bin/upper-units",
                    "mapping_source": "code:strip_value_bin+upper_units",
                }
            )
    return pl.DataFrame(rows, schema=_schema())


def _decile_index(lo: float, hi: float, deciles: list[float]) -> int | None:
    """Map an EQ value bin ``[lo, hi)`` to a 1..10 decile index.

    EQ's bin boundaries match MEDS deciles directly, so we look for the
    bin position ``i`` such that ``lo == deciles[i-1]`` (or ``-inf`` for i=1)
    and ``hi == deciles[i]`` (or ``+inf`` for i=10). Tolerant comparison.
    """
    boundaries = [float("-inf"), *deciles, float("inf")]
    for i in range(10):
        b_lo, b_hi = boundaries[i], boundaries[i + 1]
        # tolerate float drift
        lo_ok = (lo == b_lo) or (lo != float("-inf") and abs(lo - b_lo) < 1e-3 * max(1.0, abs(b_lo)))
        hi_ok = (hi == b_hi) or (hi != float("inf") and abs(hi - b_hi) < 1e-3 * max(1.0, abs(b_hi)))
        if lo_ok and hi_ok:
            return i + 1
    return None


def build_quantile(
    eq_codes: list[str],
    ethos_vocab: set[str],
    meds_quantiles: dict[str, list[float]],
) -> pl.DataFrame:
    rows: list[dict] = []
    for c in eq_codes:
        if family_of(c) not in VALUE_BIN_FAMILIES:
            continue
        parsed = parse_value_bin(c)
        if parsed is None:
            continue
        base, lo, hi = parsed
        deciles = meds_quantiles.get(base)
        if deciles is None:
            continue
        idx = _decile_index(lo, hi, deciles)
        if idx is None:
            continue
        token_lab = upper_units(base)
        token_q = f"Q{idx}"
        if token_lab not in ethos_vocab or token_q not in ethos_vocab:
            continue
        rows.append(
            {
                "query": c,
                "tier": "quantile",
                "token_pattern": f"{token_lab}|{token_q}",
                "match_kind": "lab+next_qk",
                "provenance": f"deciles/idx={idx}",
                "mapping_source": "meds-codes.parquet:values_quantiles_or_sibling_bins",
            }
        )
    return pl.DataFrame(rows, schema=_schema())


def _load_crosswalk(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        data = yaml.safe_load(f)
    return data.get("mappings", [])


def build_crosswalk(
    eq_codes: list[str],
    ethos_vocab: set[str],
    crosswalk_path: Path,
    tier_name: str,
) -> pl.DataFrame:
    """Emit literal-match rows for each (eq_code, ethos_token) pair declared in the crosswalk YAML.

    If a YAML entry's ``eq_code`` does NOT carry a ``//value_[..)`` segment,
    it acts as a prefix and applies to every EQ code that starts with
    ``eq_code + "//value_["``. This lets one YAML entry cover all value bins
    of a given lab/infusion item-id without enumerating each bin.

    Skips token patterns that aren't actually in the ETHOS vocab (so a stale
    crosswalk entry is silently dropped rather than producing predictions
    that always miss).
    """
    eq_set = set(eq_codes)
    rows: list[dict] = []
    for entry in _load_crosswalk(crosswalk_path):
        eq_pattern = entry["eq_code"]
        targets: list[str]
        if eq_pattern in eq_set:
            targets = [eq_pattern]
        elif VALUE_BIN_RE.search(eq_pattern):
            # YAML had a value-bin we don't have in EQ -- skip.
            continue
        else:
            # Treat eq_pattern as a prefix; match any binned EQ code that starts with it.
            prefix = eq_pattern + "//value_["
            targets = [c for c in eq_codes if c.startswith(prefix)]
            if not targets:
                continue
        entry_source = entry.get("source", f"unspecified:{crosswalk_path.name}")
        for tok in entry.get("ethos_tokens", []):
            if tok not in ethos_vocab:
                continue
            for eq in targets:
                rows.append(
                    {
                        "query": eq,
                        "tier": tier_name,
                        "token_pattern": tok,
                        "match_kind": "literal",
                        "provenance": f"{tier_name}/{crosswalk_path.name}",
                        "mapping_source": entry_source,
                    }
                )
    return pl.DataFrame(rows, schema=_schema())


def build_union(prior: pl.DataFrame) -> pl.DataFrame:
    if prior.is_empty():
        return pl.DataFrame(schema=_schema())
    return prior.with_columns(
        pl.lit("union").alias("tier"),
        ("from-" + pl.col("tier")).alias("provenance"),
        # mapping_source is preserved from the underlying primary-tier row
    ).unique(subset=["query", "token_pattern", "match_kind"])


def _schema() -> dict[str, type]:
    return {
        "query": pl.String,
        "tier": pl.String,
        "token_pattern": pl.String,
        "match_kind": pl.String,
        "provenance": pl.String,
        "mapping_source": pl.String,
    }


def build_coverage(eq_codes: list[str], mapping: pl.DataFrame) -> pl.DataFrame:
    """Per-code coverage report with `unmapped_reason` annotations."""

    def reason(code: str) -> str:
        fam = family_of(code)
        if fam == "DIAGNOSIS":
            return "ethos_token_missing/diagnosis_uses_descriptive_label_not_icd_code"
        if fam == "MEDICATION":
            return "drug_name_vs_atc"
        if fam in ("INFUSION_START", "INFUSION_END"):
            return "ethos_token_missing/no_infusion_prefix_in_vocab"
        if fam == "SUBJECT_FLUID_OUTPUT":
            return "ethos_token_missing/no_fluid_output_prefix_in_vocab"
        if fam == "PROCEDURE":
            return "ethos_token_missing/procedure_uses_pcs_chunked_form_not_icd_code"
        if fam == "TIMELINE":
            return "ethos_token_missing/no_timeline_start_token_in_vocab"
        if fam == "LAB":
            return "value_bin_or_units_unmatched"
        return "other"

    by_query = (
        mapping.group_by("query")
        .agg(pl.col("tier").unique().sort().alias("mapped_under_tiers"))
    )
    full = (
        pl.DataFrame({"query": eq_codes})
        .join(by_query, on="query", how="left")
        .with_columns(
            pl.col("query").map_elements(family_of, return_dtype=pl.String).alias("family"),
            pl.col("mapped_under_tiers").fill_null([]),
        )
        .with_columns(
            pl.col("mapped_under_tiers").list.len().eq(0).alias("is_unmapped"),
        )
        .with_columns(
            pl.when(pl.col("is_unmapped"))
            .then(pl.col("query").map_elements(reason, return_dtype=pl.String))
            .otherwise(pl.lit(""))
            .alias("unmapped_reason"),
        )
        .select("query", "family", "mapped_under_tiers", "is_unmapped", "unmapped_reason")
        .sort(["is_unmapped", "family", "query"])
    )
    return full


def main(upload: bool = True) -> None:
    """Build the mapping artifacts and (by default) upload them to GCS.

    Set ``upload=False`` to write the local parquet artifacts under
    ``CACHE_DIR`` without pushing to ``gs://every-query-runs/baselines/ethos/``.
    Useful when iterating on YAML crosswalks before locking the mapping in.
    """
    print("Loading EQ codes ...")
    eq_codes = load_eq_codes()
    print(f"  {len(eq_codes)} unique EQ queries")

    print("Loading MEDS deciles ...")
    meds_q = load_meds_quantiles()
    print(f"  {len(meds_q)} codes have decile metadata")

    print("Building / loading ETHOS vocab ...")
    vocab_df = build_ethos_vocab()
    ethos_vocab = set(vocab_df["token"].to_list())
    print(f"  {len(ethos_vocab)} ETHOS tokens")

    print("Building tiers ...")
    exact = build_exact(eq_codes, ethos_vocab)
    drop_bin = build_drop_bin(eq_codes, ethos_vocab)
    quantile = build_quantile(eq_codes, ethos_vocab, meds_q)
    icd_xw = build_crosswalk(eq_codes, ethos_vocab, ICD_CROSSWALK_PATH, "icd_crosswalk")
    atc_xw = build_crosswalk(eq_codes, ethos_vocab, ATC_CROSSWALK_PATH, "atc_crosswalk")
    mimic_xw = build_crosswalk(
        eq_codes, ethos_vocab, MIMIC_ITEMS_CROSSWALK_PATH, "mimic_item_crosswalk"
    )
    primary = pl.concat([exact, drop_bin, quantile, icd_xw, atc_xw, mimic_xw], how="vertical")
    union = build_union(primary)
    mapping = pl.concat([primary, union], how="vertical")

    coverage = build_coverage(eq_codes, mapping)

    out_table = CACHE_DIR / "mapping_table.parquet"
    out_cov = CACHE_DIR / "mapping_coverage.parquet"
    mapping.write_parquet(out_table)
    coverage.write_parquet(out_cov)

    print()
    print("Per-tier row counts:")
    print(mapping.group_by("tier").len().sort("tier"))
    print()
    print("Coverage by family:")
    print(
        coverage.group_by("family")
        .agg(
            pl.len().alias("n_codes"),
            pl.col("is_unmapped").sum().alias("n_unmapped"),
        )
        .sort("family")
    )
    print()
    print("Unmapped codes:")
    print(coverage.filter(pl.col("is_unmapped")).select("family", "query", "unmapped_reason"))

    print()
    if upload:
        print("Uploading to GCS ...")
        _upload(out_table, MAPPING_TABLE_BLOB)
        _upload(out_cov, MAPPING_COVERAGE_BLOB)
        print(f"  gs://{BUCKET}/{MAPPING_TABLE_BLOB}")
        print(f"  gs://{BUCKET}/{MAPPING_COVERAGE_BLOB}")
    else:
        print("Skipping GCS upload (upload=False).")
        print(f"  local: {out_table}")
        print(f"  local: {out_cov}")


if __name__ == "__main__":
    main()
