"""Ontology bridges for ETHOS <-> EQ mapping review.

This module is consumed by ``build_review.py`` to resolve, for every EQ code
and every candidate ETHOS token, an authoritative concept name and a
constituent-code expansion. Each lookup is sourced from the OHDSI Athena
bundle staged under ``data/vocab/`` (SNOMED, ICD9CM, ICD9Proc, ICD10CM,
ICD10PCS, LOINC, RxNorm, ATC) plus the MIMIC-IV item dictionaries staged
under ``data/mimic/``.

ETHOS descriptive ICD token source-of-truth
-------------------------------------------

ETHOS encodes diagnoses with tokens of the form ``ICD//CM//<LABEL>`` where
``<LABEL>`` is an upper-snake-case description rather than a numeric code
(e.g. ``ICD//CM//BRONCHIECTASIS``, ``ICD//CM//NAUSEA_AND_VOMITING``). Phase 1
of the mapping review needs to know which source vocabulary these labels
were drawn from so we can resolve them back to a set of constituent ICD-10-CM
codes for review.

Methodology: 20 distinct descriptive ``ICD//CM//*`` tokens were drawn from
``ethos_vocab.parquet`` (excluding the numeric ``ICD//CM//3-6//*`` and
``ICD//CM//SFX//*`` token families) using ``random.seed(42)``. Each was
normalized two ways and looked up against ``CONCEPT.csv`` rows with
``vocabulary_id`` in ``{ICD10CM, SNOMED}``:

* ``strict`` -- replace ``_`` with space, upper-case. Hit rate vs any
  ICD10CM concept name: 16/20 = 80%. Hit rate vs SNOMED: 6/20 = 30%.
* ``loose`` -- replace any run of non-alphanumeric characters with a single
  space, upper-case (drops commas, hyphens, parentheses, apostrophes).
  Hit rate vs ICD10CM: 20/20 = 100%, all matching ``concept_class_id`` in
  ``{3-char nonbill code, 3-char billing code}``. Hit rate vs SNOMED: 6/20 = 30%.

Conclusion: the descriptive ETHOS ICD tokens are the **ICD-10-CM 3-character
category names** with punctuation stripped and spaces replaced by
underscores. The four strict-normalization misses
(``ASSAULT_BY_RIFLE_SHOTGUN_AND_LARGER_FIREARM_DISCHARGE``,
``SUPERFICIAL_INJURY_OF_ABDOMEN_LOWER_BACK_PELVIS_AND_EXTERNAL_GENITALS``,
``POISONING_BY_ADVERSE_EFFECT_OF_AND_UNDERDOSING_OF_NONOPIOID_ANALGESICS_ANTIPYRETICS_AND_ANTIRHEUMATICS``,
``SHOCK_NOT_ELSEWHERE_CLASSIFIED``) all match an ICD-10-CM 3-char category
once commas and the equivalent of `, ` separators are stripped (e.g. the
official X94 name is ``Assault by rifle, shotgun and larger firearm
discharge``).

Implication for ``ethos_descriptive_icd_to_codes``: normalize both the
ETHOS label and every ICD-10-CM 3-char category name with the loose rule
(non-alphanumeric -> single space, upper-case), build a name-to-3-char-code
map, then enumerate descendants via ``CONCEPT_ANCESTOR.csv`` to get the
full list of constituent ICD-10-CM codes the ETHOS token represents.

Threshold check: the plan requires aborting Phase 1 if the ICD10CM hit rate
falls below 75%. With strict normalization we observe 80% and with loose
normalization (the rule we will actually use) we observe 100%, so Phase 1
proceeds.
"""

from __future__ import annotations

import hashlib
import os
import re
from functools import lru_cache
from pathlib import Path

import polars as pl

_HERE = Path(__file__).parent
VOCAB_DIR = _HERE / "data" / "vocab"
CACHE_DIR = _HERE / "data" / "cache"
MIMIC_DIR = _HERE / "data" / "mimic"

CONCEPT_CSV = VOCAB_DIR / "CONCEPT.csv"
CONCEPT_RELATIONSHIP_CSV = VOCAB_DIR / "CONCEPT_RELATIONSHIP.csv"
CONCEPT_ANCESTOR_CSV = VOCAB_DIR / "CONCEPT_ANCESTOR.csv"

D_ITEMS_CSV = MIMIC_DIR / "icu" / "d_items.csv"
D_LABITEMS_CSV = MIMIC_DIR / "hosp" / "d_labitems.csv"

# OHDSI Athena CSVs are tab-separated despite the .csv extension.
_CSV_KW: dict = {
    "separator": "\t",
    "quote_char": None,
    "infer_schema_length": 0,  # treat all columns as Utf8 to avoid mis-parses
    "null_values": [""],
}


def _hash_key(parts: tuple[str, ...]) -> str:
    """Stable short hash for cache filenames; order-independent."""
    canonical = "|".join(sorted(parts))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_concept(vocab_ids: tuple[str, ...]) -> pl.DataFrame:
    """Load ``CONCEPT.csv`` filtered to the requested ``vocabulary_id`` set.

    Output columns are typed: ``concept_id`` is cast to Int64, all other
    columns remain Utf8 (date/invalid_reason fields are not needed downstream
    and casting them would just trip on empty strings).

    Caches the filtered slice as parquet at
    ``data/cache/concept__<hash>.parquet`` keyed on the (sorted) vocab_ids.
    """
    if not vocab_ids:
        raise ValueError("vocab_ids must not be empty")

    _ensure_cache_dir()
    cache_path = CACHE_DIR / f"concept__{_hash_key(tuple(vocab_ids))}.parquet"
    if cache_path.exists():
        return pl.read_parquet(cache_path)

    df = (
        pl.scan_csv(CONCEPT_CSV, **_CSV_KW)
        .filter(pl.col("vocabulary_id").is_in(list(vocab_ids)))
        .with_columns(pl.col("concept_id").cast(pl.Int64))
        .collect()
    )
    df.write_parquet(cache_path)
    return df


def load_concept_relationship(relationship_ids: tuple[str, ...]) -> pl.DataFrame:
    """Load ``CONCEPT_RELATIONSHIP.csv`` filtered to the requested
    ``relationship_id`` set (e.g. ``("Maps to",)``, ``("Maps to", "Mapped from")``).

    ``concept_id_1`` and ``concept_id_2`` are cast to Int64. Caches the
    filtered slice as parquet at
    ``data/cache/concept_relationship__<hash>.parquet`` keyed on the
    (sorted) relationship_ids.
    """
    if not relationship_ids:
        raise ValueError("relationship_ids must not be empty")

    _ensure_cache_dir()
    cache_path = (
        CACHE_DIR
        / f"concept_relationship__{_hash_key(tuple(relationship_ids))}.parquet"
    )
    if cache_path.exists():
        return pl.read_parquet(cache_path)

    df = (
        pl.scan_csv(CONCEPT_RELATIONSHIP_CSV, **_CSV_KW)
        .filter(pl.col("relationship_id").is_in(list(relationship_ids)))
        .with_columns(
            pl.col("concept_id_1").cast(pl.Int64),
            pl.col("concept_id_2").cast(pl.Int64),
        )
        .collect()
    )
    df.write_parquet(cache_path)
    return df


def load_concept_ancestor(
    descendant_concept_ids: list[int] | None = None,
) -> pl.DataFrame:
    """Load ``CONCEPT_ANCESTOR.csv``, optionally filtered to a set of
    descendant concept ids.

    All four columns (``ancestor_concept_id``, ``descendant_concept_id``,
    ``min_levels_of_separation``, ``max_levels_of_separation``) are cast to
    Int64.

    Caches as parquet at ``data/cache/concept_ancestor__<hash>.parquet``. The
    hash key is ``"all"`` when ``descendant_concept_ids`` is None, otherwise
    the (sorted) descendant id list. The unfiltered cache is reused
    transparently to filter in-memory when a callsite later requests a
    descendant subset.
    """
    _ensure_cache_dir()

    if descendant_concept_ids is None:
        cache_path = CACHE_DIR / "concept_ancestor__all.parquet"
        if cache_path.exists():
            return pl.read_parquet(cache_path)
        df = (
            pl.scan_csv(CONCEPT_ANCESTOR_CSV, **_CSV_KW)
            .with_columns(
                pl.col("ancestor_concept_id").cast(pl.Int64),
                pl.col("descendant_concept_id").cast(pl.Int64),
                pl.col("min_levels_of_separation").cast(pl.Int64),
                pl.col("max_levels_of_separation").cast(pl.Int64),
            )
            .collect()
        )
        df.write_parquet(cache_path)
        return df

    descendants = sorted({int(c) for c in descendant_concept_ids})
    if not descendants:
        raise ValueError("descendant_concept_ids must be non-empty when provided")

    key = _hash_key(tuple(str(c) for c in descendants))
    cache_path = CACHE_DIR / f"concept_ancestor__{key}.parquet"
    if cache_path.exists():
        return pl.read_parquet(cache_path)

    full_cache = CACHE_DIR / "concept_ancestor__all.parquet"
    if full_cache.exists():
        df = pl.read_parquet(full_cache).filter(
            pl.col("descendant_concept_id").is_in(descendants)
        )
    else:
        df = (
            pl.scan_csv(CONCEPT_ANCESTOR_CSV, **_CSV_KW)
            .with_columns(
                pl.col("ancestor_concept_id").cast(pl.Int64),
                pl.col("descendant_concept_id").cast(pl.Int64),
                pl.col("min_levels_of_separation").cast(pl.Int64),
                pl.col("max_levels_of_separation").cast(pl.Int64),
            )
            .filter(pl.col("descendant_concept_id").is_in(descendants))
            .collect()
        )
    df.write_parquet(cache_path)
    return df


def _icd_eq_to_dotted_candidates(code: str) -> list[str]:
    """EQ encodes ICD codes without the decimal point (e.g. ``3320`` for
    Parkinson's, ``V503`` for ear piercing). OHDSI ``CONCEPT.concept_code``
    stores the dotted form. To bridge, we match on the dot-stripped form,
    so we only need the EQ code itself plus its dot-stripped lookup key.
    """
    return [code, code.replace(".", "")]


def icd9_dx(code: str) -> dict | None:
    """Resolve an ICD-9-CM diagnosis code (EQ format, no dot) into:

    * the source ICD9CM concept,
    * the SNOMED concept(s) it ``Maps to`` (standard concept bridge),
    * the set of ICD-10-CM codes that ``Maps to`` the same SNOMED target(s),
      i.e. the ICD-9-CM -> SNOMED -> ICD-10-CM crosswalk via Athena.

    Returns ``None`` if the source ICD-9-CM code is not present in the
    Athena bundle. Returns the dict with empty SNOMED/ICD-10-CM fields if
    the source resolves but no ``Maps to`` bridge exists.
    """
    code = code.strip()
    if not code:
        return None

    concept = load_concept(("ICD9CM",))
    src = concept.with_columns(
        pl.col("concept_code").str.replace(r"\.", "", literal=False).alias("_norm")
    ).filter(
        (pl.col("_norm") == code)
        & (pl.col("domain_id") == "Condition")
    )
    if src.is_empty():
        return None
    src_row = src.row(0, named=True)
    src_id = int(src_row["concept_id"])
    src_name = src_row["concept_name"]

    rels = load_concept_relationship(("Maps to",))
    snomed_targets = (
        rels.filter(pl.col("concept_id_1") == src_id)
        .select("concept_id_2")
        .unique()
    )
    snomed_ids = [int(x) for x in snomed_targets["concept_id_2"].to_list()]

    snomed_concepts = load_concept(("SNOMED",)).filter(
        pl.col("concept_id").is_in(snomed_ids)
    )
    snomed_name_by_id = {
        int(r["concept_id"]): r["concept_name"]
        for r in snomed_concepts.iter_rows(named=True)
    }

    snomed_target_concept_id: int | None = None
    snomed_target_name: str | None = None
    if snomed_name_by_id:
        snomed_target_concept_id = next(iter(snomed_name_by_id))
        snomed_target_name = snomed_name_by_id[snomed_target_concept_id]

    icd10_codes: list[str] = []
    icd10_targets: list[dict] = []
    if snomed_ids:
        icd10_concepts = load_concept(("ICD10CM",))
        icd10_id_set = set(
            int(c) for c in icd10_concepts["concept_id"].to_list()
        )
        bridge = rels.filter(
            pl.col("concept_id_2").is_in(snomed_ids)
            & pl.col("concept_id_1").is_in(list(icd10_id_set))
        )
        icd10_ids = [int(x) for x in bridge["concept_id_1"].unique().to_list()]
        if icd10_ids:
            icd10_rows = icd10_concepts.filter(
                pl.col("concept_id").is_in(icd10_ids)
            ).sort("concept_code")
            icd10_codes = icd10_rows["concept_code"].to_list()
            icd10_targets = [
                {"code": r["concept_code"], "name": r["concept_name"]}
                for r in icd10_rows.iter_rows(named=True)
            ]

    return {
        "concept_id": src_id,
        "concept_name": src_name,
        "snomed_target_concept_id": snomed_target_concept_id,
        "snomed_target_name": snomed_target_name,
        "icd10_target_codes": icd10_codes,
        "icd10_targets": icd10_targets,
    }


def icd9_proc(code: str) -> dict | None:
    """Resolve an ICD-9-Proc code (EQ format, no dot) into:

    * the source ICD9Proc concept,
    * the SNOMED concept(s) it ``Maps to``,
    * the set of ICD-10-PCS codes that ``Maps to`` the same SNOMED target(s).

    Returns ``None`` if the source ICD-9-Proc code is not present in the
    Athena bundle.
    """
    code = code.strip()
    if not code:
        return None

    concept = load_concept(("ICD9Proc",))
    src = concept.with_columns(
        pl.col("concept_code").str.replace(r"\.", "", literal=False).alias("_norm")
    ).filter(pl.col("_norm") == code)
    if src.is_empty():
        return None
    src_row = src.row(0, named=True)
    src_id = int(src_row["concept_id"])
    src_name = src_row["concept_name"]

    rels = load_concept_relationship(("Maps to",))
    snomed_targets = (
        rels.filter(pl.col("concept_id_1") == src_id)
        .select("concept_id_2")
        .unique()
    )
    snomed_ids = [int(x) for x in snomed_targets["concept_id_2"].to_list()]

    snomed_concepts = load_concept(("SNOMED",)).filter(
        pl.col("concept_id").is_in(snomed_ids)
    )
    snomed_name_by_id = {
        int(r["concept_id"]): r["concept_name"]
        for r in snomed_concepts.iter_rows(named=True)
    }

    snomed_target_concept_id: int | None = None
    snomed_target_name: str | None = None
    if snomed_name_by_id:
        snomed_target_concept_id = next(iter(snomed_name_by_id))
        snomed_target_name = snomed_name_by_id[snomed_target_concept_id]

    pcs_codes: list[str] = []
    pcs_targets: list[dict] = []
    if snomed_ids:
        pcs_concepts = load_concept(("ICD10PCS",))
        pcs_id_set = set(int(c) for c in pcs_concepts["concept_id"].to_list())
        bridge = rels.filter(
            pl.col("concept_id_2").is_in(snomed_ids)
            & pl.col("concept_id_1").is_in(list(pcs_id_set))
        )
        pcs_ids = [int(x) for x in bridge["concept_id_1"].unique().to_list()]
        if pcs_ids:
            pcs_rows = pcs_concepts.filter(
                pl.col("concept_id").is_in(pcs_ids)
            ).sort("concept_code")
            pcs_codes = pcs_rows["concept_code"].to_list()
            pcs_targets = [
                {"code": r["concept_code"], "name": r["concept_name"]}
                for r in pcs_rows.iter_rows(named=True)
            ]

    return {
        "concept_id": src_id,
        "concept_name": src_name,
        "snomed_target_concept_id": snomed_target_concept_id,
        "snomed_target_name": snomed_target_name,
        "icd10pcs_target_codes": pcs_codes,
        "icd10pcs_targets": pcs_targets,
    }


def icd10cm(code: str) -> dict | None:
    """Resolve an ICD-10-CM code (EQ format, no dot) into:

    * the source ICD10CM concept,
    * its 3-character parent category code and name,
    * the SNOMED concept(s) it ``Maps to``.

    Returns ``None`` if the source ICD-10-CM code is not present in the
    Athena bundle.
    """
    code = code.strip()
    if not code:
        return None

    concept = load_concept(("ICD10CM",))
    src = concept.with_columns(
        pl.col("concept_code").str.replace(r"\.", "", literal=False).alias("_norm")
    ).filter(pl.col("_norm") == code)
    if src.is_empty():
        return None
    src_row = src.row(0, named=True)
    src_id = int(src_row["concept_id"])
    src_name = src_row["concept_name"]
    src_dotted = src_row["concept_code"]

    parent_3char_code = src_dotted.split(".")[0][:3]
    parent_match = concept.filter(pl.col("concept_code") == parent_3char_code)
    parent_3char_name: str | None = None
    if not parent_match.is_empty():
        parent_3char_name = parent_match.row(0, named=True)["concept_name"]

    rels = load_concept_relationship(("Maps to",))
    snomed_targets = (
        rels.filter(pl.col("concept_id_1") == src_id)
        .select("concept_id_2")
        .unique()
    )
    snomed_ids = [int(x) for x in snomed_targets["concept_id_2"].to_list()]

    snomed_target: dict | None = None
    if snomed_ids:
        snomed_concepts = load_concept(("SNOMED",)).filter(
            pl.col("concept_id").is_in(snomed_ids)
        )
        if not snomed_concepts.is_empty():
            r = snomed_concepts.row(0, named=True)
            snomed_target = {
                "concept_id": int(r["concept_id"]),
                "concept_name": r["concept_name"],
            }

    return {
        "concept_id": src_id,
        "concept_name": src_name,
        "parent_3char_code": parent_3char_code,
        "parent_3char_name": parent_3char_name,
        "snomed_target": snomed_target,
    }


_NONALNUM_RE = re.compile(r"[^A-Za-z0-9]+")


def _loose_normalize(text: str) -> str:
    """Loose normalization rule (see module docstring): collapse runs of
    non-alphanumeric chars to a single space, strip, upper-case.
    """
    return _NONALNUM_RE.sub(" ", text).strip().upper()


@lru_cache(maxsize=1)
def _icd10cm_3char_name_index() -> dict[str, tuple[int, str, str]]:
    """Map loose-normalized ICD-10-CM 3-char category name -> (concept_id,
    raw concept_code, raw concept_name).
    """
    concept = load_concept(("ICD10CM",))
    three_char = concept.filter(
        pl.col("concept_class_id").is_in(["3-char nonbill code", "3-char billing code"])
    )
    out: dict[str, tuple[int, str, str]] = {}
    for r in three_char.iter_rows(named=True):
        key = _loose_normalize(r["concept_name"])
        if key and key not in out:
            out[key] = (int(r["concept_id"]), r["concept_code"], r["concept_name"])
    return out


def ethos_descriptive_icd_to_codes(label: str) -> dict | None:
    """Resolve an ETHOS descriptive ICD label (e.g. ``PARKINSON'S_DISEASE``,
    ``BRONCHIECTASIS``) to the source-vocab ICD-10-CM 3-character category
    plus the full descendant code list via ``CONCEPT_ANCESTOR``.

    Normalization follows the module-level methodology: collapse non-alphanumeric
    runs to a single space and upper-case both sides before matching.

    Returns ``None`` if the loose-normalized label has no match in the
    ICD-10-CM 3-char category index. The returned dict has shape::

        {
            "source_vocab": "ICD10CM",
            "source_concept_id": int,
            "source_concept_name": str,
            "source_concept_code": str,  # the 3-char parent code
            "descendant_codes": list[str],
            "descendant_count": int,
        }
    """
    if label is None:
        return None
    key = _loose_normalize(label)
    if not key:
        return None

    index = _icd10cm_3char_name_index()
    hit = index.get(key)
    if hit is None:
        return None
    src_id, src_code, src_name = hit

    # ICD-10-CM is non-standard, so OHDSI CONCEPT_ANCESTOR is not populated
    # for these concepts. Enumerate descendants via concept_code prefix
    # match within the ICD10CM vocabulary itself; the dotted child codes
    # all start with the 3-char parent (e.g. G20, G20.A, G20.A1, ...).
    icd10 = load_concept(("ICD10CM",))
    descendants = (
        icd10.filter(pl.col("concept_code").str.starts_with(src_code))
        .sort("concept_code")
        .filter(pl.col("concept_code") != src_code)
    )
    descendant_codes = descendants["concept_code"].to_list()
    descendant_pairs = [
        {"code": r["concept_code"], "name": r["concept_name"]}
        for r in descendants.iter_rows(named=True)
    ]

    return {
        "source_vocab": "ICD10CM",
        "source_concept_id": src_id,
        "source_concept_name": src_name,
        "source_concept_code": src_code,
        "descendant_codes": descendant_codes,
        "descendants": descendant_pairs,
        "descendant_count": len(descendant_codes),
    }


_DRUG_SUFFIX_RE = re.compile(
    r"\b("
    r"tablet|tablets|capsule|capsules|injection|solution|suspension|syrup|"
    r"cream|ointment|patch|patches|inhaler|spray|drops|elixir|"
    r"oral|topical|nasal|ophthalmic|otic|rectal|vaginal|"
    r"mg|mcg|g|ml|l|iu|units?|"
    r"po|iv|im|sc|sl"
    r")\b",
    re.IGNORECASE,
)
_DOSE_RE = re.compile(r"\b\d+(\.\d+)?\s*(mg|mcg|g|ml|l|iu|units?|%)\b", re.IGNORECASE)


def _normalize_drug_name(name: str) -> str:
    """Strip dose, strength, and route/form suffixes from a free-text drug
    name and lower-case for case-insensitive matching against RxNorm names.
    """
    s = name or ""
    s = _DOSE_RE.sub(" ", s)
    s = _DRUG_SUFFIX_RE.sub(" ", s)
    s = re.sub(r"[\(\)\[\]\{\}]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()


def rxnorm_drug_to_atc(drug_name: str) -> dict | None:
    """Bridge a free-text drug name to ATC class 3 (pharmacological subgroup)
    and class 4 (chemical subgroup) via RxNorm ingredient.

    Strategy:

    1. Normalize ``drug_name`` (strip dose / route / form / punctuation).
    2. Match against RxNorm ``CONCEPT`` names case-insensitively, preferring
       an exact normalized match, then a substring match.
    3. Resolve to a single RxNorm ingredient by walking ``Has ingredient`` /
       ``RxNorm has ing`` relationships if the matched concept is not itself
       an ingredient.
    4. From the ingredient concept, walk ATC ancestors via
       ``CONCEPT_ANCESTOR`` (ATC vocabulary) and pick the level-3 (length 4)
       and level-4 (length 5) ATC codes.

    Returns ``None`` if the drug cannot be matched. Empty fields are filled
    with ``None`` when the bridge partially resolves (e.g. matched the
    ingredient but found no ATC ancestor).
    """
    if not drug_name:
        return None

    target = _normalize_drug_name(drug_name)
    if not target:
        return None

    rxnorm = load_concept(("RxNorm",))

    rxnorm_lc = rxnorm.with_columns(
        pl.col("concept_name").str.to_lowercase().alias("_lc")
    )

    exact = rxnorm_lc.filter(pl.col("_lc") == target)
    candidate: dict | None = None

    if not exact.is_empty():
        ingr = exact.filter(pl.col("concept_class_id") == "Ingredient")
        candidate = (ingr if not ingr.is_empty() else exact).row(0, named=True)
    else:
        ingredients = rxnorm_lc.filter(pl.col("concept_class_id") == "Ingredient")
        ingr_hit = ingredients.filter(
            pl.col("_lc").str.contains(target, literal=True)
        )
        if ingr_hit.is_empty():
            # try the reverse direction: ingredient name appears as a substring
            # of the (multi-ingredient) target. Done in Python since polars
            # column-as-pattern requires the pattern column, not the haystack.
            ingr_rows = []
            for r in ingredients.iter_rows(named=True):
                lc = r["_lc"] or ""
                if lc and lc in target:
                    ingr_rows.append(r)
            if ingr_rows:
                ingr_rows.sort(key=lambda r: len(r["concept_name"] or ""))
                candidate = ingr_rows[0]
        if candidate is None and not ingr_hit.is_empty():
            ingr_hit = ingr_hit.with_columns(
                pl.col("concept_name").str.len_chars().alias("_len")
            ).sort("_len")
            candidate = ingr_hit.row(0, named=True)
        if candidate is None:
            sub = rxnorm_lc.filter(
                pl.col("_lc").str.contains(target, literal=True)
            )
            if sub.is_empty():
                return None
            sub = sub.with_columns(
                pl.col("concept_name").str.len_chars().alias("_len")
            ).sort("_len")
            candidate = sub.row(0, named=True)

    rxnorm_concept_id = int(candidate["concept_id"])
    rxnorm_concept_name = candidate["concept_name"]
    rxnorm_class = candidate["concept_class_id"]

    ingredient_id: int | None = None
    ingredient_name: str | None = None
    if rxnorm_class == "Ingredient":
        ingredient_id = rxnorm_concept_id
        ingredient_name = rxnorm_concept_name
    else:
        rels = load_concept_relationship(("Has ingredient", "RxNorm has ing"))
        hit = rels.filter(pl.col("concept_id_1") == rxnorm_concept_id)
        ingr_ids = [int(x) for x in hit["concept_id_2"].unique().to_list()]
        if ingr_ids:
            ingr_rows = rxnorm.filter(
                pl.col("concept_id").is_in(ingr_ids)
                & (pl.col("concept_class_id") == "Ingredient")
            )
            if not ingr_rows.is_empty():
                r = ingr_rows.row(0, named=True)
                ingredient_id = int(r["concept_id"])
                ingredient_name = r["concept_name"]

    atc_class_3_code: str | None = None
    atc_class_3_name: str | None = None
    atc_class_4_code: str | None = None
    atc_class_4_name: str | None = None

    if ingredient_id is not None:
        atc = load_concept(("ATC",))
        atc_id_set = set(int(c) for c in atc["concept_id"].to_list())

        ancestor = load_concept_ancestor([ingredient_id])
        atc_ancestor_ids = [
            int(x)
            for x in ancestor["ancestor_concept_id"].unique().to_list()
            if int(x) in atc_id_set
        ]
        if atc_ancestor_ids:
            atc_rows = atc.filter(pl.col("concept_id").is_in(atc_ancestor_ids))
            for r in atc_rows.iter_rows(named=True):
                code = r["concept_code"] or ""
                if len(code) == 4 and atc_class_3_code is None:
                    atc_class_3_code = code
                    atc_class_3_name = r["concept_name"]
                elif len(code) == 5 and atc_class_4_code is None:
                    atc_class_4_code = code
                    atc_class_4_name = r["concept_name"]

    return {
        "rxnorm_concept_id": rxnorm_concept_id,
        "rxnorm_concept_name": rxnorm_concept_name,
        "rxnorm_concept_class": rxnorm_class,
        "ingredient_concept_id": ingredient_id,
        "ingredient_name": ingredient_name,
        "atc_class_3_code": atc_class_3_code,
        "atc_class_3_name": atc_class_3_name,
        "atc_class_4_code": atc_class_4_code,
        "atc_class_4_name": atc_class_4_name,
    }


_ATC_LEVEL_BY_LEN = {1: 1, 3: 2, 4: 3, 5: 4, 7: 5}


def atc_class(atc_code: str) -> dict | None:
    """Resolve an ATC class code (any level 1-5) into the concept name and
    the full list of descendant RxNorm ingredient names.

    Level is inferred from code length using the standard ATC scheme:

    * 1 char  -> level 1 (anatomical main group)
    * 3 chars -> level 2 (therapeutic subgroup)
    * 4 chars -> level 3 (pharmacological subgroup)
    * 5 chars -> level 4 (chemical subgroup)
    * 7 chars -> level 5 (chemical substance / ingredient ATC code)

    Returns ``None`` if the code is not present in the Athena ATC vocabulary.
    """
    if not atc_code:
        return None
    code = atc_code.strip().upper()

    atc = load_concept(("ATC",))
    src = atc.filter(pl.col("concept_code") == code)
    if src.is_empty():
        return None
    src_row = src.row(0, named=True)
    src_id = int(src_row["concept_id"])
    src_name = src_row["concept_name"]
    level = _ATC_LEVEL_BY_LEN.get(len(code))

    ancestor = load_concept_ancestor(None)
    desc_ids = (
        ancestor.filter(pl.col("ancestor_concept_id") == src_id)
        .select("descendant_concept_id")
        .unique()
    )
    desc_id_list = [int(x) for x in desc_ids["descendant_concept_id"].to_list()]

    ingredients: list[str] = []
    if desc_id_list:
        rxnorm = load_concept(("RxNorm",))
        ingr_rows = rxnorm.filter(
            pl.col("concept_id").is_in(desc_id_list)
            & (pl.col("concept_class_id") == "Ingredient")
        ).sort("concept_name")
        ingredients = ingr_rows["concept_name"].to_list()

    return {
        "concept_id": src_id,
        "concept_name": src_name,
        "level": level,
        "ingredients": ingredients,
    }


@lru_cache(maxsize=1)
def _load_d_items() -> pl.DataFrame:
    return pl.read_csv(
        D_ITEMS_CSV,
        infer_schema_length=0,
        null_values=[""],
    )


@lru_cache(maxsize=1)
def _load_d_labitems() -> pl.DataFrame:
    return pl.read_csv(
        D_LABITEMS_CSV,
        infer_schema_length=0,
        null_values=[""],
    )


def mimic_item_label(item_id: int) -> dict | None:
    """Resolve a MIMIC-IV item id by looking it up in ``d_items.csv`` (ICU
    chartevents/datetimeevents/inputevents/outputevents/procedureevents) and
    ``d_labitems.csv`` (hospital labevents). The first hit wins.

    Returns a dict with the union of fields available across the two
    dictionaries (missing fields are ``None``)::

        {
            "source_table": "d_items" | "d_labitems",
            "label": str,
            "abbreviation": str | None,
            "category": str | None,
            "unitname": str | None,
            "loinc_code": str | None,
            "fluid": str | None,
            "linksto": str | None,
        }

    Note: neither MIMIC-IV dictionary CSV ships with a ``loinc_code`` column,
    so ``loinc_code`` is always ``None``; the field is preserved in the
    return shape for downstream parity with the plan.
    """
    if item_id is None:
        return None
    try:
        target = str(int(item_id))
    except (TypeError, ValueError):
        return None

    items = _load_d_items()
    hit = items.filter(pl.col("itemid") == target)
    if not hit.is_empty():
        r = hit.row(0, named=True)
        return {
            "source_table": "d_items",
            "label": r.get("label"),
            "abbreviation": r.get("abbreviation"),
            "category": r.get("category"),
            "unitname": r.get("unitname"),
            "loinc_code": None,
            "fluid": None,
            "linksto": r.get("linksto"),
        }

    labitems = _load_d_labitems()
    hit = labitems.filter(pl.col("itemid") == target)
    if not hit.is_empty():
        r = hit.row(0, named=True)
        return {
            "source_table": "d_labitems",
            "label": r.get("label"),
            "abbreviation": None,
            "category": r.get("category"),
            "unitname": None,
            "loinc_code": None,
            "fluid": r.get("fluid"),
            "linksto": None,
        }

    return None


def loinc(loinc_code: str) -> dict | None:
    """Resolve a LOINC code into its concept name plus component / system
    parts and the SNOMED concept(s) it ``Maps to``.

    The OHDSI LOINC vocabulary stores the structured analyte parts via the
    ``Has component`` and ``Has system`` relationships; we resolve those to
    the named LOINC ``Component`` / ``System`` concepts.

    Returns ``None`` if the LOINC code is not present in the Athena bundle.
    """
    if not loinc_code:
        return None
    code = loinc_code.strip()

    lc = load_concept(("LOINC",))
    src = lc.filter(pl.col("concept_code") == code)
    if src.is_empty():
        return None
    src_row = src.row(0, named=True)
    src_id = int(src_row["concept_id"])
    src_name = src_row["concept_name"]

    rels = load_concept_relationship(("Maps to", "Has component", "Has system"))

    def _first_target_name(relationship_id: str) -> str | None:
        hit = rels.filter(
            (pl.col("concept_id_1") == src_id)
            & (pl.col("relationship_id") == relationship_id)
        )
        if hit.is_empty():
            return None
        target_ids = [int(x) for x in hit["concept_id_2"].unique().to_list()]
        target_rows = lc.filter(pl.col("concept_id").is_in(target_ids))
        if target_rows.is_empty():
            return None
        return target_rows.row(0, named=True)["concept_name"]

    component = _first_target_name("Has component")
    system = _first_target_name("Has system")

    snomed_target_name: str | None = None
    snomed_hit = rels.filter(
        (pl.col("concept_id_1") == src_id)
        & (pl.col("relationship_id") == "Maps to")
    )
    if not snomed_hit.is_empty():
        snomed_ids = [int(x) for x in snomed_hit["concept_id_2"].unique().to_list()]
        snomed_concepts = load_concept(("SNOMED",)).filter(
            pl.col("concept_id").is_in(snomed_ids)
        )
        if not snomed_concepts.is_empty():
            snomed_target_name = snomed_concepts.row(0, named=True)["concept_name"]

    return {
        "concept_id": src_id,
        "concept_name": src_name,
        "component": component,
        "system": system,
        "snomed_target_name": snomed_target_name,
    }


def _ethos_vocab_cache_path() -> Path:
    """Path to the cached ETHOS token vocabulary parquet.

    Mirrors the convention used by ``build_mapping.build_ethos_vocab``: the
    parquet is staged at ``$ETHOS_MAPPING_CACHE/ethos_vocab.parquet`` (default
    ``/tmp/eq_ethos_cache``). The file is produced upstream by
    ``build_mapping.py`` and contains two columns: ``token`` (str) and
    ``count`` (int64).
    """
    base = Path(os.environ.get("ETHOS_MAPPING_CACHE", "/tmp/eq_ethos_cache"))
    return base / "ethos_vocab.parquet"


@lru_cache(maxsize=1)
def _load_ethos_vocab_counts() -> dict[str, int]:
    """Load the ETHOS token vocabulary into an in-memory ``token -> count``
    dict.

    If the parquet is not on disk yet, lazily import ``build_mapping`` and
    download it from GCS. ``build_mapping`` is imported lazily so that
    ``ontology`` can be used without google-cloud-storage installed when the
    cache file is already present.
    """
    p = _ethos_vocab_cache_path()
    if not p.exists():
        from . import build_mapping as _bm

        p.parent.mkdir(parents=True, exist_ok=True)
        _bm._download(_bm.ETHOS_VOCAB_BLOB, p)
    df = pl.read_parquet(p)
    return {row["token"]: int(row["count"]) for row in df.iter_rows(named=True)}


def ethos_token_count(token: str) -> int:
    """Return the occurrence count of ``token`` in the ETHOS training
    trajectories, or ``0`` if the token is not in the ETHOS vocabulary.

    Reads from the cached ``ethos_vocab.parquet`` produced by
    ``build_mapping.build_ethos_vocab``.
    """
    if token is None:
        return 0
    return _load_ethos_vocab_counts().get(token, 0)


@lru_cache(maxsize=1)
def ethos_icd_3char_index() -> dict[str, str]:
    """Map ICD-10-CM 3-char category code -> descriptive ETHOS ICD token.

    Enumerates every ``ICD//CM//<LABEL>`` token in the cached ETHOS vocab,
    excluding the numeric ``ICD//CM//3-6//*`` and ``ICD//CM//SFX//*`` token
    families, resolves each ``<LABEL>`` to its source ICD-10-CM 3-char
    category via ``ethos_descriptive_icd_to_codes``, and returns a dict keyed
    by that 3-char concept_code (e.g. ``G20``, ``I25``).

    When two distinct labels collide on the same 3-char (should not happen
    for the loose normalization used in ``ethos_descriptive_icd_to_codes``),
    the first label encountered wins. Tokens whose label fails to resolve
    are silently dropped.
    """
    counts = _load_ethos_vocab_counts()
    out: dict[str, str] = {}
    for token in counts:
        if not token.startswith("ICD//CM//"):
            continue
        label = token[len("ICD//CM//") :]
        if not label:
            continue
        if label.startswith("3-6//") or label.startswith("SFX//"):
            continue
        resolved = ethos_descriptive_icd_to_codes(label)
        if resolved is None:
            continue
        code = resolved.get("source_concept_code")
        if code and code not in out:
            out[code] = token
    return out


@lru_cache(maxsize=1)
def ethos_atc_index() -> dict[str, str]:
    """Map ATC concept_code -> ETHOS ATC token.

    Three observed token shapes in the cached ETHOS vocab:

    * ``ATC//<level>//<NAME>``   -- e.g. ``ATC//4//A``. Second segment is a
      level-number digit and the third segment is the ATC concept_code (a
      level-1 single letter).
    * ``ATC//<atc_code>//<NAME>`` -- e.g. ``ATC//A10//DRUGS_USED_IN_DIABETES``.
      Second segment is the ATC concept_code (level >= 2), third segment is
      the readable name.
    * ``ATC//SFX//<atc_code>``   -- e.g. ``ATC//SFX//A01``. Third segment is
      the ATC concept_code.

    When the same ``atc_code`` is represented by multiple tokens, prefer the
    descriptive ``CODE`` shape over ``LEVEL`` over ``SFX``.
    """
    counts = _load_ethos_vocab_counts()
    priority = {"CODE": 0, "LEVEL": 1, "SFX": 2}
    chosen: dict[str, tuple[int, str]] = {}
    for token in counts:
        if not token.startswith("ATC//"):
            continue
        parts = token.split("//")
        if len(parts) != 3:
            continue
        _, p1, p2 = parts
        if p1 == "SFX":
            atc_code = p2
            shape = "SFX"
        elif p1.isdigit():
            atc_code = p2
            shape = "LEVEL"
        else:
            atc_code = p1
            shape = "CODE"
        if not atc_code:
            continue
        rank = priority[shape]
        prev = chosen.get(atc_code)
        if prev is None or rank < prev[0]:
            chosen[atc_code] = (rank, token)
    return {code: token for code, (_, token) in chosen.items()}


_EQ_ICD_DX_RE = re.compile(r"^DIAGNOSIS//ICD//(9|10)//(.+)$")


def icd_to_ethos_token(eq_code: str) -> dict | None:
    """Walk an EQ ICD diagnosis code to its single most-specific ETHOS
    descriptive ICD token via the ICD-10-CM 3-char parent.

    Dispatches on the EQ ICD edition:

    * ICD-9: bridges via ``icd9_dx`` (ICD-9-CM -> SNOMED -> ICD-10-CM
      target list), then takes the unique 3-char prefixes of those targets.
    * ICD-10: looks up via ``icd10cm`` and uses ``parent_3char_code``.

    The 3-char code(s) are intersected with ``ethos_icd_3char_index``. When a
    single 3-char parent is found the corresponding ETHOS token is returned.
    When ICD-9 splits across multiple distinct 3-char parents the result is
    ambiguous: ``ethos_token`` is ``None`` and ``candidates`` lists every
    parent.

    Returns ``None`` if ``eq_code`` is not a ``DIAGNOSIS//ICD//(9|10)//*``
    code, or if the source code cannot be resolved at all. Otherwise::

        {
            "ethos_token": str | None,
            "source_3char_code": str | None,
            "descendant_count": int,
            "candidates": list[str],   # of 3-char codes
        }

    ``candidates`` always has length >= 1 once the source resolves; length
    >= 2 indicates the ambiguous ICD-9 multi-parent case that downstream
    LLM-pick logic must disambiguate.
    """
    if not eq_code:
        return None
    m = _EQ_ICD_DX_RE.match(eq_code)
    if m is None:
        return None
    edition, code = m.group(1), m.group(2)

    parents: list[str] = []
    if edition == "9":
        bridged = icd9_dx(code)
        if bridged is None:
            return None
        for target in bridged.get("icd10_target_codes") or []:
            three = target.split(".")[0][:3]
            if three and three not in parents:
                parents.append(three)
    else:
        resolved = icd10cm(code)
        if resolved is None:
            return None
        three = resolved.get("parent_3char_code")
        if three:
            parents.append(three)

    if not parents:
        return {
            "ethos_token": None,
            "source_3char_code": None,
            "descendant_count": 0,
            "candidates": [],
        }

    index = ethos_icd_3char_index()

    if len(parents) == 1:
        three = parents[0]
        ethos_token = index.get(three)
        descendant_count = 0
        if ethos_token is not None:
            label = ethos_token[len("ICD//CM//") :]
            descendants = ethos_descriptive_icd_to_codes(label)
            if descendants is not None:
                descendant_count = descendants["descendant_count"]
        return {
            "ethos_token": ethos_token,
            "source_3char_code": three,
            "descendant_count": descendant_count,
            "candidates": parents,
        }

    return {
        "ethos_token": None,
        "source_3char_code": None,
        "descendant_count": 0,
        "candidates": parents,
    }


def atc_to_ethos_token(drug_name: str) -> dict | None:
    """Walk a free-text drug name to its single most-specific ETHOS ATC
    token via the RxNorm ingredient and the OHDSI ATC ancestor table.

    Strategy:

    1. Resolve the drug name to a RxNorm ingredient via ``rxnorm_drug_to_atc``.
    2. Read every ATC-vocabulary ancestor of that ingredient from
       ``CONCEPT_ANCESTOR``.
    3. Intersect each ATC ancestor's ``concept_code`` with ``ethos_atc_index``.
    4. Reduce the hits to "leaves": atc_codes that are not the ATC-prefix of
       any other hit. These are the deepest reachable ETHOS tokens per ATC
       chain.

    A single leaf is the unambiguous deterministic answer. Multiple leaves
    indicate the ingredient sits in two or more distinct ATC chains (the
    canonical example is Mupirocin -> ``R01`` (nasal) and ``D06``
    (dermatological)); both are returned as ``candidates`` and the caller
    decides whether to commit both or invoke the LLM picker.

    Returns ``None`` when the drug cannot be resolved to a RxNorm
    ingredient. Otherwise::

        {
            "ethos_token": str | None,
            "source_atc_code": str | None,
            "ingredient_concept_id": int | None,
            "ingredient_name": str | None,
            "candidates": list[str],   # of ETHOS tokens
        }

    ``candidates`` is empty when the ingredient resolves but no ATC ancestor
    falls inside the ETHOS vocab; it has length >= 2 when multiple distinct
    ATC chains are reachable from the ingredient.
    """
    if not drug_name:
        return None
    bridge = rxnorm_drug_to_atc(drug_name)
    if bridge is None:
        return None
    ingredient_id = bridge.get("ingredient_concept_id")
    ingredient_name = bridge.get("ingredient_name")
    if ingredient_id is None:
        return {
            "ethos_token": None,
            "source_atc_code": None,
            "ingredient_concept_id": None,
            "ingredient_name": None,
            "candidates": [],
        }

    atc = load_concept(("ATC",))
    atc_id_set = {int(c) for c in atc["concept_id"].to_list()}
    atc_code_by_id = {
        int(r["concept_id"]): r["concept_code"]
        for r in atc.iter_rows(named=True)
    }

    ancestor = load_concept_ancestor([ingredient_id])
    atc_ancestor_ids = [
        int(x)
        for x in ancestor["ancestor_concept_id"].unique().to_list()
        if int(x) in atc_id_set
    ]

    index = ethos_atc_index()
    hits: list[str] = []
    for aid in atc_ancestor_ids:
        code = atc_code_by_id.get(aid) or ""
        if code and code in index and code not in hits:
            hits.append(code)

    if not hits:
        return {
            "ethos_token": None,
            "source_atc_code": None,
            "ingredient_concept_id": ingredient_id,
            "ingredient_name": ingredient_name,
            "candidates": [],
        }

    leaves = sorted(
        ac
        for ac in hits
        if not any(other != ac and other.startswith(ac) for other in hits)
    )
    leaf_tokens = [index[ac] for ac in leaves]

    if len(leaves) == 1:
        return {
            "ethos_token": leaf_tokens[0],
            "source_atc_code": leaves[0],
            "ingredient_concept_id": ingredient_id,
            "ingredient_name": ingredient_name,
            "candidates": leaf_tokens,
        }

    return {
        "ethos_token": None,
        "source_atc_code": None,
        "ingredient_concept_id": ingredient_id,
        "ingredient_name": ingredient_name,
        "candidates": leaf_tokens,
    }


if __name__ == "__main__":
    import json

    print("== load_concept(ICD9CM) ==")
    df = load_concept(("ICD9CM",))
    print(f"rows={len(df)} columns={df.columns}")

    print("== load_concept_relationship(('Maps to',)) ==")
    rels = load_concept_relationship(("Maps to",))
    print(f"rows={len(rels)}")

    print("== load_concept_ancestor(None) ==")
    anc = load_concept_ancestor(None)
    print(f"rows={len(anc)}")

    def _preview(d: dict | None, list_keys: tuple[str, ...] = ()) -> str:
        if d is None:
            return "None"
        out = dict(d)
        for k in list_keys:
            v = out.get(k)
            if isinstance(v, list) and len(v) > 10:
                out[k] = v[:10] + [f"... +{len(v) - 10} more"]
        return json.dumps(out, indent=2, default=str)

    print("== icd9_dx('3320') ==")
    print(_preview(icd9_dx("3320"), ("icd10_target_codes",)))

    print("== icd9_proc('3961') ==")
    print(_preview(icd9_proc("3961"), ("icd10pcs_target_codes",)))

    print("== icd10cm('I10') ==")
    print(_preview(icd10cm("I10")))

    print("== ethos_descriptive_icd_to_codes(\"PARKINSON'S_DISEASE\") ==")
    print(
        _preview(
            ethos_descriptive_icd_to_codes("PARKINSON'S_DISEASE"),
            ("descendant_codes",),
        )
    )

    print("== rxnorm_drug_to_atc('Carbidopa-Levodopa') ==")
    print(_preview(rxnorm_drug_to_atc("Carbidopa-Levodopa")))

    print("== atc_class('N04BA') ==")
    print(_preview(atc_class("N04BA"), ("ingredients",)))

    print("== mimic_item_label(220949) ==")
    print(_preview(mimic_item_label(220949)))

    print("== mimic_item_label(50912) ==")
    print(_preview(mimic_item_label(50912)))

    print("== loinc('2160-0') ==")
    print(_preview(loinc("2160-0")))

    print("== ethos_token_count(\"ICD//CM//PARKINSON'S_DISEASE\") ==")
    print(ethos_token_count("ICD//CM//PARKINSON'S_DISEASE"))

    print("== ethos_token_count('Q1') ==")
    print(ethos_token_count("Q1"))

    print("== ethos_token_count('NOT_A_REAL_TOKEN') ==")
    print(ethos_token_count("NOT_A_REAL_TOKEN"))
