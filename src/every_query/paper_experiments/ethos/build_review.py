"""Build the per-EQ-code mapping review markdown (Phase 2).

Reads the mapping artifacts produced by ``build_mapping.py``, calls
``ontology.py`` to resolve authoritative labels and constituent-code
expansions, and emits a single markdown file at ``review/mapping_review.md``
(one section per EQ code, mapped families first then unmapped).

Held-out AUC numbers are intentionally NOT included in the doc -- they
anchor reviewers toward post-hoc-correct decisions, but the review question
is whether the mapping is a fair semantic bridge, which is independent of
empirical performance.

Public entry points:

* ``load_mapping_artifacts`` -- pulls the four parquet inputs.
* ``_load_llm_rationales`` / ``_load_unmappable_rationales`` -- read the
  curated YAML crosswalks under ``crosswalks/``.
* ``build_review_markdown`` -- orchestrates per-section render and prepends
  a table of contents.
* ``main`` -- writes the result to ``review/mapping_review.md``.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import yaml

from . import build_mapping as _bm
from . import ontology as _ont

_THIS_DIR = Path(__file__).parent
_CROSSWALK_DIR = _THIS_DIR / "crosswalks"
_REVIEW_DIR = _THIS_DIR / "review"
_REVIEW_PATH = _REVIEW_DIR / "mapping_review.md"
_CROSSWALK_PATHS = (
    _CROSSWALK_DIR / "icd.yaml",
    _CROSSWALK_DIR / "atc.yaml",
    _CROSSWALK_DIR / "mimic_items.yaml",
)
_MIMIC_ITEMS_YAML = _CROSSWALK_DIR / "mimic_items.yaml"

CACHE_DIR = _bm.CACHE_DIR

_DESCENDANT_TRUNCATION = 30

_MEDICATION_ADMIN_MODES = {
    "START",
    "STOP",
    "Administered",
    "Delayed Administered",
    "Bolus",
}


def load_mapping_artifacts() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Download (with cache reuse via ``build_mapping._download``) the three
    parquet artifacts the review depends on, and return them as a 3-tuple in
    the order ``(mapping_table, mapping_coverage, ethos_vocab)``.

    Cache files live under ``build_mapping.CACHE_DIR`` (defaults to
    ``/tmp/eq_ethos_cache`` and overridable via the ``ETHOS_MAPPING_CACHE``
    env var). ``_download`` is a no-op if the file already exists, so this
    function is cheap to call repeatedly.
    """
    mapping_table_path = CACHE_DIR / "mapping_table.parquet"
    mapping_coverage_path = CACHE_DIR / "mapping_coverage.parquet"
    ethos_vocab_path = CACHE_DIR / "ethos_vocab.parquet"

    _bm._download(_bm.MAPPING_TABLE_BLOB, mapping_table_path)
    _bm._download(_bm.MAPPING_COVERAGE_BLOB, mapping_coverage_path)
    _bm._download(_bm.ETHOS_VOCAB_BLOB, ethos_vocab_path)

    mapping_table = pl.read_parquet(mapping_table_path)
    mapping_coverage = pl.read_parquet(mapping_coverage_path)
    ethos_vocab = pl.read_parquet(ethos_vocab_path)

    return mapping_table, mapping_coverage, ethos_vocab


def _anchor(eq_code: str) -> str:
    """Stable GitHub-flavoured-markdown anchor slug for ``eq_code``."""
    out = []
    for ch in eq_code.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", "_"):
            out.append("-")
    slug = "".join(out).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug


def _render_section_header(
    eq_code: str,
    family: str,
    coverage_row: dict,
) -> str:
    """Emit the per-section header: an H3 title anchored on the EQ code, the
    family, and the list of tiers it was mapped under.

    Held-out AUC numbers are intentionally omitted -- see module docstring.
    """
    tiers = coverage_row.get("mapped_under_tiers") or []
    tiers_no_union = [t for t in tiers if t != "union"]
    tier_str = ", ".join(tiers_no_union) if tiers_no_union else "(none)"

    lines: list[str] = []
    lines.append(f'### <a id="{_anchor(eq_code)}"></a>`{eq_code}`')
    lines.append("")
    lines.append(f"- **Family:** `{family}`")
    lines.append(f"- **Mapped tiers:** {tier_str}")
    lines.append("")

    return "\n".join(lines)


def _parse_medication_drug_name(eq_code: str) -> str:
    """Extract the drug-name segment from a ``MEDICATION//...`` EQ code.

    The EQ medication encoding interleaves the drug name with administration
    modes (``START``, ``STOP``, ``Administered``, ``Delayed Administered``,
    ``Bolus``). Strategy: split on ``//``, drop the family prefix, drop any
    parts whose value is in the known administration-mode set, and rejoin.
    Empirically every EQ medication code has exactly one non-mode segment,
    which is the drug name.
    """
    parts = eq_code.split("//")
    if not parts or parts[0] != "MEDICATION":
        return eq_code
    drug_parts = [p for p in parts[1:] if p not in _MEDICATION_ADMIN_MODES]
    return "//".join(drug_parts) if drug_parts else eq_code


def _format_truncated_list(items: list[str], cap: int = _DESCENDANT_TRUNCATION) -> str:
    """Format a list of short string items as a comma-separated inline list,
    truncated at ``cap`` items with a ``... +N more`` suffix.
    """
    if not items:
        return "(none)"
    if len(items) <= cap:
        return ", ".join(items)
    head = ", ".join(items[:cap])
    return f"{head}, ... +{len(items) - cap} more"


def _render_eq_label(eq_code: str) -> str:
    """Resolve and render the authoritative EQ-side label for ``eq_code`` as
    a markdown subsection.

    Dispatches on the family prefix:

    * ``DIAGNOSIS//ICD//9//*``   -> ``ontology.icd9_dx``  (ICD-9 -> SNOMED -> ICD-10-CM)
    * ``DIAGNOSIS//ICD//10//*``  -> ``ontology.icd10cm``  (ICD-10-CM 3-char parent + SNOMED)
    * ``PROCEDURE//ICD//9//*``   -> ``ontology.icd9_proc``  (ICD-9-Proc -> SNOMED -> ICD-10-PCS)
    * ``PROCEDURE//ICD//10//*``  -> ``ontology.icd10cm``  (best-effort; PCS code falls through)
    * ``MEDICATION//*``          -> ``ontology.rxnorm_drug_to_atc`` on the parsed drug name
    * ``LAB//<itemid>//...``     -> ``ontology.mimic_item_label`` on the item id
    * ``INFUSION_START//<id>``   -> ``ontology.mimic_item_label``
    * ``INFUSION_END//<id>``     -> ``ontology.mimic_item_label``
    * ``SUBJECT_FLUID_OUTPUT//<id>``-> ``ontology.mimic_item_label``
    * ``MEDS_DEATH``             -> static label
    * ``TIMELINE//START``        -> static label

    The returned markdown starts with the ``**Authoritative EQ label:**`` bold
    line, followed by bullet-list cross-ontology bridges where applicable.
    """
    parts = eq_code.split("//")
    family = parts[0]
    lines: list[str] = []

    if eq_code == "MEDS_DEATH":
        lines.append("**Authoritative EQ label:** Patient death (MEDS-format mortality token)")
        return "\n".join(lines)

    if eq_code == "TIMELINE//START":
        lines.append(
            "**Authoritative EQ label:** Start of patient record (first event marker)"
        )
        return "\n".join(lines)

    if family == "DIAGNOSIS" and len(parts) >= 4 and parts[1] == "ICD":
        edition = parts[2]
        code = parts[3]
        if edition == "9":
            hit = _ont.icd9_dx(code)
            if hit is None:
                lines.append(
                    f"**Authoritative EQ label:** _ICD-9-CM `{code}` not found in Athena bundle._"
                )
                return "\n".join(lines)
            lines.append(
                f"**Authoritative EQ label:** ICD-9-CM `{code}` -- {hit['concept_name']}"
            )
            if hit.get("snomed_target_name"):
                lines.append(
                    f"- SNOMED bridge: {hit['snomed_target_name']} "
                    f"(concept_id {hit.get('snomed_target_concept_id')})"
                )
            icd10_codes = hit.get("icd10_target_codes") or []
            lines.append(
                f"- ICD-10-CM crosswalk ({len(icd10_codes)} code"
                f"{'' if len(icd10_codes) == 1 else 's'}): "
                f"{_format_truncated_list(icd10_codes)}"
            )
            return "\n".join(lines)

        if edition == "10":
            hit = _ont.icd10cm(code)
            if hit is None:
                lines.append(
                    f"**Authoritative EQ label:** _ICD-10-CM `{code}` not found in Athena bundle._"
                )
                return "\n".join(lines)
            lines.append(
                f"**Authoritative EQ label:** ICD-10-CM `{code}` -- {hit['concept_name']}"
            )
            parent_code = hit.get("parent_3char_code")
            parent_name = hit.get("parent_3char_name")
            if parent_code:
                parent_label = parent_name or "(unnamed)"
                lines.append(f"- 3-char parent: `{parent_code}` -- {parent_label}")
            snomed = hit.get("snomed_target")
            if snomed:
                lines.append(
                    f"- SNOMED bridge: {snomed['concept_name']} "
                    f"(concept_id {snomed['concept_id']})"
                )
            return "\n".join(lines)

        lines.append(
            f"**Authoritative EQ label:** _Unrecognized ICD edition `{edition}` for `{eq_code}`._"
        )
        return "\n".join(lines)

    if family == "PROCEDURE" and len(parts) >= 4 and parts[1] == "ICD":
        edition = parts[2]
        code = parts[3]
        if edition == "9":
            hit = _ont.icd9_proc(code)
            if hit is None:
                lines.append(
                    f"**Authoritative EQ label:** _ICD-9-Proc `{code}` not found in Athena bundle._"
                )
                return "\n".join(lines)
            lines.append(
                f"**Authoritative EQ label:** ICD-9-Proc `{code}` -- {hit['concept_name']}"
            )
            if hit.get("snomed_target_name"):
                lines.append(
                    f"- SNOMED bridge: {hit['snomed_target_name']} "
                    f"(concept_id {hit.get('snomed_target_concept_id')})"
                )
            pcs_codes = hit.get("icd10pcs_target_codes") or []
            lines.append(
                f"- ICD-10-PCS crosswalk ({len(pcs_codes)} code"
                f"{'' if len(pcs_codes) == 1 else 's'}): "
                f"{_format_truncated_list(pcs_codes)}"
            )
            return "\n".join(lines)

        if edition == "10":
            hit = _ont.icd10cm(code)
            if hit is None:
                lines.append(
                    f"**Authoritative EQ label:** _ICD-10 procedure `{code}` not resolved "
                    f"(ICD-10-PCS lookup not staged)._"
                )
                return "\n".join(lines)
            lines.append(
                f"**Authoritative EQ label:** ICD-10 `{code}` -- {hit['concept_name']}"
            )
            return "\n".join(lines)

    if family == "MEDICATION":
        drug_name = _parse_medication_drug_name(eq_code)
        admin_modes = [p for p in parts[1:] if p in _MEDICATION_ADMIN_MODES]
        admin_str = ", ".join(admin_modes) if admin_modes else "(none)"
        lines.append(
            f"**Authoritative EQ label:** MIMIC medication `{drug_name}` "
            f"(admin modes: {admin_str})"
        )
        hit = _ont.rxnorm_drug_to_atc(drug_name)
        if hit is None:
            lines.append("- RxNorm match: _no RxNorm concept matched._")
            return "\n".join(lines)
        lines.append(
            f"- RxNorm match: {hit['rxnorm_concept_name']} "
            f"(concept_class {hit.get('rxnorm_concept_class')})"
        )
        if hit.get("ingredient_name"):
            lines.append(
                f"- Ingredient: {hit['ingredient_name']} "
                f"(concept_id {hit.get('ingredient_concept_id')})"
            )
        if hit.get("atc_class_3_code"):
            lines.append(
                f"- ATC level 3: `{hit['atc_class_3_code']}` -- {hit['atc_class_3_name']}"
            )
        if hit.get("atc_class_4_code"):
            lines.append(
                f"- ATC level 4: `{hit['atc_class_4_code']}` -- {hit['atc_class_4_name']}"
            )
        return "\n".join(lines)

    if family in ("LAB", "INFUSION_START", "INFUSION_END", "SUBJECT_FLUID_OUTPUT"):
        if len(parts) < 2:
            lines.append(f"**Authoritative EQ label:** _malformed EQ code `{eq_code}`._")
            return "\n".join(lines)
        try:
            item_id = int(parts[1])
        except ValueError:
            lines.append(
                f"**Authoritative EQ label:** _non-numeric MIMIC item-id `{parts[1]}`._"
            )
            return "\n".join(lines)
        hit = _ont.mimic_item_label(item_id)
        if hit is None:
            lines.append(
                f"**Authoritative EQ label:** _MIMIC item-id `{item_id}` "
                f"not found in d_items / d_labitems._"
            )
            return "\n".join(lines)
        lines.append(
            f"**Authoritative EQ label:** MIMIC item-id `{item_id}` -- {hit.get('label')} "
            f"(source: `{hit.get('source_table')}`)"
        )
        for field in ("abbreviation", "category", "unitname", "fluid", "linksto"):
            v = hit.get(field)
            if v:
                lines.append(f"- {field}: {v}")
        if len(parts) >= 3 and parts[2]:
            lines.append(f"- EQ-encoded units: `{parts[2]}`")
        return "\n".join(lines)

    lines.append(f"**Authoritative EQ label:** _unrecognized EQ family `{family}`._")
    return "\n".join(lines)


def _render_ethos_token(
    token: str,
    match_kind: str,
    mapping_source: str,
) -> str:
    """Emit a markdown block describing one candidate ETHOS token.

    Block contents:

    1. Token literal + ETHOS vocab count + match-kind + mapping-source.
    2. Inferred source-vocab concept (using ``ethos_descriptive_icd_to_codes``
       for ``ICD//CM//*``, ``atc_class`` for ``ATC//*``, direct passthrough
       for ``HOSPITAL_ADMISSION`` and ``MEDS_DEATH``; other token shapes
       render as ``inferred source: ETHOS-internal token``).
    3. Constituent-code expansion (truncated at 30 entries with a
       ``... +N more`` suffix).

    The LLM rationale is rendered once per EQ code by the orchestrator
    (since the YAML stores one rationale per ``eq_code`` covering all
    candidate tokens for that code), not per token.
    """
    count = _ont.ethos_token_count(token)
    lines: list[str] = []
    lines.append(f"#### Candidate ETHOS token: `{token}`")
    lines.append("")
    lines.append(f"- ETHOS vocab count: {count:,}")
    lines.append(f"- Match kind: `{match_kind}`")
    lines.append(f"- Mapping source: `{mapping_source}`")

    parts = token.split("//")

    if token == "HOSPITAL_ADMISSION":
        lines.append("- Inferred source: ETHOS hospital-admission marker (synthetic event token).")
    elif token == "MEDS_DEATH":
        lines.append("- Inferred source: ETHOS mortality token (passthrough, no ontology lookup).")
    elif len(parts) >= 3 and parts[0] == "ICD" and parts[1] == "CM":
        label = "//".join(parts[2:])
        hit = _ont.ethos_descriptive_icd_to_codes(label)
        if hit is None:
            lines.append(
                f"- Inferred source: _no ICD-10-CM 3-char category match for label `{label}`._"
            )
        else:
            lines.append(
                f"- Inferred source: ICD-10-CM 3-char category "
                f"`{hit['source_concept_code']}` -- {hit['source_concept_name']}"
            )
            descendants = hit.get("descendant_codes") or []
            lines.append(
                f"- Constituent ICD-10-CM codes ({hit.get('descendant_count', len(descendants))}): "
                f"{_format_truncated_list(descendants)}"
            )
    elif parts[0] == "ATC" and len(parts) >= 2:
        atc_code = parts[1]
        hit = _ont.atc_class(atc_code)
        if hit is None:
            lines.append(
                f"- Inferred source: _ATC code `{atc_code}` not found in Athena ATC vocab._"
            )
        else:
            lines.append(
                f"- Inferred source: ATC level {hit.get('level')} class "
                f"`{atc_code}` -- {hit['concept_name']}"
            )
            ingredients = hit.get("ingredients") or []
            lines.append(
                f"- Constituent RxNorm ingredients ({len(ingredients)}): "
                f"{_format_truncated_list(ingredients)}"
            )
    else:
        lines.append("- Inferred source: ETHOS-internal token (no Athena ontology bridge).")

    lines.append("")

    return "\n".join(lines)


def _render_status_block() -> str:
    """Return the fixed reviewer status + notes template appended to each
    mapped EQ-code section.
    """
    lines: list[str] = []
    lines.append("**STATUS:** [ ] approve  [ ] reject  [ ] modify")
    lines.append("")
    lines.append("**NOTES:**")
    lines.append("")
    lines.append("```")
    lines.append("")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _render_unmapped_section(
    eq_code: str,
    family: str,
    coverage_row: dict,
    unmappable_yaml_entry: dict | None,
) -> str:
    """Emit the per-EQ-code section for an unmapped query.

    Renders the section header (without an AUC table -- unmapped codes have
    no candidate predictions but the held-out AUC may still be available;
    that is left to the orchestrator to attach via the standard header
    helper if desired), the EQ-side authoritative label, the
    ``unmapped_reason`` from ``mapping_coverage.parquet``, and the
    ``description`` / ``reason`` from ``mimic_items.yaml``'s
    ``unmappable_with_rationale`` block when present, followed by a status
    block with the unmapped-specific options.
    """
    lines: list[str] = []
    lines.append(f'### <a id="{_anchor(eq_code)}"></a>`{eq_code}` _(unmapped)_')
    lines.append("")
    lines.append(f"- **Family:** `{family}`")
    lines.append("- **Mapped tiers:** (none)")
    lines.append("")

    lines.append(_render_eq_label(eq_code))
    lines.append("")

    unmapped_reason = coverage_row.get("unmapped_reason") or "(no reason recorded)"
    lines.append(f"**Unmapped reason (from mapping_coverage):** `{unmapped_reason}`")
    lines.append("")

    if unmappable_yaml_entry:
        desc = unmappable_yaml_entry.get("description")
        reason = unmappable_yaml_entry.get("reason")
        lines.append("**YAML rationale (`crosswalks/mimic_items.yaml`):**")
        lines.append("")
        if desc:
            lines.append(f"- description: {desc}")
        if reason:
            lines.append(f"- reason: {reason}")
        lines.append("")
    else:
        lines.append(
            "_No `unmappable_with_rationale` entry in `crosswalks/mimic_items.yaml`._"
        )
        lines.append("")

    lines.append(
        "**STATUS:** [ ] confirm no ETHOS counterpart  "
        "[ ] propose a candidate (notes below)"
    )
    lines.append("")
    lines.append("**NOTES:**")
    lines.append("")
    lines.append("```")
    lines.append("")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def _load_llm_rationales() -> dict[tuple[str, str], str]:
    """Read the three crosswalk YAMLs and return a flat
    ``{(eq_code, ethos_token): rationale_string}`` map.

    For each ``mappings`` entry in ``crosswalks/{icd,atc,mimic_items}.yaml``,
    one key is emitted per declared ``(eq_code, ethos_token)`` pair using the
    YAML's ``rationale`` field as the value. Entries without a ``rationale``
    are skipped silently.

    The YAML's ``eq_code`` is recorded verbatim. For value-binned families
    (``LAB``, ``INFUSION_*``, ``SUBJECT_FLUID_OUTPUT``) the YAML may store an
    un-binned prefix (e.g. ``"LAB//226499//mL"``) that fans out at mapping-
    build time to every binned EQ query; the orchestrator handles that
    fan-out at lookup time by stripping the trailing ``//value_[..)`` segment
    when a literal-key lookup misses.
    """
    out: dict[tuple[str, str], str] = {}
    for path in _CROSSWALK_PATHS:
        if not path.exists():
            continue
        with path.open() as f:
            data = yaml.safe_load(f) or {}
        for entry in data.get("mappings") or []:
            eq_code = entry.get("eq_code")
            rationale = entry.get("rationale")
            if not eq_code or not rationale:
                continue
            tokens = entry.get("ethos_tokens") or []
            for tok in tokens:
                out[(eq_code, tok)] = rationale
    return out


def _load_unmappable_rationales() -> dict[str, dict]:
    """Read the ``unmappable_with_rationale`` block of
    ``crosswalks/mimic_items.yaml`` and return
    ``{eq_code: {"description": str | None, "reason": str | None}}``.

    The block lists EQ codes that the LLM-curated crosswalk intentionally
    declined to map; the renderer attaches the description and reason to the
    unmapped EQ-code section so a reviewer can see why it was skipped.
    """
    out: dict[str, dict] = {}
    if not _MIMIC_ITEMS_YAML.exists():
        return out
    with _MIMIC_ITEMS_YAML.open() as f:
        data = yaml.safe_load(f) or {}
    for entry in data.get("unmappable_with_rationale") or []:
        eq_code = entry.get("eq_code")
        if not eq_code:
            continue
        out[eq_code] = {
            "description": entry.get("description"),
            "reason": entry.get("reason"),
        }
    return out


def _strip_value_bin(eq_code: str) -> str:
    """Return ``eq_code`` with any trailing ``//value_[..)`` segment removed.

    Used to fall back from a binned EQ query (e.g.
    ``"LAB//226499//mL//value_[3150.0,inf)"``) to the un-binned crosswalk
    YAML key (``"LAB//226499//mL"``) when looking up an LLM rationale.
    """
    return _bm.VALUE_BIN_RE.sub("", eq_code)


def _resolve_rationale(
    rationales: dict[tuple[str, str], str],
    eq_code: str,
    token: str,
) -> str | None:
    """Two-step lookup against the ``_load_llm_rationales`` map: try the
    literal ``(eq_code, token)`` key first, then retry with the value-bin
    suffix stripped from ``eq_code`` so binned EQ queries resolve through a
    crosswalk YAML entry that stored only the un-binned prefix.
    """
    hit = rationales.get((eq_code, token))
    if hit is not None:
        return hit
    stripped = _strip_value_bin(eq_code)
    if stripped != eq_code:
        return rationales.get((stripped, token))
    return None


def build_review_markdown() -> str:
    """Produce the full mapping-review markdown as a single string.

    Pipeline:

    1. Load the four mapping artifacts via ``load_mapping_artifacts``.
    2. Load LLM and unmappable rationales from the crosswalk YAMLs.
    3. Filter ``mapping_table`` to drop the ``union`` tier (each unique
       candidate token appears once across primary tiers; the tiers each
       candidate appeared under are aggregated separately for display).
    4. Iterate over the 41 EQ codes in the order from
       ``mapping_coverage.parquet`` (mapped families together first, then
       unmapped at the end) and dispatch to either the mapped or unmapped
       renderer.
    5. Prepend a table-of-contents linking to each EQ-code anchor.
    """
    mapping_table, mapping_coverage, _ethos_vocab = load_mapping_artifacts()
    rationales = _load_llm_rationales()
    unmappable = _load_unmappable_rationales()

    primary = mapping_table.filter(pl.col("tier") != "union")

    candidates_by_query: dict[str, list[dict]] = {}
    for query, sub in primary.group_by("query", maintain_order=True):
        query_str = query[0] if isinstance(query, tuple) else query
        sub_sorted = sub.sort("token_pattern")
        seen: dict[str, dict] = {}
        for row in sub_sorted.iter_rows(named=True):
            tok = row["token_pattern"]
            if tok in seen:
                seen[tok]["tiers"].append(row["tier"])
                continue
            seen[tok] = {
                "token": tok,
                "match_kind": row["match_kind"],
                "mapping_source": row["mapping_source"],
                "tiers": [row["tier"]],
            }
        candidates_by_query[query_str] = list(seen.values())

    sections: list[str] = []
    toc_lines: list[str] = ["## Table of contents", ""]

    for cov_row in mapping_coverage.iter_rows(named=True):
        eq_code = cov_row["query"]
        family = cov_row["family"]
        is_unmapped = bool(cov_row["is_unmapped"])
        anchor = _anchor(eq_code)
        marker = " _(unmapped)_" if is_unmapped else ""
        toc_lines.append(f"- [`{eq_code}`{marker}](#{anchor})")

        if is_unmapped:
            section = _render_unmapped_section(
                eq_code=eq_code,
                family=family,
                coverage_row=cov_row,
                unmappable_yaml_entry=unmappable.get(eq_code),
            )
            sections.append(section)
            continue

        section_parts: list[str] = []
        section_parts.append(
            _render_section_header(
                eq_code=eq_code,
                family=family,
                coverage_row=cov_row,
            )
        )
        section_parts.append(_render_eq_label(eq_code))
        section_parts.append("")

        candidates = candidates_by_query.get(eq_code, [])

        unique_rationales: list[str] = []
        seen_rationales: set[str] = set()
        for cand in candidates:
            r = _resolve_rationale(rationales, eq_code, cand["token"])
            if r and r not in seen_rationales:
                unique_rationales.append(r)
                seen_rationales.add(r)
        if unique_rationales:
            section_parts.append("**LLM rationale (verbatim):**")
            section_parts.append("")
            for r in unique_rationales:
                section_parts.append("> " + r.replace("\n", "\n> "))
                section_parts.append("")

        if not candidates:
            section_parts.append("_No candidate ETHOS tokens recorded outside the union tier._")
            section_parts.append("")
        else:
            for cand in candidates:
                tiers = sorted(set(cand["tiers"]))
                tiers_str = ", ".join(tiers)
                token_md = _render_ethos_token(
                    token=cand["token"],
                    match_kind=cand["match_kind"],
                    mapping_source=cand["mapping_source"],
                )
                section_parts.append(
                    f"_Found under tier(s): {tiers_str}_"
                )
                section_parts.append("")
                section_parts.append(token_md)

        section_parts.append(_render_status_block())
        sections.append("\n".join(section_parts))

    header_lines = [
        "# ETHOS mapping review",
        "",
        (
            "Per-EQ-code review of the EQ -> ETHOS code mapping produced by "
            "`build_mapping.py`. Each section shows the authoritative EQ-side "
            "label resolved through the staged Athena ontology, the verbatim "
            "LLM rationale (rendered once per EQ code where one was recorded), "
            "the candidate ETHOS tokens with vocab counts and constituent-code "
            "expansions, and a status block for the reviewer to mark "
            "approve / reject / modify."
        ),
        "",
    ]

    parts: list[str] = []
    parts.extend(header_lines)
    parts.extend(toc_lines)
    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append("\n\n---\n\n".join(sections))
    parts.append("")
    return "\n".join(parts)


def main() -> None:
    """Build the review markdown and write it to
    ``src/every_query/paper_experiments/ethos/review/mapping_review.md``,
    creating the ``review/`` directory if it does not exist.
    """
    _REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    md = build_review_markdown()
    _REVIEW_PATH.write_text(md)
    print(f"Wrote {_REVIEW_PATH} ({len(md):,} chars)")


if __name__ == "__main__":
    main()
