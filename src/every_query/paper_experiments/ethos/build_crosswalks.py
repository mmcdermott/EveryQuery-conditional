"""Phase 3 orchestrator: build the three EQ -> ETHOS crosswalk YAMLs.

For each EQ code in ``eval_aucs_held_out.parquet`` this module dispatches to
the right resolver and emits **at most one** ETHOS token per EQ code (with a
single allowed exception: same-source-concept ATC ties may commit two
tokens, e.g. Mupirocin nasal R01 + dermatological D06 since they share one
RxNorm ingredient). The output replaces the three on-disk crosswalk YAMLs in
[`crosswalks/`](crosswalks):

* [`crosswalks/icd.yaml`](crosswalks/icd.yaml)            -- DIAGNOSIS//ICD//* and PROCEDURE//ICD//*
* [`crosswalks/atc.yaml`](crosswalks/atc.yaml)            -- MEDICATION//*
* [`crosswalks/mimic_items.yaml`](crosswalks/mimic_items.yaml) -- INFUSION_(START|END)//*, LAB//*,
                                          SUBJECT_FLUID_OUTPUT//*, TIMELINE//*

Dispatch rules follow [`tighten_ethos_mappings_via_deterministic_walker_+_scoped_llm_picker.plan.md`](../../../../.cursor/plans/tighten_ethos_mappings_via_deterministic_walker_+_scoped_llm_picker_344884c6.plan.md):

* DIAGNOSIS  -- [`ontology.icd_to_ethos_token`](ontology.py); on
  ``icd9_multi_parent`` ambiguity, escalate to [`pick.llm_pick_one`](pick.py).
* PROCEDURE  -- ETHOS has no PCS tokens; escalate directly to ``pick.llm_pick_one``
  with reason ``no_pcs_tokens_in_ethos`` and candidate diagnosis-token list
  built from the procedure's source name (heuristic name-overlap against
  the ETHOS ICD 3-char index).
* MEDICATION -- [`ontology.atc_to_ethos_token`](ontology.py) on the EQ code's drug
  name segment; on multi-chain ties commit BOTH leaves (length-2 list).
* INFUSION_START / INFUSION_END -- resolve item-id via
  [`ontology.mimic_item_label`](ontology.py) -> drug-name -> ``atc_to_ethos_token``.
* LAB / SUBJECT_FLUID_OUTPUT -- if ``build_mapping`` would have matched the
  base via lab_no_value/lab_exact_value (i.e., the upper-units base is in the ETHOS
  vocab), no crosswalk entry is needed and we skip; otherwise the code is an
  orphan and we escalate to ``pick.llm_pick_one`` with
  ``reason='orphan_lab'`` and candidates seeded by name-overlap against the
  ETHOS ICD 3-char index. ``ethos_token=null`` from the picker writes an
  ``unmappable_with_rationale`` row instead of a mapping.
* TIMELINE//START -- direct mapping to ``HOSPITAL_ADMISSION``.
* MEDS_DEATH      -- already an exact-tier hit in
  ``build_mapping.build_exact``; no crosswalk row needed.

Run with ``python -m every_query.paper_experiments.ethos.build_crosswalks``;
requires ``ANTHROPIC_API_KEY`` to be set when any code falls through to the
LLM picker (procedures + orphan labs always do; ICD ambiguity and ATC ties
sometimes do).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from . import build_mapping as bm
from . import ontology, pick

CROSSWALK_DIR = Path(__file__).parent / "crosswalks"
ICD_CROSSWALK_PATH = CROSSWALK_DIR / "icd.yaml"
ATC_CROSSWALK_PATH = CROSSWALK_DIR / "atc.yaml"
MIMIC_ITEMS_CROSSWALK_PATH = CROSSWALK_DIR / "mimic_items.yaml"


_NONALNUM_RE = re.compile(r"[^A-Za-z0-9]+")
_STOPWORDS = frozenset(
    {
        "of",
        "the",
        "and",
        "or",
        "to",
        "in",
        "on",
        "with",
        "without",
        "by",
        "for",
        "an",
        "a",
        "at",
        "as",
        "from",
        "into",
        "other",
        "unspecified",
        "nec",
        "nos",
        "not",
        "elsewhere",
        "classified",
        "due",
        "type",
    }
)


def _content_words(text: str) -> set[str]:
    """Tokenize ``text`` into lower-case content words (>=3 chars, not in
    ``_STOPWORDS``). Used by the heuristic candidate proposer."""
    if not text:
        return set()
    words = _NONALNUM_RE.sub(" ", text).lower().split()
    return {w for w in words if len(w) >= 3 and w not in _STOPWORDS}


def _propose_diagnosis_candidates(
    text: str, top_k: int = 10
) -> list[dict[str, str]]:
    """Score every ETHOS ``ICD//CM//*`` token by jaccard word-overlap
    between ``text`` and the token's ICD-10-CM 3-char source-vocab name, and
    return the top ``top_k`` tokens as ``{"ethos_token", "icd10cm_3char",
    "icd10cm_3char_name"}`` dicts.

    This is the candidate-list seed for both the PROCEDURE escalation and
    the orphan-LAB escalation. The LLM picks one (or null) from the seed.
    """
    target_words = _content_words(text)
    if not target_words:
        return []
    index = ontology.ethos_icd_3char_index()
    scored: list[tuple[float, str, str, str]] = []
    for code, token in index.items():
        label = token[len("ICD//CM//") :]
        cand_words = _content_words(label.replace("_", " "))
        if not cand_words:
            continue
        overlap = target_words & cand_words
        if not overlap:
            continue
        union = target_words | cand_words
        jaccard = len(overlap) / len(union)
        scored.append((jaccard, code, token, label))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [
        {
            "ethos_token": tok,
            "icd10cm_3char": code,
            "icd10cm_3char_name": label.replace("_", " ").title(),
        }
        for _, code, tok, label in scored[:top_k]
    ]


def _propose_atc_candidates(
    text: str, ethos_vocab: set[str], top_k: int = 12
) -> list[dict[str, str]]:
    """Score named 3-char ETHOS ``ATC//<L3>//<NAME>`` tokens by jaccard
    word-overlap between ``text`` and the L3 class name, returning the top
    ``top_k`` as ``{"ethos_token", "atc_code", "atc_name"}`` dicts.

    Used when the deterministic ATC ancestor walker fails (no RxNorm
    ingredient match, walker returned a wrong ingredient, or the
    ingredient has no ATC ancestors). Single-letter ``ATC//4//*`` tokens
    and the ``ATC//SFX//*`` suffix tokens are skipped because their names
    are not human-readable enough for the LLM to reason about.
    """
    target_words = _content_words(text)
    if not target_words:
        return []
    scored: list[tuple[float, str, str, str]] = []
    for token in ethos_vocab:
        if not token.startswith("ATC//"):
            continue
        parts = token.split("//")
        if len(parts) != 3:
            continue
        atc_code, name = parts[1], parts[2]
        if atc_code == "4" or atc_code == "SFX":
            continue
        cand_words = _content_words(name.replace("_", " "))
        if not cand_words:
            continue
        overlap = target_words & cand_words
        if not overlap:
            continue
        union = target_words | cand_words
        jaccard = len(overlap) / len(union)
        scored.append((jaccard, atc_code, token, name))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [
        {
            "ethos_token": tok,
            "atc_code": atc_code,
            "atc_name": name.replace("_", " ").title(),
        }
        for _, atc_code, tok, name in scored[:top_k]
    ]


def _icd9_diagnosis_meaning(eq_code: str) -> str:
    """Return a human-readable description for a DIAGNOSIS//ICD//*//* code,
    falling back to the EQ code if no source vocab description exists."""
    parts = eq_code.split("//")
    if len(parts) != 4:
        return eq_code
    _, _, edition, code = parts
    if edition == "9":
        bridged = ontology.icd9_dx(code)
        if bridged is not None:
            return bridged.get("concept_name") or eq_code
    if edition == "10":
        resolved = ontology.icd10cm(code)
        if resolved is not None:
            return resolved.get("concept_name") or eq_code
    return eq_code


def _icd9_procedure_meaning(eq_code: str) -> tuple[str, dict | None]:
    """Return ``(human_readable_name, icd9_proc_bridge_dict_or_None)`` for a
    PROCEDURE//ICD//9//* code."""
    parts = eq_code.split("//")
    if len(parts) != 4 or parts[2] != "9":
        return eq_code, None
    bridged = ontology.icd9_proc(parts[3])
    if bridged is None:
        return eq_code, None
    return bridged.get("concept_name") or eq_code, bridged


def _drug_name_from_medication_code(eq_code: str) -> str | None:
    """Extract the free-text drug name from a ``MEDICATION//*`` EQ code.

    The two observed shapes are::

        MEDICATION//<drug>//<action>            -> drug
        MEDICATION//(START|STOP)//<drug>        -> drug

    Returns ``None`` if the code does not match either shape.
    """
    parts = eq_code.split("//")
    if len(parts) < 3 or parts[0] != "MEDICATION":
        return None
    if parts[1] in ("START", "STOP"):
        return parts[2] if len(parts) >= 3 else None
    return parts[1]


def _infusion_item_id(eq_code: str) -> int | None:
    """Extract the integer MIMIC item-id from an ``INFUSION_(START|END)//<id>...``
    EQ code. Returns ``None`` if the second segment is not an integer."""
    parts = eq_code.split("//")
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _is_orphan_lab(code: str, ethos_vocab: set[str]) -> bool:
    """Return ``True`` if a LAB / SUBJECT_FLUID_OUTPUT code's base form
    (after stripping the value-bin and upper-casing units) is NOT in the
    ETHOS vocab and therefore would not be matched by ``build_mapping``'s
    lab_no_value or lab_exact_value tier.
    """
    parsed = bm.parse_value_bin(code)
    base = parsed[0] if parsed is not None else code
    return bm.upper_units(base) not in ethos_vocab


def _lab_meaning(code: str) -> str:
    """Return a human-readable label for a LAB / SUBJECT_FLUID_OUTPUT code,
    looking up the MIMIC item-id when available. Falls back to the EQ code."""
    parsed = bm.parse_value_bin(code)
    base = parsed[0] if parsed is not None else code
    parts = base.split("//")
    if len(parts) >= 2:
        try:
            item_id = int(parts[1])
        except ValueError:
            return code
        info = ontology.mimic_item_label(item_id)
        if info is not None and info.get("label"):
            unit = parts[2] if len(parts) >= 3 else None
            label = info["label"]
            return f"{label} (MIMIC {parts[0]} {item_id}, units={unit})"
    return code


def _atc_token_to_concept_id(token: str) -> int | None:
    """Resolve an ETHOS ATC token (e.g. ``ATC//R01//NASAL_PREPARATIONS``) to
    the underlying OHDSI ATC ``concept_id``. Used to detect "same source
    concept" ties when multiple ATC chains are committed for a single EQ
    medication.
    """
    parts = token.split("//")
    if len(parts) != 3:
        return None
    if parts[1] == "SFX":
        atc_code = parts[2]
    elif parts[1].isdigit():
        atc_code = parts[2]
    else:
        atc_code = parts[1]
    info = ontology.atc_class(atc_code)
    if info is None:
        return None
    return int(info["concept_id"])


def handle_diagnosis(
    eq_code: str, ethos_vocab: set[str]
) -> tuple[dict | None, dict | None]:
    """Resolve a DIAGNOSIS//ICD//* code to a single ETHOS ICD token entry.

    Returns ``(mapping_or_None, unmappable_or_None)``. Walker falls through
    to the LLM picker on multi-parent ambiguity (``icd9_multi_parent``) and
    on full failure (``diagnosis_walker_unresolved``); a null pick from the
    LLM is recorded as ``unmappable_with_rationale`` so every EQ code lands
    in one of the two buckets.
    """
    walked = ontology.icd_to_ethos_token(eq_code)
    eq_label = _icd9_diagnosis_meaning(eq_code)

    if walked is not None and walked.get("ethos_token"):
        return (
            {
                "eq_code": eq_code,
                "eq_meaning": eq_label,
                "ethos_tokens": [walked["ethos_token"]],
                "source": "deterministic:icd_3char_walker",
                "rationale": (
                    f"ICD-10-CM 3-char parent {walked['source_3char_code']} "
                    f"resolved to ETHOS token via ethos_icd_3char_index."
                ),
            },
            None,
        )

    candidates = walked.get("candidates") if walked is not None else []
    candidates = candidates or []
    if len(candidates) >= 2:
        index = ontology.ethos_icd_3char_index()
        pick_candidates: list[dict[str, str]] = []
        for three in candidates:
            token = index.get(three)
            if token is None:
                continue
            label = token[len("ICD//CM//") :].replace("_", " ").title()
            pick_candidates.append(
                {
                    "ethos_token": token,
                    "icd10cm_3char": three,
                    "icd10cm_3char_name": label,
                }
            )
        if pick_candidates:
            picked = pick.llm_pick_one(
                eq_code=eq_code,
                eq_label=eq_label,
                candidates=pick_candidates,
                reason="icd9_multi_parent",
            )
            return _diagnosis_pick_to_entry(
                eq_code, eq_label, picked, "llm:icd9_multi_parent"
            )

    # Walker fully failed (no ICD-10 bridge, or ICD-9 bridge with no
    # icd10_target_codes -- e.g. paroxysmal vtach 4271). Seed an
    # ICD-3char heuristic candidate list from the source-vocab name so
    # the LLM can pick a faithful diagnosis category, with vocab-validated
    # free-text fallback if word overlap is empty.
    heuristic = _propose_diagnosis_candidates(eq_label, top_k=12)
    picked = pick.llm_pick_one(
        eq_code=eq_code,
        eq_label=eq_label,
        candidates=heuristic,
        reason="diagnosis_walker_unresolved",
        ethos_vocab=ethos_vocab,
    )
    return _diagnosis_pick_to_entry(
        eq_code, eq_label, picked, "llm:diagnosis_walker_unresolved"
    )


def _diagnosis_pick_to_entry(
    eq_code: str,
    eq_label: str,
    picked: dict,
    source: str,
) -> tuple[dict | None, dict | None]:
    """Convert an ``llm_pick_one`` result for a diagnosis into either a
    mapping entry or an ``unmappable_with_rationale`` entry, never both
    None. Centralizes the null-token -> unmappable conversion shared by the
    multi-parent and walker-unresolved paths.
    """
    token = picked.get("ethos_token")
    rationale = picked.get("rationale", "")
    if token:
        return (
            {
                "eq_code": eq_code,
                "eq_meaning": eq_label,
                "ethos_tokens": [token],
                "source": source,
                "rationale": rationale,
            },
            None,
        )
    return (
        None,
        {
            "eq_code": eq_code,
            "description": eq_label,
            "reason": rationale or "LLM declined to propose a faithful ETHOS token.",
        },
    )


_PROCEDURE_ACTION_WORDS = frozenset(
    {
        "open",
        "closed",
        "percutaneous",
        "endoscopic",
        "laparoscopic",
        "reduction",
        "fixation",
        "internal",
        "external",
        "operation",
        "operative",
        "surgery",
        "surgical",
        "procedure",
        "incision",
        "excision",
        "biopsy",
        "repair",
        "removal",
        "insertion",
        "implantation",
        "anastomosis",
        "drainage",
        "exploration",
    }
)


def _strip_procedure_action_words(text: str) -> str:
    """Drop generic procedural verbs/adjectives from ``text`` so the
    diagnosis-candidate proposer keys on the body site / clinical content
    rather than the surgical action (e.g. "Open reduction of fracture
    tibia" -> "fracture tibia"). Used only for PROCEDURE seeding.
    """
    words = _NONALNUM_RE.sub(" ", text).split()
    kept = [w for w in words if w.lower() not in _PROCEDURE_ACTION_WORDS]
    return " ".join(kept)


def handle_procedure(
    eq_code: str, ethos_vocab: set[str]
) -> tuple[dict | None, dict | None]:
    """Resolve a PROCEDURE//ICD//* code via the LLM, with candidates seeded
    from the procedure source name's word overlap with ETHOS ICD 3-char
    categories. Returns ``(mapping_or_None, unmappable_or_None)``.
    """
    eq_label, bridge = _icd9_procedure_meaning(eq_code)
    candidate_text = eq_label
    if bridge is not None:
        snomed = bridge.get("snomed_target_name")
        if snomed:
            candidate_text = f"{eq_label} {snomed}"
    candidates = _propose_diagnosis_candidates(
        _strip_procedure_action_words(candidate_text), top_k=15
    )
    picked = pick.llm_pick_one(
        eq_code=eq_code,
        eq_label=eq_label,
        candidates=candidates,
        reason="no_pcs_tokens_in_ethos",
        ethos_vocab=ethos_vocab,
    )
    return _diagnosis_pick_to_entry(
        eq_code, eq_label, picked, "llm:no_pcs_tokens_in_ethos"
    )


def _resolve_atc_for_drug(
    eq_code: str,
    eq_label: str,
    drug_name: str,
    ethos_vocab: set[str],
    fallback_reason: str,
) -> tuple[dict | None, dict | None]:
    """Walk a drug name to an ATC token entry, with LLM disambiguation when
    the ingredient sits in multiple ATC chains and an LLM fallback when the
    walker fails entirely.

    Returns ``(mapping_or_None, unmappable_or_None)`` -- exactly one is
    non-None for any successful escalation. ``fallback_reason`` is used to
    tag the LLM call when the walker can't produce a clean answer
    (``medication_walker_unresolved`` or ``infusion_walker_unresolved``).
    """
    walked = ontology.atc_to_ethos_token(drug_name)

    if walked is not None and walked.get("ethos_token"):
        return (
            {
                "eq_code": eq_code,
                "eq_meaning": eq_label,
                "ethos_tokens": [walked["ethos_token"]],
                "source": "deterministic:atc_ancestor_walker",
                "rationale": (
                    f"RxNorm ingredient '{walked.get('ingredient_name')}' walks to "
                    f"ATC {walked['source_atc_code']} via OHDSI CONCEPT_ANCESTOR."
                ),
            },
            None,
        )

    candidate_tokens = (walked.get("candidates") if walked is not None else None) or []

    if len(candidate_tokens) == 2:
        # Same-source-concept tie: the RxNorm ingredient is shared across two
        # ATC chains (canonical example: Mupirocin in R01 nasal + D06 derm).
        # Commit both tokens since they reflect a single source concept,
        # bypassing the LLM picker per the plan.
        cid_a = _atc_token_to_concept_id(candidate_tokens[0])
        cid_b = _atc_token_to_concept_id(candidate_tokens[1])
        if (
            cid_a is not None
            and cid_b is not None
            and walked is not None
            and walked.get("ingredient_concept_id") is not None
        ):
            return (
                {
                    "eq_code": eq_code,
                    "eq_meaning": eq_label,
                    "ethos_tokens": list(candidate_tokens),
                    "source": "deterministic:atc_same_source_concept_tie",
                    "rationale": (
                        f"RxNorm ingredient '{walked.get('ingredient_name')}' "
                        f"(concept_id={walked.get('ingredient_concept_id')}) sits "
                        f"in two distinct ATC chains; both are committed since "
                        f"they reflect the same source concept."
                    ),
                },
                None,
            )

    if candidate_tokens:
        pick_candidates = [
            {"ethos_token": tok, "atc_code": tok.split("//")[1]}
            for tok in candidate_tokens
        ]
        picked = pick.llm_pick_one(
            eq_code=eq_code,
            eq_label=eq_label,
            candidates=pick_candidates,
            reason="atc_multi_chain",
        )
        return _atc_pick_to_entry(eq_code, eq_label, picked, "llm:atc_multi_chain")

    # Walker yielded no candidates at all -- either no RxNorm ingredient
    # match (e.g. salt-form suffixes like "Doxycycline Hyclate", or
    # parenthetical product names like "KCl (CRRT)") or the matched
    # ingredient has no ATC ancestors. Seed ATC candidates by name overlap
    # against the ETHOS vocab and let the LLM pick (with vocab-validated
    # free-text fallback if word overlap returns nothing).
    heuristic = _propose_atc_candidates(drug_name, ethos_vocab, top_k=12)
    picked = pick.llm_pick_one(
        eq_code=eq_code,
        eq_label=eq_label,
        candidates=heuristic,
        reason=fallback_reason,
        ethos_vocab=ethos_vocab,
    )
    return _atc_pick_to_entry(eq_code, eq_label, picked, f"llm:{fallback_reason}")


def _atc_pick_to_entry(
    eq_code: str,
    eq_label: str,
    picked: dict,
    source: str,
) -> tuple[dict | None, dict | None]:
    """Mirror of ``_diagnosis_pick_to_entry`` for ATC paths: convert an
    ``llm_pick_one`` result into either a mapping entry or an
    ``unmappable_with_rationale`` entry.
    """
    token = picked.get("ethos_token")
    rationale = picked.get("rationale", "")
    if token:
        return (
            {
                "eq_code": eq_code,
                "eq_meaning": eq_label,
                "ethos_tokens": [token],
                "source": source,
                "rationale": rationale,
            },
            None,
        )
    return (
        None,
        {
            "eq_code": eq_code,
            "description": eq_label,
            "reason": rationale or "LLM declined to propose a faithful ETHOS token.",
        },
    )


def handle_medication(
    eq_code: str, ethos_vocab: set[str]
) -> tuple[dict | None, dict | None]:
    """Resolve a MEDICATION//* code; returns ``(mapping_or_None,
    unmappable_or_None)``."""
    drug_name = _drug_name_from_medication_code(eq_code)
    if not drug_name:
        return (None, None)
    return _resolve_atc_for_drug(
        eq_code,
        drug_name,
        drug_name,
        ethos_vocab,
        fallback_reason="medication_walker_unresolved",
    )


def handle_infusion(
    eq_code: str, ethos_vocab: set[str]
) -> tuple[dict | None, dict | None]:
    """Resolve an INFUSION_(START|END)//* code; returns ``(mapping_or_None,
    unmappable_or_None)``."""
    item_id = _infusion_item_id(eq_code)
    if item_id is None:
        return (None, None)
    info = ontology.mimic_item_label(item_id)
    if info is None or not info.get("label"):
        return (None, None)
    drug_name = info["label"]
    eq_label = f"{drug_name} (MIMIC item {item_id})"
    return _resolve_atc_for_drug(
        eq_code,
        eq_label,
        drug_name,
        ethos_vocab,
        fallback_reason="infusion_walker_unresolved",
    )


def handle_orphan_lab(
    eq_code: str, ethos_vocab: set[str]
) -> tuple[dict | None, dict | None]:
    """Resolve an orphan LAB / SUBJECT_FLUID_OUTPUT code via the LLM.

    Returns ``(mapping_entry_or_None, unmappable_entry_or_None)`` -- exactly
    one of the two is non-None for any successfully escalated orphan. If
    the LLM picker returns ``ethos_token=null`` we emit the ``unmappable``
    entry instead of a mapping.

    When the heuristic candidate proposer returns ``[]`` (the lab label
    shares no content words with any ICD-3char name, e.g. "CK-MB",
    "RUBIgGV", "Pinsp (Hamilton)"), we still call the LLM but route through
    the no-candidates prompt branch where the LLM may either propose a
    free-text token from clinical knowledge (validated against
    ``ethos_vocab``) or reply with a clinical rationale for why the lab
    has no faithful ETHOS counterpart.
    """
    eq_label = _lab_meaning(eq_code)
    candidates = _propose_diagnosis_candidates(eq_label, top_k=12)
    picked = pick.llm_pick_one(
        eq_code=eq_code,
        eq_label=eq_label,
        candidates=candidates,
        reason="orphan_lab",
        ethos_vocab=ethos_vocab,
    )
    token = picked.get("ethos_token")
    rationale = picked.get("rationale", "")
    if token:
        return (
            {
                "eq_code": eq_code,
                "eq_meaning": eq_label,
                "ethos_tokens": [token],
                "source": "llm:orphan_lab",
                "rationale": rationale,
            },
            None,
        )
    return (
        None,
        {
            "eq_code": eq_code,
            "description": eq_label,
            "reason": rationale or "LLM declined to propose a faithful ETHOS token.",
        },
    )


def handle_timeline(eq_code: str) -> dict | None:
    """TIMELINE//START -> HOSPITAL_ADMISSION direct mapping (preserved from
    the prior crosswalk)."""
    if eq_code != "TIMELINE//START":
        return None
    return {
        "eq_code": eq_code,
        "eq_meaning": "Start of patient record / first event token",
        "ethos_tokens": ["HOSPITAL_ADMISSION"],
        "source": "direct:event_alignment",
        "rationale": (
            "EQ's TIMELINE//START fires at the first event of a patient's "
            "record; ETHOS uses HOSPITAL_ADMISSION as its admission marker."
        ),
    }


def _yaml_header(intro: str) -> str:
    """Return a hash-prefixed comment header for a crosswalk YAML; keeps the
    pipeline's audit-trail conventions consistent across the three files.
    """
    body = "\n".join(f"# {line}" if line else "#" for line in intro.splitlines())
    return body + "\n\n"


def _write_yaml(path: Path, header: str, mappings: list[dict], unmappable: list[dict] | None = None) -> None:
    payload: dict[str, Any] = {"mappings": mappings}
    if unmappable:
        payload["unmappable_with_rationale"] = unmappable
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(header)
        yaml.safe_dump(
            payload,
            f,
            sort_keys=False,
            default_flow_style=False,
            width=88,
            allow_unicode=True,
        )


_ICD_HEADER = _yaml_header(
    "Generated by build_crosswalks.py: EQ ICD codes -> ETHOS descriptive ICD//CM\n"
    "tokens.\n"
    "\n"
    "DIAGNOSIS rows are produced by the deterministic ICD-10-CM 3-char\n"
    "walker in ontology.icd_to_ethos_token; the LLM picker is invoked only\n"
    "when an ICD-9 code crosswalks to multiple distinct 3-char parents.\n"
    "PROCEDURE rows are LLM-only since ETHOS encodes no PCS tokens; the\n"
    "candidate list is seeded by name overlap with the procedure's source\n"
    "concept name. Each entry commits exactly one ETHOS token."
)
_ATC_HEADER = _yaml_header(
    "Generated by build_crosswalks.py: EQ MEDICATION codes -> ETHOS ATC tokens.\n"
    "\n"
    "Rows are produced by the deterministic ATC ancestor walker in\n"
    "ontology.atc_to_ethos_token. When the RxNorm ingredient sits in two\n"
    "ATC chains under one source concept (e.g. Mupirocin R01/D06) both\n"
    "leaves are committed; otherwise ties are resolved by the LLM picker."
)
_MIMIC_HEADER = _yaml_header(
    "Generated by build_crosswalks.py: EQ MIMIC-item codes -> ETHOS tokens.\n"
    "\n"
    "INFUSION rows resolve the item-id to a drug-name label and walk to ATC\n"
    "via ontology.atc_to_ethos_token. LAB and SUBJECT_FLUID_OUTPUT rows are\n"
    "limited to orphan codes (base form not in the ETHOS vocab, so\n"
    "build_mapping's lab_no_value / lab_exact_value tiers cannot match) and are\n"
    "resolved by the LLM picker with diagnosis-token candidates seeded by\n"
    "name overlap. TIMELINE//START is a direct mapping. Codes the LLM\n"
    "declined are listed in unmappable_with_rationale at the bottom."
)


def main() -> None:
    print("Loading EQ codes ...")
    eq_codes = bm.load_eq_codes()
    print(f"  {len(eq_codes)} unique EQ queries")

    print("Loading ETHOS vocab ...")
    ethos_vocab = set(ontology._load_ethos_vocab_counts().keys())
    print(f"  {len(ethos_vocab)} ETHOS tokens")

    icd_entries: list[dict] = []
    icd_unmappable: list[dict] = []
    atc_entries: list[dict] = []
    atc_unmappable: list[dict] = []
    mimic_entries: list[dict] = []
    mimic_unmappable: list[dict] = []
    skipped: list[tuple[str, str]] = []

    for code in eq_codes:
        fam = bm.family_of(code)
        try:
            if fam == "DIAGNOSIS":
                mapped, unmap = handle_diagnosis(code, ethos_vocab)
                if mapped is not None:
                    icd_entries.append(mapped)
                elif unmap is not None:
                    icd_unmappable.append(unmap)
                else:
                    skipped.append((code, "diagnosis_unresolved"))
            elif fam == "PROCEDURE":
                mapped, unmap = handle_procedure(code, ethos_vocab)
                if mapped is not None:
                    icd_entries.append(mapped)
                elif unmap is not None:
                    icd_unmappable.append(unmap)
                else:
                    skipped.append((code, "procedure_unresolved"))
            elif fam == "MEDICATION":
                mapped, unmap = handle_medication(code, ethos_vocab)
                if mapped is not None:
                    atc_entries.append(mapped)
                elif unmap is not None:
                    atc_unmappable.append(unmap)
                else:
                    skipped.append((code, "medication_unresolved"))
            elif fam in ("INFUSION_START", "INFUSION_END"):
                mapped, unmap = handle_infusion(code, ethos_vocab)
                if mapped is not None:
                    mimic_entries.append(mapped)
                elif unmap is not None:
                    mimic_unmappable.append(unmap)
                else:
                    skipped.append((code, "infusion_unresolved"))
            elif fam in ("LAB", "SUBJECT_FLUID_OUTPUT"):
                if not _is_orphan_lab(code, ethos_vocab):
                    continue
                mapped, unmap = handle_orphan_lab(code, ethos_vocab)
                if mapped is not None:
                    mimic_entries.append(mapped)
                elif unmap is not None:
                    mimic_unmappable.append(unmap)
            elif fam == "TIMELINE":
                entry = handle_timeline(code)
                if entry is not None:
                    mimic_entries.append(entry)
            elif fam == "MEDS_DEATH":
                # MEDS_DEATH matches an exact ETHOS token in build_mapping's
                # exact tier; no crosswalk row is needed.
                continue
            else:
                skipped.append((code, f"unhandled_family:{fam}"))
        except Exception as exc:
            skipped.append((code, f"error:{type(exc).__name__}:{exc}"))

    icd_entries.sort(key=lambda e: e["eq_code"])
    icd_unmappable.sort(key=lambda e: e["eq_code"])
    atc_entries.sort(key=lambda e: e["eq_code"])
    atc_unmappable.sort(key=lambda e: e["eq_code"])
    mimic_entries.sort(key=lambda e: e["eq_code"])
    mimic_unmappable.sort(key=lambda e: e["eq_code"])

    print()
    print(f"icd.yaml mapped:            {len(icd_entries)}")
    print(f"icd.yaml unmappable:        {len(icd_unmappable)}")
    print(f"atc.yaml mapped:            {len(atc_entries)}")
    print(f"atc.yaml unmappable:        {len(atc_unmappable)}")
    print(f"mimic_items.yaml mapped:    {len(mimic_entries)}")
    print(f"mimic_items.yaml unmappable: {len(mimic_unmappable)}")
    print(f"skipped:                    {len(skipped)}")
    for code, reason in skipped:
        print(f"  - {code}: {reason}")

    _write_yaml(
        ICD_CROSSWALK_PATH,
        _ICD_HEADER,
        icd_entries,
        unmappable=icd_unmappable,
    )
    _write_yaml(
        ATC_CROSSWALK_PATH,
        _ATC_HEADER,
        atc_entries,
        unmappable=atc_unmappable,
    )
    _write_yaml(
        MIMIC_ITEMS_CROSSWALK_PATH,
        _MIMIC_HEADER,
        mimic_entries,
        unmappable=mimic_unmappable,
    )

    print()
    print("Wrote:")
    print(f"  {ICD_CROSSWALK_PATH}")
    print(f"  {ATC_CROSSWALK_PATH}")
    print(f"  {MIMIC_ITEMS_CROSSWALK_PATH}")


if __name__ == "__main__":
    main()
