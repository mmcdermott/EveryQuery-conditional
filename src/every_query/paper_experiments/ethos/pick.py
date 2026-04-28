"""OpenRouter-backed LLM picker for ambiguous EQ -> ETHOS token choices.

This module is the **only** code path in the ETHOS crosswalk pipeline that hits
an external LLM. It is intentionally narrow:

* ``call_llm(prompt)`` is a thin wrapper around the OpenAI Chat Completions API
  pointed at OpenRouter (``https://openrouter.ai/api/v1``). It pins the model
  + sampling settings used in the paper run and retries once on transient
  errors so that an occasional 5xx / network blip doesn't abort the build.
* ``llm_pick_one(eq_code, eq_label, candidates, reason)`` is the single entry
  point used by ``build_crosswalks.py`` for the ~10-15 EQ codes where the
  deterministic walker in ``ontology.py`` either has no answer
  (``no_pcs_tokens_in_ethos``, ``orphan_lab``) or returns multiple equally
  plausible answers (``icd9_multi_parent``, ``atc_multi_chain``). The returned
  dict is appended verbatim into the crosswalk YAML's ``rationale`` column;
  the ``ethos_token`` field is the (possibly null) tightened pick.

The prompt is JSON-only: the LLM is told to reply with a single JSON object of
shape ``{"ethos_token": <token-or-null>, "rationale": <string>}`` and any
free-text preamble is stripped before parsing. On a parse failure we re-prompt
exactly once with the previous response echoed back, then surface the error to
the caller -- the orchestrator treats a hard failure as ``unmappable`` rather
than committing a guess.

Environment: requires ``OPENROUTER_API_KEY`` to be set (loaded from ``.env``
via ``python-dotenv`` on first use). The model slug is OpenRouter's
``anthropic/claude-sonnet-4.5``.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "anthropic/claude-sonnet-4.5"
MAX_TOKENS = 1024
TEMPERATURE = 0.0

_TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _client() -> OpenAI:
    """Construct an OpenAI SDK client pointed at OpenRouter.

    ``OPENROUTER_API_KEY`` is read from the environment (``.env`` is loaded
    on first call). We surface a clear error if it is unset rather than
    letting the SDK raise a generic ``OpenAIError``.
    """
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to .env (see .env.example) "
            "or export it before running build_crosswalks."
        )
    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


def call_llm(prompt: str) -> str:
    """Send ``prompt`` to ``anthropic/claude-sonnet-4.5`` via OpenRouter and
    return the assistant text content. Single retry with a 1s back-off on
    transient API errors (connection / timeout / 5xx / 429); any other
    exception propagates.

    The prompt is sent as a single user-turn message. We don't set a system
    prompt -- ``llm_pick_one`` builds the full instruction text into the user
    message so the JSON-output contract lives in one place and is easier to
    audit.
    """
    client = _client()
    last_exc: BaseException | None = None
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
            )
            choice = response.choices[0] if response.choices else None
            if choice is None or choice.message is None:
                return ""
            return choice.message.content or ""
        except _TRANSIENT_ERRORS as e:
            last_exc = e
            if attempt == 0:
                time.sleep(1.0)
                continue
            raise
    assert last_exc is not None
    raise last_exc


def _parse_json_response(text: str) -> dict[str, Any]:
    """Extract the first JSON object from an LLM response and return it.

    The model is instructed to reply with a bare JSON object, but defensively
    we strip any prose preamble (``Here is the JSON: {...}``) and any
    fenced-code wrapper (```` ```json ... ``` ````). Raises ``ValueError`` if
    no JSON object can be extracted or the extracted text is not valid JSON.
    """
    if not text:
        raise ValueError("empty LLM response")
    match = _JSON_OBJECT_RE.search(text)
    if match is None:
        raise ValueError(f"no JSON object found in response: {text!r}")
    return json.loads(match.group(0))


_ETHOS_NAMESPACE_HINT = (
    "ETHOS tokens follow these namespaces:\n"
    "- ICD//CM//<NAME>: ICD-10-CM 3-char diagnosis category, e.g. "
    "ICD//CM//PARKINSON'S_DISEASE (G20).\n"
    "- ATC//<L3>//<NAME>: ATC level-3 drug class, e.g. "
    "ATC//N04//ANTI-PARKINSON_DRUGS.\n"
    "- HCPCS//<CODE>//<NAME>: HCPCS/CPT procedure tokens.\n"
    "- LAB//<LOINC-or-MIMIC-id>//<UNITS>: structured lab tokens.\n"
    "- HOSPITAL_ADMISSION, MEDS_DEATH, etc.: top-level event tokens."
)


def _build_pick_prompt(
    eq_code: str,
    eq_label: str,
    candidates: list[dict],
    reason: str,
    previous_attempt: str | None = None,
) -> str:
    """Build the JSON-output prompt sent to the LLM.

    Two branches:

    * **With candidates** (the common case): the LLM must pick one of the
      provided ``ethos_token`` strings or return ``null``. Used by
      ``icd9_multi_parent`` / ``atc_multi_chain`` / ``no_pcs_tokens_in_ethos``
      / ``orphan_lab`` and the walker-unresolved escalations when the
      heuristic seed produced at least one candidate.

    * **Without candidates** (escape hatch for orphan labs / oddball codes
      where heuristic word-overlap returned ``[]``): the LLM is asked to
      either propose a faithful token from its clinical knowledge as a
      free-text string OR explicitly return ``null`` with a clinical
      rationale. The caller validates the proposed token against the ETHOS
      vocab; an invalid string is treated as ``null``.
    """
    if candidates:
        candidates_json = json.dumps(candidates, indent=2)
        base = f"""You are tightening a crosswalk between EveryQuery (EQ) clinical
codes and ETHOS vocabulary tokens. Pick the single most semantically specific
ETHOS token that captures the EQ code's meaning, or return null if no
candidate is a faithful match.

EQ code: {eq_code}
EQ label / meaning: {eq_label}
Reason this case was escalated to the LLM: {reason}

Candidate ETHOS tokens (you MUST pick one of these or null):
{candidates_json}

Reply with a single JSON object and nothing else, in this exact shape:
{{"ethos_token": "<one of the candidate ethos_token strings, or null>",
  "rationale": "<one or two sentences explaining the choice>"}}
If no candidate is a defensible mapping, set ethos_token to null and explain
why in the rationale.
""".strip()
    else:
        base = f"""You are tightening a crosswalk between EveryQuery (EQ) clinical
codes and ETHOS vocabulary tokens. The automatic candidate proposer found no
plausible candidates for this EQ code, so we are asking you to reason from
clinical knowledge alone.

EQ code: {eq_code}
EQ label / meaning: {eq_label}
Reason this case was escalated to the LLM: {reason}

{_ETHOS_NAMESPACE_HINT}

Your task: decide whether any single ETHOS token from those namespaces would
be a faithful, semantically-specific match for this EQ code. If yes, propose
the exact token string (e.g. "ICD//CM//PRESSURE_ULCER" or
"ATC//N04//ANTI-PARKINSON_DRUGS"). If no faithful match exists, return null
and briefly explain (in clinical terms, not process terms) why this concept
falls outside the ETHOS vocabulary's scope.

Reply with a single JSON object and nothing else, in this exact shape:
{{"ethos_token": "<an ETHOS token string, or null>",
  "rationale": "<one or two sentences of clinical reasoning>"}}
""".strip()
    if previous_attempt is not None:
        base += (
            "\n\nYour previous reply could not be parsed as the required JSON"
            " object. The previous reply was:\n"
            f"{previous_attempt}\n\n"
            "Please respond again with ONLY the JSON object described above."
        )
    return base


def llm_pick_one(
    eq_code: str,
    eq_label: str,
    candidates: list[dict],
    reason: str,
    ethos_vocab: set[str] | None = None,
) -> dict:
    """Ask the LLM to pick exactly one ETHOS token (or none) for an EQ code.

    Parameters
    ----------
    eq_code:
        The EQ query string (e.g. ``DIAGNOSIS//ICD//9//7295``).
    eq_label:
        Human-readable meaning of the EQ code; used to give the LLM enough
        context to choose between candidates.
    candidates:
        List of ``{"ethos_token": str, "description": str, ...}`` dicts. The
        LLM may only pick one of these tokens. Pass an empty list to
        activate the no-candidates branch where the LLM is asked to propose
        a token from clinical knowledge instead.
    reason:
        Short tag describing why deterministic dispatch failed; used in the
        prompt so the model can adjust its rationale.
    ethos_vocab:
        Optional set of valid ETHOS token strings. Used **only** when
        ``candidates`` is empty: a free-text token proposed by the LLM is
        validated against this set, and an unknown token is downgraded to
        ``null`` with a rationale prefix noting the rejected proposal. Pass
        ``None`` (default) to skip validation.

    Returns
    -------
    dict with keys ``ethos_token`` (str | None) and ``rationale`` (str). On a
    JSON parse failure we re-prompt once; if the second reply is still
    unparseable a ``ValueError`` propagates so the orchestrator can mark the
    code unmappable rather than commit a guess.
    """
    prompt = _build_pick_prompt(eq_code, eq_label, candidates, reason)
    text = call_llm(prompt)
    try:
        parsed = _parse_json_response(text)
    except ValueError:
        retry_prompt = _build_pick_prompt(
            eq_code, eq_label, candidates, reason, previous_attempt=text
        )
        retry_text = call_llm(retry_prompt)
        parsed = _parse_json_response(retry_text)

    token = parsed.get("ethos_token")
    rationale = parsed.get("rationale", "")
    if token is not None and not isinstance(token, str):
        raise ValueError(
            f"ethos_token must be string or null, got {type(token).__name__}"
        )
    if not isinstance(rationale, str):
        rationale = str(rationale)

    if (
        token is not None
        and not candidates
        and ethos_vocab is not None
        and token not in ethos_vocab
    ):
        # No-candidates branch: the LLM proposed a free-text token. Try to
        # salvage near-misses where the namespace key matches a unique vocab
        # member (e.g. LLM proposes "ATC//J01//ANTIBACTERIALS_FOR_SYSTEMIC_USE"
        # but the canonical ETHOS token is
        # "ATC//J01//ANTIBIOTICS_AND_ANTIBACTERIALS_FOR_SYSTEMIC_USE"). If
        # that fails, reject rather than commit a hallucination.
        salvaged = _salvage_token_by_prefix(token, ethos_vocab)
        if salvaged is not None:
            rationale = (
                f"LLM proposed '{token}'; salvaged to canonical ETHOS token "
                f"'{salvaged}' via unique namespace-prefix match. Original "
                f"rationale: {rationale}".strip()
            )
            token = salvaged
        else:
            rationale = (
                f"LLM proposed '{token}' which is not in the ETHOS vocab; "
                f"original rationale: {rationale}".strip()
            )
            token = None

    return {"ethos_token": token, "rationale": rationale}


def _salvage_token_by_prefix(token: str, ethos_vocab: set[str]) -> str | None:
    """Salvage a near-miss free-text token proposal by checking whether the
    namespace prefix uniquely identifies an ETHOS vocab member.

    For ATC tokens of the form ``ATC//<L3>//<NAME>``, the L3 code (e.g.
    ``J01``) is the canonical key in the ETHOS vocab; each L3 has at most
    one named token. If exactly one ETHOS token starts with
    ``ATC//<L3>//``, that token is returned. For ICD tokens of the form
    ``ICD//CM//<NAME>`` we can't reverse-lookup the 3-char code from the
    name alone, so this returns ``None`` and the caller treats the proposal
    as unmappable.
    """
    parts = token.split("//")
    if len(parts) < 3:
        return None
    if parts[0] == "ATC" and parts[1] not in ("4", "SFX"):
        prefix = f"ATC//{parts[1]}//"
        matches = [t for t in ethos_vocab if t.startswith(prefix)]
        if len(matches) == 1:
            return matches[0]
    return None
