"""Anthropic-backed LLM picker for ambiguous EQ -> ETHOS token choices.

This module is the **only** code path in the ETHOS crosswalk pipeline that hits
an external LLM. It is intentionally narrow:

* ``call_llm(prompt)`` is a thin wrapper around Anthropic's Messages API that
  pins the model + sampling settings used in the paper run and retries once on
  transient errors so that an occasional 5xx / network blip doesn't abort the
  build.
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

Environment: requires ``ANTHROPIC_API_KEY`` to be set (the SDK reads it
automatically when ``anthropic.Anthropic()`` is instantiated without an
explicit ``api_key=``).
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import anthropic

MODEL = "claude-3-5-sonnet-latest"
MAX_TOKENS = 1024
TEMPERATURE = 0.0

_TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
    anthropic.RateLimitError,
)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _client() -> anthropic.Anthropic:
    """Construct an Anthropic SDK client. ``ANTHROPIC_API_KEY`` is read by the
    SDK from the environment; we don't pass it explicitly so that the same
    error message surfaces if the variable is unset.
    """
    return anthropic.Anthropic()


def call_llm(prompt: str) -> str:
    """Send ``prompt`` to ``claude-3-5-sonnet-latest`` and return the assistant
    text content. Single retry with a 1s back-off on transient API errors
    (connection / timeout / 5xx / 429); any other exception propagates.

    The prompt is sent as a single user-turn message. We don't set a system
    prompt -- ``llm_pick_one`` builds the full instruction text into the user
    message so the JSON-output contract lives in one place and is easier to
    audit.
    """
    client = _client()
    last_exc: BaseException | None = None
    for attempt in range(2):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
            )
            parts: list[str] = []
            for block in response.content:
                # Messages API returns a list of content blocks; only ``text``
                # blocks carry assistant prose. Other block types (tool_use,
                # etc.) are not requested by this prompt and are skipped.
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
            return "".join(parts)
        except _TRANSIENT_ERRORS as e:
            last_exc = e
            if attempt == 0:
                time.sleep(1.0)
                continue
            raise
    # Unreachable -- the loop either returns or re-raises -- but keeps mypy
    # / pyright honest about the function's contract.
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


def _build_pick_prompt(
    eq_code: str,
    eq_label: str,
    candidates: list[dict],
    reason: str,
    previous_attempt: str | None = None,
) -> str:
    """Build the JSON-output prompt sent to the LLM.

    ``candidates`` is a list of ``{"ethos_token": str, "description": str,
    ...}`` dicts; the LLM may pick exactly one ``ethos_token`` from this list,
    or return ``null`` to declare the EQ code unmappable. ``reason`` is one of
    the four strings used by ``build_crosswalks.py``
    (``icd9_multi_parent`` / ``atc_multi_chain`` / ``no_pcs_tokens_in_ethos``
    / ``orphan_lab``) so the model knows why deterministic dispatch fell
    through.
    """
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
        LLM may only pick one of these tokens. Pass an empty list when the
        deterministic walker yielded no candidates -- the model will then
        return ``ethos_token=null`` (i.e. unmappable).
    reason:
        Short tag describing why deterministic dispatch failed; used in the
        prompt so the model can adjust its rationale.

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
    return {"ethos_token": token, "rationale": rationale}
