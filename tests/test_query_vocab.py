"""Tests for the shared query-vocabulary seam.

:mod:`every_query.data.query_vocab` owns the single answer to "which vocabulary codes does
this query string mention?".  Its own grammar is covered by doctests; what matters here is
that the four sites where a query meets a vocabulary actually route through it, because the
upstream experiments each patched a *different* subset of those sites and shipped two
incomplete fixes for one bug.

These tests are deliberately forward-looking: no generator emits aggregate expressions yet,
so they assert the seam holds before the feature that needs it lands.
"""

import tempfile
from pathlib import Path

import polars as pl
import pytest
from omegaconf import OmegaConf

from every_query.data.query_vocab import OP_ATOM, component_codes, parse_query, unknown_codes
from every_query.generate_tasks.sample_evaluation_query_sequences import (
    SequenceSpec,
    validate_spec_codes,
)
from every_query.predict.predict import _check_vocab


def _train_cfg(tmpdir: str, codes: list[str]) -> OmegaConf:
    meta_dir = Path(tmpdir) / "metadata"
    meta_dir.mkdir(exist_ok=True)
    pl.DataFrame({"code": codes}).write_parquet(meta_dir / "codes.parquet")
    return OmegaConf.create({"datamodule": {"config": {"tensorized_cohort_dir": tmpdir}}})


# ── the seam itself ─────────────────────────────────────────────────────


def test_bare_codes_are_unchanged_by_the_seam():
    """The forms that exist today must pass through untouched — this refactor is a no-op."""
    for code in ("A", "LAB//GLUCOSE", "TIMELINE//END", "HOSPITAL_ADMISSION//EW EMER.//ER"):
        assert parse_query(code).op == OP_ATOM
        assert component_codes(code) == [code]


def test_unknown_codes_reports_components_not_the_expression():
    assert unknown_codes(["ANY(A|B)"], {"A", "B"}) == []
    assert unknown_codes(["ANY(A|B)"], {"A"}) == ["B"]
    # The wrapper string itself is never reported as a missing code.
    assert "ANY(A|B)" not in unknown_codes(["ANY(A|B)"], {"A"})


# ── site 2: predict._check_vocab ────────────────────────────────────────


def test_check_vocab_accepts_aggregate_whose_components_are_known():
    """An aggregate is not a code; validating it as one would reject a valid query."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _train_cfg(tmpdir, ["A", "B"])
        _check_vocab({"SEQ(A>B|gap=3)"}, cfg)  # must not raise


def test_check_vocab_still_rejects_a_bad_component():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _train_cfg(tmpdir, ["A"])
        with pytest.raises(ValueError, match="ZZZ"):
            _check_vocab({"ANY(A|ZZZ)"}, cfg)


def test_check_vocab_still_rejects_a_bad_bare_code():
    """The pre-existing behaviour must be preserved exactly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _train_cfg(tmpdir, ["A"])
        with pytest.raises(ValueError, match="MISSING"):
            _check_vocab({"A", "MISSING"}, cfg)


# ── site 3: validate_spec_codes ─────────────────────────────────────────


def test_validate_spec_codes_accepts_aggregate_components():
    spec = SequenceSpec("agg", ("ALL(A|B)", "C"), (30.0, 7.0))
    validate_spec_codes([spec], ["A", "B", "C"])  # must not raise


def test_validate_spec_codes_still_rejects_unknown():
    spec = SequenceSpec("bad", ("A", "NOPE"), (30.0, 7.0))
    with pytest.raises(ValueError, match="NOPE"):
        validate_spec_codes([spec], ["A", "B"])


def test_validate_spec_codes_reports_unknown_aggregate_component():
    spec = SequenceSpec("bad", ("ANY(A|NOPE)",), (30.0,))
    with pytest.raises(ValueError, match="NOPE"):
        validate_spec_codes([spec], ["A"])
