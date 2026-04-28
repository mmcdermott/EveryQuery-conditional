"""Tests for ``every_query.paper_experiments.ethos_compat.reprocess``.

The Ethos tokenizer splits ICD10/CM and ATC codes across multiple events at the same
``(subject_id, time)``.  ``EQ_reprocess_ethos`` collapses those triplets back to one
event each.  These tests verify the recombination is correct, deterministic, and
preserves the through-pass for atomic events.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from every_query.paper_experiments.ethos_compat.reprocess import (
    FAMILIES,
    reprocess_directory,
    reprocess_shard,
)

_VENV_BIN = str(Path(sys.executable).parent)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ethos_like_events() -> pl.DataFrame:
    """Synthetic events frame mirroring Ethos's split-token output shape.

    Two subjects, mixed split + atomic events, multiple ICD codes co-occurring at the same timestamp (the
    trickiest pairing case), an ATC drug code with all three sub-tokens, and a few atomic codes that should
    pass through unchanged.
    """
    rows = [
        # Subject 1 at t1: short ICD code (head only)
        {"subject_id": 1, "time": datetime(2024, 1, 1, 10), "code": "ICD//CM//I10"},
        # Subject 1 at t2: 5-char ICD (head + mid)
        {"subject_id": 1, "time": datetime(2024, 1, 2, 10), "code": "ICD//CM//E11"},
        {"subject_id": 1, "time": datetime(2024, 1, 2, 10), "code": "ICD//CM//3-6//65"},
        # Subject 1 at t2 (same time): atomic LAB event
        {"subject_id": 1, "time": datetime(2024, 1, 2, 10), "code": "LAB//Q//CREATININE"},
        # Subject 1 at t3: full 7-char ICD triplet
        {"subject_id": 1, "time": datetime(2024, 1, 3, 10), "code": "ICD//CM//K70"},
        {"subject_id": 1, "time": datetime(2024, 1, 3, 10), "code": "ICD//CM//3-6//30"},
        {"subject_id": 1, "time": datetime(2024, 1, 3, 10), "code": "ICD//CM//SFX//1"},
        # Subject 1 at t4: full ATC triplet (Aspirin)
        {"subject_id": 1, "time": datetime(2024, 1, 4, 10), "code": "ATC//N02BA01//Acetylsalicylic_Acid"},
        {"subject_id": 1, "time": datetime(2024, 1, 4, 10), "code": "ATC//4//A"},
        {"subject_id": 1, "time": datetime(2024, 1, 4, 10), "code": "ATC//SFX//01"},
        # Subject 2 at t5: two co-occurring full ICD triplets
        {"subject_id": 2, "time": datetime(2024, 2, 1, 10), "code": "ICD//CM//I10"},
        {"subject_id": 2, "time": datetime(2024, 2, 1, 10), "code": "ICD//CM//3-6//00"},
        {"subject_id": 2, "time": datetime(2024, 2, 1, 10), "code": "ICD//CM//SFX//A"},
        {"subject_id": 2, "time": datetime(2024, 2, 1, 10), "code": "ICD//CM//E11"},
        {"subject_id": 2, "time": datetime(2024, 2, 1, 10), "code": "ICD//CM//3-6//65"},
        {"subject_id": 2, "time": datetime(2024, 2, 1, 10), "code": "ICD//CM//SFX//1"},
        # Subject 2 at t6: atomic admission
        {"subject_id": 2, "time": datetime(2024, 2, 2, 10), "code": "HOSPITAL_ADMISSION"},
    ]
    return pl.DataFrame(rows, schema={"subject_id": pl.Int64, "time": pl.Datetime("us"), "code": pl.Utf8})


# ---------------------------------------------------------------------------
# Unit-level tests
# ---------------------------------------------------------------------------


class TestRecombination:
    def test_short_icd_passes_through_as_head_only(self, ethos_like_events):
        """A 3-char ICD code (head only, no mid/sfx) emerges unchanged."""
        out, mapping = reprocess_shard(ethos_like_events)
        s1_t1 = out.filter((pl.col("subject_id") == 1) & (pl.col("time") == datetime(2024, 1, 1, 10)))
        assert s1_t1.height == 1
        assert s1_t1["code"][0] == "ICD//CM//I10"

    def test_five_char_icd_combines_head_and_mid(self, ethos_like_events):
        out, mapping = reprocess_shard(ethos_like_events)
        s1_t2 = out.filter(
            (pl.col("subject_id") == 1)
            & (pl.col("time") == datetime(2024, 1, 2, 10))
            & pl.col("code").str.starts_with("ICD")
        )
        assert s1_t2.height == 1
        assert s1_t2["code"][0] == "ICD//CM//E11//3-6//65"

    def test_full_icd_triplet_combines_to_one_row(self, ethos_like_events):
        out, mapping = reprocess_shard(ethos_like_events)
        s1_t3 = out.filter((pl.col("subject_id") == 1) & (pl.col("time") == datetime(2024, 1, 3, 10)))
        assert s1_t3.height == 1
        assert s1_t3["code"][0] == "ICD//CM//K70//3-6//30//SFX//1"

    def test_full_atc_triplet_combines_to_one_row(self, ethos_like_events):
        out, mapping = reprocess_shard(ethos_like_events)
        s1_t4 = out.filter((pl.col("subject_id") == 1) & (pl.col("time") == datetime(2024, 1, 4, 10)))
        assert s1_t4.height == 1
        assert s1_t4["code"][0] == "ATC//N02BA01//Acetylsalicylic_Acid//4//A//SFX//01"

    def test_co_occurring_icd_codes_pair_by_rank(self, ethos_like_events):
        """Two full ICD triplets at the same (subject, time) recombine to two distinct rows, paired in row
        order (head[0] with mid[0]+sfx[0], head[1] with mid[1]+sfx[1])."""
        out, mapping = reprocess_shard(ethos_like_events)
        s2_t5 = out.filter(
            (pl.col("subject_id") == 2)
            & (pl.col("time") == datetime(2024, 2, 1, 10))
            & pl.col("code").str.starts_with("ICD")
        ).sort("code")
        assert s2_t5.height == 2
        codes = set(s2_t5["code"].to_list())
        assert codes == {
            "ICD//CM//E11//3-6//65//SFX//1",
            "ICD//CM//I10//3-6//00//SFX//A",
        }

    def test_atomic_events_pass_through_unchanged(self, ethos_like_events):
        out, mapping = reprocess_shard(ethos_like_events)
        atomic = out.filter(pl.col("code").is_in(["LAB//Q//CREATININE", "HOSPITAL_ADMISSION"]))
        assert atomic.height == 2
        assert set(atomic["code"].to_list()) == {"LAB//Q//CREATININE", "HOSPITAL_ADMISSION"}

    def test_total_row_count_drops_by_expected_amount(self, ethos_like_events):
        """17 input rows: 1 short + (1+1) 5-char + 3 full + 3 ATC + 6 (two co-occurring) + 2 atomic.
        Expected output: 1 + 1 + 1 + 1 + 2 + 2 = 8.
        """
        out, _ = reprocess_shard(ethos_like_events)
        assert ethos_like_events.height == 17
        assert out.height == 8

    def test_mapping_records_only_changed_codes(self, ethos_like_events):
        out, mapping = reprocess_shard(ethos_like_events)
        # Pass-through codes (LAB//Q//CREATININE, HOSPITAL_ADMISSION, ICD//CM//I10 alone)
        # must not appear as input_codes.
        inputs = set(mapping["input_code"].to_list())
        assert "LAB//Q//CREATININE" not in inputs
        assert "HOSPITAL_ADMISSION" not in inputs

        # Combined-only ICD/ATC sub-tokens MUST appear as inputs.
        assert "ICD//CM//3-6//65" in inputs
        assert "ICD//CM//SFX//1" in inputs
        assert "ATC//4//A" in inputs
        assert "ATC//SFX//01" in inputs

    def test_mapping_outputs_match_combined_codes(self, ethos_like_events):
        """Every output_code in the mapping must appear in the rewritten event stream."""
        out, mapping = reprocess_shard(ethos_like_events)
        assert mapping.height > 0
        rewritten_codes = set(out["code"].to_list())
        for output_code in mapping["output_code"].unique().to_list():
            assert output_code in rewritten_codes, (
                f"mapping declares output_code={output_code!r} but it doesn't appear in the rewritten events"
            )


class TestEdgeCases:
    def test_empty_input_returns_empty(self):
        empty = pl.DataFrame(schema={"subject_id": pl.Int64, "time": pl.Datetime("us"), "code": pl.Utf8})
        out, mapping = reprocess_shard(empty)
        assert out.height == 0
        assert mapping.height == 0

    def test_no_split_codes_passes_through(self):
        events = pl.DataFrame(
            {
                "subject_id": [1, 1, 2],
                "time": [datetime(2024, 1, 1)] * 3,
                "code": ["LAB//Q//A", "VITAL//Q//HR", "HOSPITAL_ADMISSION"],
            },
            schema={"subject_id": pl.Int64, "time": pl.Datetime("us"), "code": pl.Utf8},
        )
        out, mapping = reprocess_shard(events)
        assert out.height == 3
        assert mapping.height == 0
        assert set(out["code"].to_list()) == {"LAB//Q//A", "VITAL//Q//HR", "HOSPITAL_ADMISSION"}

    def test_orphan_mid_without_head_logged_and_dropped(self, caplog):
        """A mid token with no matching head at the same (subject, time) is dropped."""
        events = pl.DataFrame(
            {
                "subject_id": [1, 1],
                "time": [datetime(2024, 1, 1)] * 2,
                # No ICD//CM//<head> token; just an orphan mid.
                "code": ["ICD//CM//3-6//00", "LAB//Q//CREATININE"],
            },
            schema={"subject_id": pl.Int64, "time": pl.Datetime("us"), "code": pl.Utf8},
        )
        with caplog.at_level("WARNING"):
            out, _ = reprocess_shard(events)
        # Orphan mid is dropped; LAB row stays.
        assert out.height == 1
        assert out["code"][0] == "LAB//Q//CREATININE"
        assert any("orphan mid" in rec.message for rec in caplog.records)

    def test_combined_codes_are_reversible(self, ethos_like_events):
        """Splitting a combined output code on the well-known markers recovers the original triplet."""
        out, _ = reprocess_shard(ethos_like_events)
        full_triplet_row = out.filter(pl.col("code") == "ICD//CM//K70//3-6//30//SFX//1")
        assert full_triplet_row.height == 1
        code = full_triplet_row["code"][0]
        head, _, rest = code.partition("//3-6//")
        assert head == "ICD//CM//K70"
        mid, _, sfx = rest.partition("//SFX//")
        assert mid == "30"
        assert sfx == "1"


class TestFamilyDefinitions:
    def test_families_have_distinct_head_prefixes(self):
        """Family head prefixes must be non-overlapping for first-match classification."""
        prefixes = [f.head_prefix for f in FAMILIES]
        for i, p in enumerate(prefixes):
            for j, q in enumerate(prefixes):
                if i == j:
                    continue
                assert not p.startswith(q) and not q.startswith(p), (
                    f"head prefixes {p!r} and {q!r} overlap — first-match classification breaks"
                )

    def test_family_mid_and_sfx_extend_head_prefix(self):
        """Each family's mid + sfx prefixes should start with its head prefix."""
        for f in FAMILIES:
            assert f.mid_prefix.startswith(f.head_prefix)
            assert f.sfx_prefix.startswith(f.head_prefix)


# ---------------------------------------------------------------------------
# Directory-level integration test
# ---------------------------------------------------------------------------


@pytest.fixture
def ethos_like_dataset(tmp_path: Path, ethos_like_events: pl.DataFrame) -> Path:
    """Write a one-shard MEDS-style dataset under ``tmp_path/ethos/`` for integration testing."""
    root = tmp_path / "ethos"
    (root / "data" / "held_out").mkdir(parents=True)
    ethos_like_events.write_parquet(root / "data" / "held_out" / "0.parquet")
    # Also write a ``codes.parquet`` listing every code seen, so we can verify metadata
    # rewriting works end-to-end.
    codes = ethos_like_events["code"].unique().to_frame()
    (root / "metadata").mkdir(parents=True)
    codes.write_parquet(root / "metadata" / "codes.parquet")
    return root


def test_reprocess_directory_writes_expected_layout(ethos_like_dataset: Path, tmp_path: Path):
    out_dir = tmp_path / "eq_input"
    reprocess_directory(ethos_like_dataset, out_dir)
    assert (out_dir / "data" / "held_out" / "0.parquet").is_file()
    assert (out_dir / "metadata" / "ethos_code_mapping.parquet").is_file()
    assert (out_dir / "metadata" / "codes.parquet").is_file()


def test_reprocess_directory_codes_metadata_drops_mid_sfx(ethos_like_dataset: Path, tmp_path: Path):
    out_dir = tmp_path / "eq_input"
    reprocess_directory(ethos_like_dataset, out_dir)
    out_codes = pl.read_parquet(out_dir / "metadata" / "codes.parquet")["code"].to_list()
    # No code in the rewritten vocab should be a bare mid/sfx token.
    for c in out_codes:
        assert not c.startswith("ICD//CM//3-6//"), f"mid token {c!r} leaked into codes.parquet"
        assert not c.startswith("ICD//CM//SFX//"), f"sfx token {c!r} leaked into codes.parquet"
        assert not c.startswith("ATC//4//"), f"mid token {c!r} leaked into codes.parquet"
        assert not c.startswith("ATC//SFX//"), f"sfx token {c!r} leaked into codes.parquet"


def test_reprocess_directory_refuses_to_clobber(ethos_like_dataset: Path, tmp_path: Path):
    out_dir = tmp_path / "eq_input"
    out_dir.mkdir()
    (out_dir / "decoy.txt").write_text("don't clobber me")
    with pytest.raises(FileExistsError, match="overwrite=true"):
        reprocess_directory(ethos_like_dataset, out_dir, overwrite=False)


def test_reprocess_directory_overwrite_clobbers(ethos_like_dataset: Path, tmp_path: Path):
    out_dir = tmp_path / "eq_input"
    out_dir.mkdir()
    (out_dir / "stale.txt").write_text("stale")
    reprocess_directory(ethos_like_dataset, out_dir, overwrite=True)
    assert not (out_dir / "stale.txt").exists()
    assert (out_dir / "data" / "held_out" / "0.parquet").is_file()


def test_reprocess_directory_rejects_in_place_rewrite(ethos_like_dataset: Path):
    """Refuse if input_dir == output_dir to avoid clobbering source data."""
    # The CLI guard for in-place rewrite is in main(); reprocess_directory itself
    # protects via the FileExistsError when output_dir is non-empty.  This test
    # asserts the directory-API behaviour — same dir means same files exist, so we get
    # FileExistsError without overwrite=true.
    with pytest.raises(FileExistsError):
        reprocess_directory(ethos_like_dataset, ethos_like_dataset, overwrite=False)


def test_reprocess_directory_empty_data_dir_raises(tmp_path: Path):
    src = tmp_path / "empty_ethos"
    (src / "data").mkdir(parents=True)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="No parquet shards"):
        reprocess_directory(src, out)


def test_reprocess_directory_missing_data_dir_raises(tmp_path: Path):
    src = tmp_path / "missing"
    src.mkdir()
    out = tmp_path / "out"
    with pytest.raises(FileNotFoundError, match="Expected MEDS shards"):
        reprocess_directory(src, out)


# ---------------------------------------------------------------------------
# CLI subprocess test
# ---------------------------------------------------------------------------


def test_eq_reprocess_ethos_cli_end_to_end(ethos_like_dataset: Path, tmp_path: Path):
    """Run the registered console script in a subprocess; verify outputs."""
    out_dir = tmp_path / "cli_out"
    env = os.environ.copy()
    env["PATH"] = _VENV_BIN + os.pathsep + env.get("PATH", "")
    result = subprocess.run(
        [
            "EQ_reprocess_ethos",
            f"input_dir={ethos_like_dataset!s}",
            f"output_dir={out_dir!s}",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"EQ_reprocess_ethos failed (rc={result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert (out_dir / "data" / "held_out" / "0.parquet").is_file()
    assert (out_dir / "metadata" / "ethos_code_mapping.parquet").is_file()

    # Sanity: the output has fewer rows than the input.
    inp_rows = pl.read_parquet(ethos_like_dataset / "data" / "held_out" / "0.parquet").height
    out_rows = pl.read_parquet(out_dir / "data" / "held_out" / "0.parquet").height
    assert out_rows < inp_rows, "expected fewer rows after recombination"
