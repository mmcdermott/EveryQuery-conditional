"""Tests for ``every_query.paper_experiments.ethos_compat.reprocess``.

Verifies the recombination is correct, deterministic, preserves through-pass for atomic events, and re-injects
statics when configured.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from every_query.paper_experiments.ethos_compat.reprocess import (
    COMBINED_PREFIX,
    FAMILIES,
    load_static_table,
    make_combined_code,
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
# make_combined_code primitive
# ---------------------------------------------------------------------------


class TestMakeCombinedCode:
    def test_deterministic(self):
        a = make_combined_code(("ICD//CM//I10", "ICD//CM//3-6//00"))
        b = make_combined_code(("ICD//CM//I10", "ICD//CM//3-6//00"))
        assert a == b

    def test_order_invariant(self):
        a = make_combined_code(("ICD//CM//I10", "ICD//CM//3-6//00", "ICD//CM//SFX//A"))
        b = make_combined_code(("ICD//CM//SFX//A", "ICD//CM//I10", "ICD//CM//3-6//00"))
        assert a == b

    def test_unique_per_distinct_input(self):
        a = make_combined_code(("ICD//CM//I10",))
        b = make_combined_code(("ICD//CM//I11",))
        assert a != b

    def test_format(self):
        code = make_combined_code(("ICD//CM//I10",))
        assert code.startswith(COMBINED_PREFIX)
        assert len(code) == len(COMBINED_PREFIX) + 16

    def test_distinct_triplets_collide_negligibly(self):
        # Generate 1k synthetic triplets and confirm no collision (sanity bound).
        seen = {make_combined_code((f"ICD//CM//I{i}",)) for i in range(1000)}
        assert len(seen) == 1000


# ---------------------------------------------------------------------------
# Recombination
# ---------------------------------------------------------------------------


class TestRecombination:
    def test_short_icd_becomes_singleton_hash(self, ethos_like_events):
        out, mapping = reprocess_shard(ethos_like_events)
        s1_t1 = out.filter((pl.col("subject_id") == 1) & (pl.col("time") == datetime(2024, 1, 1, 10)))
        assert s1_t1.height == 1
        assert s1_t1["code"][0] == make_combined_code(("ICD//CM//I10",))
        assert s1_t1["code"][0].startswith(COMBINED_PREFIX)

    def test_five_char_icd_combines_head_and_mid(self, ethos_like_events):
        out, _ = reprocess_shard(ethos_like_events)
        s1_t2 = out.filter(
            (pl.col("subject_id") == 1)
            & (pl.col("time") == datetime(2024, 1, 2, 10))
            & pl.col("code").str.starts_with(COMBINED_PREFIX)
        )
        assert s1_t2.height == 1
        expected = make_combined_code(("ICD//CM//E11", "ICD//CM//3-6//65"))
        assert s1_t2["code"][0] == expected

    def test_full_icd_triplet_combines_to_one_row(self, ethos_like_events):
        out, _ = reprocess_shard(ethos_like_events)
        s1_t3 = out.filter((pl.col("subject_id") == 1) & (pl.col("time") == datetime(2024, 1, 3, 10)))
        assert s1_t3.height == 1
        expected = make_combined_code(("ICD//CM//K70", "ICD//CM//3-6//30", "ICD//CM//SFX//1"))
        assert s1_t3["code"][0] == expected

    def test_full_atc_triplet_combines_to_one_row(self, ethos_like_events):
        out, _ = reprocess_shard(ethos_like_events)
        s1_t4 = out.filter((pl.col("subject_id") == 1) & (pl.col("time") == datetime(2024, 1, 4, 10)))
        assert s1_t4.height == 1
        expected = make_combined_code(("ATC//N02BA01//Acetylsalicylic_Acid", "ATC//4//A", "ATC//SFX//01"))
        assert s1_t4["code"][0] == expected

    def test_co_occurring_icd_codes_pair_by_rank(self, ethos_like_events):
        """Two full ICD triplets at the same (subject, time) recombine to two distinct rows."""
        out, _ = reprocess_shard(ethos_like_events)
        s2_t5 = out.filter(
            (pl.col("subject_id") == 2)
            & (pl.col("time") == datetime(2024, 2, 1, 10))
            & pl.col("code").str.starts_with(COMBINED_PREFIX)
        )
        assert s2_t5.height == 2
        codes = set(s2_t5["code"].to_list())
        expected = {
            make_combined_code(("ICD//CM//I10", "ICD//CM//3-6//00", "ICD//CM//SFX//A")),
            make_combined_code(("ICD//CM//E11", "ICD//CM//3-6//65", "ICD//CM//SFX//1")),
        }
        assert codes == expected

    def test_atomic_events_pass_through_unchanged(self, ethos_like_events):
        out, _ = reprocess_shard(ethos_like_events)
        atomic = out.filter(pl.col("code").is_in(["LAB//Q//CREATININE", "HOSPITAL_ADMISSION"]))
        assert atomic.height == 2
        assert set(atomic["code"].to_list()) == {"LAB//Q//CREATININE", "HOSPITAL_ADMISSION"}

    def test_total_row_count_drops_by_expected_amount(self, ethos_like_events):
        """17 input rows → 8 output rows (1+1+1+1+2+2)."""
        out, _ = reprocess_shard(ethos_like_events)
        assert ethos_like_events.height == 17
        assert out.height == 8

    def test_mapping_records_only_changed_codes(self, ethos_like_events):
        _, mapping = reprocess_shard(ethos_like_events)
        inputs = set(mapping["input_code"].to_list())
        assert "LAB//Q//CREATININE" not in inputs
        assert "HOSPITAL_ADMISSION" not in inputs
        assert "ICD//CM//3-6//65" in inputs
        assert "ICD//CM//SFX//1" in inputs
        assert "ATC//4//A" in inputs
        assert "ATC//SFX//01" in inputs

    def test_mapping_outputs_match_combined_codes(self, ethos_like_events):
        out, mapping = reprocess_shard(ethos_like_events)
        assert mapping.height > 0
        rewritten_codes = set(out["code"].to_list())
        for output_code in mapping["output_code"].unique().to_list():
            assert output_code in rewritten_codes
            assert output_code.startswith(COMBINED_PREFIX)


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

    def test_orphan_mid_without_head_logged_and_dropped(self, caplog):
        events = pl.DataFrame(
            {
                "subject_id": [1, 1],
                "time": [datetime(2024, 1, 1)] * 2,
                "code": ["ICD//CM//3-6//00", "LAB//Q//CREATININE"],
            },
            schema={"subject_id": pl.Int64, "time": pl.Datetime("us"), "code": pl.Utf8},
        )
        with caplog.at_level("WARNING"):
            out, _ = reprocess_shard(events)
        assert out.height == 1
        assert out["code"][0] == "LAB//Q//CREATININE"
        assert any("no head tokens" in rec.message for rec in caplog.records)


class TestFamilyDefinitions:
    def test_families_have_distinct_head_prefixes(self):
        prefixes = [f.head_prefix for f in FAMILIES]
        for i, p in enumerate(prefixes):
            for j, q in enumerate(prefixes):
                if i == j:
                    continue
                assert not p.startswith(q) and not q.startswith(p)

    def test_family_mid_and_sfx_extend_head_prefix(self):
        for f in FAMILIES:
            assert f.mid_prefix.startswith(f.head_prefix)
            assert f.sfx_prefix.startswith(f.head_prefix)


# ---------------------------------------------------------------------------
# Static reinjection
# ---------------------------------------------------------------------------


class TestLoadStaticTable:
    def test_load_parquet(self, tmp_path):
        df = pl.DataFrame({"subject_id": [1, 2, 1], "code": ["GENDER//M", "GENDER//F", "RACE//A"]})
        p = tmp_path / "static.parquet"
        df.write_parquet(p)
        loaded = load_static_table(p)
        assert loaded.height == 3
        assert set(loaded.columns) >= {"subject_id", "code"}

    def test_load_pickle_simple_dict(self, tmp_path):
        data = {1: ["GENDER//M", "RACE//A"], 2: ["GENDER//F"]}
        p = tmp_path / "static.pkl"
        with p.open("wb") as f:
            pickle.dump(data, f)
        loaded = load_static_table(p)
        assert loaded.height == 3
        assert set(loaded["code"].to_list()) == {"GENDER//M", "GENDER//F", "RACE//A"}

    def test_load_pickle_with_numeric_values(self, tmp_path):
        data = {1: [("BMI", 27.5)], 2: [("BMI", 22.1)]}
        p = tmp_path / "static.pkl"
        with p.open("wb") as f:
            pickle.dump(data, f)
        loaded = load_static_table(p)
        assert loaded.height == 2
        assert "numeric_value" in loaded.columns
        assert set(loaded["numeric_value"].to_list()) == {27.5, 22.1}

    def test_load_unknown_extension_raises(self, tmp_path):
        p = tmp_path / "static.txt"
        p.write_text("oops")
        with pytest.raises(ValueError, match="unrecognized extension"):
            load_static_table(p)

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_static_table(tmp_path / "missing.parquet")

    def test_parquet_missing_required_columns_raises(self, tmp_path):
        df = pl.DataFrame({"foo": [1], "bar": ["x"]})
        p = tmp_path / "static.parquet"
        df.write_parquet(p)
        with pytest.raises(ValueError, match="must have"):
            load_static_table(p)


# ---------------------------------------------------------------------------
# Directory-level integration test
# ---------------------------------------------------------------------------


@pytest.fixture
def ethos_like_dataset(tmp_path: Path, ethos_like_events: pl.DataFrame) -> Path:
    """Write a one-shard MEDS-style dataset under ``tmp_path/ethos/`` for integration testing."""
    root = tmp_path / "ethos"
    (root / "data" / "held_out").mkdir(parents=True)
    ethos_like_events.write_parquet(root / "data" / "held_out" / "0.parquet")
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
    for c in out_codes:
        assert not c.startswith("ICD//CM//3-6//")
        assert not c.startswith("ICD//CM//SFX//")
        assert not c.startswith("ATC//4//")
        assert not c.startswith("ATC//SFX//")


def test_reprocess_directory_with_static_reinjection(ethos_like_dataset: Path, tmp_path: Path):
    """Static rows from a parquet land in the output as ``time=null`` rows."""
    static_df = pl.DataFrame(
        {
            "subject_id": [1, 1, 2, 99],  # subject 99 is an orphan (no timeline events)
            "code": ["GENDER//M", "RACE//A", "GENDER//F", "DROPPED//ORPHAN"],
        }
    )
    static_path = tmp_path / "static.parquet"
    static_df.write_parquet(static_path)

    out_dir = tmp_path / "eq_input"
    reprocess_directory(ethos_like_dataset, out_dir, static_data_path=static_path)
    out = pl.read_parquet(out_dir / "data" / "held_out" / "0.parquet")

    # Static rows have time=null.
    static_rows = out.filter(pl.col("time").is_null())
    assert static_rows.height == 3, f"expected 3 reinjected static rows, got {static_rows.height}"
    static_codes = set(static_rows["code"].to_list())
    assert static_codes == {"GENDER//M", "RACE//A", "GENDER//F"}
    assert "DROPPED//ORPHAN" not in static_codes  # orphan dropped

    # Timeline events still present and time is non-null for them.
    timeline = out.filter(pl.col("time").is_not_null())
    assert timeline.height > 0


def test_reprocess_directory_with_static_pickle(ethos_like_dataset: Path, tmp_path: Path):
    static_data = {1: ["GENDER//M"], 2: [("BMI", 25.0)]}
    static_path = tmp_path / "static.pkl"
    with static_path.open("wb") as f:
        pickle.dump(static_data, f)

    out_dir = tmp_path / "eq_input"
    reprocess_directory(ethos_like_dataset, out_dir, static_data_path=static_path)
    out = pl.read_parquet(out_dir / "data" / "held_out" / "0.parquet")
    static_rows = out.filter(pl.col("time").is_null())
    assert static_rows.height == 2


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


def test_reprocess_directory_rejects_in_place_rewrite(ethos_like_dataset: Path):
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


def test_eq_reprocess_ethos_cli_with_statics(ethos_like_dataset: Path, tmp_path: Path):
    static_df = pl.DataFrame({"subject_id": [1, 2], "code": ["GENDER//M", "GENDER//F"]})
    static_path = tmp_path / "static.parquet"
    static_df.write_parquet(static_path)

    out_dir = tmp_path / "cli_out"
    env = os.environ.copy()
    env["PATH"] = _VENV_BIN + os.pathsep + env.get("PATH", "")
    result = subprocess.run(
        [
            "EQ_reprocess_ethos",
            f"input_dir={ethos_like_dataset!s}",
            f"output_dir={out_dir!s}",
            f"static_data_path={static_path!s}",
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
    out = pl.read_parquet(out_dir / "data" / "held_out" / "0.parquet")
    static_rows = out.filter(pl.col("time").is_null())
    assert static_rows.height == 2
