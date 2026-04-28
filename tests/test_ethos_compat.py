"""Tests for ``every_query.paper_experiments.ethos_compat.reprocess``.

Verifies the recombination is correct, deterministic, preserves through-pass for atomic events (including ICD-
PCS sub-tokens which we explicitly do not handle), and re-injects statics when configured.
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
        {"subject_id": 1, "time": datetime(2024, 1, 1, 10), "code": "ICD//CM//I10"},
        {"subject_id": 1, "time": datetime(2024, 1, 2, 10), "code": "ICD//CM//E11"},
        {"subject_id": 1, "time": datetime(2024, 1, 2, 10), "code": "ICD//CM//3-6//65"},
        {"subject_id": 1, "time": datetime(2024, 1, 2, 10), "code": "LAB//Q//CREATININE"},
        {"subject_id": 1, "time": datetime(2024, 1, 3, 10), "code": "ICD//CM//K70"},
        {"subject_id": 1, "time": datetime(2024, 1, 3, 10), "code": "ICD//CM//3-6//30"},
        {"subject_id": 1, "time": datetime(2024, 1, 3, 10), "code": "ICD//CM//SFX//1"},
        {"subject_id": 1, "time": datetime(2024, 1, 4, 10), "code": "ATC//N02BA01//Acetylsalicylic_Acid"},
        {"subject_id": 1, "time": datetime(2024, 1, 4, 10), "code": "ATC//4//A"},
        {"subject_id": 1, "time": datetime(2024, 1, 4, 10), "code": "ATC//SFX//01"},
        {"subject_id": 2, "time": datetime(2024, 2, 1, 10), "code": "ICD//CM//I10"},
        {"subject_id": 2, "time": datetime(2024, 2, 1, 10), "code": "ICD//CM//3-6//00"},
        {"subject_id": 2, "time": datetime(2024, 2, 1, 10), "code": "ICD//CM//SFX//A"},
        {"subject_id": 2, "time": datetime(2024, 2, 1, 10), "code": "ICD//CM//E11"},
        {"subject_id": 2, "time": datetime(2024, 2, 1, 10), "code": "ICD//CM//3-6//65"},
        {"subject_id": 2, "time": datetime(2024, 2, 1, 10), "code": "ICD//CM//SFX//1"},
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
        seen = {make_combined_code((f"ICD//CM//I{i}",)) for i in range(1000)}
        assert len(seen) == 1000


# ---------------------------------------------------------------------------
# Recombination
# ---------------------------------------------------------------------------


class TestRecombination:
    def test_short_icd_becomes_singleton_hash(self, ethos_like_events):
        out, _ = reprocess_shard(ethos_like_events)
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
        assert s1_t2["code"][0] == make_combined_code(("ICD//CM//E11", "ICD//CM//3-6//65"))

    def test_full_icd_triplet_combines_to_one_row(self, ethos_like_events):
        out, _ = reprocess_shard(ethos_like_events)
        s1_t3 = out.filter((pl.col("subject_id") == 1) & (pl.col("time") == datetime(2024, 1, 3, 10)))
        assert s1_t3.height == 1
        assert s1_t3["code"][0] == make_combined_code(("ICD//CM//K70", "ICD//CM//3-6//30", "ICD//CM//SFX//1"))

    def test_full_atc_triplet_combines_to_one_row(self, ethos_like_events):
        out, _ = reprocess_shard(ethos_like_events)
        s1_t4 = out.filter((pl.col("subject_id") == 1) & (pl.col("time") == datetime(2024, 1, 4, 10)))
        assert s1_t4.height == 1
        assert s1_t4["code"][0] == make_combined_code(
            ("ATC//N02BA01//Acetylsalicylic_Acid", "ATC//4//A", "ATC//SFX//01")
        )

    def test_co_occurring_icd_codes_pair_by_rank(self, ethos_like_events):
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

    def test_mapping_has_family_column(self, ethos_like_events):
        _, mapping = reprocess_shard(ethos_like_events)
        assert "family" in mapping.columns
        assert set(mapping["family"].unique().to_list()) <= {"ICD_CM", "ATC"}
        # ICD inputs are tagged ICD_CM, ATC inputs are tagged ATC.
        icd_rows = mapping.filter(pl.col("input_code").str.starts_with("ICD//CM//"))
        atc_rows = mapping.filter(pl.col("input_code").str.starts_with("ATC//"))
        assert icd_rows["family"].unique().to_list() == ["ICD_CM"]
        assert atc_rows["family"].unique().to_list() == ["ATC"]


class TestPCSPassthrough:
    """ICD-PCS char-by-char split is explicitly out of scope (#174 follow-up).

    PCS sub-tokens (``ICD//PCS//*``) don't match any registered family head_prefix, so
    they pass through unchanged — the PR's caveat documents this, and these tests
    pin the behaviour so a future broadening of the head pattern doesn't silently
    change it.
    """

    def test_pcs_codes_pass_through_atomic(self):
        events = pl.DataFrame(
            {
                "subject_id": [1, 1, 1, 1],
                "time": [datetime(2024, 1, 1)] * 4,
                "code": [
                    "ICD//PCS//0",
                    "ICD//PCS//1//F",
                    "ICD//PCS//2//7",
                    "LAB//Q//CREATININE",
                ],
            },
            schema={"subject_id": pl.Int64, "time": pl.Datetime("us"), "code": pl.Utf8},
        )
        out, mapping = reprocess_shard(events)
        # All four rows preserved, none recombined, none in mapping.
        assert out.height == 4
        assert mapping.height == 0
        assert set(out["code"].to_list()) == set(events["code"].to_list())


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
# Static reinjection (parquet only)
# ---------------------------------------------------------------------------


class TestLoadStaticTable:
    def test_load_parquet(self, tmp_path):
        df = pl.DataFrame({"subject_id": [1, 2, 1], "code": ["GENDER//M", "GENDER//F", "RACE//A"]})
        p = tmp_path / "static.parquet"
        df.write_parquet(p)
        loaded = load_static_table(p)
        assert loaded.height == 3
        assert set(loaded.columns) >= {"subject_id", "code"}

    def test_load_parquet_with_numeric_value_column(self, tmp_path):
        df = pl.DataFrame({"subject_id": [1, 2], "code": ["BMI", "BMI"], "numeric_value": [27.5, 22.1]})
        p = tmp_path / "static.parquet"
        df.write_parquet(p)
        loaded = load_static_table(p)
        assert "numeric_value" in loaded.columns
        assert loaded["numeric_value"].to_list() == [27.5, 22.1]

    def test_load_ethos_pickle(self, tmp_path):
        """The native Ethos ``StaticDataCollector`` pickle layout — confirmed against MIMIC-IV-demo via
        ``scripts/setup_ethos_demo_data.py``."""
        import pickle

        ethos_pickle = {
            10018845: {
                "BMI": {"code": ["BMI//UNKNOWN"], "time": None},
                "GENDER": {"code": ["GENDER//M"], "time": [None]},
                "MARITAL": {"code": ["MARITAL//MARRIED"], "time": [6777484080000000]},
                "MEDS_BIRTH": {"code": ["MEDS_BIRTH"], "time": [3881606400000000]},
                "RACE": {"code": ["RACE//WHITE"], "time": [6777484080000000]},
            },
            10003046: {
                "BMI": {"code": ["BMI//Q3"], "time": None},
                "GENDER": {"code": ["GENDER//M"], "time": [None]},
            },
        }
        p = tmp_path / "static_data.pickle"
        with p.open("wb") as f:
            pickle.dump(ethos_pickle, f)
        loaded = load_static_table(p)
        # 5 entries for subject 10018845 + 2 for subject 10003046 = 7 rows.
        assert loaded.height == 7
        assert set(loaded.columns) == {"subject_id", "code"}
        codes = set(loaded["code"].to_list())
        assert "BMI//UNKNOWN" in codes
        assert "MEDS_BIRTH" in codes
        assert "GENDER//M" in codes
        # Every code is a string; no `time` column leaked through.
        assert "time" not in loaded.columns

    def test_load_unknown_extension_raises(self, tmp_path):
        p = tmp_path / "static.txt"
        p.write_text("oops")
        with pytest.raises(ValueError, match="unsupported extension"):
            load_static_table(p)

    def test_load_pickle_non_dict_raises(self, tmp_path):
        import pickle

        p = tmp_path / "static.pkl"
        with p.open("wb") as f:
            pickle.dump([1, 2, 3], f)
        with pytest.raises(ValueError, match="expected dict"):
            load_static_table(p)

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_static_table(tmp_path / "missing.parquet")

    def test_parquet_missing_required_columns_raises(self, tmp_path):
        df = pl.DataFrame({"foo": [1], "bar": ["x"]})
        p = tmp_path / "static.parquet"
        df.write_parquet(p)
        with pytest.raises(ValueError, match="missing required column"):
            load_static_table(p)


# ---------------------------------------------------------------------------
# Directory-level integration test
# ---------------------------------------------------------------------------


@pytest.fixture
def ethos_like_dataset(tmp_path: Path, ethos_like_events: pl.DataFrame) -> Path:
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


def test_reprocess_directory_mapping_is_deterministic(ethos_like_dataset: Path, tmp_path: Path):
    """Two runs over identical input produce byte-identical mapping parquets."""
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    reprocess_directory(ethos_like_dataset, out_a)
    reprocess_directory(ethos_like_dataset, out_b)
    a_bytes = (out_a / "metadata" / "ethos_code_mapping.parquet").read_bytes()
    b_bytes = (out_b / "metadata" / "ethos_code_mapping.parquet").read_bytes()
    assert a_bytes == b_bytes


def test_reprocess_directory_with_static_reinjection(ethos_like_dataset: Path, tmp_path: Path):
    static_df = pl.DataFrame(
        {
            "subject_id": [1, 1, 2, 99],
            "code": ["GENDER//M", "RACE//A", "GENDER//F", "DROPPED//ORPHAN"],
        }
    )
    static_path = tmp_path / "static.parquet"
    static_df.write_parquet(static_path)

    out_dir = tmp_path / "eq_input"
    reprocess_directory(ethos_like_dataset, out_dir, static_data_path=static_path)
    out = pl.read_parquet(out_dir / "data" / "held_out" / "0.parquet")

    static_rows = out.filter(pl.col("time").is_null())
    assert static_rows.height == 3
    static_codes = set(static_rows["code"].to_list())
    assert static_codes == {"GENDER//M", "RACE//A", "GENDER//F"}
    assert "DROPPED//ORPHAN" not in static_codes


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
# CLI subprocess tests
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


def test_eq_reprocess_ethos_cli_no_hydra_config_snapshot(ethos_like_dataset: Path, tmp_path: Path):
    """Hydra's ``output_subdir: null`` should suppress the per-run ``.hydra/`` config snapshot.

    Hydra still creates an empty ``outputs/<date>/<time>/`` parent dir under the cwd —
    that's stock behaviour all EQ CLIs share — but the per-run ``.hydra/`` config-snapshot
    subdir (which would clutter every CLI invocation with a copy of the resolved config)
    must not appear.  ``chdir: false`` separately keeps the process cwd unchanged.
    """
    out_dir = tmp_path / "cli_out"
    env = os.environ.copy()
    env["PATH"] = _VENV_BIN + os.pathsep + env.get("PATH", "")
    cwd = tmp_path / "scratch_cwd"
    cwd.mkdir()
    result = subprocess.run(
        [
            "EQ_reprocess_ethos",
            f"input_dir={ethos_like_dataset!s}",
            f"output_dir={out_dir!s}",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
        timeout=120,
    )
    assert result.returncode == 0
    # No `.hydra/` snapshot subdirs anywhere under cwd.
    hydra_snapshots = list(cwd.rglob(".hydra"))
    assert not hydra_snapshots, f"Hydra leaked .hydra/ snapshot dirs: {hydra_snapshots}"


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
    assert result.returncode == 0
    out = pl.read_parquet(out_dir / "data" / "held_out" / "0.parquet")
    static_rows = out.filter(pl.col("time").is_null())
    assert static_rows.height == 2


# ---------------------------------------------------------------------------
# Real-data integration test (gated on ETHOS_DEMO_DIR env var)
# ---------------------------------------------------------------------------


_ETHOS_DEMO_DIR = os.environ.get("ETHOS_DEMO_DIR")
_ETHOS_DEMO_STATIC = os.environ.get("ETHOS_DEMO_STATIC")


@pytest.mark.slow
@pytest.mark.skipif(
    _ETHOS_DEMO_DIR is None,
    reason="ETHOS_DEMO_DIR env var not set — run scripts/setup_ethos_demo_data.py to produce one",
)
def test_real_ethos_output_recombines_and_mtd_ingests(tmp_path: Path):
    """End-to-end test against a real Ethos-tokenized output directory.

    Set ``ETHOS_DEMO_DIR`` to the ``ethos_meds_shape/`` directory produced by
    ``scripts/setup_ethos_demo_data.py``; optionally set ``ETHOS_DEMO_STATIC`` to the
    accompanying ``static_data.pickle`` to also exercise pickle-format static
    reinjection.  Verifies:

      * ``EQ_reprocess_ethos`` runs on the real Ethos output (with statics if provided).
      * Recombined parquets have the expected MEDS schema and contain ``EQ_TOK//*`` codes.
      * If statics are provided, ``time=null`` rows appear with the expected codes.
      * ``MTD_preprocess`` ingests the recombined directory without errors.

    Confirmed against MIMIC-IV-demo-MEDS during PR-#176 development; runs in ~18s.
    """
    ethos_dir = Path(_ETHOS_DEMO_DIR).expanduser().resolve()
    assert ethos_dir.is_dir(), f"ETHOS_DEMO_DIR={ethos_dir} is not a directory"

    static_path: Path | None = None
    if _ETHOS_DEMO_STATIC:
        static_path = Path(_ETHOS_DEMO_STATIC).expanduser().resolve()
        assert static_path.is_file(), f"ETHOS_DEMO_STATIC={static_path} is not a file"

    eq_input_dir = tmp_path / "eq_input"
    reprocess_directory(ethos_dir, eq_input_dir, static_data_path=static_path)

    # Recombined output must have at least one EQ_TOK//* code (confirms split codes in
    # the input were actually collapsed, not just passed through).
    train_shard = pl.read_parquet(eq_input_dir / "data" / "train" / "0.parquet")
    eq_tok_rows = train_shard.filter(pl.col("code").str.starts_with(COMBINED_PREFIX))
    assert eq_tok_rows.height > 0, "no recombined EQ_TOK//* codes — input may not be Ethos output"
    # No leftover split-token tails — recombination is complete.
    for fam in FAMILIES:
        leftovers = train_shard.filter(
            pl.col("code").str.starts_with(fam.mid_prefix) | pl.col("code").str.starts_with(fam.sfx_prefix)
        )
        assert leftovers.height == 0, (
            f"family {fam.name}: {leftovers.height} mid/sfx tokens leaked through recombination"
        )

    # Statics: the pickle yielded a non-empty long-form table and we see time=null rows.
    if static_path is not None:
        all_data = pl.concat(
            [pl.read_parquet(p) for p in (eq_input_dir / "data").rglob("*.parquet")],
            how="diagonal_relaxed",
        )
        static_rows = all_data.filter(pl.col("time").is_null())
        assert static_rows.height > 0, "static_data_path was set but no time=null rows appeared"
        # Real Ethos statics include MEDS_BIRTH for every subject.
        assert "MEDS_BIRTH" in set(static_rows["code"].to_list())

    # Schema sanity for every shard.
    for shard in (eq_input_dir / "data").rglob("*.parquet"):
        df = pl.read_parquet(shard)
        for col in ("subject_id", "time", "code"):
            assert col in df.columns, f"Shard {shard} missing required column {col!r}"

    # The load-bearing assertion: MTD_preprocess can ingest the recombined output.
    mtd_out = tmp_path / "mtd_out"
    env = os.environ.copy()
    env["PATH"] = _VENV_BIN + os.pathsep + env.get("PATH", "")
    result = subprocess.run(
        [
            "MTD_preprocess",
            f"MEDS_dataset_dir={eq_input_dir!s}",
            f"output_dir={mtd_out!s}",
            "do_overwrite=true",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"MTD_preprocess failed (rc={result.returncode}) on real recombined Ethos output.\n"
        f"stdout:\n{result.stdout[-2000:]}\nstderr:\n{result.stderr[-2000:]}"
    )
    # MTD's canonical post-tokenization artifacts.
    for expected in ("metadata/codes.parquet", "tokenization/event_seqs", "tokenization/schemas"):
        assert (mtd_out / expected).exists(), f"MTD output missing {expected}"
