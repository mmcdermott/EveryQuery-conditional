"""Dataset integration: multitask sampler output is a drop-in ``task_labels_dir`` for the memmap dataset.

Lives at the top level because it needs ``tensorized_cohort_dir`` from the root ``conftest.py``.  The
labels are produced by the real Stage 4M kernel (``label_multitask_index`` + the atomic writer +
the driver-owned manifest) against a *synthetic* event table covering the fixture cohort's real
subject IDs, with the vocabulary taken from the cohort's actual ``codes.parquet`` - exactly the
alignment contract the dataset enforces.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from meds import train_split
from meds_torchdata.config import MEDSTorchDataConfig

from every_query.data.multitask_dataset import (
    MANIFEST_NAME,
    MultitaskBoundaryBatch,
    MultitaskBoundaryPytorchDataset,
)
from every_query.generate_tasks import sample_multitask_sequences as sms

_SUBJECTS = [239684, 1195293, 68729, 814703]
_PRED_TIMES: dict[int, datetime] = {
    239684: datetime(2010, 5, 11, 18, 0, tzinfo=UTC),
    1195293: datetime(2010, 6, 20, 20, 30, tzinfo=UTC),
    68729: datetime(2010, 5, 26, 3, 0, tzinfo=UTC),
    814703: datetime(2010, 2, 5, 6, 0, tzinfo=UTC),
}
K = 5


def _naive(pt: datetime) -> datetime:
    return pt.replace(tzinfo=None)


@pytest.fixture(scope="module")
def multitask_labels_dir(tensorized_cohort_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two metadata shards + packed sidecars + manifest under ``{dir}/train``."""
    vocab = sms.build_target_vocabulary(tensorized_cohort_dir)
    cands = vocab.boundary_candidates()
    b1, b2 = cands[0], cands[min(1, len(cands) - 1)]
    rng = np.random.default_rng(0)

    rows = []
    for subj in _SUBJECTS:
        pt = _naive(_PRED_TIMES[subj])
        for d in range(0, 40, 3):
            rows.append(
                {
                    "subject_id": subj,
                    "time": pt + timedelta(days=d),
                    "code": cands[int(rng.integers(0, len(cands)))],
                }
            )
        rows.append({"subject_id": subj, "time": pt + timedelta(days=20), "code": b1})
        rows.append({"subject_id": subj, "time": pt + timedelta(days=100), "code": b2})
    events = pl.DataFrame(rows).with_columns(
        pl.col("time").cast(pl.Datetime("us")), pl.col("subject_id").cast(pl.Int64)
    )

    dist = sms.BoundaryDistribution(K, 1.0, 60.0, "log-uniform", 0.5, tuple(cands), tuple(cands))
    root = tmp_path_factory.mktemp("multitask_labels")
    split_dir = root / train_split
    manifest = sms.write_manifest(split_dir, sms.build_manifest(dist, vocab))
    labeled_dir = root / "_labeled"
    labeled_dir.mkdir()

    shards = {"0": _SUBJECTS[:2], "1": _SUBJECTS[2:]}
    for shard, subjects in shards.items():
        contexts = [(s, _naive(_PRED_TIMES[s])) for s in subjects for _ in range(3)]
        sample = dist.sample(len(contexts), *[np.random.default_rng(i + int(shard)) for i in range(7)])
        index = pl.DataFrame(
            {
                "subject_id": pl.Series([c[0] for c in contexts], dtype=pl.Int64),
                "prediction_time": pl.Series([c[1] for c in contexts], dtype=pl.Datetime("us")),
                "durations": pl.Series(sample.durations.tolist(), dtype=pl.List(pl.Float32)),
                "bound_events": pl.Series(sample.bound_events.tolist(), dtype=pl.List(pl.Utf8)),
                "condition_codes": pl.Series(sample.condition_codes.tolist(), dtype=pl.List(pl.Utf8)),
            }
        )
        shape = (index.height, K, vocab.packed_width)
        tmp = sms._unique_tmp_path(sms.labels_path(split_dir, shard))
        mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=shape)
        metadata, _, _ = sms.label_multitask_index(index, events, vocab, K, chunk_rows=2, out=mm)
        mm.flush()
        del mm
        sms.write_labeled_shard(metadata, split_dir, shard, labels_tmp=tmp)
    assert manifest["num_bounds"] == K
    return root


def _dataset(tensorized_cohort_dir: Path, labels_dir: Path, **kw) -> MultitaskBoundaryPytorchDataset:
    cfg = MEDSTorchDataConfig(
        tensorized_cohort_dir=str(tensorized_cohort_dir),
        task_labels_dir=str(labels_dir),
        max_seq_len=64,
        seq_sampling_strategy="to_end",
        static_inclusion_mode="omit",
        batch_mode="SM",
    )
    return MultitaskBoundaryPytorchDataset(cfg, split=train_split, **kw)


def test_metadata_row_maps_to_matching_packed_row(
    tensorized_cohort_dir: Path, multitask_labels_dir: Path
) -> None:
    ds = _dataset(tensorized_cohort_dir, multitask_labels_dir)
    assert len(ds) == 12
    split_dir = multitask_labels_dir / train_split
    for i in range(len(ds)):
        item = ds[i]
        assert "targets" not in item  # never unpacked in __getitem__
        key, row = item["_source_shard"], item["_source_row"]
        shard = key.split("/")[-1]
        meta = pl.read_parquet(split_dir / f"{shard}.parquet")
        packed = np.load(split_dir / f"{shard}.labels.npy", mmap_mode="r")
        sid, pt = ds.index[i][0], ds.schema_df["prediction_time"][i]
        assert meta["subject_id"][row] == sid and meta["prediction_time"][row] == pt
        expect = np.unpackbits(packed[row], axis=-1, count=ds.vocab_size, bitorder="little").astype(bool)
        got = ds.collate([item]).targets[0].numpy()
        assert np.array_equal(got, expect)
        assert np.array_equal(
            item["q_durations"], np.asarray(meta["durations"][row].to_list(), dtype=np.float32)
        )
        assert item["condition_codes"].tolist() == [ds.code_to_index[c] for c in meta["condition_codes"][row]]
        assert item["condition_answers"].tolist() == meta["condition_answers"][row].to_list()


def test_global_shuffle_preserves_alignment_and_batch_shape(
    tensorized_cohort_dir: Path, multitask_labels_dir: Path
) -> None:
    ds = _dataset(tensorized_cohort_dir, multitask_labels_dir)
    perm = np.random.default_rng(1).permutation(len(ds)).tolist()
    batch = ds.collate([ds[i] for i in perm])
    assert isinstance(batch, MultitaskBoundaryBatch)
    assert batch.targets.shape == (len(perm), K, ds.vocab_size)
    assert batch.targets.dtype == torch.bool
    assert batch.q_durations.shape == (len(perm), K)
    assert batch.q_bound_codes.shape == (len(perm), K)
    assert batch.q_mask.all()
    assert not batch.targets[:, :, 0].any()  # PAD bit
    split_dir = multitask_labels_dir / train_split
    for b, i in enumerate(perm):
        item = ds[i]
        shard = item["_source_shard"].split("/")[-1]
        packed = np.load(split_dir / f"{shard}.labels.npy", mmap_mode="r")[item["_source_row"]]
        expect = np.unpackbits(packed, axis=-1, count=ds.vocab_size, bitorder="little").astype(bool)
        assert np.array_equal(batch.targets[b].numpy(), expect)
        assert (batch.q_bound_codes[b] != 0).tolist() == (batch.q_durations[b] == -1.0).tolist()
    # Issue #22: (B, K-1) conditioning tensors, answers == the matching unpacked target bit.
    assert batch.condition_codes.shape == (len(perm), K - 1) and batch.condition_codes.dtype == torch.long
    assert batch.condition_answers.shape == (len(perm), K - 1) and batch.condition_answers.dtype == torch.bool
    assert (batch.condition_codes > 0).all()
    expect = batch.targets[:, : K - 1].gather(2, batch.condition_codes.unsqueeze(-1)).squeeze(-1)
    assert torch.equal(expect, batch.condition_answers)
    assert batch.condition_answers.any() and not batch.condition_answers.all()


def test_memmaps_are_read_only(tensorized_cohort_dir: Path, multitask_labels_dir: Path) -> None:
    ds = _dataset(tensorized_cohort_dir, multitask_labels_dir)
    ds.collate([ds[0]])
    assert ds._memmaps
    for mm in ds._memmaps.values():
        assert isinstance(mm, np.memmap)
        assert not mm.flags.writeable


def test_manifest_vocab_mismatch_raises(
    tensorized_cohort_dir: Path, multitask_labels_dir: Path, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="vocab_size mismatch"):
        _dataset(tensorized_cohort_dir, multitask_labels_dir, expected_vocab_size=3)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        _dataset(tensorized_cohort_dir, multitask_labels_dir, expected_vocab_fingerprint="0" * 64)

    # A tampered manifest disagrees with the cohort's codes.parquet -> refuse to load.
    import shutil

    bad = tmp_path / "bad_labels"
    shutil.copytree(multitask_labels_dir, bad)
    fp = bad / train_split / MANIFEST_NAME
    manifest = json.loads(fp.read_text())
    manifest["vocab_fingerprint"] = "f" * 64
    fp.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="does not match the multitask manifest"):
        _dataset(tensorized_cohort_dir, bad)


def test_dataset_matches_manifest_vocabulary(tensorized_cohort_dir: Path, multitask_labels_dir: Path) -> None:
    vocab = sms.build_target_vocabulary(tensorized_cohort_dir)
    ds = _dataset(
        tensorized_cohort_dir,
        multitask_labels_dir,
        expected_vocab_size=vocab.size,
        expected_vocab_fingerprint=vocab.fingerprint,
    )
    assert ds.vocab_size == vocab.size and ds.num_bounds == K


def test_end_to_end_cli_run_to_batch_matches_raw_events(
    simple_static_MEDS: Path, tensorized_cohort_dir: Path, tmp_path: Path
) -> None:
    """Raw MEDS events -> ``run()`` -> dataset -> batch: targets and the input window agree with the events.

    Unlike the fixtures above, the labels here come from the real sampler driver over the fixture cohort's own
    events, subjects carry several distinct prediction times, and the expected values are recomputed from the
    raw event table rather than read back from the sidecars.
    """
    from omegaconf import OmegaConf

    from tests.multitask.conftest import scalar_oracle

    out = tmp_path / "mt"
    k = 4
    cfg = OmegaConf.create(
        {
            "data_dir": str(simple_static_MEDS),
            "out_dir": str(out),
            "query_codes": str(tensorized_cohort_dir),
            "split": train_split,
            "seed": 3,
            "num_training_examples": 40,
            "num_bounds": k,
            "duration_min": 0.01,  # the fixture's events span hours, so keep horizons short
            "duration_max": 2.0,
            "duration_distribution": "log-uniform",
            "eventbound_fraction": 0.5,
            "boundary_codes": None,
            "min_prediction_times_per_subject": 1,
            "max_workers": 1,
            "label_chunk_rows": 7,
            "ontology_dir": None,
            "overwrite": False,
        }
    )
    sms.run(cfg)

    events = pl.concat(
        [pl.read_parquet(fp) for fp in sorted((simple_static_MEDS / "data" / train_split).glob("*.parquet"))]
    ).select("subject_id", "time", "code")
    vocab = sms.build_target_vocabulary(tensorized_cohort_dir)
    code_to_index = dict(zip(vocab.codes, vocab.indices, strict=True))

    ds = _dataset(tensorized_cohort_dir, out)
    assert len(ds) == cfg.num_training_examples
    assert ds.schema_df.group_by("subject_id").n_unique()["prediction_time"].max() > 1

    perm = np.random.default_rng(0).permutation(len(ds)).tolist()
    items = [ds[i] for i in perm]
    batch = ds.collate(items)
    meta = ds.schema_df[perm]

    expect = scalar_oracle(meta, events, list(vocab.codes), k)
    assert np.array_equal(batch.targets[:, :, vocab.indices].numpy(), expect)
    assert expect.any() and not expect.all()  # the fixture actually exercises both label values

    # Input window: the tokens the encoder sees are exactly the subject's timed events at or before
    # the prediction time (SM mode, one token per measurement; static rows are omitted).
    for b, (sid, pt) in enumerate(zip(meta["subject_id"], meta["prediction_time"], strict=True)):
        visible = events.filter(
            (pl.col("subject_id") == sid) & pl.col("time").is_not_null() & (pl.col("time") <= pt)
        )
        got = sorted(batch.code[b][batch.code[b] != batch.PAD_INDEX].tolist())
        assert got == sorted(code_to_index[c] for c in visible["code"])


def _tampered(labels_dir: Path, tmp_path: Path, edit) -> Path:
    """Copy the labels dir and rewrite shard ``0``'s metadata through ``edit(df) -> df``."""
    import shutil

    bad = tmp_path / "bad_labels"
    shutil.copytree(labels_dir, bad)
    fp = bad / train_split / "0.parquet"
    edit(pl.read_parquet(fp)).write_parquet(fp)
    return bad


def test_corrupted_condition_answers_fail_in_collate(
    tensorized_cohort_dir: Path, multitask_labels_dir: Path, tmp_path: Path
) -> None:
    flip = lambda df: df.with_columns(pl.col("condition_answers").list.eval(~pl.element()))  # noqa: E731
    ds = _dataset(tensorized_cohort_dir, _tampered(multitask_labels_dir, tmp_path, flip))
    bad_rows = [i for i in range(len(ds)) if ds[i]["_source_shard"].endswith("/0")]
    assert "targets" not in ds[bad_rows[0]]  # __getitem__ still never unpacks
    with pytest.raises(ValueError, match="condition_answers disagree with the packed targets"):
        ds.collate([ds[i] for i in bad_rows])


@pytest.mark.parametrize(
    ("what", "edit"),
    [
        ("exactly", lambda df: df.with_columns(pl.col("condition_codes").list.slice(0, K - 2))),
        ("exactly", lambda df: df.with_columns(pl.col("condition_answers").list.slice(0, 1))),
        (
            "PAD, or not in",
            lambda df: df.with_columns(pl.col("condition_codes").list.eval(pl.lit("NOPE//X"))),
        ),
        (
            "PAD, or not in",
            lambda df: df.with_columns(pl.col("condition_codes").list.eval(pl.lit(None, dtype=pl.Utf8))),
        ),
        ("missing", lambda df: df.drop("condition_codes")),
    ],
)
def test_bad_condition_metadata_fails_at_init(
    tensorized_cohort_dir: Path, multitask_labels_dir: Path, tmp_path: Path, what: str, edit
) -> None:
    bad = _tampered(multitask_labels_dir, tmp_path, edit)
    with pytest.raises(ValueError, match=what):
        _dataset(tensorized_cohort_dir, bad)


# --- issue #24: window starts -----------------------------------------------------------------------


def _windowed_events(tensorized_cohort_dir: Path):
    vocab = sms.build_target_vocabulary(tensorized_cohort_dir)
    cands = vocab.boundary_candidates()
    rng = np.random.default_rng(5)
    rows = []
    for subj in _SUBJECTS:
        pt = _naive(_PRED_TIMES[subj])
        for d in range(0, 60, 2):
            rows.append(
                {
                    "subject_id": subj,
                    "time": pt + timedelta(days=d),
                    "code": cands[int(rng.integers(0, len(cands)))],
                }
            )
    events = pl.DataFrame(rows).with_columns(
        pl.col("time").cast(pl.Datetime("us")), pl.col("subject_id").cast(pl.Int64)
    )
    return vocab, cands, events


def _write_windowed_labels(
    root: Path, tensorized_cohort_dir: Path, *, drop_start_columns_in: set[str] = frozenset()
):
    """Two shards of explicit-start labels; optionally strip the start columns from some shards to emulate
    format-2 files (their bits stay whatever the explicit starts produced: a loading test, not a semantic
    one)."""
    vocab, cands, events = _windowed_events(tensorized_cohort_dir)
    dist = sms.BoundaryDistribution(
        K,
        1.0,
        30.0,
        "log-uniform",
        0.5,
        tuple(cands),
        tuple(cands),
        eventstart_fraction=0.3,
        prediction_time_start_fraction=0.3,
        start_min_duration=1.0,
        start_max_duration=30.0,
        start_event_codes=tuple(cands),
    )
    split_dir = root / train_split
    sms.write_manifest(split_dir, sms.build_manifest(dist, vocab))
    shards = {"0": _SUBJECTS[:2], "1": _SUBJECTS[2:]}
    for shard, subjects in shards.items():
        contexts = [(s, _naive(_PRED_TIMES[s])) for s in subjects for _ in range(4)]
        sample = dist.sample(len(contexts), *[np.random.default_rng(100 + i + int(shard)) for i in range(7)])
        index = pl.DataFrame(
            {
                "subject_id": pl.Series([c[0] for c in contexts], dtype=pl.Int64),
                "prediction_time": pl.Series([c[1] for c in contexts], dtype=pl.Datetime("us")),
                "start_durations": pl.Series(sample.start_durations.tolist(), dtype=pl.List(pl.Float32)),
                "start_events": pl.Series(sample.start_events.tolist(), dtype=pl.List(pl.Utf8)),
                "durations": pl.Series(sample.durations.tolist(), dtype=pl.List(pl.Float32)),
                "bound_events": pl.Series(sample.bound_events.tolist(), dtype=pl.List(pl.Utf8)),
                "condition_codes": pl.Series(sample.condition_codes.tolist(), dtype=pl.List(pl.Utf8)),
            }
        )
        tmp = sms._unique_tmp_path(sms.labels_path(split_dir, shard))
        mm = np.lib.format.open_memmap(
            tmp, mode="w+", dtype=np.uint8, shape=(index.height, K, vocab.packed_width)
        )
        metadata, _, _ = sms.label_multitask_index(index, events, vocab, K, chunk_rows=2, out=mm)
        mm.flush()
        del mm
        if shard in drop_start_columns_in:
            metadata = metadata.drop("start_durations", "start_events")
            import os

            os.replace(tmp, sms.labels_path(split_dir, shard))
            metadata.write_parquet(split_dir / f"{shard}.parquet")  # bypass the v3 schema align
        else:
            sms.write_labeled_shard(metadata, split_dir, shard, labels_tmp=tmp)
    return vocab, events


@pytest.fixture(scope="module")
def windowed_labels_dir(tensorized_cohort_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("windowed_labels")
    _write_windowed_labels(root, tensorized_cohort_dir)
    return root


def test_start_tensors_survive_metadata_loading_and_global_shuffle(
    tensorized_cohort_dir: Path, windowed_labels_dir: Path
) -> None:
    ds = _dataset(tensorized_cohort_dir, windowed_labels_dir)
    assert len(ds) == 16
    perm = np.random.default_rng(2).permutation(len(ds)).tolist()
    batch = ds.collate([ds[i] for i in perm])
    B = len(perm)
    assert batch.q_start_durations.shape == (B, K) and batch.q_start_durations.dtype == torch.float32
    assert batch.q_start_codes.shape == (B, K) and batch.q_start_codes.dtype == torch.int64
    assert batch.q_durations.dtype == torch.float32 and batch.q_bound_codes.dtype == torch.int64
    # Exactly one start and one end representation active per window.
    assert torch.equal(batch.q_start_codes != 0, batch.q_start_durations == -1.0)
    assert (batch.q_start_durations[batch.q_start_codes == 0] >= 0).all()
    assert torch.equal(batch.q_bound_codes != 0, batch.q_durations == -1.0)
    assert (
        (batch.q_start_codes > 0).any()
        and (batch.q_start_durations > 0).any()
        and (batch.q_start_durations == 0).any()
    )
    split_dir = windowed_labels_dir / train_split
    for b, i in enumerate(perm):
        item = ds[i]
        assert "targets" not in item
        shard, row = item["_source_shard"].split("/")[-1], item["_source_row"]
        meta = pl.read_parquet(split_dir / f"{shard}.parquet")
        assert batch.q_start_durations[b].tolist() == pytest.approx(meta["start_durations"][row].to_list())
        expect_codes = [0 if c is None else ds.code_to_index[c] for c in meta["start_events"][row].to_list()]
        assert batch.q_start_codes[b].tolist() == expect_codes
        packed = np.load(split_dir / f"{shard}.labels.npy", mmap_mode="r")[row]
        expect = np.unpackbits(packed, axis=-1, count=ds.vocab_size, bitorder="little").astype(bool)
        assert np.array_equal(batch.targets[b].numpy(), expect)
    expect = batch.targets[:, : K - 1].gather(2, batch.condition_codes.unsqueeze(-1)).squeeze(-1)
    assert torch.equal(expect, batch.condition_answers)


def test_legacy_parquet_without_start_columns_loads_as_zero_starts(
    tensorized_cohort_dir: Path, multitask_labels_dir: Path
) -> None:
    """The module fixture ``multitask_labels_dir`` is written by the current sampler, so strip the start
    columns and the manifest version to emulate a pre-#24 output."""
    import shutil

    legacy = multitask_labels_dir.parent / "legacy_labels"
    if legacy.exists():
        shutil.rmtree(legacy)
    shutil.copytree(multitask_labels_dir, legacy)
    for fp in (legacy / train_split).glob("*.parquet"):
        pl.read_parquet(fp).drop("start_durations", "start_events").write_parquet(fp)
    mfp = legacy / train_split / MANIFEST_NAME
    manifest = json.loads(mfp.read_text())
    manifest["format_version"] = 2
    for key in (
        "window_semantics",
        "start_reference",
        "duration_end_reference",
        "missing_event_start",
        "missing_event_end",
    ):
        manifest.pop(key)
    mfp.write_text(json.dumps(manifest))
    ds = _dataset(tensorized_cohort_dir, legacy)
    batch = ds.collate([ds[i] for i in range(len(ds))])
    assert batch.q_start_durations.dtype == torch.float32 and batch.q_start_codes.dtype == torch.int64
    assert not batch.q_start_durations.any() and not batch.q_start_codes.any()
    assert batch.q_start_durations.shape == (len(ds), K)
    # A manifest without the key at all is legacy too; an unknown version is refused.
    manifest.pop("format_version")
    mfp.write_text(json.dumps(manifest))
    assert _dataset(tensorized_cohort_dir, legacy).manifest.get("format_version") is None
    manifest["format_version"] = 4
    mfp.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="format_version 4"):
        _dataset(tensorized_cohort_dir, legacy)


def test_split_mixing_format_2_and_3_shards_loads(tensorized_cohort_dir: Path, tmp_path: Path) -> None:
    root = tmp_path / "mixed"
    _write_windowed_labels(root, tensorized_cohort_dir, drop_start_columns_in={"0"})
    ds = _dataset(tensorized_cohort_dir, root)
    batch = ds.collate([ds[i] for i in range(len(ds))])
    shard = np.array([ds[i]["_source_shard"].split("/")[-1] for i in range(len(ds))])
    legacy_rows = torch.from_numpy(shard == "0")
    assert legacy_rows.any() and (~legacy_rows).any()
    assert not batch.q_start_durations[legacy_rows].any() and not batch.q_start_codes[legacy_rows].any()
    assert (batch.q_start_durations[~legacy_rows] != 0).any()


def test_batch_with_exactly_one_start_field_raises() -> None:
    kw = {
        "code": torch.tensor([[1, 2], [3, 4]]),
        "numeric_value": torch.zeros(2, 2),
        "numeric_value_mask": torch.zeros(2, 2, dtype=torch.bool),
        "time_delta_days": torch.zeros(2, 2),
        "q_durations": torch.tensor([[30.0, -1.0], [7.0, 2.0]]),
        "q_bound_codes": torch.tensor([[0, 4], [0, 0]]),
        "q_mask": torch.ones(2, 2, dtype=torch.bool),
        "targets": torch.zeros(2, 2, 6, dtype=torch.bool),
        "condition_codes": torch.tensor([[3], [5]]),
        "condition_answers": torch.zeros(2, 1, dtype=torch.bool),
    }
    with pytest.raises(ValueError, match="given together"):
        MultitaskBoundaryBatch(**kw, q_start_durations=torch.zeros(2, 2))
    with pytest.raises(ValueError, match="given together"):
        MultitaskBoundaryBatch(**kw, q_start_codes=torch.zeros(2, 2, dtype=torch.long))
    with pytest.raises(ValueError, match="q_start_codes"):
        MultitaskBoundaryBatch(
            **kw, q_start_durations=torch.zeros(2, 2), q_start_codes=torch.zeros(2, 3, dtype=torch.long)
        )
    batch = MultitaskBoundaryBatch(
        **kw, q_start_durations=torch.zeros(2, 2), q_start_codes=torch.zeros(2, 2, dtype=torch.long)
    )
    assert "q_start_durations" in batch.LABEL_TENSOR_NAMES and "q_start_codes" in batch.LABEL_TENSOR_NAMES


@pytest.mark.parametrize(
    ("what", "edit"),
    [
        (
            "disagree",
            lambda df: df.with_columns(pl.col("start_durations").list.eval(pl.lit(-1.0, dtype=pl.Float32))),
        ),
        (
            "disagree",
            lambda df: df.with_columns(pl.col("start_events").list.eval(pl.lit(None, dtype=pl.Utf8))),
        ),
        ("PAD or not in", lambda df: df.with_columns(pl.col("start_events").list.eval(pl.lit("NOPE//X")))),
        ("exactly", lambda df: df.with_columns(pl.col("start_durations").list.slice(0, K - 1))),
        ("exactly", lambda df: df.with_columns(pl.col("start_events").list.slice(0, 1))),
        ("not all of", lambda df: df.drop("start_events")),
    ],
)
def test_bad_start_metadata_fails_at_init(
    tensorized_cohort_dir: Path, windowed_labels_dir: Path, tmp_path: Path, what: str, edit
) -> None:
    bad = _tampered(windowed_labels_dir, tmp_path, edit)
    with pytest.raises(ValueError, match=what):
        _dataset(tensorized_cohort_dir, bad)


def test_pad_start_code_is_rejected(
    tensorized_cohort_dir: Path, windowed_labels_dir: Path, tmp_path: Path
) -> None:
    ds = _dataset(tensorized_cohort_dir, windowed_labels_dir)
    pad = next((c for c, i in ds.code_to_index.items() if i == 0), None)
    if pad is None:  # PAD and unknown codes share the ``index <= 0`` rejection branch (see the NOPE//X case)
        pytest.skip("the fixture cohort's codes.parquet has no code at vocab index 0")
    bad = _tampered(
        windowed_labels_dir,
        tmp_path,
        lambda df: df.with_columns(
            pl.col("start_durations").list.eval(pl.lit(-1.0, dtype=pl.Float32)),
            pl.col("start_events").list.eval(pl.lit(pad)),
        ),
    )
    with pytest.raises(ValueError, match="PAD or not in"):
        _dataset(tensorized_cohort_dir, bad)


def test_end_to_end_run_with_starts_to_batch_matches_raw_events(
    simple_static_MEDS: Path, tensorized_cohort_dir: Path, tmp_path: Path
) -> None:
    """``run()`` with every start form -> dataset -> collated batch: targets, start tensors and conditioning
    answers recomputed from the raw event table by the naive window oracle; all six combinations reach the
    batch."""
    from omegaconf import OmegaConf

    from tests.multitask.test_multibound_labeling import _naive_from_frames

    out = tmp_path / "mt"
    k = 4
    cfg = OmegaConf.create(
        {
            "data_dir": str(simple_static_MEDS),
            "out_dir": str(out),
            "query_codes": str(tensorized_cohort_dir),
            "split": train_split,
            "seed": 5,
            "num_training_examples": 60,
            "num_bounds": k,
            "duration_min": 0.01,
            "duration_max": 2.0,
            "duration_distribution": "log-uniform",
            "eventbound_fraction": 0.5,
            "boundary_codes": None,
            "eventstart_fraction": 0.3,
            "prediction_time_start_fraction": 0.3,
            "start_duration_min": 0.01,
            "start_duration_max": 1.0,
            "start_duration_distribution": "log-uniform",
            "start_event_codes": None,
            "min_prediction_times_per_subject": 1,
            "max_workers": 1,
            "label_chunk_rows": 3,
            "ontology_dir": None,
            "overwrite": False,
        }
    )
    sms.run(cfg)
    assert json.loads((out / train_split / MANIFEST_NAME).read_text())["format_version"] == 3

    events = pl.concat(
        [pl.read_parquet(fp) for fp in sorted((simple_static_MEDS / "data" / train_split).glob("*.parquet"))]
    ).select("subject_id", "time", "code")
    vocab = sms.build_target_vocabulary(tensorized_cohort_dir)
    ds = _dataset(tensorized_cohort_dir, out)
    assert len(ds) == cfg.num_training_examples
    perm = np.random.default_rng(0).permutation(len(ds)).tolist()
    batch = ds.collate([ds[i] for i in perm])
    meta = ds.schema_df[perm]

    st, en, expect = _naive_from_frames(meta, events, vocab, k)
    assert np.array_equal(batch.targets.numpy(), expect)
    assert expect.any() and not expect.all()
    # Start tensors mirror the metadata, and the six combinations are all present in the batch.
    sd = np.array(meta["start_durations"].to_list(), dtype=np.float32)
    assert np.array_equal(batch.q_start_durations.numpy(), sd)
    codes = np.array(
        [
            [0 if c is None else vocab.code_to_index()[c] for c in row]
            for row in meta["start_events"].to_list()
        ]
    )
    assert np.array_equal(batch.q_start_codes.numpy(), codes)
    kinds = np.where(codes > 0, 2, np.where(sd > 0, 1, 0))
    ends = batch.q_bound_codes.numpy() > 0
    assert {(int(a), bool(b)) for a, b in zip(kinds.ravel(), ends.ravel(), strict=True)} == {
        (s, e) for s in (0, 1, 2) for e in (False, True)
    }
    from every_query.generate_tasks.interval_table import INF

    assert (st == INF).any()  # some event starts never resolve on this small fixture ...
    assert not batch.targets.numpy()[st == INF].any()  # ... and those windows are all-false
    got = batch.targets[:, : k - 1].gather(2, batch.condition_codes.unsqueeze(-1)).squeeze(-1)
    assert torch.equal(got, batch.condition_answers)
