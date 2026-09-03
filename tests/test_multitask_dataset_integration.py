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

    dist = sms.BoundaryDistribution(K, 1.0, 60.0, "log-uniform", 0.5, tuple(cands))
    root = tmp_path_factory.mktemp("multitask_labels")
    split_dir = root / train_split
    manifest = sms.write_manifest(split_dir, sms.build_manifest(dist, vocab))
    labeled_dir = root / "_labeled"
    labeled_dir.mkdir()

    shards = {"0": _SUBJECTS[:2], "1": _SUBJECTS[2:]}
    for shard, subjects in shards.items():
        contexts = [(s, _naive(_PRED_TIMES[s])) for s in subjects for _ in range(3)]
        sample = dist.sample(len(contexts), *[np.random.default_rng(i + int(shard)) for i in range(3)])
        index = pl.DataFrame(
            {
                "subject_id": pl.Series([c[0] for c in contexts], dtype=pl.Int64),
                "prediction_time": pl.Series([c[1] for c in contexts], dtype=pl.Datetime("us")),
                "durations": pl.Series(sample.durations.tolist(), dtype=pl.List(pl.Float32)),
                "bound_events": pl.Series(sample.bound_events.tolist(), dtype=pl.List(pl.Utf8)),
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

    Unlike the fixtures above, the labels here come from the real sampler driver over the fixture
    cohort's own events, subjects carry several distinct prediction times, and the expected values
    are recomputed from the raw event table rather than read back from the sidecars.
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
