"""Stage 0-4M orchestration tests for ``sample_multitask_sequences`` on a synthetic cohort."""

import json
import multiprocessing
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

from every_query.data.schema import MultitaskBoundarySchema
from every_query.generate_tasks import sample_multitask_sequences as sms
from every_query.generate_tasks.sample_multitask_sequences import (
    LABELS_SUFFIX,
    MANIFEST_NAME,
    BoundaryDistribution,
    TargetVocabulary,
    build_target_vocabulary,
    label_one_multitask_shard,
    read_manifest,
)
from every_query.generate_tasks.sample_tasks import INDEX_DIRNAME, LABELED_DIRNAME, default_artifacts_dir
from tests.multitask.conftest import (
    CODES,
    K,
    base_cfg,
    condition_answers_oracle,
    make_codes_parquet,
    make_index,
    scalar_oracle,
)


def _run(cohort: Path, out_dir: Path, **overrides) -> dict:
    cfg = OmegaConf.create(base_cfg(cohort, out_dir, **overrides))
    sms.run(cfg)
    return cfg


def _load_split(out_dir: Path, vocab: TargetVocabulary) -> dict[str, tuple[pl.DataFrame, np.ndarray]]:
    res = {}
    for fp in sorted((out_dir / "train").glob("*.parquet")):
        packed = np.load(out_dir / "train" / f"{fp.stem}{LABELS_SUFFIX}", mmap_mode="r")
        res[fp.stem] = (pl.read_parquet(fp), np.asarray(packed))
    return res


# --- Stage 1M -------------------------------------------------------------------------------------


def _dist(**kw) -> BoundaryDistribution:
    base = {
        "num_bounds": K,
        "min_duration": 1.0,
        "max_duration": 100.0,
        "duration_distribution": "log-uniform",
        "eventbound_fraction": 0.5,
        "boundary_codes": tuple(CODES),
        "condition_codes": tuple(CODES),
    }
    base.update(kw)
    return BoundaryDistribution(**base)


def _rngs(seed: int = 0, condition_seed: int | None = None, start_seed: int | None = None):
    """The seven caller-owned generators of ``BoundaryDistribution.sample``: three end axes, the conditioning
    axis, then the three start axes (issue #24)."""
    rngs = [np.random.default_rng(seed * 10 + i) for i in range(3)]
    rngs.append(np.random.default_rng(seed * 10 + 3 if condition_seed is None else condition_seed))
    base = seed * 10 if start_seed is None else start_seed
    rngs.extend(np.random.default_rng(base + i) for i in range(4, 7))
    return rngs


def test_boundary_distribution_fixed_seed_reproducible() -> None:
    a = _dist().sample(50, *_rngs())
    b = _dist().sample(50, *_rngs())
    assert np.array_equal(a.durations, b.durations)
    assert np.array_equal(a.bound_events, b.bound_events)
    assert np.array_equal(a.condition_codes, b.condition_codes)
    assert a.durations.shape == (50, K) and a.durations.dtype == np.float32
    assert a.condition_codes.shape == (50, K - 1)
    assert set(a.condition_codes.ravel().tolist()) <= set(CODES)
    assert any(len(set(row)) < K - 1 for row in a.condition_codes.tolist())  # repeats allowed
    assert ((a.durations == -1.0) == (a.bound_events != None)).all()  # noqa: E711


def test_boundary_distribution_streams_are_independent() -> None:
    ref = _dist().sample(200, *_rngs())
    # Changing the event fraction perturbs neither the durations nor the codes of the other slots.
    other = _dist(eventbound_fraction=0.2).sample(200, *_rngs())
    common_dur = (ref.bound_events == None) & (other.bound_events == None)  # noqa: E711
    common_ev = (ref.bound_events != None) & (other.bound_events != None)  # noqa: E711
    assert common_dur.any() and common_ev.any()
    assert np.array_equal(ref.durations[common_dur], other.durations[common_dur])
    assert np.array_equal(ref.bound_events[common_ev], other.bound_events[common_ev])
    # Changing the duration bounds leaves the forms and the codes untouched.
    wider = _dist(max_duration=1000.0).sample(200, *_rngs())
    assert np.array_equal(ref.bound_events, wider.bound_events)
    # Changing only the code pool leaves the forms and durations untouched.
    pool = _dist(boundary_codes=tuple(CODES[:3])).sample(200, *_rngs())
    assert np.array_equal(ref.durations, pool.durations)
    assert np.array_equal(ref.bound_events == None, pool.bound_events == None)  # noqa: E711
    # None of the boundary-axis changes above perturb the conditioning-code stream ...
    for other_sample in (other, wider, pool):
        assert np.array_equal(ref.condition_codes, other_sample.condition_codes)
    # ... and a different conditioning stream changes no boundary draw.
    cond = _dist().sample(200, *_rngs(condition_seed=999))
    assert not np.array_equal(ref.condition_codes, cond.condition_codes)
    assert np.array_equal(ref.durations, cond.durations)
    assert np.array_equal(ref.bound_events, cond.bound_events)


def test_boundary_distribution_validation() -> None:
    with pytest.raises(ValueError, match="num_bounds"):
        _dist(num_bounds=0)
    with pytest.raises(ValueError, match="eventbound_fraction"):
        _dist(eventbound_fraction=1.5)
    with pytest.raises(ValueError, match="boundary_codes"):
        _dist(boundary_codes=())
    with pytest.raises(ValueError, match="duration_distribution"):
        _dist(duration_distribution="normal")
    with pytest.raises(ValueError, match="condition_codes"):
        _dist(condition_codes=())
    assert _dist(num_bounds=1, condition_codes=()).sample(3, *_rngs()).condition_codes.shape == (3, 0)


# --- vocabulary -----------------------------------------------------------------------------------


def test_vocabulary_is_base_codes_only_and_rejects_lists(tmp_path: Path) -> None:
    fp = make_codes_parquet(tmp_path, ["B", "A", "TIMELINE//END"], first_index=1)
    vocab = build_target_vocabulary(tmp_path)
    assert vocab.codes == ("B", "A", "TIMELINE//END")
    assert vocab.size == 4 and vocab.packed_width == 1
    assert build_target_vocabulary(fp).fingerprint == vocab.fingerprint
    assert not any("//ANY" in c or (c.count("//") == 0 and c not in ("B", "A")) for c in vocab.codes)
    with pytest.raises(ValueError, match="explicit code list"):
        build_target_vocabulary(["A", "B"])
    with pytest.raises(ValueError, match="boundary code"):
        sms.read_boundary_codes(["NOPE"], vocab)
    assert sms.read_boundary_codes(["A", "A", "B"], vocab) == ["A", "B"]


# --- end to end ----------------------------------------------------------------------------------


def test_run_writes_layout_manifest_and_exact_count(synthetic_cohort: Path, tmp_path: Path) -> None:
    out = tmp_path / "mt"
    cfg = _run(synthetic_cohort, out)
    names = sorted(p.name for p in (out / "train").iterdir())
    assert names == ["0.labels.npy", "0.parquet", "1.labels.npy", "1.parquet", MANIFEST_NAME]

    vocab = build_target_vocabulary(synthetic_cohort)
    manifest = read_manifest(out / "train")
    assert manifest["num_bounds"] == K
    assert manifest["vocab_size"] == vocab.size == len(CODES) + 1
    assert manifest["packed_width_bytes"] == vocab.packed_width
    assert manifest["bitorder"] == "little"
    assert manifest["window"] == "open_open"
    assert manifest["missing_event_boundary"] == "infinity"
    assert manifest["datetime_unit"] == "us"
    assert manifest["vocab_fingerprint"] == vocab.fingerprint
    assert manifest["ontology_mode"] == "none"
    assert manifest["format_version"] == 3
    assert manifest["num_condition_codes"] == K - 1
    # Issue #24 semantics keys (the legacy ``window`` / ``missing_event_boundary`` keys are retained).
    assert manifest["window_semantics"] == "open_open"
    assert manifest["start_reference"] == "prediction_time"
    assert manifest["duration_end_reference"] == "resolved_start"
    assert manifest["missing_event_start"] == "empty_window"
    assert manifest["missing_event_end"] == "infinity"
    assert manifest["event_bound_duration_sentinel"] == -1.0

    shards = _load_split(out, vocab)
    total = 0
    for shard, (meta, packed) in shards.items():
        MultitaskBoundarySchema.validate(meta.to_arrow())
        assert packed.shape == (meta.height, K, vocab.packed_width)
        assert packed.dtype == np.uint8
        total += meta.height
        # Subject-shard partitioning: every context's subject lives in this shard's event file.
        events = pl.read_parquet(synthetic_cohort / "data" / "train" / f"{shard}.parquet")
        assert set(meta["subject_id"].to_list()) <= set(events["subject_id"].to_list())
        # Sorted by subject/time, sentinel representation consistent.
        assert meta.sort("subject_id", "prediction_time").equals(meta)
        d = meta["durations"].explode()
        b = meta["bound_events"].explode()
        assert ((d == -1.0) == b.is_not_null()).all()
        # Differential check of every bit against the scalar oracle.
        dense = np.unpackbits(packed, axis=-1, count=vocab.size, bitorder="little").astype(bool)
        assert dense[:, :, 0].sum() == 0
        assert np.array_equal(dense[:, :, vocab.indices], scalar_oracle(meta, events, list(vocab.codes), K))
        # Issue #22: exactly K-1 non-PAD base-vocabulary conditioning codes, answers == target bits.
        assert (meta["condition_codes"].list.len() == K - 1).all()
        assert (meta["condition_answers"].list.len() == K - 1).all()
        assert set(meta["condition_codes"].explode().to_list()) <= set(vocab.boundary_candidates())
        answers = np.array(meta["condition_answers"].to_list(), dtype=bool)
        assert np.array_equal(answers, condition_answers_oracle(meta, dense, vocab))
    assert total == cfg.num_training_examples


def test_fixed_seed_reproducibility_and_reuse(synthetic_cohort: Path, tmp_path: Path, monkeypatch) -> None:
    vocab = build_target_vocabulary(synthetic_cohort)
    _run(synthetic_cohort, tmp_path / "a")
    _run(synthetic_cohort, tmp_path / "b")
    a, b = _load_split(tmp_path / "a", vocab), _load_split(tmp_path / "b", vocab)
    assert a.keys() == b.keys()
    for shard in a:
        assert a[shard][0].equals(b[shard][0])
        assert np.array_equal(a[shard][1], b[shard][1])

    # Rerun in place: every shard is reused (worker returns "skipped"), nothing rewritten.
    labels_fp = tmp_path / "a" / "train" / f"0{LABELS_SUFFIX}"
    before = labels_fp.stat().st_mtime_ns
    statuses = {}
    real = sms._label_multitask_shards

    def spy(*args, **kwargs):
        statuses.update(real(*args, **kwargs))
        return statuses

    monkeypatch.setattr(sms, "_label_multitask_shards", spy)
    _run(synthetic_cohort, tmp_path / "a")
    assert set(statuses.values()) == {"skipped"}
    assert labels_fp.stat().st_mtime_ns == before

    # A different seed changes the index fingerprint -> relabel.
    statuses.clear()
    _run(synthetic_cohort, tmp_path / "a", seed=4)
    assert set(statuses.values()) == {"labeled"}


def test_uses_spawn_pool(synthetic_cohort: Path, tmp_path: Path, monkeypatch) -> None:
    seen = []
    real = multiprocessing.get_context

    def spy(method=None):
        seen.append(method)
        return real(method)

    monkeypatch.setattr(sms.multiprocessing, "get_context", spy)
    _run(synthetic_cohort, tmp_path / "mt", num_training_examples=20)
    assert seen == ["spawn"]


def test_ontology_dir_raises_before_stage0(synthetic_cohort: Path, tmp_path: Path) -> None:
    out = tmp_path / "mt"
    with pytest.raises(NotImplementedError, match="observable leaf codes only"):
        _run(synthetic_cohort, out, ontology_dir=str(tmp_path / "onto"))
    assert not out.exists()
    assert not default_artifacts_dir(out).exists()


def test_events_are_not_closure_expanded(synthetic_cohort: Path, tmp_path: Path, monkeypatch) -> None:
    from every_query.data import ontology

    def boom(*args, **kwargs):  # pragma: no cover - the assertion is that this never runs
        raise AssertionError("expand_events_to_query_nodes must not be called by the multitask sampler")

    monkeypatch.setattr(ontology, "expand_events_to_query_nodes", boom)
    seen = []
    real = sms.prepare_events_for_labeling

    def spy(events_df, ontology_dir=None):
        out = real(events_df, ontology_dir)
        seen.append(out is events_df)
        return out

    monkeypatch.setattr(sms, "prepare_events_for_labeling", spy)
    # Run the worker in-process so the monkeypatch is visible (a spawned worker would not see it).
    cfg = OmegaConf.create(base_cfg(synthetic_cohort, tmp_path / "mt", num_training_examples=30))
    monkeypatch.setattr(sms, "_label_multitask_shards", _inprocess_pool)
    sms.run(cfg)
    assert seen and all(seen)


def _inprocess_pool(
    shards,
    index_dir,
    data_dir,
    out_dir,
    labeled_dir,
    codes_source,
    manifest,
    overwrite,
    n_workers,
    chunk_rows,
):
    return {
        s: label_one_multitask_shard(
            s, index_dir, data_dir, out_dir, labeled_dir, codes_source, manifest, overwrite, chunk_rows
        )[1]
        for s in shards
    }


def test_driver_owns_manifest_worker_never_writes_it(synthetic_cohort: Path, tmp_path: Path) -> None:
    out = tmp_path / "mt"
    _run(synthetic_cohort, out, num_training_examples=30)
    split_dir = out / "train"
    manifest = read_manifest(split_dir)
    (split_dir / MANIFEST_NAME).unlink()
    art = default_artifacts_dir(out) / "train"
    label_one_multitask_shard(
        "0",
        art / INDEX_DIRNAME,
        synthetic_cohort / "data" / "train",
        split_dir,
        art / LABELED_DIRNAME,
        str(synthetic_cohort),
        manifest,
        True,
        7,
    )
    assert not (split_dir / MANIFEST_NAME).exists()


def test_interrupted_write_recovery(synthetic_cohort: Path, tmp_path: Path) -> None:
    out = tmp_path / "mt"
    _run(synthetic_cohort, out, num_training_examples=30)
    split_dir = out / "train"
    # Simulate a crash mid-write: an orphan temp labels file and a missing final labels file.
    orphan = split_dir / f".0{LABELS_SUFFIX}.tmp.deadbeef"
    orphan.write_bytes(b"junk")
    (split_dir / f"0{LABELS_SUFFIX}").unlink()
    _run(synthetic_cohort, out, num_training_examples=30)
    assert not orphan.exists()
    assert (split_dir / f"0{LABELS_SUFFIX}").exists()
    vocab = build_target_vocabulary(synthetic_cohort)
    meta, packed = _load_split(out, vocab)["0"]
    assert packed.shape == (meta.height, K, vocab.packed_width)


def test_stale_fingerprint_relabels(synthetic_cohort: Path, tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "mt"
    _run(synthetic_cohort, out, num_training_examples=30)
    art = default_artifacts_dir(out) / "train" / LABELED_DIRNAME
    sidecar = art / "0.json"
    rec = json.loads(sidecar.read_text())
    rec["index_fingerprint"] = "stale"
    sidecar.write_text(json.dumps(rec))
    statuses = {}
    real = sms._label_multitask_shards

    def spy(*args, **kwargs):
        statuses.update(real(*args, **kwargs))
        return statuses

    monkeypatch.setattr(sms, "_label_multitask_shards", spy)
    _run(synthetic_cohort, out, num_training_examples=30)
    assert statuses["0"] == "labeled" and statuses["1"] == "skipped"


def test_config_change_invalidates_labels(synthetic_cohort: Path, tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "mt"
    _run(synthetic_cohort, out, num_training_examples=30)
    statuses = {}
    real = sms._label_multitask_shards

    def spy(*args, **kwargs):
        statuses.update(real(*args, **kwargs))
        return statuses

    monkeypatch.setattr(sms, "_label_multitask_shards", spy)
    # Same index (contexts/forms/codes unchanged) but a different duration distribution: relabel.
    _run(synthetic_cohort, out, num_training_examples=30, duration_distribution="uniform")
    assert set(statuses.values()) == {"labeled"}


def test_vocabulary_change_invalidates_output(synthetic_cohort: Path, tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "mt"
    _run(synthetic_cohort, out, num_training_examples=30)
    old = read_manifest(out / "train")
    # Append a code to the cohort vocabulary: V, fingerprint, packed width may all change.
    make_codes_parquet(synthetic_cohort, [*CODES, "NEW//CODE"])
    statuses = {}
    real = sms._label_multitask_shards

    def spy(*args, **kwargs):
        statuses.update(real(*args, **kwargs))
        return statuses

    monkeypatch.setattr(sms, "_label_multitask_shards", spy)
    _run(synthetic_cohort, out, num_training_examples=30)
    new = read_manifest(out / "train")
    assert new["vocab_fingerprint"] != old["vocab_fingerprint"]
    assert new["vocab_size"] == old["vocab_size"] + 1
    assert set(statuses.values()) == {"labeled"}
    vocab = build_target_vocabulary(synthetic_cohort)
    for _, packed in _load_split(out, vocab).values():
        assert packed.shape[1:] == (K, vocab.packed_width)


def test_worker_rejects_manifest_vocab_mismatch(synthetic_cohort: Path, tmp_path: Path) -> None:
    out = tmp_path / "mt"
    _run(synthetic_cohort, out, num_training_examples=30)
    manifest = dict(read_manifest(out / "train"))
    manifest["vocab_fingerprint"] = "0" * 64
    art = default_artifacts_dir(out) / "train"
    with pytest.raises(ValueError, match="does not match the manifest"):
        label_one_multitask_shard(
            "0",
            art / INDEX_DIRNAME,
            synthetic_cohort / "data" / "train",
            out / "train",
            art / LABELED_DIRNAME,
            str(synthetic_cohort),
            manifest,
            True,
            7,
        )


def test_empty_shard(synthetic_cohort: Path, tmp_path: Path) -> None:
    out = tmp_path / "mt"
    _run(synthetic_cohort, out, num_training_examples=30)
    art = default_artifacts_dir(out) / "train"
    empty = (
        sms.normalize_index(make_index([], []), K)
        .with_columns(pl.Series("_ctx_id", [], dtype=pl.Int64))
        .select(sms.INDEX_COLUMNS)
    )
    empty.write_parquet(art / INDEX_DIRNAME / "9.parquet")
    events_dir = synthetic_cohort / "data" / "train"
    pl.read_parquet(events_dir / "0.parquet").head(0).write_parquet(events_dir / "9.parquet")
    manifest = read_manifest(out / "train")
    _, status, _ = label_one_multitask_shard(
        "9",
        art / INDEX_DIRNAME,
        events_dir,
        out / "train",
        art / LABELED_DIRNAME,
        str(synthetic_cohort),
        manifest,
        False,
        7,
    )
    assert status == "labeled"
    vocab = build_target_vocabulary(synthetic_cohort)
    packed = np.load(out / "train" / f"9{LABELS_SUFFIX}", mmap_mode="r")
    assert packed.shape == (0, K, vocab.packed_width)
    assert pl.read_parquet(out / "train" / "9.parquet").height == 0
    # ... and it is reusable on the next call.
    assert (
        label_one_multitask_shard(
            "9",
            art / INDEX_DIRNAME,
            events_dir,
            out / "train",
            art / LABELED_DIRNAME,
            str(synthetic_cohort),
            manifest,
            False,
            7,
        )[1]
        == "skipped"
    )


def test_index_sorting_gives_stable_ids(tmp_path: Path) -> None:
    idx = make_index(
        [(2, datetime(2024, 1, 2)), (1, datetime(2024, 1, 5)), (1, datetime(2024, 1, 1))], [[(1.0, None)]] * 3
    )
    sorted_idx = sms.sort_index_for_labeling(idx)
    assert sorted_idx["_ctx_id"].to_list() == [2, 1, 0]
    assert sorted_idx["subject_id"].to_list() == [1, 1, 2]


# --- issue #24: window starts -----------------------------------------------------------------------


def _dist24(**kw) -> BoundaryDistribution:
    base = {
        "eventstart_fraction": 0.3,
        "prediction_time_start_fraction": 0.3,
        "start_min_duration": 1.0,
        "start_max_duration": 180.0,
        "start_duration_distribution": "log-uniform",
        "start_event_codes": tuple(CODES),
    }
    base.update(kw)
    return _dist(**base)


def _start_kind(sample) -> np.ndarray:
    """0 = prediction time, 1 = positive duration, 2 = event."""
    ev = sample.start_events != None  # noqa: E711
    return np.where(ev, 2, np.where(sample.start_durations > 0, 1, 0))


def test_start_sampling_forms_and_fixed_seed_reproducibility() -> None:
    a = _dist24().sample(400, *_rngs())
    b = _dist24().sample(400, *_rngs())
    assert np.array_equal(a.start_durations, b.start_durations)
    assert np.array_equal(a.start_events, b.start_events)
    assert a.start_durations.shape == (400, K) and a.start_durations.dtype == np.float32
    assert a.start_events.shape == (400, K)
    # Exactly one start representation per slot; positive durations inside the configured bounds.
    ev = a.start_events != None  # noqa: E711
    assert ((a.start_durations == -1.0) == ev).all()
    pos = a.start_durations[~ev]
    assert ((pos == 0) | ((pos >= 1.0) & (pos <= 180.0))).all()
    assert set(a.start_events[ev].tolist()) <= set(CODES)
    kinds = _start_kind(a)
    frac = np.bincount(kinds.ravel(), minlength=3) / kinds.size
    assert abs(frac[2] - 0.3) < 0.06 and abs(frac[0] - 0.3) < 0.06 and abs(frac[1] - 0.4) < 0.06
    # All six start/end combinations are sampled.
    end_ev = a.bound_events != None  # noqa: E711
    combos = {(int(s), bool(e)) for s, e in zip(kinds.ravel(), end_ev.ravel(), strict=True)}
    assert combos == {(s, e) for s in (0, 1, 2) for e in (False, True)}


def test_legacy_defaults_put_every_start_at_the_prediction_time() -> None:
    s = _dist().sample(50, *_rngs())
    assert (s.start_durations == 0).all() and (s.start_events == None).all()  # noqa: E711


def test_start_streams_are_independent_of_end_and_condition_streams() -> None:
    ref = _dist24().sample(300, *_rngs())
    # Any start-axis change leaves every end and conditioning draw bit-identical ...
    variants = {
        "eventstart_fraction": _dist24(eventstart_fraction=0.0, prediction_time_start_fraction=0.5),
        "prediction_time_start_fraction": _dist24(prediction_time_start_fraction=0.6),
        "start_duration_bounds": _dist24(start_min_duration=5.0, start_max_duration=10.0),
        "start_duration_distribution": _dist24(start_duration_distribution="uniform"),
        "start_event_codes": _dist24(start_event_codes=tuple(CODES[:2])),
    }
    for name, dist in variants.items():
        other = dist.sample(300, *_rngs())
        assert np.array_equal(ref.durations, other.durations), name
        assert np.array_equal(ref.bound_events, other.bound_events), name
        assert np.array_equal(ref.condition_codes, other.condition_codes), name
    # ... a different start seed changes the starts but nothing else ...
    other = _dist24().sample(300, *_rngs(start_seed=777))
    assert not np.array_equal(ref.start_durations, other.start_durations)
    assert np.array_equal(ref.durations, other.durations)
    assert np.array_equal(ref.bound_events, other.bound_events)
    assert np.array_equal(ref.condition_codes, other.condition_codes)
    # ... and end / conditioning changes leave every start draw bit-identical.
    for other in (
        _dist24(eventbound_fraction=0.1).sample(300, *_rngs()),
        _dist24(max_duration=1000.0).sample(300, *_rngs()),
        _dist24(boundary_codes=tuple(CODES[:3])).sample(300, *_rngs()),
        _dist24().sample(300, *_rngs(condition_seed=999)),
    ):
        assert np.array_equal(ref.start_durations, other.start_durations)
        assert np.array_equal(ref.start_events, other.start_events)
    # Within the start axes: the pool only changes which code, the bounds only the positive durations.
    pool = _dist24(start_event_codes=tuple(CODES[:2])).sample(300, *_rngs())
    assert np.array_equal(ref.start_durations, pool.start_durations)
    bounds = _dist24(start_min_duration=5.0, start_max_duration=10.0).sample(300, *_rngs())
    assert np.array_equal(ref.start_events, bounds.start_events)
    assert np.array_equal(ref.start_durations <= 0, bounds.start_durations <= 0)


def test_start_parameter_validation() -> None:
    with pytest.raises(ValueError, match="eventstart_fraction must be >= 0"):
        _dist24(eventstart_fraction=-0.1)
    with pytest.raises(ValueError, match="prediction_time_start_fraction must be >= 0"):
        _dist24(prediction_time_start_fraction=-0.1)
    with pytest.raises(ValueError, match="must be <= 1"):
        _dist24(eventstart_fraction=0.7, prediction_time_start_fraction=0.4)
    with pytest.raises(ValueError, match="start_event_codes"):
        _dist24(start_event_codes=())
    with pytest.raises(ValueError, match="start_min_duration"):
        _dist24(start_min_duration=0.0)
    with pytest.raises(ValueError, match="start_max_duration"):
        _dist24(start_max_duration=0.5)
    with pytest.raises(ValueError, match="start_duration_distribution"):
        _dist24(start_duration_distribution="normal")
    assert _dist24(eventstart_fraction=0.0, start_event_codes=()).sample(3, *_rngs()).start_events.shape == (
        3,
        K,
    )
    assert _dist24(eventstart_fraction=0.6, prediction_time_start_fraction=0.4).eventstart_fraction == 0.6


def test_from_config_start_defaults_and_overrides() -> None:
    legacy = OmegaConf.create({"num_bounds": 3, "duration_min": 1, "duration_max": 10})
    d = BoundaryDistribution.from_config(legacy, CODES, CODES)
    assert d.eventstart_fraction == 0.0 and d.prediction_time_start_fraction == 1.0
    assert (d.start_min_duration, d.start_max_duration, d.start_duration_distribution) == (
        1.0,
        180.0,
        "log-uniform",
    )
    assert d.start_event_codes == ()
    cfg = OmegaConf.create(
        {
            "num_bounds": 3,
            "duration_min": 1,
            "duration_max": 10,
            "eventstart_fraction": 0.2,
            "prediction_time_start_fraction": 0.0,
            "start_duration_min": 2,
            "start_duration_max": 20,
            "start_duration_distribution": "uniform",
        }
    )
    d = BoundaryDistribution.from_config(cfg, CODES, CODES, CODES[:2])
    assert d.eventstart_fraction == 0.2 and d.prediction_time_start_fraction == 0.0
    assert (d.start_min_duration, d.start_max_duration, d.start_duration_distribution) == (
        2.0,
        20.0,
        "uniform",
    )
    assert d.start_event_codes == tuple(CODES[:2])


def test_read_start_event_codes(tmp_path: Path) -> None:
    make_codes_parquet(tmp_path, ["PAD", "A", "B"], first_index=0)
    vocab = build_target_vocabulary(tmp_path)
    assert sms.read_start_event_codes(None, vocab) == ["A", "B"]
    assert sms.read_start_event_codes(["B", "B", "A"], vocab) == ["B", "A"]
    with pytest.raises(ValueError, match="start_event code\\(s\\) are not in the base vocabulary"):
        sms.read_start_event_codes(["NOPE"], vocab)
    with pytest.raises(ValueError, match="PAD"):
        sms.read_start_event_codes(["PAD"], vocab)
    with pytest.raises(ValueError, match="empty"):
        sms.read_start_event_codes([], vocab)


def test_config_fingerprint_covers_every_start_parameter_and_vocab_salt_is_pinned(tmp_path: Path) -> None:
    import hashlib

    make_codes_parquet(tmp_path, ["A", "B"], first_index=1)
    vocab = build_target_vocabulary(tmp_path)
    ref = sms.config_fingerprint(_dist24(), vocab)
    changes = {
        "eventstart_fraction": _dist24(eventstart_fraction=0.31),
        "prediction_time_start_fraction": _dist24(prediction_time_start_fraction=0.31),
        "start_min_duration": _dist24(start_min_duration=2.0),
        "start_max_duration": _dist24(start_max_duration=181.0),
        "start_duration_distribution": _dist24(start_duration_distribution="uniform"),
        "start_event_codes": _dist24(start_event_codes=tuple(CODES[:3])),
        "num_bounds": _dist24(num_bounds=K + 1),
        "eventbound_fraction": _dist24(eventbound_fraction=0.4),
    }
    seen = {ref}
    for name, dist in changes.items():
        fp = sms.config_fingerprint(dist, vocab)
        assert fp not in seen, name
        seen.add(fp)
    assert sms.config_fingerprint(_dist24(), vocab) == ref
    # The vocabulary fingerprint salt is the constant "multitask-vocab-v2", not FORMAT_VERSION.
    assert sms.FORMAT_VERSION == 3 and sms.VOCAB_FINGERPRINT_VERSION == 2
    h = hashlib.sha256(b"multitask-vocab-v2:3\n" + b"1\tA\n" + b"2\tB\n")
    assert vocab.fingerprint == h.hexdigest()
    manifest = sms.build_manifest(_dist24(), vocab)
    assert manifest["format_version"] == 3 and manifest["missing_event_end"] == "infinity"
    assert manifest["missing_event_start"] == "empty_window"
    assert manifest["start_reference"] == "prediction_time"
    assert manifest["duration_end_reference"] == "resolved_start"
    assert manifest["window_semantics"] == "open_open"
    sms.validate_manifest(manifest)
    with pytest.raises(ValueError, match="format_version"):
        sms.validate_manifest({**manifest, "format_version": 2})


def _naive_from_run(meta: pl.DataFrame, events: pl.DataFrame, vocab: TargetVocabulary, k: int):
    from tests.multitask.test_multibound_labeling import _naive_from_frames

    return _naive_from_frames(meta, events, vocab, k)


START_CFG = {
    "eventstart_fraction": 0.3,
    "prediction_time_start_fraction": 0.3,
    "start_duration_min": 1,
    "start_duration_max": 60,
    "start_duration_distribution": "log-uniform",
    "start_event_codes": None,
}


def test_run_with_explicit_starts_labels_all_six_combinations(synthetic_cohort: Path, tmp_path: Path) -> None:
    out = tmp_path / "mt"
    cfg = _run(synthetic_cohort, out, **START_CFG)
    vocab = build_target_vocabulary(synthetic_cohort)
    manifest = read_manifest(out / "train")
    assert manifest["format_version"] == 3
    shards = _load_split(out, vocab)
    total = 0
    combos = set()
    stats = json.loads((default_artifacts_dir(out) / "train" / LABELED_DIRNAME / "0.json").read_text())[
        "stats"
    ]
    for key in (
        "n_event_starts",
        "frac_event_starts_unresolved",
        "frac_empty_windows",
        "mean_positives_per_window",
    ):
        assert key in stats
    assert stats["n_event_starts"] > 0 and 0 < stats["frac_empty_windows"] < 1
    for shard, (meta, packed) in shards.items():
        MultitaskBoundarySchema.validate(meta.to_arrow())
        total += meta.height
        sd = meta["start_durations"].explode()
        se = meta["start_events"].explode()
        assert ((sd == -1.0) == se.is_not_null()).all()
        assert (sd.filter(sd != -1.0) >= 0).all()
        assert set(se.drop_nulls().to_list()) <= set(vocab.boundary_candidates())
        kinds = np.where(se.is_not_null().to_numpy(), 2, np.where(sd.to_numpy() > 0, 1, 0))
        ends = meta["bound_events"].explode().is_not_null().to_numpy()
        combos |= {(int(a), bool(b)) for a, b in zip(kinds, ends, strict=True)}
        events = pl.read_parquet(synthetic_cohort / "data" / "train" / f"{shard}.parquet")
        dense = np.unpackbits(packed, axis=-1, count=vocab.size, bitorder="little").astype(bool)
        assert dense[:, :, 0].sum() == 0
        _, _, expect = _naive_from_run(meta, events, vocab, K)
        assert np.array_equal(dense, expect)
        answers = np.array(meta["condition_answers"].to_list(), dtype=bool)
        assert np.array_equal(answers, condition_answers_oracle(meta, dense, vocab))
    assert total == cfg.num_training_examples
    assert combos == {(s, e) for s in (0, 1, 2) for e in (False, True)}


def test_start_config_change_invalidates_labels_but_not_end_draws(
    synthetic_cohort: Path, tmp_path: Path, monkeypatch
) -> None:
    out = tmp_path / "mt"
    _run(synthetic_cohort, out, num_training_examples=30, **START_CFG)
    vocab = build_target_vocabulary(synthetic_cohort)
    before = {s: m for s, (m, _) in _load_split(out, vocab).items()}
    statuses = {}
    real = sms._label_multitask_shards

    def spy(*args, **kwargs):
        statuses.update(real(*args, **kwargs))
        return statuses

    monkeypatch.setattr(sms, "_label_multitask_shards", spy)
    _run(synthetic_cohort, out, num_training_examples=30, **{**START_CFG, "start_duration_max": 90})
    assert set(statuses.values()) == {"labeled"}
    after = {s: m for s, (m, _) in _load_split(out, vocab).items()}
    for s in before:
        # Same contexts, ends and conditioning codes; only the start draws moved.
        for col in ("subject_id", "prediction_time", "durations", "bound_events", "condition_codes"):
            assert before[s][col].equals(after[s][col]), col
        assert not before[s]["start_durations"].equals(after[s]["start_durations"])
    # Legacy config (no start keys) produces a format-3 manifest with zero starts.
    statuses.clear()
    _run(synthetic_cohort, tmp_path / "legacy", num_training_examples=30)
    meta = next(iter(_load_split(tmp_path / "legacy", vocab).values()))[0]
    assert (meta["start_durations"].explode() == 0).all() and meta[
        "start_events"
    ].explode().null_count() == meta.height * K
