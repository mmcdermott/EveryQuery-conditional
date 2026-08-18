"""End-to-end CLI tests for the conditional query-sequence pipeline.

Chain: ``EQ_generate_query_sequences`` → ``EQ_train --config-name=_demo_train_conditional`` →
``EQ_predict_sequences`` → ``EQ_evaluate_sequences``, all as real subprocesses against the
session fixture cohort (mirrors the single-query CLI chain in ``conftest.py``).
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml
from meds import train_split, tuning_split

from conftest import run_and_check

EVAL_V3 = Path(__file__).parent.parent / "scripts" / "eval_v3.py"


def _load_eval_v3():
    """Import ``scripts/eval_v3.py`` as a module; it is a script, not an installed entry point."""
    spec = importlib.util.spec_from_file_location("eval_v3", EVAL_V3)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


NEW_CLIS = [
    "EQ_generate_query_sequences",
    "EQ_generate_evaluation_query_sequences",
    "EQ_predict_sequences",
    "EQ_evaluate_sequences",
]


@pytest.mark.parametrize("cli", NEW_CLIS)
def test_cli_help_exits_zero(cli):
    # run_and_check prepends the active venv's bin to PATH and raises on nonzero exit.
    run_and_check([cli, "--help"], timeout=60.0)


@pytest.fixture(scope="session")
def cq_sequence_tasks_dir(eq_preprocessed_dataset: Path, tmp_path_factory) -> Path:
    """Runs ``EQ_generate_query_sequences`` for train + tuning splits on the fixture cohort."""
    intermediate = eq_preprocessed_dataset.parent / "intermediate"
    out_dir = tmp_path_factory.mktemp("cq_seq_tasks")
    for split in (train_split, tuning_split):
        run_and_check(
            [
                "EQ_generate_query_sequences",
                f"data_dir={intermediate!s}",
                f"out_dir={out_dir!s}",
                f"split={split}",
                "input_shard=0",
                "task_shard=0",
                "n_contexts=8",
                "min_queries=1",
                "max_queries=5",
                "duration_min=1",
                "duration_max=30",
                "min_context_per_subject=1",
                "seed=1",
            ],
            env={"PROCESSED": str(eq_preprocessed_dataset)},
            timeout=120.0,
        )
    return out_dir


def test_generated_sequences_conform(cq_sequence_tasks_dir: Path):
    fp = cq_sequence_tasks_dir / train_split / "0__0000.parquet"
    assert fp.exists()
    df = pl.read_parquet(fp)
    assert df.height > 0
    assert set(df.columns) >= {"subject_id", "prediction_time", "queries", "durations", "answers"}

    lens_q = df["queries"].list.len()
    lens_d = df["durations"].list.len()
    lens_a = df["answers"].list.len()
    assert (lens_q == lens_d).all() and (lens_q == lens_a).all()
    assert int(lens_q.min()) >= 1 and int(lens_q.max()) <= 5

    # Binary observed-occurrence answers: every element non-null, no privileged first query.
    assert df["answers"].explode().null_count() == 0


def test_supplied_contexts_mode(eq_preprocessed_dataset: Path, tmp_path: Path):
    """``contexts_path`` scores a user-supplied index df: N sequences per row, K queries each."""
    intermediate = eq_preprocessed_dataset.parent / "intermediate"
    shard = pl.read_parquet(next((intermediate / "data" / train_split).rglob("*.parquet")))
    cohort = shard.group_by("subject_id").agg(pl.col("time").max().alias("prediction_time")).head(3)
    cohort_fp = tmp_path / "cohort.parquet"
    cohort.write_parquet(cohort_fp)

    out_dir = tmp_path / "out"
    run_and_check(
        [
            "EQ_generate_query_sequences",
            f"data_dir={intermediate!s}",
            f"out_dir={out_dir!s}",
            f"split={train_split}",
            f"contexts_path={cohort_fp!s}",
            "n_replicates=4",
            "min_queries=5",
            "max_queries=5",
            "duration_min=1",
            "duration_max=365",
        ],
        env={"PROCESSED": str(eq_preprocessed_dataset)},
        timeout=120.0,
    )

    df = pl.read_parquet(out_dir / train_split / "cohort__0000.parquet")
    assert df.height == cohort.height * 4
    assert (df["queries"].list.len() == 5).all()
    # Every supplied context is present, each replicated exactly 4x.
    counts = df.group_by("subject_id", "prediction_time").len()
    assert counts.height == cohort.height and (counts["len"] == 4).all()


@pytest.fixture
def cq_cohort_fp(eq_preprocessed_dataset: Path, tmp_path: Path) -> Path:
    """A small supplied cohort index df: one prediction time per subject, on the train split."""
    intermediate = eq_preprocessed_dataset.parent / "intermediate"
    shard = pl.read_parquet(next((intermediate / "data" / train_split).rglob("*.parquet")))
    cohort = shard.group_by("subject_id").agg(pl.col("time").max().alias("prediction_time")).head(3)
    fp = tmp_path / "cohort.parquet"
    cohort.write_parquet(fp)
    return fp


def test_dense_grid_sampled_sequences(eq_preprocessed_dataset: Path, cq_cohort_fp: Path, tmp_path: Path):
    """``EQ_generate_evaluation_query_sequences`` labels the *same* N sequences at every context."""
    intermediate = eq_preprocessed_dataset.parent / "intermediate"
    n_contexts = pl.read_parquet(cq_cohort_fp).height
    out_dir = tmp_path / "out"
    run_and_check(
        [
            "EQ_generate_evaluation_query_sequences",
            f"data_dir={intermediate!s}",
            f"out_dir={out_dir!s}",
            f"split={train_split}",
            f"contexts_path={cq_cohort_fp!s}",
            "n_sequences=3",
            "min_queries=4",
            "max_queries=4",
            "duration_min=1",
            "duration_max=30",
        ],
        env={"PROCESSED": str(eq_preprocessed_dataset)},
        timeout=120.0,
    )

    df = pl.read_parquet(out_dir / train_split / "cohort__sampled3.parquet")
    assert df.height == n_contexts * 3
    assert (df["queries"].list.len() == 4).all()
    assert df["answers"].explode().null_count() == 0

    # The dense property: every context is asked the identical set of 3 sequences.
    spec_str = pl.concat_str(
        pl.col("queries").list.join("|"),
        pl.col("durations").cast(pl.List(pl.Utf8)).list.join("|"),
    )
    per_ctx = df.group_by("subject_id", "prediction_time").agg(spec_str.sort().alias("specs"))
    assert per_ctx.height == n_contexts
    assert per_ctx["specs"].n_unique() == 1, "all contexts must share one set of query sequences"

    # Row order is context-major: row i is (contexts[i // N], specs[i % N]).
    subjects = df["subject_id"].unique(maintain_order=True)
    assert df["subject_id"].to_list() == [s for s in subjects for _ in range(3)]
    first_specs = df.head(3).select("queries", "durations").rows()
    for c in range(n_contexts):
        assert df.slice(c * 3, 3).select("queries", "durations").rows() == first_specs


def test_dense_grid_skips_existing_output(eq_preprocessed_dataset: Path, cq_cohort_fp: Path, tmp_path: Path):
    """Re-running without ``overwrite=true`` leaves the existing parquet untouched."""
    intermediate = eq_preprocessed_dataset.parent / "intermediate"
    out_dir = tmp_path / "out"
    cmd = [
        "EQ_generate_evaluation_query_sequences",
        f"data_dir={intermediate!s}",
        f"out_dir={out_dir!s}",
        f"split={train_split}",
        f"contexts_path={cq_cohort_fp!s}",
        "n_sequences=2",
        "min_queries=2",
        "max_queries=2",
        "duration_min=1",
        "duration_max=30",
    ]
    env = {"PROCESSED": str(eq_preprocessed_dataset)}
    run_and_check(cmd, env=env, timeout=120.0)
    fp = out_dir / train_split / "cohort__sampled2.parquet"
    before = fp.stat().st_mtime_ns

    run_and_check(cmd, env=env, timeout=120.0)
    assert fp.stat().st_mtime_ns == before, "second run must skip, not rewrite"

    run_and_check([*cmd, "overwrite=true"], env=env, timeout=120.0)
    assert fp.stat().st_mtime_ns != before, "overwrite=true must regenerate"


def test_dense_grid_designed_sequences_per_spec_dirs(
    eq_preprocessed_dataset: Path, cq_cohort_fp: Path, tmp_path: Path
):
    """Designed named sequences from YAML, one independently-scoreable output dir per task."""
    intermediate = eq_preprocessed_dataset.parent / "intermediate"
    n_contexts = pl.read_parquet(cq_cohort_fp).height
    codes = pl.read_parquet(eq_preprocessed_dataset / "metadata" / "codes.parquet")["code"].to_list()
    target = codes[0]

    sequences_fp = tmp_path / "designed.yaml"
    sequences_fp.write_text(
        yaml.safe_dump(
            {
                "end_then_target": [["TIMELINE//END", 1], [target, 30]],
                "target_only": [[target, 7]],
            }
        )
    )

    out_dir = tmp_path / "out"
    run_and_check(
        [
            "EQ_generate_evaluation_query_sequences",
            f"data_dir={intermediate!s}",
            f"out_dir={out_dir!s}",
            f"split={train_split}",
            f"contexts_path={cq_cohort_fp!s}",
            f"sequences_path={sequences_fp!s}",
            "per_spec_dirs=true",
        ],
        env={"PROCESSED": str(eq_preprocessed_dataset)},
        timeout=120.0,
    )

    # Each spec gets its own `tasks_dir`-shaped directory: {out_dir}/{name}/{split}/tasks.parquet.
    two = pl.read_parquet(out_dir / "end_then_target" / train_split / "tasks.parquet")
    one = pl.read_parquet(out_dir / "target_only" / train_split / "tasks.parquet")
    assert two.height == n_contexts and one.height == n_contexts
    assert two["queries"].to_list() == [["TIMELINE//END", target]] * n_contexts
    assert two["durations"].to_list() == [[1.0, 30.0]] * n_contexts
    assert one["queries"].to_list() == [[target]] * n_contexts
    assert one["answers"].explode().null_count() == 0


def test_dense_grid_rejects_out_of_vocab_code(
    eq_preprocessed_dataset: Path, cq_cohort_fp: Path, tmp_path: Path
):
    """A typo'd designed code must fail before inference, not inside collate mid-run."""
    intermediate = eq_preprocessed_dataset.parent / "intermediate"
    sequences_fp = tmp_path / "designed.yaml"
    sequences_fp.write_text(yaml.safe_dump({"typo": [["NOT_A_REAL_CODE", 30]]}))

    with pytest.raises(RuntimeError, match=r"absent from the query vocabulary"):
        run_and_check(
            [
                "EQ_generate_evaluation_query_sequences",
                f"data_dir={intermediate!s}",
                f"out_dir={tmp_path / 'out'!s}",
                f"split={train_split}",
                f"contexts_path={cq_cohort_fp!s}",
                f"sequences_path={sequences_fp!s}",
            ],
            env={"PROCESSED": str(eq_preprocessed_dataset)},
            timeout=120.0,
        )


@pytest.fixture(scope="session")
def cq_trained_model_dir(
    eq_preprocessed_dataset: Path, cq_sequence_tasks_dir: Path, tmp_path_factory
) -> Path:
    output_dir = tmp_path_factory.mktemp("cq_train_out")
    run_and_check(
        [
            "EQ_train",
            "--config-name=_demo_train_conditional",
            f"output_dir={output_dir!s}",
            f"datamodule.config.tensorized_cohort_dir={eq_preprocessed_dataset!s}",
            f"datamodule.config.task_labels_dir={cq_sequence_tasks_dir!s}",
        ],
        timeout=300.0,
    )
    return output_dir


def test_conditional_train_produces_checkpoint(cq_trained_model_dir: Path):
    assert (cq_trained_model_dir / "checkpoints" / "last.ckpt").exists()
    assert (cq_trained_model_dir / "resolved_config.yaml").exists()


def test_conditional_predict_and_evaluate(
    cq_trained_model_dir: Path, cq_sequence_tasks_dir: Path, tmp_path: Path
):
    # Predict on the tuning split sequences (the fixture cohort has no held_out labels here).
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    src = cq_sequence_tasks_dir / tuning_split / "0__0000.parquet"
    (tasks_dir / "tasks.parquet").write_bytes(src.read_bytes())

    predictions_fp = tmp_path / "predictions.parquet"
    run_and_check(
        [
            "EQ_predict_sequences",
            f"model_run_dir={cq_trained_model_dir!s}",
            f"tasks_dir={tasks_dir!s}",
            f"output_parquet={predictions_fp!s}",
            f"split={tuning_split}",
        ],
        timeout=300.0,
    )

    preds = pl.read_parquet(predictions_fp)
    tasks = pl.read_parquet(tasks_dir / "tasks.parquet")
    assert preds.height == int(tasks["queries"].list.len().sum()), "one output row per query position"
    assert preds["answer_prob"].is_between(0.0, 1.0).all()
    # Binary observed-occurrence answers are always present (never null).
    assert preds["answer"].null_count() == 0

    metrics_stem = tmp_path / "metrics"
    run_and_check(
        [
            "EQ_evaluate_sequences",
            f"predictions_parquet={predictions_fp!s}",
            f"metrics_stem={metrics_stem!s}",
        ],
        timeout=120.0,
    )
    by_pos = pl.read_parquet(metrics_stem.with_suffix(".by_position.parquet"))
    by_query = pl.read_parquet(metrics_stem.with_suffix(".by_query.parquet"))
    assert by_pos.height >= 1 and by_pos["position"].min() == 0
    assert {"n_rows", "n_observed", "prevalence", "auroc"} <= set(by_pos.columns)
    assert by_query.height >= 1


def test_dense_grid_is_scoreable_by_predict_sequences(
    cq_trained_model_dir: Path, eq_preprocessed_dataset: Path, tmp_path: Path
):
    """The dense grid feeds ``EQ_predict_sequences`` directly: N sequences x every cohort context."""
    intermediate = eq_preprocessed_dataset.parent / "intermediate"
    shard = pl.read_parquet(next((intermediate / "data" / tuning_split).rglob("*.parquet")))
    cohort = shard.group_by("subject_id").agg(pl.col("time").max().alias("prediction_time")).head(3)
    cohort_fp = tmp_path / "cohort.parquet"
    cohort.write_parquet(cohort_fp)

    tasks_dir = tmp_path / "tasks"
    run_and_check(
        [
            "EQ_generate_evaluation_query_sequences",
            f"data_dir={intermediate!s}",
            f"out_dir={tasks_dir!s}",
            f"split={tuning_split}",
            f"contexts_path={cohort_fp!s}",
            "n_sequences=2",
            "min_queries=3",
            "max_queries=3",
            "duration_min=1",
            "duration_max=30",
        ],
        env={"PROCESSED": str(eq_preprocessed_dataset)},
        timeout=120.0,
    )
    tasks_fp = tasks_dir / tuning_split / "cohort__sampled2.parquet"
    tasks = pl.read_parquet(tasks_fp)
    assert tasks.height == cohort.height * 2

    predictions_fp = tmp_path / "predictions.parquet"
    run_and_check(
        [
            "EQ_predict_sequences",
            f"model_run_dir={cq_trained_model_dir!s}",
            f"tasks_dir={tasks_dir!s}",
            f"output_parquet={predictions_fp!s}",
            f"split={tuning_split}",
        ],
        timeout=300.0,
    )

    preds = pl.read_parquet(predictions_fp)
    # One row per (sequence, position); no cohort row silently dropped by the schema_df semi-join.
    assert preds.height == cohort.height * 2 * 3
    assert preds["answer_prob"].is_between(0.0, 1.0).all()
    # Each of the 2 shared sequences is scored once per context, so every (query, duration) pair
    # appears exactly `n_contexts` times.
    per_query = preds.group_by("query", "duration_days", "position").len()
    assert (per_query["len"] == cohort.height).all()


def test_eval_v3_scores_supplied_sequences(
    cq_trained_model_dir: Path, eq_preprocessed_dataset: Path, tmp_path: Path
):
    """``scripts/eval_v3.py`` score-last inference over a supplied QuerySeqSchema parquet.

    Uses ``contexts_path`` mode so the tasks carry replicates, exercising the alignment check and
    the extra-column passthrough that make positional attachment of probabilities sound.
    """
    intermediate = eq_preprocessed_dataset.parent / "intermediate"
    shard = pl.read_parquet(next((intermediate / "data" / tuning_split).rglob("*.parquet")))
    cohort = shard.group_by("subject_id").agg(pl.col("time").max().alias("prediction_time")).head(3)
    cohort_fp = tmp_path / "cohort.parquet"
    cohort.write_parquet(cohort_fp)

    tasks_dir = tmp_path / "tasks"
    run_and_check(
        [
            "EQ_generate_query_sequences",
            f"data_dir={intermediate!s}",
            f"out_dir={tasks_dir!s}",
            f"split={tuning_split}",
            f"contexts_path={cohort_fp!s}",
            "n_replicates=2",
            "min_queries=3",
            "max_queries=3",
            "duration_min=1",
            "duration_max=30",
        ],
        env={"PROCESSED": str(eq_preprocessed_dataset)},
        timeout=120.0,
    )
    tasks_fp = tasks_dir / tuning_split / "cohort__0000.parquet"
    n_seqs = pl.read_parquet(tasks_fp).height

    out_dir = tmp_path / "eval_v3_out"
    script = EVAL_V3
    run_and_check(
        [
            "python",
            str(script),
            "--tasks",
            str(tasks_dir),
            "--run-dir",
            str(cq_trained_model_dir),
            "--cohort-dir",
            str(eq_preprocessed_dataset),
            "--split",
            tuning_split,
            "--out-dir",
            str(out_dir),
            "--batch-size",
            "4",
            "--all-positions",
            "--min-pos",
            "1",
            "--min-neg",
            "1",
        ],
        timeout=300.0,
    )

    preds = pl.read_parquet(out_dir / "predictions.parquet")
    assert preds.height == n_seqs, "one row per supplied sequence"
    assert preds["prob"].is_between(0.0, 1.0).all()
    assert preds["true_answer"].null_count() == 0
    assert (preds["n_queries"] == 3).all()
    for col in ("prior_queries", "prior_durations", "prior_answers"):
        assert (preds[col].list.len() == 2).all()
    # The conditional label describes both priors, and is what by_conditional.parquet groups on.
    assert preds["conditional"].null_count() == 0
    assert preds["conditional"].str.count_matches(r"=(YES|NO)").to_list() == [2] * n_seqs
    # --all-positions keeps every position, and the last one is what `prob` reports.
    assert (preds["all_position_probs"].list.len() == 3).all()
    assert preds["all_position_probs"].list.last().to_numpy() == pytest.approx(
        preds["prob"].to_numpy(), abs=1e-6
    )

    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["n_sequences"] == n_seqs
    assert summary["split"] == tuning_split
    assert summary["force_prior"] == "none"
    # Per-conditional accounting is always reported, even when the cohort is too small to score:
    # n_conditionals_total counts every distinct context, scored counts those clearing the gate.
    assert summary["n_conditionals_total"] == preds["conditional"].n_unique()
    assert summary["n_conditionals_scored"] <= summary["n_conditionals_total"]
    if summary["n_conditionals_scored"]:
        by_cond = pl.read_parquet(out_dir / "by_conditional.parquet")
        assert by_cond.height == summary["n_conditionals_scored"]
        assert by_cond["auroc"].is_between(0.0, 1.0).all()

    # Counterfactual conditioning: overriding every prior answer must change the scored
    # distribution (the target query and its context are identical across the two runs).
    forced_dir = tmp_path / "eval_v3_forced"
    run_and_check(
        [
            "python",
            str(script),
            "--tasks",
            str(tasks_dir),
            "--run-dir",
            str(cq_trained_model_dir),
            "--cohort-dir",
            str(eq_preprocessed_dataset),
            "--split",
            tuning_split,
            "--out-dir",
            str(forced_dir),
            "--batch-size",
            "4",
            "--force-prior",
            "yes",
        ],
        timeout=300.0,
    )
    forced = pl.read_parquet(forced_dir / "predictions.parquet")
    assert forced.height == preds.height
    assert forced["target_query"].to_list() == preds["target_query"].to_list()
    assert forced["prob"].is_between(0.0, 1.0).all()
    assert json.loads((forced_dir / "summary.json").read_text())["force_prior"] == "yes"


def _conditional_frame(n_per_combo: int = 30) -> pl.DataFrame:
    """Rows spanning all 4 answer combos of a 2-prior sequence, plus a block of marginals."""
    rng = np.random.default_rng(0)
    combos = [(True, True), (True, False), (False, True), (False, False)]
    rows = []
    for a1, a2 in combos:
        for _ in range(n_per_combo):
            rows.append(
                {
                    "prior_queries": ["A", "B"],
                    "prior_durations": [7.0, 30.0],
                    "prior_answers": [a1, a2],
                    "target_query": "C",
                    "target_duration_days": 30.0,
                    "true_answer": bool(rng.random() < (0.8 if a1 else 0.2)),
                    "prob": float(rng.random()),
                }
            )
    for _ in range(n_per_combo):
        rows.append(
            {
                "prior_queries": [],
                "prior_durations": [],
                "prior_answers": [],
                "target_query": "C",
                "target_duration_days": 30.0,
                "true_answer": bool(rng.random() < 0.5),
                "prob": float(rng.random()),
            }
        )
    return pl.DataFrame(rows).with_columns(
        pl.col("prior_durations").cast(pl.List(pl.Float32)),
        pl.col("target_duration_days").cast(pl.Float32),
    )


def test_eval_v3_conditional_labels_are_one_per_context():
    """Each distinct prior (query, duration, answer) tuple-sequence gets its own stable label."""
    ev3 = _load_eval_v3()
    out = ev3.with_conditional(_conditional_frame(), None)

    assert out.height == 5 * 30
    labels = set(out["conditional"].unique().to_list())
    assert labels == {
        ev3.MARGINAL_LABEL,
        "A@7d=YES | B@30d=YES",
        "A@7d=YES | B@30d=NO",
        "A@7d=NO | B@30d=YES",
        "A@7d=NO | B@30d=NO",
    }, "2 conditioned queries ⇒ 4 conditionals; an empty prior list is the marginal"
    # Row order must survive the explode/group_by round trip — probs are attached positionally.
    assert out["prob"].to_list() == _conditional_frame()["prob"].to_list()


def test_eval_v3_force_prior_relabels_the_conditional():
    """Under --force-prior the label must report what the model saw, not the recorded answers."""
    ev3 = _load_eval_v3()
    forced = ev3.with_conditional(_conditional_frame(), True)
    assert set(forced["conditional"].unique().to_list()) == {
        ev3.MARGINAL_LABEL,
        "A@7d=YES | B@30d=YES",
    }, "every conditioned row collapses onto the single forced context"


def test_eval_v3_macro_auroc_splits_by_conditional():
    """Per-conditional grouping yields one AUROC per estimand, and drops thin groups honestly."""
    ev3 = _load_eval_v3()
    out = ev3.with_conditional(_conditional_frame(), None)

    tbl, macro, n_groups = ev3.macro_auroc(out, ev3.CONDITIONAL_KEYS, min_pos=3, min_neg=3)
    assert n_groups == 5 and tbl.height == 5
    assert set(tbl.columns) == {*ev3.CONDITIONAL_KEYS, "n", "prevalence", "auroc"}
    assert (tbl["n"] == 30).all()
    assert macro == pytest.approx(float(tbl["auroc"].mean()))

    # The coarse view pools all 5 contexts into one row — the eval_v2 behaviour being refined.
    qtbl, _, n_q = ev3.macro_auroc(out, ["target_query"], min_pos=3, min_neg=3)
    assert n_q == 1 and qtbl.height == 1 and int(qtbl["n"][0]) == out.height

    # A gate no group can clear scores nothing, rather than raising or reporting a bogus mean.
    empty, none_macro, n_all = ev3.macro_auroc(out, ev3.CONDITIONAL_KEYS, min_pos=10_000, min_neg=1)
    assert empty.height == 0 and none_macro is None and n_all == 5


def test_eval_v3_macro_auroc_never_scores_a_single_class_group():
    """AUROC is undefined on one class, so the floor holds even when the gate is set to 0."""
    ev3 = _load_eval_v3()
    df = pl.DataFrame(
        {
            "target_query": ["A", "A", "B", "B"],
            "true_answer": [True, True, True, False],
            "prob": [0.1, 0.2, 0.9, 0.4],
        }
    )
    tbl, macro, n_groups = ev3.macro_auroc(df, ["target_query"], min_pos=0, min_neg=0)
    assert n_groups == 2
    assert tbl["target_query"].to_list() == ["B"], "the all-positive group A must be skipped"
    assert macro == pytest.approx(1.0)


def test_eval_v3_rejects_wrong_split(
    cq_trained_model_dir: Path, eq_preprocessed_dataset: Path, cq_sequence_tasks_dir: Path, tmp_path: Path
):
    """Subjects absent from ``--split`` are dropped by the schema_df semi-join; that must not be silent."""
    tasks_dir = tmp_path / "tasks"
    (tasks_dir / train_split).mkdir(parents=True)
    src = cq_sequence_tasks_dir / train_split / "0__0000.parquet"
    (tasks_dir / train_split / "tasks.parquet").write_bytes(src.read_bytes())

    script = EVAL_V3
    with pytest.raises(RuntimeError, match=r"(?s)Dataset has .* rows but the supplied parquet"):
        run_and_check(
            [
                "python",
                str(script),
                "--tasks",
                str(tasks_dir),
                "--run-dir",
                str(cq_trained_model_dir),
                "--cohort-dir",
                str(eq_preprocessed_dataset),
                "--split",
                tuning_split,  # train-split subjects are not in tuning
                "--out-dir",
                str(tmp_path / "out"),
            ],
            timeout=300.0,
        )
