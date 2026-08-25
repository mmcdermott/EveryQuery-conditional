"""Build the designed evaluation-spec YAMLs: 20 tasks of each of the three query types,
at sequence length 1 and length 3.

The three types (the user's three new features, one per query form):
  dur_*  duration-bounded leaf query          [code, duration]
  evt_*  event-bounded query                  [code, -1, bound_event]
  anc_*  DAG/ancestor query                   [ancestor_node, duration]

Tasks are RANDOMLY SAMPLED from the model's own query universe (leaves + ancestor nodes),
subject to a prevalence floor -- a code nobody ever has yields no positives and an undefined
AUROC, which measures nothing.  Within the eligible pool the draw is uniform, so this stays a
random sample of *answerable* queries rather than a hand-picked clinical panel.

At length 3 each task keeps the SAME target at position 2, behind two randomly drawn filler
queries (themselves a random mix of the three forms).  Length 1 and length 3 therefore differ
only in conditioning depth, which is the comparison the experiment is after.

Spec NAMES are category+index only (dur_00 ... anc_19).  The code strings live in the YAML on
disk and are never printed, so downstream metrics tables are safe to read into context.

Writes:
  <out_dir>/designed_len1.yaml
  <out_dir>/designed_len3.yaml
  <out_dir>/task_manifest.parquet   (spec name -> category, for scoring)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
import yaml

N_PER_TYPE = 20
EVENT_BOUND_SENTINEL = -1

# The training sampler excludes TIMELINE ancestors as tautological, so the model never trained
# on them; they are addressable but meaningless.  Keep them out of both targets and boundaries.
TAUTOLOGICAL_PREFIXES = ("TIMELINE",)


def build_node_stats(onto_dir: Path, cohort_dir: Path) -> pl.DataFrame:
    """Per-node (leaf and ancestor) occurrence/subject counts, via the closure."""
    vocab = pl.read_parquet(onto_dir / "ontology_vocab.parquet")
    closure = pl.read_parquet(onto_dir / "event_to_query_nodes.parquet")
    codes = pl.read_parquet(cohort_dir / "metadata" / "codes.parquet").select(
        "code", "code/n_occurrences", "code/n_subjects"
    )

    # Every node's stats = sum over the event codes that map to it (a leaf maps to itself).
    stats = (
        closure.join(codes, left_on="event_code", right_on="code", how="left")
        .fill_null(0)
        .group_by("query_node")
        .agg(
            pl.col("code/n_occurrences").sum().alias("n_occ"),
            # subject counts are NOT additive across descendants (a subject may have several);
            # this is an upper bound, used only for ranking.
            pl.col("code/n_subjects").sum().alias("n_subj_ub"),
            pl.len().alias("n_desc"),
        )
    )
    return vocab.join(stats, left_on="node_name", right_on="query_node", how="left").fill_null(0)


def descendant_sets(onto_dir: Path) -> dict[str, frozenset[str]]:
    closure = pl.read_parquet(onto_dir / "event_to_query_nodes.parquet")
    out: dict[str, set[str]] = {}
    for ev, node in zip(closure["event_code"], closure["query_node"], strict=True):
        out.setdefault(node, set()).add(ev)
    return {k: frozenset(v) for k, v in out.items()}


def eligible(stats: pl.DataFrame, *, ancestors: bool, min_occ: int, min_desc: int = 1) -> pl.DataFrame:
    df = stats.filter(~pl.col("is_observed_code") if ancestors else pl.col("is_observed_code"))
    for p in TAUTOLOGICAL_PREFIXES:
        df = df.filter(~pl.col("node_name").str.starts_with(p))
    df = df.filter(pl.col("n_occ") >= min_occ)
    if ancestors:
        df = df.filter(pl.col("n_desc") >= min_desc)
    return df


def draw_duration(rng: np.random.Generator, lo: float = 1.0, hi: float = 731.0) -> float:
    """Log-uniform over [lo, hi] -- byte-identical in spirit to QueryDistribution."""
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--min-occ-leaf", type=int, default=200_000)
    ap.add_argument("--min-occ-anc", type=int, default=200_000)
    ap.add_argument("--min-occ-bound", type=int, default=1_000_000)
    args = ap.parse_args()

    onto_dir = Path(os.environ["NF_ONTOLOGY_DIR"])
    cohort_dir = Path(os.environ["TENSORIZED_COHORT_DIR"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    stats = build_node_stats(onto_dir, cohort_dir)
    desc = descendant_sets(onto_dir)

    leaf_pool = eligible(stats, ancestors=False, min_occ=args.min_occ_leaf)
    anc_pool = eligible(stats, ancestors=True, min_occ=args.min_occ_anc, min_desc=2)
    bound_pool = eligible(stats, ancestors=True, min_occ=args.min_occ_bound, min_desc=2)

    print(f"eligible leaves   : {leaf_pool.height}")
    print(f"eligible ancestors: {anc_pool.height}")
    print(f"eligible bounds   : {bound_pool.height}")

    if leaf_pool.height < N_PER_TYPE or anc_pool.height < N_PER_TYPE or bound_pool.height < 3:
        print("POOL TOO SMALL -- lower the --min-occ-* thresholds")
        return 1

    # ---- (a) duration-bounded leaf queries -------------------------------------------------
    leaf_names = leaf_pool["node_name"].to_list()
    dur_targets = list(rng.choice(leaf_names, size=N_PER_TYPE, replace=False))
    dur_horizons = [draw_duration(rng) for _ in dur_targets]

    # ---- (c) ancestor queries ---------------------------------------------------------------
    anc_names = anc_pool["node_name"].to_list()
    # The user named HOSPITAL_ADMISSION explicitly -- pin it as anc_00 if it is eligible.
    pinned = [n for n in ("HOSPITAL_ADMISSION",) if n in set(anc_names)]
    rest = [n for n in anc_names if n not in set(pinned)]
    anc_targets = pinned + list(rng.choice(rest, size=N_PER_TYPE - len(pinned), replace=False))
    anc_horizons = [draw_duration(rng) for _ in anc_targets]

    # ---- (b) event-bounded queries -----------------------------------------------------------
    # A query bounded by itself or by one of its own ancestors is unconditionally False.
    # B is an ancestor-or-self of Q  <=>  desc(Q) is a subset of desc(B).
    bound_names = bound_pool["node_name"].to_list()
    evt_pairs: list[tuple[str, str]] = []
    tries = 0
    while len(evt_pairs) < N_PER_TYPE and tries < 20000:
        tries += 1
        q = str(rng.choice(leaf_names))
        b = str(rng.choice(bound_names))
        dq, db = desc.get(q, frozenset()), desc.get(b, frozenset())
        if not dq or not db:
            continue
        if dq <= db:  # boundary is an ancestor-or-self of the query -> always False
            continue
        if (q, b) in evt_pairs:
            continue
        evt_pairs.append((q, b))
    if len(evt_pairs) < N_PER_TYPE:
        print(f"only found {len(evt_pairs)} valid event-bound pairs")
        return 1

    # ---- assemble the target entry for each spec ---------------------------------------------
    targets: dict[str, list] = {}
    for i, (c, d) in enumerate(zip(dur_targets, dur_horizons, strict=True)):
        targets[f"dur_{i:02d}"] = [str(c), round(float(d), 4)]
    for i, (q, b) in enumerate(evt_pairs):
        targets[f"evt_{i:02d}"] = [str(q), EVENT_BOUND_SENTINEL, str(b)]
    for i, (c, d) in enumerate(zip(anc_targets, anc_horizons, strict=True)):
        targets[f"anc_{i:02d}"] = [str(c), round(float(d), 4)]

    assert len(targets) == 3 * N_PER_TYPE, len(targets)

    # ---- length 1 ------------------------------------------------------------------------------
    len1 = {name: [entry] for name, entry in targets.items()}

    # ---- length 3: two RANDOM filler queries, then the target at position 2 --------------------
    # Fillers are drawn from the same universe with the same three forms, so a length-3 sequence
    # is an in-distribution random sequence whose last position is the task under test.
    def random_filler() -> list:
        form = rng.integers(0, 3)
        if form == 0:
            return [str(rng.choice(leaf_names)), round(draw_duration(rng), 4)]
        if form == 1:
            return [str(rng.choice(anc_names)), round(draw_duration(rng), 4)]
        for _ in range(200):
            q = str(rng.choice(leaf_names))
            b = str(rng.choice(bound_names))
            dq, db = desc.get(q, frozenset()), desc.get(b, frozenset())
            if dq and db and not (dq <= db):
                return [q, EVENT_BOUND_SENTINEL, b]
        return [str(rng.choice(leaf_names)), round(draw_duration(rng), 4)]

    len3 = {name: [random_filler(), random_filler(), entry] for name, entry in targets.items()}

    (out_dir / "designed_len1.yaml").write_text(yaml.safe_dump(len1, sort_keys=True))
    (out_dir / "designed_len3.yaml").write_text(yaml.safe_dump(len3, sort_keys=True))

    manifest = pl.DataFrame(
        {
            "spec": sorted(targets),
            "category": [s.split("_")[0] for s in sorted(targets)],
        }
    ).with_columns(pl.arange(0, 3 * N_PER_TYPE).alias("spec_idx"))
    manifest.write_parquet(out_dir / "task_manifest.parquet")

    # ---- report AGGREGATES only ------------------------------------------------------------------
    print(f"\nwrote {out_dir/'designed_len1.yaml'} ({len(len1)} specs, length 1)")
    print(f"wrote {out_dir/'designed_len3.yaml'} ({len(len3)} specs, length 3)")
    print(f"wrote {out_dir/'task_manifest.parquet'}")
    print("\n--- horizon distribution (days) ---")
    for label, hs in (("dur_", dur_horizons), ("anc_", anc_horizons)):
        a = np.array(hs)
        print(f"  {label}: min={a.min():.1f} median={np.median(a):.1f} max={a.max():.1f}")
    print("\n--- eligibility (n_occurrences of the drawn nodes) ---")
    for label, names in (("dur_", dur_targets), ("anc_", anc_targets),
                         ("evt_ target", [q for q, _ in evt_pairs]),
                         ("evt_ bound", [b for _, b in evt_pairs])):
        sel = stats.filter(pl.col("node_name").is_in([str(x) for x in names]))["n_occ"]
        print(f"  {label:12s}: n={sel.len()} min_occ={sel.min()} median_occ={sel.median()} max_occ={sel.max()}")
    print("\n--- pinned ---")
    print(f"  HOSPITAL_ADMISSION pinned as anc_00: {bool(pinned)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
