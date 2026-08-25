"""Scan real batches for out-of-range embedding indices.

Training died in the BACKWARD pass with
    RuntimeError: merge_sort: failed to synchronize: cudaErrorIllegalAddress
which is the signature of an out-of-range index reaching an ``nn.Embedding``: the sort inside
``embedding_dense_backward`` is where the bad index finally trips a fault.  The forward pass does
not necessarily raise, because CUDA index errors are asynchronous.

Every index that reaches an embedding in ConditionalQueryModel:

    batch.code           -> the shared token table  (V_ext rows)
    batch.q_codes        -> the same shared table
    batch.q_bound_codes  -> the same shared table
    batch.q_answers      -> answer_embed        (2 rows)
    batch.time_pos_ids   -> ModernBERT rotary positions (computed, not looked up)

Prints per-tensor min/max and the first offending batch only -- no patient rows.

Usage: probe_index_ranges.py <run_dir> [n_batches] [split]
"""

import os
import sys
from pathlib import Path

import polars as pl
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


def main() -> int:
    run_dir = Path(sys.argv[1])
    n_batches = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    split = sys.argv[3] if len(sys.argv) > 3 else "train"

    onto = Path(os.environ["NF_ONTOLOGY_DIR"])
    v_ext = int(pl.read_parquet(onto / "ontology_vocab.parquet")["token_id"].max()) + 1

    cfg = OmegaConf.load(run_dir / "resolved_config.yaml")
    vocab_size = int(cfg.lightning_module.model.config_overrides.vocab_size)
    max_queries = int(cfg.lightning_module.model.max_queries)
    print(f"V_ext={v_ext}  config vocab_size={vocab_size}  max_queries={max_queries}")

    cfg.datamodule.num_workers = 4
    dm = instantiate(cfg.datamodule)
    dm.setup("fit")
    dl = {"train": dm.train_dataloader, "tuning": dm.val_dataloader}[split]()

    agg: dict[str, tuple[int, int]] = {}
    bad = 0

    def note(name: str, t: torch.Tensor) -> tuple[int, int]:
        lo, hi = int(t.min()), int(t.max())
        p_lo, p_hi = agg.get(name, (lo, hi))
        agg[name] = (min(lo, p_lo), max(hi, p_hi))
        return lo, hi

    for i, batch in enumerate(dl):
        if i >= n_batches:
            break
        checks = []
        lo, hi = note("code", batch.code)
        checks.append(("code", lo, hi, vocab_size))
        if batch.q_codes is not None:
            lo, hi = note("q_codes", batch.q_codes)
            checks.append(("q_codes", lo, hi, vocab_size))
        if batch.q_bound_codes is not None:
            lo, hi = note("q_bound_codes", batch.q_bound_codes)
            checks.append(("q_bound_codes", lo, hi, vocab_size))
        if batch.q_answers is not None:
            lo, hi = note("q_answers", batch.q_answers)
            checks.append(("q_answers", lo, hi, 2))
        if batch.time_pos_ids is not None:
            note("time_pos_ids", batch.time_pos_ids)
        note("n_queries", torch.tensor([batch.q_codes.shape[1]]))

        for name, lo, hi, limit in checks:
            if lo < 0 or hi >= limit:
                bad += 1
                print(f"\n!! batch {i}: {name} out of range: [{lo}, {hi}] vs limit {limit}")
                if bad >= 3:
                    break
        if bad >= 3:
            break
        if (i + 1) % 100 == 0:
            print(f"  ...{i + 1} batches scanned")

    print(f"\nscanned {min(i + 1, n_batches)} batches of split={split}")
    print(f"{'tensor':<16}{'min':>12}{'max':>12}{'limit':>12}")
    limits = {"code": vocab_size, "q_codes": vocab_size, "q_bound_codes": vocab_size,
              "q_answers": 2, "time_pos_ids": None, "n_queries": max_queries}
    for name, (lo, hi) in agg.items():
        lim = limits.get(name)
        flag = ""
        if lim is not None and (lo < 0 or hi >= lim):
            flag = "  <-- OUT OF RANGE"
        print(f"{name:<16}{lo:>12}{hi:>12}{str(lim):>12}{flag}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
