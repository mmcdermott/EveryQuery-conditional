"""Prove, after the fact, that all three new features were actually ON for a training run.

Config keys can be set and still not take effect (the two `ontology_dir` keys are never
cross-checked, and a cohort with no TIMELINE//DELTA* codes makes `strip_delta_tokens` a no-op),
so this checks the run three ways:

  1. the SAVED CONFIG            -- what was requested
  2. the CHECKPOINT hyperparameters -- what the model was built with
  3. a REAL BATCH + the LOADED MODEL -- what actually reaches the forward pass

(3) is the one that matters.  Prints booleans, shapes and counts only -- no patient rows.

Usage: 09_verify_run.py <run_dir>
"""

import dataclasses
import os
import sys
from pathlib import Path

import polars as pl
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

OK, BAD = "PASS", "**FAIL**"


def main() -> int:
    run_dir = Path(sys.argv[1])
    onto_dir = Path(os.environ["NF_ONTOLOGY_DIR"])
    results: list[tuple[str, bool, str]] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        results.append((name, bool(cond), detail))

    # ---------------- 1. saved config ----------------------------------------------------
    cfg = OmegaConf.load(run_dir / "resolved_config.yaml")
    ds_kwargs = cfg.datamodule.dataset_kwargs
    mdl = cfg.lightning_module.model

    v_ext = pl.read_parquet(onto_dir / "ontology_vocab.parquet")["token_id"].max() + 1

    check("config: strip_delta_tokens=true", ds_kwargs.strip_delta_tokens is True,
          f"{ds_kwargs.strip_delta_tokens}")
    check("config: use_rope_time=true", mdl.use_rope_time is True, f"{mdl.use_rope_time}")
    check("config: dataset ontology_dir set", ds_kwargs.ontology_dir is not None, "")
    check("config: model ontology_dir set", mdl.ontology_dir is not None, "")
    check("config: the two ontology_dirs AGREE",
          str(ds_kwargs.ontology_dir) == str(mdl.ontology_dir),
          "not cross-checked by the code — drift is silent")
    check("config: ontology_dir is the one we built",
          str(mdl.ontology_dir) == str(onto_dir), "")
    check(f"config: vocab_size == V_ext ({v_ext})",
          int(mdl.config_overrides.vocab_size) == int(v_ext),
          f"got {mdl.config_overrides.vocab_size}")

    # ---------------- 2. checkpoint hyperparameters ---------------------------------------
    ckpt_path = run_dir / "best_model.ckpt"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "checkpoints" / "last.ckpt"
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ck["hyper_parameters"]["model"]
    check("ckpt: use_rope_time", hp.get("use_rope_time") is True, f"{hp.get('use_rope_time')}")
    check("ckpt: ontology_dir", hp.get("ontology_dir") is not None, "")
    sd = ck["state_dict"]
    check("ckpt: bound_marker parameter present (event-bound head)",
          any(k.endswith("bound_marker") for k in sd), "")
    emb_keys = [k for k in sd if "tok_embeddings" in k or "OntologyEmbedding" in k]
    check("ckpt: embedding rows == V_ext",
          any(tuple(sd[k].shape)[0] == int(v_ext) for k in emb_keys if sd[k].dim() == 2),
          f"{[(k, tuple(sd[k].shape)) for k in emb_keys if sd[k].dim() == 2][:3]}")
    check(f"ckpt: global_step > 0", ck.get("global_step", 0) > 0, f"{ck.get('global_step')}")

    # ---------------- 3. a real batch + the loaded model -----------------------------------
    from every_query.model.conditional_lightning import ConditionalQueryLightningModule
    from every_query.utils.model_loader import setup_model

    # setup_model returns (train_cfg, lightning_module, trainer); ckpt_name=None -> best_model.ckpt.
    train_cfg, M, _ = setup_model(
        str(run_dir), ckpt_name=None, module_cls=ConditionalQueryLightningModule
    )
    emb = M.model.HF_model.get_input_embeddings()
    check("model: input embedding is OntologyEmbedding",
          type(emb).__name__ == "OntologyEmbedding", type(emb).__name__)
    check("model: embedding table has V_ext rows",
          int(getattr(emb, "num_embeddings", -1)) == int(v_ext),
          f"{getattr(emb, 'num_embeddings', None)}")

    dm = instantiate(train_cfg.datamodule)
    dm.setup("fit")
    dl = dm.val_dataloader()
    batch = next(iter(dl))

    check("batch: time_pos_ids present (RoPE time)", batch.time_pos_ids is not None, "")
    if batch.time_pos_ids is not None:
        tp = batch.time_pos_ids
        check("batch: time_pos_ids shape == code shape",
              tuple(tp.shape) == tuple(batch.code.shape), f"{tuple(tp.shape)} vs {tuple(batch.code.shape)}")
        # Rows are left-packed and zero-padded on the right, so check only real tokens.
        keep = batch.code[0] != 0
        row = tp[0][keep]
        check("batch: time_pos_ids nondecreasing over real tokens, starting at 0",
              bool((row[1:] >= row[:-1]).all()) and int(row[0]) == 0,
              f"n_real={int(keep.sum())} first={int(row[0])} last={int(row[-1])} "
              f"max_in_batch={int(tp.max())} (elapsed hours)")

    check("batch: q_bound_codes present (event bounds)", batch.q_bound_codes is not None, "")
    if batch.q_bound_codes is not None:
        qb = batch.q_bound_codes
        n_bound = int((qb > 0).sum())
        n_q = int(batch.q_mask.sum())
        check("batch: some queries ARE event-bounded", n_bound > 0,
              f"{n_bound}/{n_q} queries bounded = {n_bound / max(n_q, 1):.3f}")
        check("batch: bounded queries carry the -1 duration sentinel",
              bool((batch.q_durations[(qb > 0)] == -1.0).all()) if n_bound else True, "")

    # ancestor queries actually present in a batch?
    vocab = pl.read_parquet(onto_dir / "ontology_vocab.parquet")
    anc_ids = set(vocab.filter(~pl.col("is_observed_code"))["token_id"].to_list())
    qc = batch.q_codes[batch.q_mask].tolist()
    n_anc = sum(1 for i in qc if i in anc_ids)
    check("batch: some queries are ANCESTOR nodes (DAG-aware)", n_anc > 0,
          f"{n_anc}/{len(qc)} = {n_anc / max(len(qc), 1):.3f}")

    # A forward pass must actually run with all of it wired together.  `load_from_checkpoint`
    # restores the module onto cuda:0 while the dataloader yields CPU tensors, and
    # MEDSTorchBatch has no `.to()` -- so move every tensor field by hand.  (In training,
    # Lightning's own transfer_batch_to_device does this generically.)
    M.eval()
    dev = next(M.parameters()).device
    # Mutate in place rather than dataclasses.replace: __post_init__ asserts each field is a
    # `torch.LongTensor`, and a CUDA long tensor fails that isinstance check.
    for f in dataclasses.fields(batch):
        v = getattr(batch, f.name)
        if torch.is_tensor(v):
            object.__setattr__(batch, f.name, v.to(dev))
    with torch.no_grad():
        _loss, out = M.model(batch)
    check("forward: runs end-to-end with all features on",
          out.answer_logits is not None, f"logits {tuple(out.answer_logits.shape)}")

    # ---------------- report ----------------------------------------------------------------
    print(f"\nrun_dir: {run_dir}")
    print(f"ckpt   : {ckpt_path.name}   global_step={ck.get('global_step')}   V_ext={v_ext}\n")
    width = max(len(n) for n, _, _ in results)
    n_fail = 0
    for name, ok, detail in results:
        n_fail += (not ok)
        print(f"  {OK if ok else BAD:<10} {name:<{width}}  {detail}")
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
