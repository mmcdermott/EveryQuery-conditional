# `paper_experiments/ethos_compat/`

`EQ_reprocess_ethos` — make Ethos-tokenized MEDS shards ingestible by EQ training infra (MTD → EQ_train → EQ_predict → EQ_evaluate).

Filed under `paper_experiments/` because the matched-tokenization comparison between EQ's training objective and Ethos's autoregressive objective is a paper-only question.

See [#174](https://github.com/payalchandak/EveryQuery/issues/174) for the design discussion.

## What it does

Two things:

1. **Collapses split codes** — Ethos's tokenizer splits each ICD10/CM diagnosis and each ATC drug code across multiple events at the same `(subject_id, time)`. EQ's task grammar is single-code, so we collapse each split family back to one event with an opaque deterministic identifier (`EQ_TOK//<16-char-hex>`). The mapping file (`metadata/ethos_code_mapping.parquet`) is the canonical reverse lookup.
2. **Re-injects statics** — Ethos extracts demographics + birth + race + marital + BMI to a side-channel pickle. We optionally read that pickle (or a parquet) and inject the rows back as `time=null` MEDS rows, routed into the shard their subject lives in.

The output is a standard MEDS directory that downstream `MTD_preprocess` + `EQ_train` consume unchanged.

## Code naming

The combined-code identifier is `EQ_TOK//<hash>` where the hash is a blake2b digest (16 hex chars / 64 bits) of the sorted input-codes tuple. This is opaque by design — the user clarification on #174 said any unique string suffices, and reversibility is provided by the mapping file rather than the string structure.

| Input rows at same `(subject_id, time)`                           | Output             |
| ----------------------------------------------------------------- | ------------------ |
| `ICD//CM//I10`                                                    | `EQ_TOK//<hash_a>` |
| `ICD//CM//E11`, `ICD//CM//3-6//65`                                | `EQ_TOK//<hash_b>` |
| `ICD//CM//K70`, `ICD//CM//3-6//30`, `ICD//CM//SFX//1`             | `EQ_TOK//<hash_c>` |
| `ATC//N02BA01//Acetylsalicylic_Acid`, `ATC//4//A`, `ATC//SFX//01` | `EQ_TOK//<hash_d>` |

Atomic events outside the split families (labs, vitals, BMI, SOFA quantile tokens, admissions, discharges, time-deltas) **pass through unchanged**.

## Pipeline position

```
Raw MEDS
  → Ethos tokenizer (drops + remaps + splits + quantizes; statics → pickle)
  → Ethos parquets (multi-token timeline)  +  static_data.pkl
  → EQ_reprocess_ethos                                             ← THIS MODULE
  → singleton-code MEDS shards (with optional time=null statics)
  → MTD_preprocess (existing EQ path)
  → EQ_train / EQ_predict / EQ_evaluate
```

## CLI

```bash
EQ_reprocess_ethos \
	input_dir=/path/to/ethos/output \
	output_dir=/path/to/eq/input \
	static_data_path=/path/to/ethos/static_data.pkl
```

Required:

- `input_dir` — directory of Ethos-tokenized MEDS shards (`{input_dir}/data/{split}/*.parquet`).
- `output_dir` — directory to write recombined shards (mirrors input layout).

Optional:

- `static_data_path` — parquet with `(subject_id, code[, numeric_value])` columns OR pickle with `{subject_id: [code, ...]}` or `{subject_id: [(code, value), ...]}`. Default `null` (no reinjection).
- `overwrite=true` — clobber `output_dir` if it exists. Default `false`.

## Output layout

```
{output_dir}/
├── data/
│   └── {split}/
│       └── {shard}.parquet              ← singleton-code shards + optional time=null statics
└── metadata/
    ├── codes.parquet                    ← rewritten vocab; mid/sfx tokens dropped, heads remapped
    ├── ethos_code_mapping.parquet       ← (input_code, output_code) for every changed code
    └── *                                ← other metadata files passed through unchanged
```

`ethos_code_mapping.parquet` schema:

| column        | type     | meaning                                           |
| ------------- | -------- | ------------------------------------------------- |
| `input_code`  | `string` | The Ethos code (head, mid, or sfx)                |
| `output_code` | `string` | The combined singleton code it now contributes to |

Pass-through codes are not in the mapping — only changed codes are recorded.

## Caveats / follow-ups

- **Only ICD10/CM and ATC families are recombined.** Ethos splits ICD10/PCS character-by-character (up to 7 sub-tokens); that's a separate family with a different prefix scheme. File a follow-up if your runs include PCS codes.
- **Other Ethos preprocessing decisions are preserved**: ICD9→ICD10 mapping, drug→ATC mapping, code filtering (infusions / weights / heights / eGFR / etc.), per-code quantile binning. Those are properties of the input, not of this module.
- **Ordering assumption**: rank-based pairing of `head[k]` with `mid[k]` and `sfx[k]` relies on Ethos's polars `.explode()` preserving per-original-event grouping. Currently true upstream; would surface as visibly garbled mappings if it ever changes.
- **Static format flexibility**: parquet (`subject_id`, `code`, optional `numeric_value`) or pickle (`{sid: [code, ...]}` or `{sid: [(code, value), ...]}`). Other shapes need a small extension to `load_static_table`.

## Related

- Design discussion: [#174](https://github.com/payalchandak/EveryQuery/issues/174)
- Ethos tokenizer source: `ipolharvard/ethos-ares` → `src/ethos/tokenize/`
