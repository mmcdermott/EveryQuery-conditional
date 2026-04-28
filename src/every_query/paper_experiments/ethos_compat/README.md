# `paper_experiments/ethos_compat/`

`EQ_reprocess_ethos` — collapse Ethos's split-token MEDS shards back into singleton-code shards so EQ_train and EQ_evaluate run unchanged on Ethos-tokenized data.

Filed under `paper_experiments/` because the matched-tokenization comparison between EQ's training objective and Ethos's autoregressive objective is a paper-only question — it's not part of the normal EveryQuery user pipeline.

See [#174](https://github.com/payalchandak/EveryQuery/issues/174) for the design discussion.

## What it does

The Ethos tokenizer (`ipolharvard/ethos-ares`) splits two code families across multiple events at the same `(subject_id, time)`:

- **ICD10/CM diagnoses** → 3 events: `ICD//CM//<head>` (chars 0–3) + `ICD//CM//3-6//<mid>` (chars 3–6) + `ICD//CM//SFX//<sfx>` (chars 6+).
- **ATC drug codes** → 3 events: `ATC//<head>` (chars 0–3) + `ATC//4//<mid>` (char 3) + `ATC//SFX//<sfx>` (chars 4+).

EQ's task grammar (`TaskQuerySchema`) treats `code` as an atom: a query for "did `I10` occur" against Ethos's split-token corpus would require conjunctive reasoning over the triplet, which the load-bearing `evaluate_index_df` cannot express.

This module reverses the splits. For each family, every `(head, optional mid, optional sfx)` triplet at a shared timestamp collapses into a single combined code:

```
ICD//CM//I10                  ← head only (3-char ICD)         → ICD//CM//I10
ICD//CM//I10
ICD//CM//3-6//00              ← head + mid (5-char ICD)       → ICD//CM//I10//3-6//00
ICD//CM//I10
ICD//CM//3-6//65
ICD//CM//SFX//1               ← full triplet (7-char ICD)     → ICD//CM//I10//3-6//65//SFX//1
```

The combined code is deterministic and reversible — splitting on `//3-6//` and `//SFX//` recovers the original triplet.

## Pipeline position

```
Raw MEDS
  → Ethos tokenizer (drops + remaps + splits + quantizes)
  → Ethos-style parquets (multi-token timeline)
  → EQ_reprocess_ethos                                    ← THIS MODULE
  → flat-code parquets (Ethos's filtering + remapping + quantization, but singleton codes)
  → MTD_preprocess (existing EQ path)
  → EQ_train / EQ_predict / EQ_evaluate
```

The output code vocabulary is **not** the raw ICD10/ATC strings — tokenization is a property of the preprocessing run, and EQ's tasks are scoped to whatever vocab their corpus has. This is fine; the goal is matched-data comparison of the two training objectives, not vocabulary parity across runs.

## CLI

```bash
EQ_reprocess_ethos \
	input_dir=/path/to/ethos/output \
	output_dir=/path/to/eq/input
```

Required:

- `input_dir` — directory containing Ethos's tokenized MEDS shards (expects `{input_dir}/data/{split}/*.parquet`).
- `output_dir` — directory to write the recombined shards into (mirrors the input layout).

Optional:

- `overwrite=true` — clobber `output_dir` if it already exists. Default `false`.

## Output layout

```
{output_dir}/
├── data/
│   └── {split}/
│       └── {shard}.parquet              ← singleton-code MEDS shards (same schema as input)
└── metadata/
    ├── codes.parquet                    ← rewritten code vocabulary (mid/sfx codes dropped, heads remapped)
    ├── ethos_code_mapping.parquet       ← (input_code, output_code) for every changed code
    └── *                                ← other metadata files passed through unchanged
```

The `ethos_code_mapping.parquet` contains exactly one row per `(input_code, output_code)` pair the run produced. Pass-through codes (atomic events that didn't change) are not included. Schema:

| column        | type     | meaning                                           |
| ------------- | -------- | ------------------------------------------------- |
| `input_code`  | `string` | The Ethos code (head, mid, or sfx)                |
| `output_code` | `string` | The combined singleton code it now contributes to |

## Caveats

- **Only ICD10/CM and ATC families are recombined.** Ethos splits ICD10/PCS character-by-character (up to 7 sub-tokens); that's a separate family with a different structure. If your run includes PCS codes, file a follow-up — the same recombination shape applies but the prefix scheme differs.
- **Ethos's static-pickle extraction is not undone here.** Demographics + birth + race + marital that Ethos pulls out of the timeline into a side-channel pickle remain absent from the output. For matched-data EQ training you may want to re-inject them via a separate stage; out of scope for this module.
- **Other Ethos preprocessing decisions are preserved**: ICD9→ICD10 mapping, drug→ATC mapping, code filtering (infusions / weights / heights / eGFR / etc.), per-code quantile binning. Those are properties of the input, not of this module.
- **Ordering assumption**: the recombination pairs `head[k]` with `mid[k]` and `sfx[k]` based on Ethos's row order. Ethos uses a polars-native `.explode()` which preserves per-original-event grouping, so this is correct in practice. Misalignment would surface as visibly garbled output codes; the test suite covers the round-trip identity for canonical inputs.

## Related

- Design discussion: [#174](https://github.com/payalchandak/EveryQuery/issues/174)
- Ethos tokenizer source: `ipolharvard/ethos-ares` → `src/ethos/tokenize/`
