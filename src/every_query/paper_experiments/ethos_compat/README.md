# `paper_experiments/ethos_compat/`

`EQ_reprocess_ethos` — make Ethos-tokenized MEDS shards ingestible by EQ training infra (MTD → EQ_train → EQ_predict → EQ_evaluate).

Filed under `paper_experiments/` because the matched-tokenization comparison between EQ's training objective and Ethos's autoregressive objective is a paper-only question.

See [#174](https://github.com/payalchandak/EveryQuery/issues/174) for the design discussion.

## What it does

The Ethos tokenizer (`ipolharvard/ethos-ares`) splits each ICD10/CM diagnosis and each ATC drug code across multiple events at the same `(subject_id, time)`. EQ's task grammar is single-code, so we collapse each split family back to one event with a deterministic opaque identifier (`EQ_TOK//<16-char-hex>`). The mapping file (`metadata/ethos_code_mapping.parquet`) records `(input_code, output_code, family)` per changed code.

Atomic events outside the split families (labs, vitals, BMI, SOFA quantile tokens, admissions, discharges, time-deltas) **pass through unchanged**.

ICD10/PCS char-by-char split (up to 7 sub-tokens) is explicitly out of scope — those events pass through atomically; if your runs include PCS codes and you need them recombined, file a follow-up.

## Static reinjection

Optional. `static_data_path=…` accepts a **parquet** with `subject_id`, `code` columns (and optional `numeric_value`). Each row is reinjected as a `time=null` MEDS row routed into its subject's home shard.

The native Ethos `StaticDataCollector` emits a **pickle** whose schema this PR has not yet inspected — only parquet input is supported here for now. To produce a real Ethos pickle and inspect its layout, run:

```bash
python scripts/setup_ethos_demo_data.py --workdir /tmp/ethos-demo
```

The script downloads the public MIMIC-IV-demo dataset (no PhysioNet credentials needed for the demo specifically), converts it to MEDS via `meds_etl`, runs Ethos's tokenization, and inspects the resulting pickle so we can extend `load_static_table` once we know the format.

## Code naming

Combined codes are `EQ_TOK//<hash>` where the hash is a blake2b digest (16 hex chars / 64 bits) of the sorted input-codes tuple. Opaque by design — the user clarification on #174 said any unique string suffices, and reversibility is provided by the mapping file rather than the string structure.

| Input rows at same `(subject_id, time)`                           | Output             |
| ----------------------------------------------------------------- | ------------------ |
| `ICD//CM//I10`                                                    | `EQ_TOK//<hash_a>` |
| `ICD//CM//E11`, `ICD//CM//3-6//65`                                | `EQ_TOK//<hash_b>` |
| `ICD//CM//K70`, `ICD//CM//3-6//30`, `ICD//CM//SFX//1`             | `EQ_TOK//<hash_c>` |
| `ATC//N02BA01//Acetylsalicylic_Acid`, `ATC//4//A`, `ATC//SFX//01` | `EQ_TOK//<hash_d>` |

## Pipeline position

```
Raw MEDS
  → Ethos tokenizer (drops + remaps + splits + quantizes; statics → pickle)
  → Ethos parquets (multi-token timeline)  +  static_data.pkl
  → EQ_reprocess_ethos                                             ← THIS MODULE
  → singleton-code MEDS shards (with optional time=null statics)
  → MTD_preprocess (existing EQ path)*
  → EQ_train / EQ_predict / EQ_evaluate
```

\* MTD ingestibility on real Ethos output is gated on the `test_real_ethos_output_recombines_and_mtd_ingests` integration test — see [Real-data integration test](#real-data-integration-test) below.

## CLI

```bash
EQ_reprocess_ethos \
	input_dir=/path/to/ethos/output \
	output_dir=/path/to/eq/input \
	static_data_path=/path/to/static.parquet
```

Required:

- `input_dir` — directory of Ethos-tokenized MEDS shards (`{input_dir}/data/{split}/*.parquet`).
- `output_dir` — directory to write recombined shards (mirrors input layout).

Optional:

- `static_data_path` — parquet with `(subject_id, code[, numeric_value])` columns. Default `null` (no reinjection). Pickle support is intentionally absent until we inspect the real Ethos format — see [Static reinjection](#static-reinjection).
- `overwrite=true` — clobber `output_dir` if it exists. Default `false`.

## Output layout

```
{output_dir}/
├── data/
│   └── {split}/
│       └── {shard}.parquet              ← singleton-code shards + optional time=null statics
└── metadata/
    ├── codes.parquet                    ← rewritten vocab; mid/sfx tokens dropped, heads remapped
    ├── ethos_code_mapping.parquet       ← (input_code, output_code, family); sorted, deterministic
    └── *                                ← other metadata files passed through unchanged
```

`ethos_code_mapping.parquet` schema:

| column        | type     | meaning                                           |
| ------------- | -------- | ------------------------------------------------- |
| `input_code`  | `string` | The Ethos code (head, mid, or sfx)                |
| `output_code` | `string` | The combined singleton code it now contributes to |
| `family`      | `string` | `ICD_CM` or `ATC` — which family it belonged to   |

Pass-through codes are not in the mapping — only changed codes are recorded. The mapping is sorted by `(family, input_code, output_code)` so two runs over identical input produce byte-identical mapping parquets (covered by `test_reprocess_directory_mapping_is_deterministic`).

## Real-data integration test

The slow test `tests/test_ethos_compat.py::test_real_ethos_output_recombines_and_mtd_ingests` runs `EQ_reprocess_ethos` + `MTD_preprocess` end-to-end on real Ethos-tokenized MIMIC-IV-demo data. It's skipped by default — set `ETHOS_DEMO_DIR` to the output of the setup script:

```bash
python scripts/setup_ethos_demo_data.py --workdir /tmp/ethos-demo
ETHOS_DEMO_DIR=/tmp/ethos-demo/ethos_output pytest -m slow tests/test_ethos_compat.py
```

This is the load-bearing assertion that the recombined output is actually consumable by EQ's downstream pipeline.

## Caveats / follow-ups

- **ICD10/PCS family** not handled — passes through atomically. Follow-up if needed.
- **Ethos pickle format inspection** outstanding — `scripts/setup_ethos_demo_data.py` is the path forward.
- **Other Ethos preprocessing decisions are preserved**: ICD9→ICD10 mapping, drug→ATC mapping, code filtering, per-code quantile binning. Properties of the input.
- **Ordering assumption**: rank-based pairing of `head[k]` / `mid[k]` / `sfx[k]` relies on Ethos's polars `.explode()` preserving per-original-event grouping. Currently true upstream; would surface as visibly garbled mappings if it ever changes.

## Related

- Design discussion: [#174](https://github.com/payalchandak/EveryQuery/issues/174)
- Ethos tokenizer source: `ipolharvard/ethos-ares` → `src/ethos/tokenize/`
- Setup script for real test data: `scripts/setup_ethos_demo_data.py`
