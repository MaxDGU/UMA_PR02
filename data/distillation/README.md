# UMA Distillation Data

This directory contains Parquet-sharded UMA distillation traces prepared for sharing through GitHub. The data are natural-language reasoning traces derived from UMA student simulations, with source-local absolute paths normalized to repo-relative identifiers in `source_file` and `source_uid`.

## Datasets

| Directory | Source | Rows | Students | Problems | Shards |
| --- | --- | ---: | ---: | ---: | ---: |
| `uma_fraction_distillation_25/` | panel25 default tracev10, SP2013 held out | 511,315 | 25 | 20,720 | 2 |
| `uma_fraction_distillation_1000/` | all1000 clean-child correct-exec distillation traces | 10,738,383 | 1,000 | 20,736 | 22 |
| `uma_decimal_bssmix_no3dpmul_repaired_v1/` | repaired decimal BSS mixture, no 3dp multiplication | 1,643,043 | 100 | 17,520 | 5 |

Each dataset directory is directly readable as a Parquet dataset:

```python
import pandas as pd

panel25 = pd.read_parquet("data/distillation/uma_fraction_distillation_25")
all1000 = pd.read_parquet("data/distillation/uma_fraction_distillation_1000")
decimal = pd.read_parquet("data/distillation/uma_decimal_bssmix_no3dpmul_repaired_v1")
```

For lower-memory reads, select columns or use PyArrow dataset scanning:

```python
import pyarrow.dataset as ds

dataset = ds.dataset("data/distillation/uma_fraction_distillation_1000", format="parquet")
table = dataset.to_table(columns=["subjid", "prob", "instruction_nl", "response_nl"])
```

## Contents

- `part-*.parquet`: zstd-compressed Parquet shards, each under 50 MiB.
- `_docs/manifest.json`: source path, row counts, schema, shard sizes, SHA256 hashes, and translation-version metadata.
- `_docs/README.md`: short per-dataset loading note.

The metadata files are stored under `_docs/` so `pd.read_parquet(<dataset_dir>)` reads only Parquet shards.

## Source Files

The exported datasets were generated from:

- `results/transformer_replication/synth_unique_panel25_even_g_rt5_ice50_seed1_tracev10_default_nosp2013_mincols.csv.gz`
- `results/transformer_replication/synth_unique_seed1_all1000_nlp_clean_child_correct_exec_mincols.csv.gz`
- `results/transformer_replication/decimal_bss_mixture_add2400_mul4800_no3dpmul_repaired_v1_k100_xlsx_distill/uma_traces_decimal_bss_mix_add2400_mul4800_no3dpmul_repaired_v1_k100_xlsx_nlp_think_aloud_child_train_mincols.csv.gz`

The raw UMA trace CSVs are not included here.
