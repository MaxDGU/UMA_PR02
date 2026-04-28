# Translated Synthetic Algebra v3 Human-Mixed Dataset

This directory contains the GitHub-friendly form of the translated synthetic
single-variable algebra distillation dataset.

Tracked data:

```text
parquet/train/part-*.parquet
parquet/val/part-00000.parquet
parquet/test/part-00000.parquet
parquet/manifest.json
singlevar_algebra_synth_v3_human_mixed_audit.json
singlevar_algebra_synth_v3_human_mixed_preview.csv
```

The full CSV files can be regenerated locally:

```bash
python cognitive_tutor/build_translated_synthetic_algebra_dataset.py
```

The Parquet shards can be regenerated from the CSV package:

```bash
python cognitive_tutor/shard_translated_synthetic_algebra_parquet.py
```

Load in Python:

```python
import pandas as pd

train = pd.read_parquet(
    "cognitive_tutor/data/processed/algebra_2006_2007/"
    "translated_synthetic_algebra_v3_human_mixed/parquet/train"
)
```

Rows use `instruction_nl` and `response_nl` for distillation. UMA parameter
columns (`g`, `d`, `c`, `rt_mu`, `ice`) are included but not exposed in the
default prompt.
