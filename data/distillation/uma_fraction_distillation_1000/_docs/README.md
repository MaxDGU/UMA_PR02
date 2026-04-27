# UMA Fraction Distillation 1000

This directory contains GitHub-friendly Parquet shards for UMA fraction distillation traces.

```python
import pandas as pd

df = pd.read_parquet("data/distillation/uma_fraction_distillation_1000")
```

Metadata lives in `_docs/manifest.json`. The `_docs` prefix keeps `pd.read_parquet(...)`
from trying to parse Markdown or JSON files as Parquet data.
