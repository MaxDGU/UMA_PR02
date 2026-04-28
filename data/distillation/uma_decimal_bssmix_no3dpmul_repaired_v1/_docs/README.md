# UMA Decimal BSS Mixture No-3dp-Mul Repaired v1

This directory contains GitHub-friendly Parquet shards for the repaired decimal UMA distillation mixture used by the Qwen3 one-epoch launcher.

```python
import pandas as pd

df = pd.read_parquet("data/distillation/uma_decimal_bssmix_no3dpmul_repaired_v1")
```

Metadata lives in `_docs/manifest.json`. The `_docs` prefix keeps `pd.read_parquet(...)` from trying to parse Markdown or JSON files as Parquet data.
