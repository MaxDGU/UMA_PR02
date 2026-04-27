# Human Fraction Fine-Tuning Data

This directory contains the shareable 8-problem human fraction fine-tuning
corpus copied from `results/human`. It is the existing sp2011-style human
bundle used by the human fine-tuning pipeline. `sp2013_human.csv` is not
included.

## Contents

| Path | Rows | Notes |
| --- | ---: | --- |
| `data.csv` | 384 | Raw corpus: 48 subjects x 8 problems. Uses `cp1252` encoding. |
| `data_train.csv` | 288 | Raw training split: 36 subjects x 8 problems. Uses `cp1252` encoding. |
| `data_val.csv` | 96 | Raw validation split: 12 subjects x 8 problems. Uses `cp1252` encoding. |
| `data_train_nlp.csv` | 288 | Training prompts and responses for supervised fine-tuning. |
| `data_val_nlp.csv` | 96 | Validation prompts and responses for supervised fine-tuning. |
| `all_data/data_trainval_nlp.csv` | 384 | Train rows followed by validation rows, no filtering. |
| `correct_only/data_train_nlp_correct.csv` | 147 | Correct-answer subset of the training NLP split. |
| `correct_only/data_val_nlp_correct.csv` | 29 | Correct-answer subset of the validation NLP split. |
| `human_strategy_targets_classifier160k.json` | - | Problem-level target strategy distributions from the 160k strategy classifier. |
| `sts_data.csv` | 384 | Headerless problem/response pairs. |
| `subjid_train.txt`, `subjid_val.txt` | - | Subject split IDs. |
| `strategy codes.txt` | - | Strategy-code notes. |
| `manifest.json` | - | File sizes, SHA256 hashes, row counts, columns, encodings, and source paths. |

The NLP CSVs use these columns:

```text
subjid, prob, resp, strategy, instruction_nl, response_nl
```

Machine-local absolute paths in copied audit JSON artifacts were normalized to
repo-relative paths before sharing.

## Loading

```python
import pandas as pd

train = pd.read_csv("data/human_ft/data_train_nlp.csv")
val = pd.read_csv("data/human_ft/data_val_nlp.csv")
trainval = pd.read_csv("data/human_ft/all_data/data_trainval_nlp.csv")

raw = pd.read_csv("data/human_ft/data.csv", encoding="cp1252")
sts = pd.read_csv("data/human_ft/sts_data.csv", header=None, names=["prob", "resp"])
```

The canonical fine-tuning entrypoint is `human_ft/finetune_humandata.py`; see
`human_ft/README.md` for pipeline commands.
