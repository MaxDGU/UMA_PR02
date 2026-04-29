# Whole-Number Distilled SmolLM2-135M

Distilled checkpoint that learns whole-number arithmetic reasoning by
imitating UMA student traces (100 students × whole-number problems × {+, -, ×}).

## Where the weights live

The full checkpoint (including `model.safetensors`) is on Hugging Face Hub:

> **https://huggingface.co/MaxDGUPTA/uma-whole-smollm135m**

Drop-in load:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "MaxDGUPTA/uma-whole-smollm135m"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo)
```

`best/` in this folder ships the small companion files (config, tokenizer,
training history, exact CLI args) so you can inspect provenance without an
HF round-trip; the weights themselves are deliberately *not* in this repo.

## Provenance

| | |
|---|---|
| Base | `HuggingFaceTB/SmolLM2-135M` (pretrained, not -Instruct) |
| Data | `uma_traces_whole_100students_nlp.csv.gz` (100 UMA students, whole-number) |
| Epochs | 1 (with mid-epoch eval; best at update 22 630) |
| Effective batch | 128 |
| LR | 5e-4 cosine, warmup 0.03, weight_decay 0.01 |
| Max length | 256 |
| Split | `split_by_problem=true`, val_frac=0.005, test_frac=0.005 |
| AMP | bf16 |
| Seed | 42 |

Final metrics at best: train_loss = 0.00932, val_loss = 0.00574, test_loss
= 0.00504. See `best/history.json` for the full eval log and
`best/train_args.json` for the exact CLI configuration the BBT pipeline ran.

## Reproducing

The BBT pipeline driver and the whole-number config are checked in alongside
this checkpoint:

- `BBT/configs/whole_number_pipeline.json`
- `BBT/pipeline.py`

The slurm wrapper used to launch this training run is
`slurm/run_whole_distill_full.slurm` in the parent fractionGPT working tree.
