# FractionGPT — Importance-Weighted SFT + KL Fine-Tuning

This folder contains the fine-tuning pipeline used to align the FractionGPT
transformer (the model trained in `results/transformer_replication/`) to the
human answer distribution on the 24-problem human eval set. The headline
configuration produced the "FractionGPT + SFT/KL (t=0.5)" bar in the SP2013
ED/UD comparison figure.

This is the FractionGPT counterpart of the Qwen-distill SFT/KL pipeline in
`human_ft/finetune_humandata.py`. Same objective (importance-weighted SFT with
a KL anchor to a frozen reference), different base model.

## Method

For each human-eval problem, every UMA trace is reweighted so the model is
trained on a distribution of (problem, answer) pairs that matches the human
answer frequencies rather than UMA's:

```
w(trace) = p_human(answer | problem) / p_uma(answer | problem)
```

Both distributions are Laplace-smoothed with `alpha = 0.05` so answers humans
give but UMA never produces still get a nonzero floor. Weights are clamped to
`max_weight` (default 5.0) and normalised per epoch.

A frozen copy of the pretrained FractionGPT acts as the KL reference. The
training loss is:

```
loss = CE(student, trace) + kl_beta * KL(student || reference)
```

The KL term is computed per-position on completion tokens only and averaged
over non-pad positions. `kl_beta` is the main knob — `0.5` is the value that
appears in the figure legend.

## Layout

```
human_ft/fractiongpt/
├── README.md
├── train_transformer_phase3.py   # FractionTokenizer, FractionGPT model, dataset
├── finetune/
│   ├── __init__.py
│   ├── config.py                 # DEFAULT_CONFIG + argparse for SFT/KL knobs
│   ├── data.py                   # load_human_data, normalize_answer, reward utilities
│   ├── generate.py               # build_prompt, generate_rollout (used by evaluate)
│   ├── evaluate.py               # evaluate_model — SP2013/Siegler MAE + token acc
│   └── sft_reweight.py           # entrypoint: importance-weighted SFT + KL
├── slurm/
│   ├── run_sft_reweight.slurm    # baseline run (kl_beta=0.5, lr=1e-5)
│   └── run_sft_hikl.slurm        # higher kl_beta variant (kl_beta=1.5, lr=1e-6)
└── scripts/
    └── plot_sft_ed_ud_comparison.py  # generates figure_ed_ud_comparison_sft.png
```

## Running

The script expects to be launched from a directory where `train_transformer_phase3`
is importable and `finetune/` is on `PYTHONPATH` as a package (it bootstraps
this in `sft_reweight.py`). Local invocation:

```bash
python finetune/sft_reweight.py \
    --sft_lr 1e-5 \
    --sft_epochs 20 \
    --batch_size 64 \
    --smooth_alpha 0.05 \
    --kl_beta 0.5 \
    --max_weight 5.0 \
    --eval_every 5 \
    --tag sft_reweight_v1
```

Required inputs (resolved via `DEFAULT_CONFIG` in `finetune/config.py`):

- A pretrained FractionGPT checkpoint (the "reference" for KL and the init).
- UMA traces (`uma_traces_all.csv.gz` or equivalent) for the 24 human-eval problems.
- Human answer-frequency tables (loaded by `data.load_human_data`).

Outputs land in `finetune/output/`:
- `best_model_<tag>.pt` — best checkpoint by validation MAE
- `history_<tag>.json` — per-epoch metrics

Then regenerate the comparison plot:

```bash
python scripts/plot_sft_ed_ud_comparison.py
```

## Key knobs

| Flag | Default | What it controls |
| --- | --- | --- |
| `--kl_beta` | 0.5 | Strength of KL anchor to frozen reference |
| `--smooth_alpha` | 0.05 | Laplace smoothing on human and UMA answer distributions |
| `--max_weight` | 5.0 | Clip on per-trace importance weight |
| `--sft_lr` | 1e-5 | Learning rate (lower for high-beta runs) |
| `--sft_epochs` | 20 | Number of epochs |
| `--tag` | `sft_reweight_v1` | Prefix for output checkpoint and history file |

## Notes

- Paths in the SLURM scripts are hard-coded to the Princeton scratch tree
  (`/scratch/gpfs/GRIFFITHS/mg7411/llm_student/fractions`). Edit before
  reusing on another cluster.
- The KL term is computed against the reference in eval mode while the student
  is in train mode; dropout mismatch inflates the absolute KL value slightly
  but the gradient direction is unchanged.
- The reference model is loaded once and kept frozen (`requires_grad_(False)`)
  for the full run.
