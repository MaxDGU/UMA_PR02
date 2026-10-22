# Panel scaling: 25 vs 1,000 UMA learners (fractions)

How the fraction distill -> human-FT pipeline changes when the UMA learner panel grows from the paper's 25 learners to the
full 1,000-profile population. Full design, run log, and corrections: `DESIGN.md`.

**Evaluation.** SP2013: 16 problems x 256 samples, T=1.0, top_p=0.95, mixed-number-aware parser (`scripts/rescore_frac_dir.py`).
MAE = mean absolute error vs. children's accuracy (51.6%) across 8 operation x denominator cells. NLL = per-token NLL on 96
held-out Siegler 2011 child responses. **Prompt format:** training writes problems spaced ("3/5 + 1/4"); long-trained models are
sensitive to the unspaced form, so all arm-C / control / KL-primary / long-FT numbers use `c1000/eval_spaced.py`. Short-trained
arms are format-insensitive (B1000 seed 42: 18.7 unspaced vs 19.9 spaced).

**Human FT** = 288 Siegler 2011 responses, lr 1e-4, full FT, 16 examples/update (18 updates/epoch). **KL human FT** adds
KL(policy || distill ckpt) with beta 1.0 and is the primary metric.

## Arms
| arm | learners x problems/learner | train rows / updates | learner tag | folder |
|---|---|---|---|---|
| A | 25 x 20,720 (paper panel) | 466K / 3,643 | 100% | `results/arm_v10_p25regen_*` |
| A control | 25 x 20,720 | 466K / 3,643 | 50% | `c1000/` (Adrop) |
| B996 / B1000 | 996 or 1,000 x 520 (matched total compute) | 466K / 3,642-3,657 | 100% | `results/`, `b1000/` |
| C | 1,000 x 20,720 (matched problems per learner) | 18.65M / 145,688 | 50% | `c1000/` |
| D | 25 random learners x 20,720 | 466K / 3,643 | 100% | `results/D_*` |
The 4 profiles missing from the saved population (92, 93, 97, 98) were re-simulated with the original UMA code (`fill4/`, c=4).
"Learner tag" = fraction of distillation prompts prefixed with `<student> g .. d .. rt .. ice .. </student>`. With 100% tags,
long training makes the model unusable on untagged prompts (the first full-data run), so arm C uses 50%.

## Primary result: KL human FT, 4 seeds (`kl_primary/`)
| arm | MAE | acc | NLL |
|---|---|---|---|
| A (25, tag 100%) | 8.1 +- 0.9 | 55.2% | 1.56 |
| A control (25, tag 50%) | **6.1 +- 0.5** | 53.5% | **1.55** |
| B1000 (1,000 x 520) | 7.4 +- 1.1 | 46.6% | 1.63 |
| C (1,000 x 20,720) | 7.3 +- 0.3 | 51.2% | 6.56 |
Welch: A vs B1000 p=0.34; A control vs C p=0.011; tag 50% vs 100% (A) p=0.012.

## Longer KL human FT for the matched-problems comparison (`longhft/`, `longhft30/`; 2 seeds)
| | 25 learners MAE / NLL | 1,000 learners MAE / NLL |
|---|---|---|
| 1 epoch (18 updates) | 6.1 / 1.55 | 7.3 / 6.56 |
| 10 epochs, best by child NLL | 4.9 / 1.44 (epochs 3-4) | 6.9 / 4.26 |
| 30 epochs, epoch 20 | - (overfits after epoch 3) | 4.8 / 3.86 |
| 30 epochs, epoch 30 | - | 4.8 / 3.84 |
With ~7x more human-FT updates the 1,000-learner model matches the 25-learner error profile, but its NLL plateaus ~2.6x higher:
the KL anchor, not training length, limits how far it moves toward children's writing.

## Standard human FT (secondary)
A 7.6 +- 2.1, B996 18.7 +- 3.0, B1000 20.0 +- 2.0, A control 11.4 +- 2.3, C 13.0 +- 3.3 (all MAE, 4 seeds); NLL ~1.37 for every
arm except C (5.76). Mechanism (`kl_test/`): standard human FT erases the multi-step procedures the wider panels learned less firmly.

## Corrections to the earlier version of this folder
* The first full-data arm C was reported as a corrupted checkpoint. It was not: it works on tagged prompts (loss 0.021) and
  fails only on untagged ones because it was trained with the tag on 100% of prompts (`diag_C2/diag3.py`).
* `results/transformer_replication/train_transformer_hf.py` and `human_ft/finetune_humandata.py` now include the versions used for
  every run here: `enable_input_require_grads()` for gradient checkpointing with LoRA, an `--attn_implementation` option
  (default `eager`, unchanged), and whole-number prompt wording (fractions unchanged).

## Folders
`scripts/` generation, translation, arm building, distillation, human FT, scoring. `c1000/` full-data arm and control
(launcher with 50% tags, follow-up array, sentinel, canary, spaced eval). `kl_primary/` 4-seed KL runs. `longhft/`, `longhft30/`
10- and 30-epoch human FT (`--epochs N --save_epoch_list ...`). `b1000/`, `fill4/` 1,000-learner arm at matched compute and the
4 regenerated learners. `valcurve/` distillation validation curves. `diag_C/`, `diag_C2/` arm C diagnostics. `fig/`,
`fig_matched/` figures and plotting scripts. `*/training_histories/` per-run train_args, summaries, and per-epoch val NLL.
Not included (size): raw traces, translated corpora, checkpoints (on HF: MaxDGUPTA/uma-cognitive-llm-checkpoints).
All scripts contain della paths.
